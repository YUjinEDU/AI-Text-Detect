import os, pathlib, random, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
                         get_linear_schedule_with_warmup, TrainingArguments)
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer  # SFTTrainer 추가
from datasets import Dataset  # Huggingface datasets 라이브러리 추가
from tqdm.auto import tqdm
import utils
from sklearn.model_selection import train_test_split
import argparse
import time
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    return parser.parse_args()

# ---------------- Main Training Function ----------------
def train_lora(cfg, epochs=None, batch_size=None, lr=None):
    utils.set_seed(cfg['seed'])
    utils.log("Qwen3-4B QLoRA 학습 시작", is_important=True)
    
    # 데이터 로드 및 NULL 값 체크/처리
    data_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(data_path):
        orig_path = os.path.join(cfg['data_dir'], 'train.csv')
        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {orig_path}")
        utils.log(f"train_folds.csv 파일이 없습니다. 먼저 split_folds.py를 실행해주세요.")
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    
    utils.log("데이터 로드 중...")
    raw = pd.read_csv(data_path)
    raw = utils.check_and_clean_data(raw, cfg=cfg)
    
    # 데이터 분할
    train_df, val_df = train_test_split(raw, test_size=0.3, stratify=raw['label'], random_state=cfg['seed'])
    # 너무 짧거나 특수문자만 있는 텍스트 샘플 제거
    def is_valid_text(x):
        import re
        return (len(x.strip()) > 2) and (re.sub(r'[^\w가-힣]', '', x).strip() != '')
    train_df = train_df[train_df['text'].apply(is_valid_text)].reset_index(drop=True)
    val_df = val_df[val_df['text'].apply(is_valid_text)].reset_index(drop=True)
    utils.log(f"[라벨 분포] train: {train_df['label'].value_counts().to_dict()} val: {val_df['label'].value_counts().to_dict()}")
    utils.log(f"데이터 분할 완료: 학습 {len(train_df)}개, 검증 {len(val_df)}개")
    
    # 모델 및 토크나이저 로드
    utils.log("Qwen3-4B 모델 및 토크나이저 로드 중...")
    start_time = time.time()
    model_id = cfg['mistral']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg['mistral']['quant']['load_in_4bit'],
        bnb_4bit_use_double_quant=cfg['mistral']['quant']['double_quant'],
        bnb_4bit_quant_type=cfg['mistral']['quant']['quant_type'],
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True 
    )
    
    # 중요: 모델 설정 추가 - 안정적인 학습을 위해 필요
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    utils.log("LoRA 어댑터 초기화 중...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg['mistral']['lora_r'],
        lora_alpha=cfg['mistral']['lora_alpha'],
        lora_dropout=0.05,  # 드롭아웃 추가
        bias="none",        # 명시적 bias 설정
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    utils.log(f"모델 준비 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    utils.log(f"학습 디바이스: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 프롬프트/타겟 생성 함수
    def make_prompt(text, label):
        return f"이 문장이 가짜인가요? {text.strip()}\n정답: {('네' if label == 1 else '아니오')}"

    # SFTTrainer를 위한 데이터셋 준비 (간단한 형식)
    def prepare_dataset(df, aug=False):
        texts = []
        for i, row in df.iterrows():
            text = row['text']
            label = row['label']
            
            # 데이터 증강 (필요시)
            if aug and cfg['augment']['enable'] and random.random() < cfg['augment']['p_noise']:
                text = utils.noise_text(text, mask_ratio=cfg['augment']['mask_ratio'], 
                                       swap_ratio=cfg['augment']['swapcase_ratio'])
            
            # SFTTrainer 형식에 맞게 데이터 구성
            formatted_text = make_prompt(text, label)
            texts.append({"text": formatted_text})
        
        return Dataset.from_pandas(pd.DataFrame(texts))

    # 데이터셋 준비
    train_dataset = prepare_dataset(train_df, aug=True)
    val_dataset = prepare_dataset(val_df, aug=False)
    
    # 학습 파라미터 설정
    epochs = epochs or int(os.environ.get('EPOCHS', cfg['mistral']['epochs']))
    batch_size = batch_size or int(os.environ.get('BATCH_SIZE', cfg['batch_size']['mistral']))
    lr = lr or float(os.environ.get('LR', 1e-5))  # 학습률 낮춤 (1e-5로 시작)
    utils.log(f"하이퍼파라미터 설정: epochs={epochs}, batch_size={batch_size}, lr={lr:.1e}")

    # TrainingArguments 구성
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg['checkpoint_dir'], 'qwen3-4b-checkpoints'),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # 배치크기가 작을 경우 증가시켜 안정성 확보
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(cfg['log_dir'], 'qwen3-4b-logs'),
        learning_rate=lr,
        weight_decay=0.01,
        fp16=False,  # 4비트 양자화와 함께 사용 시 주의
        bf16=False,  # 4비트 양자화와 함께 사용 시 주의
        max_grad_norm=0.3,  # 그래디언트 클리핑 값 낮춤
        warmup_ratio=0.03,  # 워밍업 비율
        logging_steps=20,
        save_total_limit=3,
        report_to="none"
    )

    # SFTTrainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tok,
        dataset_text_field="text",
        max_seq_length=cfg['max_length']
    )
    
    utils.log("학습 시작", is_important=True)
    
    # 학습 진행
    progress = utils.ProgressTracker(epochs, "Qwen3-4B QLoRA 학습")
    
    try:
        # 학습 실행
        trainer.train()
        
        # 모델 저장
        utils.safe_makedirs(cfg['checkpoint_dir'])
        ckpt_path = os.path.join(cfg['checkpoint_dir'], 'qwen3-4b')
        trainer.save_model(ckpt_path)
        utils.log(f"학습 완료 및 모델 저장: {ckpt_path}", is_important=True)
        progress.finish(f"모델 저장 완료: {ckpt_path}")
        
    except Exception as e:
        utils.log(f"학습 중 오류 발생: {str(e)}", level="ERROR")
        raise
    
    # 메모리 정리
    utils.log("메모리 정리 시작...")
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    utils.log("메모리 정리 완료.")
    
    return trainer.state.best_metric

# ---------------- 추론 함수 (probs_model1) ----------------
def probs_model1(texts, cfg=None):
    if cfg is None:
        cfg = utils.load_config()

    model_id = cfg['mistral']['model_id']
    utils.log(f"{model_id} 모델 로드 중 (추론용)...")

    # 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # 양자화 설정
    bnb_config_infer = BitsAndBytesConfig(
        load_in_4bit=cfg['mistral']['quant']['load_in_4bit'],
        bnb_4bit_use_double_quant=cfg['mistral']['quant']['double_quant'],
        bnb_4bit_quant_type=cfg['mistral']['quant']['quant_type'],
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg['mistral']['quant']['load_in_4bit'] else None
    )

    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config_infer if cfg['mistral']['quant']['load_in_4bit'] else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg['mistral']['lora_r'],
        lora_alpha=cfg['mistral']['lora_alpha'],
        lora_dropout=0.05,  # 드롭아웃 추가
        bias="none",        # 명시적 bias 설정
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # PEFT 모델 구성
    model = get_peft_model(base_model, lora_config)
    
    # 학습된 가중치 로드
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'qwen3-4b')
    if os.path.exists(ckpt_path):
        utils.log(f"체크포인트 로드 중: {ckpt_path}")
        try:
            # 저장 방식에 따라 적절한 로딩 방식 선택
            if os.path.isdir(ckpt_path):  # SFTTrainer로 저장된 경우 디렉토리
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    quantization_config=bnb_config_infer if cfg['mistral']['quant']['load_in_4bit'] else None,
                    device_map="auto",
                    trust_remote_code=True
                )
                utils.log("모델 디렉토리에서 가중치 로드 완료")
            else:  # state_dict로 저장된 경우 파일
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
                utils.log("state_dict에서 가중치 로드 완료")
        except Exception as e:
            utils.log(f"체크포인트 로드 오류: {e}", level="WARNING")
    else:
        utils.log(f"체크포인트를 찾을 수 없음: {ckpt_path}", level="WARNING")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()

    results = []
    batch_size = cfg.get('batch_size', {}).get('mistral_infer', 8)

    # 텍스트 정제
    cleaned_texts = []
    for t in texts:
        if pd.isna(t):
            cleaned_texts.append("")
            utils.log("NaN 텍스트를 빈 문자열로 처리", level="WARNING")
        else:
            cleaned_texts.append(utils.clean_text(str(t)))

    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=f"{model_id} 추론"):
        batch_texts = cleaned_texts[i:i+batch_size]
        prompts = [f"이 문장이 가짜인가요? {text.strip()}\n정답:" for text in batch_texts]

        encs = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=cfg['max_length'])
        input_ids = encs.input_ids.to(DEVICE)
        attention_mask = encs.attention_mask.to(DEVICE)

        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        for j in range(len(generated_outputs)):
            full_output_sequence = generated_outputs[j]
            decoded_full = tok.decode(full_output_sequence, skip_special_tokens=True)
            answer_part = decoded_full.split("\n정답:")[-1].strip()

            if "네" in answer_part:
                results.append(1.0)
            elif "아니오" in answer_part:
                results.append(0.0)
            else:
                utils.log(f"예측 불분명: '{answer_part}' -> 0.5로 처리", level="WARNING")
                results.append(0.5)

    utils.log(f"{model_id} 추론 완료")
    return np.array(results)

# ---------------- main 함수 ----------------
def main():
    cfg = utils.load_config()
    try:
        utils.init_logging(cfg)
        args = parse_args()
        train_lora(cfg, 
                  epochs=args.epochs, 
                  batch_size=args.batch_size, 
                  lr=args.lr)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
    finally:
        utils.close_logging()

if __name__ == '__main__':
    main()