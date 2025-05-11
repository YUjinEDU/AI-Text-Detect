import os, pathlib, random, re, unicodedata, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils
import time
import re
import gc


# ---------------- Main Training Function ----------------
def train_deberta(cfg):
    utils.set_seed(cfg['seed'])
    utils.log("DeBERTa-v3-Large LoRA 학습 시작", is_important=True)
    
    # 데이터 로드 및 NULL 값 체크/처리
    data_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(data_path):
        # 원본 data 파일이 있는지 확인
        orig_path = os.path.join(cfg['data_dir'], 'train.csv')
        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {orig_path}")
        utils.log(f"train_folds.csv 파일이 없습니다. 먼저 split_folds.py를 실행해주세요.")
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    
    utils.log("데이터 로드 중...")
    raw_df = pd.read_csv(data_path)
    raw_df = utils.check_and_clean_data(df=raw_df, text_col='text', cfg=cfg) # 수정된 라인
    
    # 데이터 분할
    train_df, val_df = train_test_split(raw_df, test_size=0.3, stratify=raw_df['label'], random_state=cfg['seed'])
    utils.log(f"데이터 분할 완료: 학습 {len(train_df)}개, 검증 {len(val_df)}개")
    
    def is_valid_text(x):
        return (len(x.strip()) > 2) and (re.sub(r'[^\w가-힣]', '', x).strip() != '')

    train_df = train_df[train_df['text'].apply(is_valid_text)].reset_index(drop=True)
    val_df = val_df[val_df['text'].apply(is_valid_text)].reset_index(drop=True)
    utils.log(f"[라벨 분포] train: {train_df['label'].value_counts().to_dict()} val: {val_df['label'].value_counts().to_dict()}")
    utils.log(f"유효 텍스트 필터링 후 데이터: 학습 {len(train_df)}개, 검증 {len(val_df)}개")

    # 모델 및 토크나이저 로드
    utils.log("모델 및 토크나이저 로드 중...")
    start_time = time.time()
    model_id = cfg['deberta']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    utils.log("DeBERTa-v3-Large 모델 로드 중...")
    cfg_model = {"num_labels": 2, "pad_token_id": tok.pad_token_id}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16, **cfg_model, device_map="auto")
    
    utils.log("LoRA 어댑터 초기화 중...")
    # LoRA 설정 개선 - lora_dropout 및 bias 추가
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=cfg['deberta']['lora_r'], 
        lora_alpha=cfg['deberta']['lora_alpha'], 
        lora_dropout=0.05,  # 드롭아웃 추가
        bias="none",        # 명시적 bias 설정
        target_modules=["query_proj", "key_proj", "value_proj", "dense"]
    )
    model = get_peft_model(model, lora_config)
    utils.log(f"모델 준비 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"학습 디바이스: {DEVICE}")
    
    # 데이터셋 클래스 - 메모리 효율성 개선
    class DS(Dataset):
        def __init__(self, df, aug=False):
            self.df, self.aug = df, aug
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            t = self.df.iloc[idx]['text']
            if self.aug and cfg['augment']['enable'] and random.random() < cfg['augment']['p_noise']:
                t = utils.noise_text(t, mask_ratio=cfg['augment']['mask_ratio'], swap_ratio=cfg['augment']['swapcase_ratio'])
            enc = tok(t, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt')
            # 디바이스 전송을 배치 처리 시점으로 이동
            return {k: v.squeeze() for k, v in enc.items()} | {
                'labels': torch.tensor(int(self.df.iloc[idx]['label']))
            }
    
    # 학습률 검토 및 설정
    lr = cfg['deberta'].get('lr', 1e-5)  # 기본값 낮춤
    utils.log(f"하이퍼파라미터 설정: epochs={cfg['deberta']['epochs']}, batch_size={cfg['batch_size']['deberta']}, lr={lr:.1e}")
    
    utils.log("데이터로더 초기화 중...")
    train_dl = DataLoader(DS(train_df, aug=True), batch_size=cfg['batch_size']['deberta'], shuffle=True)
    val_dl = DataLoader(DS(val_df), batch_size=cfg['batch_size']['deberta'])
    
    optim = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(optim, 
                                           num_warmup_steps=int(len(train_dl)*0.1),  # 10% 워밍업
                                           num_training_steps=len(train_dl)*cfg['deberta']['epochs'])
    early = utils.EarlyStopping(patience=3, verbose=True)
    
    utils.log("학습 시작", is_important=True)
    progress = utils.ProgressTracker(cfg['deberta']['epochs'], "DeBERTa-v3-Large LoRA 학습")
    
    model.train()
    for epoch in range(cfg['deberta']['epochs']):
        # 한 에폭 시작
        epoch_start = time.time()
        total_loss = 0
        
        # Train loop
        train_progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['deberta']['epochs']} (학습)", leave=False)
        for b in train_progress:
            # 배치 데이터를 디바이스로 이동
            b = {k: v.to(DEVICE) for k, v in b.items()}
            
            optim.zero_grad()
            out = model(**b)
            out.loss.backward()
            
            # 그래디언트 클리핑 추가 - 안정성 향상
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()
            sched.step()
            total_loss += out.loss.item()
            train_progress.set_postfix({'loss': f"{out.loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dl)
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_progress = tqdm(val_dl, desc=f"Epoch {epoch+1}/{cfg['deberta']['epochs']} (검증)", leave=False)
        with torch.no_grad():
            for b in val_progress:
                # 배치 데이터를 디바이스로 이동
                b = {k: v.to(DEVICE) for k, v in b.items()}
                
                out = model(**b)
                val_loss += out.loss.item()
                
                # 정확도 계산
                logits = out.logits
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b['labels']).sum().item()
                total += len(b['labels'])
                
                val_progress.set_postfix({'loss': f"{out.loss.item():.4f}"})
        
        val_loss /= len(val_dl)
        accuracy = correct / total * 100
        
        # 에폭 결과 출력
        epoch_time = time.time() - epoch_start
        utils.log(f"[Epoch {epoch+1}/{cfg['deberta']['epochs']}] 학습 loss: {avg_loss:.5f}, 검증 loss: {val_loss:.5f}, 정확도: {accuracy:.2f}% (소요시간: {utils.format_time(epoch_time)})")
        
        # 진행상황 업데이트
        progress.update(epoch+1, f"loss={val_loss:.5f}, acc={accuracy:.2f}%")
        
        # Early stopping
        early(val_loss)
        if early.early_stop:
            utils.log("Early stopping 발동! 학습을 조기 종료합니다.", is_important=True)
            break
        
        model.train()
    
    utils.safe_makedirs(cfg['checkpoint_dir'])
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'deberta.pt')
    
    # 모델 저장 방식 개선 - 어댑터만 저장
    try:
        # 1. 어댑터 저장 방식 (권장)
        adapter_path = os.path.join(cfg['checkpoint_dir'], 'deberta_adapter')
        utils.safe_makedirs(adapter_path)
        model.save_pretrained(adapter_path)
        utils.log(f"어댑터 저장 완료: {adapter_path}")
        
        # 2. 전체 state_dict 저장 (호환성 유지)
        torch.save(model.state_dict(), ckpt_path)
        utils.log(f"모델 state_dict 저장 완료: {ckpt_path}")
    except Exception as e:
        utils.log(f"모델 저장 중 오류: {e}", level="ERROR")
    
    progress.finish(f"모델 저장 완료: {ckpt_path}")

    # 주요 객체 명시적 삭제
    utils.log("주요 학습 객체 삭제 시도...")
    try:
        del model
        del tok
        del raw_df
        del train_df
        del val_df
        del train_dl
        del val_dl
        del optim
        del sched
        utils.log("주요 학습 객체 삭제 완료.")
    except NameError as e:
        utils.log(f"객체 삭제 중 오류 (이미 삭제되었거나 정의되지 않았을 수 있음): {e}", level="WARNING")
    except Exception as e:
        utils.log(f"객체 삭제 중 예기치 않은 오류: {e}", level="ERROR")

    # 메모리 정리 코드 추가
    utils.log("메모리 정리 시작...")
    if torch.cuda.is_available():
        utils.log("CUDA 캐시 비우는 중...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        utils.log("CUDA 캐시 비우기 완료.")
    gc.collect()
    utils.log("메모리 정리 완료.")
    
    return val_loss

# ---------------- 추론 함수 (probs_model2) ----------------
def probs_model2(texts, cfg=None):
    if cfg is None:
        cfg = utils.load_config()

    model_id = cfg['deberta']['model_id']
    utils.log(f"{model_id} 모델 로드 중 (추론용)...")

    # 1. 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # 양자화 설정 단순화
    bnb_config = None
    torch_dtype = torch.bfloat16
    
    if cfg['deberta'].get('quant', {}).get('load_in_4bit', False):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg['deberta'].get('quant', {}).get('double_quant', True),
            bnb_4bit_quant_type=cfg['deberta'].get('quant', {}).get('quant_type', "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        utils.log("DeBERTa 4비트 양자화 설정 적용")

    # 2. 베이스 모델 로드 - 단순화된 코드
    model_config_args = {
        "num_labels": 2,
        "pad_token_id": tok.pad_token_id,
        "torch_dtype": torch_dtype,
        "device_map": "auto"
    }
    
    if bnb_config:
        model_config_args["quantization_config"] = bnb_config
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, **model_config_args
    )

    # 3. LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg['deberta']['lora_r'],
        lora_alpha=cfg['deberta']['lora_alpha'],
        lora_dropout=0.05,  # 드롭아웃 추가
        bias="none",        # 명시적 bias 설정
        target_modules=["query_proj", "key_proj", "value_proj", "dense"]
    )

    # 4. 모델 구성 및 가중치 로드 - 개선된 예외 처리
    model = None
    # 먼저 어댑터 방식으로 저장된 모델 체크
    adapter_path = os.path.join(cfg['checkpoint_dir'], 'deberta_adapter')
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'deberta.pt')
    
    try:
        if os.path.exists(adapter_path):
            utils.log(f"어댑터 형식 체크포인트 로드 중: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            utils.log("어댑터 로드 성공")
        elif os.path.exists(ckpt_path):
            utils.log(f"state_dict 체크포인트 로드 중: {ckpt_path}")
            model = get_peft_model(base_model, lora_config)
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
            utils.log("state_dict 로드 성공")
        else:
            utils.log(f"체크포인트 파일을 찾을 수 없음. 기본 모델로 진행", level="WARNING")
            model = get_peft_model(base_model, lora_config)
    except Exception as e:
        utils.log(f"모델 로드 중 오류: {e}. 기본 모델로 진행", level="ERROR")
        model = get_peft_model(base_model, lora_config)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()

    # 텍스트 처리
    cleaned_texts = []
    for t in texts:
        if pd.isna(t):
            cleaned_texts.append("")
            utils.log("NaN 텍스트를 빈 문자열로 처리", level="WARNING")
        else:
            cleaned_texts.append(utils.clean_text(str(t)))

    results = []
    batch_size = cfg.get('batch_size', {}).get('deberta_infer', 32)

    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=f"{model_id} 추론"):
        batch_texts = cleaned_texts[i:i+batch_size]
        encs = tok(batch_texts, truncation=True, padding='max_length',
                   max_length=cfg['max_length'], return_tensors='pt')

        # 디바이스로 이동
        input_ids = encs.input_ids.to(DEVICE)
        attention_mask = encs.attention_mask.to(DEVICE)
        token_type_ids = encs.token_type_ids.to(DEVICE) if 'token_type_ids' in encs else None

        model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            model_inputs['token_type_ids'] = token_type_ids

        with torch.no_grad():
            outs = model(**model_inputs)
            probs = torch.softmax(outs.logits, dim=1)[:, 1].detach().cpu().numpy()
            results.extend(probs)

    utils.log(f"{model_id} 추론 완료")
    return np.array(results)

# ---------------- main 함수 ----------------
def main():
    cfg = utils.load_config()
    try:
        utils.init_logging(cfg)
        utils.set_seed(cfg['seed'])
        start_time = time.time()
        train_deberta(cfg)
        utils.log(f"DeBERTa-v3-Large LoRA 학습 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
    finally:
        utils.close_logging()

if __name__ == '__main__':
    main()