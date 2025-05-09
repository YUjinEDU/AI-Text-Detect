import os, pathlib, random, re, unicodedata, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils
import time
import re


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
    model = get_peft_model(
        model, 
        LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=cfg['deberta']['lora_r'], 
            lora_alpha=cfg['deberta']['lora_alpha'], 
            target_modules=["query_proj", "key_proj", "value_proj", "dense"]
        )
    )
    utils.log(f"모델 준비 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"학습 디바이스: {DEVICE}")
    
    class DS(Dataset):
        def __init__(self, df, aug=False):
            self.df, self.aug = df, aug
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            t = self.df.iloc[idx]['text']
            if self.aug and cfg['augment']['enable'] and random.random() < cfg['augment']['p_noise']:
                t = utils.noise_text(t, mask_ratio=cfg['augment']['mask_ratio'], swap_ratio=cfg['augment']['swapcase_ratio'])
            enc = tok(t, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt')
            return {k: v.squeeze().to(DEVICE) for k, v in enc.items()} | {
                'labels': torch.tensor(int(self.df.iloc[idx]['label']), device=DEVICE)
            }
    
    utils.log(f"하이퍼파라미터 설정: epochs={cfg['deberta']['epochs']}, batch_size={cfg['batch_size']['deberta']}, lr={cfg['deberta']['lr']:.1e}")
    
    utils.log("데이터로더 초기화 중...")
    train_dl = DataLoader(DS(train_df, aug=True), batch_size=cfg['batch_size']['deberta'], shuffle=True)
    val_dl = DataLoader(DS(val_df), batch_size=cfg['batch_size']['deberta'])
    
    optim = AdamW(model.parameters(), lr=cfg['deberta']['lr'])
    sched = get_linear_schedule_with_warmup(optim, 0, len(train_dl)*cfg['deberta']['epochs'])
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
            optim.zero_grad()
            out = model(**b)
            out.loss.backward()
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
    torch.save(model.state_dict(), ckpt_path)
    
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
        # DS 클래스 인스턴스는 DataLoader에 의해 관리되므로 별도 삭제 불필요할 수 있음
        utils.log("주요 학습 객체 삭제 완료.")
    except NameError as e:
        utils.log(f"객체 삭제 중 오류 (이미 삭제되었거나 정의되지 않았을 수 있음): {e}", level="WARNING")
    except Exception as e:
        utils.log(f"객체 삭제 중 예기치 않은 오류: {e}", level="ERROR")

    # 메모리 정리 코드 추가
    utils.log("메모리 정리 시작...")
    import gc
    if torch.cuda.is_available():
        utils.log("CUDA 캐시 비우는 중...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        utils.log("CUDA 캐시 비우기 완료.")
    utils.log("가비지 컬렉션 실행 중...")
    gc.collect()
    utils.log("메모리 정리 완료.")
    
    return val_loss

# ---------------- 추론 함수 (probs_model2) ----------------
def probs_model2(texts, cfg=None): # cfg를 인자로 받거나 내부에서 로드하도록 수정
    if cfg is None:
        cfg = utils.load_config()

    model_id = cfg['deberta']['model_id']
    utils.log(f"{model_id} 모델 로드 중 (추론용)...")

    # 1. 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # BitsAndBytesConfig 설정 (DeBERTa용)
    # config.yaml에서 deberta.quant 설정을 읽어오도록 할 수도 있습니다.
    # 여기서는 직접 설정합니다.
    load_in_4bit_deberta = cfg['deberta'].get('quant', {}).get('load_in_4bit', True) # config.yaml에 추가 권장


    if load_in_4bit_deberta:
        bnb_config_deberta = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg['deberta'].get('quant', {}).get('double_quant', True),
            bnb_4bit_quant_type=cfg['deberta'].get('quant', {}).get('quant_type', "nf4"), # "nf4" 또는 "fp4"
            bnb_4bit_compute_dtype=torch.bfloat16 # 연산 시 사용할 데이터 타입
        )
        utils.log(f"DeBERTa BitsAndBytesConfig: load_in_4bit={bnb_config_deberta.load_in_4bit}, quant_type={bnb_config_deberta.bnb_4bit_quant_type}")
        quantization_config_arg = bnb_config_deberta
        torch_dtype_arg = None # quantization_config 사용 시 torch_dtype은 bnb_4bit_compute_dtype로 관리됨
    else:
        quantization_config_arg = None
        torch_dtype_arg = torch.bfloat16 # 양자화 안 할 경우 bfloat16 사용
        utils.log("DeBERTa 4-bit 양자화를 사용하지 않습니다. BF16/FP16으로 로드합니다.")


    utils.log("DeBERTa-v3-Large 모델 로드 중 (양자화 적용)...")
    # AutoModelForSequenceClassification 설정
    # num_labels는 자동으로 감지되거나, 명시적으로 설정할 수 있습니다.
    # 일반적으로는 config.json에 정의된 num_labels를 사용하거나,
    # fine-tuning 시에는 직접 지정합니다. (여기서는 2개 클래스)
    model_config_args = {"num_labels": 2, "pad_token_id": tok.pad_token_id}
    if quantization_config_arg: # 양자화 사용 시
        model_config_args["quantization_config"] = quantization_config_arg
    if torch_dtype_arg: # 양자화 미사용 시
         model_config_args["torch_dtype"] = torch_dtype_arg

    # 2. 베이스 모델 로드
    # 학습 시와 동일하게 num_labels, pad_token_id 등을 설정
    # 추론 시에는 일반적으로 양자화 없이 float16 또는 bfloat16으로 로드하여 속도를 높임
    # 여기서는 학습 시 설정(torch_dtype=torch.float16)을 따름
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2, # 학습 시와 동일
        pad_token_id=tok.pad_token_id, # 학습 시와 동일
        torch_dtype=torch.bfloat16, # 학습 시와 동일 또는 bfloat16/float32
        # device_map="auto" # 추론 시에는 단일 GPU 사용이 일반적이므로 제거하거나 명시적 디바이스 지정
    )

    # 3. LoRA 설정 정의 (학습 시와 동일하게)
    # DeBERTa-v3-Large에 맞는 target_modules 설정
    # 이 부분은 실제 모델 구조를 확인하고 가장 적합한 모듈로 지정해야 합니다.
    # 예: ["query_proj", "key_proj", "value_proj", "dense"] 또는 DeBERTa V2/V3의 일반적인 어텐션/MLP 레이어 이름
    # 정확한 모듈명은 학습 스크립트의 LoraConfig와 동일해야 합니다.
    # 예시 (학습 스크립트와 동일하게 맞춰야 함, 아래는 일반적인 예시일 뿐임):
    deberta_target_modules = ["query_proj", "key_proj", "value_proj", "dense", "attention.self.query_layer", "attention.self.key_layer", "attention.self.value_layer", "intermediate.dense", "output.dense"]
    # 실제 사용된 target_modules를 확인하고 적용해야 합니다. config.yaml에 저장해두는 것이 좋습니다.
    # 학습 시 target_modules를 지정하지 않았다면, PEFT가 자동으로 선택한 모듈을 알아내거나,
    # 일반적인 DeBERTa LoRA 타겟 모듈을 사용해야 합니다.
    # 가장 안전한 방법은 학습 스크립트의 LoraConfig와 동일하게 설정하는 것입니다.
    # 지금은 학습 스크립트에 target_modules가 없으므로, 일반적인 DeBERTa용 LoRA 타겟을 임시로 추가합니다.
    # 하지만 이 부분은 실제 학습에 사용된 (또는 사용될) 모듈과 일치해야 합니다.
    # 예를 들어, DeBERTa v3의 경우 다음이 일반적입니다.
    default_deberta_lora_targets = ["query_proj", "key_proj", "value_proj", "intermediate.dense", "output.dense", "attention.output.dense"]


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg['deberta']['lora_r'],
        lora_alpha=cfg['deberta']['lora_alpha'],
        # target_modules 지정 필수! 학습 시 사용된 모듈과 동일하게!
        # 만약 학습 때 명시하지 않았다면, 여기서라도 적절한 값을 찾아야 함.
        # print(base_model) 로 모델 구조를 확인하여 Linear 레이어 이름을 찾아야 합니다.
        target_modules=cfg['deberta'].get('lora_target_modules', default_deberta_lora_targets) # config.yaml에 lora_target_modules 추가 권장
    )

    # 4. PEFT 모델 구성
    model = get_peft_model(base_model, lora_config)
    utils.log("PEFT 모델 구성 완료 (추론용)")

    # 5. 학습된 가중치 로드
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'deberta.pt')
    if os.path.exists(ckpt_path):
        utils.log(f"체크포인트 로드 중: {ckpt_path}")
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False) # strict=False 시도
            utils.log("state_dict 로드 성공.")
        except RuntimeError as e:
            utils.log(f"state_dict 로드 중 오류 발생: {e}. 저장 방식과 로드 방식을 확인하세요.", level="ERROR")
            # 어댑터만 저장한 경우의 로드 방식 (참고용)
            # model = PeftModel.from_pretrained(base_model, ckpt_path_directory) # ckpt_path가 디렉토리일 경우
            raise FileNotFoundError(f"체크포인트 파일 로드에 실패했습니다: {ckpt_path}")

    else:
        utils.log(f"체크포인트 파일이 없습니다: {ckpt_path}. 사전 학습된 모델 가중치를 사용합니다.", level="WARNING")
        # raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_path}") # 필요시 에러 발생

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"추론 디바이스: {DEVICE}")
    model.to(DEVICE)
    model.eval() # 평가 모드로 설정

    # 텍스트 NULL 값 체크 및 정제
    cleaned_texts = []
    for t in texts:
        if pd.isna(t):
            cleaned_texts.append("") # 또는 적절한 기본값
            utils.log("입력 텍스트 중 NaN 값을 빈 문자열로 처리했습니다.", level="WARNING")
        else:
            cleaned_texts.append(utils.clean_text(str(t))) # cfg 전달

    results = []
    # config.yaml에 deberta_infer 배치 크기 설정 권장
    batch_size = cfg.get('batch_size', {}).get('deberta_infer', cfg.get('batch_size', {}).get('deberta', 32))


    num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    utils.log(f"{model_id} 추론 시작 (총 {len(cleaned_texts)}개, {num_batches}개 배치)")

    with torch.no_grad():
        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=f"{model_id} 추론"):
            batch_texts = cleaned_texts[i:i+batch_size]
            encs = tok(batch_texts, truncation=True, padding='max_length',
                       max_length=cfg['max_length'], return_tensors='pt')

            # 생성된 텐서를 DEVICE로 이동
            input_ids = encs.input_ids.to(DEVICE)
            attention_mask = encs.attention_mask.to(DEVICE)
            # DeBERTa는 token_type_ids를 사용할 수 있음 (없어도 동작은 함)
            token_type_ids = encs.token_type_ids.to(DEVICE) if 'token_type_ids' in encs else None

            model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if token_type_ids is not None:
                model_inputs['token_type_ids'] = token_type_ids

            outs = model(**model_inputs)
            # probs = torch.softmax(outs.logits, dim=-1)[:, 1].cpu().numpy() # label 1일 확률
            # 모델이 (batch_size, num_labels) 형태의 로짓을 반환한다고 가정
            # 가짜일 확률 (label 1)을 가져옵니다.
            probs = torch.softmax(outs.logits, dim=1)[:, 1].detach().cpu().numpy()
            results.extend(probs)

    utils.log(f"{model_id} 추론 완료")
    return np.array(results)
# ---------------- main 함수 ----------------
def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 시드 설정
        utils.set_seed(cfg['seed'])
        
        # 학습 실행
        start_time = time.time()
        train_deberta(cfg)
        utils.log(f"DeBERTa-v3-Large LoRA 학습 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main() 