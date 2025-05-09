import os, pathlib, random, re, unicodedata, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils
import time

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
    raw = pd.read_csv(data_path)
    raw = utils.check_and_clean_data(raw)  # NULL 값 체크 및 처리
    
    # 데이터 분할
    train_df, val_df = train_test_split(raw, test_size=0.3, stratify=raw['label'], random_state=cfg['seed'])
    utils.log(f"데이터 분할 완료: 학습 {len(train_df)}개, 검증 {len(val_df)}개")
    
    # 모델 및 토크나이저 로드
    utils.log("모델 및 토크나이저 로드 중...")
    start_time = time.time()
    model_id = cfg['deberta']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    utils.log("DeBERTa-v3-Large 모델 로드 중...")
    cfg_model = {"num_labels": 2, "pad_token_id": tok.pad_token_id}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16, **cfg_model, device_map="auto")
    
    utils.log("LoRA 어댑터 초기화 중...")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg['deberta']['lora_r'], lora_alpha=cfg['deberta']['lora_alpha']))
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
    return val_loss

# ---------------- 추론 함수 (probs_model2) ----------------
def probs_model2(texts):
    cfg = utils.load_config()
    
    # 진행 상황 표시
    utils.log("DeBERTa-v3-Large 모델 로드 중...")
    model_id = cfg['deberta']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # 체크포인트 로드
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'deberta.pt')
    if os.path.exists(ckpt_path):
        utils.log(f"체크포인트 로드 중: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    else:
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_path}")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"추론 디바이스: {DEVICE}")
    model.to(DEVICE)
    model.eval()
    
    # 텍스트 NULL 값 체크 및 정제
    texts = [utils.clean_text(t) for t in texts]
    
    # 배치 처리로 메모리 효율 및 속도 향상
    results = []
    batch_size = 32  # 메모리에 맞게 조정
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    utils.log(f"DeBERTa-v3-Large 추론 시작 (총 {len(texts)}개, {num_batches}개 배치)")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa-v3-Large 추론"):
            batch_texts = texts[i:i+batch_size]
            encs = tok(batch_texts, truncation=True, padding='max_length', 
                       max_length=cfg['max_length'], return_tensors='pt')
            
            for k in encs:
                encs[k] = encs[k].to(DEVICE)
                
            outs = model(**encs)
            probs = torch.softmax(outs.logits, dim=-1)[:, 1].cpu().numpy()
            results.extend(probs)
    
    utils.log(f"DeBERTa-v3-Large 추론 완료")
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