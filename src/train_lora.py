import os, pathlib, random, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import utils
from sklearn.model_selection import train_test_split
import argparse
import time

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
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    bnb = BitsAndBytesConfig(load_in_4bit=cfg['mistral']['quant']['load_in_4bit'],
                            bnb_4bit_use_double_quant=cfg['mistral']['quant']['double_quant'],
                            bnb_4bit_quant_type=cfg['mistral']['quant']['quant_type'],
                            bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True 
    )
    for name, module in model.named_modules(): print(name) 
    
    utils.log("LoRA 어댑터 초기화 중...")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg['mistral']['lora_r'],
            lora_alpha=cfg['mistral']['lora_alpha'],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target modules for LoRA
        )
    )
    utils.log(f"모델 준비 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"학습 디바이스: {DEVICE}")

    # 프롬프트/타겟 생성 함수
    def make_prompt(text):
        return f"이 문장이 가짜인가요? {text.strip()}\n정답:"
    def make_target(label):
        return ("네" if label == 1 else "아니오").strip()

    class QwenDataset(Dataset):
        def __init__(self, df, aug=False):
            self.df, self.aug = df, aug
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            t = self.df.iloc[idx]['text']
            if self.aug and cfg['augment']['enable'] and random.random() < cfg['augment']['p_noise']:
                t = utils.noise_text(t, mask_ratio=cfg['augment']['mask_ratio'], swap_ratio=cfg['augment']['swapcase_ratio'])
            prompt = make_prompt(t)
            target = make_target(self.df.iloc[idx]['label'])
            full = prompt + " " + target
            enc = tok(full, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt')
            input_ids = enc.input_ids.squeeze(0)
            labels = input_ids.clone()
            # 프롬프트 부분은 -100으로 마스킹
            prompt_enc = tok(prompt, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt')
            prompt_len = (prompt_enc.input_ids.squeeze(0) != tok.pad_token_id).sum().item()
            labels[:prompt_len] = -100
            return {
                'input_ids': input_ids.to(DEVICE),
                'attention_mask': enc.attention_mask.squeeze(0).to(DEVICE),
                'labels': labels.to(DEVICE)
            }

    epochs = epochs or int(os.environ.get('EPOCHS', cfg['mistral']['epochs']))
    batch_size = batch_size or int(os.environ.get('BATCH_SIZE', cfg['batch_size']['mistral']))
    lr = lr or float(os.environ.get('LR', 5e-6))
    utils.log(f"하이퍼파라미터 설정: epochs={epochs}, batch_size={batch_size}, lr={lr:.1e}")
    utils.log("데이터로더 초기화 중...")
    train_dl = DataLoader(QwenDataset(train_df, aug=True), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(QwenDataset(val_df), batch_size=batch_size)
    opt = AdamW(model.parameters(), lr=lr, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, en(train_dl) * epochs // 20, len(train_dl)*epochs)
    early = utils.EarlyStopping(patience=3, verbose=True)
    utils.log("학습 시작", is_important=True)
    progress = utils.ProgressTracker(epochs, "Qwen3-4B QLoRA 학습")
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        train_progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} (학습)", leave=False)
        for b in train_progress:
            opt.zero_grad()
            out = model(**b)
            if torch.isnan(out.loss) or torch.isnan(out.logits).any():
                utils.log(f"[nan 감지] loss: {out.loss}, logits: {out.logits}")
                utils.log(f"[nan 감지] input_ids: {b['input_ids']}")
            train_progress.set_postfix({'loss': f"{out.loss.item():.4f}"})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            sched.step()
            total_loss += out.loss.item()
        avg_loss = total_loss / len(train_dl)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_progress = tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} (검증)", leave=False)
        with torch.no_grad():
            for b in val_progress:
                out = model(**b)
                val_loss += out.loss.item()
                # 생성 결과로 정확도 계산
                gen_out = model.generate(
                    input_ids=b['input_ids'],
                    attention_mask=b['attention_mask'],
                    max_new_tokens=3
                )
                for i in range(len(gen_out)):
                    decoded = tok.decode(gen_out[i], skip_special_tokens=True)
                    if "네" in decoded:
                        pred = 1
                    elif "아니오" in decoded:
                        pred = 0
                    else:
                        pred = -1
                    if pred == (train_df.iloc[i]['label'] if b['labels'][i][0] != -100 else val_df.iloc[i]['label']):
                        correct += 1
                    total += 1
                val_progress.set_postfix({'loss': f"{out.loss.item():.4f}"})
        val_loss /= len(val_dl)
        accuracy = correct / total * 100 if total > 0 else 0
        epoch_time = time.time() - epoch_start
        utils.log(f"[Epoch {epoch+1}/{epochs}] 학습 loss: {avg_loss:.5f}, 검증 loss: {val_loss:.5f}, 정확도: {accuracy:.2f}% (소요시간: {utils.format_time(epoch_time)})")
        progress.update(epoch+1, f"loss={val_loss:.5f}, acc={accuracy:.2f}%")
        early(val_loss)
        if early.early_stop:
            utils.log("Early stopping 발동! 학습을 조기 종료합니다.", is_important=True)
            break
        model.train()
    utils.safe_makedirs(cfg['checkpoint_dir'])
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'qwen3-4b.pt')
    torch.save(model.state_dict(), ckpt_path)
    progress.finish(f"모델 저장 완료: {ckpt_path}")
    return val_loss

# ---------------- 추론 함수 (probs_model1) ----------------
def probs_model1(texts):
    cfg = utils.load_config()
    model_id = cfg['mistral']['model_id']
    utils.log("Qwen3-4B 모델 로드 중...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'qwen3-4b.pt')
    if os.path.exists(ckpt_path):
        utils.log(f"체크포인트 로드 중: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    else:
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_path}")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"추론 디바이스: {DEVICE}")
    model.to(DEVICE)
    model.eval()
    results = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size), desc="Qwen3-4B 추론"):
        batch_texts = texts[i:i+batch_size]
        prompts = [f"이 문장이 가짜인가요? {t}\n정답:" for t in batch_texts]
        encs = tok(prompts, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=encs['input_ids'],
                attention_mask=encs['attention_mask'],
                max_new_tokens=3
            )
        for j in range(len(gen_out)):
            decoded = tok.decode(gen_out[j], skip_special_tokens=True)
            if "네" in decoded:
                results.append(1.0)
            elif "아니오" in decoded:
                results.append(0.0)
            else:
                results.append(0.5)
    utils.log(f"Qwen3-4B 추론 완료")
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