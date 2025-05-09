import os, pathlib, random, re, unicodedata, torch, pandas as pd, numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import utils

# ---------------- Main Training Function ----------------
def train_deberta(cfg):
    utils.set_seed(cfg['seed'])
    data_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    raw = pd.read_csv(data_path)
    raw['text'] = raw['text'].map(utils.clean_text)
    train_df, val_df = train_test_split(raw, test_size=0.3, stratify=raw['label'], random_state=cfg['seed'])
    model_id = cfg['deberta']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cfg_model = {"num_labels": 2, "pad_token_id": tok.pad_token_id}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16, **cfg_model, device_map="auto")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg['deberta']['lora_r'], lora_alpha=cfg['deberta']['lora_alpha']))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    train_dl = DataLoader(DS(train_df, aug=True), batch_size=cfg['batch_size']['deberta'], shuffle=True)
    val_dl = DataLoader(DS(val_df), batch_size=cfg['batch_size']['deberta'])
    optim = AdamW(model.parameters(), lr=cfg['deberta']['lr'])
    sched = get_linear_schedule_with_warmup(optim, 0, len(train_dl)*cfg['deberta']['epochs'])
    early = utils.EarlyStopping(patience=3, verbose=True)
    model.train()
    for epoch in range(cfg['deberta']['epochs']):
        total_loss = 0
        for b in tqdm(train_dl, desc=f"DeBERTa E{epoch+1}"):
            optim.zero_grad(); out = model(**b); out.loss.backward(); optim.step(); sched.step()
            total_loss += out.loss.item()
        avg_loss = total_loss / len(train_dl)
        utils.log(f"[Epoch {epoch+1}] train loss: {avg_loss:.5f}")
        # (옵션) 검증 loss/EarlyStopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b in val_dl:
                out = model(**b)
                val_loss += out.loss.item()
        val_loss /= len(val_dl)
        utils.log(f"[Epoch {epoch+1}] val loss: {val_loss:.5f}")
        early(val_loss)
        if early.early_stop:
            utils.log("Early stopping triggered!")
            break
        model.train()
    torch.save(model.state_dict(), os.path.join(cfg['checkpoint_dir'], 'deberta.pt'))
    utils.log('모델 저장 완료')

# ---------------- 추론 함수 (probs_model2) ----------------
def probs_model2(texts):
    cfg = utils.load_config()
    model_id = cfg['deberta']['model_id']
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'deberta.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    results = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, padding='max_length', max_length=cfg['max_length'], return_tensors='pt')
            for k in enc:
                enc[k] = enc[k].to(DEVICE)
            out = model(**enc)
            prob = torch.softmax(out.logits, dim=-1)[0,1].item()
            results.append(prob)
    return np.array(results)

# ---------------- main 함수 ----------------
def main():
    cfg = utils.load_config()
    train_deberta(cfg)

if __name__ == '__main__':
    main() 