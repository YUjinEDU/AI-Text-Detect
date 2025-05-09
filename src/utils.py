import re, unicodedata, random, numpy as np, torch, os, yaml
from datetime import datetime
from dotenv import load_dotenv

# ---------------- Safe makedirs ----------------
def safe_makedirs(path):
    """빈 문자열이 아니면 디렉토리 생성 (경로 문제 100% 방지)"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------------- Config Loader ----------------
def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ---------------- Text utils ----------------
def clean_text(txt: str) -> str:
    """Unicode NFKC normalize + collapse spaces."""
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def noise_text(txt: str,
               mask_ratio=0.15,
               swap_ratio=0.10) -> str:
    """Apply random [MASK] token + case swap."""
    toks = ['[MASK]' if random.random() < mask_ratio else w
            for w in txt.split()]
    txt = ' '.join(toks)
    return ''.join(ch.swapcase() if random.random() < swap_ratio and ch.isalpha() else ch
                   for ch in txt)

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------- Logging ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ---------------- EarlyStopping ----------------
class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                log(f"Validation loss improved to {val_loss:.5f}")
        else:
            self.counter += 1
            if self.verbose:
                log(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ---------------- Optuna objective helper ----------------
def objective(trial, train_fn, cfg):
    lr  = trial.suggest_loguniform("lr", 1e-5, 3e-4)
    r   = trial.suggest_categorical("lora_r", [4, 8, 16])
    alpha = trial.suggest_int("lora_alpha", 8, 64)
    # cfg를 복사해 하이퍼파라미터를 trial 값으로 덮어씀
    cfg = dict(cfg)
    cfg['mistral']['lr'] = lr
    cfg['mistral']['lora_r'] = r
    cfg['mistral']['lora_alpha'] = alpha
    acc = train_fn(cfg)
    return acc

# Optuna 사용 예시 (train_lora.py 등에서)
# import optuna
# study = optuna.create_study(direction='maximize')
# study.optimize(lambda trial: utils.objective(trial, train_lora, cfg), n_trials=cfg['optuna']['trials'])

load_dotenv()  # .env 파일 자동 로딩
# (선택) wandb 연동 예시
# import wandb
# wandb.init(project='fake-detect', config=cfg)
