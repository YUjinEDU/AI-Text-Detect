import re, unicodedata, random, numpy as np, torch, os, yaml, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
from tqdm.auto import tqdm
import sys

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
    """Unicode NFKC normalize + collapse spaces + NULL 값 처리."""
    if pd.isna(txt) or txt is None:
        return ""  # NULL 값은 빈 문자열로 처리
    txt = str(txt)  # 숫자 등의 다른 데이터 타입이 있을 경우 변환
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def noise_text(txt: str,
               mask_ratio=0.15,
               swap_ratio=0.10) -> str:
    """Apply random [MASK] token + case swap."""
    if not txt:  # NULL 값이나 빈 문자열 처리
        return ""
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
_log_file = None
_log_initialized = False

def init_logging(cfg):
    """로깅 설정을 초기화합니다."""
    global _log_file, _log_initialized
    
    # 이미 초기화된 경우 중복 실행 방지
    if _log_initialized:
        return
    
    _log_initialized = True
    
    try:
        if cfg.get('logging', {}).get('save_log', False):
            log_path = cfg.get('logging', {}).get('log_file', 'log.txt')
            # 경로 확인 및 디렉토리 생성
            log_dir = os.path.dirname(log_path)
            if log_dir:
                safe_makedirs(log_dir)
            
            # 로그 파일 열기
            _log_file = open(log_path, 'a', encoding='utf-8')
            log(f"로그 파일 열림: {log_path}", is_important=True)
    except Exception as e:
        print(f"로그 파일 초기화 중 오류 발생: {str(e)}")
        print("콘솔 로깅만 활성화됩니다.")

def log(msg, is_important=False, level="INFO"):
    global _log_file
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] [{level}] {msg}"
    
    if is_important:
        border = "="*80
        console_msg = f"\n{border}\n{formatted_msg}\n{border}"
        log_msg = f"{border}\n{formatted_msg}\n{border}"
    else:
        console_msg = formatted_msg
        log_msg = formatted_msg
    
    # 콘솔에 출력
    print(console_msg)
    
    # 로그 파일에 기록
    if _log_file:
        try:
            _log_file.write(log_msg + "\n")
            _log_file.flush()  # 즉시 기록
        except Exception as e:
            print(f"로그 파일 쓰기 중 오류 발생: {str(e)}")
            # 파일 로깅에 실패하더라도 프로그램 실행은 계속되도록 함

def close_logging():
    """로깅 자원을 정리합니다."""
    global _log_file, _log_initialized
    if _log_file:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None
    _log_initialized = False

# ---------------- KenLM 유틸리티 ----------------
def check_kenlm_available():
    """KenLM 모듈이 사용 가능한지 확인합니다."""
    try:
        import kenlm
        return True
    except ImportError:
        return False

def is_dummy_kenlm_file(file_path):
    """파일이 더미 KenLM 파일인지 확인합니다."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read(15)  # DUMMY_KENLM_FILE 확인용
        return content == b'DUMMY_KENLM_FILE'
    except Exception:
        return False

def create_dummy_kenlm_file(path):
    """더미 KenLM 파일을 생성합니다."""
    try:
        with open(path, 'wb') as f:
            f.write(b'DUMMY_KENLM_FILE')
        log(f"더미 KenLM 파일을 생성했습니다: {path}")
        return True
    except Exception as e:
        log(f"더미 KenLM 파일 생성 중 오류 발생: {str(e)}", level="ERROR")
        return False

# ---------------- 데이터 전처리/검증 ----------------
def check_and_clean_data(df, text_col='text', label_col='label', verbose=True, cfg=None):
    """데이터프레임에서 NULL 값 체크 및 처리"""
    # config 로드 (None이면 기본값 사용)
    if cfg is None:
        cfg = load_config()
    
    # 데이터 처리 설정 가져오기
    data_cfg = cfg.get('data_preprocessing', {})
    handle_null = data_cfg.get('handle_null', True)
    clean_text_enabled = data_cfg.get('clean_text', True)
    remove_duplicates = data_cfg.get('remove_duplicates', False)
    
    total_rows = len(df)
    # NULL 값 확인
    null_text = df[text_col].isna().sum()
    if label_col in df.columns:
        null_label = df[label_col].isna().sum() 
    else:
        null_label = 0
    
    # 중복 행 확인
    duplicates = df.duplicated().sum()
    
    if verbose:
        log(f"데이터 검증 결과:")
        log(f"- 총 행 수: {total_rows}")
        log(f"- NULL 텍스트: {null_text} ({null_text/total_rows*100:.2f}%)")
        if label_col in df.columns:
            log(f"- NULL 라벨: {null_label} ({null_label/total_rows*100:.2f}%)")
        log(f"- 중복 행: {duplicates} ({duplicates/total_rows*100:.2f}%)")
    
    # NULL 텍스트 처리 (빈 문자열로 대체)
    if handle_null and null_text > 0:
        df[text_col] = df[text_col].fillna("").astype(str)
        if verbose:
            log(f"- NULL 텍스트를 빈 문자열로 대체했습니다.")
    
    # 텍스트 정제
    if clean_text_enabled:
        df[text_col] = df[text_col].map(clean_text)
        if verbose:
            log(f"- 텍스트 정제를 적용했습니다.")
    
    # 중복 행 제거
    if remove_duplicates and duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        if verbose:
            log(f"- {duplicates}개의 중복 행을 제거했습니다.")
    
    return df

# ---------------- 진행상황 추적 ----------------
class ProgressTracker:
    def __init__(self, total_steps, name="작업"):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.name = name
        self.step = 0
        
        # config 로드
        cfg = load_config()
        self.show_progress = cfg.get('logging', {}).get('show_progress', True)
        
        if self.show_progress:
            log(f"{name} 시작 (총 {total_steps} 단계)", is_important=True)
        
    def update(self, step=None, msg=None):
        if not self.show_progress:
            return
            
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        elapsed = time.time() - self.start_time
        progress = self.step / self.total_steps
        eta = elapsed / progress - elapsed if progress > 0 else 0
        
        if msg:
            status = f"{self.name} 진행 중: {self.step}/{self.total_steps} ({progress*100:.1f}%) - {msg}"
        else:
            status = f"{self.name} 진행 중: {self.step}/{self.total_steps} ({progress*100:.1f}%)"
            
        log(f"{status} - 경과: {format_time(elapsed)}, 예상 남은 시간: {format_time(eta)}")
        
    def finish(self, msg=None):
        if not self.show_progress:
            return
            
        elapsed = time.time() - self.start_time
        if msg:
            log(f"{self.name} 완료 - {msg} (소요시간: {format_time(elapsed)})", is_important=True)
        else:
            log(f"{self.name} 완료 (소요시간: {format_time(elapsed)})", is_important=True)

def format_time(seconds):
    """초를 시:분:초 형식으로 변환"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

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

# 환경변수 및 dotenv 로드
load_dotenv()  # .env 파일 자동 로딩

# 프로그램 종료시 로깅 자원 정리
import atexit
atexit.register(close_logging)

# (선택) wandb 연동 예시
# import wandb
# wandb.init(project='fake-detect', config=cfg)
