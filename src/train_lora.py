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
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, padding_side='left')
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

# In train_lora.py QwenDataset class

    class QwenDataset(Dataset):
        def __init__(self, df, aug=False):
            self.df, self.aug = df, aug
            # self.tok는 외부에서 초기화된 tok 변수를 사용하도록 변경 (클래스 멤버로 저장)
            # 또는 train_lora 함수 내에서 tok을 이 클래스 생성자에 전달
            # 예: train_dl = DataLoader(QwenDataset(train_df, tok=tok, aug=True), ...)
            # 여기서는 train_lora 함수 스코프의 tok을 직접 참조한다고 가정

        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            t = self.df.iloc[idx]['text']
            label_val = self.df.iloc[idx]['label'] # 라벨 값 가져오기

            if self.aug and cfg['augment']['enable'] and random.random() < cfg['augment']['p_noise']:
                t = utils.noise_text(t, mask_ratio=cfg['augment']['mask_ratio'], swap_ratio=cfg['augment']['swapcase_ratio'])

            prompt = make_prompt(t) # "이 문장이 가짜인가요? {text}\n정답:"
            target = make_target(label_val) # "네" 또는 "아니오"

            full_text_for_input_ids = prompt + " " + target # 모델 입력으로 사용될 전체 텍스트

            # 1. 입력 ID 및 어텐션 마스크 생성 (전체 텍스트 기준, max_length로 패딩/자르기)
            # padding_side='left'가 토크나이저에 설정되어 있어야 함
            enc = tok(
                full_text_for_input_ids,
                truncation=True,
                padding='max_length',
                max_length=cfg['max_length'],
                return_tensors='pt'
            )
            input_ids = enc.input_ids.squeeze(0)
            attention_mask = enc.attention_mask.squeeze(0)

            # 2. 레이블 생성: input_ids를 복사한 후, 프롬프트 부분과 패딩 부분을 -100으로 마스킹
            labels = input_ids.clone()

            # 2a. 프롬프트 부분 마스킹:
            #       prompt 텍스트를 토큰화하여 실제 프롬프트 토큰 길이를 구함 (패딩/자르기 없이)
            #       주의: tok(prompt)와 tok(full_text_for_input_ids)에서 prompt 부분의 토큰화 결과가 다를 수 있음 (앞뒤 문맥)
            #       가장 안전한 방법은 prompt + " " 부분까지의 길이를 기준으로 하는 것.
            #       또는, target 이전까지를 모두 프롬프로 간주.
            #       여기서는 "prompt + 공백" 까지의 길이를 기준으로 마스킹 시도.
            prompt_with_space = prompt + " "
            # input_ids에서 prompt_with_space가 끝나는 지점을 찾아야 함.
            # 이는 쉽지 않으므로, target 텍스트가 시작되는 지점을 찾는 것이 더 나을 수 있음.
            # target 텍스트 ("네" 또는 "아니오") 토큰 ID를 찾아서 그 이전까지를 -100으로 마스킹.

            # 더 간단하고 일반적인 방법: 응답(target) 부분만 레이블로 남기고 나머지는 -100
            # "네" 또는 "아니오" 라는 짧은 응답을 예측하는 것이므로,
            # 전체 input_ids에서 마지막 몇 개의 토큰(응답 부분)만 레이블로 남깁니다.
            # 먼저, 프롬프트 부분의 길이를 (패딩/자르기 없는) 실제 토큰 수로 계산
            prompt_tokens = tok(prompt, add_special_tokens=False, truncation=False).input_ids
            len_prompt_tokens = len(prompt_tokens)

            # 레이블에서 프롬프트 부분은 모두 -100으로 마스킹
            # input_ids는 padding_side='left'이므로 프롬프트는 오른쪽에 있을 수 있음.
            # 토크나이저에 padding_side='left'가 잘 적용되었다면, 실제 내용은 오른쪽에 정렬됨.
            # | PAD PAD ... PAD | Prompt Text | Target Text |
            # 이 경우, 실제 내용 시작점부터 len_prompt_tokens 만큼이 프롬프트.

            # 실제 내용(non-pad)의 시작 인덱스 찾기
            if tok.padding_side == 'left':
                try:
                    first_real_token_idx = (input_ids != tok.pad_token_id).nonzero(as_tuple=True)[0][0].item()
                except IndexError: # 모두 패딩 토큰인 경우 (매우 짧은 입력)
                    first_real_token_idx = cfg['max_length'] # 전체 마스킹 유도
            else: # padding_side == 'right' (또는 기본값)
                first_real_token_idx = 0

            # 프롬프트 부분 마스킹
            # 실제 프롬프트 토큰들이 위치하는 구간을 -100으로 설정
            # 이 구간은 first_real_token_idx 부터 first_real_token_idx + len_prompt_tokens 까지.
            prompt_mask_end_idx = min(first_real_token_idx + len_prompt_tokens, cfg['max_length'])
            labels[first_real_token_idx:prompt_mask_end_idx] = -100

            # 2b. 패딩 부분 마스킹:
            #       input_ids에서 tok.pad_token_id인 부분을 모두 -100으로 마스킹
            #       (위에서 프롬프트 마스킹 후에도 패딩 영역이 남아있을 수 있으므로, 확실하게 다시 마스킹)
            labels[input_ids == tok.pad_token_id] = -100

            # 디버깅 출력 (이전 답변의 디버깅 코드 활용)
            if idx < 3: # 처음 3개 샘플에 대해 출력
                print(f"\n--- QwenDataset Debugging Sample {idx} ---")
                print(f"Raw Text: {self.df.iloc[idx]['text']}")
                print(f"Label: {label_val}")
                print(f"Prompt: {prompt}")
                print(f"Target: {target}")
                print(f"tokenizer.padding_side: {tok.padding_side}, tokenizer.pad_token_id: {tok.pad_token_id}")
                print(f"Input IDs (shape {input_ids.shape}):\n{input_ids.tolist()}")
                # print(f"Decoded Input IDs: {tok.decode(input_ids)}") # 전체 디코딩은 길 수 있음
                print(f"Attention Mask (shape {attention_mask.shape}):\n{attention_mask.tolist()}")
                print(f"Labels (shape {labels.shape}):\n{labels.tolist()}")
                active_labels = labels[labels != -100]
                print(f"Active Labels (tokens to predict, count {len(active_labels)}):\n{active_labels.tolist()}")
                if len(active_labels) > 0:
                    print(f"Decoded Active Labels: '{tok.decode(active_labels)}'")
                else:
                    print("Warning: No active labels found for this sample. All labels are -100.")
                print("---------------------------------------\n")


            # 모델은 input_ids와 attention_mask를 사용하고, labels와 비교하여 loss 계산
            return {
                'input_ids': input_ids,    # .to(DEVICE) 제거
                'attention_mask': attention_mask, # .to(DEVICE) 제거
                'labels': labels          # .to(DEVICE) 제거
            }

    epochs = epochs or int(os.environ.get('EPOCHS', cfg['mistral']['epochs']))
    batch_size = batch_size or int(os.environ.get('BATCH_SIZE', cfg['batch_size']['mistral']))
    lr = lr or float(os.environ.get('LR', 1e-4))
    utils.log(f"하이퍼파라미터 설정: epochs={epochs}, batch_size={batch_size}, lr={lr:.1e}")
    utils.log("데이터로더 초기화 중...")
    train_dl = DataLoader(QwenDataset(train_df, aug=True), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(QwenDataset(val_df), batch_size=batch_size)
    opt = AdamW(model.parameters(), lr=lr, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=len(train_dl) * epochs // 10, num_training_steps=len(train_dl)*epochs) # 수정된 라인

    early = utils.EarlyStopping(patience=3, verbose=True)
    utils.log("학습 시작", is_important=True)
    
    # NaN/Inf 감지 기능 활성화 (디버깅용)
    torch.autograd.set_detect_anomaly(True)
    
    progress = utils.ProgressTracker(epochs, "Qwen3-4B QLoRA 학습")
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        train_progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} (학습)", leave=False)
        for b in train_progress:
            opt.zero_grad()
            out = model(**b)

            if torch.isnan(out.logits).any() or torch.isinf(out.logits).any():
                utils.log(f"[nan/inf in logits DETECTED] logits sample: {out.logits.flatten()[:20]}", level="ERROR")
                # 문제가 되는 logits를 생성한 input_ids 등 로깅
                # raise ValueError("NaN/Inf in model logits") # 여기서 중단하여 확인
            if torch.isnan(out.loss) or torch.isnan(out.logits).any():
                utils.log(f"[nan 감지] loss: {out.loss}, logits: {out.logits}")
                utils.log(f"[nan 감지] input_ids: {b['input_ids']}")
            train_progress.set_postfix({'loss': f"{out.loss.item():.4f}"})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
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

    # 주요 객체 명시적 삭제
    utils.log("주요 학습 객체 삭제 시도...")
    try:
        del model
        del tok
        del raw # train_lora.py에서는 raw 변수 사용
        del train_df
        del val_df
        del train_dl
        del val_dl
        del opt # train_lora.py에서는 opt 변수 사용
        del sched
        # QwenDataset 클래스 인스턴스는 DataLoader에 의해 관리
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
    utils.log("가비지 컬렉션 실행 중...")
    gc.collect()
    utils.log("메모리 정리 완료.")

    return val_loss

# ---------------- 추론 함수 (probs_model1) ----------------
def probs_model1(texts, cfg=None): # cfg를 인자로 받거나 내부에서 로드하도록 수정
    if cfg is None:
        cfg = utils.load_config()

    model_id = cfg['mistral']['model_id'] # config.yaml에 따라 Qwen/Qwen3-4B
    utils.log(f"{model_id} 모델 로드 중 (추론용)...")

    # 1. 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True) # Qwen 모델은 trust_remote_code=True 필요
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # 2. 베이스 모델 로드 (양자화 없이 또는 추론용 양자화 설정)
    # 학습 시 사용한 BitsAndBytesConfig와 동일하게 하거나, 추론 성능에 맞춰 조정 가능
    # 여기서는 학습 시와 유사하게 설정 (더 빠른 추론을 원하면 양자화 없이 float16/bfloat16으로 로드)
    bnb_config_infer = BitsAndBytesConfig(
        load_in_4bit=cfg['mistral']['quant']['load_in_4bit'], # True 또는 False (추론 환경에 따라)
        bnb_4bit_use_double_quant=cfg['mistral']['quant']['double_quant'],
        bnb_4bit_quant_type=cfg['mistral']['quant']['quant_type'],
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg['mistral']['quant']['load_in_4bit'] else None # 양자화 시에만 유효
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config_infer if cfg['mistral']['quant']['load_in_4bit'] else None,
        torch_dtype=torch.bfloat16, # A6000에서 bfloat16 지원, 또는 float16
        device_map="auto", # 또는 특정 디바이스 지정
        trust_remote_code=True # Qwen 모델은 trust_remote_code=True 필요
    )

    # 3. LoRA 설정 정의 (학습 시와 동일하게)
    # Qwen3-4B (Qwen2 아키텍처)에 맞는 target_modules 설정
    # 이 부분은 실제 모델 구조를 확인하고 가장 적합한 모듈로 지정해야 합니다.
    # 일반적인 Qwen2 계열 타겟 모듈: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    qwen2_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # 또는 학습 시 저장된 어댑터의 config.json을 참고하여 target_modules를 가져올 수 있습니다.
    # 예: peft_config = PeftConfig.from_pretrained(os.path.join(cfg['checkpoint_dir'], 'qwen3-4b_adapter_weights')) # 어댑터만 저장했을 경우
    # target_modules = peft_config.target_modules

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg['mistral']['lora_r'],
        lora_alpha=cfg['mistral']['lora_alpha'],
        target_modules=qwen2_target_modules, # Qwen3-4B에 맞는 모듈 이름으로!
        # bias="none" # 필요시 추가
    )

    # 4. PEFT 모델 구성
    model = get_peft_model(base_model, lora_config)
    utils.log("PEFT 모델 구성 완료 (추론용)")

    # 5. 학습된 가중치 로드 (model.state_dict() 전체를 저장한 경우)
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'qwen3-4b.pt') # 파일 이름은 학습 시 저장한 이름과 동일하게
    if os.path.exists(ckpt_path):
        utils.log(f"체크포인트 로드 중: {ckpt_path}")
        # PEFT 모델의 state_dict를 로드합니다.
        # PeftModel의 state_dict는 어댑터 가중치와 베이스 모델 가중치를 포함할 수 있으나,
        # get_peft_model로 래핑된 모델에 직접 로드하는 것이 일반적입니다.
        # 만약 저장 시 model.save_pretrained()를 사용했다면, 아래처럼 로드합니다.
        # model = PeftModel.from_pretrained(base_model, ckpt_path) # ckpt_path가 어댑터 폴더일 경우
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False) # strict=False로 일부 불일치 허용
            utils.log("state_dict 로드 성공.")
        except RuntimeError as e:
            utils.log(f"state_dict 로드 중 오류 발생: {e}. 어댑터 가중치만 로드 시도...")
            # 이 부분은 어댑터 가중치만 별도로 저장했을 경우를 위한 예시입니다.
            # 현재 코드는 전체 state_dict를 저장하므로, 이 부분이 직접적으로 필요하지 않을 수 있습니다.
            # model.load_adapter(ckpt_path, adapter_name="default") # 어댑터 로드 방식
            # PeftModel.from_pretrained를 사용하는 것이 더 일반적입니다.
            utils.log(f"PeftModel.from_pretrained 방식으로 로드 시도: {ckpt_path}")
            try:
                # 이 방식은 ckpt_path가 어댑터 가중치가 저장된 디렉토리여야 합니다.
                # 현재 'qwen3-4b.pt'는 파일이므로 이 방식은 적합하지 않습니다.
                # model = PeftModel.from_pretrained(base_model, ckpt_path_directory) # ckpt_path가 디렉토리일 경우
                # 대신, 저장된 state_dict가 PEFT 모델 전체의 것이라면 위 load_state_dict가 맞습니다.
                # 만약 저장할 때 model.save_pretrained("adapter_path")로 어댑터만 저장했다면,
                # model = PeftModel.from_pretrained(base_model, "adapter_path") 로 로드합니다.
                # 현재는 model.state_dict()를 저장했으므로, get_peft_model 이후 load_state_dict가 맞습니다.
                # strict=False 옵션으로 시도해볼 수 있습니다.
                pass # 현재 구조에서는 위 load_state_dict(strict=False)로 충분할 수 있습니다.
            except Exception as e_peft:
                utils.log(f"PeftModel.from_pretrained 로드 실패: {e_peft}", level="ERROR")
                raise FileNotFoundError(f"체크포인트 파일 로드에 실패했습니다: {ckpt_path}. 저장 방식과 로드 방식을 확인하세요.")

    else:
        utils.log(f"체크포인트 파일이 없습니다: {ckpt_path}. 사전 학습된 모델 가중치를 사용합니다.", level="WARNING")
        # raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_path}") # 필요시 에러 발생

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.log(f"추론 디바이스: {DEVICE}")
    model.to(DEVICE)
    model.eval() # 평가 모드로 설정

    results = []
    batch_size = cfg.get('batch_size', {}).get('mistral_infer', 8) # 추론용 배치 크기, config.yaml에 mistral_infer 추가 권장

    # 텍스트 정제 (선택적, 학습 데이터와 동일하게 처리)
    cleaned_texts = []
    for t in texts:
        if pd.isna(t):
            cleaned_texts.append("") # 또는 적절한 기본값
            utils.log("입력 텍스트 중 NaN 값을 빈 문자열로 처리했습니다.", level="WARNING")
        else:
            cleaned_texts.append(utils.clean_text(str(t))) # cfg 전달

    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc=f"{model_id} 추론"):
        batch_texts = cleaned_texts[i:i+batch_size]
        # 프롬프트 형식은 학습 시와 동일하게
        prompts = [f"이 문장이 가짜인가요? {text.strip()}\n정답:" for text in batch_texts]

        encs = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=cfg['max_length'])
        input_ids = encs.input_ids.to(DEVICE)
        attention_mask = encs.attention_mask.to(DEVICE)

        with torch.no_grad():
            # 생성 설정은 학습 시 검증 단계와 유사하게
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,  # "네" 또는 "아니오" 와 같은 짧은 답변 생성
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                # num_beams=3, # 필요시 Beam Search 사용
                # do_sample=True, top_k=50, top_p=0.95, # 필요시 Sampling 사용
                # temperature=0.7 # 필요시 Temperature 조절
            )

        for j in range(len(generated_outputs)):
            # 생성된 전체 시퀀스에서 입력 프롬프트 부분을 제외하고 디코딩
            prompt_len = len(tok.encode(prompts[j], add_special_tokens=False)) # 입력 프롬프트의 길이
            # generated_ids_only = generated_outputs[j][prompt_len:] # generate가 입력까지 포함하여 출력하는 경우
            # 때로는 generate 함수가 입력 ID를 제외한 부분만 반환하기도 함. 확인 필요.
            # transformers 최신 버전에서는 input_ids를 제외한 부분만 반환하는 경우가 많음.
            # 만약 generate가 전체 시퀀스를 반환하면 아래처럼 슬라이싱 필요:
            # full_decoded_text = tok.decode(generated_outputs[j], skip_special_tokens=True)
            # generated_text_only = full_decoded_text[len(prompts[j]):].strip()

            # 더 안전한 방법: 생성된 토큰들 중 입력 프롬프트 이후의 부분만 디코딩
            # generate()는 기본적으로 input_ids를 포함한 전체 시퀀스를 반환하는 경우가 많음.
            # 하지만 max_new_tokens를 사용하면, 새롭게 생성된 토큰들만 반환할 수도 있음.
            # 정확한 동작은 transformers 버전에 따라 다를 수 있으므로,
            # generate()의 출력을 확인하는 것이 좋음.
            # 여기서는 generated_outputs가 새로운 토큰들만 포함한다고 가정하거나,
            # 또는 input_ids를 제외한 부분만 추출하는 로직이 필요.

            # 가장 일반적인 방식은 전체 시퀀스를 디코딩 후, 프롬프트 부분을 제거하는 것입니다.
            full_output_sequence = generated_outputs[j]
            decoded_full = tok.decode(full_output_sequence, skip_special_tokens=True)
            
            # 프롬프트가 생성된 텍스트의 시작 부분에 포함되어 있으므로, 프롬프트 길이만큼 잘라냄
            # 단, 토크나이저에 따라 정확한 프롬프트 문자열이 디코딩 결과와 다를 수 있으므로 주의
            # 여기서는 "\n정답:" 이후의 텍스트를 추출하는 방식으로 변경
            answer_part = decoded_full.split("\n정답:")[-1].strip()

            # utils.log(f"Decoded for text '{batch_texts[j][:30]}...': '{answer_part}' (Full: '{decoded_full}')") # 디버깅용

            if "네" in answer_part:
                results.append(1.0)
            elif "아니오" in answer_part:
                results.append(0.0)
            else:
                # 생성된 텍스트에 "네" 또는 "아니오"가 명확히 없으면 0.5로 처리
                # 또는 "네" 토큰과 "아니오" 토큰의 로짓 값을 직접 비교하여 확률을 계산할 수도 있음 (더 복잡)
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