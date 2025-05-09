# Fake‑Detect Pipeline

**목표**  
한밭대 AI 경진대회(가짜 텍스트 판별) 규정을 100 % 준수하면서
A600 GPU(48 GB) 환경에서 약 **2 시간** 안에 `submission.csv` 를 생성하는
Mistral‑7B QLoRA + DeBERTa‑Large LoRA + TF‑IDF LightGBM + 3‑gram KenLM 스태킹 파이프라인입니다.

---

## 📂 폴더 구조

```text
fake-detect/
├ run.sh                # ★ 한 줄 실행 스크립트 (스마트 환경 지원)
├ README.md             # (본 파일)
├ config.yaml           # 하이퍼파라미터·경로 설정 (상대경로 권장)
├ .env                  # (선택) 민감정보/외부키 환경변수 파일
├ requirements.txt      # 의존성 패키지
├ data/                 # train.csv, test.csv
├ src/                  # 파이프라인 코드
│   ├ 0_split_folds.py  # K‑Fold 분할 (옵션)
│   ├ 1_tfidf_lgbm.py   # TF‑IDF + LightGBM
│   ├ 2_build_lm.py     # 3‑gram KenLM
│   ├ train_lora.py     # Mistral‑7B QLoRA (스마트 인자/환경변수 지원)
│   ├ train_deberta.py  # DeBERTa‑Large LoRA
│   ├ stack.py          # 메타‑LogReg 스태킹
│   ├ infer.py          # submission.csv 생성
│   └ utils.py          # 공통 유틸 함수 (dotenv, wandb 예시 포함)
└ checkpoint/           # 학습 결과 자동 저장
```

---

## ⚙️ 설치 및 환경 세팅

```bash
python -m venv venv && source venv/bin/activate  # (윈도우: venv\Scripts\activate)
pip install -r requirements.txt
```
- **python-dotenv**: .env 파일 자동 로딩
- **wandb**: requirements.txt에 주석, 필요시 주석 해제 후 사용

---

## 🚀 실행 방법 (스마트 환경)

### 1. .env 파일(선택, 민감정보/외부키)
```
HUGGINGFACE_TOKEN=your_token
WANDB_API_KEY=your_wandb_key
```

### 2. config.yaml에서 경로/옵션/하이퍼파라미터 관리
- data_dir, checkpoint_dir, result_csv 등은 반드시 상대경로로!

### 3. run.sh 한 줄 실행 (전체 파이프라인)
```bash
bash run.sh
```

### 4. 커맨드라인 인자/환경변수로 실험 제어 (예시)
```bash
# 커맨드라인 인자
python src/train_lora.py --epochs 5 --batch_size 32
# 환경변수
set EPOCHS=5
set BATCH_SIZE=32
python src/train_lora.py
```
- argparse > 환경변수 > config.yaml 순서로 우선 적용

### 5. wandb 연동 (선택, 주석 해제 후 사용)
- requirements.txt, utils.py, train_lora.py 등에서 wandb 관련 코드 주석 해제
- .env 또는 환경변수에 WANDB_API_KEY 등록

---

## 📝 파일별 역할

| 스크립트                | 핵심 작업                              |
| ------------------- | ---------------------------------- |
| **0_split_folds.py**   | 5‑fold 분할 (옵션)                     |
| **1_tfidf_lgbm.py**    | 클린 텍스트 → TF‑IDF → LightGBM 학습      |
| **2_build_lm.py**      | train 텍스트로 3‑gram KenLM 빌드         |
| **train_lora.py**      | Mistral‑7B 4‑bit QLoRA (노이즈 증강 포함, 스마트 인자/환경변수 지원) |
| **train_deberta.py**   | DeBERTa‑v3‑Large LoRA 학습           |
| **stack.py**           | val 셋 예측으로 Logistic Regression 스태킹 |
| **infer.py**           | 테스트셋 예측 → submission.csv           |
| **utils.py**           | 공통 유틸 함수, dotenv, wandb 예시 포함   |

---

## 📜 민감 정보/외부 서비스 키 관리
- .env 파일에 HUGGINGFACE_TOKEN, WANDB_API_KEY 등 저장
- 코드에서는 os.environ 또는 config.yaml에서 읽어 사용
- wandb, huggingface 등 외부 서비스 연동은 환경변수만 등록하면 바로 동작

---

## 📜 실험/옵션/튜닝 관리
- config.yaml에서 모든 경로/옵션/튜닝 관리
- 커맨드라인 인자/환경변수로 실험 반복/자동화 가능
- Optuna, EarlyStopping, 증강 등 고급 기능 내장
- wandb, tensorboard 등 실험 로깅(주석 해제 시)

---

## 📜 실행 시간 (A600 GPU 기준)
| 단계               | 시간               |
| ---------------- | ---------------- |
| TF‑IDF + LGBM    | 3 min            |
| KenLM 빌드         | 2 min            |
| Mistral‑7B QLoRA | ~50 min         |
| DeBERTa LoRA     | ~25 min         |
| 스태킹 & 추론         | 5 min            |
| **총합**           | **≈ 1 h 25 min** |

(옵션) K‑Fold ON → ×5, Optuna 20 trial → +45 min

---

## ✉️ 문의
- 코드/실험/옵션 관련 문의는 언제든 환영합니다!

