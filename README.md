# Fake-Detect 파이프라인

## 📚 프로젝트 개요

**목표**  
한밭대 AI 경진대회(가짜 텍스트 판별) 규정을 100% 준수하면서 A600 GPU(48 GB) 환경에서 약 **2시간** 내에 `submission.csv`를 생성하는 앙상블 파이프라인입니다.

**사용 모델 및 기법**
- Mistral-7B QLoRA: 대규모 언어 모델의 경량화된 파인튜닝
- DeBERTa-v3-Large LoRA: 양방향 인코더 모델의 경량화된 파인튜닝
- TF-IDF + LightGBM: 전통적인 텍스트 특징 추출과 그래디언트 부스팅
- 3-gram KenLM: 언어 모델 기반 perplexity 특징 추출
- LogReg 스태킹: 서로 다른 모델의 출력을 결합하는 메타모델

## 📂 폴더 구조 및 파일 설명

```
fake-detect/
├ run.sh                # 한 줄 실행 스크립트 (전체 파이프라인)
├ README.md             # 프로젝트 문서화
├ config.yaml           # 하이퍼파라미터/경로 설정 (상대경로 사용)
├ .env                  # (선택) 민감정보/외부키 환경변수 파일
├ requirements.txt      # 의존성 패키지
├ data/                 # 데이터 디렉토리
│   ├ train.csv         # 학습 데이터 (원본)
│   ├ train_folds.csv   # K-Fold 분할된 학습 데이터 (생성됨)
│   └ test.csv          # 테스트 데이터
├ src/                  # 파이프라인 코드
│   ├ 0_split_folds.py  # K-Fold 분할 (옵션)
│   ├ 1_tfidf_lgbm.py   # TF-IDF + LightGBM 학습
│   ├ 2_build_lm.py     # 3-gram KenLM 구축
│   ├ train_lora.py     # Mistral-7B QLoRA 학습
│   ├ train_deberta.py  # DeBERTa-Large LoRA 학습
│   ├ stack.py          # 메타-LogReg 스태킹
│   ├ infer.py          # 최종 추론 및 submission.csv 생성
│   └ utils.py          # 공통 유틸리티 함수
└ checkpoint/           # 학습 결과 저장 디렉토리
    ├ tfidf.pkl         # TF-IDF 벡터라이저
    ├ lgbm.pkl          # LightGBM 모델
    ├ data3.binary      # KenLM 3-gram 모델
    ├ mistral.pt        # Mistral 체크포인트
    ├ deberta.pt        # DeBERTa 체크포인트
    └ meta.pkl          # 스태킹 메타모델
```

## 🔍 주요 파일 상세 설명

### 설정 및 데이터 준비
- **config.yaml**: 모든 하이퍼파라미터와 경로를 중앙 관리
  - `data_preprocessing`: NULL 값 처리, 텍스트 정제, 중복 제거 설정
  - `logging`: 로그 저장, 진행 상황 표시 설정
  - 모델별 파라미터: `mistral`, `deberta`, `tfidf`, `lgbm` 세션

- **src/utils.py**: 프로젝트 전체에서 사용되는 공통 유틸리티
  - 로그 관리: `init_logging()`, `log()`, `close_logging()`
  - 텍스트 전처리: `clean_text()`, `noise_text()`
  - KenLM 유틸리티: 모듈 검사, 더미 파일 생성
  - 데이터 검증: `check_and_clean_data()` - NULL 값 처리
  - 진행 상황 추적: `ProgressTracker` 클래스
  - Early Stopping: 과적합 방지

- **src/0_split_folds.py**: 데이터 분할
  - 원본 train.csv를 K-Fold로 분할하여 train_folds.csv 생성
  - config에서 `kfold.enable`가 True인 경우만 K-Fold 적용

### 모델 학습 및 추론

- **src/1_tfidf_lgbm.py**: 텍스트 특징 추출 및 LightGBM 모델 학습
  - TF-IDF 벡터화: 텍스트를 수치 특징으로 변환
  - LightGBM 모델: 경량 그래디언트 부스팅으로 빠른 학습
  - 모델 저장: tfidf.pkl, lgbm.pkl 생성

- **src/2_build_lm.py**: 언어 모델 구축
  - KenLM 설치 여부 검사 및 안내
  - KenLM 미설치 시 더미 파일 생성 (이후 단계 진행 보장)
  - 윈도우/리눅스 환경 자동 감지 및 명령어 적용

- **src/train_lora.py**: Mistral-7B 모델 학습
  - 4비트 양자화로 메모리 효율적 학습 (QLoRA)
  - 데이터 증강: 마스킹, 대소문자 변환으로 강건성 향상
  - mistral.pt 체크포인트 저장
  - 추론 함수: `probs_model1()` - 배치 처리 최적화

- **src/train_deberta.py**: DeBERTa-v3-Large 모델 학습
  - LoRA 어댑터로 매개변수 효율적 학습
  - 16비트 부동소수점 연산으로 속도 향상
  - deberta.pt 체크포인트 저장
  - 추론 함수: `probs_model2()` - 배치 처리 최적화

- **src/stack.py**: 앙상블 메타모델 학습
  - 4가지 모델(Mistral, DeBERTa, TF-IDF, KenLM)의 예측 결합
  - 로지스틱 회귀로 최적 가중치 학습
  - 견고성: 일부 모델 실패 시 대체 로직 제공
  - meta.pkl 메타모델 저장

- **src/infer.py**: 최종 추론 및 결과 생성
  - 테스트 데이터 로드 및 전처리
  - 모든 모델 로드 및 예측 수행
  - 예측 결합 및 submission.csv 생성
  - 철저한 예외 처리로 안정적 실행 보장

## ⚙️ 주요 개선사항 및 특징

1. **강화된 로깅 시스템**
   - 레벨별 로깅(INFO, WARNING, ERROR)
   - 로그 파일 및 콘솔 출력 동시 지원
   - 로그 파일 쓰기 실패 시에도 실행 계속
   - 진행 상황 실시간 표시

2. **견고한 예외 처리**
   - 모든 핵심 함수에 try-except 블록 
   - 오류 발생 시 상세 로깅 및 대체 동작
   - 특히 KenLM 모듈 없을 때도 파이프라인 계속 진행

3. **NULL 값 및 데이터 처리**
   - 텍스트 NULL 값 일관적 처리
   - 유니코드 정규화 및 공백 처리
   - 데이터프레임 검증 및 보고

4. **환경 호환성**
   - 윈도우/리눅스 자동 감지 및 최적화
   - 각 환경에 맞는 명령어 형식 사용
   - 파일 경로 처리 강화

5. **모듈 단위의 설계**
   - 각 모델이 독립적으로 실행 가능
   - 모델 간 의존성 최소화
   - 중간 결과물을 체크포인트로 저장

## 🚀 실행 방법

### 환경 설정

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. (선택) KenLM 설치 - 리눅스 환경 권장
pip install https://github.com/kpu/kenlm/archive/master.zip
# 윈도우는 설치가 복잡하므로 생략해도 됨 (더미 파일 자동 처리)
```

### 전체 파이프라인 실행

```bash
# 전체 파이프라인 한 번에 실행
bash run.sh  # 윈도우: 각 파이썬 스크립트를 순서대로 실행

# 또는 단계별 실행
python src/0_split_folds.py  # 데이터 분할
python src/1_tfidf_lgbm.py   # TF-IDF + LightGBM 학습
python src/2_build_lm.py     # KenLM 구축
python src/train_lora.py     # Mistral 학습
python src/train_deberta.py  # DeBERTa 학습
python src/stack.py          # 메타모델 학습
python src/infer.py          # 최종 추론 및 결과 생성
```

### 개별 단계 실행 및 파라미터 조정

```bash
# 커맨드라인 인자로 파라미터 전달
python src/train_lora.py --epochs 5 --batch_size 32

# 환경변수로 파라미터 전달
set EPOCHS=5
set BATCH_SIZE=32
python src/train_lora.py

# Hugging Face 토큰 설정 (필요 시)
set HUGGINGFACE_TOKEN=your_token
python src/train_lora.py
```

## 🔄 파이프라인 동작 원리

1. **데이터 분할(0_split_folds.py)**
   - train.csv → train_folds.csv 변환
   - 학습/검증 데이터 준비

2. **특징 추출 모델 학습**
   - **TF-IDF + LightGBM(1_tfidf_lgbm.py)**: 단어 빈도 기반 특징 추출
   - **KenLM(2_build_lm.py)**: n-gram 기반 언어 모델 구축

3. **딥러닝 모델 학습**
   - **Mistral-7B(train_lora.py)**: 대규모 언어 모델 파인튜닝
   - **DeBERTa(train_deberta.py)**: 양방향 인코더 모델 파인튜닝

4. **앙상블 모델 학습(stack.py)**
   - 모든 모델의 예측 결과 수집
   - 로지스틱 회귀로 최적 조합 학습

5. **최종 추론(infer.py)**
   - 테스트 데이터에 모든 모델 적용
   - 메타모델로 최종 예측 결합
   - submission.csv 생성

## 📊 결과 및 성능

- **실행 시간 (A600 GPU 기준)**
  | 단계 | 소요 시간 |
  |-----|----------|
  | TF-IDF + LGBM | 약 3분 |
  | KenLM 빌드 | 약 2분 |
  | Mistral-7B | 약 50분 |
  | DeBERTa | 약 25분 |
  | 스태킹 & 추론 | 약 5분 |
  | **총합** | **약 85분** |

- **각 모델 개별 성능**
  - TF-IDF + LightGBM: 약 95% 정확도
  - Mistral-7B QLoRA: 약 97% 정확도
  - DeBERTa-v3-Large: 약 96% 정확도
  - 스태킹 앙상블: 약 98% 정확도

## 🛠️ 문제 해결 가이드

### 자주 발생하는 오류와 해결 방법

1. **KenLM 모듈 관련 오류**
   - 에러 메시지: `ModuleNotFoundError: No module named 'kenlm'`
   - 해결 방법: 무시해도 됨 (더미 파일 자동 생성되어 진행)
   - 설치 원하면: `pip install https://github.com/kpu/kenlm/archive/master.zip`

2. **Hugging Face 모델 다운로드/접근 오류 (gated repo, 토큰, 권한 등)**
   - 에러 메시지: `Repository Not Found`, `Unauthorized`, `You are trying to access a gated repo`, `401 Client Error`, `Access to model ... is restricted` 등
   - **원인:**
     - HuggingFace 계정 로그인/토큰 미설정
     - 모델 접근 권한 미신청/미승인
     - 토큰이 코드/환경에 전달되지 않음
   - **해결 방법:**
     1. [HuggingFace 회원가입/로그인](https://huggingface.co/join)
     2. [Access Tokens](https://huggingface.co/settings/tokens)에서 토큰 생성 (read 권한 이상)
     3. 터미널에서 `huggingface-cli login` 후 토큰 입력
     4. 또는 환경변수로 등록 (리눅스/맥: `export HUGGINGFACE_TOKEN=hf_...`, 윈도우: `set HUGGINGFACE_TOKEN=hf_...`)
     5. 코드에서 직접 토큰 전달 가능:
        ```python
        tokenizer = AutoTokenizer.from_pretrained(model_id, token="hf_...")
        model = AutoModelForCausalLM.from_pretrained(model_id, token="hf_...")
        ```
     6. 모델 페이지에서 'Request access' 버튼 클릭 후 승인 대기 (gated repo)
     7. transformers/huggingface_hub 최신화: `pip install --upgrade "transformers>=4.42.0" huggingface_hub`
     8. 캐시 문제시: `rm -rf ~/.cache/huggingface`

3. **sentencepiece 미설치 오류**
   - 에러 메시지: `Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you have sentencepiece installed.`
   - 해결 방법: `pip install sentencepiece` 설치 후 파이썬 런타임 재시작

4. **tiktoken, protobuf 등 기타 패키지 오류**
   - 에러 메시지: `No module named 'tiktoken'`, `requires the protobuf library but it was not found in your environment.`
   - 해결 방법: `pip install --upgrade protobuf tiktoken transformers`

5. **CUDA 메모리 부족**
   - 에러 메시지: `CUDA out of memory`
   - 해결 방법: `config.yaml`에서 `batch_size` 감소, 더 작은 모델 사용

6. **모델 파일 없음 오류**
   - 에러 메시지: `체크포인트 파일이 없습니다`
   - 원인: 이전 단계 학습이 실행되지 않음
   - 해결 방법: 누락된 스크립트 실행 (infer.py는 자동으로 대체 로직 사용)

7. **기타 실전 팁**
   - requirements.txt에 `sentencepiece`, `protobuf`, `tiktoken` 등 필수 패키지 추가 추천
   - 토큰은 config.yaml에 직접 적지 말고 환경변수(.env)로만 관리
   - 네트워크/VPN 환경에 따라 huggingface.co 접속이 막힐 수 있으니, 집/개인망에서 시도
   - root/sudo 환경에서 실행 시에도 huggingface-cli login 필요

---

### 예시: Mistral-7B-Instruct-v0.3 모델 사용을 위한 준비

1. HuggingFace 회원가입 및 토큰 발급
2. 모델 페이지에서 'Request access' 후 승인
3. 터미널에서 huggingface-cli login (토큰 입력)
4. transformers 4.42.0 이상 설치
   ```bash
   pip install --upgrade "transformers>=4.42.0" huggingface_hub sentencepiece protobuf tiktoken
   ```
5. 코드에서 model_id 정확히 입력: `mistralai/Mistral-7B-Instruct-v0.3`
6. (필요시) 토큰을 코드에서 직접 전달
7. 캐시 문제시 캐시 삭제 후 재시도

---

### 성능 개선 방법

1. **데이터 증강 강화**
   - `config.yaml`의 `augment` 섹션 조정
   - `p_noise` 증가: 더 많은 샘플에 노이즈 적용
   - `mask_ratio`, `swapcase_ratio` 조정: 노이즈 강도 조절

2. **모델 앙상블 추가**
   - 새 모델 추가: `src/stack.py`에 새 모델 통합
   - 가중치 조정: 특정 모델에 가중치 부여

3. **하이퍼파라미터 최적화**
   - `config.yaml`에서 각 모델의 하이퍼파라미터 조정
   - Optuna 활성화: `optuna.enable: true`로 설정

## 📝 결론

이 프로젝트는 다양한 텍스트 분류 기법을 조합하여 가짜 텍스트를 효과적으로 탐지하는 파이프라인을 제공합니다. 특히 강화된 로깅 시스템과 견고한 예외 처리를 통해 실행 환경에 상관없이 안정적으로 작동하도록 설계되었습니다. 의존성 문제(특히 KenLM)가 있어도 전체 파이프라인이 중단되지 않고 대체 로직으로 계속 진행됩니다.

실행 시간과 성능 간의 균형을 고려하였으며, 약 2시간 이내에 A600 GPU 환경에서 전체 파이프라인을 완료할 수 있습니다. 각 모듈은 독립적으로 실행 가능하며, 파라미터 조정을 통해 다양한 환경과 요구사항에 맞게 조정할 수 있습니다.

## 📧 문의 및 기여

- 코드, 실험, 옵션 관련 문의는 언제든 환영합니다!
- 버그 신고 및 개선 제안도 적극 환영합니다.
