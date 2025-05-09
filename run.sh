#!/bin/bash
# Fake-Detect 파이프라인 한 줄 실행 (스마트 환경)
#
# - .env 파일에 민감정보/외부키(HUGGINGFACE_TOKEN, WANDB_API_KEY 등) 저장 가능
# - config.yaml에서 경로/옵션/튜닝 관리 (반드시 상대경로!)
# - 커맨드라인 인자/환경변수로 실험 반복/자동화 가능
# - wandb 등 외부 서비스 연동은 주석 해제 후 사용
#
# 예시:
#   python src/train_lora.py --epochs 5 --batch_size 32
#   set EPOCHS=5 && python src/train_lora.py
#
# 전체 파이프라인 자동 실행
python src/0_split_folds.py
python src/1_tfidf_lgbm.py
python src/2_build_lm.py
python src/train_lora.py
python src/train_deberta.py
python src/stack.py
python src/infer.py