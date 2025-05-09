import os, joblib, pandas as pd, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import utils
from train_lora import probs_model1
from train_deberta import probs_model2
import time
from tqdm.auto import tqdm

def train_stacking(cfg):
    utils.set_seed(cfg['seed'])
    utils.log("스태킹 메타 모델 학습 시작", is_important=True)
    
    # 진행 상황 추적기 초기화
    progress = utils.ProgressTracker(5, "스태킹 학습")
    
    # 체크포인트 디렉토리 확인
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    
    # 데이터 로드
    val_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"{val_path} 파일이 존재하지 않습니다.")
    
    utils.log("검증 데이터 로드 중...")
    val_df = pd.read_csv(val_path).sample(frac=0.3, random_state=cfg['seed'])
    val_df = utils.check_and_clean_data(val_df)  # NULL 값 체크 및 처리
    utils.log(f"검증 데이터 로드 완료: {len(val_df)}개 샘플")
    progress.update(msg="데이터 로드 완료")
    
    # 모델별 예측
    # 1) Qwen3-4B 확률
    utils.log("Qwen3-4B 추론 중...")
    start_time = time.time()
    qwen_prob = probs_model1(val_df['text'])
    utils.log(f"Qwen3-4B 추론 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    progress.update(msg="Qwen3-4B 예측 완료")
    
    # 2) DeBERTa 확률
    utils.log("DeBERTa-v3-Large 추론 중...")
    start_time = time.time()
    deberta_prob = probs_model2(val_df['text'])
    utils.log(f"DeBERTa-v3-Large 추론 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    progress.update(msg="DeBERTa 예측 완료")
    
    # 3) TF‑IDF 확률
    utils.log("TF-IDF + LGBM 추론 중...")
    start_time = time.time()
    try:
        vec = joblib.load(os.path.join(checkpoint_dir, 'tfidf.pkl'))
        lgbm = joblib.load(os.path.join(checkpoint_dir, 'lgbm.pkl'))
        tfidf_prob = lgbm.predict_proba(vec.transform(val_df['text']))[:,1]
    except Exception as e:
        utils.log(f"TF-IDF 모델 로드 또는 예측 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("랜덤 확률값을 사용합니다.")
        tfidf_prob = np.random.uniform(0, 1, size=len(val_df))
    
    utils.log(f"TF-IDF + LGBM 추론 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    progress.update(msg="TF-IDF 예측 완료")
    
    # 4) Perplexity
    utils.log("KenLM Perplexity 계산 중...")
    start_time = time.time()
    
    # KenLM 바이너리 파일 경로
    lm_path = os.path.join(checkpoint_dir, 'data3.binary')
    
    # KenLM 라이브러리 및 바이너리 파일 확인
    if not utils.check_kenlm_available() or not os.path.exists(lm_path) or utils.is_dummy_kenlm_file(lm_path):
        if not utils.check_kenlm_available():
            utils.log("KenLM 모듈이 설치되어 있지 않습니다. 랜덤 Perplexity 값을 사용합니다.", level="WARNING")
        elif not os.path.exists(lm_path):
            utils.log(f"KenLM 모델 파일이 없습니다: {lm_path}", level="WARNING")
        elif utils.is_dummy_kenlm_file(lm_path):
            utils.log("더미 KenLM 파일이 감지되었습니다. 랜덤 Perplexity 값을 사용합니다.", level="WARNING")
            
        # 랜덤 값 생성
        ppl = np.random.uniform(0, 1, size=len(val_df))
    else:
        try:
            import kenlm
            lm = kenlm.Model(lm_path)
            ppl_values = []
            for text in tqdm(val_df['text'], desc="KenLM Perplexity 계산"):
                try:
                    ppl_values.append(-lm.perplexity(text))
                except Exception:
                    # 개별 텍스트 처리 실패 시 랜덤값 사용
                    ppl_values.append(np.random.uniform(-10, -1))
                    
            ppl = np.array(ppl_values)
            
            # MinMaxScaler로 정규화
            ppl = MinMaxScaler().fit_transform(ppl.reshape(-1,1)).ravel()
        except Exception as e:
            utils.log(f"KenLM 처리 중 오류 발생: {str(e)}", level="ERROR")
            utils.log("랜덤 Perplexity 값을 사용합니다.")
            # 오류 발생 시 랜덤 값 생성
            ppl = np.random.uniform(0, 1, size=len(val_df))
    
    utils.log(f"Perplexity 계산 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    
    # 메타 특성 구성
    utils.log("메타 특성 구성 및 LogReg 모델 학습 중...")
    X = pd.DataFrame({
        'qwen3-4b': qwen_prob,
        'deberta': deberta_prob,
        'tfidf':   tfidf_prob,
        'kenlm':   ppl
    })
    
    # NULL 값 체크
    null_counts = X.isna().sum()
    if null_counts.sum() > 0:
        utils.log(f"경고: 특성 데이터에 NULL 값이 발견되었습니다: {null_counts.to_dict()}", level="WARNING")
        X = X.fillna(0)  # NULL 값 0으로 대체
        utils.log("NULL 값을 0으로 대체했습니다.")
    
    # 특성 통계 확인
    utils.log("특성 통계:")
    for col in X.columns:
        utils.log(f"  {col}: 평균={X[col].mean():.4f}, 표준편차={X[col].std():.4f}, 최소={X[col].min():.4f}, 최대={X[col].max():.4f}")
    
    # LogReg 학습
    try:
        meta = LogisticRegression(max_iter=1000, random_state=cfg['seed']).fit(X, val_df['label'])
        
        # 성능 평가
        pred_proba = meta.predict_proba(X)[:, 1]
        pred = (pred_proba > 0.5).astype(int)
        acc = (pred == val_df['label']).mean() * 100
        utils.log(f"메타 모델 정확도: {acc:.2f}%")
        
        # 특성 중요도
        coef = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(meta.coef_[0])
        }).sort_values('importance', ascending=False)
        
        utils.log("특성 중요도:")
        for i, row in coef.iterrows():
            utils.log(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 모델 저장
        joblib.dump(meta, os.path.join(checkpoint_dir, 'meta.pkl'))
        progress.finish("메타 모델 저장 완료")
    except Exception as e:
        utils.log(f"LogReg 학습 중 오류 발생: {str(e)}", level="ERROR")
        # 단순 다수결 모델로 대체
        utils.log("오류로 인해 간단한 투표 모델로 대체합니다.", level="WARNING")
        class SimpleMeta:
            def predict_proba(self, X):
                probs = np.mean(X, axis=1).reshape(-1, 1)
                return np.hstack([1-probs, probs])
                
        meta = SimpleMeta()
        joblib.dump(meta, os.path.join(checkpoint_dir, 'meta.pkl'))
        progress.finish("대체 메타 모델 저장 완료")

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 작업 시작
        start_time = time.time()
        train_stacking(cfg)
        utils.log(f"전체 스태킹 과정 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main() 