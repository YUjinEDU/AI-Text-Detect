import os, joblib, pandas as pd, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import utils
from train_lora import probs_model1
from train_deberta import probs_model2
import time
from tqdm.auto import tqdm

def train_stacking(cfg):
    utils.set_seed(cfg['seed'])
    utils.log("고급 앙상블 메타 모델 학습 시작", is_important=True)
    
    # 진행 상황 추적기 초기화
    progress = utils.ProgressTracker(6, "앙상블 학습")
    
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
    progress.update(msg="Perplexity 계산 완료")
    
    # 메타 특성 구성
    utils.log("메타 특성 구성 중...")
    X = pd.DataFrame({
        'qwen3-4b': qwen_prob,
        'deberta': deberta_prob,
        'tfidf': tfidf_prob,
        'kenlm': ppl,
        # 교차항(interaction) 추가 - 모델 간 시너지 효과 포착
        'qwen_x_deberta': qwen_prob * deberta_prob,
        'qwen_x_tfidf': qwen_prob * tfidf_prob,
        'deberta_x_tfidf': deberta_prob * tfidf_prob,
        # 차이(discrepancy) 특성 추가 - 모델 간 불일치 포착
        'qwen_deberta_diff': np.abs(qwen_prob - deberta_prob),
        'max_min_diff': np.maximum.reduce([qwen_prob, deberta_prob, tfidf_prob]) - 
                        np.minimum.reduce([qwen_prob, deberta_prob, tfidf_prob])
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
        utils.log(f"  {col}: 평균={X[col].mean():.4f}, 표준편차={X[col].std():.4f}")
    
    # 고급 앙상블 학습
    utils.log("고급 앙상블 학습 중...", is_important=True)
    labels = val_df['label'].values
    
    try:
        # 1. 교차 검증 스태킹 (더 안정적인 메타모델)
        utils.log("교차 검증 스태킹 구현 중...")
        
        # 개별 메타모델
        meta_models = {
            'LogReg': LogisticRegression(C=0.3, max_iter=1000, random_state=cfg['seed']),
            'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=cfg['seed']),
            'SVM': CalibratedClassifierCV(
                SVC(probability=True, random_state=cfg['seed']),
                cv=3, method='sigmoid'
            )
        }
        
        # 교차 검증 스태킹 모델
        cv_stacking_preds = np.zeros((len(X), len(meta_models)))
        
        # K-fold 교차 검증으로 각 메타모델 학습 및 평가
        utils.log("5-fold 교차 검증 스태킹 시작...")
        kf = KFold(n_splits=5, shuffle=True, random_state=cfg['seed'])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            utils.log(f"Fold {fold+1}/5 학습 중...")
            
            # 각 메타모델 학습
            for i, (model_name, model) in enumerate(meta_models.items()):
                model.fit(X_train, y_train)
                # 검증 세트 예측
                cv_stacking_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
        
        # 메타모델 앙상블 준비 - 모든 모델을 다시 전체 데이터로 학습
        utils.log("최종 메타모델 앙상블 구성 중...")
        for model_name, model in meta_models.items():
            model.fit(X, labels)
        
        # 2. 다양한 스태킹 기법들을 통합한 최종 앙상블
        # - 검증 데이터의 교차검증 예측을 다시 특성으로 사용하는 2단계 스태킹
        X2 = pd.DataFrame({
            # 원래 특성
            'qwen3-4b': qwen_prob,
            'deberta': deberta_prob,
            'tfidf': tfidf_prob,
            'kenlm': ppl,
            # CV 스태킹 예측
            'cv_logreg': cv_stacking_preds[:, 0],
            'cv_gbm': cv_stacking_preds[:, 1],
            'cv_svm': cv_stacking_preds[:, 2]
        })
        
        # 최종 메타모델: 보팅 앙상블 (가중치 투표)
        final_ensemble = VotingClassifier(
            estimators=[
                ('logreg', meta_models['LogReg']),
                ('gbm', meta_models['GBM']),
                ('svm', meta_models['SVM']),
            ],
            voting='soft', 
            weights=[2, 1.5, 1]  # 실험적으로 LogReg에 더 높은 가중치
        )
        
        # 최종 모델 학습
        final_ensemble.fit(X2, labels)
        
        # 모델 평가
        pred_proba = final_ensemble.predict_proba(X2)[:, 1]
        pred = (pred_proba > 0.5).astype(int)
        acc = (pred == labels).mean() * 100
        utils.log(f"최종 앙상블 정확도: {acc:.2f}%", is_important=True)
        
        # 사후 확률 보정 - 특히 확률값이 0.5 근처에 몰려있을 때 유용
        utils.log("확률값 보정 중...")
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pred_proba, labels)
        calibrated_proba = ir.transform(pred_proba)
        
        # 앙상블 객체 구성 (높은 내결함성을 위한 래퍼)
        class AdvancedEnsemble:
            def __init__(self, main_ensemble, iso_reg, meta_models, feature_cols):
                self.main_ensemble = main_ensemble
                self.iso_reg = iso_reg
                self.meta_models = meta_models
                self.feature_cols = feature_cols
                
            def predict_proba(self, X_orig):
                try:
                    # 필요한 특성 확인
                    missing_cols = set(self.feature_cols) - set(X_orig.columns)
                    if missing_cols:
                        utils.log(f"누락된 특성 감지: {missing_cols}. 0으로 채웁니다.", level="WARNING")
                        for col in missing_cols:
                            X_orig[col] = 0
                    
                    # CV 스태킹 결과 계산 (각 메타모델의 예측값)
                    cv_preds = np.zeros((len(X_orig), len(self.meta_models)))
                    for i, (_, model) in enumerate(self.meta_models.items()):
                        cv_preds[:, i] = model.predict_proba(X_orig[self.feature_cols[:4]])[:, 1]
                    
                    # 최종 특성 구성
                    X2 = pd.DataFrame({
                        'qwen3-4b': X_orig['qwen3-4b'],
                        'deberta': X_orig['deberta'],
                        'tfidf': X_orig['tfidf'],
                        'kenlm': X_orig['kenlm'], 
                        'cv_logreg': cv_preds[:, 0],
                        'cv_gbm': cv_preds[:, 1], 
                        'cv_svm': cv_preds[:, 2]
                    })
                    
                    # 최종 예측 및 보정
                    proba = self.main_ensemble.predict_proba(X2)
                    calibrated = np.column_stack([
                        1 - self.iso_reg.transform(proba[:, 1]),
                        self.iso_reg.transform(proba[:, 1])
                    ])
                    return calibrated
                    
                except Exception as e:
                    utils.log(f"앙상블 예측 중 오류 발생: {e}", level="ERROR")
                    # 대체 예측: 간단한 평균
                    fallback = np.mean(X_orig[['qwen3-4b', 'deberta', 'tfidf']], axis=1)
                    return np.column_stack([1 - fallback, fallback])
        
        # 최종 앙상블 객체 구성
        meta = AdvancedEnsemble(
            main_ensemble=final_ensemble,
            iso_reg=ir,
            meta_models=meta_models,
            feature_cols=X.columns.tolist()
        )
        
        # 모델 저장
        utils.log("앙상블 모델 저장 중...")
        joblib.dump(meta, os.path.join(checkpoint_dir, 'meta.pkl'))
        progress.finish("고급 앙상블 모델 저장 완료")
        
    except Exception as e:
        utils.log(f"앙상블 학습 중 오류 발생: {str(e)}", level="ERROR")
        
        # 확장된 대체 모델 - 단순 평균보다 개선된 버전
        utils.log("오류로 인해 강건한 대체 앙상블 모델로 전환합니다.", level="WARNING")
        
        class RobustFallbackEnsemble:
            def __init__(self, weights=None):
                # 모델별 가중치 - 기본값 설정
                self.weights = weights if weights is not None else {
                    'qwen3-4b': 0.35,  # LLM에 더 높은 가중치
                    'deberta': 0.35,   # LLM에 더 높은 가중치
                    'tfidf': 0.2,
                    'kenlm': 0.1
                }
                
            def predict_proba(self, X):
                try:
                    # 각 모델에 가중치 적용 후 합산
                    weighted_sum = np.zeros(len(X))
                    total_weight = 0
                    
                    for col, weight in self.weights.items():
                        if col in X.columns:
                            weighted_sum += X[col].values * weight
                            total_weight += weight
                    
                    # 가중치 합이 0이 아닌 경우에만 정규화
                    if total_weight > 0:
                        weighted_sum /= total_weight
                    else:
                        # 모든 모델이 실패한 경우 0.5 (불확실)
                        weighted_sum = np.full(len(X), 0.5)
                    
                    # 최종 확률값 변환 (0-1 범위 내로 클리핑)
                    final_probs = np.clip(weighted_sum, 0, 1)
                    
                    # 양쪽 클래스에 대한 확률 반환
                    return np.column_stack([1-final_probs, final_probs])
                    
                except Exception:
                    # 극단적 오류 시 균등 확률(0.5) 반환
                    utils.log("강건한 앙상블에서도 오류 발생, 균등 확률 사용", level="ERROR")
                    uniform_probs = np.full(len(X), 0.5)
                    return np.column_stack([uniform_probs, uniform_probs])
        
        # 개선된 대체 모델 저장
        meta = RobustFallbackEnsemble()
        joblib.dump(meta, os.path.join(checkpoint_dir, 'meta.pkl'))
        progress.finish("대체 앙상블 모델 저장 완료")

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 작업 시작
        start_time = time.time()
        train_stacking(cfg)
        utils.log(f"전체 앙상블 구축 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main()