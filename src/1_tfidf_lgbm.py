import os, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import utils
import time
from tqdm.auto import tqdm
import lightgbm

def train_tfidf_lgbm(cfg):
    utils.set_seed(cfg['seed'])
    utils.log("TF-IDF + LightGBM 학습 시작", is_important=True)
    
    # 데이터 로드
    data_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    
    utils.log("데이터 로드 중...")
    data = pd.read_csv(data_path)
    
    # 데이터 검증 및 정제
    data = utils.check_and_clean_data(data)
    
    # 데이터 분할
    utils.log("데이터 분할 중...")
    train_x, val_x, train_y, val_y = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=cfg['seed'], stratify=data['label'])
    
    utils.log(f"학습 샘플: {len(train_x)}개, 검증 샘플: {len(val_x)}개")
    
    # TF-IDF 벡터화
    utils.log(f"TF-IDF 피처 추출 중 (ngrams={cfg['tfidf']['ngram']}, max_features={cfg['tfidf']['max_features']})...")
    start_time = time.time()
    
    vec = TfidfVectorizer(
        ngram_range=tuple(cfg['tfidf']['ngram']), 
        max_features=cfg['tfidf']['max_features'], 
        sublinear_tf=True
    )
    
    X_train = vec.fit_transform(tqdm(train_x, desc="TF-IDF 피처 추출"))
    utils.log(f"TF-IDF 피처 추출 완료: {X_train.shape[1]} 피처 (소요시간: {utils.format_time(time.time() - start_time)})")
    
    # LightGBM 학습
    utils.log(f"LightGBM 모델 학습 중 (n_estimators={cfg['lgbm']['n_estimators']}, lr={cfg['lgbm']['learning_rate']})...")
    start_time = time.time()
    
    clf = LGBMClassifier(
        n_estimators=cfg['lgbm']['n_estimators'], 
        learning_rate=cfg['lgbm']['learning_rate'],
        verbose=-1,  # 로깅은 수동으로 처리
        random_state=cfg['seed']
    )
    
    # 최신 LightGBM API에 맞게 수정
    X_val = vec.transform(val_x)
    clf.fit(
        X_train, train_y, 
        eval_set=[(X_val, val_y)],
        eval_metric='auc',
        callbacks=[lightgbm.early_stopping(stopping_rounds=50)]
    )
    
    utils.log(f"LightGBM 학습 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    
    # 평가
    val_pred = clf.predict_proba(vec.transform(val_x))[:, 1]
    threshold = 0.5
    val_acc = (val_pred > threshold).astype(int) == val_y
    utils.log(f"검증 정확도: {val_acc.mean()*100:.2f}%")
    
    # 모델 저장
    utils.log("모델 저장 중...")
    joblib.dump(vec, os.path.join(checkpoint_dir, 'tfidf.pkl'))
    joblib.dump(clf, os.path.join(checkpoint_dir, 'lgbm.pkl'))
    utils.log('TF-IDF 및 LGBM 모델 저장 완료', is_important=True)

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 작업 시작
        start_time = time.time()
        train_tfidf_lgbm(cfg)
        utils.log(f"전체 TF-IDF + LGBM 과정 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True)
        raise
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main()