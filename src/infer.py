import os, pandas as pd, joblib, numpy as np
from sklearn.preprocessing import MinMaxScaler
import utils
from train_lora import probs_model1
from train_deberta import probs_model2
import time
from tqdm.auto import tqdm
import sys

def run_inference(cfg):
    utils.set_seed(cfg['seed'])
    utils.log("최종 추론 시작", is_important=True)
    
    # 진행 상황 추적기 초기화
    progress = utils.ProgressTracker(6, "추론 파이프라인")
    
    # 1. 테스트 데이터 로드
    utils.log("테스트 데이터 로드 중...")
    test_path = os.path.join(cfg['data_dir'], 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} 파일이 존재하지 않습니다.")
    
    test_df = pd.read_csv(test_path)
    progress.update(msg="데이터 로드 완료")
    
    # 2. 데이터 검증 및 정제
    utils.log("데이터 검증 및 정제 중...")
    test_df = utils.check_and_clean_data(test_df, label_col=None)
    progress.update(msg="데이터 검증 완료")
    
    # 3. 체크포인트 디렉토리 확인
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    
    # 모델 로드 경로 체크 - 필요한 파일이 없으면 빈 파일 생성
    required_files = ['tfidf.pkl', 'lgbm.pkl', 'meta.pkl', 'data3.binary']
    missing_files = []
    
    for file in required_files:
        path = os.path.join(checkpoint_dir, file)
        if not os.path.exists(path):
            missing_files.append(file)
            if file == 'data3.binary':
                utils.create_dummy_kenlm_file(path)
                utils.log(f"경고: {file} 파일이 없어 더미 파일을 생성했습니다.", level="WARNING")
    
    if missing_files:
        utils.log(f"경고: 다음 모델 파일이 없습니다: {', '.join(missing_files)}", level="WARNING")
        utils.log("일부 모델 없이도 계속 진행합니다. 결과가 부정확할 수 있습니다.", level="WARNING")
    
    # 4. 모델 로드 및 예측
    # TF-IDF + LGBM 예측
    utils.log("TF-IDF 모델 로드 중...")
    start_time = time.time()
    try:
        vec = joblib.load(os.path.join(checkpoint_dir, 'tfidf.pkl'))
        lgbm = joblib.load(os.path.join(checkpoint_dir, 'lgbm.pkl'))
        utils.log(f"TF-IDF 및 LGBM 모델 로드 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
        
        utils.log("TF-IDF 예측 중...")
        start_time = time.time()
        tfidf_p = lgbm.predict_proba(vec.transform(test_df['text']))[:,1]
        utils.log(f"TF-IDF 예측 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    except Exception as e:
        utils.log(f"TF-IDF 모델 로드 또는 예측 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("랜덤 확률값을 사용합니다.")
        tfidf_p = np.random.uniform(0, 1, size=len(test_df))
    
    progress.update(msg="TF-IDF 예측 완료")
    
    # Perplexity 계산
    utils.log("Perplexity 계산 중...")
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
        ppl = np.random.uniform(0, 1, size=len(test_df))
    else:
        try:
            import kenlm
            lm = kenlm.Model(lm_path)
            ppl_values = []
            for text in tqdm(test_df['text'], desc="KenLM Perplexity 계산"):
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
            ppl = np.random.uniform(0, 1, size=len(test_df))
    
    utils.log(f"Perplexity 계산 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    progress.update(msg="Perplexity 계산 완료")
    
    # Qwen3-4B 모델 예측
    utils.log("Qwen3-4B 모델 예측 중...")
    start_time = time.time()
    try:
        qwen_p = probs_model1(test_df['text'].tolist())
        utils.log(f"Qwen3-4B 예측 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    except Exception as e:
        utils.log(f"Qwen3-4B 예측 중 오류 발생: {str(e)}", level="ERROR")
        qwen_p = np.random.uniform(0, 1, size=len(test_df))
    progress.update(msg="Qwen3-4B 예측 완료")
    
    # DeBERTa 모델 예측
    utils.log("DeBERTa 모델 예측 중...")
    start_time = time.time()
    try:
        deberta_p = probs_model2(test_df['text'])
        utils.log(f"DeBERTa 모델 예측 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    except Exception as e:
        utils.log(f"DeBERTa 모델 예측 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("랜덤 확률값을 사용합니다.")
        deberta_p = np.random.uniform(0, 1, size=len(test_df))
    
    progress.update(msg="DeBERTa 예측 완료")
    
    # 메타 모델로 앙상블
    utils.log("메타 모델 로드 중...")
    try:
        meta = joblib.load(os.path.join(checkpoint_dir, 'meta.pkl'))
    except Exception as e:
        utils.log(f"메타 모델 로드 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("간단한 투표 모델로 대체합니다.", level="WARNING")
        # 간단한 대체 메타 모델
        class SimpleMeta:
            def predict_proba(self, X):
                probs = np.mean(X, axis=1).reshape(-1, 1)
                return np.hstack([1-probs, probs])
        meta = SimpleMeta()
    
    utils.log("메타 모델로 앙상블 중...")
    X = pd.DataFrame({
        'qwen3-4b': qwen_p,
        'deberta': deberta_p,
        'tfidf':   tfidf_p,
        'ppl':     ppl
    })
    
    # X 데이터프레임 검증
    null_counts = X.isna().sum()
    if null_counts.sum() > 0:
        utils.log(f"경고: 특성 데이터에 NULL 값이 발견되었습니다: {null_counts.to_dict()}", is_important=True, level="WARNING")
        # NULL 값 처리 (0으로 대체)
        X = X.fillna(0)
        utils.log("NULL 값을 0으로 대체했습니다.")
    
    # 오류 발생 가능성이 있는 코드를 try-except로 보호
    try:
        prob = meta.predict_proba(X)[:,1]
        test_df['label'] = (prob > 0.5).astype(int)
    except Exception as e:
        utils.log(f"메타 모델 예측 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("평균값으로 대체합니다.", level="WARNING")
        # 간단한 대체 방법: 각 특성의 평균 확률로 결정
        prob = X.mean(axis=1).values
        test_df['label'] = (prob > 0.5).astype(int)
    
    # 클래스별 샘플 분포 확인
    class_counts = test_df['label'].value_counts()
    utils.log(f"예측 클래스 분포: 0(진짜)={class_counts.get(0, 0)}개, 1(가짜)={class_counts.get(1, 0)}개")
    
    # 결과 저장
    try:
        result_path = cfg['result_csv'] if 'result_csv' in cfg else 'submission.csv'
        utils.safe_makedirs(os.path.dirname(os.path.abspath(result_path)))
        test_df[['id','label']].to_csv(result_path, index=False)
        
        utils.log(f"결과 통계: 총 {len(test_df)}개, 가짜 텍스트 비율: {class_counts.get(1, 0)/len(test_df)*100:.2f}%")
        progress.finish(f"submission.csv 저장 완료: {result_path}")
    except Exception as e:
        utils.log(f"결과 저장 중 오류 발생: {str(e)}", level="ERROR")
        # 마지막 시도: 현재 디렉토리에 저장
        try:
            test_df[['id','label']].to_csv('submission_emergency.csv', index=False)
            utils.log("문제 발생으로 현재 디렉토리에 저장했습니다: submission_emergency.csv", level="WARNING")
        except Exception:
            utils.log("결과 저장에 실패했습니다. 콘솔 출력을 확인하세요.", level="ERROR")

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 추론 실행
        run_inference(cfg)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True, level="ERROR")
        import traceback
        utils.log(f"상세 오류: {traceback.format_exc()}", level="ERROR")
        sys.exit(1)
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main() 