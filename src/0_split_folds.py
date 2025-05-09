# K-Fold split (disabled by default)
# -*- coding: utf-8 -*-

import os, pandas as pd
import utils
from sklearn.model_selection import StratifiedKFold
import time

def split_folds(cfg):
    utils.log("K-Fold 분할 시작", is_important=True)
    
    # 데이터 경로 확인
    data_path = os.path.join(cfg['data_dir'], 'train.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    
    # 체크포인트 디렉토리 생성
    utils.safe_makedirs(cfg['checkpoint_dir'])
    
    # 데이터 로드
    utils.log("데이터 로드 중...")
    raw = pd.read_csv(data_path)
    utils.log(f"총 {len(raw)}개 샘플 로드 완료")
    
    # 데이터 검증 및 정제
    raw = utils.check_and_clean_data(raw)
    
    # K-Fold 분할
    if cfg.get('kfold', {}).get('enable', False):
        utils.log(f"{cfg['kfold']['n_splits']}-Fold 분할 시작...")
        start_time = time.time()
        
        skf = StratifiedKFold(n_splits=cfg['kfold']['n_splits'], shuffle=True, random_state=cfg['seed'])
        raw['kfold'] = -1  # 초기화
        
        for i, (_, val_idx) in enumerate(skf.split(raw, raw['label'])):
            raw.loc[val_idx, 'kfold'] = i
            utils.log(f"  Fold {i+1}: {len(val_idx)}개 검증 샘플")
        
        # 각 폴드의 레이블 분포 체크
        for fold in range(cfg['kfold']['n_splits']):
            fold_df = raw[raw['kfold'] == fold]
            label_dist = fold_df['label'].value_counts(normalize=True) * 100
            utils.log(f"  Fold {fold+1} 레이블 분포: 0={label_dist.get(0, 0):.1f}%, 1={label_dist.get(1, 0):.1f}%")
        
        utils.log(f"K-Fold 분할 완료: {cfg['kfold']['n_splits']} folds (소요시간: {utils.format_time(time.time() - start_time)})")
    else:
        utils.log('KFold 비활성화 상태 - 전체 데이터 저장')
    
    # 저장
    out_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    utils.safe_makedirs(os.path.dirname(os.path.abspath(out_path)))
    raw.to_csv(out_path, index=False)
    utils.log(f'train_folds.csv 저장 완료: {out_path}', is_important=True)

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 시드 설정
        utils.set_seed(cfg['seed'])
        
        # 작업 실행
        split_folds(cfg)
    except Exception as e:
        utils.log(f"에러 발생: {str(e)}", is_important=True)
        raise
    finally:
        # 로깅 자원 정리
        utils.close_logging()

if __name__ == '__main__':
    main()