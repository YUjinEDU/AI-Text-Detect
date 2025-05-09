# K-Fold split (disabled by default)
# -*- coding: utf-8 -*-

import os, pandas as pd
import utils
from sklearn.model_selection import StratifiedKFold

def split_folds(cfg):
    data_path = os.path.join(cfg['data_dir'], 'train.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    raw = pd.read_csv(data_path)
    if cfg.get('kfold', {}).get('enable', False):
        skf = StratifiedKFold(n_splits=cfg['kfold']['n_splits'], shuffle=True, random_state=cfg['seed'])
        for i, (_, val_idx) in enumerate(skf.split(raw, raw['label'])):
            raw.loc[val_idx, 'kfold'] = i
        utils.log(f"KFold 분할 완료: {cfg['kfold']['n_splits']} folds")
    else:
        utils.log('KFold 미사용, 전체 데이터 저장')
    out_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    utils.safe_makedirs(os.path.dirname(os.path.abspath(out_path)))
    raw.to_csv(out_path, index=False)
    utils.log(f'train_folds.csv 저장 완료: {out_path}')

def main():
    cfg = utils.load_config()
    split_folds(cfg)

if __name__ == '__main__':
    main()