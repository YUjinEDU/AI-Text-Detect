import os, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import utils

def train_tfidf_lgbm(cfg):
    utils.set_seed(cfg['seed'])
    data_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    data = pd.read_csv(data_path)
    data['text'] = data['text'].map(utils.clean_text)
    train_x, val_x, train_y, val_y = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=cfg['seed'], stratify=data['label'])
    vec = TfidfVectorizer(ngram_range=tuple(cfg['tfidf']['ngram']), max_features=cfg['tfidf']['max_features'], sublinear_tf=True)
    X_train = vec.fit_transform(train_x)
    clf = LGBMClassifier(n_estimators=cfg['lgbm']['n_estimators'], learning_rate=cfg['lgbm']['learning_rate'])
    clf.fit(X_train, train_y)
    joblib.dump(vec, os.path.join(checkpoint_dir, 'tfidf.pkl'))
    joblib.dump(clf, os.path.join(checkpoint_dir, 'lgbm.pkl'))
    utils.log('TF-IDF 및 LGBM 모델 저장 완료')

def main():
    cfg = utils.load_config()
    train_tfidf_lgbm(cfg)

if __name__ == '__main__':
    main()