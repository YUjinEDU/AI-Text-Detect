import os, joblib, pandas as pd, numpy as np, torch, kenlm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import utils
from train_lora import probs_model1
from train_deberta import probs_model2

def train_stacking(cfg):
    utils.set_seed(cfg['seed'])
    val_path = os.path.join(cfg['data_dir'], 'train_folds.csv')
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"{val_path} 파일이 존재하지 않습니다.")
    val_df = pd.read_csv(val_path).sample(frac=0.3, random_state=cfg['seed'])
    # 1) Mistral 확률
    mistral_prob = probs_model1(val_df['text'])
    # 2) DeBERTa 확률
    deberta_prob = probs_model2(val_df['text'])
    # 3) TF‑IDF 확률
    vec = joblib.load(os.path.join(checkpoint_dir, 'tfidf.pkl'))
    lgbm = joblib.load(os.path.join(checkpoint_dir, 'lgbm.pkl'))
    tfidf_prob = lgbm.predict_proba(vec.transform(val_df['text']))[:,1]
    # 4) Perplexity
    lm = kenlm.Model(os.path.join(checkpoint_dir, 'data3.binary'))
    ppl = np.array([-lm.perplexity(t) for t in val_df['text']])
    ppl = MinMaxScaler().fit_transform(ppl.reshape(-1,1)).ravel()
    meta_X = pd.DataFrame({
        'mistral': mistral_prob,
        'deberta': deberta_prob,
        'tfidf':   tfidf_prob,
        'ppl':     ppl
    })
    meta = LogisticRegression(max_iter=1000).fit(meta_X, val_df['label'])
    joblib.dump(meta, os.path.join(checkpoint_dir, 'meta.pkl'))
    utils.log('스태킹 메타 모델 저장 완료')

def main():
    cfg = utils.load_config()
    train_stacking(cfg)

if __name__ == '__main__':
    main() 