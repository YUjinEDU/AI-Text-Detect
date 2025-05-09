import os, pandas as pd, joblib, kenlm, numpy as np
import utils
from train_lora import probs_model1
from train_deberta import probs_model2

def run_inference(cfg):
    utils.set_seed(cfg['seed'])
    test_path = os.path.join(cfg['data_dir'], 'test.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} 파일이 존재하지 않습니다.")
    test_df = pd.read_csv(test_path)
    checkpoint_dir = cfg['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    vec = joblib.load(os.path.join(checkpoint_dir, 'tfidf.pkl'))
    lgbm = joblib.load(os.path.join(checkpoint_dir, 'lgbm.pkl'))
    meta = joblib.load(os.path.join(checkpoint_dir, 'meta.pkl'))
    lm  = kenlm.Model(os.path.join(checkpoint_dir, 'data3.binary'))
    tfidf_p = lgbm.predict_proba(vec.transform(test_df['text']))[:,1]
    ppl     = np.array([-lm.perplexity(t) for t in test_df['text']])
    ppl     = (ppl - ppl.min()) / (ppl.max() - ppl.min())  # min‑max
    X = pd.DataFrame({
        'mistral': probs_model1(test_df['text']),
        'deberta': probs_model2(test_df['text']),
        'tfidf':   tfidf_p,
        'ppl':     ppl
    })
    prob = meta.predict_proba(X)[:,1]
    test_df['label'] = (prob > 0.5).astype(int)
    result_path = cfg['result_csv'] if 'result_csv' in cfg else 'submission.csv'
    os.makedirs(os.path.dirname(result_path), exist_ok=True) if os.path.dirname(result_path) else None
    test_df[['id','label']].to_csv(result_path, index=False)
    utils.log(f'submission.csv saved: {result_path}')

def main():
    cfg = utils.load_config()
    run_inference(cfg)

if __name__ == '__main__':
    main() 