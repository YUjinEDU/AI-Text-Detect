import os
import utils

def build_kenlm(cfg):
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    lm_path = os.path.join(checkpoint_dir, 'data3.binary')
    data_path = os.path.join(cfg['data_dir'], 'train.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    if not os.path.exists(lm_path):
        tmp_txt = os.path.join(checkpoint_dir, 'train.txt')
        os.system(f'cut -d, -f2- "{data_path}" > "{tmp_txt}"')
        os.system(f'lmplz -o 3 < "{tmp_txt}" > "{checkpoint_dir}/3gram.arpa"')
        os.system(f'build_binary "{checkpoint_dir}/3gram.arpa" "{lm_path}"')
        utils.log('KenLM 3-gram 모델 생성 및 저장 완료')
    else:
        utils.log('이미 KenLM 모델이 존재합니다.')

def main():
    cfg = utils.load_config()
    build_kenlm(cfg)

if __name__ == '__main__':
    main()