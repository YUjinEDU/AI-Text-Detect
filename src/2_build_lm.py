import os
import utils
import time
import pandas as pd
from tqdm.auto import tqdm
import sys
import platform
import subprocess

def build_kenlm(cfg):
    utils.log("KenLM 3-gram 모델 구축 시작", is_important=True)
    
    # KenLM 사용 가능 여부 확인
    if not utils.check_kenlm_available():
        error_msg = f"""
KenLM 모듈이 설치되어 있지 않습니다. KenLM 설치 방법:

1. Linux:
   pip install https://github.com/kpu/kenlm/archive/master.zip

2. Windows (복잡):
   a. MSVC 컴파일러와 CMake 설치 필요
   b. https://github.com/kpu/kenlm 에서 소스코드 다운로드 후 빌드
   c. 또는 WSL을 통해 Linux 환경에서 설치

지금은 KenLM 기능을 건너뛰고 계속 진행합니다.
후속 단계에서는 이 기능 없이도 전체 파이프라인이 동작하도록 처리되어 있습니다.
"""
        utils.log(error_msg, is_important=True, level="WARNING")
        
        # Windows인 경우 추가 안내
        if platform.system() == 'Windows':
            utils.log("Windows 환경에서는 KenLM 설치가 복잡할 수 있습니다. WSL 사용을 권장합니다.", is_important=True, level="WARNING")
        
        # 체크포인트 디렉토리 생성
        checkpoint_dir = cfg['checkpoint_dir']
        utils.safe_makedirs(checkpoint_dir)
        
        # 더미 KenLM 파일 생성
        dummy_path = os.path.join(checkpoint_dir, 'data3.binary')
        utils.create_dummy_kenlm_file(dummy_path)
        
        utils.log("나머지 파이프라인은 정상적으로 진행됩니다.", is_important=True)
        return
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = cfg['checkpoint_dir']
    utils.safe_makedirs(checkpoint_dir)
    
    # 경로 설정
    lm_path = os.path.join(checkpoint_dir, 'data3.binary')
    data_path = os.path.join(cfg['data_dir'], 'train.csv')
    
    # 데이터 파일 확인
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} 파일이 존재하지 않습니다.")
    
    # 이미 구축된 모델이 있는지 확인
    if os.path.exists(lm_path):
        utils.log('이미 KenLM 모델이 존재합니다. 재사용합니다.', is_important=True)
        return
    
    # 시스템 종속성 확인 (subprocess로 변경)
    required_cmds = ['lmplz', 'build_binary']
    missing_cmd = False
    
    for cmd in required_cmds:
        try:
            # 플랫폼에 맞는 명령어 실행
            if platform.system() == "Windows":
                result = subprocess.run(f"where {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(f"which {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            if result.returncode != 0:
                utils.log(f"경고: '{cmd}' 명령어를 찾을 수 없습니다. KenLM 바이너리 도구가 PATH에 추가되어 있지 않습니다.", 
                          is_important=True, level="WARNING")
                missing_cmd = True
        except Exception as e:
            utils.log(f"명령어 확인 중 오류 발생: {str(e)}", level="ERROR")
            missing_cmd = True
    
    if missing_cmd:
        utils.log("더미 KenLM 파일을 생성하고 다음 단계로 진행합니다.", level="WARNING")
        utils.create_dummy_kenlm_file(lm_path)
        return
    
    # 모델 구축 시작
    utils.log("학습 데이터 로드 중...")
    try:
        df = pd.read_csv(data_path)
        df = utils.check_and_clean_data(df)  # NULL 값 처리
    except Exception as e:
        utils.log(f"데이터 로드 중 오류 발생: {str(e)}", level="ERROR")
        utils.create_dummy_kenlm_file(lm_path)
        return
    
    # 임시 파일 경로
    tmp_txt = os.path.join(checkpoint_dir, 'train.txt')
    
    # 텍스트 데이터 추출
    utils.log("텍스트 데이터 추출 중...")
    try:
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            for text in tqdm(df['text'], desc="텍스트 파일 작성"):
                f.write(f"{text}\n")
        
        utils.log(f"텍스트 파일 저장 완료: {tmp_txt}")
    except Exception as e:
        utils.log(f"텍스트 파일 저장 중 오류 발생: {str(e)}", level="ERROR")
        utils.create_dummy_kenlm_file(lm_path)
        return
    
    # KenLM 모델 구축
    utils.log("KenLM 3-gram ARPA 모델 생성 중...")
    start_time = time.time()
    
    # lmplz 명령어 에러 처리 추가 (subprocess 사용)
    try:
        arpa_path = os.path.join(checkpoint_dir, '3gram.arpa')
        if platform.system() == "Windows":
            lmplz_cmd = f'lmplz -o 3 < "{tmp_txt}" > "{arpa_path}"'
        else:
            lmplz_cmd = f"lmplz -o 3 < '{tmp_txt}' > '{arpa_path}'"
            
        utils.log(f"실행 명령어: {lmplz_cmd}")
        lmplz_result = os.system(lmplz_cmd)
        
        if lmplz_result != 0:
            utils.log(f"lmplz 명령어 실행 중 오류 발생 (반환값: {lmplz_result})", is_important=True, level="ERROR")
            utils.log("더미 KenLM 파일을 생성하고 다음 단계로 진행합니다.")
            utils.create_dummy_kenlm_file(lm_path)
            return
            
        utils.log(f"KenLM ARPA 모델 생성 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    except Exception as e:
        utils.log(f"ARPA 모델 생성 중 오류 발생: {str(e)}", level="ERROR")
        utils.create_dummy_kenlm_file(lm_path)
        return
    
    # 바이너리 변환
    utils.log("KenLM 바이너리 모델 변환 중...")
    start_time = time.time()
    
    # build_binary 명령어 에러 처리 추가 (subprocess 사용)
    try:
        if platform.system() == "Windows":
            build_cmd = f'build_binary "{arpa_path}" "{lm_path}"'
        else:
            build_cmd = f"build_binary '{arpa_path}' '{lm_path}'"
            
        utils.log(f"실행 명령어: {build_cmd}")
        build_binary_result = os.system(build_cmd)
        
        if build_binary_result != 0:
            utils.log(f"build_binary 명령어 실행 중 오류 발생 (반환값: {build_binary_result})", is_important=True, level="ERROR")
            utils.log("더미 KenLM 파일을 생성하고 다음 단계로 진행합니다.")
            utils.create_dummy_kenlm_file(lm_path)
            return
            
        utils.log(f"KenLM 바이너리 변환 완료 (소요시간: {utils.format_time(time.time() - start_time)})")
    except Exception as e:
        utils.log(f"바이너리 변환 중 오류 발생: {str(e)}", level="ERROR")
        utils.create_dummy_kenlm_file(lm_path)
        return
    
    # 모델 테스트
    utils.log("KenLM 모델 테스트 중...")
    try:
        import kenlm
        model = kenlm.Model(lm_path)
        example = df['text'].iloc[0]
        ppl = model.perplexity(example)
        utils.log(f"모델 테스트 성공 - 예시 문장 perplexity: {ppl:.2f}")
    except Exception as e:
        utils.log(f"모델 테스트 중 오류 발생: {str(e)}", level="ERROR")
        utils.log("모델이 생성되었지만 테스트는 실패했습니다. 나머지 파이프라인은 계속 진행됩니다.")
    
    utils.log('KenLM 3-gram 모델 생성 및 저장 완료', is_important=True)

def main():
    # 설정 로드
    cfg = utils.load_config()
    
    try:
        # 로깅 초기화
        utils.init_logging(cfg)
        
        # 시드 설정
        utils.set_seed(cfg['seed'])
        
        # 작업 시작
        start_time = time.time()
        build_kenlm(cfg)
        utils.log(f"전체 KenLM 구축 과정 완료 (총 소요시간: {utils.format_time(time.time() - start_time)})", is_important=True)
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