from pathlib import Path


# 현재 파일(config.py)이 위치한 디렉터리 경로
basedir = Path(__file__).parent


class BaseConfig:
    # 파일 업로드 시 저장될 폴더 경로
    # config.py 기준으로 images 폴더를 사용
    # 예: project/config.py → project/images/
    UPLOAD_FOLDER = str(basedir / "images")