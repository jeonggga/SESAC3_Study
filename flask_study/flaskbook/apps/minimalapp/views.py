from flask import (
    Blueprint,
    render_template,
    current_app,
    send_from_directory,
    redirect,
    url_for,
    request,
)
import uuid
from pathlib import Path
from forms import UploadImageForm
from datetime import datetime


# Blueprint 생성
# "dt"는 Blueprint 이름이며 url_for에서 사용됨 (예: url_for('dt.images'))
dt = Blueprint("dt", __name__)


@dt.route("/images/<path:filename>")
def image_file(filename):
    """
    업로드된 이미지를 브라우저에서 직접 보여주기 위한 라우트
    예: /images/example.jpg
    """
    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        filename
    )


@dt.route("/upload", methods=["GET", "POST"])
def upload():
    """
    이미지 업로드 페이지
    - GET  : 업로드 폼 화면 렌더링
    - POST : 이미지 파일 업로드 처리
    """
    form = UploadImageForm()

    # 폼이 제출되었고 모든 validator를 통과했을 경우
    if form.validate_on_submit():
        # 업로드된 파일 객체
        file = form.image.data

        # 원본 파일의 확장자 추출 (.png, .jpg 등)
        ext = Path(file.filename).suffix

        # UUID를 사용해 파일명 중복 방지
        image_uuid_file_name = str(uuid.uuid4()) + ext

        # 실제 저장될 파일 경로
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"],
            image_uuid_file_name
        )

        # 파일 저장
        file.save(image_path)

        # 업로드 완료 후 이미지 목록 페이지로 이동
        return redirect(url_for("dt.images"))

    # GET 요청이거나 validation 실패 시 업로드 페이지 렌더링
    return render_template("upload.html", form=form)


# 업로드된 이미지 목록 페이지
@dt.route("/images")
def images():
    """
    업로드된 이미지 파일 목록을 조회하는 페이지
    파일명, 업로드 시간, 수정 시간, 파일 크기 표시
    """
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    images = []

    # 업로드 폴더가 존재할 경우만 처리
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.is_file():
                images.append({
                    # 파일명
                    "filename": f.name,

                    # 생성 시간 (업로드 시간으로 사용)
                    "uploaded_at": datetime.fromtimestamp(
                        f.stat().st_ctime
                    ).strftime("%Y-%m-%d %H:%M:%S"),

                    # 마지막 수정 시간
                    "modified_at": datetime.fromtimestamp(
                        f.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S"),

                    # 파일 크기 (KB 단위)
                    "size_kb": round(f.stat().st_size / 1024, 2),
                })

    return render_template("images.html", images=images)


# 이미지 삭제 처리
@dt.route("/delete/<filename>", methods=["POST"])
def delete_image(filename):
    """
    선택한 이미지 파일 삭제
    POST 요청으로만 동작하도록 제한
    """
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    file_path = upload_dir / filename

    # 파일이 존재할 경우 삭제
    if file_path.exists():
        file_path.unlink()

    # 삭제 후 이미지 목록 페이지로 이동
    return redirect(url_for("dt.images"))