from flask_wtf.file import FileAllowed, FileField, FileRequired
from flask_wtf.form import FlaskForm
from wtforms.fields.simple import SubmitField


class UploadImageForm(FlaskForm):
    # 이미지 파일 업로드 필드
    image = FileField(
        validators=[
            # 파일이 선택되지 않았을 경우 에러 메시지 출력
            FileRequired("이미지 파일을 지정해 주세요."),

            # 허용된 확장자만 업로드 가능
            # png, jpg, jpeg 외의 파일은 업로드 차단
            FileAllowed(
                ["png", "jpg", "jpeg"],
                "지원되지 않는 이미지 형식입니다."
            ),
        ]
    )

    # 폼 제출 버튼
    submit = SubmitField("업로드 하기")