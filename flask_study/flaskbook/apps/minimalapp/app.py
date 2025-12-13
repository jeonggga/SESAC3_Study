# Flask 모듈에서 Flask 클래스를 임포트합니다. 웹 애플리케이션 객체를 생성하는 데 사용됩니다.
from flask import (Flask, 
                   current_app, 
                   make_response, 
                   redirect, 
                   render_template, 
                   request, 
                   session, 
                   url_for, 
                   flash,)
import os
from flask_mail import Mail, Message
from email_validator import EmailNotValidError, validate_email

from views import dt          # views.py에 정의된 Blueprint(dt) 불러오기
from config import BaseConfig # Flask 설정값을 담은 Config 클래스 불러오기




app = Flask(__name__)
# Flask 애플리케이션 객체를 생성합니다.
# '__name__'은 현재 모듈의 이름으로, Flask가 리소스(템플릿, 정적 파일)를 찾을 위치를 결정하는 데 도움을 줍니다.
                



# ------------------------------
#   이미지 업로드 기능에 필요한 함수
# ------------------------------
def create_app():
    # Flask 애플리케이션 객체 생성
    app = Flask(__name__)

    # config.py에 정의된 BaseConfig 클래스의 설정값을 Flask 앱에 적용
    # (예: UPLOAD_FOLDER, SECRET_KEY 등)
    app.config.from_object(BaseConfig)

    # 업로드 폴더가 실제로 존재하는지 확인하고,
    # 없으면 자동으로 생성 (이미 있으면 에러 없이 넘어감)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Blueprint 등록
    # dt에 정의된 라우트(/api, /dt 등)를 Flask 앱에 연결
    app.register_blueprint(dt)

    # 설정이 완료된 Flask 앱 반환
    return app


# 애플리케이션 팩토리 패턴을 사용해 앱 생성
app = create_app()






                
# SECRET_KEY를 추가한다
# ------------------------------
# 🔐 Flask의 세션/폼 보안을 위한 SECRET_KEY 설정
#    - 쿠키, 세션, CSRF 보호에 사용됨
#    - os.urandom(24)는 24바이트의 랜덤 값을 생성해서
#      보안을 강화하기 위해 임의의 비밀키를 만든 것
# ------------------------------
app.config["SECRET_KEY"] = os.urandom(24)



# ------------------------------
# 📧 Flask-Mail 설정 (환경변수에서 값 가져오기)
#    - 메일 서버에 연결하기 위해 필요한 정보들 (코드에 비밀번호를 직접 안 적어도 되게 하기 위한 방법)
# ------------------------------
# Mail 클래스의 컨피그를 추가한다

# 메일 서버 주소 (예: smtp.gmail.com)
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")

# 메일 서버 포트 번호 (보통 587)
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")

# TLS 보안 사용 여부 (True / False)
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")

# 메일 서버 로그인 계정 (발신 이메일 주소)
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")

# 메일 서버 로그인 비밀번호 (앱 비밀번호 사용 권장)
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")

# 기본 발신자 설정 (이메일 발송 시 자동 적용)
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")



# ------------------------------
# 📮 Flask-Mail 기능을 Flask 앱과 연결하기
#    - 위에서 설정한 config를 바탕으로 Mail 객체를 만듦
#    - 이제 mail.send()로 이메일 발송 가능
# ------------------------------
# flask-mail 확장을 등록한다
mail = Mail(app)
                
                

                

# ------------------------------
# ------------------------------
#   라우트 정의 시작
# ------------------------------
# ------------------------------


@app.route('/')
# 루트 URL ('/')에 대한 라우트를 정의합니다.
# 사용자가 이 URL로 접근하면 바로 아래의 index 함수가 실행됩니다.
     
def index():
    return render_template("index.html")
# 함수가 실행될 때 웹 브라우저에 'Hi!!!'라는 문자열을 응답으로 반환합니다.


@app.route('/hello/<name>', # '/hello/' 뒤에 변수(variable part)를 포함하는 URL에 대한 라우트를 정의합니다.
                            # <name> 부분은 URL에서 추출되어 hello 함수의 인수로 전달됩니다.
             methods=['GET'], # 이 라우트는 HTTP GET 요청만 처리하도록 지정합니다. (선택 사항이지만 명시적으로 지정)
             endpoint='hello_endpoint') # 이 라우트에 'hello_endpoint'라는 고유한 이름을 부여합니다.
                                        # URL을 동적으로 생성하거나(url_for 함수) 뷰 함수를 참조할 때 사용될 수 있습니다.
                                        
def hello(name): # 라우트에 연결된 뷰 함수입니다. URL에서 추출된 name 값이 인수로 전달됩니다.
    return f'Hello, World! {name}' # 전달받은 name 값을 포함하여 포맷팅된 문자열을 응답으로 반환합니다.




# ----------------------------------------
# /admin/뒤에 이름을 넣으면 그 이름을 받아서
# admin.html 페이지에 보내주는 기능
# ----------------------------------------
@app.route("/admin/<name>")
def admin(name):
    # 이름 글자 수를 계산
    leng = len(name)  # 예: "minsu" → 5

    # admin.html 페이지에 이름과 글자 수를 같이 보내기
    # - name: 사용자가 입력한 이름
    # - leng: 이름 글자 수
    return render_template('admin.html', name=name, leng=leng)





# ------------------------------
#   구구단 기능에 필요한 함수
# ------------------------------
@app.route("/gugudan", methods=["GET"])
def dan():
    # 사용자가 입력한 값 가져오기 (예: ?num=5)
    # 아무 것도 입력하지 않으면 user_input은 None
    user_input = request.args.get("num")
    error_message = None  # 오류 메시지를 담을 변수, 처음에는 없음

    # user_input이 존재할 때만 처리
    if user_input is not None:
        if user_input.strip() == "":  
            # 사용자가 아무것도 입력하지 않고 제출한 경우
            error_message = "* 숫자를 입력해 주세요"
        elif user_input.isdigit():  
            # 입력값이 숫자인 경우
            num = int(user_input)  # 문자열을 정수로 변환
        else:  
            # 숫자가 아닌 값이 입력된 경우
            error_message = "* 숫자 형식으로 입력해 주세요"
    else:
        # 처음 페이지 접속 시 입력값이 없으면 num 변수 없음
        num = None  

    # render_template에 값을 넘김
    # num: 숫자가 입력되면 구구단 출력, 없으면 구구단 안 뜸
    # user_input: 텍스트 필드에 사용자가 입력한 값 유지
    # error_message: 오류가 있으면 화면에 표시
    return render_template(
        "gugudan.html",
        num=num if 'num' in locals() else None,  
        user_input=user_input,
        error_message=error_message
    )



# @app.route("/gugudan/<int:num>")

# def dan(num):
#     title = f'{num}단'
#     gugudan = []
#     for n in range(1, 10):
#         temp = num*n
#         gugudan.append(f'<li>{num} x {n} = {temp}</li>')
#     gugudan = "".join(gugudan)
#     return f'''<!DOCTYPE html>
# <html lang="ko">
#     <head>
#         <meta charset="UTF-8">
#         <title>구구단 : {title}</title>
#     </head>
#     <body>
#         <h1>{title}</h1>
#             <ul>
#                 {gugudan}
#             </ul>
#     </body>
# </html>
# '''


# with app.test_request_context():
#     print(url_for("index"))
#     print(url_for("hello_endpoint",name="abc"))
#     print(url_for("show_name",name="bbb"))
#     print(url_for("dan_endpoint",num=2))





# ------------------------------
#   문의하기 기능에 필요한 함수
# ------------------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")
# /contact 라는 주소로 들어오면
# contact.html 파일을 보여주는 기능
# 예) 브라우저에 /contact 를 치면
#     contact.html 화면이 열림



@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    # request.method : 현재 요청의 HTTP 메서드(GET, POST 등)를 확인
    # 만약 이 페이지로 "POST" 방식(폼 제출)으로 왔다면
    if request.method == "POST":
        # POST 요청이면, 사용자를 다시 같은 페이지로 리다이렉트
        # redirect() : 브라우저를 다른 URL로 이동시킴 (HTTP 302)
        
        # 사용자가 contact.html에서 입력한 값 받아오기
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]
        
        
        # 사용자가 입력한 내용이 제대로 되어 있는지 확인하기
        is_valid = True
        
        # 이름이 비어 있으면 오류 메시지 보여줌
        if not username:
            flash("* 사용자명은 필수입니다.")
            is_valid = False
        
        # 이메일이 비어 있으면 오류 메시지
        if not email:
            flash("* 메일 주소는 필수입니다.")
            is_valid = False
        
        # 이메일 형태가 맞는지 검사하기
        try:
            validate_email(email)
        except EmailNotValidError:
            flash("* 메일 주소의 형식으로 입력해 주세요")
            is_valid = False

        # 문의 내용이 비어 있으면 오류 메시지
        if not description:
            flash("* 문의 내용은 필수입니다.")
            is_valid = False

        # 하나라도 틀린 게 있다면 → 다시 contact 페이지로 돌려보냄
        if not is_valid:
            return redirect(url_for("contact"))
        
        # 모두 제대로 입력했으면 메시지 보여주기
        flash("문의 내용은 메일로 송신했습니다. 문의해 주셔서 감사합니다.")

        # 실제로 이메일 보내기
        send_email(email,
                   "문의 감사합니다.",
                   "contact_mail",
                   username = username,
                   description = description,)
        
        # 이메일 보낸 뒤, 같은 페이지로 다시 이동 (PRG 패턴)
        return redirect(url_for("contact_complete"))
    


    # -----------------------------------------
    # GET 요청이거나 POST 후 리다이렉트된 상태
    # 즉, 단순히 페이지에 "방문"한 상태
    # "contact_complete.html" 템플릿을 렌더링하여 사용자에게 보여줌
    # -----------------------------------------
    return render_template("contact_complete.html")


# --------------------------------
# ⭐ 초간단 요약
# --------------------------------

# POST로 왔다면 → 폼(이름, 이메일, 내용)에서 가져온 값 확인
# 값이 비어 있거나 이상하면 → contact 페이지로 다시 보내기
# 문제가 없으면 → 이메일 보내기
# 그 다음 contact_complete 페이지로 보내기
# GET이면 → 그냥 contact_complete.html 보여주기





# ------------------------------
#   이메일 보내는 함수
# ------------------------------
def send_email(to, subject, template, **kwargs):
    
    """
    - to: 이메일 받을 사람
    - subject: 이메일 제목
    - template: 사용할 템플릿 이름 (ex: "contact_mail")
    - **kwargs: 템플릿에 넣을 데이터 (이름, 내용 등)
    """
    
    # 1️⃣ 이메일 메시지 객체 만들기
    # subject: 제목
    # recipients: 받을 사람 리스트 (한 명이라도 리스트로 넣어야 함)
    msg = Message(subject, recipients=[to])
    
    # 2️⃣ 템플릿을 사용해서 이메일 내용 만들기
    # txt 파일 → 일반 텍스트 이메일
    msg.body = render_template(template + ".txt", **kwargs)
    
    # html 파일 → 꾸민 HTML 이메일
    msg.html = render_template(template + ".html", **kwargs)
    
    # 3️⃣ 이메일 실제로 보내기
    mail.send(msg)
    

# --------------------------------
# 🌟 초간단 요약
# --------------------------------

# 이메일을 만든다 → 제목, 받을 사람 지정
# 템플릿에서 내용을 만들어 넣는다 → 일반 글과 HTML
# mail.send()로 이메일 전송
# 즉, 이 함수는 “받는 사람, 제목, 내용만 주면 이메일을 보내주는 마법 함수” 라고 생각하면 쉽다








# ------------------------------
# ------------------------------
#   라우트 정의 끝
# ------------------------------
# ------------------------------


if __name__ == "__main__":
    # 이 파일을 직접 실행하면 Flask 서버 실행

    # 개발 서버를 실행합니다. 'port=8080'은 서버가 8080 포트에서 수신 대기하도록 지정합니다.
    app.run(port=8080, debug=True)
    # 디버그 모드를 활성화합니다.
    # 코드를 수정하고 저장하면 서버가 자동으로 재시작되며, 오류 발생 시 상세한 디버그 정보를 웹 페이지에 표시합니다.
