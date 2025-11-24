import pandas as pd
import korean
import matplotlib.pyplot as plt
import numpy as np

korean.korean_setup()

df = pd.read_excel("notExercise.xls", )


# '대분류' 컬럼에서 값이 '성별'인 행만 True로 선택됨
# True에 해당하는 행들만 모아서 gender_df로 저장
# 즉, 남자/여자(성별 그룹) 데이터만 뽑아서 따로 만든 DataFrame
gender_df = df[df['대분류'] == '성별']


# 파이그래프에서 각 조각에 붙일 이름(남자/여자)을 가져오는 부분
# gender_df의 '분류' 컬럼에는 '남자', '여자'가 들어있음
labels = gender_df["분류"]



# 그래프를 그릴 큰 화면(도화지) 크기를 설정하는 코드 (가로 10, 세로 10)
plt.figure(figsize=(10, 10))


# gender_df.columns[3:7] → 3~6번 컬럼(4개 컬럼)
# range(len(...)) → 0, 1, 2, 3 생성 → 총 4번 반복
for i in range(len(gender_df.columns[3:7])):
    
    
    # 2행 2열 중 i+1번째 subplot 위치 선택
    # i는 0부터 시작하므로 i+1 → 1,2,3,4 → 총 4칸 사용
    plt.subplot(2, 2, i+1)
    
    
    # 현재 그릴 그래프의 제목을 해당 컬럼 이름으로 설정
    # gender_df.columns[3:7][i] = 3~6번 컬럼 중 i번째 컬럼명
    plt.title(
        gender_df.columns[3:7][i],
        fontsize = 14,
        color="#303030",
        weight='bold'
        )
    
    colors = ["#615EFC", "#FF8A8A"]
    
    # explode = 각 조각을 얼마나 밖으로 띄울지 비율로 지정 (0 = 붙어있음, 0.1 = 10% 띄움)
    explode = [0.015, 0.015]  # 남자, 여자를 설정한만큼 띄움
    
    
    # 파이 그래프 그리기
    # values = gender_df[해당 컬럼명의 값들]  → 남/여 값 2개
    # labels = gender_df["분류"] → 남자/여자
    # autopct='%.1f%%' → 소수점 1자리 퍼센트 표시
    # startangle=90 → 위쪽(12시 방향)부터 시작
    # counterclock=False → 시계 방향으로 그리기
    plt.pie(
        gender_df[gender_df.columns[3:7][i]],   # 파이 조각 값(남자, 여자)
        labels = gender_df["분류"],              # 레이블(남자, 여자)
        autopct = '%.1f%%',                     # 퍼센트 표시 형식
        startangle = 200,                        # 시작 각도
        counterclock = False,                   # 시계방향으로 그리기
        colors = colors,
        explode=explode,
        textprops={'color':'white', 'weight':'600'}
        )
    
# 전체 subplot 간격 자동 조정 (그래프가 겹치지 않게)
plt.tight_layout()

# 최종 출력
plt.show()

