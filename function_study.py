import pandas as pd
import korean
import matplotlib.pyplot as plt
import numpy as np

korean.korean_setup()

df = pd.read_excel("notExercise.xls") # 엑셀의 내용을 데이터프레임으로 변환하여 df 변수에 저장

## 대분류별로 입력 값에 따라 동적으로 함수를 통해 그래프를 만들 수 있는 코드.
## 인자값으로 카테고리를 써야하기 때문에 함수명을 신경써서 지었다.

input_category = input("대분류명은?")
def draw_grouped_pie(df, category):  # 매개변수로 엑셀 데이터프레임과 input_category 입력값을 받음
    
    # 데이터프레임 필터링해서 입력된 값과 비교하여 True인 데이터만 가져와서 새 데이터프레임 생성해서 변수에 할당
    data_df = df[df['대분류'] == category]

    columns = data_df.columns[3:7]  # 슬라이싱한 칼럼을 가져와서 변수에 할당
    plt.figure(figsize=(10, 10))    # 그래프 화면 크기

    for i in range(len(columns)):   # 슬라이싱한 칼럼의 수로 반복함
        plt.subplot(2, 2, i+1)  # 그래프 화면내 배치 설정
        plt.title(
            columns[i],     # 특정 칼럼의 i번째
            fontsize = 14,
            color="#303030",
            weight='bold'
            )
        
        colors = ["#FF8383","#FFF574","#A1D6CB",
                  "#C0DEFF","#A19AD3", "#F3D1F4",
                  "#FADFA1", "#B7C9F2","#F6FDC3"
                  ]
        
        
        # 이 리스트는 파이 차트에서 각 조각을 중심에서 얼마나 튀어나오게 할지 지정하는 용도
        # 'data_df'의 행(row) 개수만큼 반복문을 수행하여 'explode' 리스트를 생성
        # shape → (행 개수, 열 개수) 튜플 반환하여 행의 개수로 반복함
        # 각 요소는 0.015로 설정됨
        explode = list(0.015 for exp in range(data_df.shape[0]))

        plt.pie(
            # data_df 데이터프레임의 i번째 칼럼
            data_df[columns[i]],
            labels = data_df["분류"],   # data_df 데이터프레임의 분류 라벨 표시 (ex. 여자, 남자...)
            autopct = '%.1f%%',
            startangle = 200,
            counterclock = False,
            colors = colors,
            explode = explode,
            radius = 0.9,
            textprops={'color':"#303030", 'weight':'500'}
            )

    plt.tight_layout()
    plt.show()

draw_grouped_pie(df, input_category)




















































# input_data = input("이름은?")

# def function_test(name):
#     print(name+"님, 안녕하세요!^^")
    
# function_test(input_data)



