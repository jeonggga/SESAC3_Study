import korean
import matplotlib.pyplot as plt
import numpy as np


korean.korean_setup()




height = [165, 177, 160, 180, 185, 155, 172]
weight = [62, 67, 55, 74, 90,43, 64]

plt.scatter(height, weight)
plt.xlabel('height(m)')
plt.ylabel('weight(kg)')
plt.title('Height & Weight')
plt.grid(True)

plt.scatter(height, weight, s=500, c='r') # 마커 크기는 500, 컬러는 붉은색(red)
plt.show()




size = 100 * np.arange(1,8) # 데이터별로 마커의 크기 지정
colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y'] # 데이터별로 마커의 컬러 지정
plt.scatter(height, weight, s=size, c=colors)
plt.show()





city = ['서울', '인천', '대전', '대구', '울산', '부산', '광주']
# 위도(latitude)와 경도(longitude)
lat = [37.56, 37.45, 36.35, 35.87, 35.53, 35.18, 35.16]
lon = [126.97, 126.70, 127.38, 128.60, 129.31, 129.07, 126.85]
# 인구 밀도(명/km^2): 2017년 통계청 자료
pop_den = [16154, 2751, 2839, 2790, 1099, 4454, 2995]
size = np.array(pop_den) * 0.2 # 마커의 크기 지정
colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y'] # 마커의 컬러 지정


plt.scatter(lon, lat, s=size, c=colors, alpha=0.5)
plt.xlabel('경도(longitude)')
plt.ylabel('위도(latitude)')

plt.title('지역별 인구 밀도(2017)')
for x, y, name in zip(lon, lat, city):
 plt.text(x, y, name) # 위도 경도에 맞게 도시 이름 출력
plt.show()




member_IDs = ['m_01', 'm_02', 'm_03', 'm_04'] # 회원 ID
before_ex = [27, 35, 40, 33] # 운동 시작 전
after_ex = [30, 38, 42, 37] # 운동 한 달 후

n_data = len(member_IDs) # 회원이 네 명이므로 전체 데이터 수는 4
index = np.arange(n_data) # NumPy를 이용해 배열 생성 (0, 1, 2, 3)
plt.bar(index, before_ex) # bar(x,y)에서 x=index, height=before_ex 로 지정
plt.show()

plt.bar(index, before_ex, tick_label = member_IDs)
plt.show()

colors=['r', 'g', 'b', 'm']
plt.bar(index, before_ex, color = colors, tick_label = member_IDs)
plt.show()

colors=['b', 'y', 'b', 'y']
plt.bar(index, before_ex, tick_label = member_IDs, width = 0.6, color = colors)
plt.show()




colors=['r', 'g', 'r', 'g']
plt.barh(index, before_ex, color = colors, tick_label = member_IDs)
plt.show()





barWidth = 0.4
plt.bar(index, before_ex, color='r', align='edge',
 width = barWidth, label='before')
plt.bar(index + barWidth, after_ex , color='g', align='edge',
 width = barWidth, label='after')
plt.xticks(index + barWidth, member_IDs)
plt.legend()
plt.xlabel('회원 ID')
plt.ylabel('윗몸일으키기 횟수')
plt.title('운동 시작 전과 후의 근지구력(복근) 변화 비교')
plt.show()




math = [76, 82, 84, 83, 90, 86, 85, 92, 72, 71, 100,
 87, 81, 76, 94, 78, 81, 60, 79, 69, 82, 68, 79]
plt.hist(math)
plt.hist(math, bins=4)
plt.show()

plt.hist(math, bins= 8)
plt.xlabel('시험 점수')
plt.ylabel('도수(frequency)')
plt.title('수학 시험의 히스토그램')
plt.grid()
plt.show()



fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
result = [7, 6, 3, 2, 2]

plt.pie(result)
plt.show()


plt.figure(figsize=(5,5))
plt.pie(result)
plt.show()

plt.figure(figsize=(5,5))
plt.pie(result, labels= fruit, autopct='%.1f%%')
plt.show()


plt.figure(figsize=(6,6))
colors = ['#ff6f69', '#ffcc5c', '#ff9a76', "#d8a188", '#89cff0']
plt.pie(result,
        labels= fruit,
        autopct='%.1f%%',
        startangle=90,
        textprops={'fontsize': 14, 'color': '#4e3a31', 'fontweight':'bold'},
        counterclock = False,
        colors = colors)
plt.title("<과일 비율 시각화>", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()




explode_value = (0.1, 0, 0, 0, 0)
plt.figure(figsize=(5,5))
colors = ['#ff6f69', '#ffcc5c', '#ff9a76', "#d8a188", '#89cff0']
plt.pie(result, labels= fruit, autopct='%.1f%%', startangle=90,
        textprops={'fontsize': 14, 'color': '#4e3a31', 'fontweight':'bold'},
        colors = colors,
        counterclock = False, explode=explode_value, shadow=True)
plt.title("<과일 비율 시각화>", fontsize=18, fontweight='bold', pad=20)
plt.show()



import matplotlib as mpl

mpl.rcParams['figure.figsize']
mpl.rcParams['figure.dpi']


x = np.arange(0, 5, 1)
y1 = x
y2 = x + 1
y3 = x + 2
y4 = x + 3
plt.plot(x, y1, x, y2, x, y3, x, y4)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Saving a figure')
# 그래프를 이미지 파일로 저장. dpi는 100으로 설정
plt.savefig('saveFigTest1.png', dpi = 100)
plt.show()



fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
result = [7, 6, 3, 2, 2]

explode_value = (0.1, 0, 0, 0, 0)
plt.figure(figsize=(5,5)) # 그래프의 크기를 지정
plt.pie(result, labels= fruit, autopct='%.1f%%', startangle=90,
 counterclock = False, explode=explode_value, shadow=True)
# 그래프를 이미지 파일로 저장. dpi는 200으로 설정
plt.savefig('saveFigTest2.png', dpi = 200)
plt.show()




import pandas as pd

s1 = pd.Series([1,2,3,4,5,6,7,8,9,10])
s1

s1.plot()
plt.show()



s2 = pd.Series([1,2,3,4,5,6,7,8,9,10],
 index = pd.date_range('2019-01-01', periods=10))

print(s2)
s2.plot()
plt.show()

s2.plot(grid=True)
plt.show()




df_rain = pd.read_csv('sea_rain1.csv', index_col="연도" )
print(df_rain)


rain_plot = df_rain.plot(grid = True, style = ['r--*', 'g-o', 'b:*', 'm-.p'])
rain_plot.set_xlabel("연도")
rain_plot.set_ylabel("강수량")
rain_plot.set_title("연간 강수량")
plt.show()


year = [2006, 2008, 2010, 2012, 2014, 2016] # 연도
area = [26.2, 27.8, 28.5, 31.7, 33.5, 33.2] # 1인당 주거면적
table = {'연도':year, '주거면적':area}
df_area = pd.DataFrame(table, columns=['연도', '주거면적'])
print(df_area)


df_area.plot(x='연도', y='주거면적', grid = True,
 title = '연도별 1인당 주거면적')
plt.show()




temperature = [25.2, 27.4, 22.9, 26.2, 29.5, 33.1, 30.4, 36.1, 34.4, 29.1]
Ice_cream_sales = [236500, 357500, 203500, 365200, 446600,
 574200, 453200, 675400, 598400, 463100]
dict_data = {'기온':temperature, '아이스크림 판매량':Ice_cream_sales}
df_ice_cream = pd.DataFrame(dict_data, columns=['기온', '아이스크림 판매량'])
print(df_ice_cream)


df_ice_cream.plot.scatter(x='기온', y='아이스크림 판매량', grid=True, title='최고 판매량')
plt.show()



grade_num = [5, 14, 12, 3]
students = ['A', 'B', 'C', 'D']
df_grade = pd.DataFrame(grade_num, index=students, columns = ['Student'])
print(df_grade)


grade_bar = df_grade.plot.bar(grid = True)
grade_bar.set_xlabel("학점")
grade_bar.set_ylabel("학생수")
grade_bar.set_title("학점별 학생 수 막대 그래프")
plt.show()



math = [76,82,84,83,90,86,85,92,72,71,100,87,81,76,
 94,78,81,60,79,69,74,87,82,68,79]
df_math = pd.DataFrame(math, columns = ['Student'])
math_hist = df_math.plot.hist(bins=8, grid = True)
math_hist.set_xlabel("시험 점수")
math_hist.set_ylabel("도수(frequency)")
math_hist.set_title("수학 시험의 히스토그램")
plt.show()



fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
result = [7, 6, 3, 2, 2]
df_fruit = pd.Series(result, index = fruit, name = '선택한 학생수')
print(df_fruit)

df_fruit.plot.pie()
plt.show()



explode_value = (0.1, 0, 0, 0, 0)
fruit_pie = df_fruit.plot.pie(figsize=(5, 5), autopct='%.1f%%', startangle=90,
 counterclock = False, explode=explode_value, shadow=True, table=True)
fruit_pie.set_ylabel("") # 불필요한 y축 라벨 제거
fruit_pie.set_title("과일 선호도 조사 결과")
# 그래프를 이미지 파일로 저장. dpi는 200으로 설정
plt.savefig('saveFigTest3.png', dpi = 200)
plt.show()




