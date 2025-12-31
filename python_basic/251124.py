import pandas as pd
import numpy as np

# pd.Series(집합적 자료형)
# pd.Series(리스트)
s = pd.Series([1,2,3])


# pd.Series(튜플)
s = pd.Series((1.0,2.0,3.0))
print(s)


s2 = pd.Series(['a','a','c']) #dtype: object
print(s2)


# 리스트내에 서로 다른 type의 data가 있으면 형변환 일어남- 문자열로 변환됨
s_1 = pd.Series(['a',1,3.0]) #dtype: object
print(s_1)


s = pd.Series(range(10,14)) # index 인수는 생략됨
print(s)

print(range(10,14))


print(np.arange(200))


s3 = pd.Series(np.arange(200))
print(s3)


# NaN은 np.nan 속성을 이용해서 생성
s=pd.Series([1,2,3,np.nan,6,8])
s
# dtype: float64
# 판다스가 처리하는 자료구조인 시리즈와 데이터프레임에서 결측치가 있는 경우는 datatype이 float으로
print(s)



s=pd.Series([10,20,30],index=[1,2,3])
print(s)



s= pd.Series([95,100,88], index = ['홍길동','이몽룡','성춘향'])
print(s)



s0=pd.Series([10,20,30],index=[1,2,3])
print(s0)


print(s0.index)
#Int64Index([1, 2, 3], dtype='int64')



s00 = pd.Series([1,2,3]) # index를 명시하지 않음
print(s00.index)
# 범위 인덱스가 생성



s= pd.Series([9904312,3448737,289045,2466052],
    index=["서울","부산","인천","대구"])
print(s.index)
#Index(['서울', '부산', '인천', '대구'], dtype='object')


s.index.name = '광역시'
print(s)

print(s.index)


print(s.values)
# 시리즈의 값의 전체 형태는 array(numpy의 자료구조) 형태


s.name = '인구'
print(s)



print(s.index) # 문자열형 인덱스
s['인천'] # 문자형 인덱스로 접근
s[2] # 위치 인덱스 사용 가능



# 정수형 인덱스인 경우
s03 = pd.Series([1,2,3], index=[1,2,3])
s03
s03[1] # 명시적 인덱스(정수인덱스임) 사용
# s03[0] # 위치인덱스 접근 - KeyError
# 정수인덱스인 경우 위치인덱스는 사용 불가