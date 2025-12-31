## 숫자 합 구하기

# result 변수에 0을 넣음.
result = 0

# 1부터 50까지 변수 i에 넣음.
for i in range(1, 51):
    
    # result의 초깃값 0과 i의 값 1을 더하면 result 값은 1로 업데이트됨.
    # 50까지 계속 반복함. 
    # result = result + i
    result += i

print(result)




result = 0

for i in range(101, 151):    
    result += i
print(result)




result = 1

for i in range(1, 51):
    result *= i
print(result)




## 구구단

# 정수 2부터 9까지 숫자를 입력받아서 result에 대입함.
number = int(input('2부터 9까지 숫자 입력'))
print(number, '단')

# 1부터 9까지 숫자를 변수 i에 넣음.
for i in range(1, 10):

    # 변수 i의 1부터 9까지 정수를 입력받은 result 값과 곱함.
    result = number * i
    print(number, 'x', i, '=', result)

print('완성')



'''
연습을 위한 예제 !
while 문은 제외하고,
if, elif, else 같은 조건문과
for문, input()만 사용하는 반복문 중심 예제
'''

"""
1️⃣ 나이 확인기 (기초 if)

상황: 놀이공원 입장 나이 제한
설명: 사용자에게 나이를 입력받아서

19세 이상이면 “입장 가능합니다.”

그 미만이면 “보호자 동반 필요합니다.”
출력.
확장: 숫자가 아닌 값을 입력했을 때 예외처리는 선택.
"""

age = int(input('나이 입력하세요.'))

if age >= 19:
    print('입장 가능합니다.')
else:
    print('보호자 동반 필요합니다.')


'''
2️⃣ 짝수/홀수 판별기

상황: 숫자를 입력받아 짝수인지 홀수인지 출력
설명:

입력받은 수가 2로 나누어떨어지면 “짝수입니다.”

아니면 “홀수입니다.”
확장: 0은 짝수로 처리.
'''

num = int(input('숫자 입력'))
if num % 2 == 0:
    print('짝수입니다.')
else:
    print('홀수입니다.')



'''3️⃣ 학점 계산기 (이중 조건문)

상황: 학생의 점수를 입력받아서 학점을 계산
조건:

90~100 : A

80~89 : B

70~79 : C

60~69 : D

그 이하 : F

범위를 벗어나면 “잘못된 점수입니다.”
확장: 97점 이상은 “A+”로 표시.'''

num = int(input('점수 입력'))
if num > 100 or num < 0:
    print('잘못된 점수입니다.')
elif num >= 97:
    print('A+')
elif num >= 90:
    print('A')
elif num >= 80:
    print('B')
elif num >= 70:
    print('C')
elif num >= 60:
    print('D')
else:
    print('F')



food = ['피자', '치킨', '떡볶이', '스테이크', '딸기', '파스타']
print('좋아하는 음식 개수:', range(len(food)))
print('좋아하는 음식 순서:', food)



# 철수는 국어 90점, 수학 95점, 영어 78점, 과학 55점을 맞았다.
# 과목별 등급(A, B, C, D, F)

num = int(input('점수 입력'))
if num >= 90:
    print('A등급')
elif num >= 80:
    print('B등급')
elif num >= 70:
    print('C등급')
elif num >= 60:
    print('D등급')
else:
    print('F등급')



k_point = 90
 
if k_point >= 90:
    print('A등급')
elif 70 <= k_point < 90:
    print('B등급')
else:
    print('C등급')


m_point = 60
 
if m_point >= 90:
    print('A등급')
elif 70 <= m_point < 90:
    print('B등급')
else:
    print('C등급')







k_point = 90
if k_point >= 90:
    print('국어 A등급')

elif 80 <= k_point < 90:
    print('국어 B등급')

elif 70 <= k_point < 80:
    print('국어 C등급')

elif 60 <= k_point < 70:
    print('국어 D등급')

else:
    print('국어 F등급')


m_point = 95
if m_point >= 90:
    print('수학 A등급')

elif 80 <= m_point < 90:
    print('수학 B등급')

elif 70 <= m_point < 80:
    print('수학 C등급')

elif 60 <= m_point < 70:
    print('수학 D등급')

else:
    print('수학 F등급')



e_point = 78
if e_point >= 90:
    print('영어 A등급')

elif 80 <= e_point < 90:
    print('영어 B등급')

elif 70 <= e_point < 80:
    print('영어 C등급')

elif 60 <= e_point < 70:
    print('영어 D등급')

else:
    print('영어 F등급')


s_point = 55
if s_point >= 90:
    print('과학 A등급')

elif 80 <= s_point < 90:
    print('과학 B등급')

elif 70 <= s_point < 80:
    print('과학 C등급')

elif 60 <= s_point < 70:
    print('과학 D등급')

else:
    print('과학 F등급')















scores = [90,95,78,55]


for i in range(len(scores)): 
    if scores[i] >= 90:
        print('A등급')

    elif 80 <= scores[i] < 90:
        print('B등급')

    elif 70 <= scores[i] < 80:
        print('C등급')

    elif 60 <= scores[i] < 70:
        print('D등급')

    else:
        print('F등급')







# a = [1, 2, ['a', 'b', 'c',[5,6,7]], 3]

# print(a[2][3][1])


# 철수는 국어 90점, 수학 95점, 영어 78점, 과학 55점을 맞았다.
# 영희는 국어 78점, 수학 40점, 영어 98점, 과학 35점을 맞았다.

# 철수의 최고점수와 영희의 최고점수 중 최고 점수가 더 높은 사람은 누구인가?
# 정답예시 

# 철수의 최고점수 
chul = [90, 95, 78, 55]
chul.sort()

print('철수 최고점수: ', chul[3])


yeong = [78, 40, 98, 35]
yeong.sort()

print('영희 최고점수: ', yeong[3])

# 만약 철수의 최고점수가 더 높다면 철수로 표시되고,
# 그렇지 않다면 영희로 표시된다.

if chul[3] > yeong[3] :
    print(chul[3], '점', '철수 이김')
else:
    print(yeong[3],'점', '영희 이김')


