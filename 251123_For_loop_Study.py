# for문의 실행 결과를 예측하라
과일 = ["사과", "귤", "수박"]
for 변수 in 과일:
  print(변수)

'''
사과
귤
수박
'''



# for문의 실행 결과를 예측하라
과일 = ["사과", "귤", "수박"]
for 변수 in 과일:
  print("#####")

'''
#####
#####
#####
'''



# 다음 코드를 for문으로 작성하라.
print(10)
print(20)
print(30)

for a in [10, 20, 30]:
  print(a)





# 다음 코드를 for문으로 작성하라.
print(10)
print("-------")
print(20)
print("-------")
print(30)
print("-------")


for a in [10, 20, 30]:
  print(a)
  print("-------")




# for문을 사용해서 리스트에 저장된 값에 100을 더한 값을 출력하는 코드를 작성
리스트 = ["1,200", "1,300"]

for i in ["1,200", "1,300"]:
  a = i.replace(",", "")  # 문자열에서 특정 문자를 제거할 때
  print(int(a)+100)




# "나"와 "다" 문자열만 for문으로 출력
list = ["가", "나", "다"]

# 방법1 - 슬라이싱
for i in list[1:]:
  print(i)

# 방법2 - 조건문
for i in list:
  if i != "가":
    print(i)






# 다음과 같이 판매가가 저장된 리스트가 있을 때 부가세가 포함된 가격을 화면에 출력하라.
# 단 부가세는 10원으로 가정한다.
list = [100, 200, 300]
110
210
310

for i in list:
  print(i+10)





# 리스트에 저장된 값을 다음과 같이 출력하라.
list = ["김밥", "라면", "튀김"]

'''
오늘의 메뉴: 김밥
오늘의 메뉴: 라면
오늘의 메뉴: 튀김
'''

for i in list:
  print("오늘의 메뉴: ", i)






# 저장된 문자열의 길이를 다음과 같이 출력하라.
list = ["SK하이닉스", "삼성전자", "LG전자"]

'''
6
4
4
'''

for i in list:
  print(len(i))   # 문자열의 길이를 구하는 함수 len()





# 동물 이름과 글자수를 다음과 같이 출력하라.
list = ['dog', 'cat', 'parrot']

'''
dog 3
cat 3
parrot 6
'''

for i in list:
  print(i, len(i))





# for문을 사용해서 동물 이름의 첫 글자만 출력하라.
list = ['dog', 'cat', 'parrot']

'''
d
c
p
'''

for i in list:
  print(i[0])





# for문을 사용해서 다음과 같이 출력하라.
list = [1, 2, 3]

'''
3 x 1
3 x 2
3 x 3

'''

for i in list:
  print(3, "x", i)


for i in list:
  print(3, "x", i, "=", 3*i)    # 3 x 1 = 3, 3 x 2 = 6, 3 x 3 = 9





list = ["가", "나", "다", "라"]

'''
나
다
라
'''

for i in list[1:]:
  print(i)


for i in range(len(list)):    # 가, 다 출력하려면
  if i % 2 == 0:
    print(list[i])

for i in list[::2]:    # 슬라이싱
  print(i)


# 역순으로 출력하기
for a in reversed(list):
  print(a)

for a in list[::-1]:
  print(a)

list.reverse()    # 원본 리스트를 실제로 뒤집고 싶을 때만
for a in list:
  print(a)





# for문을 사용해서 리스트의 음수를 출력하라.
list = [3, -20, -3, 44]

'''
-20
-3
'''

for i in list[1:3]:
  print(i)

for i in list:
  if i < 0:
    print(i)





# for문을 사용해서 3의 배수만을 출력하라.
list = [3, 100, 23, 44]

for i in list:
  if i % 3 == 0:
    print(i)




# 리스트에서 20 보다 작은 3의 배수를 출력하라
list = [13, 21, 12, 14, 30, 18]

for i in list:
  if (i < 20) and (i % 3 == 0):
    print(i)




# 리스트에서 세 글자 이상의 문자를 화면에 출력하라
list = ["I", "study", "python", "language", "!"]

for i in list:
  if len(i) >= 3:
    print(i)




# 리스트에서 대문자만 화면에 출력하라
list = ["A", "b", "c", "D"]

for i in list:
  a = i.isupper()   # isupper() 메서드는 대문자 여부를 판별합니다
  if a == True:   # a가 참이라면, 대문자라면 i를 출력
    print(i)


for i in list:
  if i.isupper():   # i가 대문자라서 참이라면 i를 출력
    print(i)



# 리스트에서 소문자만 화면에 출력
for i in list:
  if i.islower():   # 문자열이 모두 소문자인지 확인
    print(i)


for i in list:
  if i.isupper() == False:
    print(i)


for i in list:
  if not i.isupper():   # 논리 연산자 not을 사용
    print(i)


for i in list:
  a = i.isupper()   
  if a == False:
    print(i)


for i in list:
  a = i.isupper()   
  if a != True:
    print(i)







# 이름의 첫 글자를 대문자로 변경해서 출력하라
list = ['dog', 'cat', 'parrot']

for i in list:
  print(i[0].upper() + i[1:])   # 첫번째 문자를 대문자로 바꾸고, 첫번째 제외한 나머지 문자를 이어붙임




# 파일 이름이 저장된 리스트에서 확장자를 제거하고 파일 이름만 화면에 출력하라
list = ['hello.py', 'ex01.py', 'intro.hwp']

for i in list:
  print(i.split(".")[0])    # split() 메서드 - 입력된 구분자로 분할해서 리스트로 반환   ['hello', 'py']





# 파일 이름이 저장된 리스트에서 확장자가 .h인 파일 이름을 출력하라
list = ['intra.h', 'intra.c', 'define.h', 'run.py']

for i in list:
  if i.split(".")[1] == "h":    # split() 메서드로 문자열을 분할하고 확장자가 "h"인지 분기문으로 비교
    print(i)


# 파일 이름이 저장된 리스트에서 확장자가 .h나 .c인 파일을 화면에 출력하라
for i in list:
  if (i.split(".")[1] == "h") or (i.split(".")[1] == "c"):    # 논리 연산자 or을 사용해서 두 개의 확장자를 비교
    print(i)






# for문과 range 구문을 사용해서 0~99까지 한 라인에 하나씩 순차적으로 출력하는 프로그램을 작성
for i in range(100):
  print(i)





# 월드컵은 4년에 한 번 개최된다.
# range()를 사용하여 2002~2050년까지 중 월드컵이 개최되는 연도를 출력하라
for i in range(2002, 2051, 4):    # range의 세번 째 파라미터는 증감폭을 결정함
  print(i)





# 1부터 30까지의 숫자 중 3의 배수를 출력하라
for i in range(1, 31):
  if i % 3 == 0:
    print(i)

for i in range(3, 31, 3):
  print(i)




# 99부터 0까지 1씩 감소하는 숫자들을, 한 라인에 하나씩 출력하라
for i in reversed(range(100)):
  print(i)

for i in range(100):
  print(99 - i)




# for문을 사용해서 아래와 같이 출력하라
'''
0.0
0.1
0.2
0.3
0.4
0.5
...
0.9
'''

for i in range(0, 10):
  print(i / 10)




# 구구단 3단 출력하라
for i in range(1, 10):
  print(3, 'x', i, '=', 3*i)


# 3단 홀수만 출력
for i in range(1, 10):    # 조건문 이용
  if i % 2 != 0:
    print(3, 'x', i, '=', 3*i)

for i in range(1, 10):    # 조건문 이용
  if i % 2 == 1:
    print(3, 'x', i, '=', 3*i)

for i in range(1, 10, 2):   # 증감폭 이용
  print(3, 'x', i, '=', 3*i)




# 1~10까지의 숫자에 대해 모두 더한 값을 출력하는 프로그램을 for 문을 사용하여 작성
num = 0
for i in range(1, 11):
  num = num + i   # num += i
print("합: ", num)





# 1~10까지의 숫자 중 모든 홀수의 합을 출력하는 프로그램을 for 문을 사용하여 작성
num1 = 0
for i in range(1, 11):
  if i % 2 != 0:
    num1 += i
print("합: ", num1)


num2 = 0
for i in range(1, 11, 2):   # 증감폭 이용
  num2 += i
print("합: ", num2)




# 1~10까지의 숫자를 모두 곱한 값을 출력하는 프로그램을 for 문을 사용하여 작성
num3 = 1
for i in range(1, 11):
  num3 *= i
print(num3)




# 아래와 같이 리스트의 데이터를 출력하라. 단, for문과 range문을 사용
price_list = [32100, 32150, 32000, 32500]

'''
0 32100
1 32150
2 32000
3 32500
'''

for i in range(len(price_list)):
  print(i, price_list[i])




'''
3 32100
2 32150
1 32000
0 32500
'''

num4 = len(price_list)
for i in range(len(price_list)):
  num4 -= 1
  print(num4, price_list[i])




'''
100 32150
110 32000
120 32500
'''
num5 = 90
for i in range(1, 4):   # for i in range(1, len(price_list)): 이렇게도 가능
  num5 += 10
  print(num5, price_list[i])





# my_list를 아래와 같이 출력하라
my_list = ["가", "나", "다", "라"]

'''
가 나
나 다
다 라
'''

for i in range(3):    # 0, 1, 2
  print(my_list[i], my_list[i+1])   # 여기는 [i+1] 가능함. my_list[3]가 있기 때문임

for i in range(1, len(my_list)):
  print(my_list[i-1], my_list[i])   # 여기는 [i+1]하면 my_list[4]가 없어서 오류남




# 리스트를 아래와 같이 출력하라
my_list = ["가", "나", "다", "라", "마"]

'''
가 나 다
나 다 라
다 라 마
'''
for i in range(2, len(my_list)):
  print(my_list[i-2], my_list[i-1], my_list[i])



# 반복문과 range 함수를 사용해서 my_list를 아래와 같이 출력하라
my_list = ["가", "나", "다", "라"]

'''
라 다
다 나
나 가
'''
for i in reversed(range(1, len(my_list))):
  print(my_list[i], my_list[i-1])




# 리스트에는 네 개의 정수가 저장되어 있다.
# 각각의 데이터에 대해서 자신과 우측값과의 차분값을 화면에 출력하라
my_list = [100, 200, 400, 800]

'''
100
200
400

'''
for i in range(1, len(my_list)):
    print(my_list[i]-my_list[i-1])




# 리스트에는 6일 간의 종가 데이터가 저장되어 있다.
# 종가 데이터의 3일 이동 평균을 계산하고 이를 화면에 출력하라
my_list = [100, 200, 400, 800, 1000, 1300]

'''
233.33333333333334
466.6666666666667
733.3333333333334
1033.3333333333333
'''
for i in range(2, len(my_list)):
  print((my_list[i-2] + my_list[i-1] + my_list[i]) / 3)





# 리스트에 5일간의 저가, 고가 정보가 저장돼 있다.
# 고가와 저가의 차를 변동폭이라고 정의할 때,
# low, high 두 개의 리스트를 사용해서 5일간의 변동폭을 volatility 리스트에 저장하라
low_prices  = [100, 200, 400, 800, 1000]
high_prices = [150, 300, 430, 880, 1000]

volatility1 = []
for a, b in zip(high_prices, low_prices):
  volatility1.append(a - b)
print(volatility1)


volatility2 = []
for i in range(len(low_prices)):    # 두 리스트의 길이가 같을 때 사용 가능
  volatility2.append(high_prices[i] - low_prices[i])
print(volatility2)






