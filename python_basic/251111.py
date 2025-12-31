f = open('twoo_times_table.txt', 'w')
for num in range(1, 6):
    format_string = "2 x {0} = {1}\n".format(num, 2*num)
    f.write(format_string)
f.close()




f = open("twoo_times_table.txt")
line1 = f.readline()
line2 = f.readline()
f.close()
print(line1, end="")
print(line2, end="")




f = open("twoo_times_table.txt")
line = f.readline()

while line:
    print(line, end="")
    line = f.readline()
f.close()




f = open("twoo_times_table.txt")
lines = f.readlines()
f.close()

print(lines)




f = open("twoo_times_table.txt")
lines = f.readlines()

f.close()
for line in lines:
    print(line, end = "")




def my_friend(friendName):
    print("{}는 나의 친구입니다.".format(friendName))
my_friend("철수")
my_friend("영미")



def my_student_info(name, school_ID, phoneNumber):
    print("********************************")
    print("* 학생이름:", name)
    print("* 학급번호:", school_ID)
    print("* 전화번호:", phoneNumber)
my_student_info("현아", "01", "01-235-6789")
my_student_info("진수", "02", "01-987-6543")



def my_calc(x, y):

    z = x*y
    print(z)
    return z

print(my_calc(5, 7))




def my_student_info_list(student_info):
    print("********************************")
    print("* 학생이름:", student_info[1])
    print("* 학급번호:", student_info[0])
    print("* 전화번호:", student_info[2])
    print("********************************")

abc = ["01", "현아", "01-235-6789"]
my_student_info_list(abc)



a = 5

def func1():
    a = 1
    print("[func1] 지역 변수 a =", a)

def func2():
    a = 2
    print("[func3] 지역 변수 a =", a)

def func3():
    print("[func3] 전역 변수 a =", a)

def func4():
    global a
    a = 4
    print("[func4] 전역 변수 a =", a)


func1()
func2()
print("전역 변수 a =", a)


func3()
func4()
func3()



def print_coin():
    print("비트코인")




for a in range(1, 101) :
    print_coin()




def message() :
    print("A")
    print("B")

message()
print("C")
message()







#끝말잇기###
# 무한으로 계속 끝말잇기를 하기 때문에 범위가 없어서 while문 활용
# 반복문은 항상 참임
num = 1   # 반복 횟수를 알기위해 변수 생성
hangle1 = input('끝말잇기 단어 입력 : ')  # 처음 입력 받은 값을 저장하기 위해 변수 생성

print(num, hangle1)

while True:
    hangle2 = input('끝말잇기 단어 입력 : ')  # 두번째 입력값 받음
    if hangle2 == "그만":  # 입력 받은 값이 "그만"과 같다면
        break  # 반복문 빠져나옴
    if hangle1[-1] != hangle2[0]:  # 처음 입력 받은 값의 -1자리와 두번째 입력 받은 값의 0자리를 비교함
        print('다시 입력해라')  # 일치하지 않으면 해당 문구 출력되고,
        continue   # 반복문 첫 줄로 올라감

    num += 1   # 위의 조건문에 해당되지 않으면 num +1 증가시켜서 반복 횟수 표시함
    print(num, "{}".format(hangle2))  #print(num, hangle2)  #포매팅이 쓰고 싶어서
    hangle1 = hangle2   # 두번째 입력 값을 전역 함수에 할당함.






# 3단 입력시 3단 나오게 코드짜기
# 이중 for문을 이용하여 입력 받은 숫자의 구구단만 보여주는 코드

dan = int(input('몇 단을 보고 싶은가? : '))  # 처음 입력 받는 부분

while True:  # 무한으로 계속 입력 받을 수 있음
    if dan == "":
        print('숫자가 입력되지 않았습니다.')
        break

    print('===', "{}".format(dan), '단===')  # 몇 단인지 표시함

    for num in range(1, 10):  # 1부터 9까지 반복문으로 곱셈
        print(dan, 'x', num, '=', dan*num)
    dan = int(input('몇 단을 보고 싶은가? : '))  # 반복문이 끝나면 입력 값 받음





def print_with_smile(string):
    print(string+":D")

print_with_smile("안녕하세요")
print_with_smile("반갑습니다")


def print_upper_price(price):
    print(price * 1.3)
    
print_upper_price(45000)




def print_sum(a, b):
    print(a + b)
print_sum(3, 5)



# def print_arithmetic_operation(a, b):
#     print(a, "+", b, "=", a+b)
#     print(a, "-", b, "=", a-b)
#     print(a, "*", b, "=", a*b)
#     print(a, "/", b, "=", a/b)
# print_arithmetic_operation(3, 4)




# def print_max(a, b, c):
#     if a > b:
#         print(b)
#     if b > a:
#         print(b)
#     if c > 


# print_max(40, 60, 80)




'''
학생 5명에 학생 이름, 국어, 수학, 영어, 과학 점수를 입력 받아
1번과 2번을 출력하는 프로그램을 작성하시오.
1. 학생에 따른 점수의 평균 및 전체 평균
2. 각 과목별 최대 점수, 최소 점수
'''

# name = input("학생 이름은?")
# guk = int(input("국어 점수는?"))
# su = int(input("수학 점수는?"))
# yeong = int(input("영어 점수는?"))
# gwa = int(input("과학 점수는?"))
# info = print([f"이름: {name}, 국어: {guk}점, 수학: {su}점, 영어: {yeong}점, 과학: {gwa}점"])
# print(f"{name}의 평균 점수: ", (guk+su+yeong+gwa)/4)

# ["철수", 80, 80, 40, 60] 이름은 문자열, 점수는 int
# 리스트에 name, 국어, 수학, 영어, 과학을 차례대로 입력 받는다. sum()으로 다 더해서 len() 나온 수로 나눠준다

