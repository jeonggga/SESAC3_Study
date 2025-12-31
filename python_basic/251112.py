print((lambda x : x**2)(3))

def test(x):
    o = x**2
    return o
print(test(3))


mySquare = lambda x : x**2
print(mySquare(7))


mySimpleFunc = lambda x, y, z : 2*x +3*y + z
print(mySimpleFunc(1, 2, 3))



print([int(0.123), int(5.678), int(-1.912)])


print([int('1234'), int('4567'), int('-3975')])


print([float(0), float(123), float(-4566)])


print([float('10'), float('0.4567'), float('-39.75')])


print([str(123), str(456778), str(-987)])


list_data = ['abc', 1, 2, 'def']
tuple_data = ('abc', 1, 2, 'def')
set_data = {'abc', 1, 2, 'def'}

print(type(list_data), type(tuple_data), type(set_data))

print("리스트로 변환: ", list(tuple_data), list(set_data))

print("튜플로 변환: ", tuple(list_data), tuple(set_data))

print("세트로 변환: ", set(list_data), set(tuple_data))




print(bool(0))
print(bool(-10))
print(bool(5.12))
print(bool(-3.26))


print(bool('a'))
print(bool(""))
print(bool(" "))
print(bool(None))


# 빈 리스트
myFriends = []
print(bool(myFriends))


myFriends = ['James', 'Robert', 'Lisa', 'Mary']
print(bool(myFriends))


# 빈 튜플
myNum = ()
print(bool(myNum))


myNum = (1, 2, 3)
print(bool(myNum))


# 빈 세트
mySetA = {}
print(bool(mySetA))


mySetA = {10, 20, 30}
print(bool(mySetA))



def print_name(name):
    if bool(name):
        print("입력된 이름:", name)
    else:
        print("입력된 이름이 없습니다.")


print_name("Mina")
print_name("")




myNum = [10, 5, 12, 0, 3.5, 99.5, 42]
print([min(myNum), max(myNum)])



myStr = 'zxyabc'
print([min(myStr), max(myStr)])


myNum = (10, 5, 12, 0, 3.5, 99.5, 42)
print(min(myNum), max(myNum))


sumList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(sum(sumList))



print(len("ab cd"))

print(len([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

print(len({1:"Tomas", 2:"Edward", 3:"Henry"}))



scores = [90, 80, 95, 85]

# score_sum = 0
# subject_num = 0
# for score in scores:
#     score_sum = score_sum + score
#     subject_num = subject_num + 1

# average = score_sum / subject_num

# print("=====총점:{0}, 평균:{1}".format(score_sum, average))




scores = [90, 80, 95, 85]
print("-----총점:{0}, 평균:{1}".format(sum(scores), sum(scores)/len(scores)))



test_list =[]  #빈 리스트
score_input = int(input("점수 입력"))  #점수를 입력 받는다
test_list.append(score_input)  #입력된 점수를 test_list에 보낸다
print(test_list)


# 학생 이름이 있으면 4과목 점수를 각각 입력 받는다.
# 반복문을 통해서 점수를 입력 받는다.

# 받아야 하는 과목은 정해져 있으니 리스트를 먼저 생성함
# 그러나 리스트 안에 값이 변경되거나 수정되면 안되니까 튜플을 이용한다
score_list = ("국어", "수학", "영어", "과학")


test_list = []
total_score = 0

for score_num in range(len(score_list)) :
    score_input = int(input("점수 입력"))  #점수를 입력 받는다
    test_list.append(score_input)
    total_score = total_score + score_input

print(f"{score_list[0]}, {test_list[0]}, {score_list[1]}, {test_list[1]}, {score_list[2]}, {test_list[2]}, {score_list[3]}, {test_list[3]}")
print("총점: ", total_score, "평균: ", total_score/len(score_list))
    




'''
학생 5명에 학생 이름, 국어, 수학, 영어, 과학 점수를 입력 받아
1번과 2번을 출력하는 프로그램을 작성하시오.
1. 학생에 따른 점수의 평균 및 전체 평균
2. 각 과목별 최대 점수, 최소 점수
'''



student_dict = {}

subject_list = ("국어", "수학", "영어", "과학")


tmp_data = {"이성희": [95,90,85,70], "개발자": [90,94,80,92], "정가연": [98,98,98,98]}


def find_max_score(subject_list, student_dict):
    max_scores = {} # 각 과목별 최대 점수를 기록하기 위한 빈 딕셔너리
    all_scores = student_dict.values()  # tmp_data가 매개변수 student_dict로 값을 받아서 값만 빼오는 메서드를 써서 변수에 할당함


    for subject_idx in range(len(subject_list)):
        subject = subject_list[subject_idx]
        current_subject_scores = []

        for scores in all_scores:
            print(f"scores >>> {scores}")
            score_for_subject = scores[subject_idx]
            print(score_for_subject)

            current_subject_scores.append(score_for_subject)

        print("---------------------------")
        max_score = max(current_subject_scores)
        max_scores[subject] = max_score

    print(max_scores)


find_max_score(subject_list, tmp_data)



















# coffee_menu_str = "에스프레소   ,,,    ,,,   .아메리.카노.     카페라테.카푸치노"
# print(coffee_menu_str.split(",,,", maxsplit=10))



# coffee_menu_str = "에스프레소 아메리카노 카페라테 카푸치노"
# print(coffee_menu_str.split(" "))


# coffee_menu_str = "          &&&&                     에스프레소 아메리카노 카페라테 카푸치노"
# print(coffee_menu_str.split(maxsplit=8))


# coffee_menu_str = "      에스프레소  아메리카노 \n\n 카페라테 \n\n 카푸치노"
# print(coffee_menu_str.split(maxsplit=0))


# print("에스프레소ㅔㅔ === === === == 아메리카노 ㅔ    ==     카페라테 카푸치노".split("==", maxsplit=10))


 


# test = "나는 천재다.\t짱이다\n그렇다"

# print(test.split())

# phone_number = "+82-01-2345-6789"
# split_num = phone_number.split("-", 1)

# print(split_num)
# print("국내전화번호: {0}".format(split_num[1]))




# text = "apple,banana,grape"
# words = text.split(",")
# print(words)
# joined = " | ".join(words)
# print(type(joined))



# str = "aaaaabbbbbPythonbbbbaaaaaa"
# print(str.strip('ab'))


# text = "www.example.com"
# print(text.strip("w.moce"))




# user_input = input("이름 입력: ").strip()
# a = user_input.split(" ")
# b = "".join(a)



# user_input = " 홍 길 동 ".strip().split(" ")
# # print(user_input[0] + user_input[1] + user_input[2])
# print("".join(user_input))


# print(f"안녕하세요, {user_input}님!")

