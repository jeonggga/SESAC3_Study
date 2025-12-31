
'''
[문제]
학생 5명에 학생 이름, 국어, 수학, 영어, 과학 점수를 입력 받아
1번과 2번을 출력하는 프로그램을 작성하시오.

1. 학생에 따른 점수의 평균 및 전체 평균
2. 각 과목별 최대 점수, 최소 점수
'''


# 학생 5명이니까 범위가 정해져 있기 때문에 for문으로 각 과목씩 점수 값을 받는다
subject = ["국어", "수학", "영어", "과학"]   # 입력 받아야 되는 점수의 과목을 리스트에 담는다

num = 0   # 학생 번호 부여하기 위한 변수
student = []   # 학생 이름을 추가할 리스트
all_total_score = 0   # 전체 점수 더한 값을 받을 변수
score_group = []   # 학생들의 각 과목 점수를 받는 리스트


for index in range(1, 6):
    name = input(f"*{num+1}번 학생 이름: ")
    student.append(name)

    score = []   # 아래 for문에서 입력 받은 점수 값을 여기에 추가한다
    total_score = 0  # 아래 for문에서 학생의 총 점수를 할당할 변수 초기값


    for index in range(len(subject)):  # 과목 수에 맞춰서 반복한다
        score_input = int(input(f"{subject[index]} 점수 입력: "))  # 한 학생의 과목별 점수를 입력 받는다
        score.append(score_input)   # 입력 받은 점수 값을 score 리스트에 추가한다
        total_score = total_score + score_input   # 한 학생의 총 점수
        
    print("*평균 점수: ", total_score/len(subject))  # 한 학생의 평균 점수



    print("------------------------------------")



    all_total_score = all_total_score + total_score  # 한번 돌 때마다 한 학생의 점수가 더해짐
    num += 1  # 학생 번호 부여를 위함
    score_group.append(list(score))  # 과목별 최대점수, 최소점수를 구하기 위함


print("******* 전체 평균 점수", all_total_score/(len(student)*len(subject)), "*******")  # 학생 전체 평균 점수



print("------------------------------------")



## 학생들의 각 과목별 최대점수
for index1 in range(len(subject)): # 과목 수에 맞춰서 반복문을 돌린다
    max_list = []
    for index2 in range(len(score_group)):  #리스트안에 리스트 수 만큼 돌린다
        max_list.append(score_group[index2][index1])
    
    print(f"{subject[index1]}의 최대 점수는", max(max_list),"점")



print("====================================")
    


## 학생들의 각 과목별 최소점수
for index1 in range(len(subject)): # 과목 수에 맞춰서 반복문을 돌린다
    min_list = []
    for index2 in range(len(score_group)):  #리스트안에 리스트 수 만큼 돌린다
        min_list.append(score_group[index2][index1])
    
    print(f"{subject[index1]}의 최소 점수는", min(min_list),"점")



print(score_group, "****여기 테스트****")







'''
학생 5명에 학생 이름, 국어, 수학, 영어, 과학 점수를 입력 받아
파일(Student_Score.txt)를 저장하시오.
'''


with open("Student_Score.txt", 'w', encoding="utf-8") as f:
    txt_st_name = []
    txt_score_list = []
    txt_subject = []
    num = 0
    for txt_st_index1 in range(len(student)):
        txt_st_name.append(student[txt_st_index1])

        for txt_st_index2 in range(len(score_group)):
            print(score_group, "****여기 테스트****")
            txt_score_list.append(score_group[txt_st_index2])
            print(subject[num], "점수는?*******")
            num += 1 

print(txt_subject)
print(txt_st_name)
print(txt_score_list)
# f.write()