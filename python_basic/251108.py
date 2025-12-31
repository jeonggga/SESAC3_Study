'''
1부터 10숫자의 홀의 합, 짝의 합 구하기

정답 예시
홀의 합은 00입니다.
짝의 합은 00입니다.
'''

#1부터 while문으로 10까지 돌린다.

num = 1 #2. 1부터 돌리기 때문에 1을 할당함
even_num = 0  #3. while문을 돌리면서 짝수 합을 할당하기 위해 초깃값을 설정함. 해당 위치에 변수를 설정한 이유는 while문의 영향을 받지 않는 위치이기 때문
odd_num = 0  #3. while문을 돌리면서 홀수 합을 할당하기 위해 초깃값을 설정함. 해당 위치에 변수를 설정한 이유는 while문의 영향을 받지 않는 위치이기 때문

while num <= 10: #1.num이 10보다 작거나 같으면 참이다. 아래 조건문 실행 ( == 아래의 코드블록을 진행한다.)

    if num % 2 == 0:  #num과 2를 나눠서 나머지가 0과 같다면 아래 조건문 실행
        even_num = num + even_num #even_num 변수 값과 num 변수 값을 더해서 even_num 변수에 할당
    else : #위 조건에 해당되지 않는다면 아래 조건 실행
        odd_num = num + odd_num #odd_num 변수 값과 num 변수 값을 더해서 odd_num 변수에 할당
    num += 1 #한번 반복할 때마다 +1 증가해서 10이 넘으면 반복문 끝

print(even_num, '짝수 합')
print(odd_num, '홀수 합')






# 구구단

# 2단부터 9단까지 반복문으로 돌리며 9까지 오면 멈춘다.
# 첫번째 while문 전역변수로 바로 윗줄에 위치함.
# 초깃값은 1부터 시작해서 한번 반복될 때마다 +1 증가됨.

# title_num = 1

# while title_num < 9:
#     title_num += 1   # title_num = title_num + 1
#     print( '===', title_num, '단', '===')


#     # 1부터 9까지 곱해야하기 때문에 이중 while 쓰고 9까지 오면 다음 반복문으로 넘어간다.

#     # 두번째 while문의 변수는 지역변수로 두번째 while문 위에 있어야 함.
#     # 변수의 초깃값은 0부터 시작해서 한번 반복될 때마다 +1 증가됨.

#     sub_num = 0

#     while sub_num < 9:
#         sub_num += 1
#         print(title_num, 'x', sub_num, '=', title_num*sub_num)

#         if sub_num == 9:
#             print(title_num, '단 종료')

    
# print('=== 종료 ===')


# 변수 k의 초깃값은 0이다. 조건 비교하기 위함.
k = 0

# while문 무조건 참이다.
while True:
    k += 1 # 변수에 +1씩 더함.
    if k > 3: # 만약에 k가 3보다 크다면
        # print(k, '나오기 전')
        break # 반복문에서 빠져나와라 

    print(k) # 현재 k의 값 출력.




coffee = 10
money = 300
while money:
    print("돈을 받았으니 커피를 줍니다.")
    coffee = coffee -1
    print(f"남은 커피의 양은 {coffee}개입니다.")
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
        break



# 숫자 10까지를 출력하는데 5만 한글로 다섯으로 출력해야한다.




'''
코끼리 코 20바퀴 돈다.
중간에 10번째에서 구구단 외우고
나머지 돌아야 한다.
'''

# 20까지 찍어야하기 때문에 while문을 쓰고,
# 2단부터 9단까지 외워야하니 범위가 정해질 때 쓰는 for문을 쓰는데,
# 1부터 9까지 곱셈해야 하니까 이중for문 씀

# 조건 비교를 위해 먼저 cir 변수를 만들고 한번 반복할 때마다 +1 증가해서 20까지 만든다.
# 안에 있는 반복문의 조건 비교를 위해 num 변수를 만들어서 사용한다.


cir = 0
num = 1

while cir < 20:
    cir += 1
    print(cir)

    if cir == 10:  # 10번째에서 구구단을 외워야하기 때문에 조건 추가
        for num in range(2, 10):
            print('===',num, '단', '===')
            for sub in range(1, 10):
                print(num, 'x', sub, '=', num*sub)
            print(num, '단, 끝')
        print('구구단 끝, 나머지 돌자')




'''
코끼리 코 20바퀴 돈다.
중간에 10번째에서 구구단 외우고
지쳐서 15번째에서 쓰러졌다.
'''


# while 문을 사용해서 20바퀴 돌린다.
# 10번째에서 2단 부터 9단까지 돌릴 if문을 추가한다.
# 그리고 나머지를 돌다가 숫자 15되면 break 사용해서 while문을 나온다.

cir = 0

while cir < 20:
    cir += 1
    print(cir)

    if cir == 10:
        
        for num in range(2, 10):
            print('===', num, '단===')
            
            for sub in range(1, 10):
                print(num, 'x', sub, '=', num*sub)
        print(num, '단 끝')
    if cir == 15:
        print(cir, '넘어졌다!')
        break





    '''1부터 10까지의 숫자 중에서 홀수만 출력하는 것을
while 문을 사용해서 작성한다고 생각해 보자.'''


num = 0  # 조건 비교를 위해 변수를 생성한다.

# 10이 될 때까지 반복문이 돌아간다.
while num < 10:
    num += 1
    if num % 2 != 0:
        print(num)
    


a = 0  #10보다 작아야 하는 조건을 비교하기 위해 변수 생성함
while a < 10:
    a += 1 #반복할 때마다 변수 값이 +1 증가함
    if a % 2 == 0:  #a가 2로 나눠서 나머지가 0이라면 참이라서 if문 발동
        continue  # while문 처음으로 이동헤서 print(a)는 실행되지 않음.
    print(a)  #if문이 거짓일때마다 출력됨.





coffee = 10 # 2. 현재 커피 갯수 / 총개수를 카운팅하여 나중에 if 문에서 비교하기 위함

# 1. 조건은 항상 참이다. 그래서 계속 반복된다.
while True:
    money = int(input("돈을 넣어 주세요: "))  # 3. 입력 받은 숫자를 정수로 변환하고 money 변수에 할당함
    if money == 300:  # 4. money가 300과 같다면 아래 조건문 실행
        print("커피를 줍니다.")
        coffee = coffee -1  # coffee 변수 값에 -1 감소 후 coffee 변수에 할당함
    elif money > 300:  # 5. money가 300보다 작다면 아래 조건문 실행
        print(f"거스름돈 {money -300}를 주고 커피를 줍니다.")  # 6. 현재 money값에서 받아야되는 돈 300 배고 출력함
        coffee = coffee -1  # coffee 변수 값에 -1 감소 후 coffee 변수에 할당함
    else:  # 7. 위 조건문에 해당하지 않는다면 아래 조건문 실행
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")
        print(f"남은 커피의 양은 {coffee}개 입니다.")  # 현재 coffee 변수 값을 출력함
    if coffee == 0:  # 8. 만약 coffee 변수 값이 0과 같다면 아래 조건문 실행
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
        # 9. while 문은 True이기 때문에 계속 반복되는데 coffee 변수 값이 0이라서 커피 판매할 수 없다.
        # 10. 그래서 while문을 나와야해서 break 씀
        break







    # coffee 변수가 갯수를 알 수 없는 리스트인 경우
coffee = ["블랙커피","아아","카페라테","말차라테","자허블","피지오"]
# 2. 리스트가 가진 요소의 갯수를 알기위해 len() 사용
coffee_total_count = len(coffee)


# 1. 반복 조건은 항상 참이다.
while True:
    money = int(input("돈을 넣어 주세요: ")) #3. 입력 받은 문자열을 숫자로 변환하고 money에 할당
    if money == 300:  #4. 머니가 300과 같다면
        print("커피를 줍니다.")  #해당 문구 출력함
        coffee_total_count -= 1  #커피 변수의 요소값을 할당한 coffee_total_count 변수에 -1 감소
    elif money > 300:  #머니가 300보다 크다면
        print(f"거스름돈 {money -300}를 주고 커피를 줍니다.")  #현재 머니의 값에서 -300 감소한다
        coffee_total_count -= 1  #커피 변수의 요소값을 할당한 coffee_total_count 변수에 -1 감소
    else:  
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")   #위 조건문에 다 해당되지 않으면 해당 문구 출력함
        print(f"남은 커피의 양은 {coffee_total_count}개 입니다.")  #남은 커피 양은 coffee_total_count의 값과 일치함.
    if coffee_total_count == 0:  
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.") #만약 coffee_total_count 값이 0과 같다면 해당 문구 출력함.
        break  #항상 참인 반복문(while문)을 벗어나기 위해 break 활용
    
