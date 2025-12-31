str_f = "Python code."

print("찾는 문자열의 위치:", str_f.find("Python"))
print("찾는 문자열의 위치:", str_f.find("code."))
print("찾는 문자열의 위치:", str_f.find("n"))
print("찾는 문자열의 위치:", str_f.find("easy"))




str_f_se = "Python is powerful. Python is easy to learn."

print(str_f_se.find("Python", 10, 30))  # 시작 위치와 끝 위치
print(str_f_se.find("Python", 35))  # 찾기 위한 시작 위치 지정




str_c = "Python is powerful. Python is easy to learn. Python is open."

print("Python의 개수는?:", str_c.count("Python"))
print("powerful 개수는?:", str_c.count("powerful"))
print("IPython의 개수는?:", str_c.count("IPython"))




str_se = "Python is powerful. Python is easy to learn."

print("Python으로 시작?:", str_se.startswith("Python"))
print("is로 시작?:", str_se.startswith("is"))
print(".로 끝?:", str_se.endswith("."))
print("learn으로 끝?:", str_se.endswith("learn"))




str_a = "Python is fast. Python is friendly. Python is open."

print(str_a.replace("Python", "IPython"))
print(str_a.replace("Python", "IPython", 2))




str_b = "[Python] [is] [fast]"
str_b1 = str_b.replace('[', '')  # 문자열에서 '['를 제거
str_b2 = str_b1.replace(']', '')  # 결과 문자열에서 다시 ']'를 제거


print(str_b)
print(str_b1)
print(str_b2)




print("Python".isalpha())  # 문자열에 공백, 특수 문자, 숫자가 없음
print("Ver . 3.x".isalpha())  # 공백, 특수 문자, 숫자 중 하나가 있음




print("12345".isdigit())  # 문자열이 모두 숫자로 구성됨
print("12345abc".isdigit())  # 문자열이 숫자로만 구성되지 않음 




print("abc1234".isalnum())  # 특수 문자나 공백이 아닌 문자와 숫자로 구성
print("       abc1234".isalnum())  # 문자열에 공백이 있음



print("     ".isspace())   # 문자열이 공백으로만 구성됨
print(" 1 ".isspace())   # 문자열에 공백 외에 다른 문자가 있음



print("PYTHON".isupper())  # 문자열이 모두 대소문자로 구성됨
print("Python".isupper())  # 문자열에 대문자와 소문자가 있음
print("python".islower())  # 문자열이 모두 소문자로 구성됨
print("Python".islower())  # 문자열이 모두 대소문자로 구성됨



string1 = "Python is powerful. PYTHON IS EASY TO LEARN."
print(string1.lower())
print(string1.upper())



print("Python" == "python")
print("Python" != "python")




print("Python".lower() == "python".lower())
print("Python".upper() == "python".upper())






f = open(file_name, encoding="UTF-8")     # 파일 열기
for line in f:      # 한 줄씩 읽기
    print(line, end='')     # 한 줄씩 출력
f.close()       # 파일 닫기



f = open(file_name, encoding="UTF-8")
header = f.readline()
f.close


print(header)

header_list = header.split()  # 첫 줄의 문자열을 분리 후 리스트로 변환
print(header_list)

file_name = "coffeeShopSales.txt"



f = open(file_name, encoding="UTF-8")     # 파일 열기
header = f.readline()       # 데이터의 첫 번쨰 줄을 읽음
header_list = header.split()        # 첫 줄의 문자열을 분리한 후 리스트로 변환

for line in f:      # 두 번째 줄부터 데이터를 읽어서 반복적으로 처리 
    data_list = line.split()        # 문자열을 분리해서 리스트로 변환
    # print(data_list)    # 결과 확인을 위해 리스트 출력

f.close()





f = open(file_name, encoding="UTF-8")     # 파일 열기
header = f.readline()   # 데이터의 첫 번째 줄을 읽음
headerList = header.split()     # 첫 줄의 문자열을 분리한 후 리스트로 변환

espresso = []       # 커피 종류별로 빈 리스트 생성
americano = []
cafelatte = []
cappucino = []

for line in f:
    dataList = line.split()

    espresso.append(int(dataList[1]))
    americano.append(int(dataList[2]))
    cafelatte.append(int(dataList[3]))
    cappucino.append(int(dataList[4]))
f.close()



print("{0} : {1}".format(headerList[1], espresso))
print("{0} : {1}".format(headerList[2], americano))
print("{0} : {1}".format(headerList[3], cafelatte))
print("{0} : {1}".format(headerList[4], cappucino))