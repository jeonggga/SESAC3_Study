name = "광재"
name2 = "영희"
print("%s는 나의 친구이고, %s도 나의 친구입니다." % (name2, name))

r = 3
PI = 3.14159265358979
print("반지름: %d, 원주율: %f" %(r, PI))





input('끝말 잇기 시작할 단어')
data_0 = "가방"
data_1 = "방석"
data_2 = "석상"
data_3 = "상처"

print("끝말 잇기 : {0}, {2}, {1}, {3}".format(data_0, data_1, data_2, data_3))






name = "Tomas"
age = 10
a = 0.123456789123456789
fmt_string = "String: {0}. Integer Number: {1}. Floating Number: {2}"
print(fmt_string. format(name, age, a))




a = 0.123456789123456789
b = 0.777777777777777777777777777777777777777777
print("{1:.1000f}, {0:.1000000000f}". format(a, b))



student = ["Ailce", 15, "잠자기", 30]
print(f"이름: {student[0]}, 나이: {student[1]}, 하고 싶은 것은 잠을 {student[3]}시간 자는 것입니다.")
print("이름: {}, 나이: {}, {} 시간 최대로 {}시간 자는 것입니다.".format(*student))






# 파일 쓰기
f = open('myFile.txt', 'w') #파일명과 모드
f.write('This is my first file.\n') #파일안에 텍스트 작성
f.write('또 쓰여지는가')
f.close() #닫는다



# 파일 읽기
f = open('myFile.txt', 'r')
data = f.read()
f.close()
# print(f.closed) # 제대로 닫았는지 안닫았는지 True, False로 알려줌

print(data)




