# one = "hello"
# two = "python"
# print(one + " " + two)



# def test(a, b):
#     print(a, b)

# test("hello", "python")


# def test(a, b):
#     print(a + " " + b)


# def num(a, b):
#     print(a + b)



# num(10, 10)
# test("하하", "호호")


# warn_investment_list = [12, 12.222, "Microsoft", "Google", "Naver", "Kakao", "SAMSUNG", "LG"]

# investment_input = input("투자 종목 입력:")

# def investment(b):
        
#         if b in warn_investment_list:
#             print("투자 경고 종목입니다.")

#         else:    
#             print("투자 경고 종목이 아닙니다.")


# investment(investment_input)



class Calculator():
    def __init__(self):
        self.result = 0

    # 클래스를 만들때 쓰인 함수는 메서드
    def add(self, num):
        self.result += num
        return self.result

# 클래스를 객체 cal1에 할당
cal1 = Calculator()


# 클래스를 객체 cal2에 할당
cal2 = Calculator()


print(cal1.add(3))
print(cal1.add(4))

print(cal2.add(9))
print(cal2.add(2))



