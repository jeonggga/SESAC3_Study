# #클래스 명을 지정
# class Calculator():
#     # 생성자 메서드
#     def __init__(self):
#         # 생성자가 호출될 때마다, 해당 객체(self)의 상태를 저장하는 인스턴스 변수생성 및 초기값 지정
#         self.result = 0
#     # 클래스에서 사용될 메서드 생성 및 매개변수 지정
#     def add(self, num):
#         # 현재 객체의 result값에 입력받은 숫자 num을 더하고 그 결과를 result 변수에 누적
#         self.result += num
#         # 함수 실행 후 결과값 result를 return 함
#         return self.result
#     def sub(self, num):
#         self.result -= num
#         return self.result
#     def mul(self, num):
#         self.result *= num
#         return self.result
#     def div(self, num):
#         self.result /= num
#         return self.result
# # 클래스로부터 인스턴스를 생성하여 객체 cal1에 할당 / 할당이 되는 순간 cal1 내부의 result 값은 0으로 초기화됨
# cal1 = Calculator()
# # 클래스로부터 인스턴스를 생성하여 객체 cal2에 할당 / 할당이 되는 순간 cal2 내부의 result 값은 0으로 초기화됨
# cal2 = Calculator()
# print(f"{cal1.add(3)} \n{cal1.add(4)} \n{cal1.sub(2)} \n{cal1.div(2)} \n{cal1.mul(8)}")
# print("--------------------------------------------------")
# print(f"{cal2.add(8)} \n{cal2.add(2)} \n{cal2.sub(1)} \n{cal2.mul(2)} \n{cal2.div(3)}")







class Bicycle():  #클래스 선언
    pass



my_bicycle = Bicycle()

print(my_bicycle)

my_bicycle.wheel_size = 26
my_bicycle.color = 'black'

print("바퀴 크기:", my_bicycle.wheel_size)  # 객체의 속성 출력
print("색상:", my_bicycle.color)



class Bicycle():
    def move(self, speed):
        print("자전거: 시속 {0}킬로미터로 전진".format(speed))

    def turn(self, direction):
        print("자전거: {0}회전".format(direction))

    def stop(self):
        print("자전거({0}, {1}): 정지".format(self.wheel_size, self.color))
    
my_bicycle = Bicycle()

my_bicycle.wheel_size = 26
my_bicycle.color = "black"

my_bicycle.move(30)
my_bicycle.turn('좌')
my_bicycle.stop()


bicycle1 = Bicycle()

bicycle1.wheel_size = 27
bicycle1.color = 'red'

bicycle1.move(20)
bicycle1.turn('좌')
bicycle1.stop()


bicycle2 = Bicycle()

bicycle2.wheel_size = 24
bicycle2.color = 'blue'

bicycle2.move(15)
bicycle2.turn('우')
bicycle2.stop()



class Bicycle():
    def __init__(self, wheel_size, color):
        self.wheel_size = wheel_size
        self.color = color

    def move(self, speed):
        print("자전거: 시속 {0}킬로미터로 전진".format(speed))

    def turn(self, direction):
        print("자전거: {0}회전".format(direction))

    def stop(self):
        print("자전거({0}, {1}: 정지)".format(self.wheel_size, self.color))
    

my_bicycle = Bicycle(26, 'black') # 객체 생성과 동시에 속성값을 지정.

my_bicycle.move(30) # 객체 메서드 호출
my_bicycle.turn('좌')
my_bicycle.stop()








class Student:
    # ⭐ 클래스 변수 (모든 학생이 함께 쓰는 정보)
    school_name = "해솔초등학교"

    def __init__(self, name, age):
        # ⭐ 인스턴스 변수 (학생마다 다름)
        self.name = name
        self.age = age

    def introduce(self):
        print("안녕하세요! 저는 {0}이고, {1}살입니다.".format(self.name, self.age))
        print("제가 다니는 학교는 {0}입니다!".format(Student.school_name))


# 🎒 학생(객체) 두 명 만들기
student1 = Student("철수", 11)
student2 = Student("영희", 10)

# 각 학생 소개
student1.introduce()
student2.introduce()

# ⭐ 클래스 변수 변경해보기 (학교 이름 바꾸기)
Student.school_name = "푸른초등학교"

print("\n=== 학교 이름을 바꾼 후 ===\n")

# 다시 소개
student1.introduce()
student2.introduce()






#자동차 설계도
class Car():

    # 몇 대의 자동차가 만들어졌는지 세는 숫자 /공유되는 숫자
    instance_count = 0 # 클래스 변수 생성 및 초기화


    def __init__(self, size, color):
        self.size = size # 인스턴스 변수 생성 및 초기화
        self.color = color # 인스턴스 변수 생성 및 초기화

        Car.instance_count = Car.instance_count + 1 # 클래스 변수 이용
        print("자동차 객체의 수: {0}".format(Car.instance_count))

    def move(self):
        print("자동차({0} & {1})가 움직입니다.".format(self.size, self.color))


car1 = Car("small", "white")
car2 = Car("big", "black")

print("Car 클래스의 총 인스턴스 개수:{}".format(Car.instance_count))
print("Car 클래스의 총 인스턴스 개수:{}".format(car1.instance_count))
print("Car 클래스의 총 인스턴스 개수:{}".format(car2.instance_count))

car1.move()
car2.move()



class Car2():
    count = 0

    def __init__(self, size, num):
        self.size = size
        self.count = num
        Car2.count = Car2.count + 1
        print("자동차 객체의 수: Car2.count = {0}".format(Car2.count))
        print("인스턴스 변수 초기화: self.count = {0}".format(self.count))

    def move(self):
        print("자동차 ({0} & {1})가 움직입니다.".format(self.size, self.count))

    
car1 = Car2("big", 20)
car2 = Car2("small", 30)



# Car 클래스 선언
class Car():
    instance_count = 0  # 클래스 변수 생성 및 초기화

    # 초기화 함수(인스턴스 메서드)
    def __init__(self, size, color, speed):
        self.size = size      # 인스턴스 변수 생성 및 초기화
        self.color = color    # 인스턴스 변수 생성 및 초기화
        self.speed = speed    # 인스턴스 변수 생성 및 초기화
        Car.instance_count = Car.instance_count + 1    # 클래스 변수 이용
        # print("자동차 객체의 수: {0}".format(Car.instance_count))

    # 인스턴스 메서드
    def move(self, speed):
        self.speed = speed  # 인스턴스 변수 생성
        print("자동차({0} & {1})가 ".format(self.size, self.color))
        print("시속 {0}킬로미터로 전진".format(self.speed))

    # 인스턴스 메서드
    def auto_cruise(self, speed):
        self.speed = speed
        print("자율 주행 모드")
        self.move(self.speed)   # move() 함수의 인자로 인스턴스 변수를 입력
        


car1 = Car(size=66)  # 객체 생성
car2 = Car(size=55, color="blue")
car3 = Car(size=77, color="bluerdf", speed=300)

# car1.move(150)  # 객체 메서드 호출
# car2.move(200)
car3.move(400)

# # car1.move(22)
# car1.auto_cruise()   # 객체 메서드 호출
# car2.auto_cruise()
car3.auto_cruise(500)








# Car 클래스 선언
class Car():
    instance_count = 0

    def __init__(self, size, color):
        self.size = size
        self.color = color
        Car.instance_count = Car.instance_count + 1

    def move(self, speed):
        self.speed = speed
        print("자동차({0} & {1})가 ".format(self.size, self.color))
        print("시속 {0}킬로미터로 전진".format(self.speed))

    def auto_cruise(self, speed):
        self.speed = speed
        print("자율 주행 모드")
        self.move(self.speed)
        

    # 정적 메서드
    @staticmethod
    def check_type(model_code):
        if(model_code >= 20):
            print("이 자동차는 전기차입니다.")
        elif(10 <= model_code < 20):
            print("이 자동차는 가솔린차입니다.")
        else:
            print("이 자동차는 디젤차입니다.")



    # 클래스 메서드
    @classmethod
    def count_instance(cls):
        print("자동차 객체의 개수: {0}".format(cls.instance_count))
        
        


Car.count_instance()    # 객체 생성 전에 클래스 메서드 호출
        

car1 = Car("small", "red")  # 첫번째 객체 생성
Car.count_instance()    # 클래스 메서드 호출

car2 = Car("big", "green")  #두번째 객체 생성
Car.count_instance()    # 클래스 메서드 호출



# Car.check_type(25)

        
        
        
        

class Robot():
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        
    def move(self):
        self.pos = self.pos + 1
        print("{0} position: {1}".format(self.name, self.pos))
        
        
robot1 = Robot('R1', 0)
robot2 = Robot('R2', 10)

robot1.move()
robot2.move()

# robot3 = Robot('R3', 30)
# robot4 = Robot('R4', 40)

# robot3.move()
# robot4.move()





class Bicycle():
    def __init__(self, wheel_size, color):
        self.wheel_size = wheel_size
        self.color = color
        
        
        # 정적 메소드처럼 보이지만 데코레이션이 없기 때문에 인스턴스 메소드이다.
        # 정적 메소드와 차이점은 메모리에 저장되는 공간이 다름.
        # 문법적으로 인스턴스 메소드를 아래처럼 쓰는 것도 가능.
    def move(self, speed):
        print("자전거: 시속 {0}킬로미터로 전진".format(speed))
        
    def turn(self, direction):
        print("자전거: {0}회전".format(direction))
        
    def stop(self):
        print("자전거({0},{1}): 정지".format(self.wheel_size, self.color))
        

        
class FoldingBicycle(Bicycle):
    def __init__(self, wheel_size, color, state):   # FoldingBicycle 초기화
        Bicycle.__init__(self, wheel_size, color)   # Bicycle 의 초기화 재사용
        # super().__init__(wheel_size, color) 도 사용 가능
        
        self.state = state  # 자식 클래스에서 새로 추가한 변수
        
    def fold(self):
        self.state = 'folding'
        print("자전거: 접기, state = {0}".format(self.state))
        
    def unfold(self):
        self.state = 'unfolding'
        print("자전거: 펴기, state = {0}".format(self.state))
        
        
        
folding_bicycle = FoldingBicycle(27, 'white', 'unfolding')    # 객체 생성
folding_bicycle.move(20)   # 부모 클래스의 메서드 호출
folding_bicycle.fold()     # 자식 클래스에서 정의한 메서드 호출
folding_bicycle.unfold()
