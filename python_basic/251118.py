import random

print(random.random())


dice1 = random.randint(1, 6)
dice2 = random.randint(1, 6)
print('주사위 두 개의 숫자: {0}, {1}'.format(dice1, dice2))



num1 = random.randrange(1, 10, 2)
num2 = random.randrange(0, 100, 10)
print('num1: {0}, num2: {1}'.format(num1, num2))




menu = ['비빔밥', '된장찌개', '볶음밥', '불고기', '스파게티', '피자', '탕수육']
print(random.choice(menu))




print(random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))





'''----------------------------------------------------------------------'''





import datetime


date_obj = datetime.date(2025, 4, 20)
time_obj = datetime.time(18, 30, 55)
datetime_obj = datetime.datetime(2022, 5, 30, 14, 25, 35)

print(date_obj)
print(time_obj)
print(datetime_obj)




date_var = datetime.date.date_classmethod()
time_var = datetime.time.time.classmethod()
datetime_var = datetime.datetime.datetime_classmethod()



print(date_var)
print(time_var)
print(datetime_var)




set_day = datetime.date(2021, 8, 2)
print(set_day)


print('{0} - {1} - {2}'.format(set_day.year, set_day.month, set_day.day))



day1 = datetime.date(2021, 3, 1)
day2 = datetime.date(2021, 7, 10)
diff_day = day2 - day1
print(diff_day)


print(type(day1))
print(type(diff_day))


print("** 지정된 두 날짜의 차이는 {}일입니다. **".format(diff_day.days))




print(datetime.date.today())


today = datetime.date.today()
special_day = datetime.date(2025, 12, 24)
print(special_day - today)



set_time = datetime.time(15, 30, 45)
print(set_time)

print('{0}:{1}:{2}'.format(set_time.hour,set_time.minute,set_time.second))




set_dt = datetime.datetime(2021, 8, 2, 10, 20, 0)
print(set_dt)



print('날짜 {0}/{1}/{2}'.format(set_dt.year, set_dt.month, set_dt.day))
print('시각 {0}:{1}:{2}'.format(set_dt.hour, set_dt.minute, set_dt.second))




now = datetime.datetime.now()
print(now)


print("Date & Time: {:%Y-%m-%d, %H:%M:%S}".format(now))
print("Time: {:%H%M%S}".format(now))




now = datetime.datetime.now()
set_dt = datetime.datetime(2020, 12, 1, 12, 30, 45)

print("현재 날짜 및 시각:", now)
print("차이:", set_dt - now)





'''-------------------------------------------------------'''





import calendar




print(calendar.calendar(2025))



print(calendar.calendar(2025, m=4))



print(calendar.month(2025,11))


print(calendar.monthrange(2025,2))



print(calendar.firstweekday())

calendar.setfirstweekday(calendar.SUNDAY)
print(calendar.month(2025,11))


print(calendar.weekday(2025, 11, 17))


print(calendar.isleap(2024))
print(calendar.isleap(2025))






'''-------------------------------------------'''




def ten_div(x):
    return 10 / x

print(ten_div(2))


print(ten_div(0))



try:
    x = int(input('나눌 숫자를 입력하세요: '))
    y = 10 / x
    print(y)

except:
    print("예외가 발생했습니다.")




y = [10, 20, 30]

try:
    index, x = map(int, input('인덱스와 나눌 숫자를 입력하세요: ').split())
    print(y[index] / x)

except ZeroDivisionError:
    print('숫자를 0으로 나눌 수 없습니다.')

except IndexError:
    print('잘못된 인덱스입니다.')





y = [10, 20, 30]

try:
    index, x = map(int, input('인덱스와 나눌 숫자를 입력하세요: ').split())
    print(y[index] / x)

except ZeroDivisionError as e:
    print('숫자를 0으로 나눌 수 없습니다.', e)

except IndexError as ee:
    print('잘못된 인덱스입니다.', ee)
    
    
except Exception as e:
    print('예외가 발생했습니다.')    

    
    
    
'''-------------------------------------------------'''
    
    
    
    
    
import numpy as np


data1 = [0, 1, 2, 3, 4, 5]
a1 = np.array(data1)
print(a1)


data2 = [0.1, 5, 4, 12, 0.5]
a2 = np.array(data2)
print(a2)

print(a1.dtype)
print(a2.dtype)


print(np.array([0.5, 2, 0.01, 8]))

print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

print(np.arange(0, 10, 2))

print(np.arange(1, 10))

print(np.arange(5))

print(np.arange(12).reshape(4, 3))

b1 = np.arange(12).reshape(4, 3)
print(b1.shape)


b2 = np.arange(5)
print(b2.shape)



print(np.linspace(1, 10, 10))

print(np.linspace(0, np.pi, 20))




# arr_zero_n = np.zeros(n)
# arr_zero_mxn = np.zeros((m,n))
# arr_one_n = np.ones(n)
# arr_one_mxn = np.ones((m/n))