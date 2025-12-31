import my_area

print('pi =', my_area.PI)
print('square area =', my_area.square_area(5))
print('circle area =', my_area.circle_area(2))


dir(my_area)






from my_area import PI

print('pi =', PI)


from my_area import square_area
from my_area import circle_area

print('square area =', square_area(5))
print('circle area =', circle_area(2))






from my_area import PI, square_area, circle_area


print('pi =', PI)
print('square area =', square_area(5))
print('circle area =', circle_area(2))






from my_area import *

print('pi =', PI)
print('square area =', square_area(5))
print('circle area =', circle_area(2))






from my_area import *
from my_area2 import *

# func1()
func2()
func3()






import my_area as area # 모듈명(my_area)에 별명(area)을 붙임

print('pi =', area.PI) # 모듈명 대신 별명 이용
print('square area =', area.square_area(5))
print('circle area =', area.circle_area(2))








from my_area import PI as pi
from my_area import square_area as square
from my_area import circle_area as circle

# print('pi =', pi) # 모듈 변수의 별명 이용
# print('square area =', square(5)) # 모듈 함수의 별명 이용
# print('circle area =', circle(2))