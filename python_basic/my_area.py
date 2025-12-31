PI = 3.14
def square_area(a):
    print(a ** 2)
    return a ** 2

def circle_area(r):
    print(PI * r ** 2)
    return PI * r ** 2


# def func1():
#  print("func1 in my_module1 ")

# def func2():
#  print("func2 in my_module1 ")
 

if __name__ == "__main__":
    circle_area(8)
else: 
    square_area(5)