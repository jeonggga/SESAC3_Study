
# temp 이름의 비어있는 딕셔너리를 만들라.
temp = {}


# 다음 아이스크림 이름과 희망 가격을 딕셔너리로 구성하라.
icecream = {'메로나':1000, '폴라포':1200, '빵빠레':1800}



# 딕셔너리에 아래 아이스크림 가격정보를 추가하라.
icecream['죠스바'] = 1200
icecream['월드콘'] = 1500



# 딕셔너리를 사용하여 메로나 가격을 출력하라.
print('메로나 가격:', icecream['메로나'])



# 딕셔너리에서 메로나의 가격을 1300으로 수정하라.
icecream['메로나'] = 1300



print('메로나 가격:', icecream['메로나'])
print(icecream)



# 딕셔너리에 메로나를 삭제하라.
del icecream['메로나']

print(icecream)




# 가격과 재고 이름의 두 딕셔너리를 정의하라.
# 가격 딕셔너리는 이름(key)과 가격(value)을 저장하고 재고 딕셔너리는 이름(key)과 재고(value)를 저장한다.
price = {'메로나':1000,'폴라포':1200}
icecream = {'메로나':3, '폴라포':10}




# 딕셔너리를 사용해서 다음과 같이 출력하라.
# 메로나 1000원 재고 10개
# 폴라포 1200원 재고 3개
print('메로나', price['메로나'], '원 재고', icecream['메로나'], '개')
print('폴라포', price['폴라포'], '원 재고', icecream['폴라포'], '개')


# 딕셔너리에서 폴라포의 재고를 2로 변경하라.
icecream['폴라포'] = 2


print('메로나', price['메로나'], '원 재고', icecream['메로나'], '개')
print('폴라포', price['폴라포'], '원 재고', icecream['폴라포'], '개')


# 아이스크림 이름을 키값으로, (가격, 재고) 리스트를 딕셔너리의 값으로 저장하라.
# 딕셔너리의 이름은 inventory로 한다.
inventory = {'메로나':[300, 20],'비비빅':[400, 3], '죠스바':[250, 100]}


# inventory 딕셔너리에서 메로나의 가격을 화면에 출력하라.
print(inventory['메로나'][0], '원')


# inventory 딕셔너리에서 메로나의 재고를 화면에 출력하라.
print(inventory['메로나'][1], '개')

print(inventory)



# inventory 딕셔너리에 아래 데이터를 추가하라.
inventory['월드콘'] = [500, 7]

print(inventory)



# 다음의 딕셔너리에서 key 값으로만 구성된 리스트를 생성하라.
icecream = {'탱크보이': 1200, '폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}

print(list(icecream.keys()))



# 다음의 딕셔너리에서 values 값으로만 구성된 리스트를 생성하라.
print(list(icecream.values()))



# icecream 딕셔너리에서 아이스크림 판매 금액의 총합을 출력하라.
icecream = {'탱크보이': 1200, '폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}

a = list(icecream.values())
total = 0

for i in a:
    total = i + total
    
print(total)

print(sum(icecream.values()))




# 아래의 new_product 딕셔너리를 087번의 icecream 딕셔너리에 추가하라.
new_product = {'팥빙수':2700, '아맛나':1000}

print(icecream)


icecream['팥빙수'] = 2700
icecream['아맛나'] = 1000

icecream.update(new_product)

print(icecream)






## 아래 두 개의 튜플을 하나의 딕셔너리로 변환하라.
# keys를 키로, vals를 값으로 result 이름의 딕셔너리로 저장한다.

keys = ("apple", "pear", "peach")
vals = (300, 250, 400)



# 방법1 - 유지보수가 쉬운 방법
result = {}

result[keys[0]] = vals[0]
result[keys[1]] = vals[1]
result[keys[2]] = vals[2]

print(result)



# 방법2 - 알면 제일 쉬운 방법
result = dict(zip(keys, vals))

print(result)



# 방법3 - 딕셔너리 쌍 추가하기
result = {}

result['apple'] = 300
result['pear'] = 250
result['peach'] = 400

print(result)








## date와 close_price 두 개의 리스트를 close_table 이름의 딕셔너리로 생성하라.

date = ['09/05', '09/06', '09/07', '09/08', '09/09']
close_price = [10500, 10300, 10100, 10800, 11000]

close_table = dict(zip(date, close_price))
print(close_table)



