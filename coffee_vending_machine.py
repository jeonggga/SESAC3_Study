recipe = {
    "americano":[("bean", 25), ("water", 8)],
    "cafelatte":[("bean", 25), ("milk", 8)],
    "matchalatte":[("matcha", 25), ("milk", 8)],
    "espresso":[("bean", 30)]
}   # 음료별 레시피 딕셔너리


ingredient = {
    "bean": 50,
    "matcha": 50,
    "water": 100,
    "milk": 100
}   # 재료별 초기 재고 딕셔너리


price = {
    "americano": 2000,
    "cafelatte": 3500,
    "matchalatte": 4500,
    "espresso": 2000
}   # 음료별 가격 딕셔너리


question = """
1. americano
2. cafelatte
3. matchalatte
4. espresso
원하시는 음료의 번호를 골라주세요 : """ # 음료 선택 질문 문자열



# 원하는 음료를 선택 받는 함수 코드
def select_drink():
    selected_num = int(input(question))
    key = ""    # 조건문을 거쳐서 나온 값을 담을 빈 문자열 지정
    
    if selected_num == 1:
        key = "americano"   # 입력 받은 숫자의 문자열을 변수 key에 대입한다.
            
    elif selected_num == 2:
        key = "cafelatte"
        
    elif selected_num == 3:
        key = "matchalatte"
        
    elif selected_num == 4:
        key = "espresso"
        
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")
        
        # 지정된 숫자 외 입력 시 이 함수를 다시 호출하여 input 실행 (재귀 함수 사용 / 조건문에서는 continue를 사용할 수 없어서)
        return select_drink()
    
    return key  # 선택한 음료 문자열, 정수 반환




# 음료 제조 함수
def making_drink(key):
    print(f"{key} 제조가 완료되었습니다.")




# 재고 확인 함수
def ingredient_check(key):  # 선택된 음료 메뉴의 현재 재고를 확인함
    # 현재 남은 재고를 여기 함수에서 사용할 변수에 할당함
    used_bean = ingredient["bean"]
    used_water = ingredient["water"]
    used_milk = ingredient["milk"]
    used_matcha = ingredient["matcha"]
    
    if key == "americano":  # 선택한 음료가 아메리카노라면
        # (남은 원두 수가 필요한 원두 수보다 크거나 같다면) 그리고 (남은 뜨거운 물이 필요한 뜨거운 물보다 크거나 같다면)
        if (used_bean >= recipe["americano"][0][1]) and (used_water >= recipe["americano"][1][1]):
            # 남은 원두 수 = 현재 재고 원두 수 - 필요한 원두 수
            used_bean = ingredient["bean"] - recipe["americano"][0][1]
            # 남은 뜨거운 물의 양 = 현재 뜨거운 물의 양 - 필요한 뜨거운 물의 양
            used_water = ingredient["water"] - recipe["americano"][1][1]
            # 남은 원두 수를 현재 재고 원두 수에 할당함
            ingredient["bean"] = used_bean
            # 남은 뜨거운 물의 양을 현재 재고 남은 뜨거운 물의 양에 할당함
            ingredient["water"] = used_water
            
            print("현재 원두 재고: ", ingredient["bean"])
            print("현재 물의 양: ", ingredient["water"])
            
            # 재고 확인하고 재료 소진 후 음료 제작 함수로 이동
            return making_drink(key)
        
        else:
            print("현재 원두 재고: ", ingredient["bean"])
            print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
            
            return select_drink()
        
        
    elif key == "cafelatte":
        if used_bean >= recipe["cafelatte"][0][1] and used_milk >= recipe["cafelatte"][1][1]:
            used_bean = ingredient["bean"] - recipe["cafelatte"][0][1]
            used_milk = ingredient["milk"] - recipe["cafelatte"][1][1]
            ingredient["bean"] = used_bean
            ingredient["milk"] = used_milk
            print(ingredient["bean"])
            print(ingredient["milk"])
            return making_drink(key)
        
        else:
            print("현재 원두 재고: ", ingredient["bean"])
            print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
            return select_drink()
        
        
    elif key == "matchalatte":
        if used_bean >= recipe["matchalatte"][0][1] and used_milk >= recipe["matchalatte"][1][1]:
            used_matcha = ingredient["matcha"] - recipe["matchalatte"][0][1]
            used_milk = ingredient["milk"] - recipe["matchalatte"][1][1]
            ingredient["matcha"] = used_matcha
            ingredient["milk"] = used_milk
            print(ingredient["matcha"])
            print(ingredient["milk"])
            return making_drink(key)
        
        else:
            print("현재 말차 재고: ", ingredient["matcha"])
            print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
            return select_drink()
        
        
    elif key == "espresso":
        if used_bean >= recipe["espresso"][0][1]:
            used_bean = ingredient["bean"] - recipe["espresso"][0][1]
            ingredient["bean"] = used_bean
            print(ingredient["bean"])
            return making_drink(key)
        
        else:
            print("현재 원두 재고: ", ingredient["bean"])
            print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
            return select_drink()
        
        
ingredient_check(select_drink())



