recipe = {
    "americano":[("bean", 25), ("water", 8)],
    "cafelatte":[("bean", 25), ("milk", 8)],
    "matchalatte":[("matcha", 25), ("milk", 8)],
    "chocolatte":[("choco", 25), ("milk", 8)]
}   # 음료별 레시피 딕셔너리


ingredient = {
    "bean": 100,
    "matcha": 100,
    "choco": 100,
    "water": 100,
    "milk": 100
}   # 재료별 초기 재고 딕셔너리


price = {
    "americano": 2000,
    "cafelatte": 3500,
    "matchalatte": 4500,
    "chocolatte": 4500
}   # 음료별 가격 딕셔너리


question = """
1. americano
2. cafelatte
3. matchalatte
4. chocolatte
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
        key = "chocolatte"
        
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")
        
        # 지정된 숫자 외 입력 시 이 함수를 다시 호출하여 input 실행 (재귀 함수 사용 / 조건문에서는 continue를 사용할 수 없어서)
        return select_drink()
    
    return key  # 선택한 음료 문자열, 정수 반환


# def pay():




# 음료 제조 함수
def making_drink(key):
    print(f"{key} 제조가 완료되었습니다.")
    


# 선택한 음료 메뉴의 현재 재고를 확인하는 함수
def ingredient_check(key):
    # recipe 딕셔너리의 (메뉴명, 재료목록) 쌍을 하나씩 꺼냄 / items()는 딕셔너리의 키와 값을 쌍으로 꺼내는 메소드
    for key_menu, value_tuple in recipe.items():
        # 인자값으로 받은 key(선택한 음료)가 recipe 딕셔너리의 key(key_menu)와 일치하면
        if key == key_menu:
            # recipe 딕셔너리의 값인 value_tuple(튜플)안에 있는 2개 요소를 반복문을 통해 가져옴
            # ingr_name: 재료 이름, required_amount: 필요한 용량
            for ingr_name, required_amount in value_tuple:
                # 현재 재고가 필요한 용량(required_amount)보다 크거나 같다면
                if ingredient[ingr_name] >= required_amount:
                    # 현재 재고에서 필요한 용량을 빼서 현재 재고에 저장함
                    ingredient[ingr_name] -= required_amount
                    
                    print("사용한 용량: ", ingr_name, required_amount)
                    print("남은 용량: ", ingr_name, ingredient[ingr_name])
                
                else:   # 크거나 같지 않다면, 재고가 부족한 경우
                    print("현재 원두 재고: ", ingredient["bean"])
                    print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
                    return select_drink()   # 음료 선택 함수로 돌아감
                    
            return making_drink(key)    # 모든 재료가 충분하다면 음료 제조 함수 호출
        
    # ingredient_check 함수 실행 (select_drink 함수의 결과를 인자로 사용)



def coffee_machine():
    ingredient_check(select_drink())
    
coffee_machine()