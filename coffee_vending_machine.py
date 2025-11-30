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
원하시는 음료의 번호를 골라주세요 :""" # 음료 선택 질문 문자열



# -------------------------------------------
# 원하는 음료를 선택 받는 함수
# -------------------------------------------
def select_drink():
    selected_num = int(input(question))
    drink = ""    # 조건문 결과로 선택될 음료 이름을 저장할 변수
    
    if selected_num == 1:
        drink = "americano"   # 입력 받은 숫자의 문자열을 변수 key에 대입한다.
    elif selected_num == 2:
        drink = "cafelatte" 
    elif selected_num == 3:
        drink = "matchalatte"
    elif selected_num == 4:
        drink = "chocolatte"
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")
        # 지정된 숫자 외 입력 시 이 함수를 다시 호출하여 input 실행 (재귀 함수 사용 / 조건문에서는 continue를 사용할 수 없어서)
        return select_drink()
    
    return drink  # 선택한 음료 문자열, 정수 반환



# -------------------------------------------
# 결제 처리 함수
# -------------------------------------------
def pay(drink):
    # 선택한 음료와 금액을 안내한 뒤 결제 금액 입력받기
    input_pay = int(input(f"\n선택하신 메뉴: {drink}\n금액: {price[drink]}원\n결제 금액을 넣어주세요: ", ))

    # 충분한 금액이 들어오면 결제 완료
    if input_pay >= price[drink]:
        print("\n>>> 결제가 완료되었습니다.\n>>> 음료 제조를 시작합니다.\n")

        # 간단한 영수증 출력
        print("================== 영수증 ==================")
        print(f"주문 메뉴: {drink} --------------- {price[drink]}원")
        print("--------------------------------------------")
        print("주문해주셔서 감사합니다. 맛있는 하루되세요.\nWIFI 비밀번호는 12345입니다.")
        print("============================================\n")

    # 금액 부족 시 다시 결제 입력 받기
    else:
        print("금액이 부족합니다. 다시 시도해주세요.")
        return pay(drink)
    
    return drink



# -------------------------------------------
# 음료 제조 함수
# -------------------------------------------
def making_drink(drink):
    print(f"\n주문하신 음료: {drink}\n제조가 완료되었습니다.\n맛있게 드세요!")




# -------------------------------------------
# 재주문 여부 확인 함수
# -------------------------------------------
def continue_order():
        input_order = input("\n음료 주문을 이어서 하시겠어요?(Y/N)")

        if input_order == "Y" or input_order == "y":
            return True
        elif input_order == "N" or input_order == "n":
            print("주문이 종료되었습니다.")
            return False
        else:
            print("잘못된 입력입니다. 다시 입력해주세요.")
            return continue_order()     # 잘못 입력하면 다시 질문




# -------------------------------------------
# 음료 재료 재고 확인 함수
# 선택한 음료의 레시피를 확인하고 재고를 차감
# -------------------------------------------
def ingredient_check(drink):
    # recipe 딕셔너리의 (메뉴명, 재료목록) 쌍을 하나씩 꺼냄 / items()는 딕셔너리의 키와 값을 쌍으로 꺼내는 메소드
    # recipe: {음료명: [(재료명, 필요용량), ...]}
    for drink_menu, value_tuple in recipe.items():

        # 인자값으로 받은 drink(선택한 음료)가 recipe 딕셔너리의 drink(drink_menu)와 일치하면
        if drink == drink_menu:

            # recipe 딕셔너리의 값인 value_tuple(튜플)안에 있는 2개 요소를 반복문을 통해 가져옴
            # ingr_name: 재료 이름, required_amount: 필요한 용량
            for ingr_name, required_amount in value_tuple:

                # 현재 재고가 필요한 용량(required_amount)보다 크거나 같다면
                if ingredient[ingr_name] >= required_amount:

                    # 현재 재고에서 필요한 용량을 차감 후 현재 재고에 저장함
                    ingredient[ingr_name] -= required_amount
                    
                    print("- 사용한 용량: ", ingr_name, required_amount)
                    print("- 남은 용량: ", ingr_name, ingredient[ingr_name])
                
                else:   # 크거나 같지 않다면, 재고가 부족한 경우 → 다른 메뉴 선택 유도
                    print("- 현재 원두 재고: ", ingredient["bean"])
                    print("수량이 부족합니다. 다른 메뉴를 선택해주세요.")
                    return select_drink()   # 음료 선택 함수로 돌아감
                    
            return True    # 모든 재료가 충분한 경우
        



# -------------------------------------------
# 메인 함수 : 전체 음료 주문 프로세스 운영
# -------------------------------------------
def coffee_machine():
    while True:
        drink = select_drink()      # 음료 선택
        pay(drink)                  # 결제 진행

        # 재료 체크 후 제조
        if ingredient_check(drink):
            making_drink(drink)

        # 추가 주문 여부 확인
        if continue_order() == False:
            break


# 프로그램 실행
coffee_machine()
