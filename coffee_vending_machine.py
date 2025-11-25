recipe = {
    "americano":[("bean", 25), ("hot_water", 8)],
    "cafelatte":[("bean", 25), ("hot_milk", 8)],
    "matchalatte":[("matcha", 25), ("hot_milk", 8)],
    "espresso":[("bean", 30)]
}   # 음료별 레시피 딕셔너리

ingredient = {
    "bean": 50,
    "matcha": 50,
    "hot_water": 100,
    "hot_milk": 100,
    "ice": 90
}   # 재료별 초기 재고 딕셔너리

price = {
    "americano": 2000,
    "cafelatte": 2500,
    "matchalatte": 3000,
    "espresso": 1500
}   # 음료별 가격 딕셔너리


question = """
1. 아메리카노
2. 카페라테
3. 말차라테
4. 에스프레소
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
        return select_drink()   # 지정된 숫자 외 입력 시 이 함수를 다시 호출하여 input 실행 (재귀 함수 사용)
    
    return key  # 선택한 음료 문자열 반환

select_drink()