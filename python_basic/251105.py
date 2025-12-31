name = input('이름 :')
age = input('나이 :')
feature = input('특징 :')
print(name+'('+age+')', feature)
print(feature+'인', name)




snack_cnt = int(input('과자를 몇 개 구매하겠습니까?'))

ramen_cnt = int(input('라면을 몇 개 구매하겠습니까?'))

ham_cnt = int(input('햄을 몇 개 구매하겠습니까?'))

price = 1200 * snack_cnt + 800 * ramen_cnt + 2400 * ham_cnt
print('총 금액 :', price, '원')





snack_cnt = int(input('과자를 몇 개 구매하겠습니까?'))

ramen_cnt = int(input('라면을 몇 개 구매하겠습니까?'))

ham_cnt = int(input('햄을 몇 개 구매하겠습니까?'))

price = 1200 * snack_cnt + 800 * ramen_cnt + 2400 * ham_cnt
print('할인 전 금액 :', price, '원', '할인 후 금액 :', price * 0.75,'원')




num1 = int(input('숫자1 입력 :'))
num2 = int(input('숫자2 입력 :'))

print('두 수를 더한 값 :', num1+num2)
print('두 수를 뺀 값 :', num1-num2)
print('두 수를 곱한 값 :', num1*num2)
print('두 수를 나눈 값 :', num1/num2)




name = input('당신은 누구입니까?')
print('나는', name+'이다')


id = input('아이디 입력 :')
name = input('이름 입력 :')

print(name, '님! 회원가입을 환영합니다!')
print(id, '님에게 지금 즉시 사용 가능한 쿠폰 5개 발급!')
print(id, '님에게만 적립금 2000원 추가 지급!')




a = int(input('국어 점수는?'))
b = int(input('수학 점수는?'))
c = int(input('영어 점수는?'))

print('최종 점수는', a*0.16+b*0.34+c*0.23, '입니다!')




age = int(input('나이가 어떻게 되세요?'))
if age < 20 :
    print('미성년자는 주류를 구매할 수 없습니다.')
    print(20 - age, '년 후에 성인이 되면 오세요!')
print('감사합니다. 안녕히 가세요')




price = int(input('상품 가격은 얼마인가?'))
if price < 50000 :
    print('배송비 2500원이 추가됩니다.')
    price = 2500 + price
print('총 결제금액은', price, '원 입니다.')




bell = input('주중 또는 주말을 입력해주세요.')

if bell == '주중' :
    print('아침 7시입니다! 주인님 일어나세요!')

else :
    print('아침 10시입니다! 주인님 일어나세요!')




number = int(input('날짜를 입력하세요.'))
if number % 2 == 0:
    print('짝수번호 차량만 통행 가능합니다. 홀수번호 차주는 오늘 대중교통을 이용하세요.')
else:
    print('홀수번호 차량만 통행 가능합니다. 짝수번호 차주는 오늘 대중교통을 이용하세요.')




age = int(input('나이를 입력하세요.'))
if age >= 20 :
    print('입장료는 15,000원 입니다.')
    price = 15000
    
    price_input = int(input('얼마 넣었나요?'))
    if price_input > price :
        print('결제가 완료되었습니다.')
        print(price_input-price,'원을 거슬러 드리겠습니다.')

    elif price_input == price :
        print('결제가 완료되었습니다.')

    else :
        print('결제가 실패되었습니다.')
        print(price-price_input, '원을 더 내야합니다.')

else:
    print('입장료는 6,000원 입니다.')
    price = 6000

    price_input = int(input('얼마 넣었나요?'))
    if price_input > price :
        print('결제가 완료되었습니다.')
        print(price_input-price,'원을 거슬러 드리겠습니다.')

    elif price_input == price :
        print('결제가 완료되었습니다.')

    else :
        print('결제가 실패되었습니다.')
        print(price-price_input, '원을 더 내야합니다.')






price = int(input('구매 금액을 입력하세요.'))

if price < 20000 :
    print('주문금액이 부족합니다.')
elif price < 50000 :
    price = 2500 + price
    print('배송비 2500원이 추가됩니다.')
else :
    print('무료배송됩니다.')

print('최종 결제금액은', price, '원입니다.')





print('미세먼지 저감 조치에 따른 차량 2부제를 시행합니다.')

number = int(input('날짜를 입력하세요.'))
if number > 31:
    print('올바른 날짜를 입력하세요.')
elif number % 2 == 0:
    print('짝수번호 차량만 통행 가능합니다. 홀수번호 차주는 오늘 대중교통을 이용하세요.')
else:
    print('홀수번호 차량만 통행 가능합니다. 짝수번호 차주는 오늘 대중교통을 이용하세요.')




print('미세먼지 저감 조치에 따른 차량 2부제를 시행합니다.')

number = int(input('날짜를 입력하세요.'))
if number <= 31:
    if number % 2 == 0:
        print('짝수번호 차량만 통행 가능합니다. 홀수번호 차주는 오늘 대중교통을 이용하세요.')
    else :
        print('홀수번호 차량만 통행 가능합니다. 홀수번호 차주는 오늘 대중교통을 이용하세요.')
else:
    print('올바른 날짜를 입력하세요.')





price = int(input('구매금액 입력'))
check = input('마케팅 동의에 체크했나요? Y/N')

if price >= 100000 and check == 'Y':
    print('사은품 지급 대상입니다.')

else :
    print('사은품 지급 대상이 아닙니다.')




price = int(input('구매금액 입력'))
check = input('마케팅 동의에 체크했나요? Y/N')

if price >= 100000 or check == 'Y':
    print('무료 주차 대상입니다.')

else :
    print('무료 주차 대상이 아닙니다.')





coughing = input('기침이 있나요? (Y/N) ')
body = float(input('체온 입력'))

if body >= 38.5 and coughing == 'Y':
    print('독감일 수 있습니다.')

else :
    print('독감이 아닙니다.')




kimbab = '김밥'
ramen = '라면'

menu = input('김밥 또는 라면을 입력해주세요')

if kimbab == menu :
    print('야채김밥(2,500원)')
    print('참치김밥(3,500원)')
elif ramen == menu :
    print('기본라면(3,500원)')
    print('떡라면(4,000원)')
    print('만두라면(4,000원)')
else :
    print('올바른 메뉴를 입력해 주세요.')





number = int(input('0점부터 100점 사이의 점수 입력'))

if number > 100 :
    print('잘못된 숫자를 입력했습니다. 숫자가 초과되었습니다.')
elif 80 <= number <= 100 :
    print('A등급')
elif 40 <= number < 80:
    print('B등급')
elif number < 0 :
    print('잘못된 숫자를 입력했습니다. 음수는 표시하지 않습니다.')
elif 0 <= number < 40 :
    print('C등급')
else :
    print('잘못된 숫자를 입력했습니다.')




## 점수 커트라인을 정하기 위한 코드

# input 값을 정수형으로 바꿔서 number에 대입함
number = int(input('0점부터 100점 사이의 점수 입력'))

# 100점 이상이거나 음수인 점수는 조건값에 해당할수 없도록 설정함.
# if문을 사용하여 기준치에 따른 등급설정
if number < 0 or 100 < number :
    print('잘못된 숫자를 입력했습니다.')
elif 80 <= number :
    print('A등급')
elif 40 <= number :
    print('B등급')
else :
    print('C등급')






