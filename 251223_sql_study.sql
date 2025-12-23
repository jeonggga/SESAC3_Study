-- 기존 DB들 삭제 (있으면 삭제, 없어도 에러 안 남)
DROP DATABASE IF EXISTS ShopDB;
DROP DATABASE IF EXISTS ModelDB;
DROP DATABASE IF EXISTS sqldb;
DROP DATABASE IF EXISTS tabledb;

-- DB 생성 (⚠️ 아직 선택은 안 된 상태)
CREATE DATABASE tabledb;


-- ❌ 문제 발생 지점 ❌
-- 1) tabledb를 USE로 선택하지 않았음
-- 2) usertbl 테이블이 존재하지 않음
CREATE TABLE `tabledb`.`buytbl` (
  `num` INT NOT NULL AUTO_INCREMENT,
  `userid` CHAR(8) NOT NULL,
  `prodName` CHAR(6) NOT NULL,
  `groupName` CHAR(4) NULL,
  `price` INT NOT NULL,
  `amount` SMALLINT NOT NULL,
  PRIMARY KEY (`num`),
  FOREIGN KEY (userid) REFERENCES usertbl(userID)  
);


-- ✅ 이제 DB 선택
USE tabledb;

-- 단순 테이블 생성 (외래키 없음 → 정상 동작)
CREATE TABLE test (num INT);

-- 현재 선택된 DB 확인
SELECT DATABASE();

-- tabledb 안의 테이블 목록 확인
SHOW TABLES;


-- 기존 tabledb 데이터베이스 삭제
-- ⚠️ 존재하지 않으면 에러 발생하므로 보통 IF EXISTS를 붙임
DROP DATABASE tabledb;


-- tabledb 데이터베이스 새로 생성
CREATE DATABASE tabledb;

-- 앞으로 실행할 SQL의 대상 DB를 tabledb로 선택
USE tabledb;


-- 기존 테이블이 있다면 삭제
-- buytbl, usertbl 순서는 상관없음 (외래키 없음)
DROP TABLE IF EXISTS buytbl, usertbl;


-- =========================
-- 회원 테이블 생성
-- =========================
CREATE TABLE usertbl -- 회원 테이블
( userID  CHAR(8), -- 사용자 아이디
  name    VARCHAR(10), -- 이름
  birthYear   INT,  -- 출생년도
  addr	  CHAR(2), -- 지역(경기,서울,경남 등으로 글자만 입력)
  mobile1  CHAR(3), -- 휴대폰의국번(011, 016, 017, 018, 019, 010 등)
  mobile2  CHAR(8), -- 휴대폰의 나머지 전화번호(하이픈 제외)
  height    SMALLINT,  -- 키
  mDate    DATE  -- 회원 가입일
  -- ⚠️ 현재 PRIMARY KEY가 없음
);

-- 현재 DB(tabledb)에 존재하는 테이블 목록 확인
SHOW TABLES;


-- =========================
-- 구매 테이블 생성
-- =========================
CREATE TABLE buytbl -- 구매 테이블
(  num INT, -- 순번(PK)
   userid  CHAR(8),-- 아이디(FK)
   prodName CHAR(6), -- 물품명
   groupName CHAR(4) , -- 분류
   price     INT , -- 단가
   amount SMALLINT -- 수량
   -- ⚠️ 현재 PRIMARY KEY, FOREIGN KEY 없음
);

-- 테이블이 정상적으로 생성되었는지 확인
SHOW TABLES;


-- 다시 tabledb 사용 선언 (이미 사용 중이므로 사실상 불필요)
USE tabledb;


-- 기존에 존재하는 구매 테이블(buytbl)과 회원 테이블(usertbl)을 삭제
-- 테이블이 없어도 에러가 발생하지 않음
DROP TABLE IF EXISTS buytbl, usertbl;


-- =========================
-- 회원 테이블 생성
-- =========================
CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL,     -- 사용자 아이디 (NULL 허용 안 함)
  name    VARCHAR(10) NOT NULL, -- 사용자 이름
  birthYear   INT NOT NULL,     -- 출생년도
  addr	  CHAR(2) NOT NULL,     -- 거주 지역 (서울, 경기 등 2글자)
  mobile1 CHAR(3) NULL,         -- 휴대폰 국번 (010, 011 등, 선택 입력)
  mobile2 CHAR(8) NULL,         -- 휴대폰 나머지 번호 (선택 입력)
  height  SMALLINT NULL,        -- 키 (선택 입력)
  mDate   DATE NULL             -- 회원 가입일 (선택 입력)
  -- ⚠️ 현재 PRIMARY KEY가 정의되어 있지 않음
);

-- =========================
-- 구매 테이블 생성
-- =========================
CREATE TABLE buytbl 
(
  num INT NOT NULL,             -- 구매 번호 (보통 PK로 사용)
  userid  CHAR(8) NOT NULL,     -- 사용자 아이디 (usertbl.userID와 연결 예정)
  prodName CHAR(6) NOT NULL,    -- 상품명
  groupName CHAR(4) NULL,       -- 상품 분류
  price INT NOT NULL,           -- 상품 단가
  amount SMALLINT NOT NULL      -- 구매 수량
  -- ⚠️ 현재 PRIMARY KEY, FOREIGN KEY 없음
);

-- 현재 사용 중인 데이터베이스를 tabledb로 선택
-- 이미 선택되어 있다면 실행하지 않아도 됨
USE tabledb;




-- 기존에 존재하는 구매 테이블(buytbl)과 회원 테이블(usertbl)을 삭제
DROP TABLE IF EXISTS buytbl, usertbl;

-- =========================
-- 회원 테이블 생성
-- =========================
CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL PRIMARY KEY, -- 회원 아이디 (기본키)
  name    VARCHAR(10) NOT NULL,          -- 이름
  birthYear   INT NOT NULL,              -- 출생년도
  addr	  CHAR(2) NOT NULL,              -- 거주 지역
  mobile1 CHAR(3) NULL,                  -- 휴대폰 국번
  mobile2 CHAR(8) NULL,                  -- 휴대폰 나머지 번호
  height  SMALLINT NULL,                 -- 키
  mDate   DATE NULL                      -- 가입일
);

-- =========================
-- 구매 테이블 생성 (1차 시도)
-- =========================
CREATE TABLE buytbl 
(
  num INT NOT NULL PRIMARY KEY,           -- 구매 번호 (PK, 자동증가 아님)
  userid  CHAR(8) NOT NULL,               -- 회원 아이디
  prodName CHAR(6) NOT NULL,              -- 상품명
  groupName CHAR(4) NULL,                 -- 상품 분류
  price INT NOT NULL,                     -- 단가
  amount SMALLINT NOT NULL                -- 수량
);

-- buytbl 구조를 다시 만들기 위해 삭제
DROP TABLE IF EXISTS buytbl;

-- =========================
-- 구매 테이블 생성 (AUTO_INCREMENT 적용)
-- =========================
CREATE TABLE buytbl 
(
  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY, -- 구매 번호 (자동 증가)
  userid  CHAR(8) NOT NULL,                    -- 회원 아이디
  prodName CHAR(6) NOT NULL,                   -- 상품명
  groupName CHAR(4) NULL,                      -- 상품 분류
  price INT NOT NULL,                          -- 단가
  amount SMALLINT NOT NULL                     -- 수량
);

-- 외래 키를 추가하기 위해 다시 테이블 삭제
DROP TABLE IF EXISTS buytbl;

-- =========================
-- 구매 테이블 최종 생성 (외래 키 포함)
-- =========================
CREATE TABLE buytbl 
(
  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY, -- 구매 번호 (PK)
  userid  CHAR(8) NOT NULL,                    -- 회원 아이디 (FK)
  prodName CHAR(6) NOT NULL,                   -- 상품명
  groupName CHAR(4) NULL,                      -- 상품 분류
  price INT NOT NULL,                          -- 단가
  amount SMALLINT NOT NULL,                    -- 수량
  FOREIGN KEY(userid) REFERENCES usertbl(userID)
  -- usertbl.userID를 참조하는 외래 키
);

-- =========================
-- 회원 데이터 입력
-- =========================
INSERT INTO usertbl 
VALUES('LSG', '이승기', 1987, '서울', '011', '1111111', 182, '2008-8-8');

INSERT INTO usertbl 
VALUES('KBS', '김범수', 1979, '경남', '011', '2222222', 173, '2012-4-4');

INSERT INTO usertbl 
VALUES('KKH', '김경호', 1971, '전남', '019', '3333333', 177, '2007-7-7');

-- =========================
-- 구매 데이터 입력
-- =========================
-- num 컬럼은 AUTO_INCREMENT이므로 NULL 입력 → 자동 증가
INSERT INTO buytbl 
VALUES(NULL, 'KBS', '운동화', NULL, 30, 2);

INSERT INTO buytbl 
VALUES(NULL, 'KBS', '노트북', '전자', 1000, 1);

-- ❌ 아래 INSERT는 에러 발생
-- 이유: 'JYP'는 usertbl에 존재하지 않는 userID
-- 외래 키 제약 조건 위반
INSERT INTO buytbl 
VALUES(NULL, 'JYP', '모니터', '전자', 200, 1);






-- =========================
-- 현재 DB 내 테이블 목록 확인
-- =========================
SHOW TABLES;

-- =========================
-- 회원 데이터 입력
-- =========================
INSERT INTO usertbl VALUES('JYP', '조용필', 1950, '경기', '011', '4444444', 166, '2009-4-4');
INSERT INTO usertbl VALUES('SSK', '성시경', 1979, '서울', NULL, NULL, 186, '2013-12-12');
INSERT INTO usertbl VALUES('LJB', '임재범', 1963, '서울', '016', '6666666', 182, '2009-9-9');
INSERT INTO usertbl VALUES('YJS', '윤종신', 1969, '경남', NULL, NULL, 170, '2005-5-5');
INSERT INTO usertbl VALUES('EJW', '은지원', 1972, '경북', '011', '8888888', 174, '2014-3-3');
INSERT INTO usertbl VALUES('JKW', '조관우', 1965, '경기', '018', '9999999', 172, '2010-10-10');
INSERT INTO usertbl VALUES('BBK', '바비킴', 1973, '서울', '010', '0000000', 176, '2013-5-5');

-- =========================
-- 구매 데이터 입력
-- num 컬럼은 AUTO_INCREMENT → NULL로 입력하면 자동 증가
-- =========================
INSERT INTO buytbl VALUES(NULL, 'JYP', '모니터', '전자', 200, 1);
INSERT INTO buytbl VALUES(NULL, 'BBK', '모니터', '전자', 200, 5);
INSERT INTO buytbl VALUES(NULL, 'KBS', '청바지', '의류', 50, 3);
INSERT INTO buytbl VALUES(NULL, 'BBK', '메모리', '전자', 80, 10);
INSERT INTO buytbl VALUES(NULL, 'SSK', '책', '서적', 15, 5);
INSERT INTO buytbl VALUES(NULL, 'EJW', '책', '서적', 15, 2);
INSERT INTO buytbl VALUES(NULL, 'EJW', '청바지', '의류', 50, 1);
INSERT INTO buytbl VALUES(NULL, 'BBK', '운동화', NULL, 30, 2);
INSERT INTO buytbl VALUES(NULL, 'EJW', '책', '서적', 15, 1);
INSERT INTO buytbl VALUES(NULL, 'BBK', '운동화', NULL, 30, 2);

-- =========================
-- 데이터베이스 선택
-- =========================
USE tabledb;

-- =========================
-- 테이블 초기화 및 PK 정의 연습
-- =========================
DROP TABLE IF EXISTS buytbl, usertbl;

CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL PRIMARY KEY,  -- PRIMARY KEY 바로 지정
  name    VARCHAR(10) NOT NULL, 
  birthYear INT NOT NULL
);

-- 테이블 구조 확인
DESCRIBE usertbl;

-- =========================
-- PK 제약 조건를 CONSTRAINT로 지정하는 방식 연습
-- =========================
DROP TABLE IF EXISTS usertbl;

CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL, 
  name    VARCHAR(10) NOT NULL, 
  birthYear INT NOT NULL,  
  CONSTRAINT PRIMARY KEY PK_usertbl_userID (userID) -- 이름 있는 PK
);

-- =========================
-- ALTER TABLE로 PK 추가하는 방법 연습
-- =========================
DROP TABLE IF EXISTS usertbl;

CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL, 
  name    VARCHAR(10) NOT NULL, 
  birthYear INT NOT NULL
);

ALTER TABLE usertbl
  ADD CONSTRAINT PK_usertbl_userID 
  PRIMARY KEY (userID);

-- =========================
-- 상품 테이블 생성 및 복합 PK 지정
-- =========================
DROP TABLE IF EXISTS prodTbl;

CREATE TABLE prodTbl
(
  prodCode CHAR(3) NOT NULL,   -- 상품 코드
  prodID   CHAR(4) NOT NULL,   -- 상품 ID
  prodDate DATETIME NOT NULL,  -- 상품 생산일
  prodCur  CHAR(10) NULL        -- 상품 통화 단위 (선택)
);

-- 복합 PK 설정 (prodCode + prodID)
ALTER TABLE prodTbl
  ADD CONSTRAINT PK_prodTbl_proCode_prodID 
  PRIMARY KEY (prodCode, prodID);





-- =========================
-- 상품 테이블(prodTbl) 삭제 후 새로 생성
-- =========================
DROP TABLE IF EXISTS prodTbl;

CREATE TABLE prodTbl
(
  prodCode CHAR(3) NOT NULL,       -- 상품 코드
  prodID   CHAR(4) NOT NULL,       -- 상품 ID
  prodDate DATETIME NOT NULL,      -- 생산일
  prodCur  CHAR(10) NULL,          -- 통화 단위 (선택)
  CONSTRAINT PK_prodTbl_proCode_prodID   -- 이름 있는 PK 설정
    PRIMARY KEY (prodCode, prodID)       -- 복합 기본키: prodCode + prodID
);

-- =========================
-- prodTbl의 인덱스 정보 확인
-- =========================
SHOW INDEX FROM prodTbl;  -- PK 포함 모든 인덱스 확인 가능

-- =========================
-- 회원 및 구매 테이블 초기화
-- =========================
DROP TABLE IF EXISTS buytbl, usertbl;

-- =========================
-- 회원 테이블 생성
-- =========================
CREATE TABLE usertbl 
(
  userID  CHAR(8) NOT NULL PRIMARY KEY,  -- 회원 ID (기본키)
  name    VARCHAR(10) NOT NULL,          -- 이름
  birthYear INT NOT NULL                  -- 출생년도
);

-- =========================
-- 구매 테이블 생성
-- =========================
CREATE TABLE buytbl 
(
  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY,  -- 구매 번호 (자동 증가, PK)
  userID CHAR(8) NOT NULL,                      -- 구매자 ID (외래 키)
  prodName CHAR(6) NOT NULL,                    -- 상품명
  FOREIGN KEY(userID) REFERENCES usertbl(userID)  -- 외래 키: 회원 테이블 userID 참조
);

-- =========================
-- 핵심 포인트
-- =========================
-- 1. prodTbl: 복합 PK(prodCode + prodID) → 하나의 컬럼이 아닌 2개 이상을 묶어서 고유값 생성
-- 2. SHOW INDEX: 테이블의 PK, UNIQUE, 기타 인덱스 확인
-- 3. usertbl: 단일 PK 지정
-- 4. buytbl: num은 AUTO_INCREMENT PK
-- 5. buytbl.userID → 외래 키로 usertbl.userID 참조 / 구매 기록이 **누가 샀는지(userID)**를 연결할 때 쓰임 / 외래 키는 다른 테이블의 값과 연결하는 다리 역할
-- 6. 외래 키 사용 시 부모 테이블(usertbl)이 먼저 생성되어 있어야 함
