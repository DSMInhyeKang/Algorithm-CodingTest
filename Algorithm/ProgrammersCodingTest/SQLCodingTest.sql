-- 특정 형질을 가지는 대장균 찾기(301646) - Lv.1
SELECT COUNT(ID) AS COUNT
FROM ECOLI_DATA
WHERE !(GENOTYPE & 2) AND (GENOTYPE & 1 OR GENOTYPE & 4);  -- 비트 연산 시 3번째 형질을 나타내는 비트는 100(2), 4(10)임에 유의



-- 가장 큰 물고기 10마리 구하기(298517) - Lv.1
SELECT ID, LENGTH
FROM FISH_INFO
WHERE LENGTH IS NOT NULL
ORDER BY LENGTH DESC, ID
LIMIT 10;



-- 한 해에 잡은 물고기 수 구하기(298516) - Lv.1
SELECT COUNT(ID) AS FISH_COUNT
FROM FISH_INFO
WHERE DATE_FORMAT(TIME, '%Y') = 2021;



-- 잡은 물고기 중 가장 큰 물고기의 길이 구하기(298515) - Lv.1
SELECT CONCAT(MAX(LENGTH), 'cm') AS MAX_LENGTH
FROM FISH_INFO;



-- 잡은 물고기의 평균 길이 구하기(293259) - Lv.1
SELECT ROUND(AVG(IFNULL(LENGTH, 10)), 2) AS AVERAGE_LENGTH
FROM FISH_INFO;



-- 잔챙이 잡은 수 구하기(293258) - Lv.1
SELECT COUNT(ID) AS FISH_COUNT 
FROM FISH_INFO
WHERE LENGTH IS NULL;



-- Python 개발자 찾기(276013) - Lv.1
SELECT ID, EMAIL, FIRST_NAME, LAST_NAME
FROM DEVELOPER_INFOS
WHERE 'Python' IN (SKILL_1, SKILL_2, SKILL_3)  -- WHERE SKILL_1 LIKE'%Python%' OR SKILL_2 LIKE '%Python%' OR SKILL_3 LIKE '%Python%'
ORDER BY ID ASC;



-- 조건에 부합하는 중고거래 댓글 조회하기(164673) - Lv.1
SELECT B.TITLE, B.BOARD_ID, R.REPLY_ID, R.WRITER_ID, R.CONTENTS, DATE_FORMAT(R.CREATED_DATE, '%Y-%m-%d') AS CREATED_DATE
FROM USED_GOODS_BOARD AS B JOIN USED_GOODS_REPLY AS R ON B.BOARD_ID = R.BOARD_ID  -- 원래 JOIN이 INNER JOIN과 동의어로 쓰이지만, 여기서는 테이블 간 같은 칼럼명이 여러 개므로 ON으로 지정해야 함
WHERE B.CREATED_DATE BETWEEN '2022-10-01' AND '2022-10-31'
ORDER BY R.CREATED_DATE, B.TITLE



-- 특정 옵션이 포함된 자동차 리스트 구하기(157343) - Lv.1
SELECT *
FROM CAR_RENTAL_COMPANY_CAR
WHERE OPTIONS LIKE '%네비게이션%'
ORDER BY CAR_ID DESC;