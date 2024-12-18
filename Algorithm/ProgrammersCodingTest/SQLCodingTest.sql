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



-- 자동차 대여 기록에서 장기/단기 대여 구분하기(151138) - Lv.1
SELECT HISTORY_ID, CAR_ID, DATE_FORMAT(START_DATE, '%Y-%m-%d') AS START_DATE, DATE_FORMAT(END_DATE, '%Y-%m-%d') AS END_DATE, IF(DATEDIFF(END_DATE, START_DATE) < 29, '단기 대여', '장기 대여') AS RENT_TYPE  -- IF == CASE WHEN THEN ELSE END
FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
WHERE DATE_FORMAT(START_DATE, '%Y-%m') = '2022-09'  -- WHERE START_DATE LIKE '2022-09-%'
ORDER BY HISTORY_ID DESC;



-- 평균 일일 대여 요금 구하기(151136) - Lv.1
SELECT ROUND(AVG(DAILY_FEE), 0) AS AVERAGE_FEE
FROM CAR_RENTAL_COMPANY_CAR
WHERE CAR_TYPE = 'SUV';



-- 조건에 맞는 도서 리스트 출력하기(144853) - Lv.1
SELECT BOOK_ID, DATE_FORMAT(PUBLISHED_DATE, '%Y-%m-%d') AS PUBLISHED_DATE
FROM BOOK
WHERE CATEGORY = '인문' AND PUBLISHED_DATE LIKE '2021-%';



-- 부모의 형질을 모두 가지는 대장균 찾기(301647) - Lv.2
SELECT A.ID, A.GENOTYPE, B.GENOTYPE AS PARENT_GENOTYPE
FROM ECOLI_DATA AS A, ECOLI_DATA AS B
WHERE A.PARENT_ID = B.ID AND B.GENOTYPE & A.GENOTYPE = B.GENOTYPE
ORDER BY ID;



-- 분기별 분화된 대장균의 개체 수 구하기(299308) - Lv.2
SELECT CONCAT(QUARTER(DIFFERENTIATION_DATE), 'Q') AS QUARTER, COUNT(ID) AS ECOLI_COUNT
FROM ECOLI_DATA
GROUP BY QUARTER
ORDER BY QUARTER;



-- 과일로 만든 아이스크림 고르기(133025) - Lv.1
SELECT F.FLAVOR
FROM FIRST_HALF AS F, ICECREAM_INFO AS I
WHERE F.FLAVOR = I.FLAVOR AND F.TOTAL_ORDER > 3000 AND I.INGREDIENT_TYPE = 'fruit_based'
ORDER BY F.TOTAL_ORDER DESC;



-- 인기있는 아이스크림(133024) - Lv.1
SELECT FLAVOR
FROM FIRST_HALF
ORDER BY TOTAL_ORDER DESC, SHIPMENT_ID;



-- 흉부외과 또는 일반외과 의사 목록 출력하기(132203) - Lv.1
SELECT DR_NAME, DR_ID, MCDP_CD, DATE_FORMAT(HIRE_YMD, '%Y-%m-%d') AS HIRE_YMD
FROM DOCTOR
WHERE MCDP_CD = 'CS' OR MCDP_CD = 'GS'
ORDER BY HIRE_YMD DESC, DR_NAME;



-- 연도별 대장균 크기의 편차 구하기(299310) - Lv.2
WITH MAX_SIZE_OF_COLONY AS (
    SELECT MAX(SIZE_OF_COLONY) MAX_SIZE, YEAR(DIFFERENTIATION_DATE) YEAR 
    FROM ECOLI_DATA 
    GROUP BY YEAR
)

SELECT MS.YEAR, (MS.MAX_SIZE - ED.SIZE_OF_COLONY) YEAR_DEV, ED.ID 
FROM MAX_SIZE_OF_COLONY MS, ECOLI_DATA ED
WHERE YEAR(DIFFERENTIATION_DATE) = MS.YEAR
ORDER BY MS.YEAR, YEAR_DEV;



-- 월별 잡은 물고기 수 구하기(293260) - Lv.2
SELECT COUNT(ID) AS FISH_COUNT, MONTH(TIME) AS MONTH
FROM FISH_INFO
GROUP BY MONTH
ORDER BY MONTH;



-- 특정 물고기를 잡은 총 수 구하기(298518) - Lv.2
SELECT count(*) AS FISH_COUNT
FROM FISH_INFO AS F,FISH_NAME_INFO AS N
WHERE F.FISH_TYPE = N.FISH_TYPE AND N.FISH_NAME in ('BASS', 'SNAPPER');



-- 물고기 종류 별 잡은 수 구하기(293257) - Lv.2
SELECT COUNT(*) AS FISH_COUNT, N.FISH_NAME
FROM FISH_NAME_INFO AS N, FISH_INFO AS I
WHERE N.FISH_TYPE = I.FISH_TYPE
GROUP BY N.FISH_NAME
ORDER BY FISH_COUNT DESC;



-- 노선별 평균 역 사이 거리 조회하기(284531) - Lv.2
SELECT ROUTE, CONCAT(ROUND(SUM(D_BETWEEN_DIST), 1),'km') AS TOTAL_DISTANCE, CONCAT(ROUND(AVG(D_BETWEEN_DIST), 2), 'km') AS AVERAGE_DISTANCE
FROM SUBWAY_DISTANCE
GROUP BY ROUTE
ORDER BY ROUND(SUM(D_BETWEEN_DIST)) DESC;



-- 연도 별 평균 미세먼지 농도 조회하기(284530) - Lv.2
SELECT YEAR(YM) AS YEAR, ROUND(AVG(PM_VAL1),2) AS PM10, ROUND(AVG(PM_VAL2),2) AS 'PM2.5'
FROM AIR_POLLUTION
GROUP BY YEAR, LOCATION1, LOCATION2 HAVING LOCATION2 = '수원'
ORDER BY YEAR;



-- 조건에 맞는 사원 정보 조회하기(284527) - Lv.2
SELECT SUM(SCORE) AS SCORE, G.EMP_NO, E.EMP_NAME, E.POSITION, E.EMAIL
FROM HR_EMPLOYEES AS E, HR_GRADE AS G
WHERE E.EMP_NO = G.EMP_NO
GROUP BY YEAR, EMP_NO HAVING G.YEAR = '2022'
ORDER BY 1 DESC
LIMIT 1;



-- 조건에 맞는 개발자 찾기(276034) - Lv.2
SELECT ID, EMAIL, FIRST_NAME, LAST_NAME
FROM DEVELOPERS
WHERE SKILL_CODE & (SELECT CODE FROM SKILLCODES WHERE NAME='Python') OR SKILL_CODE & (SELECT CODE FROM SKILLCODES WHERE NAME='C#')
ORDER BY ID;



-- 조건에 맞는 도서와 저자 리스트 출력하기(144854) - Lv.2
SELECT B.BOOK_ID, A.AUTHOR_NAME, DATE_FORMAT(B.PUBLISHED_DATE, '%Y-%m-%d') AS PUBLISHED_DATE
FROM BOOK AS B, AUTHOR AS A
WHERE B.AUTHOR_ID = A.AUTHOR_ID AND B.CATEGORY = '경제'
ORDER BY PUBLISHED_DATE;



-- 대장균의 크기에 따라 분류하기 1(299307) - Lv.3
SELECT ID, CASE WHEN SIZE_OF_COLONY <= 100 THEN 'LOW' 
                WHEN SIZE_OF_COLONY > 1000 THEN 'HIGH'
                ELSE 'MEDIUM'
            END AS SIZE
FROM ECOLI_DATA
ORDER BY ID;



-- FrontEnd 개발자 찾기(276035) - Lv.4
SELECT DISTINCT ID, EMAIL, FIRST_NAME, LAST_NAME
FROM DEVELOPERS AS D
JOIN SKILLCODES AS S ON S.CODE & D.SKILL_CODE
WHERE S.CATEGORY = 'Front End'
ORDER BY ID ASC;



-- 상품 별 오프라인 매출 구하기(131533) - Lv.2
SELECT PRODUCT_CODE, SUM(PRICE * SALES_AMOUNT) AS SALES
FROM PRODUCT AS P
JOIN OFFLINE_SALE AS O ON P.PRODUCT_ID = O.PRODUCT_ID
GROUP BY PRODUCT_CODE
ORDER BY SALES DESC, PRODUCT_CODE;



-- 특정 기간동안 대여 가능한 자동차들의 대여비용 구하기(157339) - Lv.4
SELECT C.CAR_ID, C.CAR_TYPE, ROUND(C.DAILY_FEE * 30 * (100 - P.DISCOUNT_RATE) / 100) AS FEE
FROM CAR_RENTAL_COMPANY_CAR AS C
JOIN CAR_RENTAL_COMPANY_RENTAL_HISTORY AS H ON C.CAR_ID = H.CAR_ID
JOIN CAR_RENTAL_COMPANY_DISCOUNT_PLAN AS P ON C.CAR_TYPE = P.CAR_TYPE
WHERE C.CAR_ID NOT IN (
    SELECT CAR_ID
    FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
    WHERE END_DATE > '2022-11-01' AND START_DATE < '2022-12-01'
) AND P.DURATION_TYPE = '30일 이상'
GROUP BY C.CAR_ID
HAVING C.CAR_TYPE IN ('세단', 'SUV') AND (FEE >= 500000 AND FEE < 2000000) 
ORDER BY FEE DESC, CAR_TYPE, CAR_ID DESC;



-- 없어진 기록 찾기(59042) - Lv.3
SELECT AO.ANIMAL_ID, AO.NAME
FROM ANIMAL_OUTS AS AO
LEFT JOIN ANIMAL_INS AS AI ON AO.ANIMAL_ID = AI.ANIMAL_ID
WHERE AI.DATETIME IS NULL
ORDER BY AO.ANIMAL_ID ASC;



-- 있었는데요 없었습니다(59043) - Lv.3
SELECT INS.ANIMAL_ID, INS.NAME
FROM ANIMAL_INS AS INS
INNER JOIN ANIMAL_OUTS AS OUTS ON INS.ANIMAL_ID = OUTS.ANIMAL_ID
WHERE INS.DATETIME > OUTS.DATETIME
ORDER BY INS.DATETIME;



-- 오랜 기간 보호한 동물(1)(59044) - Lv.3
SELECT I.NAME, I.DATETIME
FROM ANIMAL_INS AS I
LEFT OUTER JOIN  ANIMAL_OUTS AS O ON I.ANIMAL_ID = O.ANIMAL_ID
WHERE O.ANIMAL_ID IS NULL
ORDER BY I.DATETIME 
LIMIT 3;



-- 5월 식품들의 총매출 조회하기(131117) - Lv.4
SELECT A.PRODUCT_ID, B.PRODUCT_NAME, (SUM(A.AMOUNT) * B.PRICE) AS TOTAL_SALES
FROM FOOD_ORDER AS A
JOIN FOOD_PRODUCT AS B ON A.PRODUCT_ID = B.PRODUCT_ID
WHERE YEAR(PRODUCE_DATE) = 2022 AND MONTH(PRODUCE_DATE) = 5
GROUP BY A.PRODUCT_ID
ORDER BY TOTAL_SALES DESC, A.PRODUCT_ID ASC;



-- 카테고리 별 도서 판매량 집계하기(144855) - Lv.3
SELECT CATEGORY, SUM(S.SALES) AS TOTAL_SALES
FROM BOOK AS B, BOOK_SALES AS S
WHERE B.BOOK_ID = S.BOOK_ID AND SALES_DATE BETWEEN '2022-01-01' AND '2022-01-31'
GROUP BY CATEGORY
ORDER BY CATEGORY ASC;



-- 대여 횟수가 많은 자동차들의 월별 대여 횟수 구하기(151139) - Lv.3
SELECT MONTH(START_DATE) AS MONTH, CAR_ID, COUNT(HISTORY_ID) AS RECORDS
FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
WHERE CAR_ID IN (
        SELECT CAR_ID
        FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
        WHERE (DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')
        GROUP BY CAR_ID
        HAVING COUNT(CAR_ID) >= 5
    ) AND (DATE_FORMAT(START_DATE, '%Y-%m') BETWEEN '2022-08' AND '2022-10')
GROUP BY MONTH, CAR_ID
HAVING RECORDS > 0
ORDER BY MONTH ASC, CAR_ID DESC;



-- 주문량이 많은 아이스크림들 조회하기(133027) - Lv.4
SELECT A.FLAVOR
FROM (
    SELECT * FROM FIRST_HALF
    UNION ALL
    SELECT * FROM JULY
    ) AS A
GROUP BY A.FLAVOR
ORDER BY SUM(A.TOTAL_ORDER) DESC
LIMIT 3;



-- 그룹별 조건에 맞는 식당 목록 출력하기(131124) - Lv.4
SELECT A.MEMBER_NAME,B.REVIEW_TEXT,DATE_FORMAT(B.REVIEW_DATE, "%Y-%m-%d") AS REVIEW_DATE
FROM MEMBER_PROFILE A JOIN REST_REVIEW B ON A.MEMBER_ID = B.MEMBER_ID
WHERE A.MEMBER_ID = (
    SELECT MEMBER_ID 
    FROM REST_REVIEW
    GROUP BY MEMBER_ID
    ORDER BY COUNT(*) DESC 
    LIMIT 1)
ORDER BY REVIEW_DATE ASC, REVIEW_TEXT;



-- 이름이 있는 동물의 아이디(59407) - Lv.1
SELECT ANIMAL_ID
FROM ANIMAL_INS
WHERE NAME IS NOT NULL
ORDER BY ANIMAL_ID;



-- 대장균들의 자식의 수 구하기(299305) - Lv.3
SELECT A.ID, COUNT(B.ID) AS CHILD_COUNT
FROM ECOLI_DATA AS A 
LEFT JOIN ECOLI_DATA AS B ON A.ID = B.PARENT_ID
GROUP BY A.ID
ORDER BY A.ID;



-- 상품을 구매한 회원 비율 구하기(131534) - Lv.5
SELECT DATE_FORMAT(O.SALES_DATE, '%Y') AS YEAR,
       DATE_FORMAT(O.SALES_DATE, '%m') AS MONTH,
       COUNT(DISTINCT U.USER_ID) AS PUCHASED_USERS,
       ROUND(COUNT(DISTINCT U.USER_ID) / (
        SELECT COUNT(*) 
        FROM USER_INFO 
        WHERE joined 
        LIKE '2021%'), 1
        ) AS PUCHASED_RATIO
FROM USER_INFO AS U
JOIN ONLINE_SALE AS O ON U.USER_ID = O.USER_ID
WHERE U.JOINED LIKE '2021%'
GROUP BY YEAR, MONTH
ORDER BY YEAR, MONTH;