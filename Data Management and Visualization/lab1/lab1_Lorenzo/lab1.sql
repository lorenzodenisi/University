-- 1. Select the yearly income for each phone rate, the total income for each phone rate, the total yearly
-- income and the total income.

select DATEYEAR, PHONERATETYPE,  sum(F.PRICE) as INCOME
from LORENZO.PHONERATE P , LORENZO.FACTS F, LORENZO.TIMEDIM T
where P.ID_PHONERATE = F.ID_PHONERATE and F.ID_TIME=T.ID_TIME
group by CUBE (DATEYEAR, PHONERATETYPE);


-- 2. Select the monthly number of calls and the monthly income. Associate the RANK() to each month
-- according to its income (1 for the month with the highest income, 2 for the second, etc., the last
-- month is the one with the least income).

select T.DATEMONTH, sum(NUMBEROFCALLS) as MONTHLY_NUMBER_OF_CALLS, sum(F.PRICE) as INCOME, DENSE_RANK() over (ORDER BY SUM(f.PRICE)) as RANK
from lorenzo.FACTS F, lorenzo.TIMEDIM T
where F.ID_TIME=T.ID_TIME
group by T.DATEMONTH;


-- 3. For each month in 2003, select the total number of calls. Associate the RANK() to each month
-- according to its total number of calls (1 for the month with the highest number of calls, 2 for the
-- second, etc., the last month is the one with the least number of calls).

select DATEMONTH, sum(F.NUMBEROFCALLS) as MONTHLY_NUMBER_OF_CALLS, DENSE_RANK() over (order by sum(F.NUMBEROFCALLS)) as RANK
from lorenzo.FACTS F, lorenzo.TIMEDIM T
where F.ID_TIME=T.ID_TIME AND DATEYEAR='2003'
group by DATEMONTH;


-- 4. For each day in July 2003, select the total income and the average income over the last 3 days.

select DAYOFMONTH, SUM(PRICE) as TOTAL_DAILY_INCOME, AVG(sum(PRICE)) over ( order by DAYOFMONTH rows 2 preceding ) as AVERAGE_3_DAYS
from lorenzo.FACTS F, lorenzo.TIMEDIM T
where F.ID_TIME=T.ID_TIME and DATEMONTH='7-2003'
GROUP BY DAYOFMONTH;


-- 5. Select the monthly income and the cumulative monthly income from the beginning of the year.

select T.DATEMONTH, sum(F.PRICE) as MONTHLY_INCOME, sum(sum(F.PRICE)) over ( PARTITION BY DATEYEAR order by DATEMONTH desc) CUMULATIVE_YEAR_INCOME
from lorenzo.FACTS F, lorenzo.TIMEDIM T
where F.ID_TIME=T.ID_TIME
group by T.DATEMONTH, T.DATEYEAR;


-- Estimate the cardinality of the data warehouse tables (facts and dimensions) and decide whether and which
-- materialized views are needed to improve performance of the previous queries.

-- for a 2 years time span:
-- TIMEDIM ~ 365*2 rows
-- LOCATION ~ 500 rows      it doesn't change over time
-- PHONERATE ~ 7 rows       it doesn't change
-- FACTS ~ 200000 rows

-- Create the materialized views you consider convenient and compare the execution plan cost of the queries
-- using and without using the materialized views.

CREATE MATERIALIZED VIEW mw1
    as select DATEMONTH, sum(NUMBEROFCALLS), sum(PRICE)
    from lorenzo.FACTS F , lorenzo.TIMEDIM T
    where F.ID_TIME=T.ID_TIME
    group by T.DATEMONTH;


-- evaluate FACTS row number over two years
-- data we have is related to 30 days

select count(*)*24
from lorenzo.FACTS