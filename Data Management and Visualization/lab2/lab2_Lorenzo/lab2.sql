-- 6. Consider the year 2003. Separately for phone rate and month, analyze the (i) average daily income
-- and the (ii) average income for number of calls.

select DATEMONTH, FACTS.ID_PHONERATE, PHONERATETYPE, sum(PRICE)/count(DISTINCT DAYOFMONTH) as AVG_DAILY, sum(PRICE)/sum(NUMBEROFCALLS) as AVG_FOR_N_CALLS
from lorenzo.TIMEDIM, lorenzo.PHONERATE, lorenzo.FACTS
where FACTS.ID_TIME=TIMEDIM.ID_TIME and FACTS.ID_PHONERATE=PHONERATE.ID_PHONERATE and DATEYEAR=2003
group by DATEMONTH, FACTS.ID_PHONERATE, PHONERATETYPE;


-- 7. Select the daily number of calls for each caller region and the daily number of calls for each caller
-- province.

select PROVINCE, REGION, sum(NUMBEROFCALLS) as PROVINCE_SUM, sum(sum(NUMBEROFCALLS)) over (PARTITION BY REGION) as REGION_SUM
from lorenzo.TIMEDIM, lorenzo.PHONERATE, lorenzo.FACTS, lorenzo.LOCATION
where FACTS.ID_TIME=TIMEDIM.ID_TIME and FACTS.ID_PHONERATE=PHONERATE.ID_PHONERATE and LOCATION.ID_LOCATION=FACTS.ID_LOCATION_CALLER
group by PROVINCE, REGION;


-- 8. Consider the year 2003. Separately for phone rate and month, analyze the (i) total income, (ii) the
-- percentage of income with respect to the total revenue considering all the phone rates, (iii) the percentage
-- of income with respect to the total revenue considering all the months.

select DATEMONTH, F.ID_PHONERATE, sum(PRICE), sum(PRICE)/sum(sum(PRICE)) over ( partition by DATEMONTH) as MONTLY_RATIO,
       sum(PRICE)/sum(sum(PRICE)) over ( partition by F.ID_PHONERATE) as PHONERATE_RATIO
from lorenzo.TIMEDIM T, lorenzo.PHONERATE P, lorenzo.FACTS F, lorenzo.LOCATION L
where F.ID_TIME=T.ID_TIME and F.ID_PHONERATE=P.ID_PHONERATE and L.ID_LOCATION=F.ID_LOCATION_CALLER AND T.DATEYEAR = 2003
group by DATEMONTH, F.ID_PHONERATE;


-- 9. For each caller province, analyze (i) the total number of calls and (ii) the percentage of number of
-- calls with respect to the total number of calls considering the corresponding region.

select REGION, PROVINCE, sum(NUMBEROFCALLS) as PROVINCE_CALLS, sum(NUMBEROFCALLS)/sum(sum(NUMBEROFCALLS))over ( PARTITION BY REGION) as REGION_PERCENTAGE
from lorenzo.TIMEDIM T, lorenzo.PHONERATE P, lorenzo.FACTS F, lorenzo.LOCATION L
where F.ID_TIME=T.ID_TIME and F.ID_PHONERATE=P.ID_PHONERATE and L.ID_LOCATION=F.ID_LOCATION_CALLER
group by PROVINCE, REGION;


-- 10. For each receiver region, select the monthly number of calls and the cumulative monthly number of
-- calls from the beginning of the year.


select REGION, DATEMONTH, sum(NUMBEROFCALLS) as MONTHLY_CALLS,
      sum(sum(NUMBEROFCALLS)) over ( PARTITION BY DATEYEAR order by DATEMONTH rows unbounded preceding ) as CUMULATIVE_MONTHLY_CALLS
from lorenzo.TIMEDIM T, lorenzo.PHONERATE P, lorenzo.FACTS F, lorenzo.LOCATION L
where F.ID_TIME=T.ID_TIME and F.ID_PHONERATE=P.ID_PHONERATE and L.ID_LOCATION=F.ID_LOCATION_RECEIVER
group by REGION, DATEYEAR, DATEMONTH;

-- the last one doesn't work because DATEMONTH is not saved in MM-YYYY as it should be (first 9 months have one digit instead of two)
-- so sorting results in october firstly and then july and august :(((
