# SQL基础

## **IN, BETWEEN, SUM, DISTINCT, ORDER BY, LIKE** <a id="IN,-BETWEEN,-SUM,-DISTINCT,-ORDER-BY,-LIKE"></a>

SELECT name, population FROM world

 WHERE name **IN** \('Sweden', 'Norway', 'Denmark'\)

SELECT name, area FROM world

 WHERE area **BETWEEN** 200000 **AND** 250000

 SELECT **SUM**\(population\) FROM world

SELECT **DISTINCT** region FROM bbc **ORDER BY** population **DESC**

SELECT \* FROM goal

 WHERE player **LIKE** '%Bender'

## SELECT in SELECT <a id="SELECT-in-SELECT"></a>

SELECT name FROM world

 WHERE population &gt;

 \(SELECT population FROM world WHERE name='Romania'\)

## **JOIN** <a id="JOIN"></a>

SELECT \*

 FROM game JOIN goal ON id=matchid

 SELECT a.company, a.num, a.stop, b.stop

FROM route a JOIN route b ON

 \(a.company=b.company AND a.num=b.num\)

WHERE a.stop=53

原创声明，本文系作者授权云+社区发表，未经许可，不得转载。

如有侵权，请联系 yunjia\_community@tencent.com 删除。

