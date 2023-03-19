---
title: MySQL最左前缀
date: 2023-03-19 23:27:57
tags:
  - MySQL
categories:
  - 技术
  - MySQL
top:
---

记录下MySQL中索引的最左前缀原则和联合索引有什么好处

<!--more-->

### 最左前缀原则

最左匹配原则就是指在联合索引中，如果你的 SQL 语句中用到了联合索引中的最左边的索引，那么这条 SQL 语句就可以利用这个联合索引去进行匹配。例如某表现有索引(a,b,c)，现在你有如下语句：

```mysql
select * from t where a=1 and b=1 and c =1;     #这样可以利用到定义的索引（a,b,c）,用上a,b,c

select * from t where a=1 and b=1;     #这样可以利用到定义的索引（a,b,c）,用上a,b

select * from t where b=1 and a=1;     #这样可以利用到定义的索引（a,b,c）,用上a,b（mysql有查询优化器）

select * from t where a=1;     #这样也可以利用到定义的索引（a,b,c）,用上a

select * from t where b=1 and c=1;     #这样不可以利用到定义的索引（a,b,c）

select * from t where a=1 and c=1;     #这样可以利用到定义的索引（a,b,c），但只用上a索引，b,c索引用不到
```

**当遇到范围查询(>、<、between、like)就会停止匹配**。也就是：

```mysql
select * from t where a=1 and b>1 and c =1; #这样a,b可以用到（a,b,c），c索引用不到 
```

这条语句只有 a,b 会用到索引，c 都不能用到索引。这个原因可以从联合索引的结构来解释。

但是如果是建立(a,c,b)联合索引，则a,b,c都可以使用索引，因为优化器会自动改写为最优查询语句

```mysql
select * from t where a=1 and b >1 and c=1;  #如果是建立(a,c,b)联合索引，则a,b,c都可以使用索引
#优化器改写为
select * from t where a=1 and c=1 and b >1;
```

以index （a,b,c）为例建立这样的索引相当于建立了索引a、ab、abc三个索引。

### 联合索引优点

- **减少开销。**建一个联合索引(col1,col2,col3)，实际相当于建了(col1),(col1,col2),(col1,col2,col3)三个索引。每多一个索引，都会增加写操作的开销和磁盘空间的开销。对于大量数据的表，使用联合索引会大大的减少开销！
- **覆盖索引。**对联合索引(col1,col2,col3)，如果有如下的sql: select col1,col2,col3 from test where col1=1 and col2=2。那么MySQL可以直接通过遍历索引取得数据，而无需回表，这减少了很多的随机io操作。减少io操作，特别的随机io其实是dba主要的优化策略。所以，在真正的实际应用中，覆盖索引是主要的提升性能的优化手段之一。
- **效率高。**索引列越多，通过索引筛选出的数据越少。有1000W条数据的表，有如下sql:select from table where col1=1 and col2=2 and col3=3,假设假设每个条件可以筛选出10%的数据，如果只有单值索引，那么通过该索引能筛选出1000W \* 10%=100w条数据，然后再回表从100w条数据中找到符合col2=2 and col3= 3的数据，然后再排序，再分页；如果是联合索引，通过索引筛选出1000w \*10% \*10% \*10%=1w，效率提升可想而知！
