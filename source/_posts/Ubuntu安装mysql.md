---
title: Ubuntu安装MySQL
date: 2020-04-29 11:19:00
tags: [Ubuntu, MySQL, utf8mb4]
categories:
- 技术
- MySQL
---
服务器安装MySQL有些方法搞的十分麻烦，这里记录一个Ubuntu适用的简单的方法，同时修改字符集为utf8mb4（毕竟MySQL的utf8是个赝品）

<!--more-->

参考
[Ubuntu安装mysql](https://blog.csdn.net/mr_hui_/article/details/88878836)
[更改数据库的编码为utf8mb4](https://www.cnblogs.com/silentmuh/p/11082622.html)
[MySQL字符集utf8修改为utf8mb4的方法步骤](https://www.jb51.net/article/165260.htm)

## 1.Ubuntu安装MySQL

**首先执行下面三条命令：**

```shell
sudo apt-get install mysql-server
sudo apt install mysql-client
sudo apt install libmysqlclient-dev
```
**安装成功后可以通过下面的命令测试是否安装成功：**

```shell
sudo netstat -tap | grep mysql
```

**出现如下信息则安装成功**
![image_1b6gfob7m1u4f2i0av11afd92m9](http://blog.zchsakura.top/20200429135052.png)

**可以通过如下命令进入MySQL服务：**

```shell
mysql -uroot -p你的密码
```

现在设置MySQL允许远程访问，首先编辑文件/etc/mysql/mysql.conf.d/mysqld.cnf：

**注释掉bind-address = 127.0.0.1：**

![image_1b6ggmf7h1d6b17o11iha1j1nhtem](http://blog.zchsakura.top/20200429134856.png)

**保存退出，然后进入mysql服务，执行授权命令：**

```mysql
grant all on *.* to root@'%' identified by '你的密码' with grant option;
flush privileges;
```

**然后执行quit命令退出mysql服务，执行如下命令重启mysql：**

```shell
sudo service mysql restart
```

现在在windows下可以使用navicat远程连接ubuntu下的mysql服务。

## 2.更换字符集为utf8mb4

utf8mb4编码是utf8编码的超集，兼容utf8，并且能存储4字节的表情字符。 
采用utf8mb4编码的好处是：存储与获取数据的时候，不用再考虑表情字符的编码与解码问题。

### 2.1MySQL的版本
utf8mb4的最低mysql版本支持版本为5.5.3+，若不是，请升级到较新版本。

### 2.2MySQL驱动
5.1.34可用,最低不能低于5.1.13

```mysql
SHOW VARIABLES WHERE Variable_name LIKE 'character_set_%' OR Variable_name LIKE 'collation%';
```

![20180828153217974](http://blog.zchsakura.top/20200429134724.png)

### 2.3修改MySQL配置文件

修改mysql配置文件my.cnf
my.cnf一般在etc/mysql/my.cnf位置。找到后请在以下三部分里添加如下内容： 

```
[client] 
default-character-set = utf8mb4

[mysql] 
default-character-set = utf8mb4 

[mysqld] 
character-set-client-handshake = FALSE 
character-set-server = utf8mb4 
collation-server = utf8mb4_unicode_ci 
init_connect='SET NAMES utf8mb4'
```

在这里我遇到了问题，首先用上面的方法安装的mysql的my.cnf里面并没有实际的东西而是进行引入

```
!includedir /etc/mysql/conf.d/
!includedir /etc/mysql/mysql.conf.d/
```

所以我去这两个文件夹找到了相应的内容进行了修改还有一个是在my.cnf的同级文件`debian.cnf`中

**但是！！！**

我修改完之后使用`service mysqld restart`说没有mysqld，然后我查了一下我这种安装方式应该使用`service mysql restart`,但是我使用了之后报错，第一是因为权限问题，这个可以看错误日志发现，第二就是上面添加的内容有问题，问题出在[mysqld]里，具体原因是什么我也不是很清楚，我又换了一个版本发现可以

```
[client] 
default-character-set=utf8mb4 
  
[mysqld] 
character-set-server = utf8mb4 
collation-server = utf8mb4_unicode_ci 
init_connect='SET NAMES utf8mb4'
skip-character-set-client-handshake = true 
# 一个是自己为false一个是skip自己为true，感觉上一样的，不知道啥意思
  
[mysql] 
default-character-set = utf8mb4
```

**最后**

进去mysql之后使用

```mysql
SHOW VARIABLES WHERE Variable_name LIKE 'character_set_%' OR Variable_name LIKE 'collation%';
```

![image-20200429124011522](http://blog.zchsakura.top/20200429124058.png)

**大功告成 Peace**

## 3.补充

### 3.1修改database默认的字符集

```
ALTER DATABASE database_name CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci
```
虽然修改了database的字符集为utf8mb4，但是实际只是修改了database新创建的表，默认使用utf8mb4，原来已经存在的表，字符集并没有跟着改变，需要手动为每张表设置字符集

### 3.2修改table的字符集

- 只修改表默认的字符集 `ALTER TABLE table_name DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`
- 修改表默认的字符集和所有字符列的字符集 `ALTER TABLE table_name CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`

### 3.3单独修改column默认的字符集

```
ALTER TABLE table_name CHANGE column_name column_name VARCHAR(191) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
*注：VARCHAR(191) 根据字段实例的类型填写*

### 3.4关于utf8mb4外键的坑

上个学期做课程设计的时候有一些外键创建不了，根据我层层抽丝剥茧最后发现问题在于utf8mb4上，但具体是为什么我也不是很清楚。今天在该字符集的时候在一篇文章里找到了原因。

外键的前提是索引，当索引无法创建的时候外键自然无法创建，之前在utf8的时候`VARCHAR(255)`是可以创建外键的，但是变成了utf8mb4后255个字符超出了索引的长度的限制，必须改为`VARCHAR(191)`

- 字段长度

由于从utf8升级到了utf8mb4，一个字符所占用的空间也由3个字节增长到4个字节，但是我们当初创建表时，设置的字段类型以及最大的长度没有改变。例如，你在utf8下设置某一字段的类型为`TINYTEXT`, 这中字段类型最大可以容纳255字节，三个字节一个字符的情况下可以容纳85个字符，四个字节一个字符的情况下只能容纳63个字符，如果原表中的这个字段的值有一个或多个超过了63个字符，那么转换成utf8mb4字符编码时将转换失败，你必须先将`TINYTEXT`更改为`TEXT`等更高容量的类型之后才能继续转换字符编码

- 索引

在InnoDB引擎中，最大的索引长度为767字节，三个字节一个字符的情况下，索引列的字符长度最大可以达到255，四个字节一个字符的情况下，索引的字符长度最大只能到191。如果你已经存在的表中的索引列的类型为`VARCHAR(255)`那么转换utf8mb4时同样会转换失败。你需要先将`VARCHAR(255)`更改为`VARCHAR(191)`才能继续转换字符编码

### 3.5使用Python操作mysql之1064错误

我今天使用python操作mysql的时候又出现了1064错误，之前有一次也出现了这个问题，上次发现应该是与sql语句的插入数据方式有问题，上个学期是使用的format向sql语句中插入参数的，由于上个学期的参数内容比较简单也没出什么问题，这次插入一长段带符号的文字时就出了问题，今天经过对比我才知道问题在哪。

```python 错误方法
sql3 = "INSERT INTO user_userInfo(nickname) VALUES('{}')".format(dd[3])
	try:
        # 执行SQL语句
        cursor.execute(sql3)
        # 提交修改
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()
```


```python 正确方法
sql3 = "INSERT INTO user_userInfo(nickname) VALUES(%s)"
	try:
        # 执行SQL语句
        cursor.execute(sql3, dd[3])
        # 提交修改
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()
```

大家可以看到这两者的区别就在于插入参数的方式，第一种方式大多数时间是没错的，但是一旦遇到dd[3]里面本来就带有一些特殊符号比如`/`,`'`之类的就会报错。而是用第二种方式则不会存在这个问题，要注意第二种方式不要在`%s`周围加`''`。

