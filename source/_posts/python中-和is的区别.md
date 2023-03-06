---
layout: publish
title: python中==和is的区别
date: 2023-03-06 16:09:13
tags:
  - python
categories:
  - 技术
  - Python
---

常见的一种理解是==判断值相等，is判断内存中是否指向同一个位置，但其实这个和python的交互模式，py文件模式都有点关系，这里简单总结一下网上搜罗的内容。

<!--more-->

### 问题出现

在py文件中

```python
a = "hello"
b = "hello"
print(a is b)  # 输出 True 
print(a == b)  # 输出 True

a = [1, 2, 3]
b = [1, 2, 3]
print(a is b)  # 输出 False
print(a == b)  # 输出 True 

a = [1, 2, 3]
b = a
print(a is b)  # 输出 True 
print(a == b)  # 输出 True 
```

官方文档中说 is 表示的是对象标示符（object identity），而 == 表示的是相等（equality）。is 的作用是用来检查对象的标示符是否一致，也就是比较两个对象在内存中的地址是否一样，而 == 是用来检查两个对象是否相等。

我们在检查 a is b 的时候，其实相当于检查 id(a) == id(b)。而检查 a == b 的时候，实际是调用了对象 a 的 `__eq()__`方法，a == b 相当于 `a.__eq__(b)`。

一般情况下，如果 a is b 返回True的话，即 a 和 b 指向同一块内存地址的话，a == b 也返回True，即 a 和 b 的值也相等。

接下来再看下面的代码

```python
a = "hello"
b = "hello"
print(a is b)  # 输出 True 
print(a == b)  # 输出 True

a = "hello world"
b = "hello world"
print(a is b)  # 输出 False
print(a == b)  # 输出 True 
```

这里我们又发现hello使用is判断为True，但hello world判断为False。这就和python中的小整数对象池和大整数对象池和intern机制有关系了。

### 小整数对象池

python解释器为了提升运行速度使用了小整数对象池，来避免为整数频繁定义与销毁内存空间。

小整数对象池定义的范围**[-5,256]**在这个范围的整数对象都是提前创建好的，不会被垃圾回收机制以内存垃圾回收，在这个范围内的整数使用的都是同一个对象(引用 id内存地址值相同)

```shell
>>> a=-5
>>> b=-5
>>> a is b
True
>>> a=256
>>> b=256
>>> a is b
True
```

### 大整数对象池

大整数对象池是在上述范围之外

```shell
>>> a=-6
>>> b=-6
>>> a is b
False
>>> a=257
>>> b=257
>>> a is b
False
```

### intern机制

如果两个或以上的字符串变量它们的值相同且仅由数字字母下划线组成并且长度不超过20个字符，或者值仅含有一个字符时，内存空间中只创建一个对象来让这些变量都指向该内存地址（共享引用），当字符串不满足该条件时，相同值的字符串变量在创建时都会申请一个新的内存地址来保存值。

```shell
# 单个字符或者为空以及空格的字符串，是同一个对象，共享引用

// 两个相同的单字符
>>> a = 'a'
>>> b = 'a'
>>> id(a), id(b)
(140478486488208, 140478486488208)
>>> a is b
True
// 两个相同的空字符串
>>> a = '' 
>>> b = ''
>>> id(a), id(b)
(140377617439408, 140377617439408)
>>> a is b
True
// 两个相同空格的字符串
>>> a = ' '
>>> b = ' '
>>> id(a), id(b)
(140377616103608, 140377616103608)
>>> 
>>> a is b
True
# 两个相同值的字符串，是同一个对象，共享引用
>>> a = 'liunx'
>>> b = 'liunx'
>>> id(a), id(b)
(140478486363976, 140478486363976)
>>> a is b
True
# 两个相同值但长度不超过20位的字符串，是同一个对象，共享引用
>>> a = 'linux' * 4
>>> b = 'linux' * 4
>>> id(a), id(b)
(140478486391736, 140478486391736)
>>> a is b
True
# 两个相同值但长度超过20位的字符串，不是同一个对象
>>> a = 'linux' * 5
>>> b = 'linux' * 5
>>> id(a), id(b)
(140478486354640, 140478486355280)
>>> a is b
False
# 两个相同值但包含特殊字符串(非大小写字母和数字或下划线)的，不是同一个对象
>>> a = '**'
>>> b = '**'
>>> a is b
False
>>> a = '_-'
>>> b = '_-'
>>> a is b
False
>>> a = 'a*c'
>>> b = 'a*c'
>>> a is b
False
>>> a = 'lin ux'
>>> b = 'lin ux'
>>> id(a), id(b)
(140478486364088, 140478485729384)
>>> a is b
False
```

以上是在交互模式下的结果，但是在pycharm中运行如下代码得到的结果会是True

```python
c = "hello world"
d = "hello world"
print(c == d)
True
print(c is d)
True
```

说明在交互模式下和在文件py中也还是存在区别，或者也可能是pycharm这个IDE引起的。
