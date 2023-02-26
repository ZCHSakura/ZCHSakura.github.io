---
title: python赋值&浅拷贝&深拷贝
date: 2023-02-27 00:46:25
tags:
  - python
categories:
  - 技术
  - Python
top:
---

记录Python中赋值&浅拷贝&深拷贝和区别，同时还有list.append()时浅拷贝的问题

<!--more-->

## 赋值

在python中，对象的赋值就是简单的对象引用，这点和C++是不同的

如下例子：

```python
a = ['a', 'b', 'c']
b = a   # 采用简单的=赋值
print(a==b)

# 下面是输出结果：
True
```

这种情况下，b和a是一样的，他们指向同一片内存，b不过是a的别名，是引用。我们可以使用a与b是否相同来判断，返回`True`，表明他们地址相同，内容相同。

赋值操作(包括对象作为参数、返回值)不会开辟新的内存空间，它只是复制了新对象的引用。也就是说，除了b这个名字以外，没有其它的内存开销。

修改了a，就影响了b；同理，修改了b就影响了a。下面的例子尝试对b进行修改，在后面加入新的元素’d’，通过观察输出结果发现：在修改列表b的同时，列表a也会被修改，因为两者用的是同一个内存空间。

```python
a = ['a', 'b', 'c']
b = a
b.append('d')
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', 'd']
b = ['a', 'b', 'c', 'd']
```

## 浅拷贝

浅拷贝会创建新对象，其内容是原对象的引用。

浅拷贝有三种形式：切片操作，工厂函数，`copy`模块中的copy函数。

**比如对上述a：**

- 1、切片操作：b = a[:] 或者 b = [each for each in a]
- 2、工厂函数：b = list(a)
- 3、copy函数：b = copy.copy(a) #使用时要import copy模块

浅拷贝产生的b不再是a了，使用is可以发现他们不是同一个对象，使用id查看，发现它们也不指向同一片内存。但是当我们使用 id(x) for x in a 和 id(x) for x in b 时，可以看到二者包含的元素的地址是相同的。

在这种情况下，a和b是不同的对象，修改b理论上不会影响a。比如b.append([4,5])。

**代码效果如下：**

```python
a = ['a', 'b', 'c', ['yellow', 'red']]
b = a[:]  # 采用了切片操作对列表b进行赋值
b.append('green') # 对列表b执行添加元素操作
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', ['yellow', 'red']]  # a中的元素不发生变化
b = ['a', 'b', 'c', ['yellow', 'red'], 'green']  # b中增加了一个元素'green'
```

**但是要注意：**浅拷贝之所以称为浅拷贝，是它仅仅只拷贝了一层，在a中有一个嵌套的list，如果我们修改了它，情况就不一样了。

a[3].append(“blue”)。查看b，你将发现b也发生了变化。这是因为，你修改了嵌套的list。修改外层元素，会修改它的引用，让它们指向别的位置，修改嵌套列表中的元素，列表的地址并为发生变化，指向的都是同一个位置。

**代码如下：**

```python
a = ['a', 'b', 'c', ['yellow', 'red']]
b = a[:]  # 采用了切片操作对列表b进行赋值
a[3].append('blue')  # 在a列表中的第3个元素中增加元素'blue'，由于a[3]本身也是一个列表，从而是在列表后增加了元素'blue'，从输出结果中可以看出来。
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', ['yellow', 'red', 'blue']]
b = ['a', 'b', 'c', ['yellow', 'red', 'blue']]
```

## 深拷贝

深拷贝只有一种形式，`copy`模块中的`deepcopy`函数。

和浅拷贝对应，深拷贝拷贝了对象的所有元素，包括多层嵌套的元素。因而，它的时间和空间开销要高。

同样对la，若使用`b = copy.deepcopy(a)`，再修改b将不会影响到a了。即使嵌套的列表具有更深的层次，也不会产生任何影响，因为深拷贝出来的对象根本就是一个全新的对象，不再与原来的对象有任何关联。

**实例代码如下：**

```python
import copy
a = ['a', 'b', 'c', ['yellow', 'red']]
b = copy.deepcopy(a)   # 采用深拷贝对a进行深拷贝操作
b.append('xyz')
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', ['yellow', 'red']]   # 使用深拷贝，对b的修改不会影响到a
b = ['a', 'b', 'c', ['yellow', 'red'], 'xyz']
```

**或者用下面的代码：**

```python
import copy
a = ['a', 'b', 'c', ['yellow', 'red']]
b = copy.deepcopy(a)   # 采用深拷贝对a进行深拷贝操作
a[3].append('crazy')
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', ['yellow', 'red', 'crazy']]  
b = ['a', 'b', 'c', ['yellow', 'red']]   # 对a的修改不会影响到b
```

**或者用下面的代码：**

```python
import copy
a = ['a', 'b', 'c', ['yellow', 'red']]
b = copy.deepcopy(a)   # 采用深拷贝对a进行深拷贝操作
a[3].append('crazy')
b.append('dddd')
print('a = {}'.format(a))
print('b = {}'.format(b))

# 下面是输出结果：
a = ['a', 'b', 'c', ['yellow', 'red', 'crazy']]
b = ['a', 'b', 'c', ['yellow', 'red'], 'dddd']
```

## 关于拷贝操作的提醒

- 对于非容器类型，如数字，字符，以及其它“原子”类型，没有拷贝一说。产生的都是原对象的引用。
- 如果元组变量值包含原子类型对象，即使采用了深拷贝，也只能得到浅拷贝。

## list.append()的浅拷贝问题

Python中的append方法是一个常用的方法，可以将一个对象添加到列表末尾，这里面可以存在一个浅拷贝的大坑！

```shell
>>> a = [1, 3, 5, "a"]
>>> b = []
>>> b.append(a)
>>> b
[[1, 3, 5, 'a']]
>>> a.append("aha")
>>> b    # surprise?
[[1, 3, 5, 'a', 'aha']]
```

事实上，append方法是浅拷贝。在Python中，对象赋值实际上是对象的引用，当创建一个对象，然后把它赋值给另一个变量的时候，Python并没有拷贝这个对象，而只是拷贝了这个对象的引用，这就是浅拷贝。

我们逐步来看。首先，b.append(a)就是对a进行了浅拷贝，结果为b=[[1, 3, 5, 'a']]，但b[0]与a引用的对象是相同的，这可以通过id函数进行验证：

```shell
>>> id(b[0])
3145735177480
>>> id(a)
3145735177480
```

所以，在日常使用append函数的时候，就需要将浅拷贝变为深拷贝（其实也不能叫深拷贝，应该是双重浅拷贝，可以看上面浅拷贝的内容，只拷贝一层），有两个解决方案：

- b.append(list(a))
- b.append(a[:])
