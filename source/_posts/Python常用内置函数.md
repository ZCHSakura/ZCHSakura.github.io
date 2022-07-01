---
title: Python常用内置函数
date: 2020-05-29 22:09:00
tags: [Python]
categories:
- 技术
- Python
---

记录Python常用的内置函数

<!--more-->

Python给我们内置了大量功能函数，官方文档上列出了69个，有些是我们是平时开发中经常遇到的，也有一些函数很少被用到，这里列举被开发者使用最频繁的8个函数以及他们的详细用法

![img](https:////upload-images.jianshu.io/upload_images/15801507-d86739dc54bc18fa.jpg!web?imageMogr2/auto-orient/strip|imageView2/2/w/550/format/webp)

### print()
print函数是你学Python接触到的第一个函数，它将对象输出到标准输出流，可将任意多个对象打印出来，函数的具体定义：

```python
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False) 
```

objects 是可变参数，所以你可以同时将任意多个对象打印出来

```python
>>> print(1,2,3)  
1 2 3 
```

默认使用空格分隔每个对象，通过指定sep参数可以使用逗号分隔

```python
>>> print(1,2,3, sep=',')  
1,2,3 
```

对象默认输出的是标准输出流，你也可以将内容保存到文件中

```python
>>> print(1,2,3, sep=',', file=open("hello.txt", "w")) 
```



### isinstance()

 可以用 isinstance 函数判断某个对象是否属于某个类的实例，函数的定义

```python
isinstance(object, classinfo) 
```


classinfo 既可以是单个类型对象，也可以是由多个类型对象组成的元组，只要object的类型是元组中任意一个就返回True，否则返回False

```python
>>> isinstance(1, (int, str))  
True  
>>> isinstance("", (int, str))  
True  
>>> isinstance([], dict)  
False 
```



### range()

range函数是个工厂方法，用于构造一个从[start, stop) （不包含stop）之间的连续的不可变的整数序列对象，这个序列功能上和列表非常类似，函数定义：

```python
range([start,] stop [, step]) -> range object 
```

- start 可选参数，序列的起点，默认是0
- stop 必选参数，序列的终点（不包含）
- step 可选参数，序列的步长，默认是1，生成的元素规律是 r[i] = start + step*i

生成0~5的列表

```python
>>> range(5)  
range(0, 5)     
>>> list(range(5))  
[0, 1, 2, 3, 4]
```

默认从0开始，生成0到4之间的5个整数，不包含5，step 默认是1，每次都是在前一次加1

如果你想将某个操作重复执行n遍，就可以使用for循环配置range函数实现

```python
>>> for i in range(3):  
...     print("hello python")  
...  
hello python  
hello python  
hello python 
```

步长为2

```python
>>> range(1, 10, 2)  
range(1, 10, 2)  
>>> list(range(1, 10, 2))  
[1, 3, 5, 7, 9] 
```

起点从1开始，终点10，步长为2，每次都在前一个元素的基础上加2，构成1到10之间的奇数。



### enumerate()

用于枚举可迭代对象，同时还可以得到每次元素的下表索引值，函数定义：

```python
enumerate(iterable, start=0) 
```

例如：

```python
>>> for index, value in enumerate("python"):  
...     print(index, value)  
...  
0 p  
1 y  
2 t  
3 h  
4 o  
5 n 
```

index 默认从0开始，如果显式指定参数start，下标索引就从start开始

```python
>>> for index, value in enumerate("python", start=1):  
...     print(index, value)  
...  
1 p  
2 y  
3 t  
4 h  
5 o  
6 n 
```

如果不使用enumerate函数，要获取元素的下标索引，则需要更多的代码：

```python
def my_enumerate(sequence, start=0):  
    n = start  
    for e in sequence:  
        yield n, e  
        n += 1  
```

```python
>>> for index, value in my_enumerate("python"):  
    print(index, value)  
0 p  
1 y  
2 t  
3 h  
4 o  
5 n 
```



### len

len 用于获取容器对象中的元素个数，例如判断列表是否为空可以用 len 函数

```python
>>> len([1,2,3])  
3  
>>> len("python")  
6  
>>> if len([]) == 0:  
        pass 
```

并不是所有对象都支持len操作的，例如：

```python
>>> len(True)  
Traceback (most recent call last):  
  File "<stdin>", line 1, in <module>  
TypeError: object of type 'bool' has no len() 
```

除了序列对象和集合对象，自定义类必须实现了 **len** 方法能作用在len函数上



### reversed()

reversed() 反转序列对象，你可以将字符串进行反转，将列表进行反转，将元组反转

```python
>>> list(reversed([1,2,3]))
[3, 2, 1] 
```



### open()

open 函数用于构造文件对象，构建后可对其进行内容的读写操作

```python
open(file, mode='r', encoding=None) 
```

读操作

```python
# 从当前路径打开文件 test.txt， 默认以读的方式  
>>>f = open("test.txt")  
>>>f.read()  
... 
```

有时还需要指定编码格式，否则会遇到乱码

```python
f = open("test.txt", encoding='utf8') 
```

写操作

```python
>>>f = open("hello.text", 'w', encoding='utf8')  
>>>f.write("hello python")) 
```

文件中存在内容时原来的内容将别覆盖，如果不想被覆盖，直接将新的内容追加到文件末尾，可以使用 a 模式

```python
f = open("hello.text", 'a', encoding='utf8')  
f.write("!!!") 
```



### sorted()

sroted 是对列表进行重新排序，当然其他可迭代对象都支持重新排放，返回一个新对象，原对象保持不变

```python
>>> sorted([1,4,2,1,0])  
[0, 1, 1, 2, 4] 
```
