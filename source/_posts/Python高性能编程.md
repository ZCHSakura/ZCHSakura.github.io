---
title: Python高性能编程
tags:
  - Python
  - Cython
  - numpy
categories:
  - 工程实践
  - 经验积累
top: 20
date: 2022-04-07 22:03:09
---

# Python 高性能编程



<!--more-->

## 性能分析

### 时间分析

1. 使用修饰器来自动测量时间

   ```python
   from functools import wraps
   import time
   
   def timefn(fn):
       @wraps(fn)
       def measure_time(*args, **kwargs):
           t1=time.time()
           result=fn(*args, **kwargs)
           t2=time.time()
           print("@timefn:"+fn.__name__+" took "+str(t2-t1)+" seconds")
           return result
       return measure_time
   ```

   在 IPython 内部，可以使用 %timeit 魔法函数来对函数运行进行计时

   ```python
   %timeit <function>
   ```

   

2. 使用 UNIX 的 time 命令进行计时

   ```shell
   $ /usr/bin/time -p (--verbose) python <script-name>
   ```

   

3. 使用 cProfile 模块

   cProfile 是一个标准库内建的分析工具。它钩入 CPython 的虚拟机来测量每一个函数运行花费的时间。在要分析的函数前加上 `@profile`装饰器，运行以下命令进行性能分析

   ```shell
   $ python -m cProfile -s cumulative <script-name> <options>
   ```

   也可以将分析结果保存到一个统计文件中使用其他工具进行分析

   ```shell
   $ python -m cProfile -o <outputfilename> <script-name> <options>
   ```

   

4. 使用 line_profiler 进行逐行分析

   安装：`pip install line_profiler`

   在要分析的函数前加上 `@profile`装饰器，运行以下命令进行性能分析

   ```shell
   $ kernprof -l -v <script-name>
   ```



### 空间分析

1. 使用memory_profiler 分析内存使用量

   安装：`pip install memory_profiler`

   在要分析的函数前加上 `@profile`装饰器，运行以下命令进行性能分析

   ```shell
   $ python -m memory_profiler <script-name>
   ```

   使用 `mprof` 可以绘图查看内存使用量，分为以下两个步骤：

   使用 `mprof` 运行程序，运行后在目录下会生成一个 `mprof*.dat` 文件

   ```shell
   $ mprof run <script-name>
   ```

   绘图

   ```shell
   $ mprof plot
   ```

   

## 列表和元组

列表和元组之类的数据结构被称为数组，一个数组是数据在某种内在次序下的扁平列表。

列表和元组的主要区别如下：

- 列表是动态的数组，可变且可以重设长度。元组是静态的数组，其不可变且内部数据一旦创建便无法改变。
- 元组缓存于Python运行时的环境中，这意味着每次使用元组时无需访问内核来分配内存。
- 列表相比于元组需要更多的内存开销。



### 列表

列表的查询操作是 O(1) 的。

列表的搜索操作默认方法 (`list.index()`) 为线性操作，最差情况下为 O(n)。通过对列表进行排序可以将搜索时间降至 O(log n) (采用二分查找)。

 Python 列表有一个内建的排序算法使用了 Tim 排序。Tim 排序在最佳的情况下以 O(n) (最差情况下为 O(n log n)) 的复杂度进行排序。其混用了插入排序和归并排序来达到这样的性能(对于给定数据通过探测法来猜测哪个算法性能更优)。

`bisect` 模块可以在列表进行插入操作时，保持列表的顺序不变 (默认从小到大)。`bisect` 的部分文档说明如下[2]：

> - `bisect.bisect_left(a, x, lo=0, hi=len(a), *, key=None)`
>
>   在列表 a 中找到 x 的左侧插入点
>
> - `bisect.bisect_right(a, x, lo=0, hi=len(a), *, key=None)`
>
>   在列表 a 中找到 x 的右侧插入点
>
> - `bisect.bisect(a, x, lo=0, hi=len(a), *, key=None)`
>
>   同 `bisect.bisect_left`
>
>   
>
> - `bisect.insort_left(a, x, lo=0, hi=len(a), *, key=None)`
>
>   按照排序顺序将 x 插入到列表 a 中，插入位置由`bisect.bisect_left`决定
>
> - `bisect.insort_right(a, x, lo=0, hi=len(a), *, key=None)`
>
>   按照排序顺序将 x 插入到列表 a 中，插入位置由`bisect.bisect_right`决定
>
> - `bisect.insort(a, x, lo=0, hi=len(a), ***, key=None)`
>
>   同`bisect.insort_right`
>
>   
>
> **性能说明**
>
> 当使用 `bisect()` 和 `insort()` 编写时间敏感的代码时，请记住以下概念。
>
> - 二分法对于搜索一定范围的值是很高效的。 对于定位特定的值，则字典的性能更好。
> - `insort()` 函数的时间复杂度为 O(n) 因为对数时间的搜索步骤被线性时间的插入步骤所主导。
> - 这些搜索函数都是无状态的并且会在它们被使用后丢弃键函数的结果。 因此，如果在一个循环中使用搜索函数，则键函数可能会在同一个数据元素上被反复调用。 如果键函数速度不快，请考虑用 [`functools.cache()`](https://docs.python.org/zh-cn/3/library/functools.html#functools.cache) 来包装它以避免重复计算。 另外，也可以考虑搜索一个预先计算好的键数组来定位插入点（如下面的示例节所演示的）。

列表在分配内存时会添加一定的预留空间。当一个大小为 N 的列表需要添加新的元素时，Python 会创建一个新的列表，足够存放原来的N个元素以及额外添加的元素。在实际分配时，分配的大小并不是 N+1，而是 N+M 个。其中，M 的计算公式如下：

M = (N >> 3) + (N < 9 ? 3 : 6) + 1

|  N   |  0   | 1-4  | 5-8  | 9-16 | 17-25 | 26-35 | ...  | 991-1120 |
| :--: | :--: | :--: | :--: | :--: | :---: | :---: | :--: | :------: |
| N+M  |  0   |  4   |  8   |  16  |  25   |  35   | ...  |   1120   |

当一个list长度为8时，发生append操作后：

1. new_size = 原有的size ＋ append一个对象 = 8 + 1 = 9
2. new_size为9，二进制是1001，9 >> 3 = 1
3. new_allocated = 9 >> 3 + 6 = 7
4. new_allocated += new_size，为9 + 7 ＝ 16
5. 列表的最终大小为16

### 元组

元组固定且不可变，其一旦被创建，内容便无法被修改，大小也不可被改变。

元组的静态特性的一个好处是占用了更少的内存。

元组的静态特性的另一个好处体现在一些会发生在 Python 后台的事：资源缓存。Python 是一门垃圾收集语言，这意味着当一个变量不再被使用时，Python 会将该变量使用的内存释放回操作系统。然而，对于长度为1-20的元组，即使它们不再被使用，其空间也不会立即返回给系统，而是留待未来使用。这意味着新建元组时可以直接在预留内存中存放数据，避免与操作系统进行交互。



## 字典与集合

字典是 key 和 value 的组合

集合是 key 的组合

- 字典和集合基于 key 的查询速度是 O(1) 的，但是相应的，字典和集合会占用更多的内存。
- 字典和集合的插入和查询操作依赖于散列函数，因此实际的计算速度取决于其使用的散列函数
- 字典和集合的默认最小长度为8，每次改变大小时，桶的个数增加到原来的4倍，直到5000个元素之后，每次增加到原来的2倍。

Python 的命名空间使用字典来进行管理。当Python 访问一个变量，函数或模块时，有一个体系决定去哪里寻找。

- 首先，Python 查找 `locals()` 数组，其内部保存了所有本地变量的条目。Python 对本地变量的查询做了很多优化，同时这也是整个查询过程中唯一不需要进行字典查询的部分。
- 如果不在本地数组里，那么就会查找 `globals()` 字典。如果找不到则会搜索 `__buildin__`对象（模块对象，在搜索 `__buildin__`的一个属性时，其实是在搜索它的 `locals()`字典）

```python
import math
from math import sin

def f1(x):
    '''
    %timeit f1(10)
    154 ns ± 3.93 ns per loop
    '''
    return math.sin(x)

def f2(x):
    '''
    %timeit f2(10)
    126 ns ± 0.974 ns per loop
    '''
    return sin(x)

def f3(x,sin=sin):
    '''
    %timeit f3(10)
    138 ns ± 1.85 ns per loop
    '''
    return sin(x)

```

> 将 sin 函数作为本地函数慢于直接调用 global 中的 sin。此处存疑！



## 迭代器和生成器

Python 中的 for 循环要求被循环的对象支持迭代。使用 Python 内建的 `iter()` 函数便可以生成迭代器。对于列表，元组，字典这些对象，其会返回一个由元素或者 key 组成的迭代器。对于复杂的对象，则返回对象内部的 `__iter__` 属性。

对于一个迭代器，可以调用 `__next__()` 来获取新值，直到 `StopIteration` 异常被抛出。

迭代器只有在被调用时才会进行计算，因此在对大量数据流处理时可以显著节省内存。标准库中的 `itertools` 库提供了很多便于迭代器使用的工具：

- `imap, ireduce, ifilter, izip`
- `islice` 允许对一个无穷生成器切片
- `chain` 将多个生成器链接在一起
- `takewhile` 给生成器添加一个终止条件
- `cycle` 通过不断重复将一个有穷生成器变为无穷



## 参考文献

[1]M. Micha, I. Ozsvald, S. Hu, and X. Xu, *Python高性能编程 / Python gao xing neng bian cheng*. 人民邮电出版社, Beijing: Ren Min You Dian Chu Ban She, 2017.

[2]“bisect --- 数组二分查找算法 — Python 3.10.4 文档,” *docs.python.org*. https://docs.python.org/zh-cn/3/library/bisect.html (accessed Apr. 07, 2022).
