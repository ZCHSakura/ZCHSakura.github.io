---
title: leetcode_1
date: 2020-06-30 20:13:05
tags: [python, 算法]
top: 1000
categories:
- 算法
---

这里记录在leetcode上做题的经过，包含自己的解法和优秀解法

<!--more-->

## p1_两数之和

给定一个整数数组`nums`和一个目标值`target`，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

> 示例:
>
> 给定 nums = [2, 7, 11, 15], target = 9
>
> 因为 nums[0] + nums[1] = 2 + 7 = 9
>
> 所以返回 [0, 1]

### mine

强行搜索

```python
def two_sum(nums, target):
    j = -1
    for i in range(len(nums)):
        num2 = target - nums[i]
        if num2 in nums:
            if nums.index(num2) == i and nums.count(num2) == 1: #防止两个相同的数字出现
                continue
            else:
                j = nums.index(num2, i+1)
                break
    if j >= 0:
        return [i, j]
    else:
        return []
```

### others

解题思路是在方法一的基础上，优化解法。想着，`num2` 的查找并不需要每次从 `nums` 查找一遍，只需要从 `num1 `位置之前或之后查找即可。但为了方便 index 这里选择从 `num1 `位置之前查找：

```python
def twoSum(nums, target):
    lens = len(nums)
    j=-1
    for i in range(1,lens):
        temp = nums[:i]
        if (target - nums[i]) in temp:
            j = temp.index(target - nums[i])
            break
    if j>=0:
        return [j,i]
```

参考了大神们的解法，通过哈希来求解，这里通过字典来模拟哈希查询的过程。
个人理解这种办法相较于方法一其实就是字典记录了 `num1` 和 `num2` 的值和位置，而省了再查找`num2` 索引的步骤。

```python
def twoSum(nums, target):
    hashmap={}
    for ind,num in enumerate(nums):
        hashmap[num] = ind
    for i,num in enumerate(nums):
        j = hashmap.get(target - num)
        if j is not None and i!=j:
            return [i,j]
```

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
enumerate(sequence, [start=0])

>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

遍历列表同时查字典，只用遍历一遍

```python
def two_sum(nums, target):
    """这样写更直观，遍历列表同时查字典"""
    dct = {}
    for i, n in enumerate(nums):
        if target - n in dct:
            return [dct[target - n], i]	# 由于我们是一边构建字典一边查找所以查找到的时候目前这个i应该在后面
        dct[n] = i #这句不能放在if语句之前，解决list中有重复值或target-num=num的情况
```

### summary

1. enumerate的用法，将一个可遍历对象转化为索引序列
2. `if target - n in dct`判断的是dct的key值
3. 在不确定字典里面是否包含该key值时应该使用get，直接取值会报错。`hashmap.get(target - num)`

## p1521_两个栈实现队列

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

示例 1：

```python
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：
[null,null,3,-1]
```

示例 2：

```python
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：
[null,-1,null,null,5,2]
```


提示：

- 1 <= values <= 10000
- 最多会对 appendTail、deleteHead 进行 10000 次调用

### mine

~~说实话，python基础还没学，不是很会，就很尴尬~~

### others

**解题思路**

* 栈无法实现队列功能： 栈底元素（对应队首元素）无法直接删除，需要将上方所有元素出栈。
* 双栈可实现列表倒序： 设有含三个元素的栈 A = [1,2,3] 和空栈 B = []。若循环执行 A 元素出栈并添加入栈 B ，直到栈 A 为空，则 A = [], B = [3,2,1]，即 栈 B 元素实现栈 A 元素倒序 。
* 利用栈 B 删除队首元素： 倒序后，B 执行出栈则相当于删除了 A 的栈底元素，即对应队首元素。

**复杂度分析：**

> 由于问题特殊，以下分析仅满足添加 N 个元素并删除 N 个元素，即栈初始和结束状态下都为空的情况。

* 时间复杂度： appendTail()函数为 O(1) ；deleteHead() 函数在 N 次队首元素删除操作中总共需完成 N 个元素的倒序。
* 空间复杂度 O(N) ： 最差情况下，栈 A 和 B 共保存 N 个元素。

**实现**

```python
class CQueue:
    def __init__(self):
        # A用来入栈，B用来出栈
        self.A, self.B = [], []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        # 如果B是不是空的，那么将B栈顶弹出，即队首
        if self.B:
            return self.B.pop()
        # 如果B已经空了，而A也空了，那么说明队列中没有元素了
        if not self.A:
            return -1
        # 如果B空了，但是A没空，那么把A一个个取除并压入B，则B的栈顶为队首
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()
```

### summary

1. 主要是两个栈，一个做入队，一个做出队
2. 入队直接在入队的那个栈顶入栈即可
3. 出队需要判断两个栈的情况，分别讨论

## p718_最长重复子数组

给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

示例：

```python
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
```

解释：

长度最长的公共子数组是 [3, 2, 1] 。


提示：

- 1 <= len(A), len(B) <= 1000
- 0 <= A[i], B[i] < 100

### mine

暴力方法其实很简单，只要两个for循环，两个数组中的元素一个个比较，比较通过了就比较下一位，不通过就是下一次for，只要注意防止数组越界就可以了。

```python
def findLength(self, A, B) -> int:
    result = 0
    for i in range(len(A)):
        for j in range(len(B)):
            if A[i] == B[j]:
                temp = 1
                # A[i+temp] == B[j+temp]一定要放在后面，不然一旦越界就报错
                while i+temp < len(A) and j+temp < len(B) and A[i+temp] == B[j+temp]:
                    temp += 1
                result = max(result, temp)
    return result
```

但是最大的问题就是效率太低了，数据短还可以，一旦数据量大就很难处理

### others

**滑动窗口**

![bc7d3a75a57f9abd6d1f6d789e176af0ab65f5522f6c7119178b073c67ae6494-leetcode-718-lcs-window](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200701222958.gif)

这个图很形象的展示出了什么是滑动窗口，有一个问题就是窗口的获取，python中的zip函数可以很好的解决这个问题

```python
class Solution:
    def findLength(self, A, B) -> int:
        def getMaxLen(A, B):
            result = temp = 0
            for a, b in zip(A, B):
                if a == b:
                    temp += 1   # 相同加1
                else:
                    result = max(result, temp)   # 不同时把临时赋给真正，临时清零，进行下项
                    temp = 0
            return max(result, temp)    # 不能只返回result，不然当最大temp出现在末尾不会被记录

        ans = 0
        # B不动A往左滑
        for ai in range(len(A)):  # 求 A[ai]对齐B[0] 的情况
            ans = max(ans, getMaxLen(A[ai:], B))
        # A不动B往左滑
        for bi in range(len(B)):  # A[0]对齐B[0]的已经做过了，故初始bi=1
            ans = max(ans, getMaxLen(A, B[bi:]))
        return ans
```

### summary

1. zip函数的用法

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```

## p378_有序矩阵中第k小的元素

给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。

 示例：

```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
]
k = 8,

返回 13。
```

提示：

- 你可以假设 k 的值永远是有效的，1 ≤ k ≤ n^2 。

### mine

最简单的直接解法就是将matrix中的每一个元素（列表）合并成为一个列表，然后再对这个列表排序取第k个元素即可，有趣的是可以使用`sum`将一个可迭代对象求和，但是要注意第二个参数位置要放上[]，不然默认是int类型的求和，放上一个[]就能变为列表的求和。

```python
class Solution:
    def kthSmallest(self, matrix, k) -> int:
        res = sorted(sum(matrix, []))
        return res[k-1]
```

### others

**归并排序（堆）**

首先，我们先不考虑对堆的使用，我先来概括一下使用这个方法的最关键思路：

在整个矩阵中，每次弹出矩阵中最小的值，第k个被弹出的就是我们需要的数字。
 
现在我们的目的很明确：每次弹出矩阵中最小的值。

当我们看到下面这个有序矩阵时，我们知道左上角的数字是整个矩阵最小的，但弹出它后我们如何保证接下来每一次都还能找到全矩阵最小的值呢？

![1e9354f41d0e82d81d6be6538cc7b285d31418c5f14ed7937cee5765e20b8d76-屏幕快照 2020-07-02 下午6.38.01](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702201325.png)

其实解决这个问题的关键，在于维护一组“最小值候选人”：

你需要保证最小值必然从这组候选人中产生，于是每次只要从候选人中弹出最小的一个即可。

我们来选择第一组候选人，在这里可以选择第一列，因为每一个数字都是其对应行的最小值，全局最小值也必然在其中。

![299edf92ce9acad73613ff76037c8e2c1ec4a53737b5e7ac02fe536d0713dae6-屏幕快照 2020-07-02 下午6.38.13](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702201432.png)

第一次弹出很简单，将左上角的1弹出即可。

1弹出之后，我们如何找到下一个候选人呢？

![c9980b3010f6696231c7caa016303dc815b35b14347d20ee7980fff537c30750-屏幕快照 2020-07-02 下午6.38.50](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702201455.png)


其实非常简单，刚才弹出的位置右移一格就行了，这样不是还是能保证候选人列表中每一个数字是每一行的最小值吗，那全局最小值必然在其中！

我们每次弹出候选人当中的最小值，然后把上次弹出候选人的右边一个补进来，就能一直保证全局最小值在候选人列表中产生，

示例：（穿蓝色衣服的为候选人）

（顺序是每一行都是从左向右看）(当某一行弹到没东西，候选人列表的长度就会少1)

![d550255f65fd12fb57130240046c176165c19689add0e817ccca5f88dc9340df-屏幕快照 2020-07-02 下午6.41.52](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702201510.png)

------

###################### 堆(HEAP) ######################

要具体实现这个过程，我们需要什么呢？

我们需要每次帮我们管理候选人的工具(能弹出一个候选人，还能加入一个候选人)，它就是堆了。
 
堆这个数据结构，它保证每一个父节点>=或者<=其子节点。

如果每个父节点>=其子节点，那这就是一个Max Heap

如果每个父节点<=其子节点，那这就是一个Min Heap




在这里我们需要的是Min Heap，一般Heap是二元的，也就是说每个爸爸有两个儿子，一个二元Min Heap可以参考这个图：

![2d2ea402d1a101c2186f00008fb77e98b4ecbd14f2daf633076cbb6abd20ec9b-屏幕快照 2020-07-02 下午2.29.11](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702201624.png)

(图源：https://www.youtube.com/watch?v=wptevk0bshY)

可以看到最顶上的总是最小的数字
 
其实对于这道题，你只需要知道Heap的两个操作对我们来说很实用：

1.弹出最顶上的数字（弹出候选人）

2.加入一个新数字（加入新的候选人）

------

################ Python中的heapq #################

接下来，讲一下Python中的heapq模块如何实现上述1、2两个操作，不需要的同学可以不看这一part

0.创建一个heap

1.弹出最顶上的数字（弹出候选人）

2.加入一个新数字（加入新的候选人）

0.创建一个heap

```python
heapq.heapify(x)
把List x变为一个heap
```

1.弹出最顶上的数字（弹出候选人）

```python
heapq.heappop(heap)
从heap中弹出一个候选人，并返回弹出的东西
```

2.加入一个新数字（加入新的候选人）

```python
heapq.heappush(heap,item)
在heap中加入一个新的候选人item
```


整个代码的注释：

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix) #注：题目中这个矩阵是n*n的，所以长宽都是n
        pq = [(matrix[i][0], i, 0) for i in range(n)] #取出第一列候选人
        #matrix[i][0]是具体的值，后面的(i,0)是在记录候选人在矩阵中的位置，方便每次右移添加下一个候选人

        heapq.heapify(pq) #变成一个heap

        for i in range(k - 1): #一共弹k-1次
            num, x, y = heapq.heappop(pq) #弹出候选人里最小一个
            if y != n - 1: #如果这一行还没被弹完
                heapq.heappush(pq, (matrix[x][y + 1], x, y + 1)) #加入这一行的下一个候选人

        return heapq.heappop(pq)[0]	# 弹出第k个
```

---------------

**二分查找**

由题目给出的性质可知，这个矩阵内的元素是从左上到右下递增的（假设矩阵左上角为 `matrix[0][0]`）。以下图为例：

![378_fig1](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702211621.png)

我们知道整个二维数组中` matrix[0][0]`为最小值，`matrix[n-1][n−1]`为最大值，现在我们将其分别记作 l和 r。

可以发现一个性质：任取一个数 mid 满足 l≤mid≤r，那么矩阵中不大于 mid的数，肯定全部分布在矩阵的左上角。

例如下图，取 mid=8：

![378_fig2](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702211753.png)

我们可以看到，矩阵中大于 mid的数就和不大于 mid 的数分别形成了两个板块，沿着一条锯齿线将这个矩形分开。其中左上角板块的大小即为矩阵中不大于 mid 的数的数量。

读者也可以自己取一些 mid 值，通过画图以加深理解。

我们只要沿着这条锯齿线走一遍即可计算出这两个板块的大小，也自然就统计出了这个矩阵中不大于 mid 的数的个数了。

走法演示如下，依然取 mid=8：

![378_fig3](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200702211846.png)

可以这样描述走法：

初始位置在 `matrix[n-1][0]`（即左下角）；

设当前位置为 `matrix[i][j]`。若 `matrix[i][j]≤mid`，则将当前所在列的不大于 mid 的数的数量（即 i + 1）累加到答案中，并向右移动，否则向上移动；

不断移动直到走出格子为止。

我们发现这样的走法时间复杂度为 O(n)，即我们可以线性计算对于任意一个 mid，矩阵中有多少数不大于它。这满足了二分查找的性质。

不妨假设答案为 x，那么可以知道` l≤x≤r`，这样就确定了二分查找的上下界。

每次对于「猜测」的答案 mid，计算矩阵中有多少数不大于mid ：

- 如果数量不少于 k，那么说明最终答案 x 不大于 mid；

- 如果数量少于 k，那么说明最终答案 x 大于 mid。

这样我们就可以计算出最终的结果 x 了。


```python
class Solution:
    def kthSmallest(self, matrix, k: int) -> int:
        n = len(matrix)
        right = matrix[n-1][n-1]
        left = matrix[0][0]

        # 检查该mid左上角的数字是不是大于等于k
        def chick(mid):
            i, j = n-1, 0
            num = 0
            while i >= 0 and j < n:
                # 如果这个数不大于mid，即在mid左上角
                if matrix[i][j] <= mid:
                    num += i + 1
                    j += 1
                else:
                    i -= 1
            # 返回不大于本次mid的数是否大于等于k
            print(num)
            return num >= k

        # 想象一下，mid = 7， k = 5
        # 说明matrix中小于等于7的值只有（0-4）个，
        # 举个例子：
        #     matrix中有{1 2 7 7 8 9}
        #     小于等于7的有{1， 2， 7， 7}
        #     第5大的数字，肯定就是从8开始找咯
        # 那我们要找的是第k小的数字，这个数字肯定要比mid大吧，从mid+1开始找吧
        # 否则，就从小于等于mid的这个部分来找吧
        while left < right:
            mid = (left + right)//2     # 整数除法
            print("left:", left, "mid:", mid, "right:", right)
            # 不大于本次mid的数大于等于k，说明mid需要缩小
            if chick(mid):
                right = mid
            # 不大于本次mid的数小于k，说明mid需要扩大
            else:
                left = mid + 1
        return left
```

### summary

1. sum函数可以将可迭代数据求和

```python
>>>sum([0,1,2])  
3  
>>> sum((2, 3, 4), 1)        # 元组计算总和后再加 1
10
>>> sum([0,1,2,3,4], 2)      # 列表计算总和后再加 2
12
```

2. python中堆的用法

一种著名的数据结构是堆（heap），它是一种优先队列。优先队列让你能够以任意顺序添加对象，并随时（可能是在两次添加对象之间）找出（并删除）最小的元素。相比于列表方法min，这样做的效率要高得多。
实际上，Python没有独立的堆类型，而只有一个包含一些堆操作函数的模块。这个模块名为heapq（其中的q表示队列），它包含6个函数，其中前4个与堆操作直接相关。必须使用列表来表示堆对象本身。

| 函 数                | 描 述                         |
| -------------------- | ----------------------------- |
| heappush(heap, x)    | 将x压入堆中                   |
| heappop(heap)        | 从堆中弹出最小的元素          |
| heapify(heap)        | 让列表具备堆特征              |
| heapreplace(heap, x) | 弹出最小的元素，并将x压入堆中 |
| nlargest(n, iter)    | 返回iter中n个最大的元素       |
| nsmallest(n, iter)   | 返回iter中n个最小的元素       |

## p1470_重新排列数组

给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。

请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。

示例 1：

输入：nums = [2,5,1,3,4,7], n = 3

输出：[2,3,5,4,1,7] 

解释：由于 x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 ，所以答案为 [2,3,5,4,1,7]

示例 2：

输入：nums = [1,2,3,4,4,3,2,1], n = 4

输出：[1,4,2,3,3,2,4,1]

示例 3：

输入：nums = [1,1,2,2], n = 2

输出：[1,2,1,2]

**提示：**

- `1 <= n <= 500`
- `nums.length == 2n`
- `1 <= nums[i] <= 10^3`

### mine

```python
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        result = []
        for i in range(n):
            result.append(nums[i])
            result.append(nums[i+n])
        return result
```

最简单的方法，没啥说的

### others

```python
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        nums[::2], nums[1::2] = nums[:len(nums) // 2], nums[len(nums) // 2:]
        return nums
```

切片赋值。

还有一些使用位运算的算法，直接原地修改。

因为题目限制了每一个元素 nums[i] 最大只有可能是 1000，这就意味着每一个元素只占据了 10 个 bit。（2^10 - 1 = 1023 > 1000）

而一个 int 有 32 个 bit，所以我们还可以使用剩下的 22 个 bit 做存储。实际上，每个 int，我们再借 10 个 bit 用就好了。

因此，在下面的代码中，每一个 nums[i] 的最低的十个 bit（0-9 位），我们用来存储原来 nums[i] 的数字；再往前的十个 bit（10-19 位），我们用来存储重新排列后正确的数字是什么。

在循环中，我们每次首先计算 nums[i] 对应的重新排列后的索引 j，之后，取 nums[i] 的低 10 位（nums[i] & 1023），即 nums[i] 的原始信息，把他放到 nums[j] 的高十位上。

最后，每个元素都取高 10 位的信息(e >> 10)，即是答案。

```c++
class Solution {
public:
    vector<int> shuffle(vector<int>& nums, int n) {

        for(int i = 0; i < 2 * n; i ++){
            int j = i < n ? 2 * i : 2 * (i - n) + 1;
            nums[j] |= (nums[i] & 1023) << 10;
        }
        for(int& e: nums) e >>= 10;
        return nums;
    }
};
```



### summary

- python可以使用最普通的方法，或者使用切片
- 还有空间复杂度为O(1)的方法。

## p998_最大二叉树2

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208301232481.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208301233530.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208301233578.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208301233358.png)

### mine

根据题目内容和示例可以看到基本就是三种情况。一种是插在root上面，让root做val的左子树（其中还有root为None的情况）；第二种是插入到原本树的中间；第三种是做原本树的叶子节点。

同时又根据题目给的构造树的方法和b中val放到最后的情况可以知道，val一定是作为右子节点存在的（除非val做根），原本那个位置的子树作为val的左子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 树原本为空
        if root == None:
            return TreeNode(val, root, None)
        # 树根小于val
        if root.val < val:
            return TreeNode(val, root, None)
        last_cur = None
        cur = root
        while cur:
            # val值小于当前节点则不断往右子树方向寻找
            if cur.val > val:
                last_cur = cur
                cur = cur.right
            # val值大于当前节点则放在当前节点的右子节点处，当前节点作为val节点的左子节点
            elif cur.val < val:
                last_cur.right = TreeNode(val, cur, None)
                return root
        # 遍历到了原本树的叶子节点
        last_cur.right = TreeNode(val, None, None)
        return root
```

### others

递归的方法：

1. 当前节点值<插入值（或者当前节点为空）：插入节点作为当前节点的根，当前节点作为插入节点的左子节点
2. 当前节点值>插入值：插入节点递归插入当前节点的右子树

```python
class Solution:
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root or val > root.val:
            ans = TreeNode(val)
            ans.left = root
            return ans
        
        root.right = self.insertIntoMaxTree(root.right, val)
        return root
```

### summary

- 这个题主要就是要理解题意，分清楚情况，搞清楚这个插入节点的插入规则，之后不管是正常还是递归都思路清晰

## p2_两数相加

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208302028117.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208302028925.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208302028213.png)

### mine

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        cur1 = l1
        cur2 = l2
        # ans是头，将引用传给temp，之后不断操作temp，最后返回ans
        ans = ListNode()
        temp = ans
        add_flag = 0
        while cur1 or cur2:
            # 如果某个链表先走完了，val取值0
            cur1_val = 0 if cur1 == None else cur1.val
            cur2_val = 0 if cur2 == None else cur2.val

            summ = cur1_val + cur2_val + add_flag
            # 计算val值和是否有进位
            if summ > 9:
                summ = summ - 10
                add_flag = 1
            else:
                add_flag = 0
            # 链表往后取值
            if cur1:
                cur1 = cur1.next
            if cur2:
                cur2 = cur2.next
            # 构建返回的答案链表
            temp.next = ListNode(summ)
            temp = temp.next
        # 最后一个进位
        if add_flag == 1:
            temp.next = ListNode(add_flag)
        return ans.next
```

### others

流程都差不多，但是可以优化相加后的值和进位内容

```python
summ = (cur1_val + cur2_val + add_flag)%10
add_flag = 1 if cur1_val + cur2_val + add_flag >= 10 else 0
```

### summary

整体不是太难，主要注意三点：

- python赋值直接是引用，所以直接temp=ans，之后只用不断在temp上加后续节点就行了。而像C这种赋值是传副本的需要直接malloc同一片地址`head = tail = malloc(sizeof(struct ListNode));`
- 判断两个链表是不是都结束了，没结束的才不断next，先结束的后面val直接赋值0
- 用一个变量专门记录进位，如果最后两个值又有进位后面还要再接一个节点

## p946_验证栈序列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208311124041.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208311124889.png)

### mine

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # 模拟一个空栈
        stack = []
        # poped列表index
        j = 0
        for i in pushed:
            # pushed列表内容顺序进栈
            stack.append(i)
            # 当遇到栈顶元素与poped列表指针指向元素相同时将该元素出栈，并把poped指针后移
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        # 如果模拟的栈最后能弹空则说明这个栈序列是对的
        if len(stack) == 0:
            return True
        else:
            return False
```

整体难度不是很高，就是要用一个列表来模拟栈的操作。

### others

```python
return len(stack) == 0
```

返回方式可以再优化下

### summary

难度不大，只要明白栈的工作方式，用一个列表来模拟下就可以了。注意下列表的`pop()`函数的使用。

## p3_无重复字符的最长子串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208312020996.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202208312022288.png)

### mine

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        s_list = list(s)
        max_num = 0
        for i in range(len(s_list)):
            used_list = []
            j = i
            num = 0
            while j < len(s_list) and s_list[j] not in used_list:
                used_list.append(s_list[j])
                num += 1
                j += 1
            max_num = num if num > max_num else max_num
        
        return max_num
```

我这个方法做是完全能做出来，但是开销有点大，看了题解主要有以下几点还可以优化

- 使用set来代替list，每次set不清空，从set左边开始一个个弹出，直到没有重复元素
- 右指针（也就是j）不用每次重复后从左指针处开始，只需要在弹出set中元素的时候不断移动移动左指针就行了
- 不用专门使用一个num来记录长度，可以直接通过两指针之间的距离来计算长度
- 左指针可以不用一位位右移（进阶）

### others

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 哈希集合，记录每个字符是否出现过
        occ = set()
        n = len(s)
        # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        rk, ans = -1, 0
        for i in range(n):
            if i != 0:
                # 左指针向右移动一格，移除一个字符
                occ.remove(s[i - 1])
            while rk + 1 < n and s[rk + 1] not in occ:
                # 不断地移动右指针
                occ.add(s[rk + 1])
                rk += 1
            # 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1)
        return ans
```

官方这个方法也算是中规中矩，优化了我的前三点

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:max_len = cur_len
            lookup.add(s[i])
        return max_len
```

这个方法也挺有意思的，我和官方的方法都是i作为左指针，在for循环里面通过while来增加字符串长度。而这个方法把i作为右指针，在for循环里面通过while来弹出set中的内容。

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        lst = []
        n = len(s)
        ans = 0
        for i in range(n):
            while s[i] in lst:
                del lst[0]  # 队首元素出队
            lst.append(s[i]) # 排除重复元素后 新元素入队
            ans = max(ans, len(lst))
        return ans
```

这和上个方法一样，极简版。

以上三个方法都没有优化到我说的第四点

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        k, res, c_dict = -1, 0, {}
        for i, c in enumerate(s):
            if c in c_dict and c_dict[c] > k:  # 字符c在字典中 且 上次出现的下标大于当前长度的起始下标
                k = c_dict[c]
                c_dict[c] = i
            else:
                c_dict[c] = i
                res = max(res, i-k)
        return res
```

这里相当于用dict模拟了一个哈希Map，i是右指针，k是左指针，当遇到重复的时候可以直接定位到重复元素，不用像set或list那样一个个移动左指针。

### summary

- 移动窗口思想
- 可以使用哈希Map等数据结构来优化左指针的移动，让其不要一位位的移动。

## p1475_商品折扣后的最终价格

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209011031667.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209011031567.png)

### mine

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        for i in range(len(prices)):
            j = i + 1
            while j < len(prices):
                if prices[j] <= prices[i]:
                    prices[i] = prices[i] - prices[j]
                    break
                j += 1
        
        return prices
```

暴力思路其实很简单，就按照题目要求实现就行了。

时间复杂度：O(n^2^)

空间复杂度：O(1)，返回值不算

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209011050857.png)

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        n = len(prices)
        ans = [0] * n
        st = [0]
        for i in range(n - 1, -1, -1):
            p = prices[i]
            while len(st) > 1 and st[-1] > p:
                st.pop()
            ans[i] = p - st[-1]
            st.append(p)
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209011134881.png)

### summary

普通方法比较简单，就是时间复杂度大一点。这里记录一下这个题里对单调栈的理解。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209011206613.png)

结合我这里的草稿可以看到，我们是从price的逆序开始遍历的，st是单调栈，从栈底向栈顶单调递增。单调栈实际存储了一个可能的折扣序列。

一开始单调栈里是空的，我们先判断3，发现栈是空的，不可能有折扣所以把3入栈。

之后看到2，我们发现栈顶是3，大于2，将栈顶弹出，栈空了，所以2也不能有折扣，之后把2入栈。这里解释一下为什么要弹出3，我的理解是题目要找是当前位置之后位置上第一个比自己小的元素作为折扣，如果3能作为某一个商品的折扣，而2在3之前且2比3小，所以2相较于3一定能先有机会成为这个商品的折扣，所以要把3弹出换成2。

之后我们看到6，发现栈顶是2，所以折扣就是2，然后把6入栈。

之后看4，发现栈顶是6，把6弹出，现在栈顶是2，2可以作为折扣，然后把4入栈。

最后看到8，栈顶是4，直接折扣，然后把8入栈。

结束。

## p687_最长同值路径

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209020955340.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209020955057.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209020955356.png)

### mine

我能够想到要用一个递归的深度优先搜索算法来完成这个题目，但是没有具体的思路，之前没有写过dfs。

### others

```python
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            left1 = left + 1 if node.left and node.left.val == node.val else 0
            right1 = right + 1 if node.right and node.right.val == node.val else 0
            nonlocal ans
            ans = max(ans, left1 + right1)
            return max(left1, right1)
        dfs(root)
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209020958384.png)

### summary

这道题实际上就是一个递归的深度优先搜索，之前没做过dfs确实一下子写不出来，看过一个例子就好理解多了。同时注意这个题里最长同值路径等于左最长同值路径与右最长同值路径之和。dfs返回的是有向路径长度。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209021032503.png)

这里解释下为什么dfs只返回有向路径长度，这是因为在这个题中虽然我们图上有四根线，但是我们找的是一个路径，这个路径是不能回头的，所以最多只能走完三条线，所以dfs在返回的时候只能返回left1和right1中的一个更大的值。

## p646_最长数对链

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031412769.png)

### mine

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        cur, res = -inf, 0
        for x, y in sorted(pairs, key=lambda p: p[1]):
            if cur < x:
                cur = y
                res += 1
        return res
```

贪心。要挑选最长数对链的第一个数对时，最优的选择是挑选第二个数字最小的，这样能给挑选后续的数对留下更多的空间。挑完第一个数对后，要挑第二个数对时，也是按照相同的思路，是在剩下的数对中，第一个数字满足题意的条件下，挑选第二个数字最小的。按照这样的思路，可以先将输入按照第二个数字排序，然后不停地判断第一个数字是否能满足大于前一个数对的第二个数字即可。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031418411.png)

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort()
        dp = [1] * len(pairs)
        for i in range(len(pairs)):
            for j in range(i):
                if pairs[i][0] > pairs[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031418546.png)

直接sort()是先按第一个位置排，再按第二个位置排，这样排序后所有的可能前链一定在前面。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031423980.png)

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort()
        arr = []
        for x, y in pairs:
            # 如果x能插入arr，那说明y也有可能插入arr，也就是可能有更优情况
            i = bisect_left(arr, x)
            if i < len(arr):
                arr[i] = min(arr[i], y)
            else:
                arr.append(y)
        return len(arr)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031423546.png)

如果列表中没有元素x，那么bisect_left(ls, x)和bisec_right(ls, x)返回相同的值，该值是x在ls中“**合适的插入点索引，使得数组有序**”。

如果列表中只有一个元素等于x，那么bisect_left(ls, x)的值是x在ls中的**索引**。而bisec_right(ls, x)的值是x在ls中的**索引加1**

如果列表中存在多个元素等于x，那么bisect_left(ls, x)返回**最左边的那个索引**。bisect_right(ls, x)返回**最右边的那个索引加1**

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209031503853.png)

### summary

贪心最简单，只要看出来第二个数字越小越在前就可以了。

## p1582_二进制矩阵中的特殊位置

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209041026914.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209041026321.png)

### mine

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        # 统计每行的和
        sum_row = [sum(row) for row in mat]
        # 统计每列的和
        sum_col = [sum(col) for col in zip(*mat)]
        res = 0
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                # 当前位置为1，且行和列和均为1
                if mat[i][j] == 1 and sum_row[i] == 1 and sum_col[j] == 1:
                    res += 1
        
        return res
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209041042359.png)

这个题是简单题，没什么好说的，特殊点就是**当前位置为1，且行和列和均为1**的位置。更应该注意的是获取行和，列和的过程。

获取行和十分简单，只要每行sum()即可。获取列和则要先把矩阵转置，然后获取行和，这里可以使用zip()函数完成矩阵的转置。

- zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
- 可以使用 list() 转换来输出列表。【zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。】
- 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用*号操作符，可以将元组解压为列表。

`zip(A)`相当于打包，打包为**元组的列表**：

```
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> A = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*A)          # 与 zip 相反，*A 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```

```python
A = [[1,2,3],[4,5,6],[7,8,9]]
print(*A) #[1, 2, 3] [4, 5, 6] [7, 8, 9]
#zip()返回的是一个对象。如需展示列表，需手动 list() 转换。
#print(zip(*A)) #<zip object at 0x000001CD7733A2C8>
print(list(zip(*A)))
# 输出
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209041120606.png)

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        for i, row in enumerate(mat):
            # 该行中1的数量
            cnt1 = sum(row) - (i == 0)
            # cnt1不为0
            if cnt1:
                for j, x in enumerate(row):
                    if x == 1:
                        # 该列所有1所在行中的1的数量之和
                        mat[0][j] += cnt1
        return sum(x == 1 for x in mat[0])
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209041120660.png)

### summary

直接模拟的方法很简单，主要就是计算行和，列和。计算列和时可以先使用`zip()`把矩阵转置再计算。
