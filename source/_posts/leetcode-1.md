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

## M_p998_最大二叉树2

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

## M_p2_两数相加

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

## M_p946_验证栈序列

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

## M_p3_无重复字符的最长子串

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

## E_p1475_商品折扣后的最终价格

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

## M_p687_最长同值路径

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

## M_p646_最长数对链

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

## E_p1582_二进制矩阵中的特殊位置

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

## M_p652_寻找重复的子树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051653911.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051653872.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051654550.png)

### mine

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# serial = node(node.left)(node.left)
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        res = []
        hash = dict()
        def dfs(node: Optional[TreeNode]) -> str:
            if not node:
                return ""

            serial = str(node.val) + '(' + dfs(node.left) + ')(' + dfs(node.right) + ')'
            if (hash.get(serial, None)):
                res.append(hash[serial])
            else:
                hash[serial] = node
            return serial

        dfs(root)
        return(set(res))
```

我目前遇到的几个需要遍历数的都可以使用深度优先搜索来递归的完成搜索。

这个题目是要寻找重复的子树，所以我们要把寻找过程中所有遇到的子树记录下来，每次遇到一个子树时要看之前遇到过的子树有没有一样的，寻找过的子树可以使用哈希结构来进行存储。

这里因为树结构是一个自定义的结构，没有办法直接使用树结构进行索引，所以需要将树先进行序列化，同时这个序列化要保证：

- 相同子树序列化啊结果一致
- 不同子树序列化结果不同

我们使用dfs来递归的将子树序列化，将一棵以 x 为根节点值的子树序列化为：x(左子树的序列化结果)(右子树的序列化结果)。如果子树为空，那么序列化结果为空串。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051705489.png)

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051706364.png)

```python
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            tri = (node.val, dfs(node.left), dfs(node.right))
            if tri in seen:
                (tree, index) = seen[tri]
                repeat.add(tree)
                return index
            else:
                nonlocal idx
                idx += 1
                seen[tri] = (node, idx)
                return idx
        
        idx = 0
        seen = dict()
        repeat = set()

        dfs(root)
        return list(repeat)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209051706329.png)

### summary

主要思路就是在dfs的过程中使用哈希结构保存见过的子树，然后在搜索过程中不断查询当前子树是否在哈希表中出现过。

## H_p828_统计子串中的唯一字符

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209061140467.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209061141066.png)

### mine

```python
class Solution:
    def uniqueLetterString(self, s: str) -> int:
        def getSubString(s: str) -> list:
            sub_string = []
            # length为每次取子串长度
            for length in range(len(s)):
                # 按取的长度从字符串开头向后滑动
                sub_string += [s[idx: idx + length +1] for idx in range(len(s)-length)]
            return sub_string

        def get_only_num(sub_string: str) -> int:
            # 哈希表存是否见过
            seen = dict()
            for i in sub_string:
                if i in seen:
                    seen[i] = False
                else:
                    seen[i] = True
            return sum(seen.values())

        res = 0
        for i in getSubString(s):
            res += get_only_num(i)

        return res
```

我这种就是暴力方法，先获得所有子串，然后再对每个子串计算唯一字符数。这种长度一长就要爆掉，没法ac，还是需要找到合适的算法。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209061143919.png)

```python
class Solution:
    def uniqueLetterString(self, s: str) -> int:
        index = collections.defaultdict(list)
        # 先统计每种字符在原始字符串中的下标
        for i, c in enumerate(s):
            index[c].append(i)

        res = 0
        for arr in index.values():
            # 为了解决左边没有或右边没有的情况
            arr = [-1] + arr + [len(s)]
            for i in range(1, len(arr) - 1):
                res += (arr[i] - arr[i - 1]) * (arr[i + 1] - arr[i])
        return res
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209061143242.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209061155332.png)

### summary

暴力的方法难度不高，但是时间和空间复杂度太高了，字符串一长就受不了了。

## E_p1592_重新排列单词间的空格

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209071135776.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209071135614.png)

### mine

```python
class Solution:
    def reorderSpaces(self, text: str) -> str:
        word_list = [i for i in text.split(' ') if i != '']
        word_num = len(word_list)
        space_num = len([i for i in text if i == ' '])

        if word_num == 1:
            return word_list[0] + ' ' * space_num

        avg_space = int(space_num / (word_num - 1))
        end_space = space_num % (word_num - 1)

        res = ''
        for i in range(len(word_list)):
            res += word_list[i]
            if i != len(word_list) - 1:
                res += ' ' * avg_space
        res += ' ' * end_space

        return res
```

直接计算单词数和空格数，然后构造答案。

### others

```python
class Solution:
    def reorderSpaces(self, text: str) -> str:
        words = text.split()
        space = text.count(' ')
        if len(words) == 1:
            return words[0] + ' ' * space
        per_space, rest_space = divmod(space, len(words) - 1)
        return (' ' * per_space).join(words) + ' ' * rest_space
```

思路一致，优化代码。

**divmod() **函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)。

**join()** 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

### summary

题解思路很简单，代码写法可以优化。

## M_p667_优美的排列2

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209081110294.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209081110869.png)

### mine

```python
class Solution:
    def constructArray(self, n: int, k: int) -> List[int]:
        sort_n = list(range(1, n+1))
        res = []
        for i in range(k // 2):
            res.extend([sort_n[i], sort_n[n - i - 1]])
        if k % 2 == 0:
            res.extend(list(range(n - k // 2, k // 2, -1)))
        if k % 2 != 0:
            res.extend(list(range(k // 2 + 1, n - k // 2 + 1)))
        
        return res
```

我的方法很简单，就是按照k的奇偶不同分别进行排列，按照一小一大的顺序先排出来k-1个不同的整数，之后顺序排列，最后一个整数就是1。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209081113377.png)

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209081116685.png)

```python
class Solution:
    def constructArray(self, n: int, k: int) -> List[int]:
        answer = list(range(1, n - k))
        i, j = n - k, n
        while i <= j:
            answer.append(i)
            if i != j:
                answer.append(j)
            i, j = i + 1, j - 1
        return answer
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209081116031.png)

他这个算法和我的思路其实差不多，只不过他前面先顺序，后面开始一大一小，而且他这种不用区分奇偶。

### summary

这题和数据结构没啥关系，完全就是考验构造排列的思维。

## E_p1598_文件夹操作日志搜集器

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209090909955.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209090909001.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209090910348.png)

### mine

```python
class Solution:
    def minOperations(self, logs: List[str]) -> int:
        depth = 0
        for i in logs:
            if i == '../':
                depth -= 1 if depth != 0 else 0
            elif i == './':
                continue
            else:
                depth +=1 
        return depth
```

直接模拟

### summary

没啥说的，模拟就完了。

## H_p4_寻找两个正序数组的中位数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092023968.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092023694.png)

### mine

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        sorted_nums = sorted(nums1 + nums2)
        length = len(nums1) + len(nums2)
        if length % 2 == 0:
            return (sorted_nums[length // 2] + sorted_nums[length // 2 - 1]) / 2.0
        else:
            return float(sorted_nums[length // 2])
```

我这方法只配做简单题，题本身要求的时间复杂度我这肯定达不到。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092048644.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092048103.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092048630.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092049110.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092049934.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092049980.png)

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def getKthElement(k):
            """
            - 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
            - 这里的 "/" 表示整除
            - nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
            - nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
            - 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
            - 这样 pivot 本身最大也只能是第 k-1 小的元素
            - 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
            - 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
            - 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
            """
            
            index1, index2 = 0, 0
            while True:
                # 特殊情况
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])

                # 正常情况
                newIndex1 = min(index1 + k // 2 - 1, m - 1)
                newIndex2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
                if pivot1 <= pivot2:
                    k -= newIndex1 - index1 + 1
                    index1 = newIndex1 + 1
                else:
                    k -= newIndex2 - index2 + 1
                    index2 = newIndex2 + 1
        
        m, n = len(nums1), len(nums2)
        totalLength = m + n
        if totalLength % 2 == 1:
            return getKthElement((totalLength + 1) // 2)
        else:
            return (getKthElement(totalLength // 2) + getKthElement(totalLength // 2 + 1)) / 2
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092050954.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092050104.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092051836.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092051535.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092051676.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092052110.png)

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)

        infinty = 2**40
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        # median1：前一部分的最大值
        # median2：后一部分的最小值
        median1, median2 = 0, 0

        while left <= right:
            # 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
            # // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            i = (left + right) // 2
            j = (m + n + 1) // 2 - i

            # nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            nums_im1 = (-infinty if i == 0 else nums1[i - 1])
            nums_i = (infinty if i == m else nums1[i])
            nums_jm1 = (-infinty if j == 0 else nums2[j - 1])
            nums_j = (infinty if j == n else nums2[j])

            if nums_im1 <= nums_j:
                median1, median2 = max(nums_im1, nums_jm1), min(nums_i, nums_j)
                left = i + 1
            else:
                right = i - 1

        return (median1 + median2) / 2 if (m + n) % 2 == 0 else median1
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209092052523.png)

### summary

这个题确实挺难的，看到O(log(m+n))这种复杂度就应该想到二分法，同时这个题的特殊情况也很多，需要对边界进行仔细地处理。

## M_p670_最大交换

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209132035026.png)

### mine

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        res = list(str(num))
        sorted_num = sorted(res)
        print(res)
        i = 0
        # 开始从头开始遍历输入，看是不是当前可以使用的最大值
        while res[i] and res[i] == sorted_num[-1]:
            i += 1
            sorted_num.pop()
            if i == len(res): return num
        temp = res[i]
        # 找到最右的当前最大值进行交换
        res[str(num).rfind(sorted_num[-1])] = temp
        res[i] = sorted_num[-1]
        return int(''.join((str(i) for i in res)))
```

这道题其实不难，只要想清楚如何才能构造出来最大的数就可以了，只要注意两点：

- 当前最大值应该尽可能放到前面
- 当前最大值有多个时，替换最右的那个

我这里选择先对输入进行了一个排序来方便查找当前最大值，整体思路就是从头遍历输入，一位位判断是不是当前最大值，如果是当前最大值则将辅助数组中的当前最大值弹出，接下来判断下一位。直到遇见第一个不是当前最大值的数字，此时就要将当前最大值换到这个位置，如果存在多个相同的当前最大值则要换最右边的那个。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209132044723.png)

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        ans = num
        s = list(str(num))
        for i in range(len(s)):
            for j in range(i):
                s[i], s[j] = s[j], s[i]
                ans = max(ans, int(''.join(s)))
                s[i], s[j] = s[j], s[i]
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209132045842.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209132045758.png)

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        s = list(str(num))
        n = len(s)
        maxIdx = n - 1
        idx1 = idx2 = -1
        for i in range(n - 1, -1, -1):
            if s[i] > s[maxIdx]:
                maxIdx = i
            elif s[i] < s[maxIdx]:
                idx1, idx2 = i, maxIdx
        if idx1 < 0:
            return num
        s[idx1], s[idx2] = s[idx2], s[idx1]
        return int(''.join(s))
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209132045745.png)

### summary

- 暴力方法没什么说的
- 官方第二个方法思路与我基本一致，只是实现细节上有所不同。官方采用两个指针来进行操作，感觉不如构造一个辅助数组然后顺序遍历输入的方法好理解。

## E_p1619_删除某些元素后的数组均值

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209140942424.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209140942248.png)

### mine

```python
class Solution:
    def trimMean(self, arr: List[int]) -> float:
        sorted_arr = sorted(arr)
        res_arr = sorted_arr[len(sorted_arr)//20:-len(sorted_arr)//20]
        return sum(res_arr)/len(res_arr)
```

先排序，然后切片，最后求均值。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209140945505.png)

### summary

可以先计算一次len()

## M_p672_灯泡开关2

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209151011934.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209151011779.png)

### mine

```python
class Solution:
    def flipLights(self, n: int, presses: int) -> int:
        """
        因为开关的状态决定了最后灯泡的状态，而且开关只有0，1两种状态，同一开关按2下和按4下没有任何区别，所以我们只需要分析四个开关一共有多少种可能，2^4=16，但开关3可由1，2线性表示，所以最多有八种开关状态。
        """
        # 不按开关
        if presses == 0:
            return 1

        if n == 1:
            return 2
        elif n == 2:
            if presses == 1:
                # 没法都开
                return 3
            else:
                return 4
        else:
            if presses == 1:
                # 四个开关四种结果
                return 4
            elif presses == 2:
                # 看示意图红下划线，7种
                return 7
            else:
                return 8
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209151012216.png)

黑色连线表示开关3可由1，2线性表示。红色下划线表示代码中7的由来。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209151013148.png)

```python
class Solution:
    def flipLights(self, n: int, presses: int) -> int:
        seen = set()
        for i in range(2**4):
            pressArr = [(i >> j) & 1 for j in range(4)]
            if sum(pressArr) % 2 == presses % 2 and sum(pressArr) <= presses:
                status = pressArr[0] ^ pressArr[1] ^ pressArr[3]
                if n >= 2:
                    status |= (pressArr[0] ^ pressArr[1]) << 1
                if n >= 3:
                    status |= (pressArr[0] ^ pressArr[2]) << 2
                if n >= 4:
                    status |= (pressArr[0] ^ pressArr[1] ^ pressArr[3]) << 3
                seen.add(status)
        return len(seen)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209151014034.png)

### summary

这个题如果分析出来最大情况只有八种就比较简单了，可以直接列出特殊情况得出结果。

## H_p850_矩形面积2

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209160949748.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209160949717.png)

### mine

```python
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        rect_set = set()
        for rect in rectangles:
            for x in range(rect[0], rect[2]):
                for y in range(rect[1], rect[3]):
                    rect_set.add((x, y))
        
        return len(rect_set) % (10^9 + 7)
```

简单的思路，但是显然不太符合困难题的难度……

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209160956052.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209160957245.png)

```python
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        hbound = set()
        for rect in rectangles:
            # 下边界
            hbound.add(rect[1])
            # 上边界
            hbound.add(rect[3])
        
        hbound = sorted(hbound)
        m = len(hbound)
        # 「思路与算法部分」的 length 数组并不需要显式地存储下来
        # length[i] 可以通过 hbound[i+1] - hbound[i] 得到
        seg = [0] * (m - 1)

        sweep = list()
        for i, rect in enumerate(rectangles):
            # 左边界
            sweep.append((rect[0], i, 1))
            # 右边界
            sweep.append((rect[2], i, -1))
        sweep.sort()

        ans = i = 0
        while i < len(sweep):
            j = i
            while j + 1 < len(sweep) and sweep[i][0] == sweep[j + 1][0]:
                j += 1
            if j + 1 == len(sweep):
                break
            
            # 一次性地处理掉一批横坐标相同的左右边界
            for k in range(i, j + 1):
                _, idx, diff = sweep[k]
                left, right = rectangles[idx][1], rectangles[idx][3]
                for x in range(m - 1):
                    if left <= hbound[x] and hbound[x + 1] <= right:
                        seg[x] += diff
            
            cover = 0
            for k in range(m - 1):
                if seg[k] > 0:
                    cover += (hbound[k + 1] - hbound[k])
            ans += cover * (sweep[j + 1][0] - sweep[j][0])
            i = j + 1
        
        return ans % (10**9 + 7)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209160957635.png)

### summary

学习到了很多新概念，离散化，扫描线，确实没了解过根本做不来。

## E_p1624_两个相同字符之间的最长字符串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209170943035.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209170943925.png)

### mine

```python
class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        """
        使用一个哈希表存储第一次见到的字母的下标，之后遇到的时候比较
        """
        ans = -1
        index_dict = dict()
        for index, item in enumerate(s):
            if item not in index_dict:
                index_dict[item] = index
            else:
                ans = max(ans, index - index_dict[item] - 1)
        
        return ans
```

### summary

没什么好说的，就是使用哈希来存储第一次见到的下标，之后相减比大小就行了。

## H_p827_最大人工岛

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209181154580.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209181154636.png)

### mine

```python
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        """
        1. 使用唯一标记值标记每个岛屿t=i*n+j+1,第一个搜索到的(i,j)决定了它所在岛屿的t（t为该点的位置编号，一定独一无二）
        2. 使用哈希area保存每个岛屿大小,t做索引
        3. 遍历所有为0的点，判断周围能连接的岛屿，计算连接后岛屿大小
        """
        n = len(grid)
        tag = [[0] * n for row in range(n)]
        area = dict()

        def dfs(i, j, t):
            tag[i][j] = t
            if t not in area:
                area[t] = 1
            else:
                area[t] += 1
            for x, y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
                if 0<=x<n and 0<=y<n and grid[x][y] and tag[x][y] == 0:
                    dfs(x, y, t)
        
        # 使用dfs开始遍历整个grid，获取tag和area
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                if item and tag[i][j] == 0:
                    # 每个岛屿第一个遍历到的点作为tag
                    t = i*n + j + 1
                    dfs(i, j, t)

        ans = max(area.values(), default=0)
        
        # 开始遍历grid中所有为0的地方，看四周有没有岛屿，计算连接后结果
        for i, row in enumerate(grid):
            for j, item in enumerate(row):
                if item == 0:
                    # 增加一个点，最少是1
                    add_area = 1
                    # 记录已经连接过的岛屿tag，可能四周出现同一个岛屿
                    connect = [0]
                    for x, y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
                        if 0<=x<n and 0<=y<n and grid[x][y] and tag[x][y] not in connect:
                            add_area += area[tag[x][y]]
                            connect.append(tag[x][y])
                    ans = max(ans, add_area)

        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209181158470.png)

参考官解写出来的，整体思路在开头的注释。

### summarys

1. 使用唯一标记值标记每个岛屿**t=i * n + j + 1**,第一个搜索到的(i, j)决定了它所在岛屿的t（t为该点的位置编号，一定独一无二）
2. 使用哈希area保存每个岛屿大小,t做索引
3. 遍历所有为0的点，判断周围能连接的岛屿，计算连接后岛屿大小

## E_p1636_按照频率将数组升序排列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209191116699.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209191116861.png)

### mine

```python
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        frequency_dict = dict()
        # 哈希统计词频
        for i in nums:
            if i not in frequency_dict:
                frequency_dict[i] = 1
            else:
                frequency_dict[i] += 1

        # 将词频转化为可排序的键值对
        frequency_list = []       
        for key, value in frequency_dict.items():
            frequency_list.append((key, value))

        # 对词频排序
        frequency_list.sort(key=lambda x: (x[1], -x[0]))

        ans = []
        for i in frequency_list:
            ans.extend([i[0]]*i[1])

        return ans
```

三个步骤代码里都注释了，整体思路就是统计词频然后排序最后构造答案。

### others

```python
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        cnt = Counter(nums)
        nums.sort(key=lambda x: (cnt[x], -x))
        return nums
```

和我的思路一样的，但是通过调用Python的库两行就可以把功能实现。

**Counter()**专门用来统计词频，返回一个字典。

这里主要记录下sort中key的用法：

key接受的是一个只有一个形参的函数

key接受的函数返回值，表示此元素的权值，sort将按照权值大小进行排序

### summary

合理调用库函数可以极大减少代码量。

## M_p698_划分为k个相等的子集

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209201132894.png)

### mine

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if sum(nums) % k != 0:
            return False
        average = sum(nums) // k
        nums_flag = [0] * len(nums)
        nums.sort()
        if nums[-1] > average:
            return False
        
        def divideGroups(nums: List[int], start: int, average: int, current:int, k: int):
            # print(nums_flag, start, average, current, k)
            if k == 1:
                # 前k-1个箱子填满了，第k个必能填满
                # print(333)
                return True
            if current == average:
                # 一个箱子装满了，开始装剩下的箱子
                # print(111)
                return divideGroups(nums, len(nums)-1, average, 0, k-1)
            for i in range(start, -1, -1):
                if nums_flag[i] == 1 or current + nums[i] > average:
                    # 当前元素被使用过，或放入箱子之后超出则开始判断下一个
                    continue
                # 表示该元素被占用
                nums_flag[i] = 1
                # 看使用该元素情况下能否占满一个箱子
                if divideGroups(nums, i-1, average, current + nums[i], k):
                    return True
                # 不行的话就释放该元素
                nums_flag[i] = 0
                # 例如“12333333...”，假如最右侧的“3”这个值没有匹配上，那么它左侧的剩余五个“3”都不需要再匹配了。
                while i > 0 and nums[i] == nums[i - 1]:
                    i -= 1
            return False

        return divideGroups(nums, len(nums)-1, average, 0, k)
```

参考java写的回溯，算法内容应该是对的，但是超时了，应该是剪枝没有做的很好，但是java是能通过的……

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209201502815.png)

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        all = sum(nums)
        if all % k:
            return False
        per = all // k
        nums.sort()  # 方便下面剪枝
        if nums[-1] > per:
            return False
        n = len(nums)

        @cache
        def dfs(s, p):
            if s == 0:
                return True
            for i in range(n):
                # 因为是升序排列，所以i处的不满足后面一定也不满足
                if nums[i] + p > per:
                    break
                if s >> i & 1 and dfs(s ^ (1 << i), (p + nums[i]) % per):  # p + nums[i] 等于 per 时置为 0
                    return True
            return False
        return dfs((1 << n) - 1, 0)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209201507945.png)

我一开始还有些迷惑为什么不需要判断放入桶中的总量和桶容量是否相等，但是后来明白是因为，总空间就这么大，如果能把所有球都用上（即s全0），那说明所有桶都没超，也就是所有桶都满了。

创建一个查找函数参数的字典的简单包装器。 因为它不需要移出旧值，缓存大小没有限制，所以比带有大小限制的 `lru_cache()` 更小更快。这个 `@cache` 装饰器是 Python 3.9 版中的新功能，在此之前，您可以通过 `@lru_cache(maxsize=None)` 获得相同的效果。

```python
from functools import cache

# 在一个递归函数上应用 cache 装饰器
@cache
def factorial(n):
    return n * factorial(n-1) if n else 1Q

>>> factorial(10)      # 没有以前缓存的结果，进行11次递归调用
3628800
>>> factorial(5)       # 只是查找缓存值结果
120
>>> factorial(12)      # 进行两个新的递归调用，其他10个被缓存
479001600
```

### summary

这个题本身写出回溯就不是很好写，而且还非常容易超时，需要按照官解的方法先进行压缩，再使用`@cache`进行记忆化搜索。

## H_p854_相似度为K的字符串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209211921399.png)

### mine

```python
class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        s, t = [], []
        for x, y in zip(s1, s2):
            if x != y:
                s.append(x)
                t.append(y)
        n = len(s)
        if n == 0:
            return 0

        ans = n - 1
        def dfs(i: int, cost: int) -> None:
            nonlocal ans
            if cost > ans:
                return
            while i < n and s[i] == t[i]:
                i += 1
            if i == n:
                ans = min(ans, cost)
                return
            for j in range(i + 1, n):
                if s[j] == t[i]:
                    s[i], s[j] = s[j], s[i]
                    dfs(i + 1, cost + 1)
                    s[i], s[j] = s[j], s[i]
        dfs(0, 0)
        return ans
```

暴力dfs递归能勉强通过，但一定要先对两个字符串做处理，将符合条件的字符先去掉，这样也是一种剪枝，不这样做的话会超时。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209211934523.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209211935374.png)

```python
class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        s, t = [], []
        for x, y in zip(s1, s2):
            if x != y:
                s.append(x)
                t.append(y)
        n = len(s)
        if n == 0:
            return 0

        ans = n - 1
        def dfs(i: int, cost: int) -> None:
            nonlocal ans
            if cost > ans:
                return
            while i < n and s[i] == t[i]:
                i += 1
            if i == n:
                ans = min(ans, cost)
                return
            diff = sum(s[j] != t[j] for j in range(i, len(s)))
            min_swap = (diff + 1) // 2
            if cost + min_swap >= ans:  # 当前状态的交换次数下限大于等于当前的最小交换次数
                return
            for j in range(i + 1, n):
                if s[j] == t[i]:
                    s[i], s[j] = s[j], s[i]
                    dfs(i + 1, cost + 1)
                    s[i], s[j] = s[j], s[i]
        dfs(0, 0)
        return ans
```

我的DFS就是参考官解写的，但是官解有个非常重要的剪枝是我想不到的，就是根据后面还没有换到的数量计算一个交换次数的下限，如果当前cost加当前下限已经大于了ans，那这种情况实际上不需要再走下去了，就可以return掉，这个剪枝可以极大的改善算法的耗时。

### summary

dfs递归确实好用，但是递归就意味着时间开销不会小，一定要想一些有效的剪枝策略，不然很有可能就要超时。

## E_p1640_能否连接形成数组

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210061157925.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209221140609.png)

### mine

```python
class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        for item in pieces:
            # 不止一个元素
            if len(item) != 1:
                # 元素不存在
                if item[0] not in arr:
                    return False
                start = arr.index(item[0])
                j = 1
                # 越界了直接Flase
                if start + len(item) > len(arr):
                    return False
                # 看顺序对不对
                for i in range(start + 1, start + len(item)):
                    if arr[i] != item[j]:
                        return False
                    j += 1
            # 一个元素时只要判断在不在就行了
            else:
                if item[0] not in arr:
                    return False
        
        return True
```

我的思路很简单，首先对pieces内元素分两种情况：

- 元素长度为1，我们只需要判断这个元素有没有在arr中出现。
- 元素长度不为1，我们首先定位到这个元素中item[0]在arr中的位置，然后判断后面的长度够不够和顺序对不对。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209221149256.png)

```python
class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        index = {p[0]: i for i, p in enumerate(pieces)}
        i = 0
        while i < len(arr):
            if arr[i] not in index:
                return False
            p = pieces[index[arr[i]]]
            if arr[i: i + len(p)] != p:
                return False
            i += len(p)
        return True
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209221150435.png)

思路其实都差不多，官解是使用了一个哈希表来完成index的记录，但是他这个要遍历一遍prices，我那个如果运气好一开始就会False，说不好哪个快。

### summary

这种题主要还是讲究个思路，实现起来很简单。

## M_p707_设计链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232202144.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232202266.png)

### mine

```python
class ListNode:

    def __init__(self, val):
        self.val = val
        self.next = None


class MyLinkedList:

    def __init__(self):
        # 存长度
        self.size = 0
        # 存头节点
        self.head = ListNode(0)

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        cur = self.head
        for _ in range(index + 1):
            cur = cur.next
        return cur.val


    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)


    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)


    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        add_node = ListNode(val)
        self.size += 1
        if index <= 0:
            add_node.next = self.head.next
            self.head.next = add_node
            return
        cur = self.head
        for _ in range(index):
            cur = cur.next
        add_node.next = cur.next
        cur.next = add_node

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        self.size -= 1
        cur = self.head
        for _ in range(index):     
            cur = cur.next
        cur.next = cur.next.next


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232204318.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232205236.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232206714.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232206156.png)

```python
class ListNode:

    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None


class MyLinkedList:

    def __init__(self):
        self.size = 0
        self.head, self.tail = ListNode(0), ListNode(0) 
        self.head.next = self.tail
        self.tail.prev = self.head


    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        if index + 1 < self.size - index:
            curr = self.head
            for _ in range(index + 1):
                curr = curr.next
        else:
            curr = self.tail
            for _ in range(self.size - index):
                curr = curr.prev
        return curr.val


    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)


    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)


    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        index = max(0, index)
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next
        else:
            succ = self.tail
            for _ in range(self.size - index):
                succ = succ.prev
            pred = succ.prev
        self.size += 1
        to_add = ListNode(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add


    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next.next
        else:
            succ = self.tail
            for _ in range(self.size - index - 1):
                succ = succ.prev
            pred = succ.prev.prev
        self.size -= 1
        pred.next = succ
        succ.prev = pred
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209232207156.png)

### summary

整体来说就是构造一个数据结构来模拟链表，链表主要维护节点数和头节点，并且注意头节点是不算在index里面的，头节点的后续节点是index为0的节点，操作链表时主要注意向后遍历时候的次数就行了。

## E_p1652_拆炸弹

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209241708418.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209241708486.png)

### mine

```python
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        if k == 0:
            return [0] * len(code)
        new_code = code + code
        num_code = len(code)
        ans = []
        if k > 0:
            for i in range(num_code):
                ans.append(sum(new_code[i+1:i+k+1]))
        if k < 0:
            for i in range(num_code):
                ans.append(sum(new_code[num_code+i+k:num_code+i]))
        return ans
```

简单题没什么复杂的地方，循环通过两个数组前后相连来解决，取k个值通过数组切片来取。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209241711890.png)

```python
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        if k == 0:
            return [0] * len(code)
        res = []
        n = len(code)
        code += code
        if k > 0:
            l, r = 1, k
        else:
            l, r = n + k, n - 1
        w = sum(code[l:r+1])
        for i in range(n):
            res.append(w)
            w -= code[l]
            w += code[r + 1]
            l, r = l + 1, r + 1
        return res
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209241712695.png)

也可以通过取模的操作解决循环数组，可以减少一点空间开销。

### summary

循环数组可以通过取模或者数组拼接来完成。

## M_p788_旋转数字

![image-20220925104558633](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209251046745.png)

### mine

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        """
        遍历
        1.不能有3，4，7(有效)
        2.至少有2，5，6，9其中之一(不同)
        3.分别判断有效和不同
        """
        ans = 0
        for i in range(1, n + 1):
            digital_list = [int(digital) for digital in str(i)]
            valid = True
            different = False
            for digital in digital_list:
                if digital in [3, 4, 7]:
                    valid = False
                    break
                if digital in [2, 5, 6, 9]:
                    different = True
            if valid and different:
                ans += 1
        
        return ans
```

遍历一遍N个数，对每个数遍历每一位，分别判断是否有效和是否不同。

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209251050039.png)

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209251050709.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209251050869.png)

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        check = [0, 0, 1, -1, -1, 1, 1, -1, 0, 1]
        digits = [int(digit) for digit in str(n)]

        @cache
        def dfs(pos: int, bound: bool, diff: bool) -> int:
            if pos == len(digits):
                return int(diff)
            
            ret = 0
            for i in range(0, (digits[pos] if bound else 9) + 1):
                if check[i] != -1:
                    ret += dfs(
                        pos + 1,
                        bound and i == digits[pos],
                        diff or check[i] == 1
                    )
            
            return ret
            
        
        ans = dfs(0, True, False)
        dfs.cache_clear()
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209251051083.png)

### summary

暴力遍历可以做，动态规划可以减少时间复杂度。

## M_interview17.09_第k个数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209282135399.png)

### mine

```python
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        seen = [1]
        for i in range(k - 1):
            new = min(seen)
            for factor in [3, 5, 7]:
                if new * factor not in seen:
                    seen.append(new * factor)
            seen.remove(new)
        
        return min(seen)
```

我这种方法就是暴力，而且没进行什么优化，只能说可以成功AC。

就是从1开始不断找最小的数分别乘上3，5，7。因为素因子乘积得到的数的因子只可能是这些素因子。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209282156122.png)

```python
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        factors = [3, 5, 7]
        seen = {1}
        heap = [1]

        for i in range(k - 1):
            curr = heapq.heappop(heap)
            for factor in factors:
                if (nxt := curr * factor) not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, nxt)

        return heapq.heappop(heap)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209282156676.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209282157149.png)

```python
class Solution:
    def getKthMagicNumber(self, k: int) -> int:
        dp = [0] * (k + 1)
        dp[1] = 1
        p3 = p5 = p7 = 1

        for i in range(2, k + 1):
            num3, num5, num7 = dp[p3] * 3, dp[p5] * 5, dp[p7] * 7
            dp[i] = min(num3, num5, num7)
            if dp[i] == num3:
                p3 += 1
            if dp[i] == num5:
                p5 += 1
            if dp[i] == num7:
                p7 += 1
        
        return dp[k]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202209282158915.png)

### summary

最小堆在数据结构方面对我的这种算法思路进行了优化，但是需要存储较多的数据，运行速度上不会很快。而动态规划的方法则可以通过三个指针的移动，每次只求出一个当前最小值，不用维护一片具有很多中间结果的数据。

## E_p1694_重新格式化电话号码

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210011732538.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210011733952.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210011733318.png)

### mine

```python
class Solution:
    def reformatNumber(self, number: str) -> str:
        number = number.replace(" ", "").replace("-", "")
        i = 0
        ans = []
        # 先处理前面
        while i < len(number) - 4:
            ans.append(number[i: i + 3])
            i += 3
        # 最后几个数字分类处理
        if len(number) - i == 2:
            ans.append(number[i:])
        elif len(number) - i == 3:
            ans.append(number[i:])
        elif len(number) - i == 4:
            ans.extend([number[i:i+2], number[i+2:]])

        return "-".join(ans)
```

就先统一处理前面的数字，最后分情况讨论最后几个数字就行了。

### summary

没啥说的，正常模拟一遍就完了。

## M_p777_在LR字符串中交换相邻字符

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210020853265.png)

### mine

首先要先理解这个题目，这个题目的意思就是L和R可以在X上按方向滑动，但是不能跨过其他的L或R，因此这个问题如果为True需要满足两个条件：

- start和end中LR的相对位置相同
- start和end中同一对L或R的滑动方向要符合要求

根据以上两个条件可以简单的写出以下代码，能通过，但是时空复杂度都有点大，毕竟要遍历三遍。

```python
class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        if start.replace("X", "") != end.replace("X", ""):
            return False
            
        start_index, end_index = [], []
        for i in range(len(start)):
            if start[i] != 'X':
                start_index.append(i)
        for j in range(len(end)):
            if end[j] != 'X':
                end_index.append(j)

        start_flag = list(zip(start.replace("X", ""), start_index))
        end_flag = list(zip(end.replace("X", ""), end_index))

        for i in range(len(start_flag)):
            if start_flag[i][0] == 'R' and start_flag[i][1] > end_flag[i][1]:
                return False
            if start_flag[i][0] == 'L' and start_flag[i][1] < end_flag[i][1]:
                return False

        return True
```

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210020858238.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210020859964.png)

```python
class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        n = len(start)
        i = j = 0
        while i < n and j < n:
            while i < n and start[i] == 'X':
                i += 1
            while j < n and end[j] == 'X':
                j += 1
            if i < n and j < n:
                if start[i] != end[j]:
                    return False
                c = start[i]
                if c == 'L' and i < j or c == 'R' and i > j:
                    return False
                i += 1
                j += 1
        while i < n:
            if start[i] != 'X':
                return False
            i += 1
        while j < n:
            if end[j] != 'X':
                return False
            j += 1
        return True
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210020859555.png)

双指针，只需要两个字符串各遍历一次即可。

### summary

这个题主要是要理解题意，分析出能返回True的两个必要条件，就肯定能写出来，只不过用双指针时空开销会更小一些。

## E_p1784_检查二进制字符串字段

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210031152722.png)

### mine

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        return '01' not in s
```

实际上是个脑筋急转弯。

- 第一个条件说明开头的1和后面的1之间不能有0，不然就有两个“1”组成的字段
- 第二个条件说明字符串只能是1111111110000000这种形式。

## M_p921_使括号有效的最少添加

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210041045261.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210041045792.png)

### mine

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        while '()' in s:
            s = s.replace('()', '')
        return len(s)
```

说白了就是做括号匹配，python可以使用replace来不断删除字符串中已经匹配好的括号对，剩下的就是需要添加的。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210041047270.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210041047672.png)

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        ans = cnt = 0
        for c in s:
            if c == '(':
                cnt += 1
            elif cnt > 0:
                cnt -= 1
            else:
                ans += 1
        return ans + cnt
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210041048257.png)

### summary

题不难，难度全在读题上。。。不知道啥人写的题目，就离谱。

## M_p811_子域名访问计数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210051114229.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210051121307.png)

### mine

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        seen = dict()
        for item in cpdomains:
            visit_num, domain = item.split(' ')
            if domain in seen:
                seen[domain] += int(visit_num)
            else:
                seen[domain] = int(visit_num)
            sub_domains = ['.'.join(domain.split('.')[i+1:]) for i in range(len(domain.split('.'))-1)]
            for sub_domain in sub_domains:
                if sub_domain in seen:
                    seen[sub_domain] += int(visit_num)
                else:
                    seen[sub_domain] = int(visit_num)

        return [str(value) + ' ' + str(key) for key, value in seen.items()]
```

使用哈希结构存储结果，处理域名获取所有的父域名，给父域名也加上访问次数。

### others

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        cnt = Counter()
        for domain in cpdomains:
            c, s = domain.split()
            c = int(c)
            cnt[s] += c
            while '.' in s:
                s = s[s.index('.') + 1:]
                cnt[s] += c
        return [f"{c} {s}" for s, c in cnt.items()]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210051126822.png)

思路一样只不过官解写法更简洁。

### summary

思路不难，主要是可以优化代码写法。

## H_p927_三等分

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210061200786.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210061200993.png)

### mine

```python
class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        num_1 = sum(arr)
        if num_1 % 3 != 0:
            return [-1, -1]
        if num_1 == 0:
            return [0, 2]

        part = num_1 // 3
        
        # 每个part第一次遇到1
        first = second = thrid = cur = 0
        for i, x in enumerate(arr):
            if x:
                if cur == 0:
                    first = i
                if cur == part:
                    second = i
                if cur == 2 * part:
                    thrid = i
                cur += 1
        # print(first, second, thrid)
        
        # 最后一段二进制值决定了所有的二进制值
        # 0011000110001100
        code_len = len(arr) - thrid
        if first + code_len <= second and second + code_len <= thrid:
            i = 0
            while thrid + i < len(arr):
                if arr[first+i] != arr[second+i] or arr[first+i] != arr[thrid+i]:
                    return [-1, -1]
                i += 1
            return [first+code_len-1, second + code_len]
        return [-1, -1]
```

1. 首先判断arr中的1能不能三等分，不能的话直接-1
2. 计算每个等分中第一个1出现的位置，第三个等分中的二进制代码就是所求的二进制代码
3. 判断前两个等分后面的0够不够，不够的话直接-1
4. 如果够的话开始逐位比较，有不一致的返回-1

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210061204493.png)

```python
class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        s = sum(arr)
        if s % 3:
            return [-1, -1]
        if s == 0:
            return [0, 2]

        partial = s // 3
        first = second = third = cur = 0
        for i, x in enumerate(arr):
            if x:
                if cur == 0:
                    first = i
                elif cur == partial:
                    second = i
                elif cur == 2 * partial:
                    third = i
                cur += 1

        n = len(arr)
        length = n - third
        if first + length <= second and second + length <= third:
            i = 0
            while third + i < n:
                if arr[first + i] != arr[second + i] or arr[first + i] != arr[third + i]:
                    return [-1, -1]
                i += 1
            return [first + length - 1, second + length]
        return [-1, -1]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210061205150.png)

### summary

这个题的代码部分不难，主要是分析题目得出二进制代码由最后一个等分决定。

## E_p1800_最大升序子数组和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210071335869.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210071336437.png)

### mine

```python
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        ans = 0
        cur = 0
        for i in range(len(nums)):
            if i == 0 or nums[i-1] < nums[i]:
                cur += nums[i]
            else:
                cur = nums[i]
            ans = max(cur, ans)
        
        return ans
```

循环模拟一遍就行了。

## M_p870_优势洗牌

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210081946347.png)

### mine

```python
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        ori_index2 = []
        for i in range(len(nums2)):
            ori_index2.append((i, nums2[i]))

        nums1.sort()
        ori_index2.sort(key=lambda x: x[1])
        nums1_flag = [0] * len(nums1)
        ans = []
        i = j = 0
        while i < len(nums1):
            if nums1[i] > ori_index2[j][1]:
                ans.append((ori_index2[j][0], nums1[i]))
                nums1_flag[i] = 1
                i += 1
                j += 1
            else:
                i += 1

        # 如果还有剩下的说明2中的这些值太大了，1中没有能符合的，那就从剩的中直接排列就行了
        if len(nums2)-len(ans) != 0:
            for item in ori_index2[-(len(nums2)-len(ans)):]:
                ans.append((item[0], nums1[nums1_flag.index(0)]))
                nums1_flag[nums1_flag.index(0)] = 1
        
        ans.sort(key=lambda x: x[0])
        return [i[1] for i in ans]
```

1. 首先记录nums2中原本数字的位置
2. 对两个数组都进行排序
3. 使用两个指针分别指向两个数组开头，如果nums1中数大于nums2的则两个指针都后移一位，否则nums1的指针后移一位。直到nums1的指针走到头
4. 如果nums2中还有剩余元素则将nums1中没使用的元素排列上去
5. 最后调整回nums2中原本的顺序

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210081951696.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210081951458.png)

```python
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n = len(nums1)
        idx1, idx2 = list(range(n)), list(range(n))
        idx1.sort(key=lambda x: nums1[x])
        idx2.sort(key=lambda x: nums2[x])

        ans = [0] * n
        left, right = 0, n - 1
        for i in range(n):
            if nums1[idx1[i]] > nums2[idx2[left]]:
                ans[idx2[left]] = nums1[idx1[i]]
                left += 1
            else:
                ans[idx2[right]] = nums1[idx1[i]]
                right -= 1
        
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210081952746.png)

其实整体思路差别不大的，只不过官解有两个技巧可以极大减少代码量。

- 使用两个idx数组来记录nums1和nums2中原本的顺序。
- 使用right和left两个指针来控制nums2的选择，这样子可以避免我的代码中一个数组走空了另一个数组还没走空的问题，这种方式可以保证两个数组同时走空。

### summary

在算法思路大致相同的情况下，代码的技巧也可以很大程度上影响整个算法的编写。

## M_p5_最长回文子串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082017508.png)

### mine

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        """
        暴力
        """
        if len(s) < 2:
            return s

        def isPalindrome(s: str) -> bool:
            for i in range(len(s)//2):
                if s[i] != s[-i-1]:
                    return False
            return True

        max_len = 0
        ans = 0
        for i in range(len(s)):
            for j in range(1, len(s)):
                if isPalindrome(s[i:j+1]) and len(s[i:j+1]) > max_len:
                    ans = s[i:j+1]
                    max_len = len(s[i:j+1])
        
        return ans
```

暴力方法经典超时。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082034178.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082035898.png)

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        
        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        
        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break
                    
                if s[i] != s[j]:
                    dp[i][j] = False 
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                
                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082035068.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082047432.png)

```python
class Solution:
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start: end + 1]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210082047560.png)

### summary

暴力通不过的情况下就应该考虑一些时间复杂度低一点的算法，或者考虑用空间换时间。

## M_p856_括号的分数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210091440423.png)

### mine

```python
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        st = []
        ans = 0
        for i in s:
            if i == '(':
                st.append(i)
            # 如果遇到右括号
            else:
                # 如果该右括号紧挨着左括号则加一分,之后和之前的分数相加
                if st[-1] == '(':
                    st.pop()
                    st.append(1)
                    if len(st) >= 2 and st[-2] != '(':
                        st[-2] += st[-1]
                        st.pop()
                # 如果没有紧挨着左括号则其中元素×2,之后和之前的分数相加
                else:
                    temp = 2 * st[-1]
                    st.pop()
                    st.pop()
                    st.append(temp)
                    if len(st) >= 2 and st[-2] != '(':
                        st[-2] += st[-1]
                        st.pop()
        
        return st[0]
```

使用栈的思想不断入栈出栈。

- 遇到左括号的时候很好处理，入栈就完了
- 遇到右括号的话就要进行判断
  - 如果该右括号紧挨着左括号则加一分,之后和之前的分数相加
  - 如果没有紧挨着左括号则其中元素×2,之后和之前的分数相加

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210101057784.png)

```python
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        n = len(s)
        if n == 2:
            return 1
        bal = 0
        for i, c in enumerate(s):
            bal += 1 if c == '(' else -1
            if bal == 0:
                if i == n - 1:
                    return 2 * self.scoreOfParentheses(s[1:-1])
                return self.scoreOfParentheses(s[:i + 1]) + self.scoreOfParentheses(s[i + 1:])
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210101057542.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210101057852.png)

```python
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        st = [0]
        for c in s:
            if c == '(':
                st.append(0)
            else:
                v = st.pop()
                st[-1] += max(2 * v, 1)
        return st[-1]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210101058032.png)

```python
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        ans = bal = 0
        for i, c in enumerate(s):
            bal += 1 if c == '(' else -1
            if c == ')' and s[i - 1] == '(':
                ans += 1 << bal
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210101058587.png)

### summary

官解中提供了三种不同的思路，第一种是分治递归的思想；第二种是栈的思想；第三种则是直接找到题目中的关键规律直接得到答案。

## H_p801_使序列递增的最小交换次数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210102154090.png)

### mine

```python
class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        """
        dp[i][0]表示第i位不交换情况下的操作次数
        dp[i][1]表示第i位交换情况下的操作次数
        """
        n = len(nums1)
        dp = [[0, 0] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = 1
        for i in range (1, n):
            # 原本两数组均递增，互换后仍递增(i交换与否与之前无关)
            if (nums1[i-1] < nums1[i] and nums2[i-1] < nums2[i]) and (nums1[i-1] < nums2[i] and nums2[i-1] < nums1[i]):
                dp[i][0] = min(dp[i-1][0], dp[i-1][1])
                dp[i][1] = dp[i][0] + 1
            # 原本两数组均递增，互换后不递增(i交换i-1就也得交换)
            elif nums1[i-1] < nums1[i] and nums2[i-1] < nums2[i]:
                dp[i][0] = dp[i-1][0]
                dp[i][1] = dp[i-1][1] + 1
            # 原本两数组就不递增(i交换则i-1不能交换)
            else:
                dp[i][0] = dp[i-1][1]
                dp[i][1] = dp[i-1][0] + 1
                
        return min(dp[n-1][0], dp[n-1][1])
```

困难题肯定不可能用暴力搜索，那肯定会超时。首先我们要知道题目限定了用例一定可以实现操作，那么说起来一共只有以下三种情况：

- 原本两数组均递增，互换后仍递增(i交换与否与i-1无关)
- 原本两数组均递增，互换后不递增(i交换i-1就也得交换)
- 原本两数组就不递增(i交换则i-1不能交换)

说白了就这么三种情况，而且仅与上一位状态有关，这种就需要使用动态规划来完成，使用一个辅助数组来记录每一位置上交换或不交换所需的最少次数就行了。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210102159917.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210102200459.png)

```python
class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        a, b = 0, 1
        for i in range(1, n):
            at, bt = a, b
            a = b = n
            if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
                a = min(a, at)
                b = min(b, bt + 1)
            if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
                a = min(a, bt)
                b = min(b, at + 1)
        return min(a, b)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210102200946.png)

官解的算法思路和我们的是一致的，但是官解的代码编写有两点可以学习：

- 没有使用数组来记录动态规划过程中的状态，而是使用两个常量，毕竟i位置情况只与i-1位置有关，不需要维护每个位置需要的操作次数
- 在三种情况的分支编写中，官解没有使用if-elif-else的方式，而是使用了两个if来涵盖了三个情况，更加简洁。

### summary

创建数组的时候不要使用`list_two = [[0] * 3] * 3`这种方式，不然会出现以下情况：

```python
list_two[1][1] = 2
print(list_two)

[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
[[0, 2, 0], [0, 2, 0], [0, 2, 0]]
```

为什么会出现在这种情况呢？原因是浅拷贝，我们以这种方式创建的列表，list_two 里面的三个列表的内存是指向同一块，不管我们修改哪个列表，其他两个列表也会跟着改变。

如果要使用列表创建一个二维数组，可以使用列表生成器来辅助实现。

```python
list_three = [[0 for i in range(3)] for j in range(3)]
print(list_three)
list_three[1][1] = 3
print(list_three)
```

我们对 list_three 进行更新操作，这次就能正常更新了。

```python
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
[[0, 0, 0], [0, 3, 0], [0, 0, 0]]
```

## E_p1790_仅执行一次字符串交换能否使两个字符串相等

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210112308082.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210112308113.png)

### mine

```python
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        x, y = [], []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                x.append(s1[i])
                y.append(s2[i])
        
        if len(x) == 0:
            return True
        
        if len(x) == 2 and (x[0] == y[1] and x[1] == y[0]):
            return True
        return False
```

简单题没啥说的，遍历一遍就完了。

## M_p817_链表组件

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210121037372.png)

### mine

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        cur = head
        origin = []
        while cur != None:
            origin.append(cur.val)
            cur = cur.next
        nums_index = sorted([origin.index(i) for i in nums])
        
        ans = 1
        for i in range(1, len(nums_index)):
            if nums_index[i] - nums_index[i-1] != 1:
                ans += 1
        
        return ans
```

我这个思路是先把原始链表变成数组，然后获取到nums列表中元素在原始列表中的index然后看几段连在一起的index。

有点慢，要遍历两次，还要排序。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210121044332.png)

```python
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        numsSet = set(nums)
        inSet = False
        res = 0
        while head:
            if head.val not in numsSet:
                inSet = False
            elif not inSet:
                inSet = True
                res += 1
            head = head.next
        return res
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210121044917.png)

其实遍历一遍就够了，在遍历的过程中进行如下判断：

- 如果该元素不在nums中则标志位置False
- 如果该元素在nums中并且当前标志位为False(说明该链表位置之前至少有一个位置不在nums中，即当前位置会是一个组件的开头)，标志位置为True同时res+1

### summary

这个题其实不难，但是如何用更少的时间和空间完成算法是我们更加应该考虑的。

## H_p940_不同的子序列2

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142245732.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142245489.png)

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142245101.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142246883.png)

```python
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        mod = 10**9 + 7
        last = [-1] * 26

        n = len(s)
        f = [1] * n
        for i, ch in enumerate(s):
            for j in range(26):
                if last[j] != -1:
                    f[i] = (f[i] + f[last[j]]) % mod
            last[ord(s[i]) - ord("a")] = i
        
        ans = 0
        for i in range(26):
            if last[i] != -1:
                ans = (ans + f[last[i]]) % mod
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142246896.png)

```python
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        mod = 10**9 + 7

        g = [0] * 26
        for i, ch in enumerate(s):
            total = (1 + sum(g)) % mod
            g[ord(s[i]) - ord("a")] = total
        
        return sum(g) % mod
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142247223.png)

```python
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        mod = 10**9 + 7

        g = [0] * 26
        total = 0
        for i, ch in enumerate(s):
            oi = ord(s[i]) - ord("a")
            g[oi], total = (total + 1) % mod, (total * 2 + 1 - g[oi]) % mod
        
        return total
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210142247834.png)

## M_p1441_用栈操作构建数组

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210151544457.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210151544553.png)

### mine

```python
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        cur = 0
        ans = []
        for i in range(len(target)):
            for _ in range(target[i] - cur - 1):
                ans.extend(["Push", "Pop"])
            ans.append("Push")
            cur = target[i]
        
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210151546291.png)

很简单没啥说的，就判断下一个和上一个是不是相连的就完了。

### summary

模拟一遍就完了。

## M_p886_可能的二分法

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161203145.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161203908.png)

### mine

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = [[] for _ in range(n)]
        for x, y in dislikes:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)

        color = [0] * n
        def dfs(x: int, c: int) -> bool:
            color[x] = c
            for y in g[x]:
                if color[y] == c:
                    return False
                if color[y] == 0 and not dfs(y, -c):
                    return False
            return True

        return all(c or dfs(i, 1) for i, c in enumerate(color))
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161208157.png)

一开始把题理解错了，以为是分很多组，每组里面两个。

如果只是分两组其实比较简单，不会涉及到回溯，直接挨个判断就行了。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161208845.png)

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = [[] for _ in range(n)]
        for x, y in dislikes:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        color = [0] * n  # color[x] = 0 表示未访问节点 x
        for i, c in enumerate(color):
            if c == 0:
                q = deque([i])
                color[i] = 1
                while q:
                    x = q.popleft()
                    for y in g[x]:
                        if color[y] == color[x]:
                            return False
                        if color[y] == 0:
                            color[y] = -color[x]
                            q.append(y)
        return True
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161209855.png)

```python
class UnionFind:
    def __init__(self, n: int):
        self.fa = list(range(n))
        self.rank = [1] * n

    def find(self, x: int) -> int:
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x: int, y: int) -> None:
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return
        if self.rank[fx] < self.rank[fy]:
            fx, fy = fy, fx
        self.rank[fx] += self.rank[fy]
        self.fa[fy] = fx

    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = [[] for _ in range(n)]
        for x, y in dislikes:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        uf = UnionFind(n)
        for x, nodes in enumerate(g):
            for y in nodes:
                uf.union(nodes[0], y)
                if uf.is_connected(x, y):
                    return False
        return True
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202210161209979.png)

### summary

染色法做起来不会难，毕竟只是二分，很好判断。

并查集的方法可以学习在，在这个题中不如染色简单。
