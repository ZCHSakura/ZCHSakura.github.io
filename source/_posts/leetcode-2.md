---
title: leetcode_2
date: 2022-11-05 16:36:05
tags: [python, 算法]
top: 1000
categories:
- 算法
---

这里记录在leetcode上做题的经过，包含自己的解法和优秀解法

<!--more-->

## H_p1106_解析布尔表达式

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211051637324.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211051637359.png)

### mine

```python
class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        stack = []
        for i in expression:
            if i == ',':
                continue
            if i != ')':
                stack.append(i)
                continue
            
            t_num = f_num = 0
            # 当遇到）时开始弹出直到第一个（组成一个表达式
            while stack[-1] != '(':
                pop_exp = stack.pop()
                if pop_exp == 't':
                    t_num += 1
                else:
                    f_num += 1
            
            stack.pop()
            pop_symbol = stack.pop()
            if pop_symbol == '!':
                stack.append('t' if f_num == 1 else 'f')
            elif pop_symbol == '&':
                stack.append('t' if f_num == 0 else 'f')
            elif pop_symbol == '|':
                stack.append('t' if t_num > 0 else 'f')

        return stack[-1] == 't'
```

这种表达式的题常用的解法就是栈，通过栈来判断每小式子何时闭合，然后计算完每一个小式子的结果再向后走。

### summary

今天这个题虽然是困难题，但是思路上其实没有太大的难度，比较好想到用栈来解决，只要能用代码实现出来就可以了。

## E_p1678_设计Goal解析器

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081056676.png)

### mine

```python
class Solution:
    def interpret(self, command: str) -> str:
        return command.replace('()', 'o').replace('(al)', 'al')
```

### summary

对python来说两个replace就能解决。

## M_p816_模糊坐标

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081101039.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081101052.png)

### mine

```python
class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        split_list = []
        s = s[1:-1]
        # print(s)
        for i in range(1, len(s)):
            split_list.append((s[:i], s[i:]))
        
        # print(split_list)
        # [('1', '23'), ('12', '3')]

        def set_point(s: str) -> List:
            point_list = []
            # 先计算不加小数点的情况(本身为0，不含前导0)
            if s[0] != '0' or s == '0':
                point_list.append(s)
            # 再计算加小数点的情况
            for p in range(1, len(s)):
                # 两种情况不能加点
                # 1. 以0开头但小数点不在第一位之后
                # 2. 以0结尾
                if p != 1 and s[0] == '0' or s[-1] == '0':
                    continue
                point_list.append(s[:p] + '.' + s[p:])
            return point_list
            

        ans = []
        for i in split_list:
            first_list = set_point(i[0])
            second_list = set_point(i[1])
            # print(first_list, second_list)
            if first_list and second_list:
                for first in first_list:
                    for second in second_list:
                        ans.append('(' + first + ', ' + second + ')')
        
        return ans
```

我做这个题的整体思路就是首先将输入的字符串按照逗号的位置先分成两部分，然后再对每一种划分的两部分计算有多少种合法的可能。

对于判断合法来说有两种情况：

- 本身就合法：这种主要包含两种情况，不含前导0，本身为0
- 加小数点后仍合法：这种包含两种非法情况，除了这两种情况都是能加小数点的
  - 以0开头但小数点位置不在第一位后
  - 末尾为0

### summary

python中逻辑判断符号是有优先级的not>and>or

## E_p1684_统计一致字符串的数目

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081126137.png)

### mine

```python
class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        ans = 0
        for word in words:
            flag = True
            for letter in word:
                if letter not in allowed:
                    flag = False
            if flag:
                ans += 1

        return ans
```

我的思路很简单，就是遍历每一个单词，然后再遍历每一个单词的所有字母，如果这个单词的所有字母都合法就ans加一。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081515022.png)

```python
class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        mask = 0
        for c in allowed:
            mask |= 1 << (ord(c) - ord('a'))
        res = 0
        for word in words:
            mask1 = 0
            for c in word:
                mask1 |= 1 << (ord(c) - ord('a'))
            res += (mask1 | mask) == mask
        return res
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211081515409.png)

### summary

总的来说就是遍历一遍获得答案，但是记录遍历过程的方式比较多，可以使用in判断，使用哈希集合，还可以使用位运算。

## M_p764_最大加号标志

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211092120492.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211092120967.png)

### mine

```python
class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        grid = [[1 for i in range(n)] for i in range(n)]
        for i in mines:
            grid[i[0]][i[1]] = 0

        # 上下左右
        dp = [[[1, 1, 1, 1] for i in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    dp[i][j] = [0, 0, 0, 0]
                else:
                    if j > 0:
                        dp[i][j][2] = dp[i][j-1][2] + 1
                    if i > 0:
                        dp[i][j][0] = dp[i-1][j][0] + 1
        
        for i in range(n-1, -1, -1):
            for j in range(n-1, -1, -1):
                if grid[i][j] == 0:
                    dp[i][j] = [0, 0, 0, 0]
                else:
                    if j < n-1:
                        dp[i][j][3] = dp[i][j+1][3] + 1
                    if i < n-1:
                        dp[i][j][1] = dp[i+1][j][1] + 1
        
        ans = 0
        for i in range(n):
            for j in range(n):
                dp[i][j] = min(dp[i][j])
                ans = max(ans, dp[i][j])

        return ans
```

我的思路是使用动态规划，从左上开始遍历计算每个点上面和左边的1，再从右下开始遍历计算每个点右边和下面的1。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121117920.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121118464.png)

```python
class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        dp = [[n] * n for _ in range(n)]
        banned = set(map(tuple, mines))
        for i in range(n):
            # left
            count = 0
            for j in range(n):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)
            # right
            count = 0
            for j in range(n - 1, -1, -1):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)
        for j in range(n):
            # up
            count = 0
            for i in range(n):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)
            # down
            count = 0
            for i in range(n - 1, -1, -1):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)
        return max(map(max, dp))
```

### summary

我这种方法的空间开销比较大，需要维护一个n\*n\*4的数组，可以参考官解中的方法，虽然思路是动态规划但是没有访问dp数组，而是单独使用了一个变量来存当前方向上1的数量，不同方向上只要保存最小的那个就可以了，所以只用维护一个n\*n大小的数组。

## E_p1704_判断字符串的两半是否相似

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121119022.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121119648.png)

### mine

```python
class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        VOWELS = "aeiouAEIOU"
        a, b = s[:len(s) // 2], s[len(s) // 2:]
        return sum(c in VOWELS for c in a) == sum(c in VOWELS for c in b)
```

### summary

这个比较简单，就前后分开，分别sum就行了。

## M_p790_多米诺和托米诺平铺

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121121128.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121121764.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121121731.png)

### mine

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121123384.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121124544.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121124383.png)

```python
class Solution:
    def numTilings(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        last = [0, 0, 0, 1]
        for i in range(1, n + 1):
            now = [0, 0, 0, 0]
            now[0] = last[3]
            now[1] = (last[0] + last[2]) % MOD
            now[2] = (last[0] + last[1]) % MOD
            now[3] = (((last[0] + last[1]) % MOD + last[2]) % MOD + last[3]) % MOD
            last = now
        return now[3]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121126969.png)

我这部分其实也是参考官解的，主要就是构造一个状态转移方程，但是因为只会用到前一个状态，所以我用循环数组代替了官解中的DP数组，让空间复杂度降到了O(1)。

### others

还可以通过找规律的方法来做

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121125457.png)

```python
MOD = 10 ** 9 + 7

class Solution:
    def numTilings(self, n: int) -> int:
        if n == 1: return 1
        f = [0] * (n + 1)
        f[0] = f[1] = 1
        f[2] = 2
        for i in range(3, n + 1):
            f[i] = (f[i - 1] * 2 + f[i - 3]) % MOD
        return f[n]
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211121126729.png)

### summary

在这次的DP中我吸取了p764题目中官解的做法，虽然我们的整体思路是动态规划，但是我们在计算使用前面状态的时候不一定非要从dp数组中进行读取，尤其是当前状态只与前一状态有关的情况下，我们可以单独用一个变量在遍历的过程中记录前一状态。

## M_p791_自定义字符串排序

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211132020611.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211132021137.png)

### mine

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        order_dict = Counter(order)
        other_list = []
        for letter in s:
            if letter in order:
                order_dict[letter] += 1
            else:
                other_list.append(letter)

        return ''.join([i * (order_dict[i] - 1) for i in order] + other_list)
```

我的思路是先对要排序的字符计数，然后根据计数结果进行字符串构造，最后再把不用排序的字符直接加在后面就完了。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211132022096.png)

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        val = defaultdict(int)
        for i, ch in enumerate(order):
            val[ch] = i + 1
        
        return "".join(sorted(s, key=lambda ch: val[ch]))
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211132023515.png)

### summary

官解那个代码其实可以更简单，不用循环构建排序权重，直接把下标当作权重。

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        return ''.join(sorted(s, key=lambda x: order.index(x) if x in order else 0))
```

## H_p805_数组的均值分割

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211172135852.png)

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211172136593.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211172136576.png)

```python
class Solution:
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return False

        s = sum(nums)
        for i in range(n):
            nums[i] = nums[i] * n - s

        m = n // 2
        left = set()
        for i in range(1, 1 << m):
            tot = sum(x for j, x in enumerate(nums[:m]) if i >> j & 1)
            if tot == 0:
                return True
            left.add(tot)

        rsum = sum(nums[m:])
        for i in range(1, 1 << (n - m)):
            tot = sum(x for j, x in enumerate(nums[m:]) if i >> j & 1)
            if tot == 0 or rsum != tot and -tot in left:
                return True
        return False
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211172137515.png)

这个题有一个非常重要的隐含条件，**分成的两个数组均值相等，那么这两个数组的均值一定等于原始数组的均值。**之后我们再对原始数组中的每一个元素都减去这个均值，那么此时均值就变为了0。

因为我们直接暴力搜索每种组合的均值需要$2^n$种可能性，n最大为30，这个数量级太大了，我们选择将其分为两半，这样分别遍历一半最多只有$2*2^{15}$种可能性，之后在两半中分别挑出一个子集使他们和为0，这样没挑出来的部分和自然也为0。

在遍历的过程中可以通过位运算来优化代码的编写。

### summary

这个题的隐含条件非常重要，如果没有想到这个隐含条件那么这个题就无从下手了，理解到了隐含条件之后发现直接遍历情况太多所以选择分别遍历一半的数据，之后通过位运算来优化代码的编写。

## E_p1710_卡车上的最大单元数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181004758.png)

### mine

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key=lambda x: x[1], reverse=True)
        res = 0
        for numberOfBoxes, numberOfUnitsPerBox in boxTypes:
            if numberOfBoxes >= truckSize:
                res += truckSize * numberOfUnitsPerBox
                break
            res += numberOfBoxes * numberOfUnitsPerBox
            truckSize -= numberOfBoxes
        return res
```

这个题的思路很简单，就是先装单元数量大的箱子，首先按照单元数量排序一次，然后开始放入，直到箱子个数放满就是最大单元数。

### summary

主要技巧在于使用自定义排序

## M_p775_全局倒置与局部倒置

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181008348.png)

### mine

```python
class Solution:
    def isIdealPermutation(self, nums: List[int]) -> bool:
        min_suf = nums[-1]
        for i in range(len(nums) - 2, 0, -1):
            if nums[i - 1] > min_suf:
                return False
            min_suf = min(min_suf, nums[i])
        return True
```

这个题的思路就是看能不能找到不是局部倒置的全局倒置，因为局部倒置一定是全局倒置，如果能找到一个不相邻的全局倒置，那就说明全局一定会比局部多，就可以返回**False**。

为了记录是否出现不相邻的倒置，我们从后开始遍历，记录后面的最小值，一旦遍历过程中出现前面的值大于后面的最小值就可以直接False了。从前遍历记录最大值也是可以的。

## M_p792_匹配子序列的单词数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181020741.png)

### mine

```python
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        pos = defaultdict(list)
        for i, c in enumerate(s):
            pos[c].append(i)
        ans = len(words)
        for w in words:
            if len(w) > len(s):
                ans -= 1
                continue
            p = -1
            for c in w:
                ps = pos[c]
                j = bisect_right(ps, p)
                if j == len(ps):
                    ans -= 1
                    break
                p = ps[j]
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181024541.png)

朴素的想法就是首先遍历一遍字符串s，用一个哈希结构来记录s中每一种字母出现的位置索引。

之后开始遍历words中的每一个word，通过二分查找来寻找一个满足位置索引递增的可能，如果能找到则说明该word是子序列。

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181024620.png)

```python
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        p = defaultdict(list)
        for i, w in enumerate(words):
            p[w[0]].append((i, 0))
        ans = 0
        for c in s:
            q = p[c]
            p[c] = []
            for i, j in q:
                j += 1
                if j == len(words[i]):
                    ans += 1
                else:
                    p[words[i][j]].append((i, j))
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202211181024180.png)

这种是一个多指针共同遍历的方法，整体思路就是在遍历s的过程中看words中的每一个word的当前位置是否为当前s中的字母，如果过相同则word中的指针向后移一位。

具体实现方法如下：

我们先使用一个哈希结构来保存所有word的初始情况（即每个word的第一个元素），使用（i, j）这样一个元组来记录，其中i表示该word在words数组中的下标，j表示该字母在word中的下标。

当我们在遍历s的过程中，我们访问哈希结构的对应字母的所有元组，就可以获取到当前元素为该字母所有word元组，然后逐一判断，长度走完了就ans加1，没走完就将下一个元素的（i, j）元组加入到哈希结构中，记得要清空上一个状态，代码中而可以使用一个辅助变量来记录上一状态，然后直接清空哈希表对应字母记录的元组。

### summary

朴素的办法可以实现题目的需求，而且在思路上也比较简单。多指针的方式更多的是一种思想，最终还是通过哈希结构来记录每个word的当前指向位置。

