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

## H_p1819_序列中不同最大公约数的数目

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202301141405186.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202301141405178.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202301141405821.png)

### solution

```python
class Solution:
    def countDifferentSubsequenceGCDs(self, nums: List[int]) -> int:
        """
        1.主要在于遍历每个可能的最大公约数x，看是否存在一个子序列满足该子序列的最大公约数为x。
        2.随着子序列长度的增加，该子序列的最大公约数会减小或不变，当最大公约数减小到和x相等时，说明我们找到了一个子序列满足最大公约数为x。
        3.根据最大公约数的性质，我们不用每次都在完整的子序列中进行最大公约数的计算，我们只需要将当前的最大公约数和要新加入子序列的数进行最大公约数计算即可。
        4.对于子序列数组，x倍数的个数增加，那么子序列中最大公因数一定会减小或者不变（也就是保底x），每个kx都要判断一次是否找到最大公约数为x的子序列，找到立刻跳出加强举的例子写为[6,12,15]，x=3时枚举到[6,12]时gcd仍为6，但是[6,12,15]时加入15,gcd为3判断后找到子序列。
        """
        ans, mx = 0, max(nums)
        has = [False] * (mx + 1)
        for x in nums: 
            has[x] = True
        for i in range(1, mx + 1):
            g = 0  # 0 和任何数 x 的最大公约数都是 x
            for j in range(i, mx + 1, i):  # 枚举 i 的倍数 j
                if has[j]:  # 如果 j 在 nums 中
                    g = gcd(g, j)  # 更新最大公约数
                    if g == i:  # 找到一个答案（g 无法继续减小）
                        ans += 1
                        break  # 提前退出循环
        return ans
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202301141534778.png)

## E_p2293_极大极小游戏

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082058714.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082058815.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082058214.png)

### solution

```python
class Solution:
    def minMaxGame(self, nums: List[int]) -> int:
        def min_max(sub_nums: List[int]):
            # 到了最后一步则退出递归
            if len(sub_nums) <= 2:
                return min(sub_nums)
            else:
                temp = []
                # 模拟
                for i in range(0, len(sub_nums), 4):
                    temp.append(min(sub_nums[i], sub_nums[i+1]))
                    temp.append(max(sub_nums[i+2], sub_nums[i+3]))
                return min_max(temp)
        
        return min_max(nums)
```

## M_1813_句子相似性Ⅲ

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082101018.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082101938.png)

### solution

```python
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        """
        1.固定让sentence1更短
        2.分别计算公共前缀和公共后缀的长度
        """
        sentence1 = sentence1.split(' ')
        sentence2 = sentence2.split(' ')
        if len(sentence1) > len(sentence2):
            sentence1, sentence2 = sentence2, sentence1
        
        forward_len = backward_len = 0
        # 计算公共前缀长度
        for i in range(len(sentence1)):
            if sentence1[i] == sentence2[i]:
                forward_len += 1
            else:
                break

        # 计算公共后缀长度,一定要减去前面已经匹配前缀的部分，不然会导致重复计算
        for i in range(1, len(sentence1) + 1 - forward_len):
            if sentence1[-i] == sentence2[-i]:
                backward_len += 1
            else:
                break

        print(forward_len, backward_len)

        return forward_len + backward_len == len(sentence1)
```

## M_p1814_统计一个数组中好对子的数目

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082114290.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082114814.png)

### solution

```python
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        """
        1.核心思想是把原题中的式子改写为nums[i] - rev(nums[i]) == nums[j] - rev(nums[j])
        2.令f(x) = nums[x] - rev(nums[x])
        3.计算nums中每一个f(x)的结果，并记录，看有多少相同的。
        4.因为题目中要求下标对中i<j所以先遍历到的不需要考虑后面的情况，只需要后遍历到的和前面的结果一一配对即可
        """
        res = 0
        cnt = Counter()
        for i in nums:
            j = int(str(i)[::-1])
            res += cnt[i - j]
            cnt[i - j] += 1
        return res % (10 ** 9 + 7)
```

## E_p2299_强密码检验器Ⅱ

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082115274.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082115522.png)

### solution

```python
class Solution:
    def strongPasswordCheckerII(self, password: str) -> bool:
        """
        每个情况判断
        """
        n = len(password)
        small_list = [chr(i) for i in range(97,123)]
        big_list = [chr(i) for i in range(65,91)]
        num_list = [str(i) for i in range(0,10)]
        special_list = list("!@#$%^&*()-+")
        small_flag = big_falg = num_flag = special_flag = False

        if n < 8:
            return False
        
        for i in range(n):
            if i > 0 and password[i] == password[i-1]:
                return False
            if password[i] in small_list:
                small_flag = True
            if password[i] in big_list:
                big_falg = True
            if password[i] in num_list:
                num_flag = True
            if password[i] in special_list:
                special_flag = True
            
        return small_flag and big_falg and num_flag and special_flag
```

### others

```python
class Solution:
    def strongPasswordCheckerII(self, password: str) -> bool:
        if len(password) < 8:
            return False
        
        specials = set("!@#$%^&*()-+")
        hasLower = hasUpper = hasDigit = hasSpecial = False

        for i, ch in enumerate(password):
            if i != len(password) - 1 and password[i] == password[i + 1]:
                return False

            if ch.islower():
                hasLower = True
            elif ch.isupper():
                hasUpper = True
            elif ch.isdigit():
                hasDigit = True
            elif ch in specials:
                hasSpecial = True

        return hasLower and hasUpper and hasDigit and hasSpecial
```

## M_p1817_查找用户活跃分钟数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082119167.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082120449.png)

### solution

```python
class Solution:
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
        ans = [0] * k
        # 用一个哈希表来记录每个用户在哪些分钟内活跃，使用set是为了避免重复计数
        usr_hash = defaultdict(set)

        for usr_id, active_min in logs:
            usr_hash[usr_id].add(active_min)

        for usr_id in usr_hash:
            ans[len(usr_hash[usr_id]) - 1] += 1
        return ans
```

## M_p1824_最小侧跳次数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082128488.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082128993.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082128006.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082129084.png)

### solution

```python
class Solution:
    def minSideJumps(self, obstacles: List[int]) -> int:
        # res里面保存了在当前位置在三个跑道上分别需要的横跨次数
        res = [1, 0, 1]
        for idx in range(1, len(obstacles)):
            # 这里先计算从同一个跑道上直接走到当前位置需要的横跨次数
            for i in range(3):
                # 如果当前跑道上有石头，则当前位置的该条跑道是不可达的
                if i + 1 == obstacles[idx]:
                    res[i] = inf
            
            # 这里再计算从已经走到当前位置从别的跑道横跨过来需要的横跨次数
            for i in range(3):
                # 如果当前跑道上没有有石头，则当前位置最小横跨次数再加一
                if i + 1 != obstacles[idx]:
                    # 这里比大小的两个部分分别是：直着走到当前跑道需要的横跨次数；需要从别的跑道再横跨过来的横跨次数
                    res[i] = min(res[i], min(res) + 1)

        
        return min(res)
```

## H_p1815_得到新鲜甜甜圈的最多组数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082130244.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082130811.png)

### others

```python
class Solution:
    def maxHappyGroups(self, batchSize: int, groups: List[int]) -> int:
        kWidth = 5
        kWidthMask = (1 << kWidth) - 1

        cnt = Counter(x % batchSize for x in groups)

        start = 0
        for i in range(batchSize - 1, 0, -1):
            start = (start << kWidth) | cnt[i]

        @cache
        def dfs(mask: int) -> int:
            if mask == 0:
                return 0

            total = 0
            for i in range(1, batchSize):
                amount = ((mask >> ((i - 1) * kWidth)) & kWidthMask)
                total += i * amount

            best = 0
            for i in range(1, batchSize):
                amount = ((mask >> ((i - 1) * kWidth)) & kWidthMask)
                if amount > 0:
                    result = dfs(mask - (1 << ((i - 1) * kWidth)))
                    if (total - i) % batchSize == 0:
                        result += 1
                    best = max(best, result)

            return best

        ans = dfs(start) + cnt[0]
        dfs.cache_clear()
        return ans
```

## E_p2303_计算应缴税款总额

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082132461.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082133434.png)

### solution

```python
class Solution:
    def calculateTax(self, brackets: List[List[int]], income: int) -> float:
        ans = 0.0
        brackets = [[0, 0]] + brackets
        for i in range(1, len(brackets)):
            # 如果在当前区间内，则要计算该区间内税款并返回
            if income <= brackets[i][0]:
                return ans + (income - brackets[i-1][0]) * brackets[i][1] / 100.0
            # 如果不在当前区间则计算该区间内的全部税款
            else:
                ans += (brackets[i][0] - brackets[i-1][0]) * brackets[i][1] / 100.0
```

## H_p1632_矩阵转换后的秩

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082134602.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082134690.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082134862.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082135769.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082135792.png)

### others

```python
class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        m, n = len(matrix), len(matrix[0])
        uf = UnionFind(m, n)
        for i, row in enumerate(matrix):
            num2indexList = defaultdict(list)
            for j, num in enumerate(row):
                num2indexList[num].append([i, j])
            for indexList in num2indexList.values():
                i1, j1 = indexList[0]
                for k in range(1, len(indexList)):
                    i2, j2 = indexList[k]
                    uf.union(i1, j1, i2, j2)
        for j in range(n):
            num2indexList = defaultdict(list)
            for i in range(m):
                num2indexList[matrix[i][j]].append([i, j])
            for indexList in num2indexList.values():
                i1, j1 = indexList[0]
                for k in range(1, len(indexList)):
                    i2, j2 = indexList[k]
                    uf.union(i1, j1, i2, j2)

        degree = Counter()
        adj = defaultdict(list)
        for i, row in enumerate(matrix):
            num2index = {}
            for j, num in enumerate(row):
                num2index[num] = (i, j)
            sortedArray = sorted(num2index.keys())
            for k in range(1, len(sortedArray)):
                i1, j1 = num2index[sortedArray[k - 1]]
                i2, j2 = num2index[sortedArray[k]]
                ri1, rj1 = uf.find(i1, j1)
                ri2, rj2 = uf.find(i2, j2)
                degree[(ri2, rj2)] += 1
                adj[(ri1, rj1)].append([ri2, rj2])
        for j in range(n):
            num2index = {}
            for i in range(m):
                num = matrix[i][j]
                num2index[num] = (i, j)
            sortedArray = sorted(num2index.keys())
            for k in range(1, len(sortedArray)):
                i1, j1 = num2index[sortedArray[k - 1]]
                i2, j2 = num2index[sortedArray[k]]
                ri1, rj1 = uf.find(i1, j1)
                ri2, rj2 = uf.find(i2, j2)
                degree[(ri2, rj2)] += 1
                adj[(ri1, rj1)].append([ri2, rj2])
        
        rootSet = set()
        ranks = {}
        for i in range(m):
            for j in range(n):
                ri, rj = uf.find(i, j)
                rootSet.add((ri, rj))
                ranks[(ri, rj)] = 1
        q = deque([[i, j] for i, j in rootSet if degree[(i, j)] == 0])
        while q:
            i, j = q.popleft()
            for ui, uj in adj[(i, j)]:
                degree[(ui, uj)] -= 1
                if degree[(ui, uj)] == 0:
                    q.append([ui, uj])
                ranks[(ui, uj)] = max(ranks[(ui, uj)], ranks[(i, j)] + 1)
        res = [[1] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                ri, rj = uf.find(i, j)
                res[i][j] = ranks[(ri, rj)]
        return res

class UnionFind:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.root = [[[i, j] for j in range(n)] for i in range(m)]
        self.size = [[1] * n for _ in range(m)]

    def find(self, i, j):
        ri, rj = self.root[i][j]
        if [ri, rj] == [i, j]:
            return [i, j]
        self.root[i][j] = self.find(ri, rj)
        return self.root[i][j]

    def union(self, i1, j1, i2, j2):
        ri1, rj1 = self.find(i1, j1)
        ri2, rj2 = self.find(i2, j2)
        if [ri1, rj1] != [ri2, rj2]:
            if self.size[ri1][rj1] >= self.size[ri2][rj2]:
                self.root[ri2][rj2] = [ri1, rj1]
                self.size[ri1][rj1] += self.size[ri2][rj2]
            else:
                self.root[ri1][rj1] = [ri2, rj2]
                self.size[ri2][rj2] += self.size[ri1][rj1]
```

## M_p1663_具有给定数值的最小字符串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082136932.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082136177.png)

### solution

```python
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        """
        前面尽可能多的a，后面尽可能多的z
        """
        front = middle = behind = ''
        
        # 在前面放入尽可能多的a
        i = 0
        while (i+1) * 1 + (n-i-1) * 26 >= k and i < n:
            front += 'a'
            i += 1

        k -= i
        n -= i
        # 此时全部填a填满了，直接返回
        if n <= 0:
            return front
        # 在后面放入尽可能多的z
        i = 0
        while (i+1) * 26 <= k and i < n:
            behind += 'z'
            i += 1

        k -= i * 26
        n -= i

        # 前后凑满了则返回
        if n <= 0:
            return front + behind

        # 最后计算中间那个
        middle = chr(ord('a') + k - 1)

        return front + middle + behind
```

### others

```python
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        s = []
        for i in range(1, n + 1):
            lower = max(1, k - (n - i) * 26)
            k -= lower
            s.append(ascii_lowercase[lower - 1])
        return ''.join(s)
```

## E_p2309_兼具大小写的最好英文字母

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082142890.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082142854.png)

### solution

```python
class Solution:
    def greatestLetter(self, s: str) -> str:
        cnt = defaultdict(set)
        ans = ' '
        for i in s:
            cnt[i.lower()].add(i)
            if len(cnt[i.lower()]) == 2 and ord(ans) < ord(list(cnt[i.lower()])[0].upper()):
                ans = list(cnt[i.lower()])[0].upper()

        return ans if ans != ' ' else ''
```

### others

```python
class Solution:
    def greatestLetter(self, s: str) -> str:
        s = set(s)
        for lower, upper in zip(reversed(ascii_lowercase), reversed(ascii_uppercase)):
            if lower in s and upper in s:
                return upper
        return ""
```

## M_p1664_生成平衡数组的方案数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082145831.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082145782.png)

### solution

```python
class Solution:
    def waysToMakeFair(self, nums: List[int]) -> int:
        # 奇数位置和
        sum_odd = sum(nums[1::2])
        # 偶数位置和
        sum_even = sum(nums[::2])
        # 记录走到当前位置时的奇数位置和与偶数位置和
        ans = t_odd = t_even = 0
        for i in range(len(nums)):
            # 如果当前是奇数位置且奇等于偶
            if i % 2 != 0 and t_odd + sum_even - t_even == t_even + sum_odd - t_odd - nums[i]:
                ans += 1
            # 如果当前是偶数位置且奇等于偶
            if i % 2 == 0 and t_even + sum_odd - t_odd == t_odd + sum_even - t_even - nums[i]:
                ans += 1

            t_odd += nums[i] if i % 2 != 0 else 0
            t_even += nums[i] if i % 2 == 0 else 0
        
        return ans
```

## E_p2315_统计星号

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082147145.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082148385.png)

### solution

```python
class Solution:
    def countAsterisks(self, s: str) -> int:
        in_flag = False
        ans = 0
        for letter in s:
            if not in_flag and letter == '*':
                ans += 1
            elif not in_flag and letter == '|':
                in_flag = True
            elif in_flag and letter == '|':
                in_flag = False
        
        return ans
```

## M_p1669_合并两个链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082149035.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082149099.png)

### solution

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        temp = list1
        # 先取到a节点的前一个
        for _ in range(a-1):
            temp = temp.next
        start_Node = temp

        # 取到b节点的后一个
        for _ in range(b-a+2):
            temp = temp.next
        end_Node = temp

        # 开始拼接list2
        start_Node.next = list2
        while list2.next:
            list2 = list2.next
        list2.next = end_Node

        return list1
```

## E_p2319_判断矩阵是否是一个X矩阵

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082151430.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082151905.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082152079.png)

### solution

```python
class Solution:
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        n = len(grid)
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if i == j or (i + j) == (n - 1):
                    if x == 0:
                        return False
                elif x:
                    return False
        return True
```

## E_p2325_解密消息

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082152733.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082153154.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082153836.png)

### solution

```python
class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        # 构造对应表
        real_key = list(dict.fromkeys(list(key)))
        if ' ' in real_key:
            real_key.remove(' ')

        # 根据对应表转换内容
        message = list(message)
        for i in range(len(message)):
            if message[i] != ' ':
                message[i] = chr(ord('a') + real_key.index(message[i]))

        return ''.join(message)
```

## M_p1129_颜色交替的最短路径

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082154740.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082154127.png)

### solution

```python
class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        """
        0.最短路径应该是用广度优先搜索，一层一层遍历，层数就是最短路径数
        1.首先建建立一个图g，里面节点到节点之间的颜色g中的一个元素为(y,color)，即g[x] = (y, color)，表示x到y的线条为color，用0表示红色，1表示蓝色。
        2.使用vis记录出现使用过的边，这里说明一下为什么要记录使用过的边，同时为什么不是记录使用过的节点
            如果不记录使用过的边可能会出现无限循环的情况，而且前面使用过的边后面再使用说明层数变多了，一定不是最短路径
            记录边而不是点是因为有颜色的区别，比如xyz三个点，x到y有红蓝，y到z有蓝，如果我们先搜索了x到y的蓝边并且记录了x，那么就搜索不到z点的路径了。
        3.使用q表示当前层的节点
        4.ans表示答案，开始初始化全为-1
        """
        g=[[] for _ in range(n)]
        for x, y in redEdges:
            g[x].append((y, 0))
        for x, y in blueEdges:
            g[x].append((y, 1))

        ans = [-1 for _ in range(n)]
        level = 0

        # 初始化一个记录见过边的变量
        vis = set()

        # 0到0是随便的，因为从0开始既可以走红也可以走蓝
        q = [(0, 0), (0, 1)]

        # 如果下一层还有未搜索的节点则继续搜索
        while q:
            next_q = []
            # 遍历当前层节点中的每一个节点
            for x, color in q:
                if ans[x] == -1:
                    ans[x] = level
                # 寻找当前节点的所有下一可跳节点
                for p in g[x]:
                    if p[1] != color and p not in vis:
                        vis.add(p)
                        next_q.append(p)
            
            q = next_q
            level += 1

        return ans
```

## M_p1145_二叉树着色游戏

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082155469.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082155389.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082155188.png)

### solution

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def btreeGameWinningMove(self, root: Optional[TreeNode], n: int, x: int) -> bool:
        """
        1.开始的x能将整个树分为三部分：上面和左右，我们需要找到一个部分让他的节点数大于总节点数的一半即可
        2.使用dfs计算每个部分的节点数
        """
        x_node = None

        def subTreeNum(node: TreeNode):
            if not node:
                return 0
            if node.val == x:
                nonlocal x_node
                x_node = node
            return 1 + subTreeNum(node.left) + subTreeNum(node.right)

        
        root_num = subTreeNum(root)

        left_num = subTreeNum(x_node.left)
        if left_num >= (n + 1) / 2:
            return True

        right_num = subTreeNum(x_node.right)
        if right_num >= (n + 1) / 2:
            return True

        father_num = n - left_num - right_num - 1
        if father_num >= (n + 1) / 2:
            return True

        return False
```

## M_p1798_你能构造出连续值的最大数目

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082156540.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082157259.png)

### others

```python
class Solution:
    def getMaximumConsecutive(self, coins: List[int]) -> int:
        """
        首先我们用[l,r]，0≤l,r 表示一段连续的从l到r的连续整数区间，不妨设我们现在能构造出[0,x]区间的整数，现在我们新增一个整数y，那么我们可以构造出的区间为[0,x]和[y,x+y]，那么如果y≤x+1，则加入整数 y后我们能构造出[0,x+y]区间的整数，否则我们还是只能构造出[0,x]区间的数字。因此我们每次从数组中找到未选择数字中的最小值来作为y，因为如果此时的最小值y 都不能更新区间[0,x]，那么剩下的其他元素都不能更新区间[0,x]。那么我们初始从x=0 开始，按照升序来遍历数组nums的元素来作为y，如果y≤x+1 那么我们扩充区间为[0,x+y]，否则我们无论选任何未选的数字都不能使答案区间变大，此时直接退出即可。
        """
        ans = 1
        coins.sort()
        for i in range(len(coins)):
            if coins[i] > ans:
                break
            ans += coins[i]

        return ans
```

## H_p1210_穿过迷宫的最少移动次数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082158823.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082158199.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082158604.png)

### others

```python
class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        n = len(grid)
        dist = {(0, 0, 0): 0}
        q = deque([(0, 0, 0)])

        while q:
            x, y, status = q.popleft()
            if status == 0:
                # 向右移动一个单元格
                if y + 2 < n and (x, y + 1, 0) not in dist and grid[x][y + 2] == 0:
                    dist[(x, y + 1, 0)] = dist[(x, y, 0)] + 1
                    q.append((x, y + 1, 0))
                
                # 向下移动一个单元格
                if x + 1 < n and (x + 1, y, 0) not in dist and grid[x + 1][y] == grid[x + 1][y + 1] == 0:
                    dist[(x + 1, y, 0)] = dist[(x, y, 0)] + 1
                    q.append((x + 1, y, 0))
                
                # 顺时针旋转 90 度
                if x + 1 < n and y + 1 < n and (x, y, 1) not in dist and grid[x + 1][y] == grid[x + 1][y + 1] == 0:
                    dist[(x, y, 1)] = dist[(x, y, 0)] + 1
                    q.append((x, y, 1))
            else:
                # 向右移动一个单元格
                if y + 1 < n and (x, y + 1, 1) not in dist and grid[x][y + 1] == grid[x + 1][y + 1] == 0:
                    dist[(x, y + 1, 1)] = dist[(x, y, 1)] + 1
                    q.append((x, y + 1, 1))
                
                # 向下移动一个单元格
                if x + 2 < n and (x + 1, y, 1) not in dist and grid[x + 2][y] == 0:
                    dist[(x + 1, y, 1)] = dist[(x, y, 1)] + 1
                    q.append((x + 1, y, 1))
                
                # 逆时针旋转 90 度
                if x + 1 < n and y + 1 < n and (x, y, 0) not in dist and grid[x][y + 1] == grid[x + 1][y + 1] == 0:
                    dist[(x, y, 0)] = dist[(x, y, 1)] + 1
                    q.append((x, y, 0))

        return dist.get((n - 1, n - 2, 0), -1)
```

## E_p2331_计算布尔二叉树的值

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082159407.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082159667.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082200744.png)

### solution

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        def getSubResult(node: TreeNode):
            if node.val in [0, 1]:
                return True if node.val == 1 else False
            else:
                if node.val == 2:
                    return getSubResult(node.right) or getSubResult(node.left)
                else:
                    return getSubResult(node.right) and getSubResult(node.left)

        return getSubResult(root)
```

### others

```python
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        if root.left is None:
            return root.val == 1
        if root.val == 2:
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)
        return self.evaluateTree(root.left) and self.evaluateTree(root.right)
```

## M_p1604_警告一小时内使用相同员工卡大于等于三次的人

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082203522.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082203308.png)

### solution

```python
class Solution:
    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        """
        1.首先遍历一遍两个数组，统计每个用户各自的时间，并转化为一天内的分钟数
        2.对每个用户的时间进行排序，看有没有间隔两次的时间的分钟差小于等于60
        """
        usr_hash = defaultdict(list)
        for usr, time in zip(keyName, keyTime):
            time = time.split(":")
            hour, minute = int(time[0]), int(time[1])
            usr_hash[usr].append(hour * 60 + minute)

        ans = []
        for usr, times in usr_hash.items():
            times.sort()
            for i in range(2, len(times)):
                if times[i] - times[i-2] <= 60:
                    ans.append(usr)
                    break

        ans.sort()
        return ans
```

## M_p1233_删除子文件夹

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082204897.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082205248.png)

### solution

```python
class DictTree:
    def __init__(self):
        self.children = dict()
        self.val = -1

class Solution:
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        """
        字典树
        我们可以使用字典树来解决本题。文件夹的拓扑结构正好是树形结构，即字典树上的每一个节点就是一个文件夹。对于字典树中的每一个节点，我们仅需要存储一个变量val，如果val≥0，说明该节点对应着folder[val]，否则（val=−1）说明该节点只是一个中间节点。我们首先将每一个文件夹按照“/” 进行分割，作为一条路径加入字典树中。随后我们对字典树进行一次深度优先搜索，搜索的过程中，如果我们走到了一个val≥0 的节点，就将其加入答案，并且可以直接回溯，因为后续（更深的）所有节点都是该节点的子文件夹。
        """
        root = DictTree()
        for i, path in enumerate(folder):
            path = path.split('/')
            # 先把当前指针指向根目录
            cur = root
            # 开始循环判断当前路径中的每一层有没有被记录在字典树中，并把最后一层节点的val置为i，说明有一个目录介质在该层。这里从1开始是因为最前面有个空值
            for name in path[1:]:
                if name not in cur.children:
                    cur.children[name] = DictTree()
                cur = cur.children[name]
            cur.val = i
        
        ans = []

        def dfs(cur: DictTree):
            # 不为-1说明该目录出现过，同时更深层的目录不用再去遍历了，一定是该目录的子目录
            if cur.val != -1:
                ans.append(folder[cur.val])
                return
            for path in cur.children.values():
                dfs(path)

        dfs(root)
        return ans
```

### others

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302082206375.png)

```python
class Solution:
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        folder.sort()
        ans = [folder[0]]
        for i in range(1, len(folder)):
            if not ((pre := len(ans[-1])) < len(folder[i]) and ans[-1] == folder[i][:pre] and folder[i][pre] == "/"):
                ans.append(folder[i])
        return ans
```

