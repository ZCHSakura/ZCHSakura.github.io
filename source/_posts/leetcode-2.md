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

## M_p1797_设计一个验证系统

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302091716099.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302091716516.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302091716515.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302091717133.png)

### solution

```python
class AuthenticationManager:
    """
    1.self.hash保存所有的token过期时间
    2.过期时间和currentTime一致时认为过期
    """
    def __init__(self, timeToLive: int):
        self.hash = dict()
        self.timeToLive = timeToLive


    def generate(self, tokenId: str, currentTime: int) -> None:
        self.hash[tokenId] = currentTime + self.timeToLive


    def renew(self, tokenId: str, currentTime: int) -> None:
        if tokenId in self.hash and self.hash[tokenId] > currentTime:
            self.hash[tokenId] = currentTime + self.timeToLive


    def countUnexpiredTokens(self, currentTime: int) -> int:
        ans = 0
        for outtime in self.hash.values():
            if outtime > currentTime:
                ans += 1
        return ans


# Your AuthenticationManager object will be instantiated and called as such:
# obj = AuthenticationManager(timeToLive)
# obj.generate(tokenId,currentTime)
# obj.renew(tokenId,currentTime)
# param_3 = obj.countUnexpiredTokens(currentTime)
```

统计当前存活token时可以顺便删去已经过期的token，减少后续的遍历次数。

## H_p1223_掷骰子模拟

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302101912415.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302101912906.png)

### 记忆化深搜

```python
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        """
        先尝试下递归的方法
        输入参数有：剩余骰子次数，上次点数，该点数剩余可重复次数
        """
        MOD = 10 ** 9 + 7
        # 记忆化搜索
        @cache
        def dfs(i: int, last: int, left: int) -> int:
            # 如果i为0了说明找到了一种可行方案
            if i == 0:
                return 1
            res = 0
            # 遍历这一次骰子的所有可能性
            for j, mx in enumerate(rollMax):
                # 如果该次点数和上次不一样，重置left，那么进入到(i−1,j,rollMax[j]−1)
                if j != last:
                    res += dfs(i-1, j, rollMax[j]-1)
                # 如果该次点数和上次一致，且还有剩余可重复次数，那么进入(i−1,j,left−1)
                elif left:
                    res += dfs(i-1,j,left-1)
            return res % MOD

        return sum(dfs(n - 1, j, mx - 1) for j, mx in enumerate(rollMax)) % MOD
```

### 动态dp

```python
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        """
        动态规划
        建立一个三维的动态数组[i][j][x]
        其中i代表当前是第i次掷骰子，j表示本次点数为j，x表示已经连续投掷点数为j的次数
        """
        f = [[[0] * 16 for _ in range(7)] for _ in range(n + 1)]
        for j in range(1, 7):
            f[1][j][1] = 1
        for i in range(2, n + 1):
            for j in range(1, 7):
                for x in range(1, rollMax[j - 1] + 1):
                    # 遍历本轮所有点数可能性
                    for k in range(1, 7):
                        if k != j:
                            f[i][k][1] += f[i - 1][j][x]
                        # 此时k一定等于j
                        elif x + 1 <= rollMax[j - 1]:
                            f[i][k][x + 1] += f[i - 1][k][x]
        mod = 10**9 + 7
        ans = 0
        for j in range(1, 7):
            for x in range(1, rollMax[j - 1] + 1):
                ans = (ans + f[n][j][x]) % mod
        return ans
```

## E_p2335_装满杯子需要的最短总时长

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302112321254.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302112322277.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302112322906.png)

### 贪心+排序

```python
class Solution:
    def fillCups(self, amount: List[int]) -> int:
        """
        排序，每次减少最大的两个
        """
        ans = 0
        while sum(amount):
            amount.sort(reverse=True)
            amount[0] -= 1
            amount[1] = max(0, amount[1] - 1)
            ans += 1
        return ans
```

### 数学

```python
class Solution:
    def fillCups(self, amount: List[int]) -> int:
        """
        理想情况就是一次充两个，除非有一个的数量比另外两个加起来还高。
        """
        amount.sort()
        if amount[2] > amount[1] + amount[0]:
            return amount[2]
        return (sum(amount) + 1) // 2
```

## M_p1138_字母版上的路径

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302122017418.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302122017990.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302122018799.png)

### 模拟（复杂）

```python
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        position_hash = dict()
        board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
        # 先使用哈希表保存所有字母的位置
        for i, row in enumerate(board):
            for j, x in enumerate(row):
                position_hash[x] = (i, j)
        
        def get_path(now, target):
            now, target = position_hash[now], position_hash[target]
            res = ""
            # 先变纵轴
            if (y := target[0] - now[0]) > 0:
                res += 'D' * y
            else:
                res += 'U' * -y
            # 再变横轴
            if (x := target[1] - now[1]) > 0:
                res += 'R' * x
            else:
                res += 'L' * -x
            return res

        # 构造答案
        ans = ''
        for i, letter in enumerate(target):
            # 开头要从'a'
            if i == 0:
                ans = ans + get_path('a', letter) + '!'
            # 如果起点和终点涉及z则要从u处走，但要注意排除zz相连的情况
            elif (letter == 'z' or target[i-1] == 'z') and not target[i-1] == letter:
                ans = ans + get_path(target[i-1], 'u') + get_path('u', letter) + '!'
            else:
                ans = ans + get_path(target[i-1], letter) + '!'

        return ans
```

### 模拟（简洁）

```python
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        cx = cy = 0
        res = []
        for c in target:
            c = ord(c) - ord('a')
            nx = c // 5
            ny = c % 5
            if nx < cx:
                res.append('U' * (cx - nx))
            if ny < cy:
                res.append('L' * (cy - ny))
            if nx > cx:
                res.append('D' * (nx - cx))
            if ny > cy:
                res.append('R' * (ny - cy))
            res.append('!')
            cx = nx
            cy = ny
        return ''.join(res)
```

主要优化两个方面：

- 先向上再向左可以避免z的特殊位置带来的问题
- 可以使用ASCII码之间的数字差来计算他们的相对位置，不需要使用哈希结构来记录每个字母的位置

## H_p10_正则表达式匹配

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141955957.png)

### DP

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141954874.png)

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        """
        这个题中最重要的就是写出状态转移方程
        其次就是一定要注意计算dp数组的时候i要从0开始，j从1开始。这是因为s串为空p不为空有可能是合法的（''和'a*'），p为空而s不为空的话结果一定是不合法的
        """
        m, n = len(s), len(p)

        # matches中的i,j是s和p中的实际下标加1
        def matches(i, j):
            # .和任何字符都能匹配
            if p[j-1] == '.':
                return True
            return s[i-1] == p[j-1]

        # 这里要特别注意，dp的开头加了一个dp[0][0]，所以dp[1][1]才是s和p的首字符
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        # 两个空字符串是可以匹配的
        dp[0][0] = True
        for i in range(m+1):
            for j in range(1, n+1):
                if p[j-1] == '*':
                    if matches(i, j-1):
                        dp[i][j] = dp[i-1][j] or dp[i][j-2]
                    else:
                        dp[i][j] = dp[i][j-2]
                else:
                    if matches(i, j):
                        dp[i][j] = dp[i-1][j-1] 
                    else:
                        dp[i][j] = False

        return dp[m][n]
```

## M_p11_盛最多水的容器

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141956206.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141956410.png)

### 双指针

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        有两个核心问题：
        1.能容纳的水由板子之间的距离和较短的板子高度决定
        2.当板子向内收拢时，若长板向内收拢则水量必然减少，而短板向内收拢水量可能增大，所以每次收拢短板
        """
        l, r = 0, len(height) - 1
        ans = 0
        while l != r:
            ans = max(ans, min(height[l], height[r]) * (r-l))
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        
        return ans
```

## M_P15_三数之和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141957617.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302141957365.png)

### 超时

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()

        ans = []
        l = 0
        while l < len(nums)-1:
            if nums[l] > 0:
                return ans
            r = l + 1
            while r < len(nums):
                # 如果存在满足条件的就记录下来
                if - (nums[l] + nums[r]) in nums[l+1:r]:
                    ans.append([nums[l], - (nums[l] + nums[r]), nums[r]])
                    # 只有匹配成功时右指针向右划过重复的
                    r = bisect.bisect_right(nums, nums[r])
                else:
                    r += 1
            # 左指针向右划过重复的
            l = bisect.bisect_right(nums, nums[l])

        return ans
```

我这种方式双指针指向左右两边，中间的那个每次还得用in来判断，时间复杂度有点大

### [i, L, R]

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        记得避免重复，这里通过使用二分查找来跳过重复
        """
        nums.sort()
        n=len(nums)
        # 特殊情况直接return
        if(not nums or n<3):
            return []
        res=[]
        for i in range(n):
            # 最小的元素已经大于0了，那整体肯定大于0
            if(nums[i]>0):
                return res
            # 遇到重复就continue
            if(i>0 and nums[i]==nums[i-1]):
                continue
            L=i+1
            R=n-1
            while(L<R):
                if(nums[i]+nums[L]+nums[R]==0):
                    res.append([nums[i],nums[L],nums[R]])
                    # 跳过重复部分
                    R = bisect.bisect_left(nums, nums[R])
                    L = bisect.bisect_right(nums, nums[L])
                elif(nums[i]+nums[L]+nums[R]>0):
                    R=R-1
                else:
                    L=L+1
        return res
```

这种方法每次都是确定的i，只用移动L和R就行了，少了一层循环。

## M_p17_电话号码的字母组合

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302142104262.png)

### 递归

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        """
        递归
        """
        phone_map = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }

        ans = []
        temp = []

        def backtrack(index: int):
            # 如果index超越了后界，说明某一个分支已经递归到最深处了，将该分支的答案压入最终的结果中
            if index == len(digits):
                ans.append("".join(temp))
            else:
                digit = digits[index]
                for letter in phone_map[digit]:
                    temp.append(letter)
                    # 进入更深一层
                    backtrack(index + 1)
                    # 每一个分支走完之后往回退一步，走下个分支
                    temp.pop()

        backtrack(0)
        return ans
```

## M_p19_删除链表的倒数第N个节点

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302151658964.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302151658717.png)

### 获取链表长度

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        def getLinkLen(head: Optional[ListNode]):
            """
            获取链表长度
            """
            length = 0
            while head.next:
                head = head.next
                length += 1
            length += 1
            return length

        link_len = getLinkLen(head)
        new_head = ListNode(0, head)
        temp = new_head
        for i in range(link_len - n):
            temp = temp.next
        temp.next = temp.next.next
        return new_head.next
```

## E_p20_有效的括号

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302151719735.png)

### 栈

```python
class Solution:
    def isValid(self, s: str) -> bool:
        """
        栈
        """
        my_dict = {
            '{': '}',
            '[': ']',
            '(': ')',
        }
        stack = []
        for i, x in enumerate(s):
            if x in my_dict:
                stack.append(x)
            else:
                # 这里主要是为了避免出现第一个字符就是右括号的情况下会报错
                if not stack:
                    return False
                if not my_dict[stack.pop()] == x:
                    return False
        
        return not stack
```

## E_p21_合并两个有序链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302151728190.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302151729249.png)

### 遍历

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        temp = head
        # 两个链表都没走完
        while list1 and list2:
            if list1.val >= list2.val:
                temp.next = list2
                list2 = list2.next
            else:
                temp.next = list1
                list1 = list1.next
            temp = temp.next
        
        # 其中一个走完了，把另一个剩下的补在后面
        if list1:
            temp.next = list1
        else:
            temp.next = list2
        
        return head.next
```

### 递归

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

## M_p22_括号生成

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302152109846.png)

### 深搜

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def dfs(cur_str, left, right):
            """
            :param cur_str: 从根结点到叶子结点的路径字符串
            :param left: 左括号已经使用的个数
            :param right: 右括号已经使用的个数
            :return:
            """
            if left == n and right == n:
                res.append(cur_str)
                return
            if left < right:
                return

            if left < n:
                dfs(cur_str + '(', left + 1, right)

            if right < n:
                dfs(cur_str + ')', left, right + 1)

        dfs('', 0, 0)
        return res
```

### 回溯

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def backtrack(cur_list, left, right):
            """
            :param cur_list: 从根结点到叶子结点的路径字符串
            :param left: 左括号已经使用的个数
            :param right: 右括号已经使用的个数
            :return:
            """
            if left == n and right == n:
                res.append("".join(cur_list))
                return
            if left < right:
                return

            if left < n:
                cur_list.append('(')
                backtrack(cur_list, left + 1, right)
                cur_list.pop()

            if right < n:
                cur_list.append(')')
                backtrack(cur_list, left, right + 1)
                cur_list.pop()

        backtrack([], 0, 0)
        return res
```

### 是否回溯的区别

我们可以看到深搜和回溯其实非常的相似，~~回溯使用list而不是str只是单纯的为了方便撤销，没有本质区别。~~这一段是错误的，在python中list是mutable的，如果使用list一定要注意最后要深拷贝赋值给ans，仔细看起来主要就是有没有pop，也就是有没有撤销操作的区别，为什么回溯要撤销而深搜不用撤销呢。

「回溯算法」强调了在状态空间特别大的时候，只用一份状态变量去搜索所有可能的状态，在搜索到符合条件的解的时候，通常会做一个拷贝，这就是为什么经常在递归终止条件的时候，有 `res.append("".join(cur_list))` 这样的代码。正是因为全程使用一份状态变量，因此它就有「恢复现场」和「撤销选择」的需要。

而深搜这份代码里面的区别就是我们传入`dfs`函数中的其实并不一直是一个状态变量，我们使用`dfs(cur_str + '(', left + 1, right)`时，实际上是创建了一个新的变量`cur_str + '('`，当这个dfs返回的时候`cur_str`还是进入`dfs`之前的`cur_str`，也就不用手动撤销选择了。

可以想象搜索遍历的问题其实就像是做实验，每一次实验都用新的实验材料，那么做完了就废弃了。但是如果只使用一份材料，在做完一次以后，一定需要将它恢复成原样（就是这里「回溯」的意思），才可以做下一次尝试。而`dfs(cur_str + '(', left + 1, right)`这种方式实际上相当于找了一份新的和当前一样的材料去进行下一步实验，不管结果如何现在这份材料都不会受到影响。

## H_p23_合并K个升序链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162005293.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162005473.png)

### 不断遍历每个链表的头节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ans = ListNode()
        head = ans
        while None in lists:
            lists.remove(None)

        while lists:
            min_cur = ListNode(inf)
            min_idx = -1
            # 找一圈当前最小的元素
            for i in range(len(lists)):
                # lists.remove(cur)
                if min_cur.val > lists[i].val:
                    min_cur = lists[i]
                    min_idx = i

            # 将一圈找到的最小值放在head后面
            head.next = lists[min_idx]
            head = head.next
            # 最小值那一个链表向后一位，或者从lists中删除
            if lists[min_idx].next:
                lists[min_idx] = lists[min_idx].next
            else:
                lists.pop(min_idx)

        return ans.next
```

### 优先级队列存储头节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        # 优先级队列底层使用的是堆结构，如果升序排列就对应小顶堆，降序排列对应大顶堆。
        # heapq使用小顶堆，即升序排序，如果插入元组则从前面的元素开始排序，可以以此调节元素优先级
        dummy = ListNode(0)
        p = dummy
        head = []

        # 先将所有不为空的链表的首元素如队列
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next

        while head:
            # 取出当前队列中值最小的
            val, idx = heapq.heappop(head)
            # 插入答案中
            p.next = ListNode(val)
            p = p.next
            # 如果最小值所在链表还有后续节点则将该节点加入优先级队列，并向后移一个
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next
```

## M_p31_下一个排列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162054468.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162054858.png)

### 扫描

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。

        同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。
        """

        n = len(nums)
        left, right = n - 2, n - 1
        # 先找到最靠右的一组升序，left的位置就是「较小数」
        while left > 0:
            if nums[left] < nums[right]:
                break
            else:
                left -= 1
                right -= 1
        
        # 保证nums是完全倒序的
        if left == 0 and right != n - 1 and nums[left] > nums[right]:
            nums.sort()
            return

        smaller = nums[left]
        biger_idx = n - 1
        # 寻找「较小数」右侧范围内的「较大数」
        for i in range(right + 1, n):
            if nums[i] <= smaller:
                biger_idx = i - 1
                # 找到最小的「较大数」时停止寻找
                break

        # 交换「较大数」和「较小数」
        nums[left], nums[biger_idx] = nums[biger_idx],  nums[left]
        # 将「较小数」之后的部分调整为升序
        for i in range(right, n - 1):
            for j in range(i+1, n):
                if nums[i] > nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
```

### 优化扫描

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        # 因为后面本身一定是降序的，这里变为升序只需要两两互换，可以O(N)
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

## H_p32_最长有效括号

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162124951.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162124956.png)

### 栈

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # 一开始先放一个占位符，保证栈中首元素为右括号
        stack = [-1]
        ans = 0
        for i in range(len(s)):
            # 如果是左括号则入栈
            if s[i] == '(':
                stack.append(i)
            # 如果是右括号则从栈里面弹出来一个
            else:
                last_idx = stack.pop()
                # 如果此时栈为空了，那说明本次弹出来的是占位的右括号，将本次的右括号放入占位
                if not stack:
                    stack.append(i)
                # 如果没空说明弹出来了一个有效的左括号
                else:
                    ans = max(ans, i - stack[-1])
        
        return ans
```

## M_p33_搜索旋转排序数组

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162201125.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162201308.png)

### 二分

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """
        二分，旋转有序的数组经过二分后至少有一半是有序的，只要区分一下目标是不是在有序的部分就可以了
        """
        left, right = 0, len(nums) - 1
        mid = (left + right) // 2

        if not nums:
            return -1

        while left <= right:
            mid = (left + right) // 2
            print(left, mid, right)
            if nums[mid] == target:
                return mid
            # 如果左边是有序的，这里有等号是因为//除法的结果会更倾向于左边（当数量为偶数时），导致mid可能和left重合
            if nums[left] <= nums[mid]:
                # 且目标在左边
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                # 目标在右边
                else:
                    left = mid + 1
            # 如果右边是有序的
            else:
                # 且目标在右边
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                # 目标在左边
                else:
                    right = mid - 1

        return -1
```

## M_p39_组合总和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162253009.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302162253906.png)

### 深搜

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []

        def dfs(target, combine, idx):
            if idx == len(candidates):
                return 
            if target == 0:
                print('**', target, combine, idx)
                # 重点中的重点，要深拷贝出来一份，不然后面都会变成[]
                ans.append(combine[:])
                return
            # 不使用该位置
            dfs(target, combine, idx+1)
            # 使用该位置
            if target - candidates[idx] >= 0:
                combine.append(candidates[idx])
                dfs(target - candidates[idx], combine, idx)
                combine.pop()
        
        dfs(target, [], 0)

        return ans
```

**一定一定要注意python的list是mutable的，最后一定要深拷贝出来，不然会变成空的。**

## M_p34_在排序数组中查找元素的第一个和最后一个位置

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302171911026.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302171911779.png)

### 二分

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        logN肯定是二分法
        二分两次，一次找到target的第一个位置，一次找到第一个大于target的位置。
        """
        if not nums:
            return [-1, -1]

        def getFirstTarget(nums: List[int], target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                # print(left, mid, right)
                # 如果左右指针重合了并且当前位置元素等于target说明找到了该元素的第一个位置
                if target == nums[mid] and left == right:
                    return mid
                # 没找到但重合了说明没有这个元素
                elif left == right:
                    return -1
                # 这里使用小于等于是为了确保right一定能圈到target值，同时我们为了寻找第一个target希望right在能圈住的情况下尽可能地左缩
                if target <= nums[mid]:
                    right = mid
                # mid处小于target，left可以直接mid+1，此时不会错过第一个target
                else:
                    left = mid + 1

        def getFirstBiger(nums: List[int], target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                # print(left, mid, right)
                # 如果左右指针重合了并且当前位置元素大于target说明找到了大于target元素的第一个位置
                if target < nums[mid] and left == right:
                    return mid
                # 这种情况主要是为了防止这个列表中没有比target元素更大的元素，这时候right和left会汇合在最右端，为了适配后面的-1，这里返回right+1
                elif left == right:
                    return right + 1
                # 要保证right位置上的元素永远大于target
                if target < nums[mid]:
                    right = mid
                # 等号在下面是因为，这里我们希望left尽可能地右缩，同时不必保证target一定被left圈住，所以可以+1
                else:
                    left = mid + 1


        # print(getFirstTarget(nums, target))
        # print(getFirstBiger(nums, target))
        if (first_idx := getFirstTarget(nums, target)) != -1:
            return [first_idx, getFirstBiger(nums, target) - 1]
        return [-1, -1]
```

## H_p42_接雨水

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172044284.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172044613.png)

### 动规

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        """
        动规
        先分别从两边扫描一遍，获取leftMax数组和rightMax数组，其中元素代表着在i位置左边的最大高度和右边的最大高度
        """
        if not height:
            return 0
        
        n = len(height)
        leftMax = [height[0]] + [0] * (n - 1)
        for i in range(1, n):
            leftMax[i] = max(leftMax[i - 1], height[i])

        rightMax = [0] * (n - 1) + [height[n - 1]]
        for i in range(n - 2, -1, -1):
            rightMax[i] = max(rightMax[i + 1], height[i])

        ans = sum(min(leftMax[i], rightMax[i]) - height[i] for i in range(n))
        return ans
```

### 双指针

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        left, right = 0, len(height) - 1
        leftMax = rightMax = 0

        while left < right:
            leftMax = max(leftMax, height[left])
            rightMax = max(rightMax, height[right])
            # 此时leftMax < rightMax，应该移动左指针
            # 此时存在更高的右边界，左指针指的位置到leftMax中间的水必能接住
            if height[left] < height[right]:
                ans += leftMax - height[left]
                left += 1
            # 此时leftMax >= rightMax，应该移动右指针
            # 此时存在更高的左边界，右指针指的位置到rightMax中间的水必能接住
            else:
                ans += rightMax - height[right]
                right -= 1
        
        return ans
```

双指针的优势在于可以使用两个变量来代替两个数组，节省空间开销。

## M_p46_全排列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172053507.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172054198.png)

### 深搜

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        深搜
        """
        def dfs(nums, combine):
            if len(combine) == len(nums):
                # 注意深拷贝，如果ans.append(combine)最后结果全是指向同一个位置的空列表
                ans.append(combine[:])
                return
            for number in nums:
                if number not in combine:
                    combine.append(number)
                    dfs(nums, combine)
                    combine.pop()
        
        ans = []
        dfs(nums, [])
        return ans
```

**注意深拷贝**

## M_p48_旋转图像

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172100913.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172101892.png)

### 翻转

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        先水平翻转再主对角线翻转
        """
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

## M_p49_字母异位词分组

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172111477.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172111563.png)

### 排序

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        直接把每个元素排序，结果相同的放到一个哈希表位置
        """
        hash_dict = collections.defaultdict(list)

        for st in strs:
            key = "".join(sorted(st))
            hash_dict[key].append(st)
        
        return list(hash_dict.values())
```

## M_p53_最大子数组和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172122302.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172122344.png)

### 动规

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """
        从头到尾遍历一遍，计算以每个位置结尾的子数组最大和
        """
        ans = nums[0]
        temp = 0
        for i in range(len(nums)):
            temp = max(nums[i], temp + nums[i])
            ans = max(ans, temp)
        
        return ans
```

## M_p55_跳跃游戏

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172133947.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172133824.png)

### 贪心

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        贪心，记录一个能达到的最远距离，遍历每一个能到达的点的位置加上它自身数值和当前最远距离对比，不断更新最远距离，直到超过n-1或者遍历结束。
        """
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
```

## M_p56_合并区间

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172146526.png)

### 排序

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 先对左端点进行排序，这样能保证最后合并的大区间中的小区间一定是连续的
        intervals.sort(key=lambda x: x[0])
        ans = []
        for i, x in enumerate(intervals):
            if i == 0:
                ans.append(x)
            # 如果当前左端点小于ans中最后一个元素的右端点则说明该小区间可以合并入ans中最后一个区间
            if x[0] <= ans[-1][1]:
                # 判断合并后区间有没有变长
                ans[-1][1] = max(ans[-1][1], x[1])
            # 不能合并就直接加入ans
            else:
                ans.append(x)
        
        return ans
```

## M_p62_不同路径

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172201189.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172201044.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302172201177.png)

### 动规

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        动规
        走到每一个位置可能性只和走到它左边的而可能性和走到它上面的可能性有关
        初始化的时候记得把最左边和最上边初始化为1
        """
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        print(f)
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[m - 1][n - 1]
```

### 数学

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        排列组合
        一共走m + n - 2步，其中n - 1步向右，找出所有不同向右的数量
        """
        return comb(m + n - 2, n - 1)
```

## M_p64_最小路径和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192013267.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192013055.png)

### DP

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        """
        DP
        """
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        # print(dp)
        for row in range(m):
            for col in range(n):
                # 初始位置
                if row == 0 and col == 0:
                    dp[row][col] = grid[row][col]
                # 最上面一行
                elif row == 0:
                    dp[row][col] = dp[row][col - 1] + grid[row][col]
                # 最左边一列
                elif col == 0:
                    dp[row][col] = dp[row - 1][col] + grid[row][col]
                else:
                    dp[row][col] = min(dp[row - 1][col], dp[row][col - 1]) + grid[row][col]
        
        # print(dp)
        return dp[m-1][n-1]
```

## E_p70_爬楼梯

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192032072.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192032234.png)

### DP滚动数组

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        """
        DP滚动数组
        到第x阶的方案数为到x-1和x-2的方案数之和
        """
        # 到第零阶和第一阶的方案数都为1
        p, q = 1, 1
        ans = 1
        # 滚动数组，只保存连续的三个位置p,q,ans
        for i in range(1, n):
            ans = p + q
            p = q
            q = ans
        return ans
```

## H_p72_编辑距离

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192140271.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192140766.png)

### DP

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192141293.jpg)

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        二维DP
        dp[i][j]表示word1的前i个字母和word2的前i个字母的匹配最小操作数
        
        """
        m = len(word1)
        n = len(word2)

        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

        # 初始化dp[0][j]
        for j in range(n+1):
            dp[0][j] = j

        # 初始化dp[i][0]
        for i in range(m+1):
            dp[i][0] = i

        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    # 分别对应对word1增，删，改（相同时不用改）
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] - 1)
                else:
                    # 分别对应对word1增，删，改
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
        
        return dp[m][n]
```

## M_p75_颜色分类

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192231163.png)

### 双指针（快排思想）

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        采用双指针，一个指向2，一个指向0
        注意必须要先比较当前元素是不是2，再比较是不是0，因为我们有可能从后面换过来一个0，但是我们不可能从i走过的地方换过来一个2，所以必须要先比较2再比较0。
        比较2的时候还要注意换过来的是不是2，如果换过来的是2还要接着换，不然i就后移了。
        但是比较0的时候换过来的只可能是1或者i和ptr_0在一个位置，这是因为：
            ptr_0一定是等于或者落后于i的，i已经划过的位置中不可能还有2，只有前面全是0或者把2换了0过来的情况下i才可能一直等于ptr_0，一旦i超过了ptr_0，说明i划过的位置出现过1，而此时ptr_0一定卡在1的那个位置上。
        """
        ptr_0 = 0
        ptr_2 = len(nums) - 1

        i = 0
        while i <= ptr_2:
            # print(ptr_0, i, ptr_2)
            # 如果换过来的是2还要接着换，不然i就后移了
            while i <= ptr_2 and nums[i] == 2:
                nums[i], nums[ptr_2] = nums[ptr_2], nums[i]
                ptr_2 -= 1
            if nums[i] == 0:
                nums[i], nums[ptr_0] = nums[ptr_0], nums[i]
                ptr_0 += 1
            i += 1

        return nums
```

## H_p76_最小覆盖字串

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192258321.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302192258325.png)

### 滑动窗口

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
        滑动窗口
        两个哈希表，一个记录要求，一个记录当前窗口内情况
        """
        if len(s) < len(t):
            return ""

        t_dict = Counter(t)
        s_dict = Counter()

        def check(s_dict, t_dict):
            # print(s_dict, t_dict)
            for key in t_dict:
                # 如果某一个字母没满足要求就返回False
                if t_dict[key] > s_dict[key]:
                    return False
            return True

        left = right = 0
        ans = " " * len(s)
        flag = False
        while right < len(s):
            # print(left, right)
            # 先将右节点加入哈希表中
            s_dict[s[right]] += 1
            # 开始检查，看能否回缩左窗口
            while check(s_dict, t_dict):
                s_dict[s[left]] -= 1
                left += 1
                # 实际上满足条件的是left左边的那个位置
                if len(ans) >= len(s[left-1:right+1]):
                    flag = True
                    ans = s[left-1:right+1]
            # 当前不满足条件，将右窗口扩大
            right += 1

        return ans if flag else ""
```

