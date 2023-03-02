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

**注意浅拷贝**

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

## M_p78_子集

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302201935825.png)

### 回溯

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        回溯
        """
        ans = []
        def dfs(cur, idx):
            if idx == len(nums):
                ans.append(cur[:])
                return
            # 选取当前位置元素
            cur.append(nums[idx])
            dfs(cur, idx + 1)
            cur.pop()
            # 不选取当前位置元素
            dfs(cur, idx + 1)

        dfs([], 0)
        return ans
```

## M_p79_单词搜索

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202108354.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202108817.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202109918.png)

### 回溯

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        回溯
        """
        around_b = [[-1, 0], [1, 0], [0, 1], [0, -1]]
        
        # 当前判断第idx元素，used中保存使用过的位置，i,j为当前坐标
        def dfs(idx, used, i, j):
            if word[idx] != board[i][j]:
                return False

            if idx == len(word) - 1:
                return True
            
            # 先将这个节点加入used中
            used.append([i, j])
            for bi, bj in around_b:
                newi, newj = i+bi, j+bj
                # 遍历当前位置的四个邻居，看有没有没被用过的而且后续能成功的
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]) and [newi, newj] not in used:
                    if dfs(idx+1, used, i+bi, j+bj):
                        return True
            # 如果这个节点四个方向都搜索完了都不行，那就把这个节点弹出去
            used.pop()
            return False

        m, n = len(board), len(board[0])

        # 避免极端情况，看要不要反转word
        head_or_tail = 0
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    head_or_tail += 1
                elif board[i][j] == word[-1]:
                    head_or_tail -= 1

        if head_or_tail > 0:
            word = word[::-1]

        # 以每个点为起点看能不能找到一个答案
        for i in range(m):
            for j in range(n):
                if dfs(0, [], i, j):
                    return True

        return False
```

## H_p84_柱状图中最大的矩形

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202152919.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202152905.png)

### 两次单调栈

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        两次单调栈
        记录以每个位置的高度为高可构建的矩形的左右边界分别在哪里
        """
        n = len(heights)

        if n < 1:
            return 0

        left, right = [0] * n, [0] * n

        stack = []
        # 先从左向右遍历一遍,找到每个heights[i]当作高的左边界在哪里
        for i in range(n):
            # 找到一个高度比当前位置小的位置作为左边界,这就是单调栈维护单调递增的意义
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            left[i] = stack[-1] if stack else -1
            stack.append(i)

        stack = []
        # 再从右向左遍历一遍,找到每个heights[i]当作高的右边界在哪里
        for i in range(n-1, -1, -1):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            right[i] = stack[-1] if stack else n
            stack.append(i)

        print(left)
        print(right)
        
        return max([(right[i] - left[i] - 1) * heights[i] for i in range(n)])
```

### 一次单调栈

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        一次单调栈
        我们在对位置i进行入栈操作时，确定了它的左边界。从直觉上来说，与之对应的我们在对位置 i进行出栈操作时可以确定它的右边界！仔细想一想，这确实是对的。当位置i被弹出栈时，说明此时遍历到的位置i0的高度小于等于height[i]，并且在i0​与i之间没有其他高度小于height[i]的柱子。
        """
        n = len(heights)
        left, right = [0] * n, [n] * n

        mono_stack = list()
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                # 被弹出的时候记录右边界,但是我们这里只要相等就会弹出,所以当有一排高度相等的时候可能无法获取到正确的右边界,但其实不会影响最终结果.
                #在答案对应的矩形中，如果有若干个柱子的高度都等于矩形的高度，那么最右侧的那根柱子是可以求出正确的右边界的
                right[mono_stack[-1]] = i
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)
        
        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans
```

## H_p85_最大矩阵

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202247503.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202247478.png)

### 每一行向上截取，转化为p84

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        """
        使用84题的思路,每一行往上截取连续的1作为柱状图的高
        """
        def largestRectangleArea(heights: List[int]) -> int:
            """
            一次单调栈
            我们在对位置i进行入栈操作时，确定了它的左边界。从直觉上来说，与之对应的我们在对位置 i进行出栈操作时可以确定它的右边界！仔细想一想，这确实是对的。当位置i被弹出栈时，说明此时遍历到的位置i0的高度小于等于height[i]，并且在i0​与i之间没有其他高度小于height[i]的柱子。
            """
            n = len(heights)
            left, right = [0] * n, [n] * n

            mono_stack = list()
            for i in range(n):
                while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                    # 被弹出的时候记录右边界,但是我们这里只要相等就会弹出,所以当有一排高度相等的时候可能无法获取到正确的右边界,但其实不会影响最终结果.
                    #在答案对应的矩形中，如果有若干个柱子的高度都等于矩形的高度，那么最右侧的那根柱子是可以求出正确的右边界的
                    right[mono_stack[-1]] = i
                    mono_stack.pop()
                left[i] = mono_stack[-1] if mono_stack else -1
                mono_stack.append(i)
            
            ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
            return ans

        # 初始化up矩阵,用来记录每个点往上的连续1个数
        up = [[0 for j in range(len(matrix[0]))] for i in range(len(matrix))]
        
        # 计算up矩阵
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (matrix[i][j] == '1'):
                    up[i][j] = up[i-1][j] + 1 if i != 0 else 1

        # print(up)
        ans = 0
        # 取每一行计算柱状图中最大矩阵
        for row in up:
            ans = max(ans, largestRectangleArea(row))

        return ans
```

## E_p94_二叉树的中序遍历

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302202206148.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

## M_p96_不同的二叉搜索树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212135515.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212135125.png)

### 递归+记忆化

```python
class Solution:
    @cache
    def numTrees(self, n: int) -> int:
        """
        递归与记忆化
        以N个节点中的每一个节点为根节点，以该节点为根节点
        的树的数目为其左边节点组成的树数目×右边节点组成的树的数目
        注意记忆化，当输入的n一致时返回树的数量也一致。
        """
        if n == 0 or n == 1:
            return 1

        ans = 0
        for i in range(n):
            left_num = self.numTrees(i)
            right_num = self.numTrees(n - i - 1)
            ans += left_num * right_num

        return ans
```

## M_p98_验证二叉搜索树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212203235.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212203887.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        递归
        每次进行递归的时候要传入下一个节点和当前的上下界
        """
        def check(node: TreeNode, lower: int, upper: int):
            if not node:
                return True
            
            val = node.val
            # 当前节点不满足要求
            if val <= lower or val >= upper:
                return False
            # 左子树不满足
            if not check(node.left, lower, val):
                return False
            # 右子树不满足
            if not check(node.right, val, upper):
                return False

            return True
        
        return check(root, -inf, inf)
```

## E_p101_对称二叉树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212218568.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212218708.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        递归
        复制一份该二叉树，如果对称需要满足两个条件：
        1.根节点的值相同
        2.左子树等于对方右子树，右子树等于对方左子树
          2         2
  		 / \       / \
  		3   4     4   3
	   / \ / \   / \ / \
	  8  7 6  5 5  6 7  8
        """
        if not root:
            return True
        
        def check(node1: TreeNode, node2:TreeNode):
            # 如果都为空
            if (not node1 and not node2):
                return True
            # 有一个为空
            if (not node1 or not node2):
                return False
            # 根节点的值相同;
            # 左子树等于对方右子树，右子树等于对方左子树;
            return node1.val == node2.val and check(node1.right, node2.left) and check(node1.left, node2.right)

        return check(root, root)
```

## M_p102_二叉树的层序遍历

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212238981.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212239711.png)

### 队列迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        队列
        from queue import Queue
        初始化队列：node_list = Queue()
        返回队列长度：node_list.qsize()
        入队：node_list.put(node.left)
        出队：node = node_list.get()
        """
        if not root:
            return []

        from queue import Queue
        # 初始化一个先进先出队列
        node_list = Queue()
        ans = []
        node_list.put(root)
        while not node_list.empty():
            # 存该层结果
            temp = []
            node_num = node_list.qsize()
            # 遍历该层所有的节点，并将子节点压入队列
            for _ in range(node_num):
                node = node_list.get()
                temp.append(node.val)
                # 先如左
                if node.left:
                    node_list.put(node.left)
                # 再入右
                if node.right:
                    node_list.put(node.right)

            ans.append(temp)
        
        return ans
```

## E_p104_二叉树的最大深度

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212241043.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root):
        """
        递归
        分别计算左右子树深度，取大的+1
        """
        if root is None: 
            return 0 
        else: 
            left_height = self.maxDepth(root.left) 
            right_height = self.maxDepth(root.right) 
            return max(left_height, right_height) + 1 
```

## M_p105_从前序和中序遍历序列构造二叉树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212302368.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302212302617.png)

### 递归+分支

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        递归+分支
        先序遍历定位根节点
        中序遍历定位左右子树范围
        """
        # 构造哈希映射，帮助我们快速定位根节点
        index = {element: i for i, element in enumerate(inorder)}

        def subBuildTree(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:
                return None

            # 先序遍历的第一个就是根节点
            preorder_root = preorder_left
            
            # 查询根节点在中序遍历中的位置
            inorder_root = index[preorder[preorder_root]]

            # 建造根节点
            root = TreeNode(preorder[preorder_root])

            # 计算左子树节点数
            num_leftsub = inorder_root - inorder_left

            # 构造左子树
            root.left = subBuildTree(preorder_left + 1, preorder_left + num_leftsub, inorder_left, inorder_root - 1)

            # 构造右子树
            root.right = subBuildTree(preorder_left + num_leftsub + 1, preorder_right, inorder_root + 1, inorder_right)

            return root
        
        n = len(preorder)
        return subBuildTree(0, n-1, 0, n-1)
```

## M_p114_二叉树展开为链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222007080.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222007086.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return root

        ans = TreeNode()
        temp = ans
        def dfs(node: TreeNode):
            nonlocal temp 
            if not node:
                return

            temp.right = TreeNode(node.val)
            temp.left = None
            temp = temp.right

            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)

        dfs(root)
        root.left = None
        root.right = ans.right.right
```

### 寻找前驱节点（原地）

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222007156.png)

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        curr = root
        while curr:
            if curr.left:
                predecessor = nxt = curr.left
                while predecessor.right:
                    predecessor = predecessor.right
                predecessor.right = curr.right
                curr.left = None
                curr.right = nxt
            curr = curr.right
```

## E_p121_买卖股票的最佳时机

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222008463.png)

### 遍历

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        遍历
        使用一个值记录当前遇到的最小值，每次和当前最小值比较，更新答案
        """
        min_num = inf
        ans = 0

        for i in range(len(prices)):
            min_num = min(min_num, prices[i])
            if prices[i] - min_num > ans:
                ans = prices[i] - min_num

        return ans
```

## H_p124_二叉树中的最大路径和

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222028980.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222029171.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
        递归
        递归的过程中计算每个节点的贡献度，也就是该节点作为子节点时能提供的最大路径和
        这里要注意贡献度和最大路径和是不一样的，路径中的根节点的最大路径和可以同时包含左右节点的贡献度
        但是一个节点作为子节点时意味着他只能带左右孩子中的一个，那要选择贡献度大的孩子
        """
        max_path = -inf

        def get_contribution(node: TreeNode) -> int:
            """
            该函数的返回值为该node节点作为子节点时的贡献度，而不是最大路径和！
            """
            # 空节点的贡献度为0
            if not node:
                return 0

            # 如果左儿子贡献度小于0那不如直接不选
            left_contr = max(0, get_contribution(node.left))
            right_contr = max(0, get_contribution(node.right))

            nonlocal max_path
            # 这里记录的是该node节点作为根节点时（即左右儿子都能选）的最大路径和
            max_path = max(max_path, node.val + left_contr + right_contr)

            # 函数返回的是该节点的贡献度，只能左右孩子中取其一
            return node.val + max(left_contr, right_contr)
        
        get_contribution(root)
        return max_path
```

## M_p128_最长连续序列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222118083.png)

### 遍历+优化

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        """
        遍历并优化
        以每一个元素为序列开头，向后寻找有没有连续的序列
        同时我们在外层循环中做了判断，当前元素为x，如果x-1在set中则不用将x作为序列开头来寻找序列
        这是因为x-1在set中时，要么x已经被记录过，要么将来x-1做序列头时会被记录，且一定比x-1开头要短，所以没必要寻找
        因此虽然我们代码有两重循环，但实际上每个元素在内循环中最多出现一次，仍然能保证时间是O(n)
        """
        # in 查询set的时间复杂度为O(1)
        nums = set(nums)
        ans = 0

        for x in nums:
            if x - 1 not in nums:
                cur = x
                cur_len = 1
                while cur + 1 in nums:
                    cur += 1
                    cur_len += 1
                ans = max(ans, cur_len)
            
        return ans
```

### 遍历+哈希

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        """
        用哈希表存储每个端点值对应连续区间的长度
        若数已在哈希表中：跳过不做处理
        若是新数加入：
            取出其左右相邻数已有的连续区间长度 left 和 right
            计算当前数的区间长度为：cur_length = left + right + 1
            根据 cur_length 更新最大长度 max_length 的值
            更新区间两端点的长度值为cur_length
        
        注：这里说明为什么只更新端点处长度值，这是因为新出现的节点若能和当前区间结合则
        一定出现在当前区间端点左右，所以每次只用更新端点处长度即可。

        """
        # 哈希表中存储当前节点所在区间的区间长度值
        hash_dict = dict()
        ans = 0
        for x in nums:
            # 如果x不在hash中
            if x not in hash_dict:
                # 取其左右相邻区间长度，没有就没0
                left_len = hash_dict.get(x - 1, 0)
                right_len = hash_dict.get(x + 1, 0)

                # 计算当前区间长度
                cur_length = left_len + 1 + right_len

                # 更新最大区间长度
                ans = max(ans, cur_length)

                # 更新区间端点长度值
                hash_dict[x] = cur_length
                hash_dict[x-left_len] = cur_length
                hash_dict[x+right_len] = cur_length
            
        return ans
```

## E_p136_只出现一次的数字

### 神之异或！！！

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        异或
        异或具有交换律和结合律，a^b^a=a^a^b=b
        最后会变成0^ans，最后答案还是ans
        python中的reduce函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
            用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
            得到的结果再与第三个数据用 function 函数运算，最后得到一个结果
        """
        return reduce(lambda x, y: x ^ y, nums)
```

## M_p139_单词拆分

### DP

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        DP
        1.初始化dp=[False,⋯,False]，长度为n+1。n为字符串长度。dp[i] 表示s的前i位(s[0,⋯,i))是否可以用wordDict中的单词表示。
        2.初始化dp[0]=True，空字符可以被表示。
        3.遍历字符串的所有子串，遍历开始索引i，遍历区间[0,n)：
            遍历结束索引j，遍历区间[i+1,n+1)：
                若dp[i]=True 且s[i,⋯,j) 在wordlist 中：dp[j]=True。
                解释：dp[i]=True 说明s的前i位可以用wordDict表示，则s[i,⋯,j) 出现在wordDict中，说明s的前j位可以表示。
        4.返回dp[n]
        注意dp[i]和s[i]中的i是不一致的
        """
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(n):
            for j in range(i+1, n+1):
                if dp[i] and (s[i:j] in wordDict):
                    dp[j] = True
        
        return dp[n]
```

## E_p141_环形链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222202806.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302222202898.png)

### 快慢指针（Floyd判圈算法）
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        快慢指针（Floyd判圈算法）
        两个指针，快指针每次走两步，慢指针每次走一步。
        最终只有两种情况，快指针走到头，或者快指针遇到了慢指针。
        前者说明该链表是无圈的，后者则说明该链表有圈
        """
        if not head or not head.next:
            return False

        slow_ptr = head
        fast_ptr = head.next

        # 只要没相遇就一直走
        while slow_ptr != fast_ptr:
            # 快指针走到头
            if not fast_ptr or not fast_ptr.next:
                return False

            slow_ptr = slow_ptr.next
            fast_ptr= fast_ptr.next.next
        
        return True
```

## M_p142_环形链表Ⅱ

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231826306.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231826212.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231827616.png)

### 哈希O(n)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
<<<<<<< HEAD
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        快慢指针（Floyd判圈算法）
        两个指针，快指针每次走两步，慢指针每次走一步。
        最终只有两种情况，快指针走到头，或者快指针遇到了慢指针。
        前者说明该链表是无圈的，后者则说明该链表有圈
        """
        if not head or not head.next:
            return False

        slow_ptr = head
        fast_ptr = head.next

        # 只要没相遇就一直走
        while slow_ptr != fast_ptr:
            # 快指针走到头
            if not fast_ptr or not fast_ptr.next:
                return False

            slow_ptr = slow_ptr.next
            fast_ptr= fast_ptr.next.next
        
        return True
=======
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        哈希
        """

        hash_node = set()

        while head:
            if head not in hash_node:
                hash_node.add(head)
            else:
                return head
            head = head.next
```

### 双指针O(1)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        双指针
        快慢指针一开始都从head开始，快指针每次走两步，慢指针每次走一步
        设链表共有a+b个节点，其中链表头部到链表入口有a个节点（不计链表入口节点），链表环有b个节点
        fast的步数是slow的两倍：f=2s
        fast的步数是slow步数加n倍的圈长：f=s+nb
        所以s=nb，当s=a+nb的时候慢指针就回到了环的入口，所以需要让慢指针再走a步
        此时我们将快指针移回head让并让他每次走一步，这样当快指针走了a步之后他会和慢指针在环入口处相遇
        """

        fast, slow = head, head

        # 题目保证有环
        while True:
            if not fast or not fast.next:
                return
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        # 此时s=nb，把fast放回head上走a步
        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next

        return fast
```

## M_p146_LRU缓存

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231953559.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231953305.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302231954452.png)

### 哈希+双向链表

```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    """
    哈希+双向链表
    哈希表存的是双向链表的节点对象
    每次get后，移到头节点
    每次put后，移到头节点，如果长度溢出需要删除尾节点
    """

    def __init__(self, capacity: int):
        self.hash = dict()
        # 伪头部和伪尾部节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0


    def get(self, key: int) -> int:
        """
        先通过hash判断在不在链表中
        如果在链表中则返回节点value，同时将节点移至头部
        """
        if key not in self.hash:
            return -1
        node = self.hash[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        先通过hash判断在不在链表中
            如果在链表中则更新节点value，同时将节点移至头部
            如果不在则插入，同时判断size是否超过capacity
                如果超过则移除尾部节点，同时删除hash中对应内容
        """
        if key in self.hash:
            node = self.hash[key]
            node.value = value
            self.moveToHead(node)

        else:
            node = DLinkedNode(key, value)
            self.hash[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                removed_node = self.removeTail()
                self.hash.pop(removed_node.key)
                self.size -= 1


    def addToHead(self, node: DLinkedNode):
        """
        在头部增加一个节点
        """
        node.next = self.head.next
        self.head.next = node
        node.next.prev = node
        node.prev = self.head

    def removeNode(self, node: DLinkedNode):
        """
        删除当前节点
        """
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def moveToHead(self, node: DLinkedNode):
        """
        在原位置删除，在头节点新增
        """
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

## M_p148_排序链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302232024691.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302232024201.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302232024257.png)

### 自顶向下，递归版归并

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        """
        1.找到链表的中点，以中点为分界，将链表拆分成两个子链表。
        寻找链表的中点可以使用快慢指针的做法，快指针每次移动2步，慢指针每次移动1步，
        当快指针到达链表末尾时，慢指针指向的链表节点即为链表的中点。
        2.对两个子链表分别排序。
        3.将两个排序后的子链表合并，得到完整的排序后的链表。
        可以使用「21. 合并两个有序链表」的做法，将两个有序的子链表进行合并。
        """
        def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
            if not head:
                return head
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))
            
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
        
        return sortFunc(head, None)
```

### 自底向上归并，空间O(1)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        """
        1.用subLength表示每次需要排序的子链表的长度，初始时subLength=1。
        2.每次将链表拆分成若干个长度为subLength的子链表（最后一个子链表的长度可以小于subLength），
        按照每两个子链表一组进行合并，合并后即可得到若干个长度为subLength×2
        的有序子链表（最后一个子链表的长度可以小于subLength×2）。
        合并两个子链表仍然使用「21. 合并两个有序链表」的做法。
        3.将subLength的值加倍，重复第 2 步，对更长的有序子链表进行合并操作，
        直到有序子链表的长度大于或等于length，整个链表排序完毕。
        """
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
        
        if not head:
            return head
        
        length = 0
        node = head
        while node:
            length += 1
            node = node.next
        
        dummyHead = ListNode(0, head)
        subLength = 1
        while subLength < length:
            prev, curr = dummyHead, dummyHead.next
            while curr:
                head1 = curr
                for i in range(1, subLength):
                    if curr.next:
                        curr = curr.next
                    else:
                        break
                head2 = curr.next
                curr.next = None
                curr = head2
                for i in range(1, subLength):
                    if curr and curr.next:
                        curr = curr.next
                    else:
                        break
                
                succ = None
                if curr:
                    succ = curr.next
                    curr.next = None
                
                merged = merge(head1, head2)
                prev.next = merged
                while prev.next:
                    prev = prev.next
                curr = succ
            subLength <<= 1
        
        return dummyHead.next
```

## M_p152_乘积最大子数组

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302241945767.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302241946382.png)

### DP

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        """
        DP
        由于存在负数，如果只记录每个位置结尾的最大值的话可能会漏掉很小的负值，
        而这个值在后面如果再遇到负值可能会变得很大。
        所以需要两个动态数组分别记录每个位置结尾的子数组的最小值和最大值。
        又由于只与前一个位置有关所以只用记录两个变量即可。
        """
        max_dp = min_dp = ans = nums[0]
        for i in range(1, len(nums)):
            mx, mn = max_dp, min_dp
            max_dp = max(nums[i], mx * nums[i], mn * nums[i])
            min_dp = min(nums[i], mx * nums[i], mn * nums[i])
            ans = max(ans, max_dp)
        
        return ans
```

## M_p155_最小栈

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242004021.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242004914.png)

### List实现栈

```python
class MinStack:
    """
    list实现栈
    不能直接维护一个最小值，不然把最小值弹出去了之后不知道次小值是多少
    把握栈先进后出的思想，每一个位置的元素都会知道当自己作为栈顶的时候栈的最小值
    每次压栈的时候压一个元组(val, min_value)
    """

    def __init__(self):
        self.stack = [(-1, inf)]

    def push(self, val: int) -> None:
        min_value = min(val, self.stack[-1][1])
        self.stack.append((val, min_value))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

## E_p160_相交链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242039650.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242041216.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242041967.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242041696.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242042041.png)

### 我吹过你吹过的晚风

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        走到尽头见不到你，于是走过你来时的路，等到相遇时才发现，你也走过我来时的路。
        若相遇:a+c+b = b+c+a
        若不相遇:a+b = b+a
        """
        ptr_a, ptr_b = headA, headB
        while ptr_a != ptr_b:
            # 走到头就换路
            if not ptr_a:
                ptr_a = headB
            else:
                ptr_a = ptr_a.next

            if not ptr_b:
                ptr_b = headA
            else:
                ptr_b = ptr_b.next
        
        return ptr_a
```

## E_p169_多数元素

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242120338.png)

### 哈希计数

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        哈希计数
        max()函数可以通过key参数指定排序规则
        """
        counts = Counter(nums)
        return max(counts.keys(), key=counts.get)
```

### 排序

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
    """
    如果将数组nums中的所有元素按照单调递增或单调递减的顺序排序，那么下标为⌊n/2⌋的元素（下标从0开始）一定是众数。
    """
        nums.sort()
        return nums[len(nums) // 2]
```

### 投票

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """
        投票算法
        有一个团伙在擂台上
        遍历每个人：
            如果发现当前擂台上是自己人那就自己上台，台上人数加1
            如果发现台上不是自己人就上去拉下来一个，台上人数减1
            最终还能站在台上的一定是多数元素
            因为本身就会有小党派间的争斗会消耗人数，就算只有两个党派这样下来站在台上的一定也是多数人的党派
        """
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
```

## M_p198_打家劫舍

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242259904.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302242259789.png)

### DP

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        """
        DP
        dp[i]表示前i间房屋能偷窃到的最高总金额
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        只和前两个相邻位置有关，可以使用两个变量代替
        """
        if len(nums) <= 2:
            return max(nums)

        first = nums[0]
        second = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            temp = max(second, first + nums[i])
            first = second
            second = temp

        return second
```

## M_p200_岛屿数量

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251404281.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251404031.png)

### DFS

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        DFS
        每一次dfs将一片岛屿置为0
        """
        row = len(grid)
        col = len(grid[0])


        def dfs(r, c):
            """
            深搜目的是将一片岛屿置0
            """
            grid[r][c] = '0'
            for x, y in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if 0 <= x <= row-1 and 0 <= y <= col-1 and grid[x][y] == '1':
                    dfs(x, y)
        
        ans = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    # 深搜了几次就有几个岛屿
                    dfs(r, c)
                    ans += 1
        
        return ans
```

## E_p206_反转链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251452504.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251453893.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251453607.png)

### 迭代

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        迭代
        1 2 3 none
        3 2 1 none
        """
        # ans和cur用来构造新链表
        ans = None
        cur = head
        while cur:
            # 提前保存一个cur的后续节点，原始的cur马上会被用掉
            # next_node始终位于原始链表
            next_node = cur.next
            # 将当前答案放到cur后边
            cur.next = ans
            # 将答案指针指向cur
            ans = cur
            # 把cur取回到原始链表上的下一个节点
            cur= next_node

        return ans
```

### 递归

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        递归
        """
        if head == None or head.next == None:
            return head
        newHead = self.reverseList(head.next)
        # print(newHead)
        # 此时head右边已经被反转完成，我们希望右边的next是head
        head.next.next = head
        # 最后要指向None，不然可能会产生环
        head.next = None
        return newHead
```

## M_p207_课程表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251543180.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251543735.png)

### 拓扑排序

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        拓扑排序
        1.根据依赖关系，构建邻接表、入度数组。
        2.选取入度为 0 的数据，根据邻接表，减小依赖它的数据的入度。
        3.找出入度变为 0 的数据，重复第 2 步。
        4.直至所有数据的入度为 0，得到排序，如果还有数据的入度不为 0，说明图中存在环。
        """
        # 邻接矩阵
        adjacency_dict = defaultdict(list)
        # 入度数组
        indeg = [0] * numCourses
        # y->x
        for x, y in prerequisites:
            adjacency_dict[y].append(x)
            indeg[x] += 1
        
        # 定义一个队列，里面保存所有当前入度为空（即没有前置要求，或已满足前置要求）的节点，
        q = deque([u for u in range(numCourses) if indeg[u] == 0])

        # 记录能成功完成学习的门数
        visited = 0

        while q:
            visited += 1
            # 弹出一个入度为0的课程y，并完成该课程学习
            y = q.popleft()
            # 所有需要提前学习y的课程入度减1
            for x in adjacency_dict[y]:
                indeg[x] -= 1
                # 如果一个可能入度减为0，则将该课程也入队列
                if indeg[x] == 0:
                    q.append(x)

        return visited == numCourses
```

## M_p208_实现Trie(前缀树)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251613240.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251614425.png)

### 字典树

```python
class Trie:
    """
    该字典树与普通树不同
    每个节点维护一个是否是单词结尾的标志
    每个节点维护一个26种可能的数组
    """
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False
    
    def searchPrefix(self, prefix: str) -> "Trie":
        node = self
        for p in prefix:
            idx = ord(p) - ord('a')
            # 如果下一个查询的单词不在字典树中，返回空节点
            if not node.children[idx]:
                return None
            node = node.children[idx]
        return node

    def insert(self, word: str) -> None:
        node = self
        for p in word:
            idx = ord(p) - ord('a')
            # 如果下一个查询的单词不在字典树中，构造节点
            if not node.children[idx]:
                node.children[idx] = Trie()
            node = node.children[idx]
        node.isEnd = True
        print(self.children)

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        return self.searchPrefix(prefix) is not None


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

## M_p215_数组中的第K个最大元素

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251742494.png)

### 快排分区

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        快速排序的分区思想，快排的思想是一次找出一个数的正确位置，
        并使得该数左边的元素都比它小，该数右边的元素都比它大，要找出第k
	    大的元素，只需要在快排的时候采用降序排序，找到下标为k-1的元素即可。
        """
        def quickPartition(start, end, target):
            """
            在[start, end)中寻找下标为target的元素，降序
            """
            # 随机选取一个元素进行定位
            random_idx = random.randint(start, end-1)
            nums[random_idx], nums[start] = nums[start], nums[random_idx]

            cur_idx = start
            base = nums[start]
            # 遍历整个[start+1, end)数组，保证所有大于等于base的数都在cur_idx前面
            for i in range(start + 1, end):
                if nums[i] >= base:
                    nums[cur_idx+1], nums[i] = nums[i], nums[cur_idx+1]
                    cur_idx += 1
            
            # 把base换到cur_idx位置
            nums[cur_idx], nums[start] = nums[start], nums[cur_idx]

            # 如果该次元素位置小于目标，则对右半部分排序
            if cur_idx < target:
                quickPartition(cur_idx+1, end, target)
            # 如果该次元素位置大于目标，则对左半部分排序
            elif cur_idx > target:
                quickPartition(start, cur_idx, target)
            # 如果等于目标位置就不用操作了，说明找到了

        quickPartition(0, len(nums), k-1)
        return nums[k-1]
```

### 优先级队列

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        优先级队列
        遍历一遍nums，维护一个大小为k的小顶堆，最后的堆顶就是答案
        """
        import heapq
        heap = []

        for i in nums:
            # 如果当前入队元素小于k个就直接入队
            if len(heap) < k:
                heapq.heappush(heap, i)
            # 如果已经到达了k个就pushpop
            else:
                heapq.heappushpop(heap, i)
        
        return heapq.heappop(heap)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251804151.png)

### 直接调用优先级队列

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        优先级队列
        遍历一遍nums，维护一个大小为k的小顶堆，最后的堆顶就是答案
        """
        import heapq
        
        # 返回的是前k个最大的
        return heapq.nlargest(k, nums)[-1]
```

## M_p221_最大正方形

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251913829.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251913251.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251913550.png)

### DP

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """
        DP
        dp[i][j]存储以(i, j)为右下角时可构建的最大正方形边长
        dp[i][j]为左边元素，左上元素，上方元素中最小的值+1
        """
        row = len(matrix)
        col = len(matrix[0])

        dp = [[0 for _ in range(col)] for _ in range(row)]
        max_edge = 0

        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    max_edge = max(max_edge, dp[i][j])

        return max_edge * max_edge
```

## E_p226_翻转二叉树

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251923297.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302251923464.png)

### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        递归
        交换自己的左右子树
        """
        if root is None:
            return
        
        # 左右互换
        root.right, root.left = root.left, root.right
        # 两个子树也各自递归
        self.invertTree(root.right)
        self.invertTree(root.left)
        return root
```

## E_p234_回文链表

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262051312.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262051256.png)

### 遍历

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        """
        遍历
        把值复制到数组中，然后翻转数组
        """
        vals = []
        current_node = head
        while current_node is not None:
            vals.append(current_node.val)
            current_node = current_node.next
        return vals == vals[::-1]
```

### 反转后半段输入

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        """
        反转后半段输入
        1.找到前半部分链表的尾节点。
        2.反转后半部分链表。
        3.判断是否回文。
        4.恢复链表。
        5.返回结果。
        该方法虽然可以将空间复杂度降到O(1)，但是在并发环境下，该方法也有缺点。在并发环境下，
        函数运行时需要锁定其他线程或进程对链表的访问，因为在函数执行过程中链表会被修改。
        """
        if head is None:
            return True

        # 找到前半部分链表的尾节点并反转后半部分链表
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # 判断是否回文
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        # 还原链表并返回结果
        first_half_end.next = self.reverse_list(second_half_start)
        return result    

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head):
        previous = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        return previous
```

## M_p236_二叉树的最近公共祖先

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262253093.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262254117.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262254873.png)

### 记录父节点

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        记录祖先
        深搜的过程中记录自己的祖先，最后对比双方祖先
        """
        import copy
        step_1 = []
        step_2 = []

        def dfs(node: 'TreeNode', target: 'TreeNode', visited: list, step: list, flag = False):
            # 搜索剪枝
            if flag:
                return
            # 记录访问过的节点
            if node is not None:
                visited.append(node)

            if node.val == target.val:
                step.extend(visited[:])
                flag = True
                return

            if node.left:
                dfs(node.left, target, visited, step, flag)
                visited.pop()

            if node.right:
                dfs(node.right, target, visited, step, flag)
                visited.pop()

        dfs(root, p, [], step_1, False)
        dfs(root, q, [], step_2, False)

        ans = TreeNode()
        i = 0
        # 对比双方父节点
        while i < min(len(step_2), len(step_1)) and step_1[i] == step_2[i]:
            ans = step_1[i]
            i += 1

        return ans
```

### 直接深搜

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        直接深搜
        深搜的过程找有没有满足题意的答案
        一开始我也会疑惑虽然是深搜，第一次的答案是最深的，但是后面难道不会覆盖掉最深的答案吗
        其实是不会的，这里是因为判断条件十分巧妙
        必须要满足（一个节点在左子树中，一个节点在右子树中）或（一个节点在当前位置，一个节点在子树）
        更上面的父节点其实是不会满足这个条件的，因为两个节点必在其同一个子树中，也就不会覆盖正确答案
        """
        def dfs(node: 'TreeNode', p: 'TreeNode', q: 'TreeNode'):
            if node is None:
                return False

            # 当前位置是否有目标
            inCurrentNode = node.val == p.val or node.val == q.val
            # 左子树中是否有目标
            inLeft = dfs(node.left, p, q)
            # 右子树中是否有目标
            inRight = dfs(node.right, p, q)

            # 更新答案
            # （一个节点在左子树中，一个节点在右子树中）或（一个节点在当前位置，一个节点在子树）
            if (inLeft and inRight) or (inCurrentNode and (inLeft or inRight)):
                nonlocal ans
                ans = node

            return inCurrentNode or inLeft or inRight

        ans = TreeNode()
        dfs(root, p, q)
        return ans
```

### 简洁递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        简洁递归
        配图理解，能往上一直return的一定是p和q分居不同子树的一个公共先祖节点，或者一个本身就是另一个的先祖
        """
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left: return right
        if not right: return left
        return root
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262256757.png)

## M_P238_除自身以外数组的乘积

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262307393.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262307158.png)

### 前缀积，后缀积

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        前缀积和后缀积
        可以分别左右遍历两边，最后再乘起来
        也可以直接在输出数组中计算一边前缀积，然后用一个变量储存后缀积的同时遍历这样可以空间O(1)
        """
        length = len(nums)
        answer = [0]*length

        # 先计算一遍前缀积
        answer[0] = 1
        for i in range(1, length):
            answer[i] = answer[i-1] * nums[i-1]

        # 开始乘后缀积，R记录后缀积
        R = 1
        for i in range(length-1, -1, -1):
            answer[i] *= R
            R *= nums[i]

        return answer
```

## H_p239_滑动窗口最大值

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262341256.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302262341115.png)

### 优先级队列

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        优先级队列
        维护一个大顶堆，每次取窗口内的堆顶元素
        """
        import heapq
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)

        ans = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            # 保证最大元素在窗口内
            while q[0][1] <= i - k:
                heapq.heappop(q)
            ans.append(-q[0][0])
        
        return ans
```

### 单调减双端队列

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        单调减双端队列
        我们可以观察到新入队的元素如果比前面还在窗口内的元素大时，
        前面这些元素就失去了作用，因为他们在窗口内时，后来的更大的元素也一定在窗口内
        """
        from collections import deque
        
        n = len(nums)
        q = deque()
        for i in range(k):
            # 保证单调减
            while q and nums[q[-1]] < nums[i]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            # 保证单调减
            while q and nums[q[-1]] < nums[i]:
                q.pop()
            q.append(i)
            # 弹出过期的最大元素
            while q[0] <= i-k:
                q.popleft()
            # 加入最大值
            ans.append(nums[q[0]])

        return ans
```

## M_p240_搜索二维矩阵Ⅱ

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302270005107.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302270005571.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302270005047.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302270006097.png)

### Z字形查找

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        Z字形查找
        由于矩阵特殊的有序性，从矩阵的最右上角开始取点(i, j)
        若target大于点，则点下移
        若target小于点，则点左移
        直到找到target或点越界
        """
        m, n = len(matrix), len(matrix[0])

        i, j = 0, n-1
        while i < m and j >= 0:
            if target == matrix[i][j]:
                return True
            elif target > matrix[i][j]:
                i += 1
            else:
                j -= 1

        return False
```

## M_279_完全平方数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302272154863.png)

### DP

```python
class Solution:
    def numSquares(self, n: int) -> int:
        """
        DP
        维护一个dp[i]表示整数i需要的最少完全平方数
        dp[i] = min(dp[i-j*j] + 1) , 1 <= j * j <= i
        """
        dp = [inf] * (n+1)
        dp[0] = 0

        for i in range(1, n+1):
            j = 1
            min_temp = inf
            while j * j <= i:
                min_temp = min(min_temp, dp[i-j*j] + 1)
                j += 1
            dp[i] = min_temp

        return dp[n]
```

### BFS

```python
class Solution:
    def numSquares(self, n: int) -> int:
        """
        BFS
        维护一个双端队列，队列里面记录当前值和已经经过的步数(n, step)
        每次从队列头部取出一个元素，然后记录这个元素所有的减去平方数后的可能targets，并将step加1后入队
        可以维护一个visited哈希结构，记录见过的所有可能，因为是BFS，后见的相同可能需要的步数不会更小，所以不用入队
        """
        from collections import deque
        q = deque()
        visited = set()

        q.append((n, 0))

        while q:
            number, step = q.popleft()
            targets = [number - i * i for i in range(1, int(number ** 0.5) + 1)]
            for target in targets:
                if target == 0:
                    return step + 1
                if target not in visited:
                    visited.add(target)
                    q.append((target, step + 1))

        return -1
```

## E_p283_移动零

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282019303.png)

### 双指针

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        双指针
        左指针保证走过的位置不包含0
        右指针每次遇到不为0的数就和左指针位置交换
        要么左右指针齐头并进，说明前面没遇到过0
        要么需要交换时，左指针一定停在0上，左指针左边不可能出现0
        """
        left = right = 0

        while right < len(nums):
            # 如果右指针遇到不为0的shu
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            
            right += 1

        return nums
```

## M_p287_寻找重复数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282058244.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282058073.png)

### 二分

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        """
        二分
        取left和right中间数mid，看输入数组中小于等于mid的数字个数是否大于mid
        如果大于mid则说明在目标在左边，否则目标在右边
        """
        n = len(nums) - 1
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            cnt = 0
            for num in nums:
                if num <= mid:
                    cnt += 1
            # 如果个数大于mid，说明在[left, mid]
            if cnt > mid:
                right = mid
            # 不然在[mid+1, right]
            else:
                left = mid + 1

        return right
```

### Floyd判圈

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        """
        Floyd判圈
        我们对nums数组建图，每个位置i连一条i→nums[i]的边。由于存在的重复的数字target，
        因此target这个位置一定有起码两条指向它的边，因此整张图一定存在环，
        且我们要找到的target就是这个环的入口，那么整个问题就等价于带环链表的环入口

        同时还要注意链表的起点，我们一定要找一个入度为0的点作为起点
        在该题中0位置的入度为，这样可以保证我们的起点一定不在环内
        """
        fast = slow = 0
        # 保证有环
        while True:
            fast = nums[nums[fast]]
            slow = nums[slow]
            if fast == slow:
                break

        fast = 0
        while slow != fast:
            fast = nums[fast]
            slow = nums[slow]

        return fast
```

## H_p297_二叉树的序列化与反序列化

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282218463.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282219761.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282219116.png)

### 先序遍历+递归构造

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    """
    先序遍历+递归构造
    """

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        return str(self._serialize(root, []))
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        return self._deserialize(eval(data))
    
    def _serialize(self, root, str_list):
        """
        先序遍历
        """
        if root is None:
            str_list.append(None)
        else:
            str_list.append(root.val)
            self._serialize(root.left, str_list)
            self._serialize(root.right, str_list)
        
        return str_list

    def _deserialize(self, data: list) -> 'TreeNode':
        """
        根据先序遍历结果递归构造
        """
        if data[0] == None:
            data.pop(0)
            return None
        
        new_node = TreeNode(data[0])
        data.pop(0)
        new_node.left = self._deserialize(data)
        new_node.right = self._deserialize(data)
        return new_node


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```

## M_p300_最长递增子序列

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282311492.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202302282311306.png)

### DP

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        DP
        dp[i]表示以i结尾的最长严格递增子序列的长度（包含i位置）
        dp[i] = max(dp[j]) + 1, 0<=j<i且nums[i]>nums[j]
        """
        if not nums:
            return 0
        
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```

### 贪心+二分

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        贪心+二分
        当新员工的贡献可以替代老员工时就把老员工开了
        以输入序列[0,8,4,12,2] 为例：
        第一步插入0，d=[0]；
        第二步插入8，d=[0,8]；
        第三步插入4，d=[0,4]；
        第四步插入12，d=[0,4,12]；
        第五步插入2，d=[0,2,12]。
        """
        d = []
        for n in nums:
            if not d or n > d[-1]:
                d.append(n)
            else:
                # idx = bisect.bisect_left(d, n)
                left, right = 0, len(d) - 1
                while left < right:
                    mid = (left+right) // 2
                    if d[mid] >= n:
                        right = mid
                    else:
                        left = mid + 1
                d[left] = n
        return len(d)
```

## H_p301_删除无效的括号

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303011923092.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303011923530.png)

### 递归

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        递归
        1.计算要删去多少左括号和多少有括号
        2.写一个函数判断序列是否有效
        3.回溯搜索所有删除括号方式
        """
        # 先计算左右括号各需要删除多少
        lremove, rremove = 0, 0
        for i in s:
            if i == '(':
                lremove += 1
            elif i == ')':
                if lremove > 0:
                    lremove -= 1
                else:
                    rremove += 1

        def isValid(str):
            """
            判断括号序列是否合法
            """
            cnt = 0
            for c in str:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0

        def dfs(s, start, lremove, rremove):
            """
            字符串s，当前判断start位置，左右括号还需要删除lremove和rremove
            """
            if lremove == 0 and rremove == 0:
                if isValid(s):
                    ans.append(s)
                return
            
            for i in range(start, len(s)):
                # 剪枝1，((()这种情况下删除前面三个中的任何一个结果都一样
                if i > start and s[i] == s[i-1]:
                    continue

                # 剪枝2，如果还需要删除的个数比剩下的个数还多肯定不对了
                # 出现这种情况是因为剪枝1，在剪枝1中可能滑过了很多连续括号
                if lremove + rremove > len(s) - i:
                    break

                # 删除左括号
                if lremove > 0 and s[i] == '(':
                    dfs(s[:i] + s[i+1:], i, lremove-1, rremove)

                # 删除右括号
                if rremove > 0 and s[i] == ')':
                    dfs(s[:i] + s[i+1:], i, lremove, rremove-1)
                
                # 不删除括号
                # 这里不用专门写不删除括号的情况，因为本身就在进行for循环
                # start到i中间这一部分在i这个循环里面就是没有删除括号的
                # dfs(s, i+1, lremove, rremove)

        ans = []
        dfs(s, 0, lremove, rremove)
        return ans







class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        递归
        1.计算要删去多少左括号和多少有括号
        2.写一个函数判断序列是否有效
        3.回溯搜索所有删除括号方式
        """
        # 先计算左右括号各需要删除多少
        lremove, rremove = 0, 0
        for i in s:
            if i == '(':
                lremove += 1
            elif i == ')':
                if lremove > 0:
                    lremove -= 1
                else:
                    rremove += 1

        def isValid(str):
            """
            判断括号序列是否合法
            """
            cnt = 0
            for c in str:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0

        def dfs(s, start, lremove, rremove):
            """
            字符串s，当前判断start位置，左右括号还需要删除lremove和rremove
            """
            if lremove == 0 and rremove == 0:
                if isValid(s):
                    ans.append(s)
                return
            
            for i in range(start, len(s)):
                # 剪枝1，((()这种情况下删除前面三个中的任何一个结果都一样
                if i > start and s[i] == s[i-1]:
                    continue

                # 剪枝2，如果还需要删除的个数比剩下的个数还多肯定不对了
                # 出现这种情况是因为剪枝1，在剪枝1中可能滑过了很多连续括号
                if lremove + rremove > len(s) - i:
                    break

                # 删除左括号
                if lremove > 0 and s[i] == '(':
                    dfs(s[:i] + s[i+1:], i, lremove-1, rremove)

                # 删除右括号
                if rremove > 0 and s[i] == ')':
                    dfs(s[:i] + s[i+1:], i, lremove, rremove-1)
                
                # 不删除括号
                # 这里不用专门写不删除括号的情况，因为本身就在进行for循环
                # start到i中间这一部分在i这个循环里面就是没有删除括号的
                # dfs(s, i+1, lremove, rremove)

        ans = []
        dfs(s, 0, lremove, rremove)
        return ans
```

### BFS

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        BFS
        1.写一个函数判断序列是否有效
        2.每次保存上一轮搜索的结果，然后对上一轮已经保存的结果中的每一个字符串
        尝试所有可能的删除一个括号的方法，然后将保存的结果进行下一轮搜索。
        """
        def isValid(str):
            """
            判断括号序列是否合法
            """
            cnt = 0
            for c in str:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0

        ans = []
        last_set = set()
        last_set.add(s)
        while True:
            # 先查看上一轮次中有没有满足要求的，如果有就是有效的最小次数
            for ss in last_set:
                if isValid(ss):
                    ans.append(ss)
            
            # 如果ans中出现了答案，说明上一轮次出现了最优次数
            if ans:
                return ans

            cur_set = set()
            for ss in last_set:
                # 遍历字符串所有的删除一个括号的可能
                for i in range(len(ss)):
                    # 剪枝，((()前三个位置结果一样
                    if i > 0 and ss[i] == ss[i-1]:
                        continue
                    # 删除一个位置的括号
                    if ss[i] in ['(', ')']:
                        cur_set.add(ss[:i] + ss[i+1:])

            last_set = cur_set

        return -1
```

## M_p309_最佳买卖股票时机含冷冻期

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012008754.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012008137.png)

### DP

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        DP
        我们用f[i] 表示第i天结束之后的「累计最大收益」
        f[i][0]: 手上持有股票的最大收益
        f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
        f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
        只与前一天有关，可以只保留三个变量
        """
        if not prices:
            return 0

        f0 = -prices[0]
        # 这里初始化的第0天后的f1实际上不存在的，但是初始化一定不能比f2大，不然影响统一的转移方程
        f1 = -inf
        f2 = 0
        for i in range(len(prices)):
            # 今天过完手上还有股票，说明要么买入了，要么昨天就持有
            temp0 = max(f0, f2 - prices[i])
            # 今天过完手上不持有股票，且冷却，说明今天卖出了
            temp1 = f0 + prices[i]
            # 今天过完手上不持有股票，且没冷却，说明今天啥都没干，且昨天过完就没股票
            temp2 = max(f1, f2)

            f0, f1, f2 = temp0, temp1, temp2

        return max(f0, f1, f2)
```

## H_p312_戳气球

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012038835.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012038943.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012040684.png)

### 记忆化搜索

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012040644.png)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        val = [1] + nums + [1]
        
        @lru_cache(None)
        def solve(left: int, right: int) -> int:
            if left >= right - 1:
                return 0
            
            best = 0
            for i in range(left + 1, right):
                total = val[left] * val[i] * val[right]
                total += solve(left, i) + solve(i, right)
                best = max(best, total)
            
            return best

        return solve(0, n + 1)
```

### 区间DP

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012041871.png)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        rec = [[0] * (n + 2) for _ in range(n + 2)]
        val = [1] + nums + [1]

        for i in range(n - 1, -1, -1):
            for j in range(i + 2, n + 2):
                for k in range(i + 1, j):
                    total = val[i] * val[k] * val[j]
                    total += rec[i][k] + rec[k][j]
                    rec[i][j] = max(rec[i][j], total)
        
        return rec[0][n + 1]
```

## M_p322_零钱兑换

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012116338.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012116946.png)

### 记忆化搜索

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        记忆化搜索
        使用记忆化搜索必须要把状态的答案作为返回值
        """
        @lru_cache(amount)
        def dp(cur) -> int:
            if cur == 0: 
                return 0
            mini = inf
            for coin in coins:
                if cur - coin >= 0:
                    res = dp(cur - coin)
                    # 只有返回来的结果是有效的，并且更优时才更新mini
                    if res >= 0 and mini > res:
                        mini = res + 1
            return mini if mini != inf else -1

        if amount < 1:
            return 0
        return dp(amount)
```

### DP

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        DP
        DP[i]记录amount为i的时候需要的最小硬币数
        """
        dp = [inf] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                # 如果i可以通过之前的某个最小硬币数再加一个硬币，更新i需要的最小硬币数
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        
        return dp[amount] if dp[amount] != inf else -1
```

## M_p337_打家劫舍Ⅲ

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012206904.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012206610.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012206257.png)

### DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        """
        DFS
        对每一个节点进行深搜，返回选他和不选他的两种结果
        """
        def dfs(node: TreeNode) -> tuple:
            if node is None:
                return 0, 0
            
            # 返回左右子树选不选的结果
            left_s, left_ns = dfs(node.left)
            right_s, right_ns = dfs(node.right)

            # 选该节点
            node_s = node.val + left_ns + right_ns
            # 不选该节点
            node_ns = max(left_s, left_ns) + max(right_s, right_ns)

            return node_s, node_ns

        return max(dfs(root))
```

## E_p338_比特位计数

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012215404.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303012215577.png)

### DP+规律

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        """
        DP+规律
        如果 i 为偶数，那么f(i) = f(i/2) ,因为 i/2 本质上是i的二进制左移一位，低位补零
        如果 i 为奇数，那么f(i) = f(i - 1) + 1， 因为如果i为奇数，
        那么 i - 1必定为偶数，而偶数的二进制最低位一定是0，
        那么该偶数 +1 后最低位变为1且不会进位，所以奇数比它上一个偶数bit上多一个1
        """
        ans = [0]
        for i in range(1, n + 1):
            if i % 2 == 0: # 偶数
                ans.append(ans[i//2])
            else: # 奇数
                ans.append(ans[i - 1] + 1)
        return ans
```

## M_p347_前K个高频元素

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022009882.png)



### Counter+排序

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        counter+排序
        """
        from collections import Counter
        cnt = Counter(nums)
        # 使用字典值对字典键进行排序
        sorted_cnt = sorted(cnt.keys(), key=cnt.get, reverse=True)
        return sorted_cnt[:k]
```

### Counter+小顶堆

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        counter+小顶堆
        """
        from collections import Counter
        import heapq

        cnt = Counter(nums)
        heap = []

        for key in cnt:
            # 如果堆内元素小于k个则直接入堆
            if len(heap) < k:
                heapq.heappush(heap, (cnt[key], key))
            # 堆内已经有k个元素，将当前元素与堆顶元素比较，把小的弹出，始终保留最大的k个
            else:
                heapq.heappushpop(heap, (cnt[key], key))

        # 获取结果并反转，原始结果是从小到大
        return [heapq.heappop(heap)[1] for _ in range(k)][::-1]
```

### 快排分区

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def mySort(count, start, end, res, k):
            randIndex = random.randint(start, end - 1)  # 随机挑一个下标作为中间值开始找
            count[start], count[randIndex] = count[randIndex], count[start] # 先把这个随机找到的中间元素放到开头
            
            midVal = count[start][1] # 选中的中间值
            index = start + 1
            for i in range(start + 1, end):
                if count[i][1] > midVal: # 把所有大于中间值的放到左边
                    count[index], count[i] = count[i], count[index]
                    index += 1
            count[start], count[index - 1] = count[index - 1], count[start] # 中间元素归位

            if k < index - start: # 随机找到的top大元素比k个多，继续从前面top大里面找k个
                mySort(count, start, index, res, k)
            elif k > index - start: # 随机找到的比k个少
                for i in range(start, index): # 先把top大元素的key存入结果
                    res.append(count[i][0])
                mySort(count, index, end, res, k - (index - start)) # 继续往后找
            else: # 随机找到的等于k个
                for i in range(start, index): # 把topk元素的key存入结果
                    res.append(count[i][0])
                return
        
        num_map = collections.defaultdict(int)  # 次数字典
        for num in nums:
            num_map[num] += 1
        
        count = list(num_map.items())  # 转换为列表
        res = [] # 结果
        mySort(count, 0, len(count), res, k)  # 迭代函数求前k个
        return res
```

## M_394_字符串解码

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022141853.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022141249.png)

### 栈

```python
class Solution:
    def decodeString(self, s: str) -> str:
        """
        栈
        一个字符栈，一个数字栈
        """
        letter_stack = []
        multiple_stack = []
        ans = []
        # 用来保存未完整的数字
        temp_num = ''

        for i in range(len(s)):
            if s[i].islower():
                # 如果当前栈是空的而且元素是字符，说明不需要重复，直接写入ans
                if len(letter_stack) == 0:
                    ans.append(s[i])
                # 不是空的就入栈
                else:
                    letter_stack.append(s[i])

            
            elif s[i].isdigit():
                # 如果后面也是数字就先等等，组成完整的数字就入multiple_stack
                if i+1<len(s) and s[i+1].isdigit():
                    temp_num = temp_num + s[i]
                # 如果后面不是数字
                else:
                    # 这里就是一个单个数字
                    if len(temp_num) == 0:
                        multiple_stack.append(int(s[i]))
                    # 之前的一组数字
                    else:
                        multiple_stack.append(int(temp_num + s[i]))
                        temp_num = ''

            # 如果当前元素是']'就开始处理栈里面的内容
            elif s[i] == ']':
                temp = []
                while (x := letter_stack.pop()) != '[':
                    temp = [x] + temp
                letter_stack.extend(temp * int(multiple_stack.pop()))
                # 如果multiple_stack里面空了就说明当前栈里面没有需要解码的内容，全部移到ans中
                if len(multiple_stack) == 0:
                    ans += letter_stack
                    letter_stack = []
            
            # 当前元素是'['
            else:
                letter_stack.append(s[i])

        return ''.join(ans)
```

## M_p399_除法求值

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022214318.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022215367.png)

### DFS

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """
        DFS
        使用visit记录每一次计算除法时候遇到过的被除数，
        如果被一个数已经被作为过被除数那就不要让他作为被除数继续dfs
        """
        graph=defaultdict(dict)
        for (a,b),v in zip(equations,values):
            graph[a][b]=v
            graph[b][a]=1/v

        def dfs(s,e):
            if s not in graph or e not in graph:
                return -1
            if s==e:
                return 1
            visited.add(s)
            for i in graph[s]:
                if i==e:
                    return graph[s][i]
                if i not in visited:
                    ans=dfs(i,e)
                    if ans!=-1:
                        return graph[s][i]*ans
            return -1
        res=[]
        for a,b in queries:
            visited=set()
            res.append(dfs(a,b))
        return res
```

### 并查集

https://leetcode.cn/problems/evaluate-division/solution/399-chu-fa-qiu-zhi-nan-du-zhong-deng-286-w45d/

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022217843.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022217586.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022217444.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022217615.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022218464.png)

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022218240.png)

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:

        equationSize = len(equations)

        unionfind = UnionFind(2 * equationSize)

        # 第一步 预处理 将变量的值与id进行映射 方便编码

        hash_dict = dict()
        id = 0
        for i in range(equationSize):
            var1 = equations[i][0]
            var2 = equations[i][1]

            if var1 not in hash_dict and len(hash_dict) <= 2 * equationSize:
                hash_dict[var1] = len(hash_dict)
            if var2 not in hash_dict and len(hash_dict) <= 2 * equationSize:
                hash_dict[var2] = len(hash_dict)

            # 合并
            # print(hash_dict[var1], hash_dict[var2], values[i])
            unionfind.union(hash_dict[var1], hash_dict[var2], values[i])

        # 做查询
        queriesSize = len(queries) # 
        res = [0.0] * queriesSize # 结果

        for i in range(queriesSize):
            var1 = queries[i][0]
            var2 = queries[i][1]

            id1 = hash_dict.get(var1, -1)
            id2 = hash_dict.get(var2, -1)
            #  print(var1, var2, id1, id2)
            if id1 == -1 or id2 == -1:
                res[i] = -1.0
            else:
                res[i] = unionfind.isConnected(id1, id2)
            
        return res


class UnionFind:
    def __init__(self, n):

        # eg a / b  = 2的表示方法
        # a --> 0
        # b --> 1
        # self.parent[0] = 1
        # self.weight[0] = 2
        self.parent = [i for i in range(n)]
        # 这题额外加入weight 数组
        self.weight = [1.0 for i in range(n)] # i / i = 1.0  

    # 有没有老哥解释一下 为什么隔代路径压缩不行？ 有几个示例没法通过 --> 懂了 因为调用isconnected的时候有些是直接输出结果 隔代压缩没法得到最后的除法的值 如果多次调用就可以 算是吸取一个教训。
    # def find(self, x):
    #     while x != self.parent[x]:
    #         origin = self.parent[x]
    #         self.parent[x] = self.parent[self.parent[x]]
    #         self.weight[x] = self.weight[x] * self.weight[origin]
    #         x = self.parent[x]   
    #     return x
    
    def find(self, x):
        if x != self.parent[x]:
            origin = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.weight[x] = self.weight[x] * self.weight[origin]
        return self.parent[x]

    def union(self, x, y, value):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
            self.weight[root_x] = value * self.weight[y] / self.weight[x]
    def isConnected(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return self.weight[x] / self.weight[y]
        else:
            return -1.0
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202303022219045.png)
