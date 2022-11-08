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
