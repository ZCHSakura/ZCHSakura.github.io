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
