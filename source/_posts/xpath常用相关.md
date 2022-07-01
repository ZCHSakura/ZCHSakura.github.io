---
title: xpath常用相关
tags:
  - scrapy
  - python
categories:
  - 技术
  - 爬虫
date: 2020-05-09 17:34:34
---


xpath常用语法

<!--more-->

### 1.匹配文本

```python
a[contains(text(),"百度搜索")]
```

### 2.匹配属性

```python
input[contains(@name,'na')]
```

### 3.节点关系

| 轴名称             | 结果                                                     |
| :----------------- | :------------------------------------------------------- |
| ancestor           | 选取当前节点的所有先辈（父、祖父等）。                   |
| ancestor-or-self   | 选取当前节点的所有先辈（父、祖父等）以及当前节点本身。   |
| attribute          | 选取当前节点的所有属性。                                 |
| child              | 选取当前节点的所有子元素。                               |
| descendant         | 选取当前节点的所有后代元素（子、孙等）。                 |
| descendant-or-self | 选取当前节点的所有后代元素（子、孙等）以及当前节点本身。 |
| following          | 选取文档中当前节点的结束标签之后的所有节点。             |
| following-sibling  | 选取当前节点之后的所有兄弟节点                           |
| namespace          | 选取当前节点的所有命名空间节点。                         |
| parent             | 选取当前节点的父节点。                                   |
| preceding          | 选取文档中当前节点的开始标签之前的所有节点。             |
| preceding-sibling  | 选取当前节点之前的所有同级节点。                         |
| self               | 选取当前节点。                                           |

```html
<div>
    <a id="1" href="www.baidu.com">我是第1个a标签</a>
    <p>我是p标签</p>
    <a id="2" href="www.baidu.com">我是第2个a标签</a>
    <a id="3" href="www.baidu.com">我是第3个a标签</a>
    <a id="4" href="www.baidu.com">我是第4个a标签</a>
    <p>我是p标签</p>
    <a id="5" href="www.baidu.com">我是第5个a标签</a>
</div>
```

获取第三个a标签的下一个a标签："//a[@id='3']/following-sibling::a[1]"

获取第三个a标签后面的第N个标签："//a[@id='3']/following-sibling::*[N]"

获取第三个a标签的上一个a标签："//a[@id='3']/preceding-sibling::a[1]"

获取第三个a标签的前面的第N个标签："//a[@id='3']/preceding-sibling::*[N]"

获取第三个a标签的父标签："//a[@id=='3']/.."