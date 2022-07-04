---
title: lxml相关操作
tags:
  - python
  - 爬虫
  - 信息提取
categories:
  - 技术
  - 爬虫
date: 2020-04-30 17:12:39
---

scrapy作为爬虫框架还是比较全能的，但是我之前遇到过一个问题就是要将一个节点的某些子节点删除，这个操作好像scrapy的selector好像不能实现，也有可能是我自己没找到方法。我的解决方法是直接将response的内容构建成lxml然后再进行节点删除和信息提取。

<!--more-->

## 1.如何构建对象

首先我们要引入lxml（html也是lxml的一种）包，然后调用`etree.HTML()`函数解析html来构建Element

```python
from lxml import etree
element = etree.HTML(response.text).xpath('//div[@class="con_main"]')[0]
```

## 2.去除特定节点

构建完对象后，我们选择`find`,`findall`,`getchildren`等方法选定想要剔除的节点，然后再调用`remove`函数将其去掉，Element还有很多的方法我也没有全部用过，如果有需要可以参见[lxml官方文档](https://lxml.de/api/lxml.etree._Element-class.html)。

```python
for pp in element.findall('p[@style]'):
    element.remove(pp)
```

## 3.完成信息提取

之后就是正常的完成对所需信息的处理，去毛刺，格式化，列表化等操作：

```python
content = element.xpath('string(.)').replace('\xa0', '').replace('a("conten");', '').split('\n')
# 这一句是去除列表中的空元素
item['content'] = [i for i in content if i != '']
```

**注意：**

- 虽然`Element`和scrapy的`selector`都可以是调用`xpath`方法形式上也很类似但是`Element`对象`xpath('string(.)')`之后是不用`extract_first()`的

## 4.extract（）和extract_first（）

如果您是Scrapy的长期用户，则可能熟悉`.extract()`和`.extract_first()`选择器方法。许多博客文章和教程也正在使用它们。Scrapy仍支持这些方法，**没有计划**弃用它们。

但是，Scrapy用法文档现在使用`.get()`和 `.getall()`方法编写。我们认为这些新方法可以使代码更简洁易读。

以下示例显示了这些方法如何相互映射：

1. `SelectorList.get()`与`SelectorList.extract_first()`：
```shell
>>> response.css('a::attr(href)').get()
'image1.html'
>>> response.css('a::attr(href)').extract_first()
'image1.html'
```

2. `SelectorList.getall()`与`SelectorList.extract()`：

```shell
>>> response.css('a::attr(href)').getall()
['image1.html', 'image2.html', 'image3.html', 'image4.html', 'image5.html']
>>> response.css('a::attr(href)').extract()
['image1.html', 'image2.html', 'image3.html', 'image4.html', 'image5.html']
```

3. `Selector.get()`与`Selector.extract()`：

```shell
>>> response.css('a::attr(href)')[0].get()
'image1.html'
>>> response.css('a::attr(href)')[0].extract()
'image1.html'
```

4. 为了保持一致性，还有`Selector.getall()`，它返回一个列表：

```shell
>>> response.css('a::attr(href)')[0].getall()
['image1.html']
```

因此，主要区别在于`.get()`和`.getall()`方法的输出更可预测：`.get()`始终返回单个结果，`.getall()` 始终返回所有提取结果的列表。使用`.extract()`method时，结果是否为列表并不总是很明显；得到一个结果`.extract()`或者`.extract_first()`应该被调用。