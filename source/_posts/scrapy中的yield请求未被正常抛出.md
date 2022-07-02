---
title: scrapy中的yield请求未被正常抛出
date: 2020-05-08 21:19:36
tags: [scrapy]
categories:
- 技术
- 爬虫
---

有一次爬医书的时候有一个函数不会被执行，找了半天没有找到问题所在，只知道是上一个`scrapy.Request`执行完并没有正确的触发接下来的函数，根据查找的信息应该是由于scrapy自带的去重机制将我的请求给抛弃了。

<!--more-->

参考：https://blog.csdn.net/qq_32670879/article/details/85042464

问题来源：我在爬取医书的时候因为是书籍所以存在章节信息，在书籍的介绍页既有书籍简介，又有目录，我想要写两个函数来分别处理，当书籍简介获取完之后要进入获取目录信息函数的时候不能正确进入。

![image-20200509152349586](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200509152400.png)

经过在网上查找信息发现应该是我调用下面函数的时候，因为又请求了之前请求过的`response.url`导致scrapy自带的去重机制将本次请求全部拦截了。

```python
yield scrapy.Request(response.url, callback=self.chapterList, meta={'bookName': item['name']})
```

![20181216232557785](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20200509153757.png)

解决方法：在函数参数中加入`dont_filter=True`，结果如下

```python
yield scrapy.Request(response.url, callback=self.chapterList, meta={'bookName': item['name']}, dont_filter=True)
```

