---
title: NexT写作常用样式
date: 2020-03-13 23:32:16
tags: [NexT, Hexo,markdown,]
categories:
- 工程实践
- 经验积累
top: 10
---
一些markdown之外的语法规则，让文章更美观

> 本文摘录自 {% label success@周宇峰的博客 %} [原文链接](https://yfzhou.coding.me/2018/08/27/Hexo-Next%E6%90%AD%E5%BB%BA%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2%EF%BC%88%E4%B8%BB%E9%A2%98%E4%BC%98%E5%8C%96%EF%BC%89/)
![8MAadJ.md](https://download.kezhi.tech/img/8MAadJ.md.jpg)
<!--more-->

### 1.文本居中

效果:
{% cq %}
人生乃是一面镜子，
从镜子里认识自己，
我要称之为头等大事，
也只是我们追求的目的！
{% endcq %}
源码:

```
{% cq %}
人生乃是一面镜子，
从镜子里认识自己，
我要称之为头等大事，
也只是我们追求的目的！
{% endcq %}
```

更多NexT主题自带的标签，点击：[http://theme-next.iissnan.com/tag-plugins.html](http://theme-next.iissnan.com/tag-plugins.html)

### 2.主题自带样式note标签

```
<div class="note default"><p>default</p></div>
```

<div class="note default"><p>default</p></div>

```
<div class="note primary"><p>primary</p></div>
```

<div class="note primary"><p>primary</p></div>

```
<div class="note success"><p>success</p></div>
```

<div class="note success"><p>success</p></div>

```
<div class="note info"><p>info</p></div>
```

<div class="note info"><p>info</p></div>

```
<div class="note warning"><p>warning</p></div>
```

<div class="note warning"><p>warning</p></div>

```
<div class="note danger"><p>danger</p></div>
```

<div class="note danger"><p>danger</p></div>

```
<div class="note danger no-icon"><p>danger no-icon</p></div>
```
<div class="note danger no-icon"><p>danger no-icon</p></div>

首先在主题配置文件中配置：
```
# Note tag (bs-callout).
note:
  # 风格
  style: flat
  # 要不要图标
  icons: true
  # 圆角矩形
  border_radius: 3
  light_bg_offset:
```

### 3.自带label

{% label default@default %}
```
{% label default@default %}
```
{% label primary@primary %}
```
{% label primary@primary %}
```
{% label success@success %}
```
{% label success@success %}
```
{% label info@info %}
```
{% label info@info %}
```
{% label warning@warning %}
```
{% label warning@warning %}
```
{% label danger@danger %}
```
{% label danger@danger %}
```

### 4.选项卡

{% tabs 选项卡, 2 %}
<!-- tab -->
**这是选项卡 1** 呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈……
<!-- endtab -->
<!-- tab -->
**这是选项卡 2**
<!-- endtab -->
<!-- tab -->
**这是选项卡 3** 哇，你找到我了！φ(≧ω≦*)♪～
<!-- endtab -->
{% endtabs %}
```
{% tabs 选项卡, 2 %}
<!-- tab -->
**这是选项卡 1** 呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈呵呵哈哈哈哈哈哈哈哈……
<!-- endtab -->
<!-- tab -->
**这是选项卡 2**
<!-- endtab -->
<!-- tab -->
**这是选项卡 3** 哇，你找到我了！φ(≧ω≦*)♪～
<!-- endtab -->
{% endtabs %}
```

### 5.主题自带tabs


源码:
```
{% btn https://www.baidu.com, 点击下载百度, download fa-lg fa-fw %}
```
效果:{% btn https://www.baidu.com, 点击下载百度, download fa-lg fa-fw %}
