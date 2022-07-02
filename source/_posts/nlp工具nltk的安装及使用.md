---
title: nlp工具nltk的安装及使用
tags:
  - python
  - nlp
categories:
  - 工程实践
  - 经验积累
top: 20
date: 2021-10-21 21:24:53
---


## nltk简介

NLTK是构建Python程序以使用人类语言数据的领先平台。它为50多种语料库和词汇资源（如WordNet）提供了易于使用的界面，还提供了一套用于分类，标记化，词干化，标记，解析和语义推理的文本处理库。NLTK是Python上著名的⾃然语⾔处理库 ⾃带语料库，具有词性分类库 ⾃带分类，分词，等等功能。

<!--more-->

## package安装

首先使用pip安装nltk包

```bash
pip install nltk
```

可以使用清华源对其加速

```bash
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## nltk-data下载

安装好的nltk包是不能拿来直接使用的，还需要下载相关数据模型才可以使用。下载方法如下。

nltk包安装完成后打开python命令行运行以下命令（也可以新建python文件写入以下命令并运行）

```python
import nltk
nltk.download()
```

会出现以下界面：

![](http://download.kezhi.tech/img/202110212046568.png)

最开始这个列表是空白的，点击右下方`refresh`后出现nltk-data的列表。

点击左下角的`Download`开始下载数据，等下载完成后即可正常使用

## 国内加速下载

在国内下载可能会出现找不到DNS或者下载到一半出错的情况。遇到该情况最便捷的解决思路如下：

- 执行以下命令之一下载nltk-data到本地，大小700M左右

  ```bash
  git clone https://github.com/nltk/nltk_data.git
  # 无法链接到GitHub的也可以使用如下链接之一进行clone
  git clone http://gitclone.com/github.com/nltk/nltk_data.git
  git clone https://hub.fastgit.org/nltk/nltk_data.git
  ```

- 进入下载到本地的nltk-data目录，修改nltk_data目录下的index.xml文件，将所有的

  ```bash
  s://raw.githubusercontent.com/nltk/nltk_data/gh-pages
  ```

  替换为：

  ```bash
  ://localhost:8000
  ```

- 在该目录下运行：

  ```bash
  python -m http.server 8000
  ```

  这个时候我们会在本机提供nltk_data数据下载服务的服务器。nltk下载器通过访问本地地址既可以获取到需要的文件。

- 重新在python中执行如下语句：

  ```python
  import nltk
  nltk.download()
  ```
  
- 将server index中的地址替换为`http://localhost:8000/index.xml`如下图:

  ![image-20211021212307774](http://download.kezhi.tech/img/202110212123817.png)

  依次点击`refresh`和`Download`即可开始安装。

## Reference

[1] [国内下载GITHUB库加速方法及快速安装NLTK - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/347931749)

[2] [直接快速下载NLTK数据_今春一别难相逢-CSDN博客_nltk下载](https://blog.csdn.net/qiang12qiang12/article/details/81254595)

[3] [nltk/nltk_data: NLTK Data (github.com)](https://github.com/nltk/nltk_data)

[4] [自然语言处理| NLTK - 简书 (jianshu.com)](https://www.jianshu.com/p/a3cb9f986e69)

