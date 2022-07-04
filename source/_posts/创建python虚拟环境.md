---
title: 创建python虚拟环境
date: 2020-04-26 15:04:38
tags: [python,虚拟环境]
categories:
 - 技术
 - Python
---

如果在一台电脑上，想开发多个不同的项目， 需要用到同一个包的不同版本， 如果不使用虚拟环境， 在同一个目录下安装或者更新， 新版本会覆盖以前的版本， 其它的项目就无法运行了。

<!--more-->

## 1.安装虚拟环境

```
sudo pip install virtualenv
sudo pip install virtualenvwrapper
```

virtualenvwrapper类似于一个虚拟环境管理工具会比原生的virtualenv方便一些

安装完虚拟环境如果找不到mkvirtualenv命令，须配置环境变量：

```
# 1、在~（家目录）下创建目录用来存放虚拟环境
mkdir .virtualenvs

# 2、打开~/.bashrc文件，并添加如下：
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

# 3、运行
source ~/.bashrc
```

## 2.创建虚拟环境

如果不指定Python版本，默认安装的是Python2的虚拟环境

```
# 在python2中，创建虚拟环境
mkvirtualenv 虚拟环境名称
例 ：
mkvirtualenv py_flask
```

如果想要指定python版本

```
mkvirtualenv -p python3 虚拟环境名称
例 ：
mkvirtualenv -p python3 py3_flask
```

提示：

所有的虚拟环境都位于/~/下的隐藏目录.virtualenvs下

## 3.使用虚拟环境

### 3.1查看已有虚拟环境

```
workon
```

### 3.2进入虚拟环境

```
workon 虚拟环境名称
```

### 3.3退出虚拟环境

```
deativate
```

### 3.4删除虚拟环境

```
先退出：deactivate
再删除：rmvirtualenv py3_flask
```

## 4.虚拟环境工具包位置

~/.virtualenvs/虚拟环境名称/lib/python3.5/site-packages

