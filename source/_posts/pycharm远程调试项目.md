---
title: pycharm远程调试项目
tags:
  - python
  - pycharm
categories:
  - 技术
  - Python
date: 2020-05-20 22:27:45
---

服务器上没有IDE，编写代码很不方便，所以我选择在本地用pycharm远程调试程序，但是我没有选择在本地pycharm使用ssh连接Terminal，因为真的太卡了，我还是用xshell作为命令行使用。

参考：https://www.cnblogs.com/weihengblog/p/9656257.html

<!--more-->

pycharm是一个非常强大的python开发工具，现在很多代码最终在线上跑的环境都是linux，而开发环境可能还是windows下开发，这就需要经常在linux上进行调试，或者在linux对代码进行编写，而pycharm提供了非常便捷的方式。具体实现在windows上远程linux开发和调试的代码步骤如下：

1. 本地和远程同步

2. 配置Project Interpreter（使用远程的Python解释器）

3. 设置Terminal运行的Python版本

4. 使用Terminal登陆到Linux服务器

### 一、本地和远程代码同步

首先，在本地和远程拥有相同的项目代码：

![1321568-20180916144244833-1490970117](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105507.png)

**在windows平台使用Pycharm打开项目,然后：Tools -> Deployment -> configuration，然后新建远程服务器**

![1321568-20180916144731393-549531962](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105650.png)

**然后进行服务器配置**

![1321568-20180916145358775-782073356](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105658.png)

![1321568-20180916145549234-784619187](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105705.png)

**点击Ok**，**经过上面步骤的配置后，我们可以在PyCharm 界面的右边查看远端代码，如下图：**

![1321568-20180916145725516-1886186686](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105836.png)

**切记！勾选 Automatic Upload  实现本地自动同步到远端**

### 二、配置Project Interpreter（使用远程的Python解释器）

![1321568-20180916145905889-619698935](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105842.png)

![1321568-20180916150051936-1448125681](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105846.png)

 ![1321568-20180916150149777-1149968755](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105852.png)

 ![1321568-20180916150602788-417269814](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012105900.png)

**点击 OK 保存，点击Finish完成。然后在编辑新添加的Python Interpreter，如下图所示：**

![1321568-20180916151536382-1761728586](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012110011.png)

 ![1321568-20180916151652947-1922094213](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012110017.png)

 **这样，本地和远程的项目 以来的pip都是相同的，解释器也是相同的。**

### 三、设置Terminal运行的Python版本

**File -> Settings -> Tools -> SSH Terminal，在 Deployment server 选择Linux服务器的Python版本路径。（配置已经存在，只要选择即可）**

![1321568-20180916151811986-1349908586](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012110030.png)

### 四、使用Terminal登陆到Linux服务器

 **选择 Tools -> Start SSH session，默认会开启Linux ssh会话窗口，如下图：**

![1321568-20180916151852453-923493946](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012110034.png)

**就可以执行Linux命令了，在远程Linux主机上，如下图：**

 ![1321568-20180916152101674-143914985](http://zchsakura-blog.oss-cn-beijing.aliyuncs.com/20201012110042.png)