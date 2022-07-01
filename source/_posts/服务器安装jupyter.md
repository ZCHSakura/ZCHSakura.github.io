---
title: 服务器安装jupyter
date: 2020-10-15 10:07:28
tags: [服务器,jupyter]
categories:
- 技术
- 服务器配置
---

属实不想用xshell上的ipython，命令行也确实没有jupyter方便，毕竟jupyter还可以保存运行过程，还可以进行拓展。所以我在服务器上搭建了一个jupyter

<!--more-->

## 参考

https://blog.csdn.net/qq_42137895/article/details/104283459?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param

## 1. 安装jupyter服务

```shell
pip3.7 install jupyter
```

## 2. 配置远程访问jupyter

先输入ipython进入交互模式 ，见图输入以下内容，设置好密码过后

会自动生成一个**Verify password**复制保存下来

![aHR0cHM6Ly9waWMuZG93bmsuY2MvaXRlbS81ZTQzZDQwZTJmYjM4YjhjM2NkNDMwOTcucG5n](http://blog.zchsakura.top/20201015101310.png)

### 2.1生成配置文件

在输入以下内容

```shell
jupyter notebook --generate-config
```

就会自动在根目录下生成文件（~/.jupyter/jupyter_notebook_config.py），如果看不到，选项中选择显示隐藏文件

### 2.2修改配置文件

建议先把jupyter_notebook_config.py文件传输到本地修改完成再上传

找到以下字符串进行修改

```python
c.NotebookApp.ip = '*' # 如果这里修过过后启动服务报错 则修改为c.NotebookApp.ip='0.0.0.0'
c.NotebookApp.password = u'sha1****' #就之前保存的验证密码
c.NotebookApp.open_browser = False # 设置是否自动打开浏览器
c.NotebookApp.port = 9999  # 设置端口
c.NotebookApp.allow_remote_access = True
c.ContentsManager.root_dir = '/var/www/jupyter_python'	# 设置jupyter显示目录
```

### 3. 启动服务

这里推荐两种

1. 入门：`jupyter notebook --allow-root`

    > 但这种会一直占着窗口，无法执行其他命令
    >
    > Ctrl + C 即可结束

2. 进阶: `nohup jupyter notebook --allow-root &`

    > nohup表示no hang up, 就是不挂起, 于是这个命令执行后即使终端退出, 也不会停止运行.
    >
    > 但要手动结束
    >
    > lsof -i : {端口号}
    >
    > 然后 kill -9 {对应pid} # 9的意思是发送KILL信号，立刻结束，可能会有数据丢失

启动以后在本机 输入`http://{服务器ip}:9999`进行访问

## 4. 安装增强功能（自动补全之类）

在终端中依次执行以下4行代码

```python
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user
```

**执行完成以后，重启jupyter，即可看到附加项**

![aHR0cHM6Ly9waWMuZG93bmsuY2MvaXRlbS81ZTQzZDhiYTJmYjM4YjhjM2NkNGVlZjUucG5n](http://blog.zchsakura.top/20201015101630.png)