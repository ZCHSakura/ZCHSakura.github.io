---
title: 将flask部署到Ubuntu服务器上
date: 2020-04-26 20:59:00
tags: [python, Flask, 部署至服务器]
categories:
- 技术
- Flask
---

我们做的项目最终都是要跑在服务器上的，我们不可能在服务器上一直跑着我们的Flask程序，而且程序也可能因为某些错误而挂掉，我们需要保证它的可用性，这篇文章就是我根据别人的[Flask部署教程](https://www.cnblogs.com/Ray-liang/p/4173923.html)总结出来的东西还有几个特别需要注意的小坑。

<!--more-->

总体部署方案是：

- Web 服务器采用 uwsgi host Flask

- 用 Supervisor 引用 uwsgi 作常规启动服务

- 基于 Nginx 作反向代理

# 1.安装python并创建虚拟环境

安装python的教程有很多也很简单这里就不再赘述了。

要说一下的是python的虚拟环境创建，大家可能都知道virtualenv，但是其实原生的virtualenv使用起来不是那么方便，所以我推荐使用virtualenvwrapper来管理我们的虚拟环境。具体操作可以参考我的另一篇文章{% post_link 创建python虚拟环境 [创建python虚拟环境] %}。

# 2.安装 uWSGI

安装uwsgi还是比较简单的，记得进入相应的虚拟环境

```shell
(venv)my_flask root$ pip install uwsgi
```

应该是秒装，安装完我们先不管他，后面再来配置他。

# 3.安装Flask

我是用清单文件一次性安装Flask和他的相关依赖的，这样会更快。我的引用清单是这样的：

```[] requirements.txt
Flask==0.10.1
Flask-Login==0.2.11
Flask-Mail==0.9.1
Flask-Moment==0.4.0
Flask-PageDown==0.1.5
Flask-SQLAlchemy==2.0
Flask-Script==2.0.5
Flask-WTF==0.10.2
Flask-Cache==0.13.1
Flask-Restless==0.15.0
Flask-Uploads==0.1.3
Jinja2==2.7.3
Mako==1.0.0
Markdown==2.5.1
MarkupSafe==0.23
SQLAlchemy==0.9.8
WTForms==2.0.1
Werkzeug==0.9.6
html5lib==1.0b3
itsdangerous==0.24
six==1.8.0
awesome-slugify==1.6
```

安装清单文件（记得在虚拟环境中操作）

```shell
(venv)my_flask root$ pip install -r requirements.txt
```

# 4.项目文件

接下来就是上传 Flask的项目文件，之前我在各大的“转载专业户”里找了好多的资料，在这一步中大多只是在上面加个标准的Flask运行文件，虽说做个范例可以但说实在的这很让人迷惑，为什么？先看看代码吧：

```python
from flask import Flask
 
app = Flask(__name__)
 
@app.route("/")
def hello():
    return "Hello World!"
```

生产环境内，谁会用这样的代码呢，这只是Flask 的最简入门范，我的Flask项目中 app 是被做在包内的，相信很多人都是这样做的，在包外我们采用 Flask Script 写一个 `manage.py` 文件 作为启动文件，这更方便于支持各种的项目

```python manage.py
from flask_script import Manager,Server
from app import app

manager = Manager(app)
manager.add_command('runserver',Server(host='0.0.0.0',port=5000,use_debugger=True))

if __name__ == '__main__':
    manager.run()
```

此时如果我们使用命令：

```shell
python manage.py runserver
```

已经可以运行在本地 `http://127.0.0.1:5000`（记得开启5000端口）

# 5.配置uWSGI

uWSGI有两种启动方式，在这里我们选了通过配置文件启动的方法

在项目目录中新建`config.ini`，写入如下内容：

```python config.ini
[uwsgi]

# uwsgi 启动时所使用的地址与端口
#http=0.0.0.0:9102
socket=0.0.0.0:9102

# 指向网站目录
chdir = /var/www/flask

home = /root/.virtualenvs/py3_flask

# python 启动程序文件
wsgi-file = manage.py

# python 程序内用以启动的 application 变量名
callable = app

#daemonize=/www/wwwroot/www.chineseculture.xyz/flask_test/my_flask.log

# 处理器数
processes = 4

# 线程数
threads = 2
```

这里的9102就是我们外网访问时服务器监听的地址，记得在控制台开启9102端口。

配置好端口之后，输入命令直接运行uWSGI：

```shell
uwsgi config.ini
```

到此为止，我们已经可以通过`服务器公网ip:9102`访问你的Flask应用

**注意！！这里出现了第一个坑**

uwsgi 启动时所使用的地址与端口前面可以写http也可以写socket，当我们这里写http的时候我们是可以通过`服务器公网ip:9102`访问Flask的，但如果是socket则不可以。相应的我们后期使用nginx进行端口转发到uwsgi的时候我们这里必须是socket，如果是http将不成功。

OK， 此时已经正常启动 uwsgi 并将 Flask 项目载入其中了，ctrl+c 关闭程序。但这只是命令启动形式，要使其随同服务器启动并作为后台服务运行才是运营环境的实际所需要。因此接下来我们需要安装另一个工具来引导 uwsgi 。

# 6.安装 Supervisor

[Supervisor](http://supervisord.org/configuration.html)可以同时启动多个应用，最重要的是，当某个应用Crash的时候，他可以自动重启该应用，保证可用性。

```shell
sudo apt-get install supervisor
```

Supervisor 的全局的配置文件位置在：

```
/etc/supervisor/supervisor.conf
```

正常情况下我们并不需要去对其作出任何的改动，只需要添加一个新的 *.conf 文件放在

```
/etc/supervisor/conf.d/
```

下就可以，那么我们就新建立一个用于启动 my_flask 项目的 uwsgi 的 supervisor 配置 (命名为：flask_supervisor.conf)：

```python flask_supervisor.conf
[program:my_flask]
# 启动命令入口
command=/root/.virtualenvs/py3_flask/bin/uwsgi /var/www/flask/config.ini

# 命令程序所在目录
directory=/var/www/flask
#运行命令的用户名
user=root

stopasgroup = true
killasgroup = true

autostart=true
autorestart=true
#日志地址
stdout_logfile=/var/www/flask/uwsgi_supervisor.log      
```

**注意！！这里是第二个坑**

```python
stopasgroup = true  # 用于停止进程组，即停止所有通过“uwsgi.ini”配置启动的进程。
killasgroup = true  # 用于关闭进程组，即关闭所有通过“uwsgi.ini”配置启动的进程。
```

这两句话在很多教程里是没有的，如果没有这两局代码也就意味着我们使用`supervisorctl`命令stop了`my_flask`之后这个uwsgi进程占用的`9102端口`不会被释放也就意味着我们无法再start `my_flask`。这个问题一开始困扰了我好久，我一直不明白哪里出了问题，每次restart都不成功，希望看到的有缘人能注意一下。

**启动服务**

```shell
sudo service supervisor start
```

**终止服务**

```shell
sudo service supervisor stop
```

## Supervisor补充

supervisor：要安装的软件的名称。
supervisord：装好supervisor软件后，supervisord用于启动supervisor服务。
supervisorctl：用于管理supervisor配置文件中program。

**启动supervisor服务：**

```shell
supervisord -c /etc/supervisor/supervisord.conf
```

**supervisorctl 操作**

```shell
supervisorctl status

supervisorctl reload

supervisorctl stop program_name

supervisorctl start program_name

supervisorctl restart program_name
```

 # 7.安装并配置nginx

[Nginx](http://nginx.com/)是轻量级、性能强、占用资源少，能很好的处理高并发的反向代理软件。

```shell
sudo apt-get install nginx
```

Ubuntu 上配置 Nginx 也是很简单，不要去改动默认的 nginx.conf 只需要将

```
/etc/nginx/sites-available/default
```

文件替换掉就可以了。

新建一个 default 文件:

```python
server {
    listen  80;
    server_name XXX.XXX.XXX; #公网地址
    
    location / {
        include uwsgi_params;#转发到那个地址，转发到uwgi的地址，在通过uwsgi来启动我们的项目
        uwsgi_pass 0.0.0.0:9102;
        uwsgi_connect_timeout 60;
    }
}
```

将default配置文件替换掉就大功告成了！
还有，更改配置还需要记得重启一下nginx:

```shell
nginx -s reload
```