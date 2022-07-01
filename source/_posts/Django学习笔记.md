---
title: Django学习笔记
date: 2020-05-18 18:32:33
tags: [python, Django]
categories:
- 技术
- Django
---

记录Django学习笔记

https://www.bilibili.com/video/BV1rx411X717

<!--more-->

### 创建工程

```python
django-admin startproject HelloDjango
```

![image-20200513182805245](http://blog.zchsakura.top/20200513182816.png)

### 创建应用

```python
python manage.py startapp App
```

![image-20200513182930988](http://blog.zchsakura.top/20200513182932.png)

在setting.py中注册应用：

```diff setting.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
+   'learn1',
]
```

### 开发者服务器

```python
python manage.py runserver 0.0.0.0:8000
```

### 允许访问范围

```python settings.py
ALLOWED_HOSTS = ["*"]
```

### 语言和时区设置

```python settings.py
LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'
```

### 迁移

```shell
python manage.py migrate
```

### 注册路由

```python urls.py
from learn1 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', views.hello),
]
```

```python views.py
def hello(request):
    return HttpResponse("hello Django")
```

### html模板

两种

- 在App中进行模板配置
    - 只需在App的根目录创建templates文件夹即可
    - 如果想让代码自动提示，我们应该标记文件夹为模板文件夹
- 在项目目录中进行模板配置
    - 需要在项目目录中创建templates文件夹并标记
    - 需要在settings中进行注册
- 在开发中使用第二种
    - 模板可以继承，复用

--------

在HelloDjango中新建templates文件夹并设置为模板文件夹（方便代码提示），在其中新建html模板

![image-20200518175638301](http://blog.zchsakura.top/20200518175701.png)

然后再settings.py中加入templates路径

```diff settings.py
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
+           os.path.join(BASE_DIR, 'templates'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```



**HTML小技巧：**

```html
ul>li*4(tab键)
```

```python
<ul>
    <li>今天</li>
    <li>天气</li>
    <li>真好</li>
    <li>嘿嘿嘿</li>
</ul>
```

### 分发路由

这样就可以访问ip/learn2/hello了

- 项目如果逻辑过于复杂，可以进行拆分
- 拆分为多个App
- 继续拆分路由器 urls
    - 在App中创建自己的urls
        - urlpatterns 路由规则列表
        - 在根urls中进行子路由的包含 
    - 子路由使用
        - 根路由规则 + 子路由的规则

---------

在新建的应用learn2里面新建urls.py

```python learn2/urls.py
from django.urls import path
from learn2 import views

urlpatterns = [
    path('hello/', views.hello),
]
```

在HelloDjango的urls.py中导入learn2中的路由

```diff HelloDjango/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', views.hello),
    path('index/', views.index),
+   path('learn2/', include('learn2.urls')),
]
```

### models 使用了ORM技术

- Object Relational Mapping 对象关系映射
- 将业务逻辑进行了一个解耦合
    - object.save()
    - object.delete()
- 关系型数据库
    - DDL
    - 通过models定义实现  数据库表的定义
- 数据操作
    - 增删改查
    - 存储
        - save()
    - 查询 
        - 查所有  objects.all()
        - 查单个 objects.get(pk=xx)
    - 更新
        - 基于查询的
        - 查好的对象，修改属性，然后save()
    - 删除
        - 基于查询的
        - 调用 delete()

### 快捷键

- control + p 
    - 参数提示

![image-20200608171048103](http://blog.zchsakura.top/20200608171058.png)


- shift + f6 重命名，重构
- .re  快捷生成return
- .if   多用点看看世界的美好

### 连接mysql驱动

- mysqlclient
    - python2,3都能直接使用
    - 致命缺点
        - 对mysql安装有要求，必须指定位置存在配置文件
- python-mysql
    - python2 支持很好
    - python3 不支持
- pymysql
    - python2，python3都支持
    - 它还可以伪装成前面的库

```python __init__.py
import pymysql
pymysql.version_info = (1, 4, 6, 'final', 0)  # change mysqlclient version
pymysql.install_as_MySQLdb()
```

### 重新建立 migration 文件

1.首先要保证,目前的migration文件和数据库是同步的，通过执行

```shell
python manage.py makemigrations
```

如果看到 这样的提示: No changes detected，则可以继续接下来的步骤

2.通过执行

```shell
python manage.py showmigrations
```


结果，可以看到当前项目，所有的app及对应的已经生效的migration文件如

```shell
git_hook
 [X] 0001_initial
guardian
 [X] 0001_initial
kombu_transport_django
 [X] 0001_initial
message
 (no migrations)
order
 [X] 0001_initial
pay
 [X] 0001_initial
 [x] 0002_add_model
sessions
 [X] 0001_initial
```

3.通过执行

```shell
python manage.py migrate –fake pay zero
```

这里的 pay就是你要重置的app
4.之后再执行 `python manage.pu showmigrations`，你会发现 文件前的 [x] 变成了[ ]

现在，你可以删除pay 这个 app下的migrations模块中 除 init.py 之外的所有文件。

5.之后，执行

```shell
python manage.py makemigrations
```

程序会再次为这个app 生成 0001_initial.py 之类的文件

6.最重要的一步来了, 执行

```shell
python manage.py migrate –fake-inital
```

–fake-inital 会在数据库中的 migrations表中记录当前这个app 执行到 0001_initial.py ，但是它不会真的执行该文件中的 代码。
这样就做到了，既不对现有的数据库改动，而又可以重置 migraion 文件，妈妈再也不用在 migration模块中看到一推文件了。

### 重新生成数据库中的某张表

1.删除数据库中的django_migration 表中对应的记录以及你要重新导的表

2.将你要导的那个app中的migrate 文件删除掉

3.重新导入你需要的表

```shell
python manage.py makemigration shop(你要导的app)
python manage.py migrate shop
```

这样就完成了。



### 表关系

- 分类

    - ForeignKey：一对多，将字段定义在多的端中

    - ManyToManyField：多对多，将字段定义在两端

    - OneToOneField：一对一，将字段定义在任意一端中

- 用一访问多
    - 格式
        - 对象.模型类小写_set
    - 示例
        - grade.students_set

- 用一访问一
	- 格式
		- 对象.模型类小写
	- 示例
		- grade.students

- 访问id
	- 格式
		- 对象.属性_id
	- 示例
		- student.sgrade_id

### 模型过滤

- filter()：返回符合筛选条件的结果
- exclude()：返回不符合筛选条件的结果
- all()：返回所有数据
- order_by('id')：排序，如果要逆序就加个`-`
- values()：一条数据就是一个字典，返回一个列表
- 连续使用
    - 链式调用
    - Person.objects.filter().filter().xxxx.eclude().exclude().yyyy

### 方法

- 对象方法

    - 可以调用对象的属性，也可以调用类的属性

- 类方法

    - 不能调用对象属性，只能调用类属性

    ```pyhton
    @classmethod
    def create(cls, p_name, p_age=100):
    	return cls(p_name=p_name, p_age=p_age)
    ```

    

- 静态方法
    - 啥都不能调用，不能获取对象属性，也不能获取类属性
    - 只是寄生在我们这个类上而已

### 获取单个对象

- get
    - 查询条件没有匹配的对象，会抛异常，DoesNotExist
    - 如果查询条件对应多个对象，会抛异常，MultipleObjectsReturned
- first
- last
- count
- exist



### first和last

- 默认情况下可以正常从QuerySet中获取
- 隐藏bug
    - 可能会出现 first和last获取到的是相同的对象
        - 显式，手动写排序规则（先order_by）



### 切片

- 和python中的切片不太一样
- QuerySet[5:15]  获取第五条到第十五条数据
    - 相当于SQL中limit和offset

### 缓存集

- filter
- exclude
- all
- 都不会真正的去查询数据库
- 只有我们在迭代结果集，或者获取单个对象属性的时候，它才会去查询数据库
- 懒查询
    - 为了优化我们结构和查询



### 查询条件

- 属性__运算符=值
- gt 大于
- lt 小于
- gte 大于等于
- lte 小于等于
- in 在某一集合中
- contains  类似于 模糊查询 like
- startswith  以xx开始  本质也是like
- endswith 以 xx 结束  也是like
- exact 
- 前面同时添加i , ignore 忽略
    - iexact
    - icontains
    - istartswith
    - iendswith
- django中查询条件有时区问题
    - 关闭django中自定义的时区
    - 在数据库中创建对应的时区表

### 模型成员

- 显性属性
    - 开发者手动书写的属性
- 隐性属性
    - 开发者没有书写，ORM自动生成的
    - 如果你把隐性属性手动声明了，系统就不会为你产生隐性属性了

### 从MySql到model

```shell
python manage.py inspectdb > backstage/models.py
```

### sort排序

2）key参数/函数
从python2.4开始，list.sort()和sorted()函数增加了key参数来指定一个函数，此函数将在每个元素比较前被调用。 例如通过key指定的函数来忽略字符串的大小写：
代码如下:

```python
sorted("This is a test string from Andrew".split(), key=str.lower)
['a', 'Andrew', 'from', 'is', 'string', 'test', 'This']
```

key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较。这个技术是快速的因为key指定的函数将准确地对每个元素调用。

更广泛的使用情况是用复杂对象的某些值来对复杂对象的序列排序，例如：

代码如下:

```python
student_tuples = [
('john', 'A', 15),
('jane', 'B', 12),
('dave', 'B', 10),
]
sorted(student_tuples, key=lambda student: student[2])   # sort by age
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
```

### Paginator分页

```python
from django.core.paginator import Paginator

# 进行分页
pagenator = Paginator(result_list, limit)
result = pagenator.page(page).object_list
```

### pop()间接修改键的key值

pop()一般用作删除列表中元素，但是它的返回值很有趣

```python
pop(key[,default])
```

- key: 要删除的键/值对所对应的键
- default: 可选参数，给定键不在字典中时必须设置，否者会报错(没有默认值)，此时返回default值，

Python 字典 pop() 方法删除给定键所对应的键/值对，并返回被删除的值。给定键如果不在字典中，则必须设置一个default值，否则会报错，此时返回的就是default值。

### model_to_dict()单个对象转字典

```
from django.forms import model_to_dict

result = UserUserinfo.objects.filter(nickname__contains=key).first()
result = model_to_dict(result)
```

### 取日期区间

```python
import datetime
from dateutil.relativedelta import relativedelta

now_date = datetime.datetime.now().date()
dates = {
    '1': now_date - datetime.timedelta(days=1),  # 近一天
    '2': now_date - datetime.timedelta(weeks=1),  # 近一周
    '3': now_date - relativedelta(months=1)  # 近一月
}

parameter_time = dates.get('%s' % time)
obj_list = Customer.objects.filter(deal_date__gte=parameter_time, deal_date__lte=cur_date)
```

### django中通过model名字获取model

```shell

>>> from django.apps import apps
>>> apps.get_app_config('auth') 
<AuthConfig: auth>
# 注意得到的结果是迭代器(iterator)
>>> auth = apps.get_app_config('auth')
>>> auth.get_models()
<generator object get_models at 0x31422d0>
>>> for i in auth.get_models():
...  print i
... 
<class 'django.contrib.auth.models.Permission'>
<class 'django.contrib.auth.models.Group'>
<class 'django.contrib.auth.models.User'>
>>> auth.get_model('User')         
<class 'django.contrib.auth.models.User'>
>>> User = auth.get_model('User')
>>> User.objects.all()[0]
<User: root>
```

### 时间类型

1. default=datetime.now()

    model每次初始化，都会自动设置该字段的默认值为初始化时间。

2. default=datetime.now

    model每次进行新增或修改操作，都会自动设置该字段的值为操作时间。设置后仍可以使用ORM手动修改该字段。

3. auto_now_add=True

    默认值为False，若设置为True，model每次进行新增操作，都会自动设置该字段的值为操作时间。设置为True后无法使用ORM手动修改该字段，哪怕填充了字段的值也会被覆盖。

4. auto_now=True

    默认值为False，若设置为True，model每次进行新增或修改操作，都会自动设置该字段的值为操作时间。设置为True后无法使用ORM手动修改该字段，哪怕填充了字段的值也会被覆盖。

5. 要注意的点

    除非想设置动态默认时间为项目的启动时间，否则default=datetime.now()这种用法是错误的，会得到期望之外的结果。

    使用User.objects.update方法时，设置的default=datetime.now和auto_now=True都不会生效，由于设置了auto_now=True的字段不能手动修改，此时只能使用save方法修改数据，这对于多个数据的更新是不友好的。

    因此如果设置动态默认时间的字段，应该使用default=datetime.now和auto_now_add=True来实现。

### djang获取models字段方法

通过._meta.fields获取
以Student这个model为例

```shell
In [59]: Student._meta.fields
Out[59]: 
(<django.db.models.fields.AutoField: id>,
 <django.db.models.fields.CharField: stu_name>,
 <django.db.models.fields.CharField: stu_no>,
 <django.db.models.fields.CharField: stu_sex>,
 <django.db.models.fields.IntegerField: stu_age>,
 <django.db.models.fields.DateTimeField: stu_birth>)
```

获取字段名：

```shell
In [62]: stu = Student._meta.fields 
In [62]: [stu[i].name for i in range(len(stu))]
Out[62]: [u'id', 'stu_name', 'stu_no', 'stu_sex', 'stu_age', 'stu_birth']
```

### 修改数据库中部分信息

```python
def add_work(request):
    try:
        data = simplejson.loads(request.body)
    except Exception:
        return JsonResponse({"code": -1, "msg": '参数接受失败'})
    print(data)
    DataModel = firsttypeid_to_model(data['firsttypeid']).objects.create()
    # 获取该模型内所有字段名数据
    fields_data = DataModel._meta.fields
    # 这里是将当前的model转换成数据字典，方便后面修改后提交
    data_dict = DataModel.__dict__
    for key in data:
        for field in fields_data:
            # 这样或输出这条记录的所有字段名，需要的话还可以输出verbose_name
            if field.name == key:
                # 进行匹配，将前端传来的字段匹配到，然后修改数据库里面的数据
                if data[key] == '':
                    data_dict[key] = None
                else:
                    data_dict[key] = data[key]
    data_dict['secondtypeid_id'] = data['secondtypeid']
    # 保存数据到数据库，这样的好处就是提高效率，避免过多重复操作
    DataModel.save()

    return JsonResponse({"code": 1, "msg": '成功'})
```

### 外键不能直接插入

```python
publisher = UserAdmin.objects.get(id=publisherid)
    obj = UserNotice.objects.create(noticetitle=title, noticecontent=content, publisherid=publisher)
```

### 去重查询

```python
works = MainDataTougaolanmu.objects.all().values('userid_id').distinct()
```

### 解决跨域

在整个项目的setting.py中进行修改

```diff
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
+   'corsheaders',  # 跨域
    'learn1',
    'backstage',
]
```

```diff
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
+   'corsheaders.middleware.CorsMiddleware',  # 解决跨域
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

```python
# 跨域增加忽略
CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_ALLOW_ALL = True

CORS_ALLOW_METHODS = (
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
    'VIEW',
)
CORS_ALLOW_HEADERS = (
    'XMLHttpRequest',
    'X_FILENAME',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'token',    # token
    'authentication',   # token
)
```