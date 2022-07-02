---
title: 基于redis使用scrapy下载中间件实现简单的增量爬虫
tags:
  - python
  - redis
  - scrapy
  - 爬虫
  - 增量爬取
categories:
  - 技术
  - 爬虫
date: 2020-04-21 12:00:00
---


在爬虫中有一种很重要的操作就是增量爬取，scrapy自带的管道中的去重只能实现该次爬取链接的去重，并不能实现对过往爬取数据的去重，所以选择使用redis来实现增量爬取，这也是大家现在使用较多的方法

<!--more-->

参考https://www.jianshu.com/p/f03479b9222d

# 1.官方去重

上面提起过，官方文档中的在管道里预设了去重模块，但只能在本次爬取内容中去重，无法对比以往爬取的数据。

```python 官方去重DuplicatesPipeline
class DuplicatesPipeline(object):
  
    def __init__(self):
        self.url_seen = set()

    def process_item(self, item, spider):
        if item['art_url'] in self.url_seen: #这里替换成你的item['#']
            raise DropItem("Duplicate item found: %s" % item)
           
        else:
            self.url_seen.add(item['art_url']) #这里替换成你的item['#']
            return item     
```

可以看出官方自带的DuplicatesPipeline管道中间件没办法完成我们所需要的增量爬取（对比曾经爬过的内容而不只是本次）

# 2.使用管道中间件

我在网上找到的大部分方法都是在管道中间件中通过使用redis实现了简单的增量爬取，原理就是在redis中存储之前爬虫爬取过的页面url，然后item进入管道的时候判断这个item对应的url在不在redis里，如果在说明爬过，把这个item丢弃；如果不在就继续下一步。

但是这里并没有解决页面url没变但是页面内容发生了变化的情况，如果要解决这个问题那么还应该将item内容编成md5编码，然后对比url的同时对比md5编码，url相同md5不同就覆盖掉原来的。但是我本次并没有实现这个功能，这个功能只适用于内容会发生变化的详情页，一般都用不上，只是提出来解决方法。

```python pipelines.py
import mysql.connector
import pandas as pd  #用来读MySQL
import redis 
redis_db = redis.Redis(host='127.0.0.1', port=6379, db=4) #连接redis，相当于MySQL的conn
redis_data_dict = "f_url"  #key的名字，写什么都可以，这里的key相当于字典名称，而不是key值。


class DuplicatesPipeline(object):
    conn = mysql.connector.connect(user = 'root', password='yourpassword', database='dbname', charset='utf8')    

    def __init__(self):
        redis_db.flushdb() #删除全部key，保证key为0，不然多次运行时候hlen不等于0，刚开始这里调试的时候经常出错。
        if redis_db.hlen(redis_data_dict) == 0: #
            sql = "SELECT url FROM your_table_name;"  #从你的MySQL里提数据，我这里取url来去重。
            df = pd.read_sql(sql, self.conn) #读MySQL数据
            for url in df['url'].get_values(): #把每一条的值写入key的字段里
                redis_db.hset(redis_data_dict, url, 0) #把key字段的值都设为0，你要设成什么都可以，因为后面对比的是字段，而不是值。


    def process_item(self, item, spider):
    	if redis_db.hexists(redis_data_dict, item['url']): #取item里的url和key里的字段对比，看是否存在，存在就丢掉这个item。不存在返回item给后面的函数处理
             raise DropItem("Duplicate item found: %s" % item)

        return item
```

注意要将新写的管道中间件在setting中注册：

```diff setting.py
ITEM_PIPELINES = {
	'shaoerkepu.pipelines.ShaoerkepuPipeline': 300,
+	'shaoerkepu.pipelines.DuplicatesPipeline': 200,
}
```

这里主要使用reids中的hash类型，原因是速度，但是为什么速度快俺也不懂，没有深入了解过。个人感觉这里的hash其实有点像是字典，redis_db.hset(redis_data_dict, url, 0)，这里面redis_data_dict类似字典名，然后url是key，0是value。一个redis_data_dict里面有很多对key，value。这里由于我们只需要使用url作为key所以所有的value都写了0。详细的python操作redis可以看我的另一篇博客。

# 3.使用下载中间件

上面介绍了大部分去重工作都在管道中间件中实现，如果涉及到内容的去重也就是md5那么就必须在管道中间件中实现，但是如果只涉及到对url的判断我觉得可以在下载中间件中实现，这样可以省去更多的时间（理论上应该是这样的吧，毕竟下载中间件在request的时候就起作用，管道在最后面才起作用）。

```python middlewares.py
from scrapy import signals
from scrapy.exceptions import IgnoreRequest
import redis

redis_db = redis.Redis(host='127.0.0.1', port=6379, db=4)

class IngoreRequestMiddleware(object):
    def process_request(self, request, spider):
        # 通过这一行代码实现下载中间件的自定义，不可能每一个请求都调用该中间件，我们是多层爬虫，会导致第一层就被抛弃，到不了详情页
        if request.meta.get('middleware') == 'IngoreRequestMiddleware':
            if redis_db.hexists('urls', request.url):
                # 调用异常抛弃request
                raise IgnoreRequest("IgnoreRequest : %s" % request.url)
            else:
                redis_db.hset('urls', request.url, 0)
                # 返回None进行接下来的操作
                return None
```

注意要将新写的下载中间件在setting中注册：

```diff settings.py
DOWNLOADER_MIDDLEWARES = {
#    'shaoerkepu.middlewares.ShaoerkepuDownloaderMiddleware': 543,
+   'shaoerkepu.middlewares.IngoreRequestMiddleware': 543,
}
```

