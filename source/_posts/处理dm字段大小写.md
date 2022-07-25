---
title: 处理dm字段大小写
tags:
  - 达梦
  - Python
categories:
  - 技术
  - 达梦
date: 2022-07-25 10:52:51
top:
---


记录下使用达梦数据库遇到的大小写问题，达梦数据库在建库的时候可以选择不区分大小写，默认是选择区分的，我看网上的博客很多都建议取消区分大小写，但是我对接的用户这边建库的时候没有取消，然后dm好像有把字段名全部大写，就算他们建表时写的小写，没有仔细研究，我这里主要只是做了有无大写判断。

<!--more-->

#### cursor.fetchall()

对从数据库中获得的列数较少的数据实际上都不用判断里面字段名是大写还是小写，可以直接用try、expect来处理

```python
try:
    folder_path = save_folder[0]['SAVE_FOLDER']
except:
    folder_path = save_folder[0]['SAVE_FOLDER'.lower()]
```

#### df = pd.DataFrame.from_records(db_data)

为了对数据进行更好的处理，在Python中经常会使用到pandas来处理数据，当我们在`dmPython.connect`中设置了`cursorclass=dmPython.DictCursor`时，我们`cursor.fetchall()`回来的数据会是一个list包裹若干dict的形式，每一个dict是数据库中的一行，dict中的key就是数据表中的列名。此时我们可以采用`pd.DataFrame.from_records()`将这种list直接变为pandas的DataFrame格式来进行处理。

这里我需要将数据库中取出来的数据先修改类型，使用`df.astype()`可以直接修改df的列数据类型，这里需要用到一个dict来完成数据类型修改，dict的key是df的列名，value是要修改成的数据类型，如下：

```
data_types_dict = {
    "time_start": float,
    "ip_client": str,
    "byte_up": float,
    "byte_dn": float
}
```

这里就涉及到判断从数据库中取出来的数据字段名是大写还是小写，可以用以下代码完成数据修改，主要思路就是当我判断数据库字段名是大写的时候把我的dict的key也改为大写：

```
if 'time_start' in df.columns.tolist():
    df = df.astype(data_types_dict)
elif 'time_start'.upper() in df.columns.tolist():
    data_types_dict_upper = {}
    for i, j in data_types_dict.items():
        data_types_dict_upper[i.upper()] = j
    # print('=====', data_types_dict_upper)
    df = df.astype(data_types_dict_upper)
```

如果需要进行列名修改也是类似的：

```
rename_col = {
    'time_start': 'time',
    'ip_client': 'sIP',
    'byte_up': 'outlen',
    'byte_dn': 'inlen'
}

if 'time_start' in df.columns.tolist():
    df = df.rename(columns=rename_col)
elif 'time_start'.upper() in df.columns.tolist():
    rename_col_upper = {}
    for i, j in rename_col.items():
        rename_col_upper[i.upper()] = j
    # print('=====', rename_col_upper)
    df = df.rename(columns=rename_col_upper)
```

