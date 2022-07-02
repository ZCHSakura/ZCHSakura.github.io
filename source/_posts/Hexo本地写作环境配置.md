---
title: Hexo本地写作环境配置
tags:
  - Hexo
  - node.js
categories:
  - 工程实践
  - 经验积累
top: 20
date: 2022-03-20 23:21:53# 
---

## 写作缘由

由于经常换电脑设备，每次更换设备时都需要重新配置Hexo环境，故对配置流程进行记录，以供后续使用。

<!--more-->

## 本地环境搭建

1. 安装node.js （这里最好安装版本 v12 否则可能出现部署不成功的情况）

   window 安装：[Download | Node.js (nodejs.org)](https://nodejs.org/en/download/)

   Linux 安装：

   ```sh
   $ sudo apt-get install nodejs
   $ sudo apt-get install npm
   ```

   安装完成后输入以下命令检查是否安装成功：

   ```sh
   $ node -v
   $ npm -v
   ```

   

2. 安装Hexo

   首先修改 npm 到国内源：

   ```sh
   $ npm config set registry http://registry.npm.taobao.org
   ```

   输入以下命令进行确认：

   ```sh
   $npm get registry
   ```

   在 Blog 所在文件夹根目录打开命令行，输入以下命令强制安装Hexo：

   ```sh
   $ npm install -g hexo-cli --force
   ```

   安装完成后输入以下命令检查是否安装成功：

   ```sh
   $ hexo -v
   ```



## 插件

### steam 展示

项目链接：https://github.com/HCLonely/hexo-steam-games

1. 安装插件

   ```shell
   $ npm install hexo-steam-games --save
   ```



2. 配置`hexo`的`_config.yml`文件

   ```yaml
   steam:
     enable: true
     steamId: '*****' #steam 64位Id
     path:
     title: Steam游戏库
     quote: '人人都是头号玩家'
     tab: all
     length: 1000
     imgUrl: '*****'
     proxy:
       host:
       port:
     extra_options:
       key: value
   ```

   - **enable**: 是否启用
   - **steamId**: steam 64位Id(需要放在引号里面，不然会有BUG), ***需要将steam库设置为公开！***
   - **path**: 页面路径，默认`steamgames/index.html`
   - **title**: 该页面的标题
   - **quote**: 写在页面开头的一段话,支持html语法
   - **tab**: `all`或`recent`, `all: 所有游戏`, `recent: 最近游玩的游戏`
   - **length**: 要显示游戏的数量，游戏太多的话可以限制一下
   - **imgUrl**: 图片链接，在`quote`下面放一张图片，图片链接到Steam个人资料，可留空
   - proxy: 如果无法访问steam社区的话请使用代理
     - **host**: 代理ip或域名
     - **port**: 代理端口
   - **extra_options**: 此配置会扩展到Hexo的`page`变量中

3. 使用

   1. 在`hexo generate`或`hexo deploy`之前使用`hexo steam -u`命令更新steam游戏库数据！
   2. 删除游戏库数据指令:`hexo steam -d`

### 添加 Bilibili 追番/追剧列表

参考自：https://opensourcelibs.com/lib/hexo-bilibili-bangumi

1. 安装

   ```shell
   $ npm install hexo-bilibili-bangumi --save
   ```

2. 配置`hexo`的`_config.yml`文件

   ```yaml
   # 追番设置
   bangumi:
     enable: true
     path:
     vmid:
     title: '追番列表'
     quote: '生命不息，追番不止！'
     show: 1
     lazyload: true
     loading:
     metaColor:
     color:
     webp:
     progress:
     extra_options:
       key: value
   
   # 追剧设置
   cinema:
     enable: true
     path:
     vmid:
     title: '追剧列表'
     quote: '生命不息，追剧不止！'
     show: 1
     lazyload: true
     loading:
     metaColor:
     color:
     webp:
     progress:
     extra_options:
       key: value
   ```

   - **enable**: 是否启用
   - **path**: 页面路径，默认`bangumis/index.html`, `cinemas/index.html`
   - **vmid**: 哔哩哔哩的 `vmid(uid)`,[如何获取？](https://opensourcelibs.com/lib/hexo-bilibili-bangumi#获取uid)
   - **title**: 该页面的标题
   - **quote**: 写在页面开头的一段话，支持 html 语法，可留空。
   - **show**: 初始显示页面：`0: 想看`, `1: 在看`, `2: 看过`，默认为`1`
   - **lazyload**: 是否启用图片懒加载，如果与主题的懒加载冲突请关闭，默认`true`
   - **loading**: 图片加载完成前的 loading 图片，需启用图片懒加载
   - **metaColor**: meta 部分(简介上方)字体颜色
   - **color**: 简介字体颜色
   - **webp**: 番剧封面使用`webp`格式(此格式在`safari`浏览器下不显示，但是图片大小可以缩小 100 倍左右), 默认`true`
   - **progress**: 获取番剧数据时是否显示进度条，默认`true`
   - **extra_options**: 此配置会扩展到Hexo`page`变量中

3. 使用

   1. 在`hexo generate`或`hexo deploy`之前使用`hexo bangumi -u`命令更新追番数据，使用`hexo cinema -u`命令更新追剧数据！
   2. 删除数据命令:`hexo bangumi -d`/`hexo cinema -d`

## 小结

以上步骤结束后一般即可以进行写作和部署，若依然存在问题可以访问[Troubleshooting | Hexo](https://hexo.io/docs/troubleshooting.html)进行错误排查。

