---
title: NexT标签云配置
tags:
  - Hexo
  - NexT
categories:
  - 工程实践
  - 经验积累
top: 20
date: 2022-03-22 17:11:28
---

本文引自[Next 7.0+ TagCanvas 标签云 | Alex_McAvoy (alex-mcavoy.github.io)](https://alex-mcavoy.github.io/hexo/7be258c0.html)

![](https://download.kezhi.tech/img/20220322172925.jpg)

<!--more-->

Next 7.0+ 可以对标签进行自定义设置，在此基础上，可以使用球形标签云的 `tagcanvas.js` 插件进行样式修改，具体修改步骤如下



1. 下载插件

   关于球形标签云 `tagcanvas.js` 插件的详细介绍：[点击这里](http://www.goat1000.com/tagcanvas.php)

   将该插件下载后，放入 `/theme/next/source/js` 目录下



2. 新建标签云 swig 文件

   在 `/theme/next/layout/_partials` 目录下，建一个名为 `tagcanvas.swig` 的文件，并写入如下内容：

   ```html
   <div class="tags" id="myTags">
     <canvas width="500" height="500" id="my3DTags">
       <p>Anything in here will be replaced on browsers that support the canvas element</p>
     </canvas>
   </div>
   <div class="tags" id="tags">
     <ul style="display: none">
       {{ tagcloud({
           min_font   : theme.tagcloud.min,
           max_font   : theme.tagcloud.max,
           amount     : theme.tagcloud.amount,
           color      : true,
           start_color: theme.tagcloud.start,
           end_color  : theme.tagcloud.end})
       }}
     </ul>
   </div>
   <script type="text/javascript" src="/js/tagcanvas.js"></script>
   <script type="text/javascript" >
     window.onload = function() {
       try {
         TagCanvas.Start('my3DTags','tags',{
           textFont: 'Georgia,Optima',
           textColour: null,
           outlineColour: 'black',
           weight: true,
           reverse: true,
           depth: 0.8,
           maxSpeed: 0.05,
           bgRadius: 1,
           freezeDecel: true
         });
       } catch(e) {
         document.getElementById('myTags').style.display = 'none';
       }
     };
   </script>
   ```

3. 修改页面配置文件

   对 `/theme/next/layout/` 中的 `page.swig` 文件按照下图所示进行进行修改

   ![](https://download.kezhi.tech/img/20220322171837.png)

   添加代码为：

   ```html
   {# tagcanvas plugin 球型云标签 #}
   {% include '_partials/tagcanvas.swig' %}
   ```

4. 修改主题配置文件

   打开 `/theme/config.yml`，找到 `tagcloud` 字段，根据实际需要进行修改即可：

   ```yaml
   # TagCloud settings for tags page.
   tagcloud:
     # All values below are same as default, change them by yourself.
     min: 20 # Minimun font size in px
     max: 30 # Maxium font size in px
     start: "#19CAAD" # Start color (hex, rgba, hsla or color keywords)
     end: "#F4606C" # End color (hex, rgba, hsla or color keywords)
     amount: 200 # Amount of tags, change it if you have more than 200 tags
   ```

   
