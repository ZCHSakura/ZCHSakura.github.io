---
title: Vue学习笔记
tags:
  - Vue
  - 前端框架
categories:
  - 工程实践
  - Web
top: 10
date: 2021-04-13 11:32:27
---


## Vue

### 与Flask一起使用时的注意事项

Vue与Flask一起使用时会出现冲突，更改配置即可解决。

- Flask

```python
if __name__ == '__main__':
    app.jinja_env.variable_start_string = '[['
    app.jinja_env.variable_end_string = ']]'
    app.run(debug=True)
```

- Vue

```javascript
new Vue({
    delimiters: ['[[',']]']
})
```
