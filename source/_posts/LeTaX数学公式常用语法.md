---
title: LeTaX数学公式常用语法
date: 2020-03-14 17:35:38
tags: [LetaX]
categories:
- 工程实践
- 经验积累
top: 10
---
<center>码一些经常使用的LeTaX数学公式语法</center>

![](https://download.kezhi.tech/img/8QXOvd.jpg)

<!--more-->

### 1.上标与下标

>上标命令为 "^"， 下标命令为 "_"

$$x_1$$ $$x_1^2$$ $$x^2_1$$ $$x_{22}^{(n)}$$ $${}^*\!x^*$$

```letax
$$x_1$$
$$x_1^2$$
$$x^2_1$$
$$x_{22}^{(n)}$$
$${}^*\!x^*$$
```

### 2.分式

>输入较短的分式时，可以使用 "/"来输入
对于输入带有水平分数线的分式，使用 "\frac{分子}{分母}"

$$ (x+y)/2 $$$$ \frac{x+y}{2} $$

```letax
$$ (x+y)/2 $$
$$ \frac{x+y}{2} $$
```

### 3.根式

>排版根式的命令是：开平方：\sqrt{表达式}；开 n 次方：\sqrt[n]{表达式}

$$\sqrt{2}<\sqrt[3]{3}$$$$\sqrt{1+\sqrt[p]{1+a^2}}$$$$\sqrt{1+\sqrt[^p\!]{1+a^2}}$$

```letax
$$\sqrt{2}<\sqrt[3]{3}$$
$$\sqrt{1+\sqrt[p]{1+a^2}}$$
$$\sqrt{1+\sqrt[^p\!]{1+a^2}}$$
```

### 4.求和和积分

>排版求和符号与积分符号的命令分别为 \sum 和 \int，它们通常都有上下限，在排版上就是上标和下标。

$$\sum_{k=1}^{n}\frac{1}{k}$$$$\sum_{k=1}^n\frac{1}{k}$$$$\int_a^b f(x)dx$$$$\int_a^b f(x)dx$$$$\int_a^b f(x)\mathrm{d}x$$

```letax
$$\sum_{k=1}^{n}\frac{1}{k}$$
$$\sum_{k=1}^n\frac{1}{k}$$
$$\int_a^b f(x)dx$$
$$\int_a^b f(x)dx$$
$$\int_a^b f(x)\mathrm{d}x$$
```

### 5.空格

>LaTeX 能够自动处理公式中的大多数字符之间的空格，但是有时候需要自己手动进行控制。

紧贴  $a\!b$
没有空格 $ab$
小空格 $a\,b$
中等空格 $a\;b$
大空格 $a\ b$
quad空格 $a\quad b$
两个quad空格 $a\qquad b$

```letax
紧贴  $a\!b$
没有空格 $ab$
小空格 $a\,b$
中等空格 $a\;b$
大空格 $a\ b$
quad空格 $a\quad b$
两个quad空格 $a\qquad b$
```

### 6.矩阵

>对于少于 10 列的矩阵，可使用 matrix，pmatrix，bmatrix，Bmatrix，vmatrix 和 Vmatrix 等环境。

$$\begin{matrix}1 & 2\\3 &4\end{matrix}$$$$\begin{pmatrix}1 & 2\\3 &4\end{pmatrix}$$$$\begin{bmatrix}1 & 2\\3 &4\end{bmatrix}$$$$\begin{Bmatrix}1 & 2\\3 &4\end{Bmatrix}$$$$\begin{vmatrix}1 & 2\\3 &4\end{vmatrix}$$$$\begin{Vmatrix}1 & 2\\3 &4\end{Vmatrix}$$

```letax
$$\begin{matrix}1 & 2\\3 &4\end{matrix}$$
$$\begin{pmatrix}1 & 2\\3 &4\end{pmatrix}$$
$$\begin{bmatrix}1 & 2\\3 &4\end{bmatrix}$$
$$\begin{Bmatrix}1 & 2\\3 &4\end{Bmatrix}$$
$$\begin{vmatrix}1 & 2\\3 &4\end{vmatrix}$$
$$\begin{Vmatrix}1 & 2\\3 &4\end{Vmatrix}$$
```

### 7.排版数组

>当矩阵规模超过 10 列，或者上述矩阵类型不符需求，可使用 array 环境。该环境可把一些元素排列成横竖都对齐的矩形阵列。

$$
\mathbf{X} =
\left( \begin{array}{ccc}
x_{11} & x_{12} & \ldots \\
x_{21} & x_{22} & \ldots \\
\vdots & \vdots & \ddots
\end{array} \right)
$$

```letax
$$
\mathbf{X} =
\left( \begin{array}{ccc}
x_{11} & x_{12} & \ldots \\
x_{21} & x_{22} & \ldots \\
\vdots & \vdots & \ddots
\end{array} \right)
$$
```

>\mathbf大写控制符，\\表示换行，{ccc}表示列样式。array 环境也可以用来排版这样的表达式，表达式中使用一个“.” 作为其隐藏的\right 定界符。

$$
y = \left\{ \begin{array}{ll}
a & \textrm{if $d>c$}\\
b+x & \textrm{in the morning}\\
l & \textrm{all day long}
\end{array} \right.
$$

```letax
$$
y = \left\{ \begin{array}{ll}
a & \textrm{if $d>c$}\\
b+x & \textrm{in the morning}\\
l & \textrm{all day long}
\end{array} \right.
$$
```

>你也可以在array 环境中画线，如分隔矩阵中元素。

$$
\left(\begin{array}{c|c}
1 & 2 \\
\hline
3 & 4
\end{array}\right)
$$

```letax
$$
\left(\begin{array}{c|c}
1 & 2 \\
\hline
3 & 4
\end{array}\right)
$$
```

### 8.极限符号

$$
{\lim_{x \to +\infty}}
$$

```letax
$${\lim_{x \to +\infty}}$$
```

$$
{\lim_{x \to -\infty}}
$$

```letax
$${\lim_{x \to -\infty}}$$
```

$$
{\lim_{x \to 0}}
$$

```letax
$${\lim_{x \to 0}}$$
```

$$
{\lim_{x \to 0^+}}
$$

```letax
$${\lim_{x \to 0^+}}$$
```
