---
title: ssh 22 time out
tags:
  - git
categories:
  - 技术
  - 服务器配置
date: 2022-10-09 10:31:36
top:
---


这里记录一下git使用过程中总是碰到的一个错误，`ssh: connect to host github.com port 22: Connection timed out`。

<!--more-->

我碰过的这个错误基本是由两种原因引起的，一种是公钥没配置好，另一种就是网络原因。

这里主要记录下网络原因引起的time out该怎么办。

### 检测

`ssh -T git@github.com`

如果出现：You’ve successfully authenticated，那么连接成功可以使用了。如果出现：ssh: connect to host github.com port 22: Connection timed out，就说明连接超时。

连接失败后，可以同样试试`ssh -T -p 443 git@github.com`，检查是否有异常。

如果加上443之后能够success的话基本就是网络问题，我们需要使用443端口去连接git。

### 配置

1. cd ~/.ssh
2. 配置或新建config文件
3. 编辑文件内容并保存退出

```shell
Host github.com
User 你的邮箱
Hostname ssh.github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa
Port 443
```

4. 使用`ssh -T git@github.com`测试下
