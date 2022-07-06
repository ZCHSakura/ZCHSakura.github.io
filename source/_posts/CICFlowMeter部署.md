---
title: CICFlowMeter部署
tags:
  - CICFlowMeter
categories:
  - 技术
  - CICFlowMeter
date: 2022-07-06 10:54:45
top:
---


记录下在X86和arm架构上部署CICFlowMeter的整体流程和各种问题

<!--more-->

## 部署整体流程

这里先说一下联网部署的整体流程，在联网情况下部署一般只有一个问题，就是jnetpcap依赖的安装。

### 环境配置

etc/profile

/home/user(用户名)/.bashrc

在上述两个文件末尾中添加下列环境变量，添加完成后使用source etc/profile和source /home/user/.bashrc

- 配置jdk 1.8.0_311

```
export JAVA_HOME=/***/Traffic_arm/jdk1.8.0_333
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
```

- 配置maven 3.8.4

```
export MAVEN_HOME=/***/Traffic_arm/apache-maven-3.8.4
export PATH=$PATH:$MAVEN_HOME/bin
```

- 配置gradle 3.3

```
export GRADLE_HOME=/***/Traffic_arm/gradle-3.3
export PATH=$GRADLE_HOME/bin:$PATH
```

### 验证环境是否配置成功

- 使用java -version查看java版本是否与环境变量中一致

- 使用which java查看路径是否与环境变量一致

- 如果which java显示与环境变量不一致，而是/usr/bin/java之类的路径则使用以下命令（这种情况主要是机子上本身存在jdk，理论上将如果在配置环境变量时如果把新的路径放在$PATH前的话是不会出现这个问题的）

```
sudo rm /usr/bin/java（which java 结果）
sudo ln -s /***/Traffic_x86/jdk1.8.0_311/bin/java /usr/bin/java（which java 结果）
```

- 再使用java -version和which java验证环境配置结果
- 如果还没有生效尝试关闭shell重开一个shell或切换用户看有无生效
- 如果出现权限不够的情况，使用sudo chmod 777 path 修改权限

### 导入jnetpcap

- Dfile路径改为/Traffic_arm/CICFlowMeter-master/jnetcap/linux/jnetpcap-1.4.r1425下的jnetpcap.jar

```
mvn install:install-file -Dfile=/***/jnetpcap.jar -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar
```

- 在/Traffic_arm/CICFlowMeter-master下构建项目

```
./gradlew distZip
```

- 在/Traffic_arm/CICFlowMeter-master/build/distributions里解压缩CICFlowMeter-4.0.zip

- 将/CICFlowMeter-master/jnetcap/linux/jnetpcap-1.4.r1425里的libjnetpcap.so和libjnetpcap-pcap100.so复制到/Traffic_arm/jdk1.8.0_333/jre/lib/aarch64（或amd64，和平台架构相关）目录下

- 在/***/Traffic_arm/CICFlowMeter-master/build/distributions/CICFlowMeter-4.0/bin运行cfm文件即可

```
./cfm [pcap_file] [target_path]
/***/Traffic/CICFlowMeter-master/build/distributions/CICFlowMeter-4.0/bin/cfm ***.pcap /***/
```

### 相关描述

- 按照上述操作在有网的x86的Ubuntu中应该是不会出现问题的,笔者已经在多台Ubuntu中成功部署.
- 

## X86架构

## ARM架构

## jnetpcap编译安装

## 其他问题



