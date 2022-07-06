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


记录下在X86和arm架构上部署CICFlowMeter的整体流程和各种问题，笔者在部署的时候真的是遇到过很多问题，部署到ARM上更是难搞，这里记录下顺便看能不能帮助其他人。

<!--more-->

## 部署整体流程

下面我把CICFlowMeter简写为CIC。这里先说一下联网部署的整体流程，在联网情况下部署一般只有一个问题，就是jnetpcap依赖的安装。

> 在这里特别提醒一下，尽量不要使用open jdk，笔者使用openjdk就会有问题，建议最好使用Oracle的JDK！！！！！！
>
> 还有就是笔者尝试过在相同操作系统，相同架构的电脑上完成打包的CIC是可以直接复制到别的电脑上的，这就意味着在另一台电脑上不用配置mvn和gradle这些内容，只需要配置JDK然后把打出来的压缩包复制过去，然后配置好libpcap-dev和jnetpcap依赖就可以使用了，不同系统和架构的笔者没有尝试过。

### 环境配置

笔者使用是自己下载的软件包，所以要自己配置下环境

- etc/profile

- /home/user(用户名)/.bashrc

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
- 安装libpcap-dev依赖（centos里面好像是libpcap-devel）

```
sudo apt-get install libpcap-dev
yum install libpcap-devel
```

- 将/CICFlowMeter-master/jnetcap/linux/jnetpcap-1.4.r1425里的libjnetpcap.so和libjnetpcap-pcap100.so复制到/Traffic_arm/jdk1.8.0_333/jre/lib/aarch64（或amd64，和平台架构相关）目录下
- 在/***/Traffic_arm/CICFlowMeter-master/build/distributions/CICFlowMeter-4.0/bin运行cfm文件即可

```
./cfm [pcap_file] [target_path]
/***/Traffic/CICFlowMeter-master/build/distributions/CICFlowMeter-4.0/bin/cfm ***.pcap /***/
```

### 相关描述

- 按照上述操作在有网的x86的Ubuntu中应该是不会出现问题的，笔者已经在多台X86的Ubuntu中成功部署。
- 在一个全新的机子上部署是要走完以上全部流程的，但是笔者本身不会JAVA所以不是很清楚mvn和gradle这样使用的原理，[CICgithub](https://github.com/ahlashkari/CICFlowMeter)上作者就是这么用的，github上的流程是在为了在IDE上能使用，但是我们希望把他变成一个工具在任何地方都可以通过命令行使用，所以要把两个so动态库放到jre里面去，这个大概就是要我们在命令行全局使用的时候能找到jnetpcap的动态链接库，这就涉及到了JAVA本身的依赖管理，~~笔者确实不懂JAVA，都是师兄教的😀。~~
- 在X86的Ubuntu上使用上述步骤理论上讲不存在问题了，但是在别的操作系统或者非X86架构上还是存在问题，一般主要是遇到下面这个问题

```
cic.cs.unb.ca.ifm.Cmd You select: /Integ/pcapsource/d1/1_00001_20210104112753.pcap
cic.cs.unb.ca.ifm.Cmd Out folder: ./
cic.cs.unb.ca.ifm.Cmd CICFlowMeter received 1 pcap file
Exception in thread "main" java.lang.UnsatisfiedLinkError: com.slytechs.library.NativeLibrary.dlopen(Ljava/lang/String;)J
        at com.slytechs.library.NativeLibrary.dlopen(Native Method)
        at com.slytechs.library.NativeLibrary.<init>(Unknown Source)
        at com.slytechs.library.JNILibrary.<init>(Unknown Source)
        at com.slytechs.library.JNILibrary.loadLibrary(Unknown Source)
        at com.slytechs.library.JNILibrary.register(Unknown Source)
        at com.slytechs.library.JNILibrary.register(Unknown Source)
        at com.slytechs.library.JNILibrary.register(Unknown Source)
        at org.jnetpcap.Pcap.<clinit>(Unknown Source)
        at cic.cs.unb.ca.jnetpcap.PacketReader.config(PacketReader.java:58)
        at cic.cs.unb.ca.jnetpcap.PacketReader.<init>(PacketReader.java:52)
        at cic.cs.unb.ca.ifm.Cmd.readPcapFile(Cmd.java:128)
        at cic.cs.unb.ca.ifm.Cmd.main(Cmd.java:80)
```

- 这个问题根据笔者自己的尝试和网上的 [参考](https://blog.csdn.net/lizheng2017/article/details/121455590) 基本可以确定是由于jnetpcap这个东西引起的，应该是jnetpcap这个东西在不同架构下不兼容的问题，在arm架构上使用CICgithub中提供的so是不行的，必须要自己重新编译，甚至我之后在部署一台X86的centos的时候这个os也不能用最后还是笔者自己重新编译之后才能使用，这个东西的兼容性确实很差，实在不行了需要编译安装的时候可以看下面的内容。

## X86架构

> 对X86架构下的Ubuntu系统来讲，按照上述流程应该不会存在问题，github上提供的jnetpcap的so本身是适配ubuntu的。
>
> 对X86架构下的Centos和其他系统来讲，可能会遇到上面jnetpcap报错的问题，那应该就是github上提供的so依赖和系统不兼容，需要重新编译，或者看网上有没有好心人提供编译好的，或者看下面的jnetpcap编译安装部分。

## ARM架构

> ARM架构下的部署，最主要的问题还是在jnetpcap上，github上提供的so肯定是用不了的，笔者也亲身体验过，确实不行。但是笔者这里在CSDN上搜到了一个好心老哥自己在银河麒麟V10， aarch64架构上编译出来的结果，这里贴出来供大家参考。

[ARM架构依赖](https://blog.csdn.net/lizheng2017/article/details/121455590)

## jnetpcap编译安装

以下内容全部来源于上面ARM架构中提到的好心老哥，而且他还提供了他编译好的so，真的是大善人，我怕他的博客哪天没了所以我这里复制一遍[ARM架构依赖](https://blog.csdn.net/lizheng2017/article/details/121455590)

-----------

**编译环境**：银河麒麟V10， aarch64架构
网上很容易下载到jnetpcap的包，但是却没有arm64架构的编译好的so，于是下载源码包自己编译。下面是遇到的问题解决，整个流程耗时6个多小时：

1. 下载jnetpcap-src-1.4.r1425-1.zip并解压：

![解压的文件](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061639724.png)

2. 安装ant， 命令大概是：

```
sudo apt install ant
```

3. 编译，问题1，xml:119: taskdef class org . vafer.jdeb . ant DebAntTask cannot be found using the classloader AntclassL oader[ ]

![问题1](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061639887.png)

​	方法：直接将build.xml第119行删除，不是debian系统。

4. 编译，问题2，use aresiurce collection to copy directories.

![问题2](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061640674.png)

​	根据提示，xml的611行有问题，直接找到你系统的libpcap.so的路径，填入xml：

![问题行](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061640724.png)

![解决方法](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061640701.png)

5. 编译，问题3，网上的答案都说是xml的问题，但在这里其实是少了cpptask.jar的原因。

```
Problem: failed to create task or type cc
Cause: The name is undefined.
Action: Check the spelling.
Action: Check that any custom tasks/types have been declared.
Action: Check that any <presetdef>/<macrodef> declarations have taken place.
```

![问题3](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061641290.png)

​	解决办法：下载cpptasks-1.0b4.jar，放到jnetpcap的lib文件夹下。

​	这里的具体cpptasks.jar的版本需要你查看xml文件，放的路径也是要看xml，直接在xml中搜索cpptask相信你能找到线索的。

​	ps：我这里是放的路径是[项目路径]/cpptasks-1.0b4/cpptasks.jar，然后在xml里面路径直接写死

6. 编译，问题四：

```
<C ommand- line>:0: 19: error: token “”is not valid in preprocessor expressions/home/ka/ jnetpcap-src-1.4.1425-1/src/c/jnetpcap_pcap100.cpp:87:6: note: in expansion of macro ' L IBPCAP_ VERSION '
#if (L IBPCAP VERSION < L IBPCAP PCAP CREATE )
```

![问题四](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061643419.png)

​	解决，明显是找不到LIBPCAP_VERSION，即libpcap的版本号没有，我们编辑Linux.properties文件，添加如下行

```
complier.LIBPCAP_VERSION = 174
(具体版本可用apt list|grep libpcap获取，填个很大的数就行)，这里一定是一个数字，不能像1.7.4这样。
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061644038.png)

7. 问题五

```
	[javac] 注: 有关详细信息, 请使用 -Xlint:deprecation 重新编译。
    [javac] 注: 某些输入文件使用了未经检查或不安全的操作。
    [javac] 注: 有关详细信息, 请使用 -Xlint:unchecked 重新编译。
    [javac] 78 个错误

BUILD FAILED
/home/ka/jnetpcap-src-1.4.r1425-1/build.xml:1090: Compile failed; see the compiler error output for details.
        at org.apache.tools.ant.taskdefs.Javac.compile(Javac.java:1181)
        at org.apache.tools.ant.taskdefs.Javac.execute(Javac.java:936)
        at org.apache.tools.ant.UnknownElement.execute(UnknownElement.java:293)
        at sun.reflect.GeneratedMethodAccessor4.invoke(Unknown Source)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:498)
        at org.apache.tools.ant.dispatch.DispatchUtils.execute(DispatchUtils.java:106)
        at org.apache.tools.ant.Task.perform(Task.java:348)
        at org.apache.tools.ant.Target.execute(Target.java:435)
        at org.apache.tools.ant.Target.performTasks(Target.java:456)
        at org.apache.tools.ant.Project.executeSortedTargets(Project.java:1405)
        at org.apache.tools.ant.Project.executeTarget(Project.java:1376)
        at org.apache.tools.ant.helper.DefaultExecutor.executeTargets(DefaultExecutor.java:41)
        at org.apache.tools.ant.Project.executeTargets(Project.java:1260)
        at org.apache.tools.ant.Main.runBuild(Main.java:853)
        at org.apache.tools.ant.Main.startAnt(Main.java:235)
        at org.apache.tools.ant.launch.Launcher.run(Launcher.java:285)
        at org.apache.tools.ant.launch.Launcher.main(Launcher.java:112)
```

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061645003.png)

​	这个问题可能是我没有下载正确版本的cpptask的原因，要根据build.xml里的要求来下载。
但是检查./build/obj/我已经得到了我们想要的so文件

![大功告成](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061645493.png)

程序终于跑起来了！！！！！！！

![](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061646710.png)

### 附编译好的文件链接

https://download.csdn.net/download/lizheng2017/46217421

## 离线部署

离线部署其实挺麻烦的，如果平时没有经常使用Linux服务器的话还是比较恼火的，最好的方法还是让机子联网，但是确实有很多情况下没法联网，笔者这里也只能提供我在尝试离线部署过程中的一些经验

- 尽量找到一台相同系统相同架构的能联网的机器，这会让离线部署简单得多，apt和yum都提供了下载安装包的功能，我们可以在能联网的机子上先下载好适配的离线包然后复制到不能联网的机子上进行编译安装，这样的成功率就会很高，也比较方便，这里贴两个参考[apt离线包下载](https://blog.csdn.net/qq_17576885/article/details/122070612)，[yum离线包下载](https://www.codeleading.com/article/78986008408/)
- 相同系统相同架构的机器不用每个都使用mvn和gradle，只要有一台成功打出了压缩包后面就可以直接复制，只要配置新机器的JDK和各种依赖就行了

## 其他问题

### 问题1

> 问题描述：经过上面的一通操作，CIC已经成功跑了起来，我把CIC用python封装成了一个接口供后端调用，但是出现了一个非常奇怪的问题，那就我直接跑我的接口CIC稳定能用，但是后端调我的接口就时而能用时而不行，一旦不行之后就稳定不行。

> 这个问题的产生是因为后端使用Linux的/etc/crontab做了后端服务的定时重启，但是crontab里面有一个自己的$PATH，并且它不会去读取/etc/profile里面的路径，所以会导致读不到我们的JDK和各种依赖，解决方法也很简单，只要把我们在profile里面写的JDK路径加到crontab里面就行了

### 问题2

> 问题描述：在实际使用中我们发现提取出来的协议号基本只有0，6，17。6代表TCP，17代表UDP，剩下其他所有协议全部被识别为0，例如ARP，ICMP，SEP之类的全部识别为0，而且组流的情况看起来也不是很好，绝大部分列都是全为0，简单看了下代码发现CIC本事就只对有限的几种protocol做了处理，泛用性不是很广。

![protocol](https://zchsakura-blog.oss-cn-beijing.aliyuncs.com/202207061724179.png)
