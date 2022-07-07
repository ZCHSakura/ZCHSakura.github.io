---
title: python调用CIC
tags:
  - python
  - CICFlowMeter
categories:
  - 技术
  - CICFlowMeter
date: 2022-07-07 16:48:11
top:
---


在这里记录下使用python通过命令行调用CIC的过程，包含给pcap整体打标签，多个pcap整合等内容

<!--more-->

> 整体思路比较简单，使用python自带的**subprocess**来通过命令行调用我们部署好的CIC程序，然后将多个pcap包提取出的CSV进行打标签和整合

### 部分介绍

首先是引入必要的库，其中的**CIC_dfm**就是CIC部署成功后dfm文件的绝对路径

```python
import subprocess
import os
import pandas as pd
import time
from utils.db_config import CIC_dfm
```

然后是调用CIC部分

```python
cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
# 调CIC
p = subprocess.Popen(cmd, shell=True)
# 阻塞等待CIC执行完成
return_code = p.wait()
```

之后就是对PCAP列表中的每一个PCAP包调用CIC进行处理，然后将处理完的结果放到一个df里最后再存成CSV

```python
for pcap_path in pcap_list:
    pcap_name = pcap_path.split(os.sep)[-1].replace('.pcap', '')
    cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
    # 调CIC
    p = subprocess.Popen(cmd, shell=True)
    return_code = p.wait()
    csv_path = os.path.join(csv_save_folder, pcap_name + '.pcap_Flow.csv')
    if pcap_path == pcap_list[0]:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    else:
        temp_df = pd.read_csv(csv_path, skipinitialspace=True)
        df = pd.concat([df, temp_df], ignore_index=True)
    # 删除中间结果
    os.remove(csv_path)

csv_save_path = os.path.join(csv_save_folder, '%s.csv' % time_now)
df.to_csv(csv_save_path, index=0)
```

### 完整程序

```python
import subprocess
import os
import pandas as pd
import time
from utils.db_config import CIC_dfm


def unsupervise_extract(mode_id: int, pcap_list: list, csv_save_folder: str):
    """
    无监督算法流量特征提取，包含训练，离线测试和在线测试
    :param mode_id:
        1为训练;
        2为离线测试;
        3为在线测试;
    :param pcap_list: 要进行特征提取的pcap包路径列表(在线测试时里面只能包含一个)
    :param csv_save_folder: 提取出的csv文件要保存的目录
    :return: csv_save_path: 提取出的csv文件路径
    """

    time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(csv_save_folder, exist_ok=True)
    if mode_id not in [1, 2, 3]:
        print('mode_id出错')
        return 0
    if mode_id == 3:
        try:
            assert len(pcap_list) == 1
        except:
            print('在线测试时pcap_list中只能包含一个pcap包')
            return 0
    for pcap_path in pcap_list:
        pcap_name = pcap_path.split(os.sep)[-1].replace('.pcap', '')
        cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
        # 调CIC
        p = subprocess.Popen(cmd, shell=True)
        return_code = p.wait()
        csv_path = os.path.join(csv_save_folder, pcap_name + '.pcap_Flow.csv')
        if pcap_path == pcap_list[0]:
            df = pd.read_csv(csv_path, skipinitialspace=True)
        else:
            temp_df = pd.read_csv(csv_path, skipinitialspace=True)
            df = pd.concat([df, temp_df], ignore_index=True)
        # 删除中间结果
        os.remove(csv_path)

    csv_save_path = os.path.join(csv_save_folder, '%s.csv' % time_now)
    df.to_csv(csv_save_path, index=0)
    return csv_save_path


def supervise_extract(mode_id: int, pcap_list: list, norm_pcap_list: list,
                      abnorm_pcap_list: list, csv_save_folder: str):
    """
    有监督算法流量特征提取，包含训练，离线测试和在线测试
    :param mode_id:
        1为训练;
        2为离线测试;
        3为在线测试;
    :param pcap_list: 在线测试要进行特征提取的pcap包路径列表(里面只能包含一个)
    :param norm_pcap_list: 训练和离线测试要进行特征提取的全正常流量pcap包路径列表(至少包含一个)
    :param abnorm_pcap_list: 训练和离线测试要进行特征提取的全异常流量pcap包路径列表(至少包含一个)
    :param csv_save_folder: 提取出的csv文件要保存的目录
    :return: csv_save_path: 提取出的csv文件路径
    """

    time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(csv_save_folder, exist_ok=True)
    if mode_id not in [1, 2, 3]:
        print('mode_id出错')
        return 0
    if mode_id == 3:
        try:
            assert len(pcap_list) == 1
        except:
            print('在线测试时pcap_list中只能包含一个pcap包')
            return 0
    else:
        try:
            assert len(norm_pcap_list) > 0 and len(abnorm_pcap_list) > 0
        except:
            print('需要同时包含正常和异常pcap包')
            return 0
    if mode_id == 3:
        pcap_path = pcap_list[0]
        pcap_name = pcap_path.split(os.sep)[-1].replace('.pcap', '')
        cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
        # 调CIC
        p = subprocess.Popen(cmd, shell=True)
        return_code = p.wait()
        csv_path = os.path.join(csv_save_folder, pcap_name + '.pcap_Flow.csv')
        df = pd.read_csv(csv_path, skipinitialspace=True)
    else:
        for pcap_path in norm_pcap_list:
            pcap_name = pcap_path.split(os.sep)[-1].replace('.pcap', '')
            cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
            # 调CIC
            p = subprocess.Popen(cmd, shell=True)
            return_code = p.wait()
            csv_path = os.path.join(csv_save_folder, pcap_name + '.pcap_Flow.csv')
            if pcap_path == norm_pcap_list[0]:
                df_norm = pd.read_csv(csv_path, skipinitialspace=True)
            else:
                temp_df = pd.read_csv(csv_path, skipinitialspace=True)
                df_norm = pd.concat([df_norm, temp_df], ignore_index=True)
            df_norm['Label'] = 0
            # 删除中间结果
            os.remove(csv_path)

        for pcap_path in abnorm_pcap_list:
            pcap_name = pcap_path.split(os.sep)[-1].replace('.pcap', '')
            cmd = '%s %s %s' % (CIC_dfm, pcap_path, csv_save_folder)
            # 调CIC
            p = subprocess.Popen(cmd, shell=True)
            return_code = p.wait()
            csv_path = os.path.join(csv_save_folder, pcap_name + '.pcap_Flow.csv')
            if pcap_path == abnorm_pcap_list[0]:
                df_abnorm = pd.read_csv(csv_path, skipinitialspace=True)
            else:
                temp_df = pd.read_csv(csv_path, skipinitialspace=True)
                df_abnorm = pd.concat([df_abnorm, temp_df], ignore_index=True)
            df_abnorm['Label'] = 1
            # 删除中间结果
            os.remove(csv_path)

        df = pd.concat([df_abnorm, df_norm], ignore_index=True)

    csv_save_path = os.path.join(csv_save_folder, '%s.csv' % time_now)
    df.to_csv(csv_save_path, index=0)
    return csv_save_path


if __name__ == '__main__':
    pcap_list = ['/home/*/test1.pcap', '/home/*/test2.pcap', '/home/*/test3.pcap']
    # pcap_list = ['/home/*/test1.pcap']
    csv_save_folder = '/home/*'
    mode_id = 1
    _ = unsupervise_extract(mode_id, pcap_list, csv_save_folder)

    norm_pcap_list = ['/home/*/test1.pcap', '/home/*/test3.pcap']
    abnorm_pcap_list = ['/home/*/test2.pcap']
    pcap_list = ['/home/*/test1.pcap']
    mode_id = 2
    _ = supervise_extract(mode_id, pcap_list, norm_pcap_list, abnorm_pcap_list, csv_save_folder)

    print(_)
```

