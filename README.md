# SOC_FaultDetection_demo

关键词：18650电池充放电的实时状态估计和故障诊断；集员滤波；

飞书文档链接： https://s08emqdplxz.feishu.cn/wiki/JKp0wuLNyiOJVqk8PVscOvRZnHb?from=from_copylinkgai

## 一、运行环境：

该项目运行在lubancat4开发板上，芯片为瑞芯微rk3588s，架构为arm64（aarch64）。

* cpp程序需要运行可执行程序，运行时进入bin文件夹目录下；
* py代码运行直接使用`python3+文件名`

## 二、复现步骤：

### 1  状态估计和故障诊断功能

1.使用MobaXterm登录开发板：

```
ssh cat@192.168.31.131
pwd:temppwd
```

2.拉取仓库：

```
cd ~
git clone https://github.com/Mayq1028/SOC_FaultDetection_demo.git
```

3.编译

```
cd SOC_FaultDetection_demo
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/eigen3/ 
make
```

4.运行

```
cd ../bin
sudo chmod 777 /dev/ttyUSB0
在线：sudo ./online
离线：sudo ./offline
```

5.按 `ctrl+C`结束运行

6.绘图预览

状态估计生成的数据存储在 data/output/*.csv

```
cd ../data/output
python3 plot_output.py online.csv
python3 plot_output.py offline.csv
```

### 2   电池放电数据采集步骤：

```
cd src/tty_proc
```

采集并保存：

```
python3 save_tty_data.py
```

读取串口数据：

```
python3 read_tty.py
```

## 三、工程目录解释

```
cat@lubancat:~/SOC_FaultDetection_demo$ tree .
.
├── bin                           # 存放可执行文件
│   ├── deltaT_adjust   
│   ├── offline                   # 数据来源：测试集
│   └── online                    # 数据来源：实时读取串口数据
├── data                          # 存放所有数据文件、数据处理程序  
│   ├── collect_data              # 采集电池放电数据
│   │   ├── collection_1          # 第一次采集（下午），原始数据，未处理，曲线下降漂亮，但最低只降到3.95V，不合理
│   │   │   ├── Cur.txt
│   │   │   └── Vol.txt
│   │   ├── collection_2          # 第二次采集（晚上），考虑串口数据和oled显示数据有出入，故对串口数据微调，最低只到3.4V
│   │   │   ├── Cur.txt
│   │   │   └── Vol.txt
│   │   ├── Cur.txt               # 次路径下直接存的是最新一次的采集数据，如需保存，需要手动移进新建文件夹；不保存，则下一次运行则会覆盖此路径下的**.txt文件
│   │   ├── plot_raw.py           # 绘制串口采集的原始数据图像
│   │   └── Vol.txt
│   ├── output                    # 状态估计输出数据
│   │   ├── offline.csv   
│   │   ├── online.csv
│   │   └── plot_output.py        # 绘制状态估计输出曲线
│   └── pre_data                  # 离线模式所需的测试集
│       ├── current.txt           # 电流
│       ├── mat_convert.py        # 把mat格式文件转换成txt
│       ├── pfit_offline.txt      # 数据集中，拟合得到的U_ocv关于SOC的非线性系数
│       ├── pfit_online.txt       # 根据串口采集的放电数据，拟合得到的U_ocv关于SOC的非线性系数
│       └── voltage.txt           # 端电压 
├── Makefile                      # 编译规则
├── obj                           # 编译过程object
│   ├── deltaT_adjust.o
│   ├── offline.o
│   └── online.o
├── README.md
├── src                           # 源码
│   ├── offline_pkg               # 离线功能包
│   │   ├── offline.cpp           # 结构体版本，可读性和可移植性更强（待完善，中心值发散）
│   │   └── offline_origin.cpp    # 单步过程，简单，但不易移植（待完善,之前跑出来过准确的状态估计中心值，但上下界发散）
│   ├── online_pkg                # 在线功能包
│   │   ├── deltaT_adjust.cpp     # 采样时间不稳定，此版本用于测试调整采样时间（待完善）
│   │   ├── online.cpp            # 主版本
│   │   └── online.py             # py版本采样时间比较准，考虑将cpp代码转成py（待完善）
│   └── tty_proc                  # 串口数据处理
│       ├── read_tty.py           # 读取串口数据
│       └── save_tty_data.py      # 采集并保存串口数据
└── test1
```
