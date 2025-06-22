#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图脚本：从 CSV 读取 EKF 估计结果，生成状态量与电压对比图。

用法：
    python plot_results.py data/plot_1.csv
"""

# ======== 1. 标准库 / 第三方库导入 ========
import csv                      # 处理 CSV 文件
import argparse                 # 命令行参数解析
import matplotlib.pyplot as plt # 绘图

# ======== 2. 读取数据函数 ========
def read_csv_data(csv_path):
    """
    从 csv_path 指定的 CSV 中读取所需列，返回一个字典。
    """
    # 初始化各列的列表
    k          = []
    Up_est     = []; Up_low  = []; Up_up  = []; Up_true  = []
    SOC_est    = []; SOC_low = []; SOC_up = []; SOC_true = []
    Vol_data   = []; y_est   = []; y_low  = []; y_up     = []; y_true = []

    # 打开并逐行读取 CSV
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)      # DictReader 用列名取值
        for row in reader:                    # 遍历文件中每一行
            # 采样序号
            k.append(float(row['k']))

            # x1 = Up
            Up_est .append(float(row['Up_est']))
            Up_low .append(float(row['Up_low']))
            Up_up  .append(float(row['Up_up']))
            Up_true.append(float(row['Up_true']))

            # x2 = SOC
            SOC_est .append(float(row['SOC_est']))
            SOC_low .append(float(row['SOC_low']))
            SOC_up  .append(float(row['SOC_up']))
            SOC_true.append(float(row['SOC_true']))

            # 测量与估计输出
            Vol_data.append(float(row['Vol']))
            y_est   .append(float(row['y_est']))
            y_low   .append(float(row['y_low']))
            y_up    .append(float(row['y_up']))
            y_true  .append(float(row['y_true']))

    # 将所有列表打包成 dict，便于后续调用
    return dict(k=k,
                Up_est=Up_est, Up_low=Up_low, Up_up=Up_up, Up_true=Up_true,
                SOC_est=SOC_est, SOC_low=SOC_low, SOC_up=SOC_up, SOC_true=SOC_true,
                Vol_data=Vol_data, y_est=y_est, y_low=y_low, y_up=y_up, y_true=y_true)

# ======== 3. 绘图函数 ========
def plot_results(data):
    """
    根据 data 字典绘制两张图：
    (1) Up / SOC 估计与真值
    (2) 电压观测与估计
    """
    # 为了书写简洁，解包常用字段
    k          = data['k']
    Up_est     = data['Up_est'];  Up_low  = data['Up_low'];  Up_up  = data['Up_up'];  Up_true  = data['Up_true']
    SOC_est    = data['SOC_est']; SOC_low = data['SOC_low']; SOC_up = data['SOC_up']; SOC_true = data['SOC_true']
    Vol_data   = data['Vol_data']; y_est = data['y_est'];    y_low  = data['y_low'];  y_up     = data['y_up']

    # ---------- 图 1：状态量 ----------
    plt.figure(figsize=(10, 8))              # 整体窗口大小

    # --- (a) x1 = Up ---
    plt.subplot(2, 1, 1)                     # 两行一列，第一行
    plt.plot(k, Up_true, 'k', label='x1')    # 真值黑线
    plt.plot(k, Up_est,  'r', label='x1_')   # 估计红线
    plt.plot(k, Up_low,  'b', label='x1_u,l')# 区间下界蓝线
    plt.plot(k, Up_up,   'b')                # 区间上界蓝线
    plt.title('x1 Up,k')                     # 标题
    # plt.ylim(-0.04, 0.1)                    # y 轴范围
    # plt.xlim(0, 10000)                        # x 轴范围
    plt.legend(); plt.grid(True)             # 图例 & 网格

    # --- (b) x2 = SOC ---
    plt.subplot(2, 1, 2)                     # 第二行
    plt.plot(k, SOC_true, 'k', label='x2')
    plt.plot(k, SOC_est,  'r', label='x2_')
    plt.plot(k, SOC_low,  'b', label='x2_u,l')
    plt.plot(k, SOC_up,   'b')
    plt.title('x2 SOC_k')
    # plt.ylim(0, 1) 
    # plt.xlim(0, 10000)
    plt.legend(); plt.grid(True)

    plt.tight_layout()                       # 子图自动排版

    # ---------- 图 2：电压 ----------
    plt.figure(figsize=(10, 6))
    plt.plot(k, Vol_data, 'k', label='Vol')
    plt.plot(k, y_est,    'r', label='y_')
    plt.plot(k, y_low,    'b', label='y_u,l')
    plt.plot(k, y_up,     'b')
    # plt.plot(k, data['y_true'], 'm', label='y')  # 如需真值可取消注释
    plt.legend()
    plt.title('y_k(U_k)')
    plt.xlim(0, 500)
    plt.ylim(-10, 20)
    plt.grid(True)

    plt.tight_layout()
    plt.show()                               # 阻塞式显示所有窗口

# ======== 4. 主函数 ========
def main():
    """
    解析命令行参数，读取数据并绘图。
    """
    # ---- 4.1 解析命令行 ----
    parser = argparse.ArgumentParser(
        description='读取 CSV 并绘制 EKF 结果')           # 描述信息
    parser.add_argument('csv_path',            # 位置参数
                        help='CSV 数据文件路径，例如 data/plot_1.csv')
    args = parser.parse_args()                 # 解析参数

    # ---- 4.2 读取数据 & 绘图 ----
    data = read_csv_data(args.csv_path)        # 读取 CSV
    plot_results(data)                         # 绘图

# ======== 5. 程序入口 ========
if __name__ == '__main__':
    main()





# import csv
# import matplotlib.pyplot as plt

# # 读取数据
# k = []
# Up_est = []; Up_low = []; Up_up = []; Up_true = []
# SOC_est = []; SOC_low = []; SOC_up = []; SOC_true = []
# Vol_data = []; y_est = []; y_low = []; y_up = []; y_true = []

# with open('../data/plot_1.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         k.append(float(row['k']))
        
#         Up_est.append(float(row['Up_est']))
#         Up_low.append(float(row['Up_low']))
#         Up_up.append(float(row['Up_up']))
#         Up_true.append(float(row['Up_true']))
        
#         SOC_est.append(float(row['SOC_est']))
#         SOC_low.append(float(row['SOC_low']))
#         SOC_up.append(float(row['SOC_up']))
#         SOC_true.append(float(row['SOC_true']))
        
#         Vol_data.append(float(row['Vol']))
#         y_est.append(float(row['y_est']))
#         y_low.append(float(row['y_low']))
#         y_up.append(float(row['y_up']))
#         y_true.append(float(row['y_true']))

# # 创建绘图
# plt.figure(figsize=(10, 8))

# # Up 估计
# plt.subplot(2, 1, 1)
# plt.plot(k, Up_true, 'k', label='x1')
# plt.plot(k, Up_est, 'r', label='x1_')
# plt.plot(k, Up_low, 'b', label='x1_u,l')
# plt.plot(k, Up_up, 'b')
# plt.title('x1 Up,k')
# plt.ylim(-0.04, 0.08)
# plt.xlim(0, 9200)
# plt.legend()
# plt.grid(True)

# # SOC 估计
# plt.subplot(2, 1, 2)
# plt.plot(k, SOC_true, 'k', label='x2')
# plt.plot(k, SOC_est, 'r', label='x2_')
# plt.plot(k, SOC_low, 'b', label='x2_u,l')
# plt.plot(k, SOC_up, 'b')
# plt.title('x2 SOC_k')
# plt.xlim(0, 9200)
# plt.legend()
# plt.grid(True)

# plt.tight_layout()

# # 第二张图：电压比较
# plt.figure(figsize=(10, 6))
# plt.plot(k, Vol_data, 'k', label='Vol')
# plt.plot(k, y_est, 'r', label='y_')
# plt.plot(k, y_low, 'b', label='y_u,l')
# plt.plot(k, y_up, 'b')
# # plt.plot(k, y_true, 'm', label='y')
# plt.legend()
# plt.title('y_k(U_k)')
# plt.xlim(0, 9200)
# plt.ylim(0, 8)    
# plt.grid(True)

# plt.tight_layout()
# plt.show()


