import serial
import time

# 打开串口
ser = serial.Serial('/dev/ttyUSB0', 115200)  # 根据实际情况修改串口号和波特率

k = 1  # 初始化k为1

while True:
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').strip()  # 读取一行数据并去掉换行符

        # 全通道： <all>:3998,4003,0
        if data.startswith('<all>:'):
            # 提取数据部分，并分割为不同的数值
            values = data.split(':')[1].split(',')
            U = int(values[2]) / 1000.0  # 电池端电压（V）通道三
            U_p = int(values[1]) / 1000.0  # 极性电压（V）通道二
            Rp = U
            # 使用欧姆定律计算电流
            I_k = U_p / Rp  # 电流（单位：安培）

            print(f"U({k}): {U:.3f}V, I({k}): {I_k:.3f}A")

            # 增加k值
            k += 1

    # 每秒打印一次数据
    time.sleep(1)
