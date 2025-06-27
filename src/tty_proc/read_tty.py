import serial
import time

# --- 配置区 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
R_SAMPLE = 1.0  # 假设为1.0欧姆，请务必修改为你的实际采样电阻值！

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) # 使用一个短的超时
    print(f"成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")
except serial.SerialException as e:
    print(f"错误：无法打开串口 {SERIAL_PORT}。")
    print(e)
    exit()

k = 1

while True:
    try:
        latest_line_in_interval = None # 用于存储当前间隔内的最新数据行

        # 核心逻辑：快速读取缓冲区中的所有数据，只保留最后一个
        # 只要缓冲区(in_waiting)里有数据，就一直读
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line: # 确保不是空行
                latest_line_in_interval = line

        # 如果在上一个周期中接收到了数据，则处理最新的一条
        if latest_line_in_interval:
            data = latest_line_in_interval
            # print(f"接收到的最新原始数据: '{data}'") # 调试用，可以看看读到的是什么

            if data.startswith('<all>:'):
                parts = data.split(':')
                if len(parts) >= 2:
                    values_str = parts[1].split(',')
                    if len(values_str) >= 3:
                        U_battery = int(values_str[0]) / 1000.0
                        U_sample_resistor = int(values_str[1]) / 1000.0

                        if R_SAMPLE <= 0:
                            I_k = 0.0
                        else:
                            I_k = U_sample_resistor / R_SAMPLE

                        print(f"k={k}, Uk={U_battery:.3f}V, Ik={I_k:.3f}A")
                        k += 1

        # 主循环的节拍器，控制每秒处理一次
        time.sleep(1)

    except (ValueError, IndexError) as e:
        print(f"数据解析错误: '{latest_line_in_interval}' -> {e}。")
        # 错误发生后，继续等待下一个1秒周期
        time.sleep(1)
        continue
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
        break

if ser.is_open:
    ser.close()
    print("串口已关闭。")