import serial
import time
import os

# --- 配置区 (Configuration Area) ---

# 根据你的设备修改串口号和波特率
# Modify the serial port and baud rate according to your device
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

# !!! 物理定律警告 (Physics Law Warning) !!!
# 下面的电阻值 Rp 是电流计算的关键。在你的原始代码中，Rp被错误地设置为了一个电压值 (Rp = U_k)。
# 要正确计算电流 (I = V/R)，你需要在这里填入电路中用于电流测量的【采样电阻】的【实际阻值】（单位：欧姆）。
# 如果你不确定，请查阅电路图。这里暂时设置为1.0，请务必修改为你的真实值。
# The resistance value Rp below is crucial for current calculation. In your original code,
# Rp was incorrectly set to a voltage value (Rp = U_k).
# To calculate current correctly (I = V/R), you must enter the ACTUAL RESISTANCE of the
# shunt resistor used for current sensing in your circuit (unit: Ohms).
# If you are unsure, please check your circuit diagram. It's set to 1.0 temporarily, please modify it.
R_SAMPLE = 1.0

# 数据保存路径
# Data save path
VOLTAGE_FILE_PATH = "../../data/collect_data/Vol.txt"
CURRENT_FILE_PATH = "../../data/collect_data/Cur.txt"

# 停止采集的电压阈值
# Voltage threshold to stop collection
STOP_VOLTAGE_THRESHOLD = 2.40

# --- 主程序 (Main Program) ---

def main():
    """
    主函数，包含程序所有逻辑
    Main function containing all program logic
    """
    # --- 1. 检查并创建文件夹 (Check and create directories) ---
    try:
        # os.path.dirname 获取文件所在的目录路径
        # os.path.dirname gets the directory path of the file
        os.makedirs(os.path.dirname(VOLTAGE_FILE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CURRENT_FILE_PATH), exist_ok=True)
    except OSError as e:
        print(f"创建目录时出错 (Error creating directory): {e}")
        return # 如果无法创建目录，则退出程序

    # --- 2. 初始化串口 (Initialize Serial Port) ---
    ser = None # 先声明变量
    try:
        # 使用 timeout=0.1s，避免 readline 长时间阻塞
        # Use a short timeout to prevent readline from blocking for too long
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"成功打开串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")
    except serial.SerialException as e:
        print(f"错误：无法打开串口 {SERIAL_PORT}。请检查设备连接或权限。")
        print(e)
        return # 无法打开串口则退出

    k = 1  # 初始化数据点计数器

    # --- 3. 打开文件并开始主循环 (Open files and start main loop) ---
    try:
        # 使用 'with' 语句能确保程序退出时文件被正确关闭
        # Using 'with' statement ensures files are properly closed on exit
        with open(VOLTAGE_FILE_PATH, "w") as vol_file, open(CURRENT_FILE_PATH, "w") as cur_file:
            print("文件已打开，开始采集数据...")
            print(f"当电压低于 {STOP_VOLTAGE_THRESHOLD:.2f}V 时，采集将自动停止。")
            print("按 Ctrl+C 可以手动停止程序。")

            while True:
                latest_line_in_interval = None

                # 核心逻辑：快速清空缓冲区，只保留最新一行数据
                # Core Logic: Quickly drain the buffer, keeping only the latest line
                while ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:  # 确保不是空行
                        latest_line_in_interval = line

                # 如果在刚才的周期内读到了数据，则处理它
                # If data was read in the last cycle, process it
                if latest_line_in_interval:
                    data = latest_line_in_interval
                    
                    if data.startswith('<all>:'):
                        # a. 解析数据 (Parse data)
                        parts = data.split(':')
                        if len(parts) < 2: continue
                        
                        values_str = parts[1].split(',')
                        if len(values_str) < 2: continue
                        
                        # b. 提取数值并计算 (Extract values and calculate)
                        U_k = int(values_str[0]) / 1000.0  # 假设通道一为电压
                        U_p = int(values_str[1]) / 1000.0  # 假设通道二为用于计算电流的电压

                        # !!! 注意：正确的电流计算 !!!
                        # 使用预设的采样电阻值，而不是用一个变量电压值
                        # Note: Correct current calculation!
                        # Use the predefined sample resistance, not a variable voltage.
                        I_k = U_p / R_SAMPLE
                        
                        # c. 打印到控制台 (Print to console)
                        print(f"第 {k} 秒最新数据 -> U: {U_k:.3f}V, I: {I_k:.3f}A")

                        # d. 将数据写入文件 (Write data to files)
                        vol_file.write(f"{U_k:.3f}\n")
                        cur_file.write(f"{I_k:.3f}\n")
                        
                        # 立即将数据从内存缓冲区刷入磁盘，确保实时写入
                        # Flush data from memory buffer to disk immediately for real-time writing
                        vol_file.flush()
                        cur_file.flush()

                        # e. 检查停止条件 (Check stop condition)
                        if U_k < STOP_VOLTAGE_THRESHOLD:
                            print(f"\n电压 {U_k:.3f}V 低于阈值 {STOP_VOLTAGE_THRESHOLD:.2f}V，停止采集。")
                            break # 退出 while 循环

                        k += 1

                # 采样周期的节拍器，固定为1秒
                # The beat of the sampling cycle, fixed at 1 second
                time.sleep(1)

    except (ValueError, IndexError) as e:
        # 捕获数据格式错误，例如 "a,b,c" 解析失败
        print(f"\n数据解析错误，已跳过此行: '{latest_line_in_interval}' -> {e}")
    except KeyboardInterrupt:
        # 允许用户通过 Ctrl+C 来优雅地停止脚本
        print("\n程序被用户中断。")
    except Exception as e:
        # 捕获其他所有未知异常
        print(f"\n发生未知错误: {e}")
    finally:
        # 无论程序如何退出，都确保关闭串口
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭。")
        print("程序结束。")


if __name__ == '__main__':
    main()
