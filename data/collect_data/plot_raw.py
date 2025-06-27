import matplotlib.pyplot as plt

# 读取数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

# 读取电压和电流数据
# voltage_data = read_data("./collection_2/Vol.txt")
# current_data = read_data("./collection_2/Cur.txt")

voltage_data = read_data("Vol.txt")
current_data = read_data("Cur.txt")

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制电压图
# plt.subplot(2, 1, 1)
plt.plot(voltage_data, color='blue', label="Voltage (V)")
plt.title("Voltage vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()

# 绘制电流图
# plt.subplot(2, 1, 2)
# plt.plot(current_data, color='red', label="Current (A)")
# plt.title("Current vs. Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Current (A)")
# plt.grid(True)
# plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
