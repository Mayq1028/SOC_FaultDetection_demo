# 打开 Vol.txt 和 Cur.txt 文件，读取数据
with open('Vol.txt', 'r') as vol_file, open('Cur_raw.txt', 'r') as cur_file:
    vol_lines = vol_file.readlines()
    cur_lines = cur_file.readlines()

# 确保两个文件有相同的行数
if len(vol_lines) != len(cur_lines):
    raise ValueError("Vol.txt 和 Cur_raw.txt 的行数不一致！")

# 计算比值并写入新文件
with open('Cur.txt', 'w') as ratio_file:
    for vol_line, cur_line in zip(vol_lines, cur_lines):
        # 去除行尾的换行符并转换为浮点数
        vol = float(vol_line.strip())
        cur = float(cur_line.strip())

        # 计算比值
        ratio = vol / cur

        # 写入到新文件，保留三位小数
        ratio_file.write(f"{ratio:.3f}\n")

print("比值计算完毕，结果已保存至 Cur.txt")
