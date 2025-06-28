# 编译器和选项
CC = g++
CFLAGS = -Wall -g -I/usr/include/eigen3  # 在此处将 Eigen 头文件路径加入

# 目录定义
SRC_DIR = ./src
TTY_PROC_SRC = $(SRC_DIR)/tty_proc
OBJ_DIR = ./obj
BIN_DIR = ./bin
DATA_DIR = ./data
OUTPUT_DIR = $(DATA_DIR)/output

# 离线程序
OFFLINE_SRC = $(SRC_DIR)/offline_pkg/offline.cpp
OFFLINE_OBJ = $(OBJ_DIR)/offline.o
OFFLINE_BIN = $(BIN_DIR)/offline

# 实时程序
ONLINE_SRC = $(SRC_DIR)/online_pkg/online.cpp
ONLINE_OBJ = $(OBJ_DIR)/online.o
ONLINE_BIN = $(BIN_DIR)/online

# # 实时程序_调整deltaT
# ONLINE_SRC = $(SRC_DIR)/online_pkg/on_test.cpp
# ONLINE_OBJ = $(OBJ_DIR)/on_test.o
# ONLINE_BIN = $(BIN_DIR)/on_test

# # 离线程序
# OFFLINE_SRC = $(SRC_DIR)/offline_pkg/off_test.cpp
# OFFLINE_OBJ = $(OBJ_DIR)/off_test.o
# OFFLINE_BIN = $(BIN_DIR)/off_test

# 默认目标：编译并生成所有可执行文件
all: dirs $(OFFLINE_BIN) $(ONLINE_BIN)

dirs:
	@echo "Creating directories: $(BIN_DIR) and $(OBJ_DIR)..."
	mkdir -p $(BIN_DIR) $(OBJ_DIR)

# 编译离线程序
$(OFFLINE_BIN): $(OFFLINE_OBJ)
	@echo "Linking $(OFFLINE_BIN)..."
	$(CC) $(OFFLINE_OBJ) -o $(OFFLINE_BIN)

$(OFFLINE_OBJ): $(OFFLINE_SRC)
	@echo "Compiling $(OFFLINE_SRC) -> $(OFFLINE_OBJ)..."
	$(CC) $(CFLAGS) -c $(OFFLINE_SRC) -o $(OFFLINE_OBJ)

# 编译实时程序
$(ONLINE_BIN): $(ONLINE_OBJ)
	@echo "Linking $(ONLINE_BIN)..."
	$(CC) $(ONLINE_OBJ) -o $(ONLINE_BIN)

$(ONLINE_OBJ): $(ONLINE_SRC)
	@echo "Compiling $(ONLINE_SRC) -> $(ONLINE_OBJ)..."
	$(CC) $(CFLAGS) -c $(ONLINE_SRC) -o $(ONLINE_OBJ)

# 清理目标：删除中间文件和可执行文件
clean:
	@echo "Cleaning up $(OBJ_DIR) and $(BIN_DIR)..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# 运行离线程序
offline: $(OFFLINE_BIN)
	@echo "Running $(OFFLINE_BIN)..."
	./$(OFFLINE_BIN) 

# 运行实时程序
online: $(ONLINE_BIN)
	@echo "Running $(ONLINE_BIN)..."
	./$(ONLINE_BIN) 

# 运行串口数据采集脚本
tty_save:
	@echo "Running tty data save script..."
	python3 $(TTY_PROC_SRC)/save_tty_data.py

tty_read:
	@echo "Running tty read script..."
	python3 $(TTY_PROC_SRC)/read_tty.py

plot_offline:
	@echo "Plotting offline data..."
	python3 $(OUTPUT_DIR)/plot_output.py $(OUTPUT_DIR)/offline.csv

plot_online:
	@echo "Plotting online data..."
	python3 $(OUTPUT_DIR)/plot_output.py $(OUTPUT_DIR)/online.csv