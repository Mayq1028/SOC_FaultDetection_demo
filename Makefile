# 编译器和选项
CC = g++
CFLAGS = -Wall -g -I/usr/include/eigen3  # 在此处将 Eigen 头文件路径加入

# 目录定义
SRC_DIR = ./src
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
# ONLINE_SRC = $(SRC_DIR)/online_pkg/deltaT_adjust.cpp
# ONLINE_OBJ = $(OBJ_DIR)/deltaT_adjust.o
# ONLINE_BIN = $(BIN_DIR)/deltaT_adjust

# 离线程序
OFFLINE_SRC = $(SRC_DIR)/offline_pkg/offline_origin.cpp
OFFLINE_OBJ = $(OBJ_DIR)/offline_origin.o
OFFLINE_BIN = $(BIN_DIR)/offline_origin

# 默认目标：编译并生成所有可执行文件
all: $(OFFLINE_BIN) $(ONLINE_BIN)

# 编译离线程序
$(OFFLINE_BIN): $(OFFLINE_OBJ)
	$(CC) $(OFFLINE_OBJ) -o $(OFFLINE_BIN)

$(OFFLINE_OBJ): $(OFFLINE_SRC)
	$(CC) $(CFLAGS) -c $(OFFLINE_SRC) -o $(OFFLINE_OBJ)

# 编译实时程序
$(ONLINE_BIN): $(ONLINE_OBJ)
	$(CC) $(ONLINE_OBJ) -o $(ONLINE_BIN)

$(ONLINE_OBJ): $(ONLINE_SRC)
	$(CC) $(CFLAGS) -c $(ONLINE_SRC) -o $(ONLINE_OBJ)

# 清理目标：删除中间文件和可执行文件
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/*

# 运行离线程序
run_offline: $(OFFLINE_BIN)
	./$(OFFLINE_BIN) $(DATA_DIR)/offline_data/input_data.txt

# 运行实时程序
run_online: $(ONLINE_BIN)
	./$(ONLINE_BIN) 

# 运行串口数据采集脚本
run_tty_proc:
	python3 $(TTY_PROC_SRC)
