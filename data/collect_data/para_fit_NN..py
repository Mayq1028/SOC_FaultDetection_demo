import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 假设你已经从 Vol.txt 和 Cur.txt 加载了数据
# 例如:
# true_yk = np.loadtxt('Vol.txt')
# uk = np.loadtxt('Cur.txt')
# N = len(true_yk)

# 临时模拟数据用于演示
N = 1000
uk = np.sin(np.linspace(0, 100, N)) + np.random.randn(N) * 0.1
# 假设真实参数
true_Rp = 10.0
true_Cp = 50.0
true_R0 = 0.05
true_a0 = 0.1
true_a1 = -0.02
true_a2 = 0.005
true_a3 = -0.0001
true_a4 = 0.00002
true_a5 = -0.000003
true_a6 = 0.0000004
true_a7 = -0.00000005
true_a8 = 0.000000006

# 模拟生成真实 yk (用于训练的“ground truth”)
deltaT = 1.0
eta = 1.0
Qn = 1.5 * 3600
xk = np.zeros((N, 2))
xk[0, 1] = 1.0 # x2,0 = 1
true_yk = np.zeros(N)

for k in range(N - 1):
    # 状态更新
    A_true = np.array([[np.exp(-deltaT / (true_Rp * true_Cp)), 0],
                       [0, 1]])
    B_true = np.array([[true_Rp * (1 - np.exp(-deltaT / (true_Rp * true_Cp)))],
                       [-eta * deltaT / Qn]])
    
    xk[k+1, :] = np.dot(A_true, xk[k, :]) + B_true.flatten() * uk[k]

    # 观测方程
    x2k_pow_terms = sum(true_a_i * (xk[k, 1] ** i) for i, true_a_i in enumerate([true_a0, true_a1, true_a2, true_a3, true_a4, true_a5, true_a6, true_a7, true_a8]))
    true_yk[k] = x2k_pow_terms - xk[k, 0] + (-true_R0) * uk[k]
true_yk[-1] = true_yk[-2] # 最后一帧观测方程可能需要特殊处理，或者只考虑N-1个点

# 转换为 PyTorch 张量
uk_tensor = torch.tensor(uk, dtype=torch.float32).unsqueeze(1) # (N, 1)
true_yk_tensor = torch.tensor(true_yk, dtype=torch.float32).unsqueeze(1) # (N, 1)

class SystemIDModel(nn.Module):
    def __init__(self):
        super(SystemIDModel, self).__init__()
        # 将未知参数定义为可训练的 nn.Parameter
        self.Rp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) # 初始猜测值
        self.Cp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.R0 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a3 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a4 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a5 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a6 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a7 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.a8 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 常量
        self.deltaT = torch.tensor(1.0, dtype=torch.float32)
        self.eta = torch.tensor(1.0, dtype=torch.float32)
        self.Qn = torch.tensor(1.5 * 3600, dtype=torch.float32)

    def forward(self, uk_sequence, initial_x2_0=1.0):
        N = uk_sequence.shape[0]
        predicted_yk = torch.zeros(N, 1, dtype=torch.float32)
        # xk 是 (N, 2) 维度的张量，存储每个时间步的状态
        xk = torch.zeros(N, 2, dtype=torch.float32)
        xk[0, 1] = initial_x2_0 # x2,0 = 1

        for k in range(N):
            # 计算当前时刻的观测值 yk
            x1k = xk[k, 0]
            x2k = xk[k, 1]
            
            # 观测方程
            # 使用torch.pow确保梯度可以回传
            x2k_pow_terms = self.a0 + \
                            self.a1 * x2k + \
                            self.a2 * torch.pow(x2k, 2) + \
                            self.a3 * torch.pow(x2k, 3) + \
                            self.a4 * torch.pow(x2k, 4) + \
                            self.a5 * torch.pow(x2k, 5) + \
                            self.a6 * torch.pow(x2k, 6) + \
                            self.a7 * torch.pow(x2k, 7) + \
                            self.a8 * torch.pow(x2k, 8)
            
            predicted_yk[k] = x2k_pow_terms - x1k + (-self.R0) * uk_sequence[k]

            # 状态更新 (如果不是最后一个时间步)
            if k < N - 1:
                # A 矩阵
                A_elem_00 = torch.exp(-self.deltaT / (self.Rp * self.Cp))
                A_matrix = torch.tensor([[A_elem_00, 0.0],
                                         [0.0, 1.0]], dtype=torch.float32)

                # B 矩阵
                B_elem_0 = self.Rp * (1.0 - torch.exp(-self.deltaT / (self.Rp * self.Cp)))
                B_elem_1 = -self.eta * self.deltaT / self.Qn
                B_matrix = torch.tensor([[B_elem_0], [B_elem_1]], dtype=torch.float32)

                # 确保矩阵乘法维度正确，并使用 @ 运算符进行矩阵乘法
                xk_current_vec = xk[k, :].unsqueeze(1) # (2, 1)
                uk_current_val = uk_sequence[k] # (1)

                # x_{k+1} = A x_k + B u_k
                xk_next_vec = torch.matmul(A_matrix, xk_current_vec) + B_matrix * uk_current_val
                xk[k+1, :] = xk_next_vec.squeeze(1) # 赋值给下一个时间步的状态

        return predicted_yk

# 实例化模型、定义损失函数和优化器
model = SystemIDModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # 学习率可能需要调整

# 训练循环
num_epochs = 5000 # 迭代次数可能需要很多
for epoch in range(num_epochs):
    model.train() # 设置为训练模式
    optimizer.zero_grad() # 梯度清零

    predicted_yk = model(uk_tensor, initial_x2_0=1.0)
    loss = criterion(predicted_yk, true_yk_tensor)

    loss.backward() # 反向传播计算梯度
    optimizer.step() # 更新参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 训练结束后，打印辨识出的参数
print("\n--- 辨识出的参数 ---")
print(f"Rp: {model.Rp.item():.4f} (真实值: {true_Rp})")
print(f"Cp: {model.Cp.item():.4f} (真实值: {true_Cp})")
print(f"R0: {model.R0.item():.4f} (真实值: {true_R0})")
print(f"a0: {model.a0.item():.4f} (真实值: {true_a0})")
print(f"a1: {model.a1.item():.4f} (真实值: {true_a1})")
print(f"a2: {model.a2.item():.4f} (真实值: {true_a2})")
print(f"a3: {model.a3.item():.4f} (真实值: {true_a3})")
print(f"a4: {model.a4.item():.4f} (真实值: {true_a4})")
print(f"a5: {model.a5.item():.4f} (真实值: {true_a5})")
print(f"a6: {model.a6.item():.4f} (真实值: {true_a6})")
print(f"a7: {model.a7.item():.4f} (真实值: {true_a7})")
print(f"a8: {model.a8.item():.4f} (真实值: {true_a8})")