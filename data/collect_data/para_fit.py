import numpy as np
from scipy.optimize import minimize

# Vol.txt 和 Cur.txt 就在当前目录下
Vol = np.loadtxt('Vol.txt')
Cur = np.loadtxt('Cur.txt')

# --- 模拟数据用于演示，实际请从文件读取 ---
N = len(Vol)
deltaT_val = 1
eta_val = 1
Qn_val = 1.5 * 3600

# 初始状态
x_initial = np.array([0.1, 1.0])
x_k = x_initial

def system_model(params, Cur, x_initial, N, deltaT_val, eta_val, Qn_val):
    """
    状态空间模型，根据给定的参数和输入计算预测的输出。
    params: 包含 [Rp, Cp, a, b, c, d, R0] 的数组
    """
    Rp, Cp, a, b, c, d, R0 = params
    
    y_pred = np.zeros(N)
    x_k = np.copy(x_initial) # 确保不修改原始初始状态
    
    for k in range(N):
        # 状态转移矩阵 A 和 B
        exp_term = np.exp(-deltaT_val / (Rp * Cp))
        A = np.array([[exp_term, 0], [0, 1]])
        B = np.array([[Rp * (1 - exp_term)], [-eta_val * deltaT_val / Qn_val]])
        
        # 状态更新
        x_k_next = np.dot(A, x_k) + B.flatten() * Cur[k] # B是列向量，需要flatten
        
        # 输出观测方程
        D = -R0
        y_pred[k] = a + b * x_k_next[1] + c * (x_k_next[1]**2) + \
                    d * (x_k_next[1]**3) - x_k_next[0] + D * Cur[k]
        
        x_k = x_k_next
        
    return y_pred

def cost_function(params, Cur, Vol, x_initial, N, deltaT_val, eta_val, Qn_val):
    """
    代价函数：计算预测输出与实际输出之间的均方误差。
    """
    y_pred = system_model(params, Cur, x_initial, N, deltaT_val, eta_val, Qn_val)
    return np.sum((y_pred - Vol)**2)

# 初始参数猜测 (请根据实际情况调整这些值)
# 顺序对应 system_model 函数中的 params: [Rp, Cp, a, b, c, d, R0]
initial_params = np.array([1, 3000.0, 0.1, 0.005, 0.00005, -0.0000005, 1])

# 定义参数的边界 (可选，但推荐，特别是对于 Rp, Cp 这种物理量应为正)
bounds = [
    (1e-3, 100.0),   # Rp 应该为正
    (1e-3, 10000.0), # Cp 应该为正
    (-10.0, 10.0),   # a
    (-1.0, 1.0),    # b
    (-0.1, 0.1),    # c
    (-0.01, 0.01),  # d
    (1e-3, 1.0)      # R0 应该为正
]

# 运行优化
print("开始拟合参数...")
result = minimize(
    cost_function,
    initial_params,
    args=(Cur, Vol, x_initial, N, deltaT_val, eta_val, Qn_val),
    method='L-BFGS-B', # 推荐使用L-BFGS-B, TNC, 或 SLSQP 对于有边界的优化
    bounds=bounds,
    options={'disp': True, 'maxiter': 1000}
)

# 打印拟合结果
print("\n非线性优化算法拟合结果:")
if result.success:
    fitted_params = result.x
    print(f"成功拟合参数。最小化误差: {result.fun:.4e}")
    print(f"拟合的 Rp: {fitted_params[0]:.4f}")
    print(f"拟合的 Cp: {fitted_params[1]:.4f}")
    print(f"拟合的 a: {fitted_params[2]:.4f}")
    print(f"拟合的 b: {fitted_params[3]:.4f}")
    print(f"拟合的 c: {fitted_params[4]:.4f}")
    print(f"拟合的 d: {fitted_params[5]:.4f}")
    print(f"拟合的 R0: {fitted_params[6]:.4f}")
    
    # 验证拟合效果 (可选)
    import matplotlib.pyplot as plt
    y_fitted = system_model(fitted_params, Cur, x_initial, N, deltaT_val, eta_val, Qn_val)
    plt.figure(figsize=(10, 6))
    plt.plot(Vol, label='Vol_k')
    plt.plot(y_fitted, label='y_k')
    plt.title('contrast of y_k and Vol_k')
    plt.xlabel('time k')
    plt.ylabel('y_k')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("参数拟合失败。原因:", result.message)