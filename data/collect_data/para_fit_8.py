import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
    params: 包含 [Rp, Cp, a0, a1, a2, a3, a4, a5, a6, a7, a8, R0] 的数组
    """
    Rp, Cp = params[0], params[1]
    a_coeffs = params[2:11] # a0 to a8
    R0 = params[11]
    
    y_pred = np.zeros(N)
    x_k = np.copy(x_initial) # 确保不修改原始初始状态
    
    for k in range(N):
        # 状态转移矩阵 A 和 B
        exp_term = np.exp(-deltaT_val / (Rp * Cp))
        A = np.array([[exp_term, 0], [0, 1]])
        B = np.array([[Rp * (1 - exp_term)], [-eta_val * deltaT_val / Qn_val]])
        
        # 状态更新
        x_k_next = np.dot(A, x_k) + B.flatten() * Cur[k]
        
        # 输出观测方程
        D = -R0
        
        # Calculate the polynomial part of the output equation
        poly_term = sum(a_coeffs[i] * (x_k_next[1]**i) for i in range(len(a_coeffs)))
        
        y_pred[k] = poly_term - x_k_next[0] + D * Cur[k]
        
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
initial_params = np.array([
    5.0,        # Rp
    500.0,      # Cp
    0.1,        # a0
    0.005,      # a1
    0.00005,    # a2
    -0.0000005, # a3
    0.0,        # a4
    0.0,        # a5
    0.0,        # a6
    0.0,        # a7
    0.0,        # a8
    0.01        # R0
])

# 定义参数的边界 (可选，但推荐，特别是对于 Rp, Cp 这种物理量应为正)
bounds = [
    (1e-3, 1.0),    # Rp 应该为正
    (1e-3, 10000.0),  # Cp 应该为正
    (-2000.0, 2000.0),    # a0
    (-2000.0, 2000.0),      # a1
    (-2000.0, 2000.0),      # a2
    (-2000.0, 2000.0),    # a3
    (-1000.0, 1000.0),  # a4
    (-1000.0, 1000.0),# a5
    (-100.0, 100.0),    # a6
    (-50.0, 50.0),    # a7
    (-50.0, 50.0),    # a8
    (1e-3, 1.0)       # R0 应该为正
]

# 运行优化
print("开始拟合参数...")
result = minimize(
    cost_function,
    initial_params,
    args=(Cur, Vol, x_initial, N, deltaT_val, eta_val, Qn_val),
    method='L-BFGS-B', # 推荐使用L-BFGS-B, TNC, 或 SLSQP 对于有边界的优化
    bounds=bounds,
    options={'disp': True, 'maxiter': 5000}
)

# 打印拟合结果
print("\n非线性优化算法拟合结果:")
if result.success:
    fitted_params = result.x
    print(f"成功拟合参数。最小化误差: {result.fun:.4e}")
    print(f"拟合的 Rp: {fitted_params[0]:.4f}")
    print(f"拟合的 Cp: {fitted_params[1]:.4f}")
    print(f"拟合的 a0: {fitted_params[2]:.4e}")
    print(f"拟合的 a1: {fitted_params[3]:.4e}")
    print(f"拟合的 a2: {fitted_params[4]:.4e}")
    print(f"拟合的 a3: {fitted_params[5]:.4e}")
    print(f"拟合的 a4: {fitted_params[6]:.4e}")
    print(f"拟合的 a5: {fitted_params[7]:.4e}")
    print(f"拟合的 a6: {fitted_params[8]:.4e}")
    print(f"拟合的 a7: {fitted_params[9]:.4e}")
    print(f"拟合的 a8: {fitted_params[10]:.4e}")
    print(f"拟合的 R0: {fitted_params[11]:.4f}")
    
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