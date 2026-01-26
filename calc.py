import numpy as np
from scipy.integrate import quad

def laplace_pdf_normalized(z):
    """
    实现了归一化后的拉普assed分布的概率密度函数 (PDF) for z >= 0。
    公式: p(z) = e^(-z)
    """
    return np.exp(-z)

def calculate_gamma_laplace(a_hat):

    # --- 1. 计算 MXFP4 舍入误差 E_r,mxfp4(â) ---
    # 根据公式，我们只需计算积分部分，公共因子 b² 会在求比值时消掉。
    
    # 定义MXFP4的基础网格点和分段边界点
    Q_plus_base = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    M_base = np.array([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])

    # 计算归一化后的缩放因子 S_hat 和对应的网格/边界点
    S_hat = a_hat / 6.0
    Q_hat = Q_plus_base * S_hat  # 归一化后的量化点 q_hat_i
    M_hat = M_base * S_hat      # 归一化后的积分边界 m_hat_i

    integral_sum_mxfp4 = 0.0
    # 循环遍历8个积分区间
    for i in range(8):
        m_hat_i = M_hat[i]
        m_hat_i_plus_1 = M_hat[i+1]
        
        # 对应的量化点 q_hat_i
        q_hat_i = Q_hat[i] 
        
        # 定义被积函数: (z - q_hat_i)^2 * e^(-z)
        integrand = lambda z: ((z - q_hat_i)**2) * laplace_pdf_normalized(z)
        
        # 使用 quad 工具进行积分
        segment_integral, _ = quad(integrand, m_hat_i, m_hat_i_plus_1)
        integral_sum_mxfp4 += segment_integral
        
    # E_r,mxfp4 的核心部分就是这个积分和
    # (公式中的 b² 在计算gamma时会被消掉)
    error_mxfp4 = integral_sum_mxfp4*2

    # --- 2. 计算 INT4 舍入误差 E_r,int4(â) ---
    # 使用拉普拉斯分布进行积分计算（与 MXFP4 方式一致）
    # INT4 有 16 个量化级别，在 [0, a_hat] 范围内有 8 个正量化点
    # 使用枚举值和中间值，并乘以 S_hat
    
    # 定义 INT4 的量化点（枚举值，相对于 6.0 归一化）
    # INT4 在 [0, 6] 范围内有 8 个量化点：0, 6/8, 12/8, 18/8, 24/8, 30/8, 36/8, 42/8
    S_hat = a_hat / 7.0
    Q_int4_base = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # 定义 INT4 的边界点（中间值，量化点之间的中点）
    # 边界点：0, (0+0.75)/2, (0.75+1.5)/2, ..., (5.25+6.0)/2, 6.0
    M_int4_base = np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7])
    
    # 使用相同的 S_hat 进行归一化
    Q_int4_hat = Q_int4_base * S_hat  # 归一化后的量化点
    M_int4_hat = M_int4_base * S_hat  # 归一化后的积分边界
    
    integral_sum_int4 = 0.0
    # 循环遍历 8 个量化区间
    for i in range(8):
        m_hat_i = M_int4_hat[i]
        m_hat_i_plus_1 = M_int4_hat[i+1]
        
        # 对应的量化点 q_hat_i
        q_hat_i = Q_int4_hat[i]
        
        # 定义被积函数: (z - q_hat_i)^2 * e^(-z)
        integrand = lambda z: ((z - q_hat_i)**2) * laplace_pdf_normalized(z)
        
        # 使用 quad 工具进行积分
        segment_integral, _ = quad(integrand, m_hat_i, m_hat_i_plus_1)
        integral_sum_int4 += segment_integral
    
    # E_r,int4 的核心部分（考虑对称性，乘以 2）
    error_int4_base = integral_sum_int4 * 2
    # 避免除以零
    error_int4_base = max(error_int4_base, 1e-12)

    # --- 3. 计算效率增益 gamma(â) ---
    # 使用基于拉普拉斯分布计算的 error_int4_base
    error_int4 = error_int4_base
    gamma = error_mxfp4 / error_int4

    return (gamma, error_mxfp4, error_int4)

# --- 主程序：枚举 â 值并进行积分计算 ---

print("### 基于拉普拉斯分布的 Gamma 计算结果")
print("| â (阈值) | γ (增益) ")
print("|:---|:---|")

# 遍历一系列 â 值
for a_hat_val in np.arange(0.0, 20.1, 0.5):

    gamma, err_mxfp4, err_int4 = calculate_gamma_laplace(a_hat_val)
    
    # 高亮关键点
    a_str =  f"{a_hat_val:.1f}"
    gamma_str = f"{gamma:.4f}"
    
    print(f"| {a_str.ljust(10)} | {gamma_str.ljust(10)} |")

