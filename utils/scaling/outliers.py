import torch
import numpy as np
import sys
import argparse

def analyze_pt_file(file_path):
    # 1. 加载数据
    data = torch.load(file_path)
    
    # 确保是浮点型并展平
    if isinstance(data, dict):
        print("提示: 加载的是一个字典，尝试提取第一个张量...")
        data = next(iter(data.values()))
    
    x = data.detach().cpu().float().flatten()
    N = x.numel()
    
    # 2. 计算基本统计量
    mu = torch.mean(x).item()
    sigma = torch.std(x).item()
    s_max = torch.max(torch.abs(x)).item()
    
    # 3. 现场估算拉普拉斯参数 b
    # MLE 估计: b = mean(|x - median|) 或者是 mean(|x - mean|)
    # 通常用 mean(|x - mu|) 足够精确
    b_estimated = torch.mean(torch.abs(x - mu)).item()
    
    # 4. 计算倍数关系
    num_sigma = s_max / sigma
    num_b = s_max / b_estimated
    
    # 5. 理论预测 (如果是纯 i.i.d. Laplace 分布)
    # 对于 N 个样本，最大值的期望约为 mu + b * ln(N)
    theoretical_max_b = np.log(N)
    theoretical_max_sigma = theoretical_max_b / np.sqrt(2)

    print(f"========================================")
    print(f"文件分析报告: {file_path}")
    print(f"样本总数 N: {N}")
    print(f"----------------------------------------")
    print(f"实际观测值:")
    print(f"  均值 (mu):         {mu:.6f}")
    print(f"  标准差 (sigma):    {sigma:.6f}")
    print(f"  拉普拉斯 b:        {b_estimated:.6f}")
    print(f"  最大值 S_max:      {s_max:.6f}")
    print(f"----------------------------------------")
    print(f"离群强度分析:")
    print(f"  S_max 位于 {num_sigma:.2f} 个标准差 (sigma) 之外")
    print(f"  S_max 位于 {num_b:.2f} 个尺度单位 (b) 之外")
    print(f"----------------------------------------")
    print(f"理想 Laplace 分布对比 (同规模 N={N}):")
    print(f"  理论预期 S_max 应为: {theoretical_max_sigma:.2f} sigma")
    print(f"  理论预期 S_max 应为: {theoretical_max_b:.2f} b")
    print(f"========================================")

    # 逻辑判定
    if num_b > theoretical_max_b * 2:
        print("⚠️ 判定: 该张量含有显著离群值 (符合你的 12b 逻辑)。")
        print(f"建议: 采用 Half-scaling，将截断点设为 {s_max/2:.4f} (约为 {s_max/2/b_estimated:.2f}b)")
    else:
        print("✅ 判定: 该张量分布较为平滑 (符合你的 3.5b 逻辑)。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 PyTorch 张量文件的离群值")
    parser.add_argument("tensor_file", type=str, help="要分析的 .pt 文件路径")
    args = parser.parse_args()
    
    analyze_pt_file(args.tensor_file)
