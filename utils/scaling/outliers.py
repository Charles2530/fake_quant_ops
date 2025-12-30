import torch
import numpy as np
import argparse
from pathlib import Path

def compute_smax_gamma(file_path):
    """
    计算单个 tensor 文件的 Smax 在几个 gamma (标准差) 之外
    返回: num_sigma (Smax / sigma)
    """
    # 1. 加载数据
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"警告: 无法加载文件 {file_path}: {e}")
        return None
    
    # 确保是浮点型并展平
    if isinstance(data, dict):
        data = next(iter(data.values()))
    
    x = data.detach().cpu().float().flatten()
    
    # 2. 计算基本统计量
    sigma = torch.std(x).item()
    s_max = torch.max(torch.abs(x)).item()
    
    # 3. 计算 Smax 在几个 gamma (标准差) 之外
    if sigma == 0:
        print(f"警告: 文件 {file_path} 的标准差为0，跳过")
        return None
    
    num_sigma = s_max / sigma
    return num_sigma


def analyze_folder(folder_path):
    """
    统计文件夹下所有 tensor 文件的 Smax/gamma 的最小值和平均值
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"错误: {folder_path} 不是一个有效的文件夹路径")
        return
    
    # 查找所有 .pt 文件
    pt_files = list(folder.glob("*.pt"))
    if len(pt_files) == 0:
        print(f"警告: 在 {folder_path} 中没有找到 .pt 文件")
        return
    
    print(f"找到 {len(pt_files)} 个 .pt 文件")
    print(f"开始分析...\n")
    
    # 计算所有文件的 num_sigma
    num_sigma_list = []
    for pt_file in pt_files:
        num_sigma = compute_smax_gamma(pt_file)
        if num_sigma is not None:
            num_sigma_list.append(num_sigma)
    
    if len(num_sigma_list) == 0:
        print("错误: 没有成功处理任何文件")
        return
    
    # 计算统计量
    min_num_sigma = min(num_sigma_list)
    avg_num_sigma = np.mean(num_sigma_list)
    
    # 输出结果
    print(f"========================================")
    print(f"文件夹统计报告: {folder_path}")
    print(f"成功处理的文件数: {len(num_sigma_list)}")
    print(f"----------------------------------------")
    print(f"S_max / gamma (标准差倍数) 统计:")
    print(f"  最小值: {min_num_sigma:.4f}")
    print(f"  平均值: {avg_num_sigma:.4f}")
    print(f"========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计文件夹下所有 tensor 文件的 Smax/gamma 统计信息")
    parser.add_argument("path", type=str, help="要分析的文件夹路径或单个 .pt 文件路径")
    args = parser.parse_args()
    
    path = Path(args.path)
    if path.is_file() and path.suffix == ".pt":
        # 如果是单个文件，计算并输出
        num_sigma = compute_smax_gamma(path)
        if num_sigma is not None:
            print(f"S_max / gamma = {num_sigma:.4f}")
    elif path.is_dir():
        # 如果是文件夹，统计所有文件
        analyze_folder(path)
    else:
        print(f"错误: {args.path} 不是一个有效的文件夹或 .pt 文件路径")
