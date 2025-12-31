import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import tqdm for progress bar, fallback to simple print if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple tqdm replacement
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total
            self.desc = desc or ""
            self.unit = unit or ""
            self.n = 0
            self.postfix = {}
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit}")
            else:
                print(f"{self.desc}: {self.n} {self.unit}")
        def set_postfix(self, **kwargs):
            self.postfix = kwargs
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

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


def analyze_folder(folder_path, output_dir=None, num_workers=32):
    """
    统计文件夹下所有 tensor 文件的 Smax/gamma 的最小值和平均值，并绘制分布图
    使用多线程加速处理
    
    Args:
        folder_path: 文件夹路径
        output_dir: 输出目录
        num_workers: 线程数，默认32。推荐：0.25-0.5x CPU核心数用于CPU密集型任务
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
    print(f"使用 {num_workers} 个线程进行处理...\n")
    
    # 使用多线程计算所有文件的 num_sigma
    num_sigma_list = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(compute_smax_gamma, pt_file): pt_file
            for pt_file in pt_files
        }
        
        # 处理完成的任务，显示进度条
        with tqdm(total=len(pt_files), desc="Processing", unit="file") as pbar:
            for future in as_completed(future_to_file):
                pt_file = future_to_file[future]
                try:
                    num_sigma = future.result()
                    if num_sigma is not None:
                        num_sigma_list.append(num_sigma / 2)
                    pbar.set_postfix({'success': len(num_sigma_list), 
                                    'file': pt_file.name[:30]})
                except Exception as e:
                    print(f"  ⚠️  处理文件 {pt_file.name} 时出错: {e}")
                
                pbar.update(1)
    
    if len(num_sigma_list) == 0:
        print("错误: 没有成功处理任何文件")
        return
    
    # 转换为 numpy 数组便于计算
    num_sigma_array = np.array(num_sigma_list)
    
    # 计算统计量
    min_num_sigma = np.min(num_sigma_array)
    max_num_sigma = np.max(num_sigma_array)
    avg_num_sigma = np.mean(num_sigma_array)
    median_num_sigma = np.median(num_sigma_array)
    std_num_sigma = np.std(num_sigma_array)
    p25_num_sigma = np.percentile(num_sigma_array, 25)
    p75_num_sigma = np.percentile(num_sigma_array, 75)
    p95_num_sigma = np.percentile(num_sigma_array, 95)
    p99_num_sigma = np.percentile(num_sigma_array, 99)
    
    # 输出结果
    print(f"========================================")
    print(f"文件夹统计报告: {folder_path}")
    print(f"成功处理的文件数: {len(num_sigma_list)}")
    print(f"----------------------------------------")
    print(f"S_max / sigma (标准差倍数) 统计:")
    print(f"  最小值: {min_num_sigma:.4f}")
    print(f"  最大值: {max_num_sigma:.4f}")
    print(f"  平均值: {avg_num_sigma:.4f}")
    print(f"  中位数: {median_num_sigma:.4f}")
    print(f"  标准差: {std_num_sigma:.4f}")
    print(f"  25%分位数: {p25_num_sigma:.4f}")
    print(f"  75%分位数: {p75_num_sigma:.4f}")
    print(f"  95%分位数: {p95_num_sigma:.4f}")
    print(f"  99%分位数: {p99_num_sigma:.4f}")
    print(f"========================================")
    
    # 绘制分布图（柱状图，按区间分段统计）
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    # 定义更细的区间（每隔1个单位）
    max_val = np.max(num_sigma_array)
    # 创建每隔1个单位的区间
    if max_val <= 20:
        # 如果最大值不超过20，创建到最大值的区间
        bin_max = int(np.ceil(max_val))
        bins = list(range(0, bin_max + 1))
        bin_labels = [f'{i}-{i+1}' for i in range(bin_max)]
    else:
        # 如果最大值超过20，创建到20的区间，然后20+
        bins = list(range(0, 21)) + [np.inf]
        bin_labels = [f'{i}-{i+1}' for i in range(20)] + ['20+']
    
    # 按区间统计频数
    counts, bin_edges = np.histogram(num_sigma_array, bins=bins)
    
    # 使用渐变色（viridis colormap）
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bin_labels)))
    
    # Paper-ready size (调整宽度以适应更多柱子)
    num_bins = len(bin_labels)
    fig_width = max(5.5, num_bins * 0.25)  # 根据柱子数量调整宽度
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    
    # 绘制柱状图（美化样式，更细的柱子）
    bars = ax.bar(range(len(bin_labels)), counts, alpha=0.85, color=colors, 
                  edgecolor='white', linewidth=1.0, width=0.6)
    
    # 在柱状图上添加数值标签（美化样式）
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            height = bar.get_height()
            # 只显示数值，简洁明了
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2C3E50')
    
    # 添加统计线（需要转换为柱状图的x坐标）
    # 找到平均值、中位数、95%分位数所在的区间
    def find_bin_index(value):
        for i in range(len(bins)-1):
            if i == len(bins) - 2:  # 最后一个区间（20+）
                if value >= bins[i]:
                    return i
            else:
                if bins[i] <= value < bins[i+1]:
                    return i
        return len(bins) - 2  # 默认返回最后一个区间
    
    avg_bin_idx = find_bin_index(avg_num_sigma)
    median_bin_idx = find_bin_index(median_num_sigma)
    p95_bin_idx = find_bin_index(p95_num_sigma)
    
    # 添加统计线（美化样式，使用更细的线条）
    ax.axvline(avg_bin_idx + 0.5, color='#E74C3C', linestyle='--', linewidth=1.5, 
               alpha=0.8, label=f'Mean: {avg_num_sigma:.2f}')
    ax.axvline(median_bin_idx + 0.5, color='#27AE60', linestyle='--', linewidth=1.5, 
               alpha=0.8, label=f'Median: {median_num_sigma:.2f}')
    # ax.axvline(p95_bin_idx + 0.5, color='#F39C12', linestyle='--', linewidth=1.2, 
            #    alpha=0.7, label=f'95th: {p95_num_sigma:.2f}')
    
    # 设置x轴标签（美化样式）
    # 如果柱子太多，可以只显示部分标签或旋转
    ax.set_xticks(range(len(bin_labels)))
    if len(bin_labels) > 15:
        # 柱子太多时，只显示部分标签（每隔一个显示）
        tick_positions = range(0, len(bin_labels), 2)
        tick_labels = [bin_labels[i] if i < len(bin_labels) else '' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, fontweight='bold', rotation=45, ha='right')
    else:
        ax.set_xticklabels(bin_labels, fontsize=7.5, fontweight='bold', rotation=45, ha='right')
    
    # 设置标签和标题（美化样式，匹配论文格式）
    ax.set_xlabel('S_max / σ Range', fontsize=9, fontweight='normal', color='#000000')
    ax.set_ylabel('Frequency', fontsize=9, fontweight='normal', color='#000000')
    
    # Make y-axis tick labels bold for visibility
    ax.tick_params(axis='y', labelsize=8, colors='#333333', which='major')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.set_title(f'S_max / σ Distribution\n{folder.name} ({len(num_sigma_list)} files)', 
                 fontsize=10, fontweight='bold', pad=12, color='#000000')
    
    # 添加网格（美化样式，更轻的网格）
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, color='#CCCCCC', axis='y')
    ax.set_axisbelow(True)
    
    # 添加图例（美化样式）
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor='#E0E0E0', 
              facecolor='white', frameon=True)
    
    # 设置背景色
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # 边框样式（美化样式，更细的边框）
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1.0)
    
    # 使用 tight_layout 并调整边距（为旋转的标签留出更多底部空间）
    if len(bin_labels) > 15:
        plt.tight_layout(pad=1.2, rect=[0, 0.1, 1, 1])  # 底部留出更多空间
    else:
        plt.tight_layout(pad=1.2)
    
    # 保存图片（高分辨率，适合论文）
    if output_dir is None:
        output_dir = Path("./draw/outliers")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / f'sigma_distribution_{folder.name}.pdf'
    plt.savefig(plot_path, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05,
                metadata={'Creator': 'Outliers Analyzer', 
                         'Title': f'S_max / σ Distribution - {folder.name}'})
    plt.close()
    
    print(f"\n✅ 分布图已保存到: {plot_path}")
    print(f"\n区间统计结果:")
    for label, count in zip(bin_labels, counts):
        percentage = (count / len(num_sigma_list)) * 100
        print(f"  {label:12s}: {count:5d} 个文件 ({percentage:5.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计文件夹下所有 tensor 文件的 Smax/gamma 统计信息并绘制分布图")
    parser.add_argument("path", type=str, help="要分析的文件夹路径或单个 .pt 文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，用于保存分布图 (默认: ./draw/outliers/)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="线程数，用于并行处理文件 (默认: 32)。"
                             "推荐：0.25-0.5x CPU核心数用于CPU密集型任务，"
                             "或0.5-1x用于I/O密集型任务。")
    args = parser.parse_args()
    
    path = Path(args.path)
    if path.is_file() and path.suffix == ".pt":
        # 如果是单个文件，计算并输出
        num_sigma = compute_smax_gamma(path)
        if num_sigma is not None:
            print(f"S_max / sigma = {num_sigma:.4f}")
    elif path.is_dir():
        # 如果是文件夹，统计所有文件并绘制分布图
        analyze_folder(path, output_dir=args.output_dir, num_workers=args.num_workers)
    else:
        print(f"错误: {args.path} 不是一个有效的文件夹或 .pt 文件路径")
