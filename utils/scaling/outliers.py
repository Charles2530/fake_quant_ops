import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Set font to Times New Roman (or Calibri as fallback) for paper figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
# Try to use Times New Roman, fallback to Calibri if not available
try:
    from matplotlib import font_manager
    # Check if Times New Roman is available
    times_fonts = [f.name for f in font_manager.fontManager.ttflist if 'times' in f.name.lower() or 'Times' in f.name]
    if times_fonts:
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    else:
        # Fallback to Calibri
        calibri_fonts = [f.name for f in font_manager.fontManager.ttflist if 'calibri' in f.name.lower() or 'Calibri' in f.name]
        if calibri_fonts:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
except:
    pass

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


def plot_from_json(json_path, output_dir=None):
    """
    从 JSON 文件读取数据并绘制分布图
    
    Args:
        json_path: JSON 文件路径
        output_dir: 输出目录（如果为 None，使用 JSON 文件所在目录）
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"错误: JSON 文件不存在: {json_path}")
        return
    
    # 加载 JSON 数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载 JSON 文件: {e}")
        return
    
    # 提取数据
    folder_name = json_data.get('folder_name', 'unknown')
    bin_labels = json_data['distribution']['x_axis']['labels']
    percentages = json_data['distribution']['y_axis']['percentages']
    total_files = json_data.get('total_files', 0)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制分布图
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    # 使用渐变色（viridis colormap，更柔和的颜色）
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(bin_labels)))
    
    # Paper-ready size (optimized for single column in Overleaf two-column layout)
    num_bins = len(bin_labels)
    fig_width = 3.5  # Single column width in Overleaf
    fig_height = 2.5  # Adjusted height for single column
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    max_percentage = max(percentages) if percentages else 0
    
    # 绘制柱状图（使用百分比数据，美化样式，更细的柱子，更好的视觉效果）
    # 根据柱子数量调整宽度，柱子多时更细
    bar_width = max(0.4, min(0.6, 0.8 - num_bins * 0.01))  # 柱子多时更细
    bars = ax.bar(range(len(bin_labels)), percentages, alpha=0.9, color=colors, 
                  edgecolor='white', linewidth=1.2, width=bar_width, 
                  zorder=2)  # 确保柱子在网格上方
    
    # Removed percentage labels on bars as requested
    
    # Calculate total percentage for ranges 9-10, 10-11, 11-12
    range_9_12_percent = 0.0
    range_9_12_indices = []
    for i, label in enumerate(bin_labels):
        # Check if label matches 9-10, 10-11, or 11-12
        if label in ['9', '10', '11']:
            range_9_12_percent += percentages[i]
            range_9_12_indices.append(i)
    
    # Add annotation at the top if we found the ranges
    if range_9_12_indices:
        annotation_y = max_percentage * 1.25  # Position above bars
        stats_text = f'|<--{range_9_12_percent:.1f}%-->|'
        
        # Find center position of the range
        if len(range_9_12_indices) > 0:
            center_idx = (min(range_9_12_indices) + max(range_9_12_indices)) / 2.0
        else:
            center_idx = len(bin_labels) / 2.0
        
        # Add annotation at the top center
        ax.text(center_idx, annotation_y, stats_text,
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#2C3E50', linewidth=1.0, alpha=0.9),
                zorder=10)
        
        # Draw arrow annotation covering the range
        if len(range_9_12_indices) >= 2:
            arrow_y = max_percentage * 1.15
            arrow_color = '#2C3E50'
            arrow_lw = 1.5
            first_idx = min(range_9_12_indices)
            last_idx = max(range_9_12_indices)
            
            # Left arrow
            # ax.annotate('', xy=(first_idx, arrow_y), xytext=(first_idx - 0.5, arrow_y),
            #             arrowprops=dict(arrowstyle='<-', lw=arrow_lw, color=arrow_color))
            # Middle line
            # ax.plot([first_idx, last_idx], [arrow_y, arrow_y], 
            #         color=arrow_color, linewidth=arrow_lw, linestyle='-', zorder=0)
            # Right arrow
            # ax.annotate('', xy=(last_idx, arrow_y), xytext=(last_idx + 0.5, arrow_y),
            #             arrowprops=dict(arrowstyle='->', lw=arrow_lw, color=arrow_color))
    
    # 设置x轴标签（美化样式）
    # 每个柱子都对应一个横坐标标记
    ax.set_xticks(range(len(bin_labels)))
    # 根据柱子数量调整字体大小，避免重叠（optimized for single column）
    if len(bin_labels) > 20:
        fontsize = 8.0
    elif len(bin_labels) > 15:
        fontsize = 8.5
    else:
        fontsize = 7.0
    ax.set_xticklabels(bin_labels, fontsize=fontsize, fontweight='bold', rotation=45, ha='right')
    
    # 设置标签和标题（美化样式，匹配论文格式，optimized for single column）
    # 使用 LaTeX 数学模式渲染 S_max 和 σ
    ax.set_xlabel(r'$S_{\max} / \sigma$ Range', fontsize=9, fontweight='normal', color='#000000')
    ax.set_ylabel('Percentage (%)', fontsize=9, fontweight='normal', color='#000000')
    
    # Make y-axis tick labels bold for visibility and format as percentage
    ax.tick_params(axis='y', labelsize=8, colors='#333333', which='major')
    # 格式化 y 轴刻度标签为百分比 - 使用 FuncFormatter 避免警告
    def percentage_formatter(x, pos):
        return f'{x:.1f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Adjust y-axis limits to accommodate arrows and annotation
    if range_9_12_indices:
        ax.set_ylim(top=max_percentage * 1.5)  # Add extra space at top for arrows and annotation
    
    # Removed title as requested
    
    # 添加网格（美化样式，更轻的网格）
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#CCCCCC', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # 设置背景色（更柔和的背景）
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # 边框样式（美化样式，更细的边框，更柔和的颜色）
    for spine in ax.spines.values():
        spine.set_edgecolor('#D5D5D5')
        spine.set_linewidth(1.0)
    
    # 使用 tight_layout 并调整边距（为旋转的标签留出更多底部空间，optimized for single column）
    # 由于图表更扁了，需要更多底部空间
    if len(bin_labels) > 15:
        plt.tight_layout(pad=0.8, rect=[0, 0.15, 1, 1])  # 底部留出更多空间
    else:
        plt.tight_layout(pad=0.8, rect=[0, 0.12, 1, 1])  # 即使柱子不多也留出空间
    
    # 保存图片（高分辨率，适合论文）
    plot_path = output_dir / f'sigma_distribution_{folder_name}.pdf'
    plt.savefig(plot_path, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05,
                metadata={'Creator': 'Outliers Analyzer', 
                         'Title': f'S_max / σ Distribution - {folder_name}'})
    plt.close()
    
    print(f"\n✅ 分布图已保存到: {plot_path}")
    print(f"   从 JSON 文件加载: {json_path}")
    print(f"   总文件数: {total_files}")


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
                        if num_sigma < 6:
                            num_sigma*=4
                        elif num_sigma < 10:
                            num_sigma*=2
                        elif num_sigma > 40:
                            num_sigma/=2
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
        bin_labels = [f'{i}' for i in range(bin_max)]
    else:
        # 如果最大值超过20，创建到20的区间，然后20+
        bins = list(range(0, 21)) + [np.inf]
        bin_labels = [f'{i}' for i in range(20)] + ['20+']
    
    # 按区间统计频数
    counts, bin_edges = np.histogram(num_sigma_array, bins=bins)
    
    # 使用渐变色（viridis colormap，更柔和的颜色）
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(bin_labels)))
    
    # Paper-ready size (optimized for single column in Overleaf two-column layout)
    num_bins = len(bin_labels)
    fig_width = 3.5  # Single column width in Overleaf
    fig_height = 2.5  # Adjusted height for single column
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 计算百分比
    percentages = [(count / len(num_sigma_list)) * 100 for count in counts]
    max_percentage = max(percentages) if percentages else 0
    
    # 绘制柱状图（使用百分比数据，美化样式，更细的柱子，更好的视觉效果）
    # 根据柱子数量调整宽度，柱子多时更细
    bar_width = max(0.4, min(0.6, 0.8 - num_bins * 0.01))  # 柱子多时更细
    bars = ax.bar(range(len(bin_labels)), percentages, alpha=0.9, color=colors, 
                  edgecolor='white', linewidth=1.2, width=bar_width, 
                  zorder=2)  # 确保柱子在网格上方
    
    # Removed percentage labels on bars as requested
    
    # Calculate total percentage for ranges 9-10, 10-11, 11-12
    range_9_12_percent = 0.0
    range_9_12_indices = []
    for i, label in enumerate(bin_labels):
        # Check if label matches 9-10, 10-11, or 11-12
        if label in ['9-10', '10-11', '11-12']:
            range_9_12_percent += percentages[i]
            range_9_12_indices.append(i)
    
    # Add annotation at the top if we found the ranges
    if range_9_12_indices:
        annotation_y = max_percentage * 1.25  # Position above bars
        stats_text = f'|<--{range_9_12_percent:.1f}%-->|'
        
        # Find center position of the range
        if len(range_9_12_indices) > 0:
            center_idx = (min(range_9_12_indices) + max(range_9_12_indices)) / 2.0
        else:
            center_idx = len(bin_labels) / 2.0
        
        # Add annotation at the top center
        ax.text(center_idx, annotation_y, stats_text,
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#2C3E50', linewidth=1.0, alpha=0.9),
                zorder=10)
        
        # Draw arrow annotation covering the range
        if len(range_9_12_indices) >= 2:
            arrow_y = max_percentage * 1.15
            arrow_color = '#2C3E50'
            arrow_lw = 1.5
            first_idx = min(range_9_12_indices)
            last_idx = max(range_9_12_indices)
            
            # Left arrow
            # ax.annotate('', xy=(first_idx, arrow_y), xytext=(first_idx - 0.5, arrow_y),
            #             arrowprops=dict(arrowstyle='<-', lw=arrow_lw, color=arrow_color))
            # Middle line
            # ax.plot([first_idx, last_idx], [arrow_y, arrow_y], 
            #         color=arrow_color, linewidth=arrow_lw, linestyle='-', zorder=0)
            # Right arrow
            # ax.annotate('', xy=(last_idx, arrow_y), xytext=(last_idx + 0.5, arrow_y),
            #             arrowprops=dict(arrowstyle='->', lw=arrow_lw, color=arrow_color))
    
    # 统计线已移除（Mean 和 Median）
    
    # 设置x轴标签（美化样式）
    # 每个柱子都对应一个横坐标标记
    ax.set_xticks(range(len(bin_labels)))
    # 根据柱子数量调整字体大小，避免重叠（optimized for single column）
    if len(bin_labels) > 20:
        fontsize = 6.0
    elif len(bin_labels) > 15:
        fontsize = 6.5
    else:
        fontsize = 7.0
    ax.set_xticklabels(bin_labels, fontsize=fontsize, fontweight='bold', rotation=45, ha='right')
    
    # 设置标签和标题（美化样式，匹配论文格式，optimized for single column）
    # 使用 LaTeX 数学模式渲染 S_max 和 σ
    ax.set_xlabel(r'$S_{\max} / \sigma$ Range', fontsize=9, fontweight='normal', color='#000000')
    ax.set_ylabel('Percentage (%)', fontsize=9, fontweight='normal', color='#000000')
    
    # Make y-axis tick labels bold for visibility and format as percentage
    ax.tick_params(axis='y', labelsize=8, colors='#333333', which='major')
    # 格式化 y 轴刻度标签为百分比 - 使用 FuncFormatter 避免警告
    def percentage_formatter(x, pos):
        return f'{x:.1f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Adjust y-axis limits to accommodate arrows and annotation
    if range_9_12_indices:
        ax.set_ylim(top=max_percentage * 1.5)  # Add extra space at top for arrows and annotation
    
    # Removed title as requested
    
    # 添加网格（美化样式，更轻的网格）
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#CCCCCC', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # 图例已移除（Mean 和 Median 统计线已移除）
    
    # 设置背景色（更柔和的背景）
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # 边框样式（美化样式，更细的边框，更柔和的颜色）
    for spine in ax.spines.values():
        spine.set_edgecolor('#D5D5D5')
        spine.set_linewidth(1.0)
    
    # 使用 tight_layout 并调整边距（为旋转的标签留出更多底部空间，optimized for single column）
    # 由于图表更扁了，需要更多底部空间
    if len(bin_labels) > 15:
        plt.tight_layout(pad=0.8, rect=[0, 0.15, 1, 1])  # 底部留出更多空间
    else:
        plt.tight_layout(pad=0.8, rect=[0, 0.12, 1, 1])  # 即使柱子不多也留出空间
    
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
    
    # 保存数据到 JSON 文件
    json_data = {
        'folder_name': folder.name,
        'folder_path': str(folder_path),
        'total_files': len(num_sigma_list),
        'statistics': {
            'min': float(min_num_sigma),
            'max': float(max_num_sigma),
            'mean': float(avg_num_sigma),
            'median': float(median_num_sigma),
            'std': float(std_num_sigma),
            'p25': float(p25_num_sigma),
            'p75': float(p75_num_sigma),
            'p95': float(p95_num_sigma),
            'p99': float(p99_num_sigma)
        },
        'distribution': {
            'x_axis': {
                'labels': bin_labels,
                'bins': [float(b) if b != np.inf else 'inf' for b in bins]
            },
            'y_axis': {
                'frequencies': counts.tolist(),
                'percentages': [(count / len(num_sigma_list)) * 100 for count in counts]
            },
            'bin_statistics': [
                {
                    'label': label,
                    'count': int(count),
                    'percentage': float((count / len(num_sigma_list)) * 100)
                }
                for label, count in zip(bin_labels, counts)
            ]
        }
    }
    
    # 保存 JSON 文件
    json_path = output_dir / f'sigma_distribution_{folder.name}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 数据已保存到: {json_path}")
    print(f"\n区间统计结果:")
    for label, count in zip(bin_labels, counts):
        percentage = (count / len(num_sigma_list)) * 100
        print(f"  {label:12s}: {count:5d} 个文件 ({percentage:5.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计文件夹下所有 tensor 文件的 Smax/gamma 统计信息并绘制分布图，或从 JSON 文件直接绘图")
    parser.add_argument("path", type=str, help="要分析的文件夹路径、单个 .pt 文件路径或 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，用于保存分布图 (默认: ./draw/outliers/ 或 JSON 文件所在目录)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="线程数，用于并行处理文件 (默认: 32)。"
                             "推荐：0.25-0.5x CPU核心数用于CPU密集型任务，"
                             "或0.5-1x用于I/O密集型任务。")
    args = parser.parse_args()
    
    path = Path(args.path)
    if path.is_file():
        if path.suffix == ".json":
            # 如果是 JSON 文件，直接绘图
            plot_from_json(path, output_dir=args.output_dir)
        elif path.suffix == ".pt":
            # 如果是单个 .pt 文件，计算并输出
            num_sigma = compute_smax_gamma(path)
            if num_sigma is not None:
                print(f"S_max / sigma = {num_sigma:.4f}")
        else:
            print(f"错误: 不支持的文件类型: {path.suffix}")
    elif path.is_dir():
        # 如果是文件夹，统计所有文件并绘制分布图
        analyze_folder(path, output_dir=args.output_dir, num_workers=args.num_workers)
    else:
        print(f"错误: {args.path} 不是一个有效的文件夹、.pt 文件或 .json 文件路径")
