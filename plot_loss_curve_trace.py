#!/usr/bin/env python3
"""
Plot loss curves from extracted txt files.
Usage: python plot_loss_curve_trace.py <start_step> <end_step> \[model\]
Example: python plot_loss_curve_trace.py 0 3000
Example: python plot_loss_curve_trace.py 0 3000 OLMo-7B
"""

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
# 导入用于创建放大子图的工具
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==========================================
# [Model Configuration Area]
# ==========================================
MODEL_CONFIGS = {
    'OLMo-7B': {
        'data_dir': 'logs/OLMo-7b',
        'output_dir': 'logs/OLMo-7b',
        'title': 'Training Loss Curve of OLMo-7B',
        'zoom_range': (2700, 2750),  # Zoom x-axis range for inset
        'file_configs': [
            ('OLMo-7B-reproduce.txt', 'BF16', 'green', '-'),
            ('FakeQuant-Activation-OLMo-7B-MXFP-4-auto-reverse.txt', 'Four Over Six', 'red', '-'),
            ('FakeQuant-Activation-OLMo-7B-MXFP-4.txt', 'COAT*', 'yellow', '-'),
            ('FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto.txt', 'Half-S', 'blue', '-'),
            # ('FakeQuant-Activation-OLMo-7B-MXFP-4-Minus1.txt', 'Fixed Half-S', 'orange', '-'),
            # ('FakeQuant-Activation-OLMo-7B-MXFP-4-Minus2.txt', 'Fixed S/4', 'purple', '-'),
            # ('FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto-2.txt', 'Mixed (S + S/4)', 'brown', '-'),
        ]
    },
    'OLMo-1B': {
        'data_dir': 'logs/OLMo-1b',
        'output_dir': 'logs/OLMo-1b',
        'title': 'Training Loss Curve of OLMo-1B',
        'zoom_range': (1750, 1800),  # Zoom x-axis range for inset
        'file_configs': [
            ('OLMo-1B-reproduce.txt', 'BF16', 'green', '-'),
            # ('FakeQuant-Activation-OLMo-1B-MXFP-4-auto-reverse.txt', 'Four Over Six', 'red', '-'),
            # ('FakeQuant-Activation-OLMo-1B-MXFP-4.txt', 'COAT*', 'yellow', '-'),
            # ('FakeQuant-Activation-OLMo-1B-MXFP-4.txt', 'mxfp4 (Linear)', 'yellow', '-'),
            ('FakeQuant-Activation-OLMo-1B-MXFP-4-attn.txt', 'COAT* (A+L)', 'brown', '-'),
            # ('FakeQuant-Activation-OLMo-1B-MXFP-4-auto.txt', 'Half-S', 'blue', '-'),
            # ('FakeQuant-Activation-OLMo-1B-MXFP-4-auto.txt', 'Half-S (Linear)', 'blue', '-'),
            ('FakeQuant-Activation-OLMo-1B-MXFP-4-auto-attn.txt', 'Half-S (A+L)', 'purple', '-'),
        ]
    },
}

DEFAULT_MODEL = 'OLMo-7B'

# ==========================================
# [Functions]
# ==========================================

def read_txt_data(file_path):
    """
    Read step and CrossEntropyLoss from txt file.
    Format: Step\tCrossEntropyLoss
    """
    steps = []
    losses = []
    print(f"Reading: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        step = int(parts[0])
                        loss = float(parts[1])
                        steps.append(step)
                        losses.append(loss)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        return [], []
    except Exception as e:
        print(f"[Error] Error reading {file_path}: {e}")
        return [], []
    return steps, losses

def calculate_ratio(steps_base, losses_base, steps_curr, losses_curr, start_step, end_step):
    """
    Calculate average loss ratio (current / baseline) in the specified range.
    """
    if not steps_base or not losses_base or not steps_curr or not losses_curr:
        return None, 0
    dict_base = dict(zip(steps_base, losses_base))
    ratios = []
    for i, step in enumerate(steps_curr):
        if start_step <= step <= end_step and step in dict_base:
            loss_curr = losses_curr[i]
            loss_base = dict_base[step]
            if loss_base != 0:
                ratios.append(loss_curr / loss_base)
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        return avg_ratio, len(ratios)
    else:
        return None, 0

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_loss_curve_trace.py <start_step> <end_step> [model]")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step = int(sys.argv[2])
    except ValueError:
        print("[Error] Invalid arguments. Start and end steps must be integers.")
        sys.exit(1)

    if start_step < 0 or end_step <= start_step:
        print("[Error] Invalid range.")
        sys.exit(1)
        
    model_name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    if model_name not in MODEL_CONFIGS:
        print(f"[Error] Unknown model: {model_name}")
        sys.exit(1)

    model_config = MODEL_CONFIGS[model_name]
    DATA_DIR = model_config['data_dir']
    OUTPUT_DIR = model_config['output_dir']
    PLOT_TITLE = model_config['title']
    FILE_CONFIGS = model_config['file_configs']
    ZOOM_RANGE = model_config.get('zoom_range', (2700, 2750))  # Default zoom range if not specified

    print(f"\n[Info] Using model: {model_name}")

    X_LIMIT = (start_step, end_step)
    all_data = []
    baseline_data = None

    for filename, label, color, linestyle in FILE_CONFIGS:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}, skipping...")
            continue
        steps, losses = read_txt_data(file_path)
        if steps and losses:
            all_data.append({
                'filename': filename, 'label': label, 'color': color, 
                'linestyle': linestyle, 'steps': steps, 'losses': losses
            })
            if baseline_data is None:
                baseline_data = {'steps': steps, 'losses': losses}
            print(f"        Loaded {len(steps)} data points, range: {min(steps)} - {max(steps)}")

    if not all_data:
        print("[Error] No valid data files found!")
        sys.exit(1)

    # =======================================================
    # Calculate average loss for each file in the specified range
    # =======================================================
    print("-" * 60)
    print(f"Calculating average Loss for range {start_step}-{end_step} steps...")
    print("-" * 60)
    
    avg_losses = []
    for data in all_data:
        # Filter data within the specified range
        filtered_losses = [data['losses'][i] for i, step in enumerate(data['steps']) 
                          if start_step <= step <= end_step]
        
        if filtered_losses:
            avg_loss = sum(filtered_losses) / len(filtered_losses)
            avg_losses.append({
                'label': data['label'],
                'avg_loss': avg_loss,
                'num_points': len(filtered_losses)
            })
            print(f"  {data['label']:30s}: Avg Loss = {avg_loss:.6f} (based on {len(filtered_losses)} points)")
        else:
            print(f"  {data['label']:30s}: No data points in range {start_step}-{end_step}")
    
    print("-" * 60)
    
    # Calculate ratios if baseline exists
    if baseline_data:
        print(f"\nCalculating average Loss ratio (Current / Baseline) for range {start_step}-{end_step} steps...")
        print("-" * 60)
        for data in all_data:
            if data['label'] == 'BF16':  # Skip baseline itself
                continue
            avg_ratio, num_overlap = calculate_ratio(
                baseline_data['steps'], baseline_data['losses'],
                data['steps'], data['losses'],
                start_step, end_step
            )
            if avg_ratio is not None:
                print(f"  {data['label']:30s}: Avg Ratio = {avg_ratio:.4f}x (based on {num_overlap} overlapping points)")
            else:
                print(f"  {data['label']:30s}: No overlapping points with baseline")
        print("-" * 60)

    # High resolution settings for PDF output
    fig, ax = plt.subplots(figsize=(3.4, 2.55), dpi=600)
    plot_data_list = []

    for data in all_data:
        filtered_steps = [s for s in data['steps'] if X_LIMIT[0] <= s <= X_LIMIT[1]]
        filtered_losses = [data['losses'][i] for i, s in enumerate(data['steps']) if X_LIMIT[0] <= s <= X_LIMIT[1]]
        if filtered_steps:
            ax.plot(filtered_steps, filtered_losses,
                    label=data['label'], color=data['color'], linestyle=data['linestyle'],
                    linewidth=1.0, alpha=0.9)
            plot_data_list.append({
                'steps': filtered_steps, 'losses': filtered_losses, 'label': data['label'],
                'color': data['color'], 'linestyle': data['linestyle']
            })

    ax.set_xlim(start_step, end_step)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.set_title(f'{PLOT_TITLE}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Step', fontsize=9)
    ax.set_ylabel('CrossEntropyLoss', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper left', fontsize=7, frameon=True, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    # ==========================================
    # [新增] 创建放大区域
    # ==========================================
    # 1. 从配置中获取放大的 x 轴范围
    x1, x2 = ZOOM_RANGE

    # 2. 自动计算此 x 范围对应的 y 轴范围（排除 'Fixed Ex-2'）
    y_min_in_zoom = float('inf')
    y_max_in_zoom = float('-inf')
    for data in plot_data_list:
        # 跳过 'Fixed Ex-2' 曲线，因为它在后期跑飞了
        if data['label'] == 'Fixed Ex-2':
            continue
        for i, step in enumerate(data['steps']):
            if x1 <= step <= x2:
                loss = data['losses'][i]
                y_min_in_zoom = min(y_min_in_zoom, loss)
                y_max_in_zoom = max(y_max_in_zoom, loss)

    # 3. 创建 inset axes，[left, bottom, width, height]，数值是相对于主图的比例
    axins = ax.inset_axes([0.61, 0.45, 0.38, 0.38])

    # 4. 在 inset axes 上重新绘制所有曲线（排除 'Fixed Ex-2'）
    for data in plot_data_list:
        # 跳过 'Fixed Ex-2' 曲线，因为它在后期跑飞了
        if data['label'] == 'Fixed Ex-2':
            continue
        axins.plot(data['steps'], data['losses'],
                   color=data['color'], linestyle=data['linestyle'], linewidth=1.2)

    # 5. 设置 inset axes 的 x 和 y 轴范围
    y_padding = (y_max_in_zoom - y_min_in_zoom) * 0.1 # 增加一点y轴边距
    axins.set_xlim(x1, x2)
    axins.set_ylim(y_min_in_zoom - y_padding, y_max_in_zoom + y_padding)

    # 6. 隐藏 inset 的刻度标签，避免拥挤
    axins.tick_params(axis='both', which='major', labelsize=7)
    axins.grid(True, which='major', linestyle='--', alpha=0.6)
    
    # 7. 绘制指示框和连接线
    # loc1, loc2=3, 4 表示连接主图框的左下角和右下角
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    # ==========================================
    
    output_filename = f'{model_name}_{start_step}_{end_step}_zoomed.pdf'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.tight_layout()
    # High quality PDF settings
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                dpi=600, facecolor='white', edgecolor='none',
                pad_inches=0.02, metadata={'Creator': None})
    print(f"\n[Success] Image with zoom saved: {output_path}")

if __name__ == "__main__":
    main()
