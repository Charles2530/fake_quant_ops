import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

# ==========================================
# [Configuration Area]
# ==========================================
X_LIMIT = (0,700) 
FILE_BF16 = 'bf16_deepseek_1118.log'      # Baseline log (green background)
FILE_CURRENT = 'mxfp8_deepseek.log'  # Current log (red main)
OUTPUT_IMG = f'draw/compare_loss_{X_LIMIT[0]}_{X_LIMIT[1]}.png' # Output image name

# [Key] Only show first 2000 steps (locked range)
# ==========================================
def read_log_data(file_path):
    """
    Read log function
    Regex logic: match "iteration number / ... lm loss: value"
    """
    iters = []
    losses = []
    
    # This regex requires iteration to be followed by "/" to prevent matching nan iterations
    pattern = re.compile(r"iteration\s*(\d+)\s*/\s*\d+.*?lm loss:\s*([\d\.E\+\-]+)", re.IGNORECASE)

    print(f"Reading: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        iter_val = int(match.group(1))
                        loss_val = float(match.group(2))
                        iters.append(iter_val)
                        losses.append(loss_val)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        return [], []

    return iters, losses

def main():
    # 1. Read data
    iters_bf16, losses_bf16 = read_log_data(FILE_BF16)
    iters_curr, losses_curr = read_log_data(FILE_CURRENT)

    # Debug: Print data statistics
    print(f"\n[Debug] BF16 data: {len(iters_bf16)} points")
    if iters_bf16:
        print(f"        Range: {min(iters_bf16)} - {max(iters_bf16)}")
    print(f"[Debug] Current data: {len(iters_curr)} points")
    if iters_curr:
        print(f"        Range: {min(iters_curr)} - {max(iters_curr)}")

    if not iters_bf16:
        print(f"[Warning] BF16 log file '{FILE_BF16}' has no data or file not found!")
        print("         Will only plot current run (red line).")
    if not iters_curr:
        print(f"[Error] Current log file '{FILE_CURRENT}' has no data or file not found!")
        print("        Cannot generate plot.")
        return

    # =======================================================
    # 2. [Calculation] Calculate Loss ratio (Current / BF16) in range 0 ~ X_LIMIT
    # =======================================================
    ratios = []
    ratio_str = ""
    
    if iters_bf16 and losses_bf16:
        # Convert BF16 to dictionary {step: loss} for quick lookup
        dict_bf16 = dict(zip(iters_bf16, losses_bf16))
        
        print("-" * 40)
        print(f"Calculating average Loss ratio for range {X_LIMIT[0]}-{X_LIMIT[1]} steps...")

        for i, step in enumerate(iters_curr):
            # Filter condition: within X_LIMIT range, and BF16 log also has data for this step
            if step >= X_LIMIT[0] and step <= X_LIMIT[1] and step in dict_bf16:
                loss_now = losses_curr[i]
                loss_base = dict_bf16[step]
                
                if loss_base != 0:
                    # Calculate ratio
                    ratios.append(loss_now / loss_base)
        
        # Generate statistical text for title
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            ratio_str = f"Avg Ratio ({X_LIMIT[0]}-{X_LIMIT[1]}): {avg_ratio:.4f}x"
            print(f"[Result] Average ratio: {avg_ratio:.4f} (based on {len(ratios)} overlapping points)")
            print("       (>1.0 means current Loss is high, <1.0 means current Loss is low)")
        else:
            ratio_str = "No Overlap Data"
            print(f"[Warning] No overlapping steps found in range {X_LIMIT[0]}-{X_LIMIT[1]}, cannot calculate ratio.")
        print("-" * 40)
    else:
        ratio_str = "BF16 data unavailable"
        print("[Info] Skipping ratio calculation (no BF16 data)")

    # 3. Start plotting
    plt.figure(figsize=(12, 8), dpi=300) # High-resolution canvas

    # --- Draw baseline line (Blue) ---
    # Blue, placed at bottom layer as reference
    if iters_bf16 and losses_bf16:
        # Filter data within X_LIMIT range for display
        filtered_bf16_iters = [x for x in iters_bf16 if X_LIMIT[0] <= x <= X_LIMIT[1]]
        filtered_bf16_losses = [losses_bf16[i] for i, x in enumerate(iters_bf16) if X_LIMIT[0] <= x <= X_LIMIT[1]]
        
        if filtered_bf16_iters:
            plt.plot(filtered_bf16_iters, filtered_bf16_losses, 
                     label='Baseline (BF16)', 
                     color='red',      # [Color] Blue
                     linestyle='-',      # Solid line
                     linewidth=0.6,)     # [Line width] Thin line (show details)
                    #  alpha=1.0,          # [Transparency] Opaque (1.0)
                    #  zorder=1)           # Layer level 1 (bottom layer)
            print(f"[Debug] Plotted BF16 line with {len(filtered_bf16_iters)} points in range {X_LIMIT[0]}-{X_LIMIT[1]}")
        else:
            print(f"[Warning] BF16 data has no points in range {X_LIMIT[0]}-{X_LIMIT[1]}")
    else:
        print("[Warning] Skipping BF16 line (no data)")

    # --- Draw current line (Green) ---
    # Green, placed at top layer
    if iters_curr and losses_curr:
        # Filter data within X_LIMIT range for display
        filtered_curr_iters = [x for x in iters_curr if X_LIMIT[0] <= x <= X_LIMIT[1]]
        filtered_curr_losses = [losses_curr[i] for i, x in enumerate(iters_curr) if X_LIMIT[0] <= x <= X_LIMIT[1]]
        
        if filtered_curr_iters:
            plt.plot(filtered_curr_iters, filtered_curr_losses, 
                     label='Current Run', 
                     color='green',        # [Color] Green
                     linestyle='-',      # Solid line
                     linewidth=0.6)      # [Line width] Thin line
                    #  alpha=1.0,          # [Transparency] Opaque (1.0)
                    #  zorder=1)          # Layer level 10 (top layer)
            print(f"[Debug] Plotted Current line with {len(filtered_curr_iters)} points in range {X_LIMIT[0]}-{X_LIMIT[1]}")
        else:
            print(f"[Warning] Current data has no points in range {X_LIMIT[0]}-{X_LIMIT[1]}")
    else:
        print("[Error] Cannot plot current line (no data)")
        return

    # 4. Set axis range and ticks
    ax = plt.gca()
    
    # [Key] Force lock x-axis display range: 0 to 2000
    plt.xlim(0, X_LIMIT[1])
    
    # Smart scale: make x-axis show only about 8 numbers, clean and not crowded
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    # 5. Decorate image
    # Write the calculated ratio directly in the title
    plt.title(f'Training Loss Comparison\n{ratio_str}', fontsize=18, fontweight='bold')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('LM Loss', fontsize=14)
    
    # Place legend in upper right corner
    plt.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
    
    # Grid lines
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # 6. Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Image generated: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()