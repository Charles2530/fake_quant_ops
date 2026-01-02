#!/usr/bin/env python3
"""
Multi-threaded tensor distribution analysis.
Processes all tensor files in a directory, aggregates data, filters outliers (|value| > 6),
and creates a combined distribution plot.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import laplace

# Import data format information from layer_analysis
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_analysis import DATA_TYPE_INFO

def load_and_process_tensor(filepath):
    """Load and process a single tensor file."""
    try:
        filename = os.path.basename(filepath)
        
        # Detect data format from filename
        data_format = detect_data_format(filename)
        if data_format is None:
            data_format = 'bf16'
        
        # Load tensor
        tensor = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Handle case where loaded object is not a tensor
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, dict) and 'tensor' in tensor:
                tensor = tensor['tensor']
            elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
                tensor = tensor[0]
            else:
                return None, None, None
        
        # Convert BFloat16 and other unsupported types to Float32 for CPU processing
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        elif tensor.dtype in [torch.float16, torch.half]:
            tensor = tensor.float()
        elif tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            tensor = tensor.float()
        elif tensor.dtype in [torch.uint8]:
            tensor = tensor.float()
        
        # Convert to numpy and flatten
        # Detach from computation graph if tensor requires grad
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = tensor.numpy()
        
        # Handle empty tensors
        if tensor_np.size == 0:
            return None, None, None
        
        # Handle complex tensors
        if tensor_np.dtype in [np.complex64, np.complex128]:
            tensor_np = np.abs(tensor_np)
        
        # Flatten for distribution analysis
        flat_tensor = tensor_np.flatten()
        
        return flat_tensor, data_format, filename
        
    except Exception as e:
        return None, None, None

def detect_data_format(filename):
    """Extract data format from filename."""
    for fmt in DATA_TYPE_INFO.keys():
        if fmt in filename:
            return fmt
    return None

def process_single_file(filepath):
    """Process a single tensor file and return its data."""
    data, data_format, filename = load_and_process_tensor(filepath)
    if data is not None:
        return data, data_format, filename
    return None, None, None

def create_aggregated_distribution_plot(all_data, data_format, output_path, 
                                       total_files, processed_files, 
                                       original_count, filtered_count):
    """
    Create aggregated distribution plot from all tensor data.
    
    Args:
        all_data (np.array): Aggregated flattened tensor data (after outlier removal)
        data_format (str): Data format identifier
        output_path (Path): Output file path
        total_files (int): Total number of tensor files found
        processed_files (int): Number of successfully processed files
        original_count (int): Total number of values before filtering
        filtered_count (int): Number of outliers removed (|value| > 6)
    """
    if data_format not in DATA_TYPE_INFO:
        raise ValueError(f"Unknown data format: {data_format}")
    
    format_info = DATA_TYPE_INFO[data_format]
    
    # Calculate data statistics
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    data_mean = np.mean(all_data)
    data_std = np.std(all_data)
    data_median = np.median(all_data)
    
    data_range = data_max - data_min
    
    # Dynamic range calculation - focus on data distribution
    if data_range > 0:
        margin = data_range * 0.15  # 15% margin
        plot_min = data_min - margin
        plot_max = data_max + margin
    else:
        plot_min, plot_max = data_min - 1, data_max + 1
    
    # Create figure (adjusted for single-column paper format)
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate histogram
    n_bins = min(200, max(50, int(np.sqrt(len(all_data)))))
    counts, bins, patches = plt.hist(all_data, bins=n_bins, alpha=0.7, 
                                   color=format_info['color'], density=True,
                                   label=f'Aggregated Tensor Values (n={len(all_data):,})',
                                   linewidth=0.3, edgecolor='black')
    
    # Set dynamic x-axis range
    plt.xlim(plot_min, plot_max)
    
    # Add representable values as vertical red lines (filtered to plot range)
    if format_info['representable_values'] is not None:
        rep_values = np.array(format_info['representable_values'])
        visible_rep_values = rep_values[(rep_values >= plot_min) & (rep_values <= plot_max)]
        
        print(f"Showing {len(visible_rep_values)} representable values in range [{plot_min:.3f}, {plot_max:.3f}]")
        
        # Add vertical lines for representable values
        for val in visible_rep_values:
            plt.axvline(val, color='red', alpha=0.6, linewidth=0.8, zorder=3)
    
    # Fit Laplace distribution
    # Laplace distribution: f(x) = (1/(2b)) * exp(-|x - μ|/b)
    # where μ is location (median) and b is scale parameter
    laplace_loc = np.median(all_data)
    laplace_scale = np.mean(np.abs(all_data - laplace_loc))
    
    # Generate x values for Laplace fit curve
    x_fit = np.linspace(plot_min, plot_max, 1000)
    laplace_pdf = laplace.pdf(x_fit, loc=laplace_loc, scale=laplace_scale)
    
    # Plot Laplace fit curve as gray dashed line (denser dashes, thinner line)
    plt.plot(x_fit, laplace_pdf, color='gray', linewidth=1.0, 
             alpha=0.8, label='Laplace Fit', zorder=5, dashes=(2, 1.5))
    
    # Add zero reference line
    if plot_min < 0 < plot_max:
        plt.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='Zero')
    
    outlier_percent = (filtered_count / original_count) * 100 if original_count > 0 else 0
    
    # Set labels (adjusted for single-column paper)
    plt.xlabel('Value', fontsize=9, fontweight='normal')
    plt.ylabel('Density', fontsize=9, fontweight='normal')
    
    # Set tick label font size
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=7)
    
    # Add legend (adjusted for single-column paper)
    legend_elements = [
        # plt.Line2D([0], [0], color=format_info['color'], alpha=0.7, linewidth=3, label='Aggregated Tensor Values'),
        plt.Line2D([0], [0], color='gray', linewidth=1.0, label='Laplace Fit', dashes=(2, 1.5)),
        # plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=1.5, label='Outlier Boundary (|value| = 6)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Improve layout (adjusted for single-column paper)
    plt.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout(pad=0.5)
    
    # Save plot (high DPI for publication quality)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Aggregated distribution plot saved to: {output_path}")
    
    # Return analysis summary
    return {
        'data_format': data_format,
        'total_files': total_files,
        'processed_files': processed_files,
        'total_elements': len(all_data),
        'original_count': original_count,
        'filtered_count': filtered_count,
        'outlier_percent': outlier_percent,
        'value_range': [data_min, data_max],
        'mean_std': [data_mean, data_std],
        'median': data_median,
        'laplace_loc': laplace_loc,
        'laplace_scale': laplace_scale
    }

def main():
    """Main function for aggregated distribution analysis."""
    parser = argparse.ArgumentParser(
        description='Generate aggregated tensor value distribution plots from all .pt files in a directory'
    )
    parser.add_argument('input_dir', help='Path to directory containing tensor files (.pt)')
    parser.add_argument('--output-dir', default='./draw/distribution_aggregated/', 
                        help='Output directory for plots (default: ./draw/distribution_aggregated/)')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of worker threads (default: 32)')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                        help='Outlier threshold (default: 3.0, values with |value| > threshold will be removed)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Print detailed statistics to console')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1
    
    # Find all .pt files
    pt_files = list(input_dir.glob('*.pt'))
    if len(pt_files) == 0:
        print(f"Error: No .pt files found in directory: {input_dir}")
        return 1
    
    print(f"Found {len(pt_files)} tensor files in {input_dir}")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_filename = f'aggregated_distribution_{input_dir.name}.png'
    output_path = output_dir / output_filename
    
    # Process all files in parallel
    all_data_list = []
    data_format_counts = {}
    processed_count = 0
    
    print(f"Processing {len(pt_files)} files with {args.num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, str(filepath)): filepath 
            for filepath in pt_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(pt_files), desc="Loading tensors") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    data, data_format, filename = future.result()
                    if data is not None:
                        all_data_list.append(data)
                        data_format_counts[data_format] = data_format_counts.get(data_format, 0) + 1
                        processed_count += 1
                except Exception as e:
                    print(f"\nError processing {filepath.name}: {e}")
                finally:
                    pbar.update(1)
    
    if len(all_data_list) == 0:
        print("Error: No valid tensor data was loaded from any files.")
        return 1
    
    print(f"\nSuccessfully processed {processed_count}/{len(pt_files)} files")
    
    # Determine most common data format
    if len(data_format_counts) > 0:
        data_format = max(data_format_counts, key=data_format_counts.get)
        print(f"Most common data format: {data_format.upper()} ({data_format_counts[data_format]} files)")
    else:
        data_format = 'bf16'
        print(f"Using default data format: {data_format.upper()}")
    
    # Concatenate all data
    print("\nConcatenating all tensor data...")
    all_data = np.concatenate(all_data_list)
    original_count = len(all_data)
    print(f"Total elements before filtering: {original_count:,}")
    
    # Filter outliers (|value| > threshold)
    print(f"\nFiltering outliers (|value| > {args.outlier_threshold})...")
    mask = np.abs(all_data) <= args.outlier_threshold
    filtered_count = np.sum(~mask)
    all_data = all_data[mask]
    print(f"Removed {filtered_count:,} outliers ({filtered_count/original_count*100:.4f}%)")
    print(f"Total elements after filtering: {len(all_data):,}")
    
    # Create aggregated distribution plot
    print("\nCreating aggregated distribution plot...")
    analysis_summary = create_aggregated_distribution_plot(
        all_data, data_format, output_path,
        len(pt_files), processed_count,
        original_count, filtered_count
    )
    
    # Print summary statistics if requested
    if args.show_stats:
        print("\nDetailed Analysis Summary:")
        print("-" * 40)
        print(f"Data Format: {analysis_summary['data_format'].upper()}")
        print(f"Total Files: {analysis_summary['total_files']}")
        print(f"Processed Files: {analysis_summary['processed_files']}")
        print(f"Total Elements (after filtering): {analysis_summary['total_elements']:,}")
        print(f"Original Elements: {analysis_summary['original_count']:,}")
        print(f"Outliers Removed: {analysis_summary['filtered_count']:,} ({analysis_summary['outlier_percent']:.4f}%)")
        print(f"Value Range: [{analysis_summary['value_range'][0]:.6f}, {analysis_summary['value_range'][1]:.6f}]")
        print(f"Mean ± Std: {analysis_summary['mean_std'][0]:.6f} ± {analysis_summary['mean_std'][1]:.6f}")
        print(f"Median: {analysis_summary['median']:.6f}")
        print(f"Laplace Fit - Location (μ): {analysis_summary['laplace_loc']:.6f}")
        print(f"Laplace Fit - Scale (b): {analysis_summary['laplace_scale']:.6f}")
    
    print(f"\nVisualization complete!")
    print(f"Plot saved to: {output_path}")
    
    # Provide usage assessment
    print("\nUsability Assessment:")
    print("-" * 30)
    
    outlier_pct = analysis_summary['outlier_percent']
    
    if outlier_pct < 1.0:
        print("🟢 EXCELLENT: Very few outliers detected")
    elif outlier_pct < 5.0:
        print("🟡 GOOD: Acceptable number of outliers")
    elif outlier_pct < 10.0:
        print("🟠 CAUTION: Some outliers detected")
    else:
        print("🔴 POOR: Significant number of outliers detected")
    
    return 0

if __name__ == "__main__":
    exit(main())

