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
import json
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

def process_single_file(filepath, outlier_threshold=3.0):
    """
    Process a single tensor file and return statistics and histogram.
    Memory is released immediately after processing.
    """
    data, data_format, filename = load_and_process_tensor(filepath)
    if data is None:
        return None
    
    # Filter outliers immediately
    original_count = len(data)
    mask = np.abs(data) <= outlier_threshold
    filtered_count = np.sum(~mask)
    filtered_data = data[mask]
    
    # Calculate statistics before releasing memory
    data_min = np.min(filtered_data) if len(filtered_data) > 0 else 0.0
    data_max = np.max(filtered_data) if len(filtered_data) > 0 else 0.0
    data_sum = np.sum(filtered_data)
    data_sum_sq = np.sum(filtered_data ** 2)
    data_count = len(filtered_data)
    
    # Calculate histogram (this is memory efficient)
    # Use a reasonable number of bins
    n_bins = min(200, max(50, int(np.sqrt(data_count))))
    if data_count > 0 and data_max > data_min:
        hist, bin_edges = np.histogram(filtered_data, bins=n_bins, range=(data_min, data_max))
    else:
        hist = np.array([])
        bin_edges = np.array([data_min, data_max])
    
    # Release memory immediately
    del data, filtered_data
    
    return {
        'data_format': data_format,
        'filename': filename,
        'original_count': original_count,
        'filtered_count': filtered_count,
        'data_count': data_count,
        'data_min': data_min,
        'data_max': data_max,
        'data_sum': data_sum,
        'data_sum_sq': data_sum_sq,
        'hist': hist,
        'bin_edges': bin_edges
    }

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
    
    # Process all files in parallel with memory-efficient streaming
    stats_list = []
    data_format_counts = {}
    processed_count = 0
    original_count = 0
    filtered_count = 0
    
    # Accumulated statistics
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    global_min = float('inf')
    global_max = float('-inf')
    
    print(f"Processing {len(pt_files)} files with {args.num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, str(filepath), args.outlier_threshold): filepath 
            for filepath in pt_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(pt_files), desc="Processing tensors") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    stats = future.result()
                    if stats is not None:
                        stats_list.append(stats)
                        data_format_counts[stats['data_format']] = data_format_counts.get(stats['data_format'], 0) + 1
                        processed_count += 1
                        
                        # Accumulate statistics
                        original_count += stats['original_count']
                        filtered_count += stats['filtered_count']
                        total_count += stats['data_count']
                        total_sum += stats['data_sum']
                        total_sum_sq += stats['data_sum_sq']
                        
                        if stats['data_count'] > 0:
                            global_min = min(global_min, stats['data_min'])
                            global_max = max(global_max, stats['data_max'])
                except Exception as e:
                    print(f"\nError processing {filepath.name}: {e}")
                finally:
                    pbar.update(1)
    
    if processed_count == 0:
        print("Error: No valid tensor data was loaded from any files.")
        return 1
    
    print(f"\nSuccessfully processed {processed_count}/{len(pt_files)} files")
    print(f"Total elements before filtering: {original_count:,}")
    print(f"Total outliers removed: {filtered_count:,} ({filtered_count/original_count*100:.4f}%)")
    print(f"Total elements after filtering: {total_count:,}")
    
    # Determine most common data format
    if len(data_format_counts) > 0:
        data_format = max(data_format_counts, key=data_format_counts.get)
        print(f"Most common data format: {data_format.upper()} ({data_format_counts[data_format]} files)")
    else:
        data_format = 'bf16'
        print(f"Using default data format: {data_format.upper()}")
    
    # Reconstruct a representative dataset from histograms for plotting
    # We'll sample from the merged histogram to create a dataset
    print("\nReconstructing representative dataset from histograms...")
    
    # Merge all histograms into one
    n_bins = 200  # Number of bins for merged histogram
    if global_max > global_min and total_count > 0:
        # Create unified bin edges
        bin_edges = np.linspace(global_min, global_max, n_bins + 1)
        hist_merged = np.zeros(n_bins)
        
        # Merge histograms by redistributing counts to unified bins
        for stats in stats_list:
            if stats['data_count'] > 0 and len(stats['hist']) > 0:
                # Get bin centers from original histogram
                bin_centers_orig = (stats['bin_edges'][:-1] + stats['bin_edges'][1:]) / 2
                # Distribute counts to new bins
                for i, center in enumerate(bin_centers_orig):
                    if i < len(stats['hist']) and stats['hist'][i] > 0:
                        bin_idx = np.searchsorted(bin_edges, center) - 1
                        bin_idx = max(0, min(n_bins - 1, bin_idx))
                        hist_merged[bin_idx] += stats['hist'][i]
        
        # Sample from histogram to create representative dataset
        # Limit to reasonable size (e.g., 1M points max) to save memory
        max_samples = min(1000000, total_count)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        probabilities = hist_merged / np.sum(hist_merged) if np.sum(hist_merged) > 0 else hist_merged
        if np.sum(probabilities) > 0:
            sampled_indices = np.random.choice(len(bin_centers), size=max_samples, p=probabilities, replace=True)
            all_data = bin_centers[sampled_indices]
        else:
            all_data = np.array([])
    else:
        all_data = np.array([])
        bin_edges = np.array([global_min, global_max]) if global_max > global_min else np.array([-1, 1])
    
    if len(all_data) > 0:
        print(f"Created representative dataset with {len(all_data):,} samples")
    else:
        print("Warning: No data available for plotting")
        return 1
    
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
    
    # Save statistics to JSON file
    json_filename = output_path.stem + '.json'
    json_path = output_dir / json_filename
    
    # Prepare statistics for JSON (convert numpy types to Python native types)
    json_stats = {
        'input_directory': str(input_dir),
        'data_format': analysis_summary['data_format'],
        'total_files': analysis_summary['total_files'],
        'processed_files': analysis_summary['processed_files'],
        'original_count': int(analysis_summary['original_count']),
        'filtered_count': int(analysis_summary['filtered_count']),
        'outlier_percent': float(analysis_summary['outlier_percent']),
        'outlier_threshold': float(args.outlier_threshold),
        'total_elements_after_filtering': int(analysis_summary['total_elements']),
        'value_range': {
            'min': float(analysis_summary['value_range'][0]),
            'max': float(analysis_summary['value_range'][1])
        },
        'statistics': {
            'mean': float(analysis_summary['mean_std'][0]),
            'std': float(analysis_summary['mean_std'][1]),
            'median': float(analysis_summary['median'])
        },
        'laplace_fit': {
            'location_mu': float(analysis_summary['laplace_loc']),
            'scale_b': float(analysis_summary['laplace_scale'])
        },
        'data_format_counts': data_format_counts
    }
    
    # Save to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {json_path}")
    
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

