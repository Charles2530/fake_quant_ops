#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import laplace
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import data format information from layer_analysis
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_analysis import DATA_TYPE_INFO

def load_and_process_tensor(filepath):
    try:
        filename = os.path.basename(filepath)
        
        # Detect data format from filename
        data_format = detect_data_format(filename)
        if data_format is None:
            data_format = 'bf16'
        #     print(f"Warning: Could not detect data format from filename: {filename}")
        #     return None, None, None
        
        # Load tensor
        tensor = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Handle case where loaded object is not a tensor
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, dict) and 'tensor' in tensor:
                tensor = tensor['tensor']
            elif isinstance(tensor, (list, tuple)) and len(tensor) > 0:
                tensor = tensor[0]
            else:
                print(f"Warning: Loaded object is not a tensor: {filename}")
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
            print(f"Warning: Empty tensor: {filename}")
            return None, None, None
        
        # Handle complex tensors
        if tensor_np.dtype in [np.complex64, np.complex128]:
            tensor_np = np.abs(tensor_np)
        
        # Flatten for distribution analysis
        flat_tensor = tensor_np.flatten()
        
        return flat_tensor, data_format, filename
        
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return None, None, None

def detect_data_format(filename):
    """Extract data format from filename."""
    for fmt in DATA_TYPE_INFO.keys():
        if fmt in filename:
            return fmt
    return None

def process_single_file(filepath, output_dir):
    """Process a single tensor file and create its distribution plot."""
    try:
        # Load and process tensor
        tensor_data, data_format, filename = load_and_process_tensor(str(filepath))
        
        if tensor_data is None:
            return None, f"Failed to load or process: {filepath.name}"
        
        # Setup output path
        output_filename = filepath.stem + '.pdf'
        output_path = output_dir / output_filename
        
        # Create distribution plot
        analysis_summary = create_distribution_plot(tensor_data, data_format, filename, output_path)
        
        return analysis_summary, None
    except Exception as e:
        return None, f"Error processing {filepath.name}: {str(e)}"

def create_distribution_plot(tensor_data, data_format, filename, output_path):
    """
    Create distribution plot with representable values overlay.
    
    Args:
        tensor_data (np.array): Flattened tensor data
        data_format (str): Data format identifier
        filename (str): Original filename
        output_path (Path): Output file path
    """
    if data_format not in DATA_TYPE_INFO:
        raise ValueError(f"Unknown data format: {data_format}")
    
    format_info = DATA_TYPE_INFO[data_format]
    
    # Calculate data statistics for dynamic range adjustment
    data_min, data_max = np.min(tensor_data), np.max(tensor_data)
    data_range = data_max - data_min
    
    # Dynamic range calculation - focus on data distribution
    if data_range > 0:
        margin = data_range * 0.15  # 15% margin
        plot_min = data_min - margin
        plot_max = data_max + margin
    else:
        plot_min, plot_max = data_min - 1, data_max + 1
    
    # Create figure - optimized for single-column display in Overleaf
    # Single column width is typically ~3.5 inches
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate histogram
    n_bins = min(200, max(50, int(np.sqrt(len(tensor_data)))))
    counts, bins, patches = plt.hist(tensor_data, bins=n_bins, alpha=0.7, 
                                   color=format_info['color'], density=True,
                                   label=f'Tensor Values (n={len(tensor_data):,})')
    
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
    laplace_loc = np.median(tensor_data)
    laplace_scale = np.mean(np.abs(tensor_data - laplace_loc))
    
    # Generate x values for Laplace fit curve
    x_fit = np.linspace(plot_min, plot_max, 1000)
    laplace_pdf = laplace.pdf(x_fit, loc=laplace_loc, scale=laplace_scale)
    
    # Plot Laplace fit curve as gray dashed line
    plt.plot(x_fit, laplace_pdf, color='gray', linestyle='--', linewidth=1.2, 
             alpha=0.8, label='Laplace Fit', zorder=5)
    
    # Add zero reference line
    if plot_min < 0 < plot_max:
        plt.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='Zero')
    
    # Calculate statistics for return value
    tensor_min = np.min(tensor_data)
    tensor_max = np.max(tensor_data)
    tensor_mean = np.mean(tensor_data)
    tensor_std = np.std(tensor_data)
    tensor_median = np.median(tensor_data)
    
    # Mark minimum and maximum values with symbols
    # Get the maximum y-value from histogram to position markers above the plot
    ax = plt.gca()
    y_max_hist = np.max(counts) if len(counts) > 0 else 1.0
    # Position markers slightly above the histogram maximum
    y_marker_pos = y_max_hist * 1.15
    
    # Add markers for min and max values
    # Use different symbols: 'v' (downward triangle) for min, '^' (upward triangle) for max
    # plt.plot(tensor_min, y_marker_pos, marker='v', markersize=12, color='blue', 
    #          markeredgecolor='darkblue', markeredgewidth=2, zorder=10, label='Min')
    # plt.plot(tensor_max, y_marker_pos, marker='^', markersize=12, color='red', 
    #          markeredgecolor='darkred', markeredgewidth=2, zorder=10, label='Max')
    
    # Set labels and title - smaller fonts for single-column display
    plt.xlabel('Value', fontsize=9)
    plt.ylabel('Density', fontsize=9)
    # plt.title(f'Tensor Value Distribution vs {data_format.upper()} Representable Values\n'
    #          f'File: {filename}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend - smaller font and linewidth for single-column display
    legend_elements = [
        plt.Line2D([0], [0], color=format_info['color'], alpha=0.7, linewidth=2.5, label='Tensor Values'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label='Laplace Fit'),
        # plt.Line2D([0], [0], marker='v', markersize=10, color='blue', markeredgecolor='darkblue', 
        #            markeredgewidth=2, linestyle='None', label='Min'),
        # plt.Line2D([0], [0], marker='^', markersize=10, color='red', markeredgecolor='darkred', 
        #            markeredgewidth=2, linestyle='None', label='Max')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Improve layout
    plt.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)
    
    # Adjust tick label font size
    ax = plt.gca()
    ax.tick_params(labelsize=8)
    
    # Save plot
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Distribution plot saved to: {output_path}")
    
    # Return analysis summary
    return {
        'filename': filename,
        'data_format': data_format,
        'total_elements': len(tensor_data),
        'value_range': [tensor_min, tensor_max],
        'mean_std': [tensor_mean, tensor_std],
        'median': tensor_median,
        'laplace_loc': laplace_loc,
        'laplace_scale': laplace_scale,
        'representable_values_shown': len(visible_rep_values) if format_info['representable_values'] is not None else 0
    }

def main():
    """Main function for distribution analysis."""
    parser = argparse.ArgumentParser(description='Generate tensor value distribution plots with representable values overlay')
    parser.add_argument('input_path', help='Path to tensor file (.pt) or directory containing .pt files')
    parser.add_argument('--output-dir', default='./draw/distribution_tensor/', 
                        help='Output directory for plots (default: ./draw/distribution_tensor/)')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of worker threads for parallel processing (default: 32)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Print detailed statistics to console')
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single file processing
        print(f"Analyzing tensor distribution: {input_path.name}")
        print("=" * 60)
        
        # Load and process tensor
        tensor_data, data_format, filename = load_and_process_tensor(str(input_path))
        
        if tensor_data is None:
            print("Failed to load or process tensor file.")
            return 1
        
        print(f"Data format detected: {data_format.upper()}")
        print(f"Tensor elements: {len(tensor_data):,}")
        print(f"Value range: [{np.min(tensor_data):.6f}, {np.max(tensor_data):.6f}]")
        
        # Generate output filename
        output_filename = input_path.stem + '.pdf'
        output_path = output_dir / output_filename
        
        # Create distribution plot
        analysis_summary = create_distribution_plot(tensor_data, data_format, filename, output_path)
        
        # Print summary statistics if requested
        if args.show_stats:
            print("\nDetailed Analysis Summary:")
            print("-" * 40)
            print(f"Data Format: {analysis_summary['data_format'].upper()}")
            print(f"Total Elements: {analysis_summary['total_elements']:,}")
            print(f"Value Range: [{analysis_summary['value_range'][0]:.6f}, {analysis_summary['value_range'][1]:.6f}]")
            print(f"Mean ± Std: {analysis_summary['mean_std'][0]:.6f} ± {analysis_summary['mean_std'][1]:.6f}")
            print(f"Median: {analysis_summary['median']:.6f}")
            print(f"Laplace Fit - Location (μ): {analysis_summary['laplace_loc']:.6f}")
            print(f"Laplace Fit - Scale (b): {analysis_summary['laplace_scale']:.6f}")
            print(f"Representable Values Shown: {analysis_summary['representable_values_shown']}")
        
        print(f"\nVisualization complete!")
        print(f"Plot saved to: {output_path}")
        
    elif input_path.is_dir():
        # Directory processing with multithreading
        pt_files = list(input_path.glob('*.pt'))
        if len(pt_files) == 0:
            print(f"Error: No .pt files found in directory: {input_path}")
            return 1
        
        print(f"Found {len(pt_files)} tensor files in {input_path}")
        print("=" * 60)
        print(f"Processing {len(pt_files)} files with {args.num_workers} workers...")
        
        successful_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, filepath, output_dir): filepath 
                for filepath in pt_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(pt_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        analysis_summary, error = future.result()
                        if analysis_summary is not None:
                            successful_count += 1
                            if args.show_stats:
                                print(f"\n{filepath.name}:")
                                print(f"  Data Format: {analysis_summary['data_format'].upper()}")
                                print(f"  Total Elements: {analysis_summary['total_elements']:,}")
                                print(f"  Value Range: [{analysis_summary['value_range'][0]:.6f}, {analysis_summary['value_range'][1]:.6f}]")
                        else:
                            failed_count += 1
                            if error:
                                print(f"\nError: {error}")
                    except Exception as e:
                        failed_count += 1
                        print(f"\nError processing {filepath.name}: {e}")
                    finally:
                        pbar.update(1)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful_count}/{len(pt_files)}")
        if failed_count > 0:
            print(f"Failed: {failed_count}")
        print(f"Plots saved to: {output_dir}")
    else:
        print(f"Error: Input path is neither a file nor a directory: {input_path}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
