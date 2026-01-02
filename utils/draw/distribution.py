#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import laplace

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
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
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
            plt.axvline(val, color='red', alpha=0.6, linewidth=1.0, zorder=3)
    
    # Fit Laplace distribution
    # Laplace distribution: f(x) = (1/(2b)) * exp(-|x - μ|/b)
    # where μ is location (median) and b is scale parameter
    laplace_loc = np.median(tensor_data)
    laplace_scale = np.mean(np.abs(tensor_data - laplace_loc))
    
    # Generate x values for Laplace fit curve
    x_fit = np.linspace(plot_min, plot_max, 1000)
    laplace_pdf = laplace.pdf(x_fit, loc=laplace_loc, scale=laplace_scale)
    
    # Plot Laplace fit curve as gray dashed line
    plt.plot(x_fit, laplace_pdf, color='gray', linestyle='--', linewidth=2, 
             alpha=0.8, label='Laplace Fit', zorder=5)
    
    # Add zero reference line
    if plot_min < 0 < plot_max:
        plt.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Zero')
    
    # Calculate and display statistics
    tensor_min = np.min(tensor_data)
    tensor_max = np.max(tensor_data)
    tensor_mean = np.mean(tensor_data)
    tensor_std = np.std(tensor_data)
    tensor_median = np.median(tensor_data)
    
    # Add statistics text box
    stats_text = (
        f'Data Format: {data_format.upper()} ({format_info["description"]})\n'
        f'Tensor Shape: {tensor_data.shape if hasattr(tensor_data, "shape") else "Flattened"}\n'
        f'Total Elements: {len(tensor_data):,}\n\n'
        f'Value Statistics:\n'
        f'  Min: {tensor_min:.6f}\n'
        f'  Max: {tensor_max:.6f}\n'
        f'  Mean: {tensor_mean:.6f}\n'
        f'  Median: {tensor_median:.6f}\n'
        f'  Std: {tensor_std:.6f}\n\n'
        f'Laplace Fit:\n'
        f'  Location (μ): {laplace_loc:.6f}\n'
        f'  Scale (b): {laplace_scale:.6f}'
    )
    
    # Position stats box
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Set labels and title
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Tensor Value Distribution vs {data_format.upper()} Representable Values\n'
             f'File: {filename}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=format_info['color'], alpha=0.7, linewidth=5, label='Tensor Values'),
        plt.Line2D([0], [0], color='red', alpha=0.6, linewidth=1, label='Representable Values'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Laplace Fit')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Improve layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    parser.add_argument('input_file', help='Path to tensor file (.pt)')
    parser.add_argument('--output-dir', default='./draw/distribution_tensor/', 
                        help='Output directory for plots (default: ./draw/distribution_tensor/)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Print detailed statistics to console')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        return 1
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename: same as input but with .png extension
    output_filename = input_path.stem + '.png'
    output_path = output_dir / output_filename
    
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
    
    return 0

if __name__ == "__main__":
    exit(main())
