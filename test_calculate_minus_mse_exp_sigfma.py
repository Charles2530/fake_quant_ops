#!/usr/bin/env python3
"""
Test script for calculate_minus_mse_exp_sigfma function.
Usage: python test_calculate_minus_mse_exp_sigfma.py [tensor_file_path]
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant.ops.mxfp import calculate_minus_mse_exp_sigfma

def load_tensor(file_path):
    """Load tensor from .pt file"""
    print(f"Loading tensor from: {file_path}")
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Handle different tensor file formats
        if isinstance(data, dict):
            if 'tensor' in data:
                tensor = data['tensor']
            else:
                # Take the first tensor value
                tensor = list(data.values())[0]
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")
        
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}")
        print(f"Tensor mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        raise

def test_calculate_minus_mse_exp_sigfma(tensor, block_size=16, axes=-1, elem_format='fp4_e2m1'):
    """Test the calculate_minus_mse_exp_sigfma function"""
    print("\n" + "="*80)
    print("Testing calculate_minus_mse_exp_sigfma")
    print("="*80)
    
    # Test parameters
    scale_bits = 8
    shared_exp_method = "max"
    minus_level = 1.0
    
    print(f"\nTest Configuration:")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Block size: {block_size}")
    print(f"  Axes: {axes}")
    print(f"  Scale bits: {scale_bits}")
    print(f"  Element format: {elem_format}")
    print(f"  Shared exp method: {shared_exp_method}")
    print(f"  Minus level: {minus_level}")
    
    # Calculate minus_exp
    try:
        result = calculate_minus_mse_exp_sigfma(
            A=tensor,
            scale_bits=scale_bits,
            elem_format=elem_format,
            shared_exp_method=shared_exp_method,
            axes=axes,
            block_size=block_size,
            round="nearest",
            flush_fp32_subnorms=False,
            minus_level=minus_level,
        )
        
        print(f"\nResult:")
        if isinstance(result, torch.Tensor):
            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")
            print(f"  Result min: {result.min().item():.6f}, max: {result.max().item():.6f}")
            print(f"  Result mean: {result.mean().item():.6f}")
            
            # Count how many blocks use half scale vs original scale
            num_blocks = result.numel()
            num_half_scale = (result == minus_level).sum().item()
            num_original_scale = (result == 0.0).sum().item()
            
            print(f"\nBlock Statistics:")
            print(f"  Total blocks: {num_blocks}")
            print(f"  Blocks using half scale (minus_level={minus_level}): {num_half_scale} ({100*num_half_scale/num_blocks:.2f}%)")
            print(f"  Blocks using original scale (0): {num_original_scale} ({100*num_original_scale/num_blocks:.2f}%)")
        else:
            print(f"  Result (scalar): {result}")
        
        return result
        
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Default tensor file path
    default_tensor_path = "data/real/fwd_in_140541200228864.pt"
    
    # Get tensor file path from command line or use default
    if len(sys.argv) > 1:
        tensor_path = sys.argv[1]
    else:
        tensor_path = default_tensor_path
    
    # Check if file exists
    if not os.path.exists(tensor_path):
        print(f"Error: Tensor file not found: {tensor_path}")
        print(f"\nUsage: python {sys.argv[0]} [tensor_file_path]")
        print(f"Example: python {sys.argv[0]} {default_tensor_path}")
        sys.exit(1)
    
    # Load tensor
    tensor = load_tensor(tensor_path)
    
    print("\n" + "="*80)
    print("Test 2: With block_size=32, axes=-1, fp4_e2m1")
    print("="*80)
    result2 = test_calculate_minus_mse_exp_sigfma(tensor, block_size=32, axes=-1, elem_format='fp4_e2m1')
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)

if __name__ == "__main__":
    main()
