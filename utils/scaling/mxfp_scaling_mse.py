import torch
from enum import Enum, IntEnum
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import os
import logging
from datetime import datetime

# Add the parent directory to path to import mxfp module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

# Enum for scalar data formats
class ElemFormat(Enum):
    int8 = 1
    int4 = 2
    int2 = 3
    fp8_e5m2 = 4
    fp8_e4m3 = 5
    fp6_e3m2 = 6
    fp6_e2m3 = 7
    fp4 = 8
    fp4_e2m1 = 8
    float16 = 9
    fp16 = 9
    bfloat16 = 10
    bf16 = 10

    @staticmethod
    def from_str(s):
        assert(s != None), "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)

def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin

def _get_max_norm(ebits, mbits):
    """ Valid only for floats that define NaN """
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    emax = 0 if ebits==0 else 2**(ebits - 1) - 1
    return 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)

_FORMAT_CACHE = {}

def _get_format_params(fmt):
    """ Allowed formats:
        - intX:         2 <= X <= 32, assume sign-magnitude, 1.xxx representation
        - floatX/fpX:   16 <= X <= 28, assume top exp is used for NaN/Inf
        - bfloatX/bfX:  9 <= X <= 32
        - fp4,                  no NaN/Inf
        - fp6_e3m2/e2m3,        no NaN/Inf
        - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior
        Returns:
          ebits: exponent bits
          mbits: mantissa bits: includes sign and implicit bits
          emax: max normal exponent
          max_norm: max normal number
          min_norm: min normal number
    """
    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)
    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2**(ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % fmt)

    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)
    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)
    return ebits, mbits, emax, max_norm, min_norm

def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)

def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)

def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """
    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A

def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))
        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)
    out = _round_mantissa(out, bits, round, clamp=False)
    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    if A_is_sparse:
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)
    return out

def _shared_exponents(A, method="max", axes=None, ebits=0, elem_format='fp8_e5m2', minus_exp=None, heuristic_level=None):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
      heuristic_level {int}   -- Level of heuristic to use for 'auto' minus_exp (0-3).
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """
    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    if minus_exp is not None:
        shared_exp = torch.ceil(
            torch.log2(
                shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
            )
        )
        shared_exp = shared_exp - minus_exp
    else:
        shared_exp = torch.floor(
            torch.log2(
                shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
            )
        )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        shared_exp[shared_exp > emax] = float("NaN")
        shared_exp[shared_exp < -emax] = -emax
    return shared_exp

def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]
    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)
    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

def _quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    minus_exp=None,
    heuristic_level=None
):
    """Function used for MX* quantization
    """
    if elem_format == None:
        return A

    assert(scale_bits > 0)
    if axes is None:
        axes = []
    else:
        axes = [axes] if type(axes) == int else axes
        axes = [x + A.ndim if x < 0 else x for x in axes]

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    if block_size > 0:
        A_reshaped, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )
    else:
        A_reshaped = A

    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(
        A_reshaped, method=shared_exp_method, axes=shared_exp_axes, ebits=0,elem_format=elem_format, 
        minus_exp=0, heuristic_level=heuristic_level,
    )

    if flush_fp32_subnorms:
        A_reshaped = A_reshaped * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    q_A = A_reshaped / (2**shared_exp)
    q_A = _quantize_elemwise_core(
        q_A, mbits, ebits, max_norm, round='nearest',
        allow_denorm=True, saturate_normals=True
    )
    
    # Undo scaling (same as _quantize_mx line 437)
    q_A = q_A * (2**shared_exp)
    
    # Calculate MSE per block with A_reshaped and q_A
    if block_size > 0:
        # Calculate squared error
        squared_error = (A_reshaped - q_A) ** 2
        # Calculate MSE per block by averaging over block_size dimensions
        # Use shared_exp_axes which correspond to the block dimensions
        # (shared_exp_axes = [x + 1 for x in axes] when block_size > 0)
        mse_per_block = squared_error
        for axis in shared_exp_axes:
            mse_per_block = torch.mean(mse_per_block, dim=axis, keepdim=True)
        # Calculate overall MSE from per-block MSE
        mse_scale = torch.mean(mse_per_block).item()
    else:
        # No blocks, calculate overall MSE
        mse_scale = torch.mean((A_reshaped - q_A) ** 2).item()
    
    # ========== Test with scale_exp - 1 (half_scale) ==========
    # Create half_scale shared_exp by subtracting 1
    shared_exp_half = shared_exp.clone() - 1
    
    # Clamp again to ensure it's within range
    shared_exp_half[shared_exp_half > scale_emax] = float("NaN")
    shared_exp_half[shared_exp_half < -scale_emax] = -scale_emax
    
    # Apply scaling with half_scale
    q_A_half = A_reshaped / (2**shared_exp_half)
    
    # Quantize element-wise
    q_A_half = _quantize_elemwise_core(
        q_A_half, mbits, ebits, max_norm, round='nearest',
        allow_denorm=True, saturate_normals=True
    )
    
    # Undo scaling
    q_A_half = q_A_half * (2**shared_exp_half)
    
    # Calculate MSE per block with A_reshaped and q_A_half
    if block_size > 0:
        # Calculate squared error
        squared_error_half = (A_reshaped - q_A_half) ** 2
        # Calculate MSE per block by averaging over block_size dimensions
        mse_per_block_half = squared_error_half
        for axis in shared_exp_axes:
            mse_per_block_half = torch.mean(mse_per_block_half, dim=axis, keepdim=True)
        # Calculate overall MSE from per-block MSE
        mse_half_scale = torch.mean(mse_per_block_half).item()
    else:
        # No blocks, calculate overall MSE
        mse_half_scale = torch.mean((A_reshaped - q_A_half) ** 2).item()
        mse_per_block = None
        mse_per_block_half = None
    
    # select per block mse_per_block and mse_per_block_half to get best minus_exp
    best_minus_exp_per_block = None
    best_shared_exp = None
    best_mse = None
    blocks_using_half_scale = 0
    total_blocks = 0
    
    if block_size > 0 and mse_per_block is not None and mse_per_block_half is not None:
        # Compare MSE per block and select the better one
        # mse_per_block_half < mse_per_block means half_scale is better (minus_exp = -1)
        # Otherwise, original scale is better (minus_exp = 0)
        better_is_half = mse_per_block_half < mse_per_block
        
        # Create best_minus_exp_per_block: -1 for blocks where half_scale is better, 0 otherwise
        best_minus_exp_per_block = torch.where(better_is_half, 
                                                torch.full_like(mse_per_block, 1.0), 
                                                torch.zeros_like(mse_per_block))
        
        # Create best_shared_exp by selecting shared_exp or shared_exp_half for each block
        best_shared_exp = torch.where(better_is_half, shared_exp_half, shared_exp)
        
        # Calculate best MSE per block
        best_mse = torch.where(better_is_half, mse_per_block_half, mse_per_block)
        
        # Count statistics
        blocks_using_half_scale = torch.sum(better_is_half).item()
        total_blocks = torch.numel(better_is_half)
        
        # Calculate overall best MSE
        best_mse_overall = torch.mean(best_mse).item()
    else:
        # No blocks, compare overall MSE
        if mse_half_scale < mse_scale:
            best_minus_exp = 1
            best_mse_overall = mse_half_scale
            best_shared_exp = shared_exp_half
        else:
            best_minus_exp = 0
            best_mse_overall = mse_scale
            best_shared_exp = shared_exp
        best_minus_exp_per_block = best_minus_exp
        best_mse = torch.tensor(best_mse_overall)  # Convert to tensor for consistency
        blocks_using_half_scale = 1 if mse_half_scale < mse_scale else 0
        total_blocks = 1
    
    return {
        'mse_scale': mse_scale,
        'mse_half_scale': mse_half_scale,
        'mse_per_block': mse_per_block,
        'mse_per_block_half': mse_per_block_half,
        'best_minus_exp_per_block': best_minus_exp_per_block,
        'best_shared_exp': best_shared_exp,
        'best_mse': best_mse,
        'best_mse_overall': best_mse_overall,
        'blocks_using_half_scale': blocks_using_half_scale,
        'total_blocks': total_blocks,
        'half_scale_ratio': blocks_using_half_scale / total_blocks if total_blocks > 0 else 0.0,
    }



def main():
    """Main function for MXFP scaling test."""
    parser = argparse.ArgumentParser(description='Test different scaling strategies for MXFP quantization')
    parser.add_argument('--input_tensor', default='data/bf16/20250923_100142_0001_iter000_linear_L1_forward_pre_linear_bf16_rank00_group000_input.pt', help='Path to input tensor file (.pt)')
    parser.add_argument('--elem-format', default='fp4_e2m1', 
                        choices=['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'fp6_e3m2', 'fp6_e2m3'],
                        help='Element format for quantization (default: fp8_e4m3)')
    parser.add_argument('--scale-bits', type=int, default=8,
                        help='Number of scale bits (default: 8)')
    parser.add_argument('--max-scale-exp', type=int, default=10,
                        help='Maximum scale exponent (default: auto-calculated from tensor max if using default value)')
    parser.add_argument('--min-scale-exp', type=int, default=-10,
                        help='Minimum scale exponent (default: auto-calculated from tensor min if using default value)')
    parser.add_argument('--block-size', type=int, default=32,
                        help='Block size for tiling (default: 32, use 0 for no tiling)')
    parser.add_argument('--axes', type=int, default=-1,
                        help='Axes for shared exponent calculation (default: -1)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()

    try:
        # Load tensor
        input_tensor = torch.load(args.input_tensor, map_location='cpu')['tensor']
        
        if not isinstance(input_tensor, torch.Tensor):
            print(f"❌ Error: File does not contain a torch.Tensor")
            return 1
        
        print(f"Tensor shape: {input_tensor.shape}")
        print(f"Tensor dtype: {input_tensor.dtype}")
        print(f"Tensor range: [{input_tensor.min().item():.6f}, {input_tensor.max().item():.6f}]")
        
        # Get format parameters
        ebits, mbits, emax, max_norm, _ = _get_format_params(args.elem_format)
        print(f"\nElement format: {args.elem_format}")
        print(f"  ebits: {ebits}, mbits: {mbits}, max_norm: {max_norm}")
        
        # Convert axes to list if needed
        axes = args.axes if isinstance(args.axes, list) else [args.axes] if args.axes is not None else None
        
        # Determine scale_exp to test
        # Use max_scale_exp as the primary scale_exp, and test scale_exp - 1 as half_scale
        scale_exp = args.max_scale_exp
        
        print(f"\nTesting scale_exp = {scale_exp} vs scale_exp - 1 = {scale_exp - 1}")
        print("=" * 80)
        
        result = _quantize_mx(
            A = input_tensor,
            scale_bits=args.scale_bits,
            elem_format=args.elem_format,
            shared_exp_method="max",
            axes=axes,
            block_size=args.block_size,
            round="nearest",
            flush_fp32_subnorms=False,
            minus_exp=None,
            heuristic_level=None
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  MSE (scale_exp={scale_exp}):     {result['mse_scale']:.6e}")
        print(f"  MSE (scale_exp={scale_exp-1}):  {result['mse_half_scale']:.6e}")
        print(f"  Best MSE (per-block selection): {result['best_mse_overall']:.6e}")
        
        if result['mse_scale'] > 0:
            improvement = ((result['mse_scale'] - result['mse_half_scale']) / result['mse_scale'] * 100)
            print(f"  Improvement (half vs original): {improvement:.2f}%")
            
            best_improvement = ((result['mse_scale'] - result['best_mse_overall']) / result['mse_scale'] * 100)
            print(f"  Improvement (best vs original): {best_improvement:.2f}%")
            
            if result['mse_half_scale'] < result['mse_scale']:
                print(f"  ✅ Half scale (scale_exp - 1) has better overall MSE")
            else:
                print(f"  ✅ Original scale has better overall MSE")
        
        # Print per-block statistics
        if args.block_size > 0 and result['total_blocks'] > 0:
            print(f"\nPer-block selection statistics:")
            print(f"  Total blocks: {result['total_blocks']}")
            print(f"  Blocks using half_scale (minus_exp=-1): {result['blocks_using_half_scale']}")
            print(f"  Blocks using original scale (minus_exp=0): {result['total_blocks'] - result['blocks_using_half_scale']}")
            print(f"  Half scale ratio: {result['half_scale_ratio']*100:.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
     exit(main())