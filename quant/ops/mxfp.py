import torch
from enum import Enum, IntEnum
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        if isinstance(minus_exp, str) and "auto" in minus_exp:
            if minus_exp == "auto":
                minus_exp_result = calculate_minus_mse_exp(
                    A, scale_bits=8, elem_format=elem_format, 
                    shared_exp_method=method, axes=axes, block_size=32, 
                    round="nearest", flush_fp32_subnorms=False,minus_level=1.0
                )
            elif minus_exp == "auto-reverse":
                minus_exp_result = calculate_minus_mse_exp(
                    A, scale_bits=8, elem_format=elem_format, 
                    shared_exp_method=method, axes=axes, block_size=32, 
                    round="nearest", flush_fp32_subnorms=False, minus_level=-1.0
                )
            elif minus_exp == "auto-2":
                minus_exp_result = calculate_minus_mse_exp(
                    A, scale_bits=8, elem_format=elem_format, 
                    shared_exp_method=method, axes=axes, block_size=32, 
                    round="nearest", flush_fp32_subnorms=False, minus_level=2.0
                )
            # Ensure minus_exp has the same shape as shared_exp
            if isinstance(minus_exp_result, torch.Tensor):
                # If shapes don't match, try to reshape
                if minus_exp_result.shape != shared_exp.shape:
                    # Try to squeeze extra trailing dimensions of size 1
                    while (minus_exp_result.ndim > shared_exp.ndim and 
                        minus_exp_result.shape[-1] == 1):
                        minus_exp_result = minus_exp_result.squeeze(-1)
                    # If still don't match, try to view or broadcast
                    if minus_exp_result.shape != shared_exp.shape:
                        try:
                            minus_exp_result = minus_exp_result.view(shared_exp.shape)
                        except:
                            # If view fails, try broadcast_to
                            minus_exp_result = torch.broadcast_to(minus_exp_result, shared_exp.shape).clone()
                minus_exp = minus_exp_result
            else:
                # Scalar value, create tensor with shared_exp shape
                minus_exp = torch.full(shared_exp.shape, minus_exp_result, dtype=shared_exp.dtype, device=shared_exp.device)
            # print(f"minus_exp is auto, calculated per-block with heuristic level {heuristic_level}.")
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
        minus_exp=minus_exp, heuristic_level=heuristic_level,
    )

    if flush_fp32_subnorms:
        A_reshaped = A_reshaped * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A_q = A_reshaped / (2**shared_exp)
    A_q = _quantize_elemwise_core(
            A_q, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True)
    A_q = A_q * (2**shared_exp)

    if block_size > 0:
        A_q = _undo_reshape_to_blocks(A_q, padded_shape, orig_shape, axes)

    return A_q
import torch
from torch.autograd import Function

class MXFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                elem_format: str = 'fp8_e5m2', block_size: int = 32, minus_exp=None, heuristic_level=3):
        ctx.save_for_backward(A, B)
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        ctx.minus_exp = minus_exp
        ctx.heuristic_level = heuristic_level
        
        A_q = _quantize_mx(
            A, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-1, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False, minus_exp=minus_exp, heuristic_level=heuristic_level
        )
        B_q = _quantize_mx(
            B, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-2, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False, minus_exp=minus_exp, heuristic_level=heuristic_level
        )
        return torch.matmul(A_q, B_q)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        A_q = _quantize_mx(
            A, scale_bits=8, elem_format=ctx.elem_format,
            shared_exp_method="max", axes=-1, block_size=ctx.block_size,
            round="nearest", flush_fp32_subnorms=False, minus_exp=ctx.minus_exp, heuristic_level=ctx.heuristic_level
        )
        B_q = _quantize_mx(
            B, scale_bits=8, elem_format=ctx.elem_format,
            shared_exp_method="max", axes=-2, block_size=ctx.block_size,
            round="nearest", flush_fp32_subnorms=False, minus_exp=ctx.minus_exp, heuristic_level=ctx.heuristic_level
        )
        grad_output_q = _quantize_mx(
            grad_output, scale_bits=8, elem_format=ctx.elem_format,
            shared_exp_method="max", axes=-1, block_size=ctx.block_size,
            round="nearest", flush_fp32_subnorms=False, minus_exp=ctx.minus_exp, heuristic_level=ctx.heuristic_level
        )
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output_q, B_q.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A_q.transpose(-2, -1), grad_output_q)
        return grad_A, grad_B, None, None, None, None

def mxfp_matmul(A, B, elem_format='fp8_e5m2', block_size=32, minus_exp=None, heuristic_level=3):
    return MXFPMatMul.apply(A, B, elem_format, block_size, minus_exp, heuristic_level)

def _get_kappa(elem_format: str, distribution: str='gaussian',miu:float=0.0,gamma:float=1.0) -> float:
    # if elem_format == 'fp4_e2m1' and distribution == 'gaussian':
    #     kappa = 0.020974709594546316
    # else:
    from quant.ops.utils import _calculate_kappa
    kappa = _calculate_kappa(elem_format, distribution, miu, gamma)
    return kappa

def calculate_minus_mse_exp(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    minus_level:float=1.0,
):
    """
    Calculate optimal minus_exp based on MSE comparison between minus_exp=0 and minus_exp=-1.
    
    This function quantizes the input tensor with two different minus_exp values (0 and -1),
    compares their MSE per block, and returns the optimal minus_exp for each block.
    
    Args:
        A: Input tensor to quantize
        scale_bits: Number of bits for scale
        elem_format: Element format (e.g., 'fp8_e5m2', 'fp4_e2m1')
        shared_exp_method: Method for shared exponent calculation ("max" or "none")
        axes: Axes along which to calculate shared exponents
        block_size: Block size for block-wise quantization (0 means no blocking)
        round: Rounding method
        flush_fp32_subnorms: Whether to flush FP32 subnormals
    
    Returns:
        best_minus_exp_per_block: Tensor with optimal minus_exp for each block (-1 or 0)
                                 If block_size=0, returns a scalar (0 or -1)
    """
    if elem_format == None:
        # If no quantization format, return 0 (no minus_exp adjustment needed)
        return 0
    
    assert(scale_bits > 0)
    if axes is None:
        axes = []
    else:
        axes = [axes] if type(axes) == int else axes
        axes = [x + A.ndim if x < 0 else x for x in axes]
    
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)
    
    # Check if axes are valid for current A shape
    # If axes contain values >= A.ndim, it means A is already reshaped and axes are already adjusted
    if block_size > 0 and axes:
        axes_valid = all(0 <= ax < A.ndim for ax in axes)
        if axes_valid:
            # A is not reshaped yet, do reshape
            try:
                A_reshaped, axes_reshaped, orig_shape, padded_shape = _reshape_to_blocks(
                    A, axes, block_size
                )
                shared_exp_axes = [x + 1 for x in axes_reshaped]
            except (IndexError, AssertionError):
                # If reshape fails (e.g., axes out of range), assume A is already reshaped
                A_reshaped = A
                shared_exp_axes = axes
        else:
            # Axes are out of range, assume A is already reshaped
            A_reshaped = A
            shared_exp_axes = axes
    else:
        A_reshaped = A
        shared_exp_axes = axes if axes is not None else []
    
    # Calculate shared_exp with minus_exp=0
    shared_exp = _shared_exponents(
        A_reshaped, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
        elem_format=elem_format, minus_exp=0, heuristic_level=None,
    )
    
    if flush_fp32_subnorms:
        A_reshaped = A_reshaped * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)
    
    shared_exp = shared_exp - emax
    
    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    
    # Quantize with minus_exp=0 (original scale)
    q_A = A_reshaped / (2**shared_exp)
    q_A = _quantize_elemwise_core(
        q_A, mbits, ebits, max_norm, round=round,
        allow_denorm=True, saturate_normals=True
    )
    q_A = q_A * (2**shared_exp)
    
    # Calculate MSE per block with minus_exp=0
    if block_size > 0:
        squared_error = (A_reshaped - q_A) ** 2
        mse_per_block = squared_error
        for axis in shared_exp_axes:
            mse_per_block = torch.mean(mse_per_block, dim=axis, keepdim=True)
    else:
        mse_scale = torch.mean((A_reshaped - q_A) ** 2).item()
        mse_per_block = None
    
    # ========== Test with scale_exp - 1 (minus_exp = -1) ==========
    # Create half_scale shared_exp by subtracting 1
    shared_exp_half = shared_exp.clone() - minus_level
    
    # Clamp again to ensure it's within range
    shared_exp_half[shared_exp_half > scale_emax] = float("NaN")
    shared_exp_half[shared_exp_half < -scale_emax] = -scale_emax
    
    # Apply scaling with half_scale
    q_A_half = A_reshaped / (2**shared_exp_half)
    
    # Quantize element-wise
    q_A_half = _quantize_elemwise_core(
        q_A_half, mbits, ebits, max_norm, round=round,
        allow_denorm=True, saturate_normals=True
    )
    
    # Undo scaling
    q_A_half = q_A_half * (2**shared_exp_half)
    
    # Calculate MSE per block with minus_exp=-1
    if block_size > 0:
        squared_error_half = (A_reshaped - q_A_half) ** 2
        mse_per_block_half = squared_error_half
        for axis in shared_exp_axes:
            mse_per_block_half = torch.mean(mse_per_block_half, dim=axis, keepdim=True)
    else:
        mse_half_scale = torch.mean((A_reshaped - q_A_half) ** 2).item()
        mse_per_block_half = None
    
    # Select best minus_exp per block based on MSE comparison
    if block_size > 0 and mse_per_block is not None and mse_per_block_half is not None:
        # Compare MSE per block and select the better one
        # mse_per_block_half < mse_per_block means minus_exp=-1 is better
        # Otherwise, minus_exp=0 is better
        better_is_half = mse_per_block_half < mse_per_block
        
        # Create best_minus_exp_per_block with the same shape as shared_exp
        # Ensure better_is_half has the same shape as shared_exp
        if better_is_half.shape != shared_exp.shape:
            # Try to match shapes by squeezing extra trailing dimensions
            if better_is_half.ndim > shared_exp.ndim:
                while (better_is_half.ndim > shared_exp.ndim and 
                       better_is_half.shape[-1] == 1):
                    better_is_half = better_is_half.squeeze(-1)
            # If still don't match, try broadcast
            if better_is_half.shape != shared_exp.shape:
                better_is_half = torch.broadcast_to(better_is_half, shared_exp.shape).clone()
        
        best_minus_exp_per_block = torch.where(
            better_is_half,
            torch.full_like(shared_exp, minus_level, dtype=torch.float),
            torch.zeros_like(shared_exp)
        )
    else:
        # No blocks, compare overall MSE
        if mse_half_scale < mse_scale:
            best_minus_exp = minus_level
        else:
            best_minus_exp = 0
        # Return scalar value when block_size=0, similar to reference code
        best_minus_exp_per_block = best_minus_exp
    
    return best_minus_exp_per_block

def calculate_minus_exp(
    tensor_block: torch.Tensor,
    elem_format: str = 'fp8_e4m3',
    distribution: str = 'gaussian',
    axis: int = -1,
    heuristic_level: int = 3
) -> torch.Tensor:
    """
    Calculates the 'minus_exp' for scaling per-block along a specified axis,
    incorporating a multi-stage heuristic with controllable depth.
    """
    x_abs = torch.abs(tensor_block.to(torch.float32))
    mu_aggressive = torch.mean(x_abs, dim=axis, keepdim=True)
    gamma = torch.std(x_abs, dim=axis, keepdim=True)

    kappa = _get_kappa(elem_format, distribution,mu_aggressive,gamma)
    if tensor_block.numel() == 0:
        return torch.tensor(0, dtype=torch.int, device=tensor_block.device)

    # Use float32 for calculations to avoid overflow and precision issues
    epsilon = 1e-9

    # --- Pre-calculate statistics needed for aggressive scaling ---
    V_max_aggressive, _ = torch.max(x_abs, dim=axis, keepdim=True)
    
    # --- Heuristic Decision Logic ---
    if heuristic_level == 0:
        # No heuristic, always proceed with aggressive scaling for non-zero blocks
        proceed_mask = V_max_aggressive >= epsilon
    else:
        # Initialize masks
        V_max = V_max_aggressive
        proceed_mask = torch.zeros_like(V_max, dtype=torch.bool)
        undecided_mask = V_max >= epsilon # Only non-zero blocks are candidates

        # --- Stage 1: Pre-check via V_max Threshold ---
        if heuristic_level >= 1:
            V_th = (2.0 / kappa)**(1/3.0)
            s1_pass = (V_max > V_th) & undecided_mask
            proceed_mask[s1_pass] = True
            undecided_mask[s1_pass] = False

        # --- Stage 2: Quick Rejection via Upper Bound ---
        if heuristic_level >= 2:
            # Only calculate for blocks that are still undecided
            undecided_indices = torch.where(undecided_mask)
            if undecided_indices[0].numel() > 0:
                V_max_s2 = V_max[undecided_indices]
                mu_s2 = mu_aggressive[undecided_indices]
                
                C_vmax_s2 = (2.0 / (7.0 * kappa * V_max_s2 + epsilon)) - (V_max_s2.pow(2) / 7.0)
                n = tensor_block.shape[axis]
                E_upper_s2 = n * mu_s2 * V_max_s2
                
                s2_reject = E_upper_s2 <= C_vmax_s2
                
                # Update the main undecided_mask by setting rejected blocks to False
                temp_undecided = undecided_mask.clone()
                temp_undecided[undecided_indices] = ~s2_reject
                undecided_mask = temp_undecided

        # --- Stage 3: Exact Calculation ---
        if heuristic_level >= 3:
            # Only calculate for blocks that are still undecided
            undecided_indices = torch.where(undecided_mask)
            if undecided_indices[0].numel() > 0:
                V_max_s3 = V_max[undecided_indices]
                # Create a view of x_abs that matches the undecided_mask shape
                mask_shape = undecided_mask.shape
                view_shape = list(x_abs.shape)
                if axis != -1 and axis != len(view_shape) - 1:
                    # Make the axis of reduction the last one for easier indexing
                    x_abs_swapped = x_abs.transpose(axis, -1)
                    view_shape[axis], view_shape[-1] = view_shape[-1], view_shape[axis]
                else:
                    x_abs_swapped = x_abs

                # This is complex to do generally. Let's revert to a slightly less efficient but correct logic:
                # Pre-calculate C_vmax for all undecided blocks and E for all undecided blocks.
                C_vmax = (2.0 / (7.0 * kappa * V_max + epsilon)) - (V_max.pow(2) / 7.0)
                E = torch.sum(x_abs.pow(2), dim=axis, keepdim=True)

                s3_pass = (E > C_vmax) & undecided_mask
                proceed_mask[s3_pass] = True

    # --- Final Decision ---
    # Default minus_exp is 0.
    # import pdb; pdb.set_trace()
    minus_exp_final = torch.zeros_like(V_max_aggressive)
    # Use the aggressive value only for blocks that passed the heuristic.
    minus_exp_final = torch.where(proceed_mask, 1, 0)
    # import pdb; pdb.set_trace()
    # print(f"minus_exp_final: {minus_exp_final}")

    return minus_exp_final


if __name__ == '__main__':
    A = torch.load("data/bf16/20250923_100142_0001_iter000_linear_L1_forward_pre_linear_bf16_rank00_group000_input.pt", map_location='cpu')['tensor'].cuda()
    B = torch.load("data/bf16/20250923_100142_0002_iter000_linear_L1_forward_pre_linear_bf16_rank00_group000_weight.pt", map_location='cpu')['tensor'].cuda() 
    # Test different minus_exp settings including 'auto' with different heuristic levels
    test_settings = [None,0, 1, 'auto']
    # test_settings = [None, 0, 1, ('auto', 1), ('auto', 2), ('auto', 3)]
    block_size_list = [32]
    
    for setting in test_settings:
        for block_size in block_size_list:
            if isinstance(setting, tuple):
                minus_exp, level = setting
            else:
                minus_exp, level = setting, None

            print(f"minus_exp: {minus_exp}, heuristic_level: {level}, block_size: {block_size}")
            
            # Test Matrix A
            mxfp8_A = _quantize_mx(A, scale_bits=8, elem_format='fp4_e2m1', shared_exp_method="max", axes=-1, 
                                   block_size=block_size, round="nearest", flush_fp32_subnorms=False, 
                                   minus_exp=minus_exp, heuristic_level=level)
            loss_A = torch.mean((A - mxfp8_A) ** 2)
            print(f"loss_A: {loss_A}")

            # Test Matrix B
            mxfp8_B = _quantize_mx(B, scale_bits=8, elem_format='fp4_e2m1', shared_exp_method="max", axes=-1, 
                                   block_size=block_size, round="nearest", flush_fp32_subnorms=False, 
                                   minus_exp=minus_exp, heuristic_level=level)
            loss_B = torch.mean((B - mxfp8_B) ** 2)
            print(f"loss_B: {loss_B}")

            # Test MatMul
            C_mxfp8 = mxfp_matmul(A, B, elem_format='fp4_e2m1', block_size=block_size, minus_exp=minus_exp, heuristic_level=level)
            C_bf16 = torch.matmul(A,B).to(torch.bfloat16)
            loss_mxfp = torch.mean((C_bf16 - C_mxfp8) ** 2)
            print(f"loss_mxfp: {loss_mxfp}")
            print("-" * 30)