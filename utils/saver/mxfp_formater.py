import torch
# import torch_npu
from enum import Enum, IntEnum
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Set global font to Times New Roman (or Calibri as fallback) for paper-ready plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Calibri', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math text


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


def _analyze_overflow_underflow_before_quantization(A, elem_format, mbits, ebits, max_norm, verbose=True):
    """
    Analyze tensor for overflow and underflow conditions before quantization.
    This function is called right before element-wise quantization to detect
    potential overflow and underflow issues that might be caused by scaling.
    
    Args:
        A (torch.Tensor): Input tensor after scaling but before quantization
        elem_format (str): Element format identifier
        mbits (int): Number of mantissa bits
        ebits (int): Number of exponent bits
        max_norm (float): Maximum normal value for the format
        verbose (bool): Whether to print analysis results immediately
        
    Returns:
        dict: Analysis results containing overflow and underflow statistics
    """
    analysis_result = {
        'elem_format': elem_format,
        'total_elements': 0,
        'underflow_count': 0,
        'underflow_percent': 0.0,
        'flush_count': 0,
        'flush_percent': 0.0,
        'overflow_count': 0,
        'overflow_percent': 0.0,
        'min_denormal': 0.0,
        'min_norm': 0.0,
        'max_norm': max_norm,
        'tensor_range': [0.0, 0.0],
        'has_significant_underflow': False,
        'has_significant_overflow': False,
        'severity': 'none',  # 'none', 'moderate', 'high'
        'error': None
    }
    
    try:
        # Calculate minimum representable values
        min_norm = _get_min_norm(ebits)
        min_denormal = min_norm / (2 ** (mbits - 2)) if mbits > 2 else min_norm
        
        # Convert to numpy for analysis (handle BFloat16)
        if A.dtype == torch.bfloat16:
            A_float = A.float()
        else:
            A_float = A
            
        # Detach from computation graph if needed, then convert to numpy
        if A_float.is_cuda:
            A_np = A_float.detach().cpu().numpy()
        else:
            A_np = A_float.detach().numpy()
        
        # Handle empty tensors
        if A_np.size == 0:
            analysis_result['total_elements'] = 0
            return analysis_result
        
        # Count underflow conditions
        total_elements = A_np.size
        non_zero_mask = A_np != 0.0
        abs_A = np.abs(A_np)
        
        # Underflow: non-zero values closer to zero than smallest representable
        underflow_mask = non_zero_mask & (abs_A < min_denormal)
        underflow_count = np.sum(underflow_mask)
        underflow_percent = (underflow_count / total_elements) * 100
        
        # Also check for values that would be flushed to zero
        flush_mask = non_zero_mask & (abs_A < min_norm)
        flush_count = np.sum(flush_mask)
        flush_percent = (flush_count / total_elements) * 100
        
        # Check for overflow: values larger than maximum representable
        overflow_mask = abs_A > max_norm
        overflow_count = np.sum(overflow_mask)
        overflow_percent = (overflow_count / total_elements) * 100
        
        # Store analysis results
        analysis_result.update({
            'total_elements': total_elements,
            'underflow_count': int(underflow_count),
            'underflow_percent': float(underflow_percent),
            'flush_count': int(flush_count),
            'flush_percent': float(flush_percent),
            'overflow_count': int(overflow_count),
            'overflow_percent': float(overflow_percent),
            'min_denormal': float(min_denormal),
            'min_norm': float(min_norm),
            'max_norm': float(max_norm),
            'tensor_range': [float(np.min(A_np)), float(np.max(A_np))],
            'has_significant_underflow': underflow_percent > 0.1 or flush_percent > 0.1,
            'has_significant_overflow': overflow_percent > 0.1
        })
        
        # Determine severity based on both overflow and underflow
        max_issue_percent = max(underflow_percent, overflow_percent)
        if max_issue_percent > 1.0:
            analysis_result['severity'] = 'high'
        elif max_issue_percent > 0.1:
            analysis_result['severity'] = 'moderate'
        else:
            analysis_result['severity'] = 'none'
        
        # Print analysis if verbose and significant issues detected
        if verbose and (analysis_result['has_significant_underflow'] or analysis_result['has_significant_overflow']):
            print(f"\n⚠️  OVERFLOW/UNDERFLOW ANALYSIS ({elem_format}):")
            print(f"    Total elements: {total_elements:,}")
            print(f"    Min denormal: {min_denormal:.2e}")
            print(f"    Min normal: {min_norm:.2e}")
            print(f"    Max normal: {max_norm:.2e}")
            print(f"    Underflow count: {underflow_count:,} ({underflow_percent:.2f}%)")
            print(f"    Flush to zero count: {flush_count:,} ({flush_percent:.2f}%)")
            print(f"    Overflow count: {overflow_count:,} ({overflow_percent:.2f}%)")
            print(f"    Tensor range: [{np.min(A_np):.2e}, {np.max(A_np):.2e}]")
            
            if max_issue_percent > 1.0:
                if underflow_percent > overflow_percent:
                    print(f"    🔴 HIGH UNDERFLOW RATE: {underflow_percent:.2f}%")
                else:
                    print(f"    🔴 HIGH OVERFLOW RATE: {overflow_percent:.2f}%")
                print(f"       Consider adjusting scaling strategy!")
            elif max_issue_percent > 0.1:
                if underflow_percent > overflow_percent:
                    print(f"    🟡 MODERATE UNDERFLOW: {underflow_percent:.2f}%")
                else:
                    print(f"    🟡 MODERATE OVERFLOW: {overflow_percent:.2f}%")
            
    except Exception as e:
        # Don't let analysis errors break the quantization process
        analysis_result['error'] = str(e)
        if verbose:
            print(f"Warning: Underflow analysis failed: {str(e)}")
    
    return analysis_result


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

    # handle Inf/NaN
    # out[A == float("Inf")] = float("Inf")
    # out[A == -float("Inf")] = -float("Inf")
    # out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)

    return out


def _shared_exponents(A, method="max", axes=None, ebits=0, scaling_control="max"):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            max_val = torch.max(torch.abs(A))
            if scaling_control == "max_minus_1":
                # Use max - 1 strategy to avoid potential overflow
                shared_exp = max_val - 1.0
            else:  # default "max"
                shared_exp = max_val
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
            if scaling_control == "max_minus_1":
                # Use max - 1 strategy to avoid potential overflow
                shared_exp = shared_exp - 1.0
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
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
        # Don't pad if the axis is short enough to fit inside one tile
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
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
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
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    scaling_control="max",
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    if axes is None:
        axes = []
    else:
        axes = [axes] if type(axes) == int else axes
        axes = [x + A.ndim if x < 0 else x for x in axes]

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0, scaling_control=scaling_control,
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)

    # Add underflow analysis before quantization
    # _analyze_underflow_before_quantization(A, elem_format, mbits, ebits, max_norm)
    
    A = _quantize_elemwise_core(
            A, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True)

    A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A

import torch
from torch.autograd import Function
from typing import Optional, Dict, Any

class MXFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                elem_format: str = 'fp8_e5m2', block_size: int = 32,
                layer_type: Optional[str] = None, layer_idx: Optional[int] = None,
                operation: str = "forward", phase: str = "pre", component: str = "linear",
                rank: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None,
                scaling_control: str = "max"):
        # 保存tensor和参数到ctx
        ctx.save_for_backward(A, B)
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        ctx.layer_type = layer_type
        ctx.layer_idx = layer_idx
        ctx.operation = operation
        ctx.phase = phase
        ctx.component = component
        ctx.rank = rank
        ctx._metadata = metadata
        ctx.scaling_control = scaling_control
        
        # 量化tensor
        A_q = _quantize_mx(
            A, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-1, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False, scaling_control=scaling_control
        )
        B_q = _quantize_mx(
            B, scale_bits=8, elem_format=elem_format,
            shared_exp_method="max", axes=-2, block_size=block_size,
            round="nearest", flush_fp32_subnorms=False, scaling_control=scaling_control
        )
        
        # 执行矩阵乘法
        output = torch.matmul(A_q, B_q)
        
        # 自动保存forward阶段的tensor
        if layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 根据component类型确定tensor名称
                if component == "FA" or component == "attention":
                    # attention操作：A是attention_probs，B是value
                    tensor_name_A = "attention_probs"
                    tensor_name_B = "value"
                else:
                    # linear操作：使用通用名称
                    tensor_name_A = "input"
                    tensor_name_B = "weight"
                
                # 保存输入tensor A
                save_tensor(
                    tensor=A,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type=f"mxfp_{elem_format}",
                    tensor_name=tensor_name_A,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存输入tensor B
                save_tensor(
                    tensor=B,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type=f"mxfp_{elem_format}",
                    tensor_name=tensor_name_B,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存量化后的tensor A_q
                save_tensor(
                    tensor=A_q,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type=f"mxfp_{elem_format}_quantized",
                    tensor_name="input_A_quantized",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存量化后的tensor B_q
                save_tensor(
                    tensor=B_q,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type=f"mxfp_{elem_format}_quantized",
                    tensor_name="input_B_quantized",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存输出tensor
                save_tensor(
                    tensor=output,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type=f"mxfp_{elem_format}",
                    tensor_name="output",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                pass  # Silently ignore tensor saving errors
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        
        # 计算梯度
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        
        # 自动保存backward阶段的tensor
        if ctx.layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存梯度输出
                save_tensor(
                    tensor=grad_output,
                    layer_type=ctx.layer_type,
                    operation="backward",
                    quant_type=f"mxfp_{ctx.elem_format}",
                    tensor_name="grad_output",
                    layer_idx=ctx.layer_idx,
                    phase="post",
                    component=ctx.component,
                    rank=ctx.rank,
                    metadata=ctx._metadata
                )
                
                # 根据component类型确定backward tensor名称
                if ctx.component == "FA" or ctx.component == "attention":
                    # attention操作：grad_A是grad_attention_probs，grad_B是grad_value
                    grad_tensor_name_A = "grad_attention_probs"
                    grad_tensor_name_B = "grad_value"
                else:
                    # linear操作：使用通用名称
                    grad_tensor_name_A = "grad_input_A"
                    grad_tensor_name_B = "grad_input_B"
                
                # 保存梯度A
                if grad_A is not None:
                    save_tensor(
                        tensor=grad_A,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type=f"mxfp_{ctx.elem_format}",
                        tensor_name=grad_tensor_name_A,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度B
                if grad_B is not None:
                    save_tensor(
                        tensor=grad_B,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type=f"mxfp_{ctx.elem_format}",
                        tensor_name=grad_tensor_name_B,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                pass  # Silently ignore tensor saving errors
        
        return grad_A, grad_B, None, None, None, None, None, None, None, None, None, None  # None对应所有额外参数（12个）

class MXFPBAddBmm(Function):
    @staticmethod
    def forward(ctx, input, batch1, batch2, beta=1.0, alpha=1.0,
                elem_format='fp8_e5m2', block_size=32,
                layer_type=None, layer_idx=None, operation="forward", 
                phase="pre", component="attention", rank=None, metadata=None,
                scaling_control="max"):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta, ctx.alpha = beta, alpha
        ctx.elem_format = elem_format
        ctx.block_size = block_size
        ctx.layer_type = layer_type
        ctx.layer_idx = layer_idx
        ctx.operation = operation
        ctx.phase = phase
        ctx.component = component
        ctx.rank = rank
        ctx._metadata = metadata
        ctx.scaling_control = scaling_control
        
        # 使用集成了tensor保存的MXFPMatMul
        mm_out = MXFPMatMul.apply(batch1, batch2, elem_format, block_size,
                                  layer_type, layer_idx, operation, phase, component, rank, metadata, scaling_control)
        output = beta * input + alpha * mm_out
        
        # 自动保存forward阶段的tensor
        if layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 根据component类型确定tensor名称
                if component == "FA" or component == "attention":
                    # attention操作：input是matmul_input_buffer，batch1是query，batch2是key
                    tensor_name_input = "matmul_input_buffer"
                    tensor_name_batch1 = "query"
                    tensor_name_batch2 = "key"
                else:
                    # 其他操作：使用通用名称
                    tensor_name_input = "input"
                    tensor_name_batch1 = "batch1"
                    tensor_name_batch2 = "batch2"
                
                # 保存输入tensor
                save_tensor(
                    tensor=input,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="mxfp",
                    tensor_name=tensor_name_input,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存batch1 tensor
                save_tensor(
                    tensor=batch1,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="mxfp",
                    tensor_name=tensor_name_batch1,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存batch2 tensor
                save_tensor(
                    tensor=batch2,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="mxfp",
                    tensor_name=tensor_name_batch2,
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
                # 保存最终输出
                save_tensor(
                    tensor=output,
                    layer_type=layer_type,
                    operation=operation,
                    quant_type="mxfp",
                    tensor_name="output",
                    layer_idx=layer_idx,
                    phase=phase,
                    component=component,
                    rank=rank,
                    metadata=metadata
                )
                
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                pass  # Silently ignore tensor saving errors
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output
            grad_batch1 = torch.matmul(mm_grad, batch2.transpose(-2, -1))
            grad_batch2 = torch.matmul(batch1.transpose(-2, -1), mm_grad)
        
        # 自动保存backward阶段的tensor
        if ctx.layer_type is not None:
            try:
                from megatron.core.tensor_saver import save_tensor
                
                # 保存梯度输出
                save_tensor(
                    tensor=grad_output,
                    layer_type=ctx.layer_type,
                    operation="backward",
                    quant_type="mxfp",
                    tensor_name="grad_output",
                    layer_idx=ctx.layer_idx,
                    phase="post",
                    component=ctx.component,
                    rank=ctx.rank,
                    metadata=ctx._metadata
                )
                
                # 根据component类型确定backward tensor名称
                if ctx.component == "FA" or ctx.component == "attention":
                    # attention操作：grad_input是grad_matmul_input_buffer，grad_batch1是grad_query，grad_batch2是grad_key
                    grad_tensor_name_input = "grad_matmul_input_buffer"
                    grad_tensor_name_batch1 = "grad_query"
                    grad_tensor_name_batch2 = "grad_key"
                else:
                    # 其他操作：使用通用名称
                    grad_tensor_name_input = "grad_input"
                    grad_tensor_name_batch1 = "grad_batch1"
                    grad_tensor_name_batch2 = "grad_batch2"
                
                # 保存梯度input
                if grad_input is not None:
                    save_tensor(
                        tensor=grad_input,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="mxfp",
                        tensor_name=grad_tensor_name_input,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度batch1
                if grad_batch1 is not None:
                    save_tensor(
                        tensor=grad_batch1,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="mxfp",
                        tensor_name=grad_tensor_name_batch1,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                
                # 保存梯度batch2
                if grad_batch2 is not None:
                    save_tensor(
                        tensor=grad_batch2,
                        layer_type=ctx.layer_type,
                        operation="backward",
                        quant_type="mxfp",
                        tensor_name=grad_tensor_name_batch2,
                        layer_idx=ctx.layer_idx,
                        phase="post",
                        component=ctx.component,
                        rank=ctx.rank,
                        metadata=ctx._metadata
                    )
                    
            except ImportError:
                pass  # 如果tensor_saver不可用，静默跳过
            except Exception as e:
                pass  # Silently ignore tensor saving errors
        
        return grad_input, grad_batch1, grad_batch2, None, None, None, None, None, None, None, None, None, None, None, None # None对应所有额外参数（15个）

def mxfp_matmul(A, B, elem_format='fp8_e5m2', block_size=32, scaling_control='max', **tensor_save_kwargs):
    """
    MXFP矩阵乘法函数，支持tensor保存
    
    Args:
        A, B: 输入tensor
        elem_format: 元素格式
        block_size: 块大小
        **tensor_save_kwargs: tensor保存相关参数
            - layer_type: 层类型
            - layer_idx: 层索引
            - operation: 操作类型
            - phase: 阶段
            - component: 组件类型
            - rank: GPU rank
            - metadata: 元数据
    """
    # 如果有tensor保存参数，使用集成算子
    if tensor_save_kwargs and any(key in tensor_save_kwargs for key in 
                                 ['layer_type', 'layer_idx', 'operation', 'phase', 'component', 'rank', 'metadata']):
        return MXFPMatMul.apply(
            A, B, elem_format, block_size,
            tensor_save_kwargs.get('layer_type'),
            tensor_save_kwargs.get('layer_idx'),
            tensor_save_kwargs.get('operation', 'forward'),
            tensor_save_kwargs.get('phase', 'pre'),
            tensor_save_kwargs.get('component', 'linear'),
            tensor_save_kwargs.get('rank'),
            tensor_save_kwargs.get('metadata'),
            scaling_control
        )
    else:
        # 否则使用原始调用方式
        return MXFPMatMul.apply(A, B, elem_format, block_size, None, None, "forward", "pre", "linear", None, None, scaling_control)

def mxfp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0,
                 elem_format='fp8_e5m2', block_size=32, scaling_control='max', **tensor_save_kwargs):
    """
    MXFP Batch Add Batch Matrix Multiplication函数，支持tensor保存
    
    Args:
        input, batch1, batch2: 输入tensor
        beta, alpha: 参数
        elem_format: 元素格式
        block_size: 块大小
        **tensor_save_kwargs: tensor保存相关参数
    """
    # 如果有tensor保存参数，使用集成算子
    if tensor_save_kwargs and any(key in tensor_save_kwargs for key in 
                                 ['layer_type', 'layer_idx', 'operation', 'phase', 'component', 'rank', 'metadata']):
        return MXFPBAddBmm.apply(
            input, batch1, batch2, beta, alpha, elem_format, block_size,
            tensor_save_kwargs.get('layer_type'),
            tensor_save_kwargs.get('layer_idx'),
            tensor_save_kwargs.get('operation', 'forward'),
            tensor_save_kwargs.get('phase', 'pre'),
            tensor_save_kwargs.get('component', 'attention'),
            tensor_save_kwargs.get('rank'),
            tensor_save_kwargs.get('metadata'),
            scaling_control
        )
    else:
        # 否则使用原始调用方式
        return MXFPBAddBmm.apply(input, batch1, batch2, beta, alpha, elem_format, block_size, None, None, "forward", "pre", "attention", None, None, scaling_control)

def _remove_scaling_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    scaling_control="max",
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    if axes is None:
        axes = []
    else:
        axes = [axes] if type(axes) == int else axes
        axes = [x + A.ndim if x < 0 else x for x in axes]

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0, scaling_control=scaling_control,
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)
    return A

def analyze_quantized_value_distribution(quantized_tensor, target_values, tolerance=1e-6):
    """
    Analyze the distribution of quantized values in specific ranges.
    
    Args:
        quantized_tensor (torch.Tensor): Quantized tensor after _quantize_elemwise_core
        target_values (list): List of target values to count (e.g., [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        tolerance (float): Tolerance for matching values
        
    Returns:
        dict: Dictionary with counts and percentages for each value (including negative)
    """
    # Convert to numpy for analysis
    # Convert to float32 first to handle BFloat16 and other types
    if quantized_tensor.dtype == torch.bfloat16:
        quantized_tensor = quantized_tensor.float()
    
    # Detach from computation graph if needed, then convert to numpy
    if quantized_tensor.is_cuda:
        values = quantized_tensor.detach().cpu().numpy().flatten()
    else:
        values = quantized_tensor.detach().numpy().flatten()
    
    total_elements = len(values)
    if total_elements == 0:
        return {}
    
    # Count distribution
    distribution = {}
    
    for target_val in target_values:
        # Count positive values
        pos_mask = np.abs(values - target_val) < tolerance
        pos_count = np.sum(pos_mask)
        pos_percent = (pos_count / total_elements) * 100
        
        # Count negative values
        neg_mask = np.abs(values + target_val) < tolerance
        neg_count = np.sum(neg_mask)
        neg_percent = (neg_count / total_elements) * 100
        
        # Count zero (only for target_val == 0)
        if target_val == 0:
            zero_mask = np.abs(values) < tolerance
            zero_count = np.sum(zero_mask)
            zero_percent = (zero_count / total_elements) * 100
            distribution[0.0] = {
                'count': int(zero_count),
                'percent': float(zero_percent)
            }
        else:
            distribution[target_val] = {
                'count': int(pos_count),
                'percent': float(pos_percent)
            }
            distribution[-target_val] = {
                'count': int(neg_count),
                'percent': float(neg_percent)
            }
    
    return distribution


def _quantize_mx_with_statistics(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    minus_exp=0,
    scaling_control="max",
    target_values=None
):
    """
    Quantize tensor and return both quantized tensor and value distribution statistics.
    This is a wrapper around _quantize_mx that captures the quantized values after _quantize_elemwise_core.
    
    Args:
        A: Input tensor
        scale_bits, elem_format, etc.: Same as _quantize_mx
        target_values: List of target values to analyze (e.g., [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        
    Returns:
        tuple: (quantized_tensor, distribution_stats)
        distribution_stats includes 'pre_scaling' and 'post_scaling' keys with statistics and sampled data
        about values before and after scaling, including count and percentage of values with |value| > 6
    """
    if elem_format == None:
        return A, {}
    
    assert(scale_bits > 0)
    
    if axes is None:
        axes = []
    else:
        axes = [axes] if type(axes) == int else axes
        axes = [x + A.ndim if x < 0 else x for x in axes]
    
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)
    
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, block_size)
    
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes
    
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0, scaling_control=scaling_control,
    )
    
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)
    
    shared_exp = shared_exp - emax - minus_exp
    
    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    
    # Collect data before scaling
    pre_scaling_stats = {}
    pre_scaling_data = None
    if target_values is not None:
        # Flatten tensor for analysis (before scaling)
        A_before_flat = A.detach().cpu().float().flatten().numpy()
        
        # Filter outliers (|value| > 6) for analysis
        mask_before = np.abs(A_before_flat) <= 6.0
        A_before_filtered = A_before_flat[mask_before]
        outliers_count_before = np.sum(~mask_before)
        total_count_before = len(A_before_flat)
        
        # Calculate statistics (use full data for min/max, filtered data for mean/std/median)
        pre_scaling_stats = {
            'total_elements': int(total_count_before),
            'outliers_6plus_count': int(outliers_count_before),
            'outliers_6plus_percent': float(outliers_count_before / total_count_before * 100) if total_count_before > 0 else 0.0,
            'filtered_elements': int(len(A_before_filtered)),
            'min': float(np.min(A_before_flat)) if len(A_before_flat) > 0 else 0.0,  # Use full data for min
            'max': float(np.max(A_before_flat)) if len(A_before_flat) > 0 else 0.0,  # Use full data for max
            'mean': float(np.mean(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'std': float(np.std(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'median': float(np.median(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0
        }
        
        # Store pre-scaling data for later plotting (use full data, not filtered)
        # Sample to save memory
        max_samples = min(100000, len(A_before_flat))
        if len(A_before_flat) > max_samples:
            indices = np.random.choice(len(A_before_flat), size=max_samples, replace=False)
            pre_scaling_data = A_before_flat[indices]
        else:
            pre_scaling_data = A_before_flat.copy() if len(A_before_flat) > 0 else None
    
    # Apply scaling
    A = A / (2**shared_exp)
    
    # Collect data after scaling
    post_scaling_data = None
    if target_values is not None:
        # Flatten tensor for analysis (after scaling)
        A_after_flat = A.detach().cpu().float().flatten().numpy()
        
        # Filter outliers (|value| > 6) for analysis
        mask_after = np.abs(A_after_flat) <= 6.0
        A_after_filtered = A_after_flat[mask_after]
        
        # Store post-scaling data for aggregation (sample to save memory)
        # Sample at most 100k points to avoid memory issues
        max_samples = min(100000, len(A_after_filtered))
        if len(A_after_filtered) > max_samples:
            indices = np.random.choice(len(A_after_filtered), size=max_samples, replace=False)
            post_scaling_data = A_after_filtered[indices]
        else:
            post_scaling_data = A_after_filtered.copy() if len(A_after_filtered) > 0 else None
    
    # Quantize - this is where we want to capture the distribution
    A_quantized = _quantize_elemwise_core(
        A, mbits, ebits, max_norm, round=round,
        allow_denorm=True, saturate_normals=True
    )
    
    # Analyze distribution if target_values provided
    distribution_stats = {}
    if target_values is not None:
        distribution_stats = analyze_quantized_value_distribution(
            A_quantized, target_values, tolerance=1e-5
        )
        # Add pre-scaling statistics and data
        distribution_stats['pre_scaling'] = pre_scaling_stats
        # Add sampled data for aggregation (convert to list for JSON serialization)
        if pre_scaling_data is not None:
            # Sample to save memory
            max_samples = min(100000, len(pre_scaling_data))
            if len(pre_scaling_data) > max_samples:
                indices = np.random.choice(len(pre_scaling_data), size=max_samples, replace=False)
                pre_scaling_data_sampled = pre_scaling_data[indices].tolist()
            else:
                pre_scaling_data_sampled = pre_scaling_data.tolist()
            distribution_stats['pre_scaling']['data'] = pre_scaling_data_sampled
        if post_scaling_data is not None:
            distribution_stats['post_scaling'] = {'data': post_scaling_data.tolist()}
    
    # Scale back
    A_quantized = A_quantized * (2**shared_exp)
    
    if block_size:
        A_quantized = _undo_reshape_to_blocks(A_quantized, padded_shape, orig_shape, axes)
    
    return A_quantized, distribution_stats


def save_value_distribution_data(all_results, data_file_path):
    """
    Save value distribution data to JSON file for later reuse.
    
    Args:
        all_results (dict): Results from analyze_folder_value_distribution
        data_file_path (Path): Path to save the data file
    """
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    plot_data = convert_to_json_serializable(all_results)
    
    with open(data_file_path, 'w', encoding='utf-8') as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Value distribution data saved to: {data_file_path}")

def load_value_distribution_data(data_file_path):
    """
    Load value distribution data from JSON file.
    
    Args:
        data_file_path (Path): Path to the data file
        
    Returns:
        dict: Plot data dictionary, or None if file doesn't exist
    """
    if not data_file_path.exists():
        return None
    
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            plot_data = json.load(f)
        print(f"✅ Value distribution data loaded from: {data_file_path}")
        return plot_data
    except Exception as e:
        print(f"⚠️  Error loading value distribution data: {e}")
        return None

def _process_single_tensor(tensor_file, minus_exp, elem_format, scale_bits, block_size, 
                          axes, target_values, return_quantized=False):
    """
    Process a single tensor file and return distribution statistics.
    
    Args:
        tensor_file: Path to tensor file
        minus_exp: minus_exp value to use
        elem_format: Element format
        scale_bits: Number of scale bits
        block_size: Block size for tiling
        axes: Axes for shared exponent calculation
        target_values: List of target values to analyze
        return_quantized: If True, also return quantized tensor
        
    Returns:
        dict: Result dictionary with 'success', 'distribution', 'tensor_file', 'error', etc.
        distribution includes 'pre_scaling' and 'post_scaling' with sampled data for aggregation
    """
    result = {
        'success': False,
        'tensor_file': str(tensor_file),
        'tensor_name': tensor_file.name if hasattr(tensor_file, 'name') else str(tensor_file),
        'minus_exp': minus_exp,
        'distribution': None,
        'quantized_tensor': None,
        'error': None
    }
    
    try:
        # Load tensor
        data = torch.load(str(tensor_file), map_location='cpu', weights_only=False)
        
        if isinstance(data, dict) and 'tensor' in data:
            input_tensor = data['tensor']
        elif isinstance(data, torch.Tensor):
            input_tensor = data
        else:
            result['error'] = 'Invalid format'
            return result
        
        # Convert to bfloat16 if needed
        if input_tensor.dtype != torch.bfloat16:
            input_tensor = input_tensor.bfloat16()
        
        # Quantize and get statistics
        quantized_tensor, distribution = _quantize_mx_with_statistics(
            input_tensor,
            scale_bits=scale_bits,
            elem_format=elem_format,
            shared_exp_method="max",
            axes=axes,
            block_size=block_size,
            minus_exp=minus_exp,
            round="nearest",
            flush_fp32_subnorms=False,
            scaling_control="max",
            target_values=target_values
        )
        
        if distribution:
            result['success'] = True
            result['distribution'] = distribution
            if return_quantized:
                result['quantized_tensor'] = quantized_tensor
        else:
            result['error'] = 'No distribution data'
            
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def analyze_folder_value_distribution(folder_path, elem_format='fp4_e2m1', 
                                      target_values=[0, 0.5, 1, 1.5, 2, 3, 4, 6],
                                      output_dir=None, scale_bits=8, block_size=32, axes=-1,
                                      minus_exp_list=[0, 1, 2], num_workers=32):
    """
    Analyze value distribution for all tensor files in a folder.
    Supports multiple minus_exp values and generates separate plots for each.
    Uses multithreading to speed up processing.
    
    Args:
        folder_path (str): Path to folder containing .pt tensor files
        elem_format (str): Element format (default: 'fp4_e2m1')
        target_values (list): List of target values to analyze
        output_dir (str): Output directory for plots (default: ./draw/value_distribution/)
        scale_bits (int): Number of scale bits
        block_size (int): Block size for tiling
        axes (int): Axes for shared exponent calculation
        minus_exp_list (list): List of minus_exp values to test (default: [0, 1, 2])
        num_workers (int): Number of worker threads for parallel processing (default: 32).
                           Recommended: 0.25-0.5x CPU cores for CPU-bound tasks, or 0.5-1x for I/O-bound tasks.
                           Adjust based on available memory and tensor sizes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
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
    
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find all .pt files
    tensor_files = list(folder_path.glob("*.pt"))
    if not tensor_files:
        print(f"No .pt files found in {folder_path}")
        return
    
    print(f"Found {len(tensor_files)} tensor files in {folder_path}")
    print(f"Testing minus_exp values: {minus_exp_list}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("./draw/value_distribution") / folder_path.name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data file path
    data_dir = Path("./draw/input/value_distribution")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file_path = data_dir / f"{folder_path.name}_{elem_format}_plot_data.json"
    
    # Check if plot data exists
    plot_data = load_value_distribution_data(data_file_path)
    if plot_data is not None:
        print(f"Found existing plot data, skipping computation")
        print(f"To recompute, delete: {data_file_path}")
        
        # Generate plots from loaded data
        for minus_exp_str, result_data in plot_data.items():
            # Convert minus_exp from string to int (JSON loads keys as strings)
            minus_exp = int(minus_exp_str)
            aggregated_dist = result_data['aggregated_dist']
            successful_count = result_data['successful_count']
            
            # Prepare data for plotting
            # Filter out non-numeric keys first
            numeric_keys = [k for k in aggregated_dist.keys() if isinstance(k, (int, float))]
            # Convert keys to float for sorting, but keep original keys for access
            # Sort by float value but use original keys for access
            key_value_pairs = [(float(k), k) for k in numeric_keys]
            key_value_pairs.sort(key=lambda x: x[0])  # Sort by float value
            
            sorted_values = [v for v, _ in key_value_pairs]  # Float values for labels
            sorted_keys = [k for _, k in key_value_pairs]    # Original keys for access
            
            percentages = [aggregated_dist[k]['avg_percent'] for k in sorted_keys]
            std_percentages = [aggregated_dist[k].get('std_percent', 0) for k in sorted_keys]
            labels = [f'{v:+.1f}' if v != 0 else '0' for v in sorted_values]
            
            # Create beautiful plot
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                try:
                    plt.style.use('seaborn-darkgrid')
                except:
                    plt.style.use('default')
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Use gradient colors
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_values)))
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(sorted_values)), percentages, 
                          yerr=std_percentages,
                          color=colors, alpha=0.8, edgecolor='white', linewidth=2,
                          error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capsize': 5})
            
            # Add value labels on bars
            for i, (bar, pct, std) in enumerate(zip(bars, percentages, std_percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                        f'{pct:.2f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Styling
            ax.set_xlabel('Quantized Values', fontsize=14, fontweight='bold', color='#333333')
            ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold', color='#333333')
            ax.set_xticks(range(len(sorted_values)))
            ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
            ax.set_title(f'Value Distribution Analysis - {elem_format.upper()}\n'
                         f'minus_exp={minus_exp} ',
                         fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, color='#CCCCCC', axis='y')
            ax.set_axisbelow(True)
            
            # Background
            ax.set_facecolor('#F8F9FA')
            fig.patch.set_facecolor('white')
            
            # Border
            for spine in ax.spines.values():
                spine.set_edgecolor('#E0E0E0')
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            
            # Save plot with minus_exp in filename (high-resolution PDF)
            plot_path = output_dir / f'value_distribution_{elem_format}_minus_exp_{minus_exp}_{folder_path.name}.pdf'
            plt.savefig(plot_path, 
                       format='pdf',
                       dpi=600,  # High DPI for any rasterized elements
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       metadata={'Creator': 'MXFP Formatter', 'Title': f'Value Distribution - {elem_format} (minus_exp={minus_exp})'})
            plt.close()
            
            print(f"\n✅ Plot saved to: {plot_path}")
        
        # 即使从缓存加载，也需要重新处理tensor进行zeroing loss分析
        # 所以不在这里返回，继续执行zeroing loss分析
        all_results = plot_data
    
    # Process each minus_exp value
    all_results = {}
    
    # Collect scaling data for aggregation (use first minus_exp)
    all_pre_scaling_data = []
    all_post_scaling_data = []
    all_quantized_distributions = []  # Collect quantized value distributions for bar chart
    scaling_data_collected = False
    
    for minus_exp in minus_exp_list:
        print(f"\n{'='*60}")
        print(f"Processing minus_exp = {minus_exp}")
        print(f"{'='*60}")
        
        # Collect statistics from all tensors for this minus_exp using multithreading
        all_distributions = []
        successful_count = 0
        
        print(f"Processing {len(tensor_files)} tensors with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_tensor = {
                executor.submit(
                    _process_single_tensor,
                    tensor_file, minus_exp, elem_format, scale_bits,
                    block_size, axes, target_values, return_quantized=False
                ): tensor_file
                for tensor_file in tensor_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tensor_files), desc=f"minus_exp={minus_exp}", 
                     unit="tensor") as pbar:
                for future in as_completed(future_to_tensor):
                    tensor_file = future_to_tensor[future]
                    result = future.result()
                    
                    if result['success']:
                        all_distributions.append(result['distribution'])
                        successful_count += 1
                        pbar.set_postfix({'success': successful_count, 
                                        'file': result['tensor_name'][:30]})
                        
                        # Collect scaling data for aggregation (only for first minus_exp to avoid duplication)
                        if not scaling_data_collected and result['distribution']:
                            dist = result['distribution']
                            if 'pre_scaling' in dist and 'data' in dist['pre_scaling']:
                                all_pre_scaling_data.extend(dist['pre_scaling']['data'])
                            if 'post_scaling' in dist and 'data' in dist['post_scaling']:
                                all_post_scaling_data.extend(dist['post_scaling']['data'])
                            # Collect quantized distribution (exclude 'pre_scaling' and 'post_scaling' keys)
                            quantized_dist = {k: v for k, v in dist.items() 
                                            if isinstance(k, (int, float)) and k not in ['pre_scaling', 'post_scaling']}
                            if quantized_dist:
                                all_quantized_distributions.append(quantized_dist)
                    else:
                        if result['error']:
                            print(f"  ⚠️  {result['tensor_name']}: {result['error']}")
                        if 'traceback' in result:
                            print(f"  ❌ Error in {result['tensor_name']}:")
                            print(result['traceback'])
                    
                    pbar.update(1)
        
        if not all_distributions:
            print(f"No valid distribution data collected for minus_exp={minus_exp}.")
            continue
        
        print(f"\nSuccessfully processed {successful_count}/{len(tensor_files)} tensors for minus_exp={minus_exp}")
        
        # Mark scaling data as collected after first minus_exp
        if not scaling_data_collected:
            scaling_data_collected = True
        
        # Aggregate statistics
        aggregated_dist = {}
        for dist in all_distributions:
            for value, stats in dist.items():
                # Skip non-numeric keys like 'pre_quantization'
                if not isinstance(value, (int, float)):
                    continue
                # Skip if stats doesn't have 'count' key (shouldn't happen, but safety check)
                if not isinstance(stats, dict) or 'count' not in stats:
                    continue
                if value not in aggregated_dist:
                    aggregated_dist[value] = {'count': 0, 'percent': []}
                aggregated_dist[value]['count'] += stats['count']
                aggregated_dist[value]['percent'].append(stats['percent'])
        
        # Calculate average percentages
        for value in aggregated_dist:
            aggregated_dist[value]['avg_percent'] = np.mean(aggregated_dist[value]['percent'])
            aggregated_dist[value]['std_percent'] = np.std(aggregated_dist[value]['percent'])
        
        # Store results for this minus_exp
        all_results[minus_exp] = {
            'aggregated_dist': aggregated_dist,
            'successful_count': successful_count
        }
        
        # Prepare data for plotting
        # Filter out non-numeric keys (safety check)
        numeric_keys = [k for k in aggregated_dist.keys() if isinstance(k, (int, float))]
        sorted_values = sorted(numeric_keys)
        percentages = [aggregated_dist[v]['avg_percent'] for v in sorted_values]
        std_percentages = [aggregated_dist[v].get('std_percent', 0) for v in sorted_values]
        labels = [f'{v:+.1f}' if v != 0 else '0' for v in sorted_values]
        
        # Create beautiful plot
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_values)))
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(sorted_values)), percentages, 
                      yerr=std_percentages,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=2,
                      error_kw={'elinewidth': 2, 'ecolor': '#333333', 'capsize': 5})
        
        # Add value labels on bars
        for i, (bar, pct, std) in enumerate(zip(bars, percentages, std_percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{pct:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Quantized Values', fontsize=14, fontweight='bold', color='#333333')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold', color='#333333')
        ax.set_xticks(range(len(sorted_values)))
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.set_title(f'Value Distribution Analysis - {elem_format.upper()}\n'
                     f'minus_exp={minus_exp} ',
                     fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, color='#CCCCCC', axis='y')
        ax.set_axisbelow(True)
        
        # Background
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor('#E0E0E0')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save plot with minus_exp in filename (high-resolution PDF)
        plot_path = output_dir / f'value_distribution_{elem_format}_minus_exp_{minus_exp}_{folder_path.name}.pdf'
        # Use high DPI for rasterized elements and ensure vector format
        plt.savefig(plot_path, 
                   format='pdf',
                   dpi=600,  # High DPI for any rasterized elements
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   metadata={'Creator': 'MXFP Formatter', 'Title': f'Value Distribution - {elem_format} (minus_exp={minus_exp})'})
        plt.close()
        
        print(f"\n✅ Plot saved to: {plot_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"VALUE DISTRIBUTION SUMMARY (minus_exp={minus_exp})")
        print("=" * 60)
        for value in sorted_values:
            stats = aggregated_dist[value]
            print(f"Value {value:+.1f}: {stats['avg_percent']:.2f}% ± {stats.get('std_percent', 0):.2f}% "
                  f"(total count: {stats['count']:,})")
        print("=" * 60)
    
    # Save plot data for future reuse (only if not loaded from cache)
    if plot_data is None:  # Only save if we computed new data
        save_value_distribution_data(all_results, data_file_path)
    
    # ========== 新增功能：分析 zeroing loss 占比最大的情况 ==========
    print("\n" + "=" * 60)
    print("ANALYZING ZEROING LOSS (4,5,6比例最小的情况)")
    print("=" * 60)
    
    # 收集所有tensor的0和4、5、6比例数据
    zeroing_analysis_data = []
    
    for minus_exp in minus_exp_list:
        print(f"\nProcessing minus_exp = {minus_exp} for zeroing loss analysis...")
        
        # 使用多线程重新处理每个tensor以获取详细统计
        # 注意：不需要保存quantized_tensor，只保存distribution即可，避免内存溢出
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_tensor = {
                executor.submit(
                    _process_single_tensor,
                    tensor_file, minus_exp, elem_format, scale_bits,
                    block_size, axes, target_values, return_quantized=False
                ): tensor_file
                for tensor_file in tensor_files
            }
            
            with tqdm(total=len(tensor_files), desc=f"zeroing_loss minus_exp={minus_exp}", 
                     unit="tensor") as pbar:
                for future in as_completed(future_to_tensor):
                    result = future.result()
                    
                    if result['success'] and result['distribution']:
                        distribution = result['distribution']
                        
                        # 计算0的比例
                        zero_percent = distribution.get(0.0, {}).get('percent', 0.0)
                        
                        # 计算4、5、6的比例（包括正负值）
                        # 注意：5可能不在target_values中，所以只统计存在的值
                        large_values = [4.0, 5.0, 6.0]
                        large_percent = 0.0
                        for val in large_values:
                            # 只统计在distribution中存在的值
                            if val in distribution:
                                large_percent += distribution[val].get('percent', 0.0)
                            if -val in distribution:
                                large_percent += distribution[-val].get('percent', 0.0)
                        
                        zeroing_analysis_data.append({
                            'tensor_file': result['tensor_name'],
                            'minus_exp': minus_exp,
                            'zero_percent': zero_percent,
                            'large_percent': large_percent,  # 4,5,6的总比例
                            'distribution': distribution
                            # 不再保存quantized_tensor，避免内存溢出
                        })
                    
                    pbar.update(1)
    
    if not zeroing_analysis_data:
        print("No data collected for zeroing loss analysis.")
        return all_results
    
    # 找出4、5、6比例最小的tensor（即zeroing loss占比最大）
    min_large_percent_idx = np.argmin([d['large_percent'] for d in zeroing_analysis_data])
    max_zeroing_loss_data = zeroing_analysis_data[min_large_percent_idx]
    
    print(f"\n找到 zeroing loss 占比最大的 tensor:")
    print(f"  Tensor: {max_zeroing_loss_data['tensor_file']}")
    print(f"  minus_exp: {max_zeroing_loss_data['minus_exp']}")
    print(f"  0的比例: {max_zeroing_loss_data['zero_percent']:.2f}%")
    print(f"  4,5,6的比例: {max_zeroing_loss_data['large_percent']:.2f}%")
    
    # 绘制zeroing loss占比最大的tensor的分布图
    distribution = max_zeroing_loss_data['distribution']
    # Filter out non-numeric keys (like 'pre_scaling')
    numeric_keys = [k for k in distribution.keys() if isinstance(k, (int, float))]
    sorted_values = sorted(numeric_keys)
    percentages = [distribution[v]['percent'] for v in sorted_values]
    labels = [f'{v:+.1f}' if v != 0 else '0' for v in sorted_values]
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_values)))
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_values)), percentages, 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        if height > 0.5:  # Only show label if bar is significant
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight 0 and 4,5,6 bars
    for i, val in enumerate(sorted_values):
        abs_val = abs(float(val))
        if abs_val < 1e-6:  # Zero
            bars[i].set_color('#FF6B6B')  # Red for zero
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2.5)
        elif abs_val in [4.0, 5.0, 6.0]:
            bars[i].set_color('#4A90E2')  # Blue for large values
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2.5)
    
    # Styling
    ax.set_xlabel('Quantized Values', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_xticks(range(len(sorted_values)))
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_title(f'Zeroing Loss Analysis - Maximum Zeroing Loss Case\n'
                 f'{elem_format.upper()} | minus_exp={max_zeroing_loss_data["minus_exp"]} | '
                 f'Tensor: {max_zeroing_loss_data["tensor_file"]}\n'
                 f'Zero: {max_zeroing_loss_data["zero_percent"]:.2f}% | '
                 f'Large (4,5,6): {max_zeroing_loss_data["large_percent"]:.2f}%',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, color='#CCCCCC', axis='y')
    ax.set_axisbelow(True)
    
    # Background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save plot
    zeroing_plot_path = output_dir / f'zeroing_loss_max_{elem_format}_{folder_path.name}.pdf'
    plt.savefig(zeroing_plot_path, 
               format='pdf',
               dpi=600,
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none',
               metadata={'Creator': 'MXFP Formatter', 
                        'Title': f'Zeroing Loss Analysis - Maximum Case - {elem_format}'})
    plt.close()
    
    print(f"\n✅ Zeroing loss plot saved to: {zeroing_plot_path}")
    
    # 统计0和4、5、6比例的比较
    print("\n" + "=" * 60)
    print("STATISTICS: Zero vs Large Values (4,5,6) Comparison")
    print("=" * 60)
    
    zero_percents = [d['zero_percent'] for d in zeroing_analysis_data]
    large_percents = [d['large_percent'] for d in zeroing_analysis_data]
    
    # 统计0比较多的情况
    zero_dominant_count = sum(1 for z, l in zip(zero_percents, large_percents) if z > l)
    zero_dominant_percent = (zero_dominant_count / len(zeroing_analysis_data)) * 100
    
    print(f"Total tensors analyzed: {len(zeroing_analysis_data)}")
    print(f"\nZero percentage statistics:")
    print(f"  Mean: {np.mean(zero_percents):.2f}%")
    print(f"  Std:  {np.std(zero_percents):.2f}%")
    print(f"  Min:  {np.min(zero_percents):.2f}%")
    print(f"  Max:  {np.max(zero_percents):.2f}%")
    
    print(f"\nLarge values (4,5,6) percentage statistics:")
    print(f"  Mean: {np.mean(large_percents):.2f}%")
    print(f"  Std:  {np.std(large_percents):.2f}%")
    print(f"  Min:  {np.min(large_percents):.2f}%")
    print(f"  Max:  {np.max(large_percents):.2f}%")
    
    print(f"\nZero > Large (4,5,6) cases: {zero_dominant_count}/{len(zeroing_analysis_data)} ({zero_dominant_percent:.2f}%)")
    
    # 绘制0和4、5、6比例的对比散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(zero_percents, large_percents, 
                        alpha=0.6, s=100, c=range(len(zero_percents)), 
                        cmap='viridis', edgecolors='black', linewidths=0.5)
    
    # Highlight the max zeroing loss case
    ax.scatter([max_zeroing_loss_data['zero_percent']], 
               [max_zeroing_loss_data['large_percent']],
               s=300, c='red', marker='*', edgecolors='black', 
               linewidths=2, label='Max Zeroing Loss', zorder=5)
    
    # Add diagonal line (zero = large)
    max_val = max(max(zero_percents), max(large_percents))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, label='Zero = Large')
    
    # Styling
    ax.set_xlabel('Zero Percentage (%)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Large Values (4,5,6) Percentage (%)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title(f'Zero vs Large Values Comparison\n'
                 f'{elem_format.upper()} | {len(zeroing_analysis_data)} tensors',
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper right')
    
    # Background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = output_dir / f'zero_vs_large_comparison_{elem_format}_{folder_path.name}.pdf'
    plt.savefig(comparison_plot_path, 
               format='pdf',
               dpi=600,
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none',
               metadata={'Creator': 'MXFP Formatter', 
                        'Title': f'Zero vs Large Values Comparison - {elem_format}'})
    plt.close()
    
    print(f"\n✅ Comparison plot saved to: {comparison_plot_path}")
    print("=" * 60)
    
    # Aggregate and plot scaling distribution (pre and post scaling)
    print("\n" + "=" * 60)
    print("Creating aggregated scaling distribution plot...")
    print("=" * 60)
    
    # Create aggregated scaling distribution plot using collected data
    if len(all_pre_scaling_data) > 0 and len(all_quantized_distributions) > 0:
            try:
                # Convert to numpy arrays
                pre_data = np.array(all_pre_scaling_data)
                
                # Aggregate quantized distributions
                # Sum counts across all tensors, then calculate total percentage
                total_count = 0
                aggregated_quantized_dist = {}
                for dist in all_quantized_distributions:
                    for value, stats in dist.items():
                        if value not in aggregated_quantized_dist:
                            aggregated_quantized_dist[value] = {'count': 0}
                        aggregated_quantized_dist[value]['count'] += stats.get('count', 0)
                        total_count += stats.get('count', 0)
                
                # Calculate percentages based on total count
                if total_count > 0:
                    for value in aggregated_quantized_dist:
                        aggregated_quantized_dist[value]['percent'] = (aggregated_quantized_dist[value]['count'] / total_count) * 100
                
                # Create two separate figures instead of subplots
                
                # ========== Figure 1: Before Scaling ==========
                # Size optimized for single column in Overleaf two-column layout (width ~3.5 inches)
                # Font: Times New Roman/Calibri (set globally via plt.rcParams)
                fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
                
                # Plot pre-scaling distribution - histogram
                n_bins_pre = min(200, max(50, int(np.sqrt(len(pre_data)))))
                if len(pre_data) > 0:
                    counts, bins, patches = ax1.hist(pre_data, bins=n_bins_pre, alpha=0.7, 
                            color='#1f77b4', density=True,
                            linewidth=0.3, edgecolor='black')
                    
                    # Add zero reference line
                    if np.min(pre_data) < 0 < np.max(pre_data):
                        ax1.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='Zero')
                    
                    # Set labels and styling (using Times New Roman/Calibri from global settings)
                    ax1.set_xlabel('Value', fontsize=12, fontweight='normal')
                    ax1.set_ylabel('Density', fontsize=12, fontweight='normal')
                    ax1.tick_params(axis='both', which='major', labelsize=12)
                    ax1.tick_params(axis='both', which='minor', labelsize=11)
                    ax1.grid(True, alpha=0.25, linewidth=0.5)
                    # ax1.set_title('Before Scaling', fontsize=13, fontweight='normal')
                    # Don't clip x-axis - let it show full range
                    
                    # Mark minimum and maximum values above x-axis (after setting labels to get proper y-axis range)
                    data_min = np.min(pre_data)
                    data_max = np.max(pre_data)
                    
                    # Get y-axis limits to position markers above the axis
                    y_min_axis, y_max_axis = ax1.get_ylim()
                    # Position markers slightly above the x-axis (about 2% of y-axis range)
                    y_marker_pos = y_min_axis + (y_max_axis - y_min_axis) * 0.05
                    
                    # Mark minimum with blue downward triangle
                    ax1.plot(data_min, y_marker_pos, marker='v', markersize=9, color='blue', 
                             markeredgecolor='darkblue', markeredgewidth=2, zorder=10, label='Min')
                    
                    # Mark maximum with red upward triangle
                    ax1.plot(data_max, y_marker_pos, marker='^', markersize=9, color='red', 
                             markeredgecolor='darkred', markeredgewidth=2, zorder=10, label='Max')
                    
                    # Add legend after markers (using Times New Roman/Calibri from global settings)
                    ax1.legend(fontsize=10, framealpha=0.9)
                
                # Improve layout and save first plot
                fig1.tight_layout(pad=1.0)
                fig1.patch.set_facecolor('white')
                
                scaling_plot_path_before = output_dir / f'scaling_distribution_before_{elem_format}_{folder_path.name}.pdf'
                fig1.savefig(scaling_plot_path_before, dpi=600, bbox_inches='tight', pad_inches=0.05, 
                           facecolor='white', edgecolor='none')
                plt.close(fig1)
                
                print(f"\n✅ Before scaling distribution plot saved to: {scaling_plot_path_before}")
                
                # ========== Figure 2: After Scaling ==========
                if aggregated_quantized_dist:
                    # Optimized for single column display in Overleaf (double column layout)
                    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.0))
                    
                    # Filter out non-numeric keys and sort (show all values)
                    numeric_keys = [k for k in aggregated_quantized_dist.keys() if isinstance(k, (int, float))]
                    sorted_values = sorted(numeric_keys)
                    percentages = [aggregated_quantized_dist[v]['percent'] for v in sorted_values]
                    
                    # Create labels: show all values on x-axis (preserve decimals)
                    def format_value(val):
                        """Format value, preserving decimals if needed"""
                        if abs(float(val)) < 1e-6:
                            return '0'
                        # Check if value is effectively an integer
                        if abs(float(val) - round(float(val))) < 1e-6:
                            return f'{int(round(val)):+d}' if val != 0 else '0'
                        else:
                            # Has decimal part, show with 1 decimal place
                            return f'{val:+.1f}'
                    
                    labels = [format_value(v) for v in sorted_values]
                    
                    # Calculate statistics for three ranges: < -3, [-3, 3], > 3
                    range_less_neg3_percent = 0.0
                    range_neg3_to_3_percent = 0.0
                    range_greater_3_percent = 0.0
                    range_less_neg3_indices = []
                    range_neg3_to_3_indices = []
                    range_greater_3_indices = []
                    
                    for i, val in enumerate(sorted_values):
                        val_float = float(val)
                        if val_float < -3.0:
                            range_less_neg3_percent += percentages[i]
                            range_less_neg3_indices.append(i)
                        elif -3.0 <= val_float <= 3.0:
                            range_neg3_to_3_percent += percentages[i]
                            range_neg3_to_3_indices.append(i)
                        else:  # val_float > 3.0
                            range_greater_3_percent += percentages[i]
                            range_greater_3_indices.append(i)
                    
                    # Use gradient colors from a single color scheme
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_values)))
                    
                    # Create bar plot (show all values)
                    bars = ax2.bar(range(len(sorted_values)), percentages, 
                                  color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
                    
                    # Calculate y_max once for positioning annotations
                    y_max = max(percentages) if percentages else 0
                    annotation_y = y_max * 1.25  # Position above bars
                    
                    # Create three-segment annotation with vertical separators
                    # Format: |<---X.X%-->|<---X.X%--->|<---X.X%-->|
                    stats_text = f'|<-{range_less_neg3_percent:.1f}%->|<---------------{range_neg3_to_3_percent:.1f}%--------------->|<-{range_greater_3_percent:.1f}%->|'
                    
                    # Add annotation at the top center
                    ax2.text(len(sorted_values) / 2.0 - 0.5, annotation_y, stats_text,
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='#2C3E50', linewidth=1.0, alpha=0.9),
                            zorder=10)
                    
                    # Draw unified arrow from -6 to 6 with vertical separators at -6, -3, 3, 6
                    arrow_y = y_max * 1.15
                    arrow_color = '#2C3E50'
                    arrow_lw = 1.5
                    
                    # Find indices for -6, -3, 3, 6 (these are the x-coordinates of bar centers)
                    # Use first occurrence for -6 and last occurrence for 6 to cover the full range
                    idx_neg6_first = None
                    idx_neg6_last = None
                    idx_neg3 = None
                    idx_3 = None
                    idx_6_first = None
                    idx_6_last = None
                    
                    for i, val in enumerate(sorted_values):
                        val_float = float(val)
                        if abs(val_float + 6.0) < 1e-6:
                            if idx_neg6_first is None:
                                idx_neg6_first = i
                            idx_neg6_last = i
                        elif abs(val_float + 4.0) < 1e-6:
                            idx_neg3 = i
                        elif abs(val_float - 4.0) < 1e-6:
                            idx_3 = i
                        elif abs(val_float - 6.0) < 1e-6:
                            if idx_6_first is None:
                                idx_6_first = i
                            idx_6_last = i
                    
                    # Use first -6 and last 6 to cover the full range
                    idx_neg6 = idx_neg6_first if idx_neg6_first is not None else idx_neg6_last
                    idx_6 = idx_6_last if idx_6_last is not None else idx_6_first
                    
                    # Draw unified arrow from -6 to 6 (using exact bar center positions)
                    if idx_neg6 is not None and idx_6 is not None:
                        # Left arrow at first -6 (pointing right, starting from bar center)
                        ax2.annotate('', xy=(idx_neg6, arrow_y), xytext=(idx_neg6 - 0.5, arrow_y),
                                    arrowprops=dict(arrowstyle='-', lw=arrow_lw, color=arrow_color))
                        # Middle line from first -6 to last 6 (exact bar center to bar center)
                        ax2.plot([idx_neg6, idx_6], [arrow_y, arrow_y], 
                                color=arrow_color, linewidth=arrow_lw, linestyle='-', zorder=0)
                        # Right arrow at last 6 (pointing left, ending at bar center)
                        ax2.annotate('', xy=(idx_6, arrow_y), xytext=(idx_6 + 0.5, arrow_y),
                                    arrowprops=dict(arrowstyle='-', lw=arrow_lw, color=arrow_color))
                    
                    # Draw vertical separators at -6, -3, 3, 6 (aligned exactly with bar centers)
                    # Use the first occurrence of -6 and last occurrence of 6 for separators
                    separator_bottom = y_max * 1.05  # Just above the highest bar
                    separator_top = arrow_y + 0.2     # Just above the arrow line
                    
                    # Draw separator at first -6
                    if idx_neg6_first is not None:
                        ax2.plot([idx_neg6_first, idx_neg6_first], [separator_bottom, separator_top], 
                                color=arrow_color, linewidth=arrow_lw, zorder=0)
                    # Draw separator at -3
                    if idx_neg3 is not None:
                        ax2.plot([idx_neg3, idx_neg3], [separator_bottom, separator_top], 
                                color=arrow_color, linewidth=arrow_lw, zorder=0)
                    # Draw separator at 3
                    if idx_3 is not None:
                        ax2.plot([idx_3, idx_3], [separator_bottom, separator_top], 
                                color=arrow_color, linewidth=arrow_lw, zorder=0)
                    # Draw separator at last 6
                    if idx_6_last is not None:
                        ax2.plot([idx_6_last, idx_6_last], [separator_bottom, separator_top], 
                                color=arrow_color, linewidth=arrow_lw, zorder=0)
                    
                    # Styling optimized for single column display in Overleaf
                    ax2.set_xlabel('Quantized Values', fontsize=10, fontweight='normal')
                    ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='normal')
                    ax2.set_xticks(range(len(sorted_values)))
                    # Show all values on x-axis, slight rotation to prevent overlap
                    ax2.set_xticklabels(labels, fontsize=9, fontweight='bold', rotation=30, ha='right')
                    # Remove title as requested
                    # ax2.set_title('After Scaling', fontsize=15, fontweight='normal')
                    # Set tick labels for both axes (optimized for Overleaf paper)
                    ax2.tick_params(axis='both', which='major', labelsize=9)
                    ax2.tick_params(axis='both', which='minor', labelsize=8)
                    # Make y-axis tick labels bold
                    for label in ax2.get_yticklabels():
                        label.set_fontweight('bold')
                    
                    # Adjust y-axis limits to accommodate arrows and annotation
                    ax2.set_ylim(top=y_max * 1.5)  # Add extra space at top for arrows and annotation
                    
                    # Grid
                    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, axis='y')
                    ax2.set_axisbelow(True)
                    
                    # Background
                    ax2.set_facecolor('#F8F9FA')
                    
                    # Improve layout and save second plot (optimized for single column)
                    fig2.tight_layout(pad=0.5)
                    fig2.patch.set_facecolor('white')
                    
                    scaling_plot_path_after = output_dir / f'scaling_distribution_after_{elem_format}_{folder_path.name}.pdf'
                    fig2.savefig(scaling_plot_path_after, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close(fig2)
                    
                    print(f"✅ After scaling distribution plot saved to: {scaling_plot_path_after}")
            except Exception as e:
                print(f"Warning: Failed to create aggregated scaling distribution plot: {e}")
                import traceback
                traceback.print_exc()
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze value distribution for fp4_e2m1 quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Test all default minus_exp values (0, 1, 2)
  python utils/saver/mxfp_formater.py data/bf16
  
  # Test only minus_exp=0
  python utils/saver/mxfp_formater.py data/bf16 0
  
  # Test multiple minus_exp values using --minus-exp-list
  python utils/saver/mxfp_formater.py data/bf16 --minus-exp-list 0 1 2 3
        '''
    )
    parser.add_argument('folder_path', type=str, help='Path to folder containing .pt tensor files')
    parser.add_argument('minus_exp', type=int, nargs='?', default=None,
                        help='Single minus_exp value to test (optional). If provided, overrides --minus-exp-list. Example: python utils/saver/mxfp_formater.py data/bf16 0')
    parser.add_argument('--elem-format', default='fp4_e2m1', 
                        choices=['fp4_e2m1', 'fp8_e4m3', 'fp8_e5m2'],
                        help='Element format (default: fp4_e2m1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: ./draw/value_distribution/)')
    parser.add_argument('--scale-bits', type=int, default=8,
                        help='Number of scale bits (default: 8)')
    parser.add_argument('--block-size', type=int, default=32,
                        help='Block size for tiling (default: 32)')
    parser.add_argument('--axes', type=int, default=-1,
                        help='Axes for shared exponent calculation (default: -1)')
    parser.add_argument('--minus-exp-list', type=int, nargs='+', default=[0, 1, 2],
                        help='List of minus_exp values to test (default: 0 1 2). Ignored if minus_exp positional argument is provided.')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of worker threads for parallel processing (default: 32). '
                             'Recommended: 0.25-0.5x CPU cores for CPU-bound tasks, or 0.5-1x for I/O-bound tasks. '
                             'Adjust based on available memory and tensor sizes.')
    
    args = parser.parse_args()
    
    # If minus_exp positional argument is provided, use it instead of --minus-exp-list
    if args.minus_exp is not None:
        minus_exp_list = [args.minus_exp]
        print(f"Using single minus_exp value: {args.minus_exp}")
    else:
        minus_exp_list = args.minus_exp_list
        print(f"Using minus_exp_list: {minus_exp_list}")
    
    # Default target values for fp4_e2m1
    target_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    
    analyze_folder_value_distribution(
        folder_path=args.folder_path,
        elem_format=args.elem_format,
        target_values=target_values,
        output_dir=args.output_dir,
        scale_bits=args.scale_bits,
        block_size=args.block_size,
        axes=args.axes,
        minus_exp_list=minus_exp_list,
        num_workers=args.num_workers
    )
