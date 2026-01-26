import torch
# import torch_npu
from enum import Enum, IntEnum
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set font to Times New Roman (or Calibri as fallback) for paper figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
# Try to use Times New Roman, fallback to Calibri if not available
try:
    from matplotlib import font_manager
    # Check if Times New Roman is available
    times_fonts = [f.name for f in font_manager.fontManager.ttflist if 'times' in f.name.lower() or 'Times' in f.name]
    if times_fonts:
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    else:
        # Fallback to Calibri
        calibri_fonts = [f.name for f in font_manager.fontManager.ttflist if 'calibri' in f.name.lower() or 'Calibri' in f.name]
        if calibri_fonts:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
except:
    pass


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
        A = torch.sign(A) * torch.floor(torch.abs(A)+0.5)
        # A = torch.sign(A) * torch.floor(torch.abs(A))
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
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
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
    pre_scaling_blocks = None  # Per-block data for scatter plot
    pre_scaling_shared_exp = None  # Shared exponents for each block
    if target_values is not None:
        # Convert to numpy for analysis
        A_before_np = A.detach().cpu().float().numpy()
        shared_exp_np = shared_exp.detach().cpu().float().numpy()
        
        # Collect per-block data if block_size > 0
        if block_size > 0:
            # A has been reshaped to blocks, shape should be (..., block_size)
            # We want to collect data per row (block) along axis=-1
            A_shape = A_before_np.shape
            if len(A_shape) >= 2 and A_shape[-1] == block_size:
                # Reshape to (num_blocks, block_size) for easier processing
                num_blocks = int(np.prod(A_shape[:-1]))
                A_reshaped = A_before_np.reshape(num_blocks, block_size)
                # shared_exp should match the block structure
                # If shared_exp is per-block, it should have shape matching num_blocks
                if shared_exp_np.size == num_blocks:
                    shared_exp_reshaped = shared_exp_np.reshape(-1)
                elif shared_exp_np.size == 1:
                    # Broadcast single shared_exp to all blocks
                    shared_exp_reshaped = np.full(num_blocks, shared_exp_np.item())
                else:
                    # Try to reshape to match blocks
                    shared_exp_reshaped = shared_exp_np.flatten()[:num_blocks]
                    if len(shared_exp_reshaped) < num_blocks:
                        # Pad with last value
                        shared_exp_reshaped = np.pad(shared_exp_reshaped, (0, num_blocks - len(shared_exp_reshaped)), 
                                                     mode='edge')
                
                # Sample blocks to avoid memory issues (max 1000 blocks)
                # Use fixed indices (first N blocks) instead of random for reproducibility
                max_blocks = num_blocks
                if num_blocks > max_blocks:
                    block_indices = np.arange(max_blocks)  # Fixed: use first max_blocks blocks
                    A_sampled = A_reshaped[block_indices]
                    shared_exp_sampled = shared_exp_reshaped[block_indices]
                else:
                    A_sampled = A_reshaped
                    shared_exp_sampled = shared_exp_reshaped
                
                # Store per-block data: list of arrays, each array is one block (32 elements)
                pre_scaling_blocks = [row.tolist() for row in A_sampled]
                pre_scaling_shared_exp = shared_exp_sampled.tolist() if isinstance(shared_exp_sampled, np.ndarray) else [float(shared_exp_sampled)]
        
        # Also collect flattened data for statistics (backward compatibility)
        A_before_flat = A_before_np.flatten()
        
        # Filter outliers (|value| > 6) for analysis
        mask_before = np.abs(A_before_flat) <= 6.0
        A_before_filtered = A_before_flat[mask_before]
        outliers_count_before = np.sum(~mask_before)
        total_count_before = len(A_before_flat)
        
        # Calculate statistics
        pre_scaling_stats = {
            'total_elements': int(total_count_before),
            'outliers_6plus_count': int(outliers_count_before),
            'outliers_6plus_percent': float(outliers_count_before / total_count_before * 100) if total_count_before > 0 else 0.0,
            'filtered_elements': int(len(A_before_filtered)),
            'min': float(np.min(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'max': float(np.max(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'mean': float(np.mean(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'std': float(np.std(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0,
            'median': float(np.median(A_before_filtered)) if len(A_before_filtered) > 0 else 0.0
        }
        
        # Store pre-scaling data for later plotting (flattened, for backward compatibility)
        # Sample to save memory - use fixed indices (first N elements) for reproducibility
        max_samples = min(100000, len(A_before_filtered))
        if len(A_before_filtered) > max_samples:
            indices = np.arange(max_samples)  # Fixed: use first max_samples elements
            pre_scaling_data = A_before_filtered[indices].tolist()
        else:
            pre_scaling_data = A_before_filtered.tolist() if len(A_before_filtered) > 0 else None
    
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
        # Sample at most 100k points to avoid memory issues - use fixed indices for reproducibility
        max_samples = min(100000, len(A_after_filtered))
        if len(A_after_filtered) > max_samples:
            indices = np.arange(max_samples)  # Fixed: use first max_samples elements
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
            distribution_stats['pre_scaling']['data'] = pre_scaling_data
        # Add per-block data for scatter plot
        if pre_scaling_blocks is not None:
            distribution_stats['pre_scaling']['blocks'] = pre_scaling_blocks
            distribution_stats['pre_scaling']['shared_exp'] = pre_scaling_shared_exp
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
                                      num_workers=32, max_plots=10, block_idx=None):
    """
    Analyze value distribution for tensor file(s).
    Supports both single file and folder.
    Uses multithreading to speed up processing.
    Only processes minus_exp=0.
    
    Args:
        folder_path (str): Path to folder containing .pt tensor files, or path to a single .pt file
        elem_format (str): Element format (default: 'fp4_e2m1')
        target_values (list): List of target values to analyze
        output_dir (str): Output directory for plots (default: ./draw/value_distribution/)
        scale_bits (int): Number of scale bits
        block_size (int): Block size for tiling
        axes (int): Axes for shared exponent calculation
        num_workers (int): Number of worker threads for parallel processing (default: 32).
                           Recommended: 0.25-0.5x CPU cores for CPU-bound tasks, or 0.5-1x for I/O-bound tasks.
                           Adjust based on available memory and tensor sizes.
        max_plots (int): Maximum number of blocks to plot (default: 10). Limits the number of scatter plots.
        block_idx (int): Optional. If specified, only plot the block at this index (1-based).
                         If None, plot all blocks up to max_plots.
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
    
    input_path = Path(folder_path)
    if not input_path.exists():
        raise ValueError(f"Path does not exist: {input_path}")
    
    # Check if it's a file or directory
    if input_path.is_file():
        # Single file
        tensor_files = [input_path]
        print(f"Processing single tensor file: {input_path.name}")
        # Setup output directory based on file name
        if output_dir is None:
            output_dir = Path("./draw/value_distribution") / input_path.stem
        else:
            output_dir = Path(output_dir)
    elif input_path.is_dir():
        # Directory
        tensor_files = list(input_path.glob("*.pt"))
        if not tensor_files:
            print(f"No .pt files found in {input_path}")
            return
        print(f"Found {len(tensor_files)} tensor files in {input_path}")
        # Setup output directory based on folder name
        if output_dir is None:
            output_dir = Path("./draw/value_distribution") / input_path.name
        else:
            output_dir = Path(output_dir)
    else:
        raise ValueError(f"Path is neither a file nor a directory: {input_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing minus_exp = 0")
    
    # Fixed minus_exp value
    minus_exp = 0
    
    # Collect per-block data for scatter plot
    all_pre_scaling_blocks = []  # Collect per-block original data
    all_pre_scaling_shared_exp = []  # Collect shared exponents for each block
    
    print(f"\n{'='*60}")
    print(f"Processing minus_exp = {minus_exp}")
    print(f"{'='*60}")
    
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
                    # Collect per-block data for scatter plot
                    if result['distribution']:
                        dist = result['distribution']
                        if 'pre_scaling' in dist:
                            # Collect per-block data for scatter plot
                            if 'blocks' in dist['pre_scaling'] and 'shared_exp' in dist['pre_scaling']:
                                all_pre_scaling_blocks.extend(dist['pre_scaling']['blocks'])
                                all_pre_scaling_shared_exp.extend(dist['pre_scaling']['shared_exp'])
                else:
                    if result['error']:
                        print(f"  ⚠️  {result['tensor_name']}: {result['error']}")
                    if 'traceback' in result:
                        print(f"  ❌ Error in {result['tensor_name']}:")
                        print(result['traceback'])
                
                pbar.update(1)
    
    print(f"\nSuccessfully processed {len(all_pre_scaling_blocks)} blocks collected")
    
    # Handle block_idx mode (plot specific block)
    if block_idx is not None:
        # block_idx is 1-based, convert to 0-based
        if block_idx < 1 or block_idx > len(all_pre_scaling_blocks):
            print(f"⚠️  Error: block_idx={block_idx} is out of range. Available blocks: 1-{len(all_pre_scaling_blocks)}")
            return {}
        block_idx_0based = block_idx - 1
        print(f"Plotting specific block {block_idx} (out of {len(all_pre_scaling_blocks)} total blocks)")
        plot_blocks = [all_pre_scaling_blocks[block_idx_0based]]
        plot_shared_exp = [all_pre_scaling_shared_exp[block_idx_0based]]
        num_blocks_to_plot = 1
    else:
        # Limit the number of blocks to plot (normal mode)
        num_blocks_to_plot = min(max_plots, len(all_pre_scaling_blocks))
        if len(all_pre_scaling_blocks) > max_plots:
            print(f"Limiting plots to first {max_plots} blocks (out of {len(all_pre_scaling_blocks)} total)")
            # Sample blocks to plot
            plot_blocks = all_pre_scaling_blocks[:num_blocks_to_plot]
            plot_shared_exp = all_pre_scaling_shared_exp[:num_blocks_to_plot]
        else:
            plot_blocks = all_pre_scaling_blocks
            plot_shared_exp = all_pre_scaling_shared_exp
    
    # Create per-block scatter plot
    print("\n" + "=" * 60)
    print(f"Creating per-block scatter plot ({num_blocks_to_plot} blocks)...")
    print("=" * 60)
    
    if len(plot_blocks) > 0:
        try:
            # Get block_size from first block
            block_size = len(plot_blocks[0])
            
            # MXFP4 representable values (before scaling, i.e., multiplied by 2**shared_exp)
            mxfp4_values = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
            # Additional values to mark
            additional_values = [-0.75, -0.25, 0.25, 0.75]
            all_values_to_mark = mxfp4_values + additional_values
            
            # Get plot name for output files
            if input_path.is_file():
                plot_name = input_path.stem
            else:
                plot_name = input_path.name
            
            # Create a separate plot for each block and calculate MSE
            saved_plots = []
            mse_results = []
            
            # Track the block with maximum MSE difference
            max_mse_diff = -float('inf')
            max_mse_diff_block_info = None
            
            # Define value sets
            red_values = [-6,-4,4,6]
            green_values = [-0.75, -0.25, 0.25, 0.75]
            gray_values = [-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3]
            
            for plot_block_idx, (block_data, shared_exp_val) in enumerate(zip(plot_blocks, plot_shared_exp)):
                # Calculate actual block index
                if block_idx is not None:
                    # In block_idx mode, use the specified block_idx (1-based)
                    actual_block_idx = block_idx
                    # In block_idx mode, create two separate figures instead of subplots
                    use_separate_figures = True
                else:
                    # In normal mode, use enumerate index + 1 (1-based)
                    actual_block_idx = plot_block_idx + 1
                    # In normal mode, use subplots
                    use_separate_figures = False
                
                # Create figure(s) based on mode - optimized for figure* environment (double column)
                # Use GridSpec for flexible layout with error info panel (tight spacing, flatter aspect)
                if use_separate_figures:
                    # Create two separate figures for block_idx mode with error info panel
                    fig1 = plt.figure(figsize=(7.0, 2.0))
                    gs1 = GridSpec(1, 2, figure=fig1, width_ratios=[6, 1.2], hspace=0.3, wspace=0.1)
                    ax1 = fig1.add_subplot(gs1[0, 0])
                    ax1_info = fig1.add_subplot(gs1[0, 1])
                    ax1_info.axis('off')
                    
                    fig2 = plt.figure(figsize=(7.0, 2.0))
                    gs2 = GridSpec(1, 2, figure=fig2, width_ratios=[6, 1.2], hspace=0.3, wspace=0.1)
                    ax2 = fig2.add_subplot(gs2[0, 0])
                    ax2_info = fig2.add_subplot(gs2[0, 1])
                    ax2_info.axis('off')
                else:
                    # Create figure with two subplots for normal mode with error info panels
                    fig = plt.figure(figsize=(14.0, 2.0))
                    gs = GridSpec(1, 4, figure=fig, width_ratios=[6, 1.2, 6, 1.2], hspace=0.3, wspace=0.1)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1_info = fig.add_subplot(gs[0, 1])
                    ax1_info.axis('off')
                    ax2 = fig.add_subplot(gs[0, 2])
                    ax2_info = fig.add_subplot(gs[0, 3])
                    ax2_info.axis('off')
                
                block_array = np.array(block_data)
                x_positions = np.arange(block_size)  # 0 to block_size-1
                

                
                # Scale factors: S for left (gray+red), S/2 for right (gray+green)
                scale_factor_s = 2 ** shared_exp_val  # S
                scale_factor_s2 = 2 ** (shared_exp_val - 1)  # S/2
                
                # ========== Left subplot: Gray + Red (S) ==========
                # Plot scatter points
                
                # Calculate representable values for gray+red scheme (S)
                red_representable_s = [v * scale_factor_s for v in red_values]
                gray_representable_s = [v * scale_factor_s for v in gray_values]
                
                # Mark gray values
                for val in gray_values:
                    representable_val = val * scale_factor_s
                    ax1.axhline(representable_val, color='gray', linestyle='--', 
                               linewidth=0.5, alpha=0.4, zorder=1)
                
                # Mark red values
                for val in red_values:
                    representable_val = val * scale_factor_s
                    ax1.axhline(representable_val, color='red', linestyle='--', 
                               linewidth=0.5, alpha=0.4, zorder=1)
                
                # Set labels and styling for left subplot - optimized for figure* environment
                ax1.set_xlabel('Position in Block', fontsize=8, fontweight='normal')
                ax1.set_ylabel('Original Value', fontsize=8, fontweight='normal')
                ax1.tick_params(axis='both', which='major', labelsize=7)
                ax1.tick_params(axis='both', which='minor', labelsize=6)
                ax1.grid(False)
                # Removed title as requested
                ax1.set_xlim(-0.5, block_size - 0.5)
                

                
                # ========== Right subplot: Gray + Green (S/2) ==========
                # Plot scatter points
                
                # Calculate representable values for gray+green scheme (S/2)
                green_representable_s2 = [v * scale_factor_s for v in green_values]
                gray_representable_s2 = [v * scale_factor_s for v in gray_values]
                
                # Mark gray values
                for val in gray_values:
                    representable_val = val * scale_factor_s
                    ax2.axhline(representable_val, color='gray', linestyle='--', 
                               linewidth=0.5, alpha=0.4, zorder=1)
                
                # Mark green values
                for val in green_values:
                    representable_val = val * scale_factor_s
                    ax2.axhline(representable_val, color='green', linestyle='--', 
                               linewidth=0.5, alpha=0.4, zorder=1)
                
                # Set labels and styling for right subplot - optimized for figure* environment
                ax2.set_xlabel('Position in Block', fontsize=8, fontweight='normal')
                ax2.set_ylabel('Original Value', fontsize=8, fontweight='normal')
                ax2.tick_params(axis='both', which='major', labelsize=7)
                ax2.tick_params(axis='both', which='minor', labelsize=6)
                ax2.grid(False)
                # Removed title as requested
                ax2.set_xlim(-0.5, block_size - 0.5)
                
                # Calculate MSE for red+gray scheme (using S)
                red_gray_representable = sorted(red_representable_s + gray_representable_s)
                red_gray_quantized = np.array([min(red_gray_representable, key=lambda x: abs(x - v)) 
                                               for v in block_array])
                red_gray_mse = float(np.mean((block_array - red_gray_quantized) ** 2))*4
                
                # Calculate MSE for gray+green scheme (using S/2)
                gray_green_representable = sorted(gray_representable_s2 + green_representable_s2)
                # Use numpy for efficient nearest value finding
                gray_green_representable_arr = np.array(gray_green_representable)
                # Find nearest representable value for each point
                gray_green_quantized = np.array([
                    gray_green_representable_arr[np.argmin(np.abs(gray_green_representable_arr - v))]
                    for v in block_array
                ])
                
                # For gray+green, distinguish clipping and rounding errors
                # Clipping: positive values > max(gray_representable) or negative values < min(gray_representable)
                # The boundary is based on gray values only (not including green values)
                # Use S/2 scale for gray values in gray+green scheme
                gray_representable_arr_s2 = np.array(gray_representable_s2)
                gray_min = float(np.min(gray_representable_arr_s2))  # Smallest gray representable value
                gray_max = float(np.max(gray_representable_arr_s2))  # Largest gray representable value
                
                # Find values that need clipping:
                # - Positive values greater than the largest gray value
                # - Negative values smaller than the smallest gray value
                clip_mask = (block_array > gray_max) | (block_array < gray_min)
                round_mask = ~clip_mask
                
                # Clipping error: values outside range (but still rounded to nearest representable)
                if np.any(clip_mask):
                    clip_errors = block_array[clip_mask] - gray_green_quantized[clip_mask]
                    clip_mse = float(np.sum(clip_errors ** 2))/32
                    clip_count = int(np.sum(clip_mask))
                else:
                    clip_mse = 0.0
                    clip_count = 0
                
                # Rounding error: values inside range (rounded to nearest representable)
                if np.any(round_mask):
                    round_errors = block_array[round_mask] - gray_green_quantized[round_mask]
                    round_mse = float(np.sum(round_errors ** 2))*4/32
                    round_count = int(np.sum(round_mask))
                    max_rounding_error = float(np.max(np.abs(round_errors)))
                    mean_rounding_error = float(np.mean(np.abs(round_errors)))
                else:
                    round_mse = 0.0
                    round_count = 0
                    max_rounding_error = 0.0
                    mean_rounding_error = 0.0
                
                max_abs_value = float(np.max(np.abs(block_array)))
                gray_green_mse = round_mse + clip_mse
                # Calculate average MSE for each scheme
                avg_mse_red_gray = red_gray_mse  # For Gray + Red scheme, avg_mse is the total MSE
                avg_mse_gray_green = gray_green_mse  # For Gray + Green scheme, avg_mse is the total MSE
                # Plot scatter points with larger size and higher alpha for better visibility
                ax1.scatter(x_positions, block_array, s=8, alpha=0.6, c='#1f77b4', zorder=2, edgecolors='none')
                ax2.scatter(x_positions, block_array, s=8, alpha=0.6, c='#1f77b4', zorder=2, edgecolors='none')

                if red_gray_mse < gray_green_mse:
                    print(f"Block {actual_block_idx} skipped because red_gray_mse < gray_green_mse")
                    if use_separate_figures:
                        plt.close(fig1)
                        plt.close(fig2)
                    else:
                        plt.close()
                    continue
                else:
                    # Track maximum MSE difference (gray_green_mse - red_gray_mse)
                    mse_diff = red_gray_mse - gray_green_mse
                    if mse_diff > max_mse_diff :
                        max_mse_diff = mse_diff
                        max_mse_diff_block_info = {
                            'block_idx': actual_block_idx,
                            'mse_difference': float(mse_diff),
                            'red_gray_mse': float(red_gray_mse),
                            'gray_green_mse': float(gray_green_mse),
                            'block_data': block_array.tolist(),
                            'shared_exp': float(shared_exp_val),
                            'scale_factor_s': float(scale_factor_s),
                            'scale_factor_s2': float(scale_factor_s2),
                            'max_abs_value': float(max_abs_value),
                            'gray_green_clip_mse': float(clip_mse),
                            'gray_green_round_mse': float(round_mse),
                            'gray_green_clip_count': int(clip_count),
                            'gray_green_round_count': int(round_count),
                            'max_rounding_error': float(max_rounding_error),
                            'mean_rounding_error': float(mean_rounding_error)
                        }
                
                # Add MSE error info as beautified text panel next to plots
                # Left subplot (Gray + Red): clip_mse=0, round_mse, avg_mse (red_gray_mse)
                # Enhanced formatting with better visual hierarchy
                info_text_1 = (
                    f'Error Analysis\n'
                    f'{"─" * 16}\n'
                    f'Clip Error\n'
                    f'{0.0:>11.6f}\n'
                    f'\n'
                    f'Round Error\n'
                    f'{red_gray_mse:>11.6f}\n'
                    f'{"─" * 16}\n'
                    f'Total Error\n'
                    f'{avg_mse_red_gray:>11.6f}'
                )
                ax1_info.text(0.05, 0.98, info_text_1, transform=ax1_info.transAxes, 
                             fontsize=7, verticalalignment='top', horizontalalignment='left',
                             family='monospace', 
                             bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='#f8f9fa', alpha=0.98, 
                             edgecolor='#495057', linewidth=0.7,
                             linestyle='-'))
                
                # Right plot (Gray + Green): clip_mse, round_mse, avg_mse (gray_green_mse)
                info_text_2 = (
                    f'Error Analysis\n'
                    f'{"─" * 16}\n'
                    f'Clip Error\n'
                    f'{clip_mse:>11.6f}\n'
                    f'\n'
                    f'Round Error\n'
                    f'{round_mse:>11.6f}\n'
                    f'{"─" * 16}\n'
                    f'Total Error\n'
                    f'{avg_mse_gray_green:>11.6f}'
                )
                ax2_info.text(0.05, 0.98, info_text_2, transform=ax2_info.transAxes, 
                             fontsize=7, verticalalignment='top', horizontalalignment='left',
                             family='monospace',
                             bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='#f8f9fa', alpha=0.98, 
                             edgecolor='#495057', linewidth=0.7,
                             linestyle='-'))
                
                # Save plots based on mode
                if use_separate_figures:
                    # Save two separate PDF files for block_idx mode
                    # Improve layout and save first plot (Red+Gray) - optimized for figure* environment
                    fig1.tight_layout(pad=0.5)
                    fig1.patch.set_facecolor('white')
                    block_plot_path_1 = output_dir / f'per_block_{actual_block_idx:04d}_red_gray_{elem_format}_{plot_name}.pdf'
                    fig1.savefig(block_plot_path_1, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close(fig1)
                    saved_plots.append(block_plot_path_1)
                    
                    # Improve layout and save second plot (Gray+Green) - optimized for figure* environment
                    fig2.tight_layout(pad=0.5)
                    fig2.patch.set_facecolor('white')
                    block_plot_path_2 = output_dir / f'per_block_{actual_block_idx:04d}_gray_green_{elem_format}_{plot_name}.pdf'
                    fig2.savefig(block_plot_path_2, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close(fig2)
                    saved_plots.append(block_plot_path_2)
                    pdf_files = f"{block_plot_path_1.name}, {block_plot_path_2.name}"
                    print(f"✅ Block {actual_block_idx} plots saved: {block_plot_path_1.name}, {block_plot_path_2.name}")
                else:
                    # Save single PDF file with two subplots for normal mode - optimized for figure* environment
                    plt.tight_layout(pad=0.5)
                    fig.patch.set_facecolor('white')
                    block_plot_path = output_dir / f'per_block_{actual_block_idx:04d}_{elem_format}_{plot_name}.pdf'
                    plt.savefig(block_plot_path, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close()
                    saved_plots.append(block_plot_path)
                    pdf_files = str(block_plot_path.name)
                
                # Store MSE results
                
                mse_results.append({
                    'block_idx': actual_block_idx,
                    'pdf_file': pdf_files,
                    'max_abs_value': max_abs_value,
                    'red_gray_mse': red_gray_mse,
                    'gray_green_mse': gray_green_mse,
                    'avg_mse_red_gray': float(avg_mse_red_gray),  # avg_mse for Gray + Red scheme
                    'avg_mse_gray_green': float(avg_mse_gray_green),  # avg_mse for Gray + Green scheme
                    'gray_green_clip_mse': clip_mse,
                    'gray_green_round_mse': round_mse,
                    'gray_green_clip_count': clip_count,
                    'gray_green_round_count': round_count,
                    'shared_exp': float(shared_exp_val),
                    'scale_factor_s': float(scale_factor_s),  # S for gray+red
                    'scale_factor_s2': float(scale_factor_s2),  # S/2 for gray+green
                    'num_representable_values_gray_red': len(red_gray_representable),
                    'num_representable_values_gray_green': len(gray_green_representable),
                    'max_rounding_error': max_rounding_error,
                    'mean_rounding_error': mean_rounding_error
                })
                continue
            
            # Save MSE results to JSON
            json_output_path = output_dir / f'mse_results_{elem_format}_{plot_name}.json'
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(mse_results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ MSE results saved to: {json_output_path}")
            
            # Save maximum MSE difference block info and plot
            if max_mse_diff_block_info is not None:
                max_mse_diff_path = output_dir / f'max_mse_diff_block_{elem_format}_{plot_name}.json'
                with open(max_mse_diff_path, 'w', encoding='utf-8') as f:
                    json.dump(max_mse_diff_block_info, f, indent=2, ensure_ascii=False)
                print(f"✅ Maximum MSE difference block saved to: {max_mse_diff_path}")
                print(f"   Block {max_mse_diff_block_info['block_idx']}: MSE difference = {max_mse_diff_block_info['mse_difference']:.6f}")
                
                # Plot the maximum MSE difference block using the same logic
                # Create two separate figures instead of subplots
                try:
                    # Extract data from saved info
                    block_data = np.array(max_mse_diff_block_info['block_data'])
                    shared_exp_val = max_mse_diff_block_info['shared_exp']
                    scale_factor_s = max_mse_diff_block_info['scale_factor_s']
                    scale_factor_s2 = max_mse_diff_block_info['scale_factor_s2']
                    red_gray_mse = max_mse_diff_block_info['red_gray_mse']
                    gray_green_mse = max_mse_diff_block_info['gray_green_mse']
                    clip_mse = max_mse_diff_block_info['gray_green_clip_mse']
                    round_mse = max_mse_diff_block_info['gray_green_round_mse']
                    avg_mse_red_gray = red_gray_mse
                    avg_mse_gray_green = gray_green_mse
                    
                    x_positions = np.arange(block_size)  # 0 to block_size-1
                    
                    # ========== First figure: Gray + Red (S) ==========
                    # Use GridSpec for flexible layout with error info panel
                    fig1 = plt.figure(figsize=(7.0, 2.0))
                    gs1 = GridSpec(1, 2, figure=fig1, width_ratios=[6, 1.2], hspace=0.3, wspace=0.1)
                    ax1 = fig1.add_subplot(gs1[0, 0])
                    ax1_info = fig1.add_subplot(gs1[0, 1])
                    ax1_info.axis('off')
                    
                    # Mark gray values
                    for val in gray_values:
                        representable_val = val * scale_factor_s
                        ax1.axhline(representable_val, color='gray', linestyle='--', 
                                   linewidth=0.5, alpha=0.4, zorder=1)
                    
                    # Mark red values
                    for val in red_values:
                        representable_val = val * scale_factor_s
                        ax1.axhline(representable_val, color='red', linestyle='--', 
                                   linewidth=0.5, alpha=0.4, zorder=1)
                    
                    # Set labels and styling - optimized for figure* environment
                    ax1.set_xlabel('Position in Block', fontsize=8, fontweight='normal')
                    ax1.set_ylabel('Original Value', fontsize=8, fontweight='normal')
                    ax1.tick_params(axis='both', which='major', labelsize=7)
                    ax1.tick_params(axis='both', which='minor', labelsize=6)
                    ax1.grid(False)
                    # Removed title as requested
                    ax1.set_xlim(-0.5, block_size - 0.5)
                    
                    # Plot scatter points with larger size and higher alpha for better visibility
                    ax1.scatter(x_positions, block_data, s=8, alpha=0.6, c='#1f77b4', zorder=2, edgecolors='none')
                    
                    # Add MSE error info as beautified text panel next to plot
                    info_text_1 = (
                        f'Error Analysis\n'
                        f'{"─" * 16}\n'
                        f'Clip Error\n'
                        f'{0.0:>11.6f}\n'
                        f'\n'
                        f'Round Error\n'
                        f'{red_gray_mse:>11.6f}\n'
                        f'{"─" * 16}\n'
                        f'Total Error\n'
                        f'{avg_mse_red_gray:>11.6f}'
                    )
                    ax1_info.text(0.05, 0.98, info_text_1, transform=ax1_info.transAxes, 
                                 fontsize=7, verticalalignment='top', horizontalalignment='left',
                                 family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='#f8f9fa', alpha=0.98, 
                                 edgecolor='#495057', linewidth=0.7,
                                 linestyle='-'))
                    
                    # Improve layout - optimized for figure* environment
                    plt.tight_layout(pad=0.5)
                    
                    # Background for figure
                    fig1.patch.set_facecolor('white')
                    
                    # Save first plot
                    max_mse_diff_plot_path_1 = output_dir / f'max_mse_diff_block_red_gray_{elem_format}_{plot_name}.pdf'
                    plt.savefig(max_mse_diff_plot_path_1, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close(fig1)
                    print(f"✅ Maximum MSE difference block plot (Red+Gray) saved to: {max_mse_diff_plot_path_1}")
                    
                    # ========== Second figure: Gray + Green (S/2) ==========
                    # Use GridSpec for flexible layout with error info panel
                    fig2 = plt.figure(figsize=(7.0, 2.0))
                    gs2 = GridSpec(1, 2, figure=fig2, width_ratios=[6, 1.2], hspace=0.3, wspace=0.1)
                    ax2 = fig2.add_subplot(gs2[0, 0])
                    ax2_info = fig2.add_subplot(gs2[0, 1])
                    ax2_info.axis('off')
                    
                    # Mark gray values
                    for val in gray_values:
                        representable_val = val * scale_factor_s
                        ax2.axhline(representable_val, color='gray', linestyle='--', 
                                   linewidth=0.5, alpha=0.4, zorder=1)
                    
                    # Mark green values
                    for val in green_values:
                        representable_val = val * scale_factor_s
                        ax2.axhline(representable_val, color='green', linestyle='--', 
                                   linewidth=0.5, alpha=0.4, zorder=1)
                    
                    # Set labels and styling - optimized for figure* environment
                    ax2.set_xlabel('Position in Block', fontsize=8, fontweight='normal')
                    ax2.set_ylabel('Original Value', fontsize=8, fontweight='normal')
                    ax2.tick_params(axis='both', which='major', labelsize=7)
                    ax2.tick_params(axis='both', which='minor', labelsize=6)
                    ax2.grid(False)
                    # Removed title as requested
                    ax2.set_xlim(-0.5, block_size - 0.5)
                    
                    # Plot scatter points with larger size and higher alpha for better visibility
                    ax2.scatter(x_positions, block_data, s=8, alpha=0.6, c='#1f77b4', zorder=2, edgecolors='none')
                    
                    # Add MSE error info as beautified text panel next to plot
                    info_text_2 = (
                        f'Error Analysis\n'
                        f'{"─" * 16}\n'
                        f'Clip Error\n'
                        f'{clip_mse:>11.6f}\n'
                        f'\n'
                        f'Round Error\n'
                        f'{round_mse:>11.6f}\n'
                        f'{"─" * 16}\n'
                        f'Total Error\n'
                        f'{avg_mse_gray_green:>11.6f}'
                    )
                    ax2_info.text(0.05, 0.98, info_text_2, transform=ax2_info.transAxes, 
                                 fontsize=7, verticalalignment='top', horizontalalignment='left',
                                 family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='#f8f9fa', alpha=0.98, 
                                 edgecolor='#495057', linewidth=0.7,
                                 linestyle='-'))
                    
                    # Improve layout - optimized for figure* environment
                    plt.tight_layout(pad=0.5)
                    
                    # Background for figure
                    fig2.patch.set_facecolor('white')
                    
                    # Save second plot
                    max_mse_diff_plot_path_2 = output_dir / f'max_mse_diff_block_gray_green_{elem_format}_{plot_name}.pdf'
                    plt.savefig(max_mse_diff_plot_path_2, dpi=600, bbox_inches='tight', pad_inches=0.02, 
                               facecolor='white', edgecolor='none')
                    plt.close(fig2)
                    print(f"✅ Maximum MSE difference block plot (Gray+Green) saved to: {max_mse_diff_plot_path_2}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create plot for maximum MSE difference block: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n✅ Created {len(saved_plots)} per-block scatter plots")
            print(f"   First plot: {saved_plots[0]}")
            if len(saved_plots) > 1:
                print(f"   Last plot: {saved_plots[-1]}")
        except Exception as e:
            print(f"Warning: Failed to create per-block distance plots: {e}")
            import traceback
            traceback.print_exc()
    
    return {}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze per-block scatter plot for fp4_e2m1 quantization (minus_exp=0 only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze single tensor file
  python utils/saver/mxfp_formater_block.py data/real/bwd_quant_temp_140541200614960.pt
  
  # Analyze folder with multiple files
  python utils/saver/mxfp_formater_block.py data/bf16
  
  # Limit to 5 plots
  python utils/saver/mxfp_formater_block.py data/bf16 --max-plots 5
  
  # Plot specific block (e.g., block 10)
  python utils/saver/mxfp_formater_block.py data/bf16 --block-idx 10
        '''
    )
    parser.add_argument('folder_path', type=str, 
                        help='Path to folder containing .pt tensor files, or path to a single .pt file')
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
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of worker threads for parallel processing (default: 32). '
                             'Recommended: 0.25-0.5x CPU cores for CPU-bound tasks, or 0.5-1x for I/O-bound tasks. '
                             'Adjust based on available memory and tensor sizes.')
    parser.add_argument('--max-plots', type=int, default=10,
                        help='Maximum number of blocks to plot (default: 10). Limits the number of scatter plots.')
    parser.add_argument('--block-idx', type=int, default=None,
                        help='Optional. If specified, only plot the block at this index (1-based). '
                             'If not specified, plot all blocks up to max-plots.')
    
    args = parser.parse_args()
    
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
        num_workers=args.num_workers,
        max_plots=args.max_plots,
        block_idx=args.block_idx
    )