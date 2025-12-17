import torch
from enum import Enum, IntEnum
import numpy as np
import math

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
        assert s != None, "String elem_format == None"
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
    assert ebits >= 5, "invalid for floats that don't define NaN"
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
    elif fmt == ElemFormat.fp4_e2m1:
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

def _get_representable_values(elem_format):
    """
    Generates all positive, non-zero representable values for a given float format.
    """
    ebits, mbits, _, _, _ = _get_format_params(elem_format)
    if ebits == 0: # Not a float format
        return []

    mbits_explicit = mbits - 2
    bias = 2**(ebits - 1) - 1
    
    values = set()

    # Denormalized numbers
    E_val = 1 - bias
    for m_val in range(1, 2**mbits_explicit):
        mantissa_val = m_val / (2**mbits_explicit)
        values.add((2**E_val) * mantissa_val)

    # Normalized numbers
    for e_val in range(1, 2**ebits):
        E_val = e_val - bias
        for m_val in range(0, 2**mbits_explicit):
            mantissa_val = 1 + m_val / (2**mbits_explicit)
            values.add((2**E_val) * mantissa_val)
            
    return sorted(list(values))

# def _calculate_kappa(elem_format: str, distribution: str='gaussian') -> float:
#     """
#     Calculates the format-specific constant kappa based on the quantization points
#     and the data distribution, as described in the paper.
#     """
#     # 1. Get the positive, non-zero representable values (q_i)
#     q_values = _get_representable_values(elem_format)
#     # import pdb; pdb.set_trace()
#     if not q_values:
#         return 2.0 # Default for non-float formats

#     # 2. Define the probability density function p(x) for the distribution
#     if distribution.lower() == 'gaussian':
#         def p(x):
#             return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2.0)
#     else:
#         # For now, only Gaussian is implemented in detail
#         raise ValueError(f"Detailed kappa calculation for distribution '{distribution}' is not implemented.")

#     # 3. Calculate the total rounding error D_round
#     total_error = 0.0
#     q_with_zero = [0.0] + q_values

#     # Iterate through each quantization point to calculate its error contribution
#     for i in range(1, len(q_with_zero)):
#         q_i = q_with_zero[i]
        
#         # 4. Calculate left and right intervals (delta_L, delta_R)
#         delta_L = q_i - q_with_zero[i-1]
        
#         if i == len(q_with_zero) - 1:
#             # For the last point, assume the right interval is symmetric to the left
#             # This is a simplification for the clipping region
#             delta_R = delta_L
#         else:
#             delta_R = q_with_zero[i+1] - q_i
            
#         # 5. Calculate error contribution using the formula from the paper
#         # D_round ≈ Σ p(q_i) * ( (Δ_i,R³ + Δ_i,L³) / 24 )
#         prob = p(q_i)
#         error_contrib = prob * (delta_L**3 + delta_R**3) / 24.0
#         total_error += error_contrib
        
#     # 6. Multiply by 2 for negative values and return kappa
#     # For S=1 and Energy=1, kappa is approximately D_round
#     kappa = 2.0 * total_error
#     return kappa

# if __name__ == '__main__':
#     print("Demonstrating the new _calculate_kappa function:")
    
#     # Example for fp4_e2m1 as discussed
#     kappa_fp4 = _calculate_kappa('fp4_e2m1', 'gaussian')
#     print(f"Calculated kappa for fp4_e2m1 (Gaussian): {kappa_fp4}")
#     kappa_fp8_e4m3 = _calculate_kappa('fp8_e4m3', 'gaussian')
#     print(f"Calculated kappa for fp8_e4m3 (Gaussian): {kappa_fp8_e4m3}")
#     kappa_fp8_e5m2 = _calculate_kappa('fp8_e5m2', 'gaussian')
#     print(f"Calculated kappa for fp8_e5m2 (Gaussian): {kappa_fp8_e5m2}")


# 新增一个缓存来存储预计算的 Kappa 常量
_KAPPA_CONSTANTS_CACHE = {}

def _get_kappa_constants(elem_format: str,distribution: str='gaussian'):
    if type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)
    
    # 如果缓存中已有结果，直接返回
    if elem_format in _KAPPA_CONSTANTS_CACHE:
        return _KAPPA_CONSTANTS_CACHE[elem_format]

    # 获取量化值
    q_values = _get_representable_values(elem_format)
    if not q_values:
        # 对于非浮点格式，缓存一个空列表并返回
        _KAPPA_CONSTANTS_CACHE[elem_format] = []
        return []

    constants = []
    q_with_zero = [0.0] + q_values
    
    # 循环一次，计算所有仅与格式相关的常量
    for i in range(1, len(q_with_zero)):
        q_i = q_with_zero[i]
        
        delta_L = q_i - q_with_zero[i-1]
        
        if i == len(q_with_zero) - 1:
            delta_R = delta_L  # 最后一个点的简化处理
        else:
            delta_R = q_with_zero[i+1] - q_i
            
        # C_i 是仅依赖于数据格式的常量
        C_i = (delta_L**3 + delta_R**3) / 24.0
        constants.append((q_i, C_i))
        
    # 将结果存入缓存
    _KAPPA_CONSTANTS_CACHE[elem_format] = constants
    return constants

def _calculate_kappa(elem_format: str, distribution: str='gaussian', miu: float = 0.0, gamma: float = 1.0) -> float:
    kappa_constants = _get_kappa_constants(elem_format,distribution)
    if not kappa_constants:
        return 2.0  
    pdf_norm_factor = 1.0 / (math.sqrt(gamma * 2 * math.pi))

    total_error = 0.0
    for q_i, C_i in kappa_constants:
        # exponent = -((q_i - miu)**2) / (2 * gamma)
        exponent = -(q_i**2 - 2*miu*q_i + miu**2) / (2 * gamma)
        prob = pdf_norm_factor * math.exp(exponent)
        
        total_error += prob * C_i
        
    kappa = 2.0 * total_error
    return kappa

if __name__ == '__main__':
    kappa1 = _calculate_kappa('fp4_e2m1','gaussian', miu=0.2, gamma=1.0)
    v1=math.pow(2/kappa1,1/3)
    print(f"Calculated kappa for fp4_e2m1 (μ=0.2, γ=1): {kappa1}, v1: {v1}")
    
    kappa2 = _calculate_kappa('fp4_e2m1','gaussian', miu=0.5, gamma=1.0)
    v2=math.pow(2/kappa2,1/3)
    print(f"Calculated kappa for fp4_e2m1 (μ=0.5, γ=1): {kappa2}, v2: {v2}")

    # 第二次调用相同格式时，会直接使用缓存，速度更快
    kappa3 = _calculate_kappa('fp4_e2m1','gaussian', miu=5, gamma=1.0)
    v3=math.pow(2/kappa3,1/3)
    print(f"Calculated kappa for fp4_e2m1 (μ=5, γ=1): {kappa3}, v3: {v3}")
