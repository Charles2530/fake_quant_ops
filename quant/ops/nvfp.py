import torch
from torch import Tensor
from torch.autograd import Function
from typing import Optional, Dict, Union
import re

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def _get_nvfp_params(nvfp_format: str):
    """
    Get NVFP format parameters and corresponding PyTorch dtype
    
    Supported formats:
    - 'nvfp4' or 'nvfp4_e2m1': 4-bit, E2M1 format
    - 'nvfp8_e4m3': 8-bit, E4M3 format (default for nvfp8)
    - 'nvfp8_e5m2': 8-bit, E5M2 format
    
    Args:
        nvfp_format: NVFP format string
    Returns:
        ebits: exponent bits
        mbits: mantissa bits (including sign)
        emax: max exponent
        max_norm: max representable value
        torch_dtype: Corresponding PyTorch dtype (if available)
    """
    nvfp_format_lower = nvfp_format.lower()
    
    # Parse format string
    if nvfp_format_lower == 'nvfp4' or nvfp_format_lower == 'nvfp4_e2m1':
        ebits, mbits = 2, 3  # 4-bit: 2 exp, 1 mantissa + 1 sign
        emax = 2**(ebits - 1)  # 2
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)  # 2^2 * 1.0 = 4.0
        torch_dtype = None  # No direct PyTorch dtype for FP4
    elif nvfp_format_lower == 'nvfp8' or nvfp_format_lower == 'nvfp8_e4m3':
        ebits, mbits = 4, 5  # 8-bit: 4 exp, 3 mantissa + 1 sign
        emax = 2**(ebits - 1)  # 8
        max_norm = 448.0  # FP8 E4M3 max value
        torch_dtype = torch.float8_e4m3fn
    elif nvfp_format_lower == 'nvfp8_e5m2':
        ebits, mbits = 5, 4  # 8-bit: 5 exp, 2 mantissa + 1 sign
        emax = 2**(ebits - 1) - 1  # 15
        max_norm = 57344.0  # FP8 E5M2 max value
        torch_dtype = torch.float8_e5m2
    else:
        # Try to parse old format for backward compatibility
        res = re.match(r"^nvfp([0-9]+)$", nvfp_format_lower)
        if res is None:
            raise ValueError(
                f"Invalid NVFP format: {nvfp_format}. "
                f"Expected nvfp4, nvfp4_e2m1, nvfp8, nvfp8_e4m3, or nvfp8_e5m2"
            )
        n_bits = int(res.group(1))
        if n_bits == 4:
            return _get_nvfp_params('nvfp4')
        elif n_bits == 8:
            return _get_nvfp_params('nvfp8_e4m3')  # Default to E4M3
        else:
            raise ValueError(f"NVFP{n_bits} not supported. Only nvfp4 and nvfp8 are supported.")
    
    return ebits, mbits, emax, max_norm, torch_dtype


def _round_mantissa(A, bits, round='nearest'):
    """Round mantissa to nearest bits"""
    if round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "even":
        absA = torch.abs(A)
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception(f"Unrecognized round method {round}")
    return A


def _quantize_nvfp_with_torch_dtype(x: Tensor, torch_dtype, max_norm: float, amax: Optional[Tensor] = None):
    """
    Quantize using PyTorch's native FP8 types (for NVFP8 E4M3/E5M2)
    
    This uses per-tensor scaling: compute amax, scale, convert to FP8, convert back.
    
    Args:
        x: Input tensor
        torch_dtype: torch.float8_e4m3fn or torch.float8_e5m2
        max_norm: Maximum representable value
        amax: Optional pre-computed amax (for static quantization)
    Returns:
        Quantized and dequantized tensor
    """
    eps = torch.finfo(torch.float32).eps
    
    # Compute or use provided amax
    if amax is not None:
        if isinstance(amax, (int, float)):
            amax_val = float(amax)
        elif isinstance(amax, torch.Tensor):
            amax_val = amax.item() if amax.numel() == 1 else amax
        else:
            amax_val = float(amax)
    else:
        # Dynamic: compute per-tensor amax
        amax_val = torch.amax(torch.abs(x)).item()
    
    if amax_val < eps:
        return x.clone()
    
    # Compute scale: scale so that amax maps to max_norm
    scale = max_norm / (amax_val + eps)
    
    # Scale input
    x_scaled = x.float() * scale
    
    # Convert to FP8 and back (this performs the quantization)
    x_fp8 = x_scaled.to(torch_dtype)
    x_dequantized = x_fp8.float() / scale
    
    return x_dequantized


def _quantize_nvfp4_manual(x: Tensor, max_norm: float, amax: Optional[Tensor] = None):
    """
    Manual quantization for NVFP4 (E2M1) since PyTorch doesn't have native FP4 type
    
    Args:
        x: Input tensor
        max_norm: Maximum representable value (4.0 for E2M1)
        amax: Optional pre-computed amax (for static quantization)
    Returns:
        Quantized and dequantized tensor
    """
    eps = torch.finfo(torch.float32).eps
    ebits, mbits = 2, 3  # E2M1: 2 exp, 1 mantissa + 1 sign
    
    # Compute or use provided amax
    if amax is not None:
        if isinstance(amax, (int, float)):
            amax_val = float(amax)
        elif isinstance(amax, torch.Tensor):
            amax_val = amax.item() if amax.numel() == 1 else amax
        else:
            amax_val = float(amax)
    else:
        # Dynamic: compute per-tensor amax
        amax_val = torch.amax(torch.abs(x)).item()
    
    if amax_val < eps:
        return x.clone()
    
    # Compute scale
    scale = max_norm / (amax_val + eps)
    
    # Scale input
    x_scaled = x.float() * scale
    
    # Manual quantization for E2M1
    out = x_scaled.clone()
    zero_mask = (x_scaled == 0)
    abs_x = torch.abs(x_scaled)
    non_zero_mask = ~zero_mask
    
    # Calculate private exponent for each element
    private_exp = torch.zeros_like(abs_x)
    private_exp[non_zero_mask] = torch.floor(torch.log2(abs_x[non_zero_mask] + FP32_MIN_NORMAL))
    
    # Clip exponent range
    min_exp = -(2**(ebits-1)) + 1  # -1
    max_exp = 2**(ebits-1)  # 2
    private_exp = private_exp.clamp(min=min_exp, max=max_exp)
    
    # Scale up to integer portion
    scale_up = 2**(mbits - 1)  # 2^2 = 4
    out = torch.where(non_zero_mask,
                     out / (2**private_exp) * scale_up,
                     torch.zeros_like(out))
    
    # Round mantissa
    out = _round_mantissa(out, mbits, 'nearest')
    
    # Scale back
    out = torch.where(non_zero_mask,
                     out / scale_up * (2**private_exp),
                     torch.zeros_like(out))
    
    # Clamp to max_norm
    out = torch.clamp(out, min=-max_norm, max=max_norm)
    
    # Preserve zeros
    out = torch.where(zero_mask, torch.zeros_like(out), out)
    
    # Handle Inf/NaN
    out = torch.where(torch.isinf(x_scaled) | torch.isnan(x_scaled), x_scaled, out)
    
    # Scale back to original range
    out = out / scale
    
    return out


@torch.no_grad()
def quant_nvfp_core(
    x: Tensor, 
    nvfp_format: str = 'nvfp4_e2m1', 
    round: str = 'nearest',
    amax: Optional[Tensor] = None,
    scale: Optional[Tensor] = None
) -> Tensor:
    """
    Core NVFP quantization function (per-tensor scaling)
    
    NVFP uses per-tensor scaling: compute amax for entire tensor, then scale and quantize.
    For NVFP8, uses PyTorch's native FP8 types (float8_e4m3fn or float8_e5m2).
    For NVFP4, uses manual quantization since PyTorch doesn't have native FP4.
    
    Supports both dynamic and static quantization:
    - Dynamic: amax is computed from input tensor
    - Static: amax is provided (from calibration data)
    
    Args:
        x: Input tensor
        nvfp_format: 'nvfp4', 'nvfp4_e2m1', 'nvfp8', 'nvfp8_e4m3', 'nvfp8_e5m2'
        round: Rounding method (only used for manual quantization, ignored for FP8)
        amax: Optional pre-computed amax value for static quantization
        scale: Optional pre-computed scale value (if provided, amax is ignored)
    Returns:
        Quantized and dequantized tensor
    """
    # Handle empty tensor
    if x.numel() == 0:
        return x.clone()
    
    # Get format parameters
    ebits, mbits, emax, max_norm, torch_dtype = _get_nvfp_params(nvfp_format)
    
    # If scale is provided, compute amax from it
    if scale is not None:
        eps = torch.finfo(torch.float32).eps
        if isinstance(scale, torch.Tensor):
            scale_val = scale.item() if scale.numel() == 1 else scale
        else:
            scale_val = float(scale)
        amax = torch.tensor(max_norm / scale_val - eps, device=x.device, dtype=x.dtype)
    
    # Use PyTorch native FP8 types for NVFP8
    if torch_dtype is not None:
        return _quantize_nvfp_with_torch_dtype(x, torch_dtype, max_norm, amax)
    else:
        # Manual quantization for NVFP4
        return _quantize_nvfp4_manual(x, max_norm, amax)


def quant_nvfp(
    x: Tensor, 
    nvfp_format: str = 'nvfp4_e2m1', 
    round: str = 'nearest',
    amax: Optional[Tensor] = None,
    scale: Optional[Tensor] = None
) -> Tensor:
    """
    NVFP quantization with gradient support (per-tensor)
    
    Supports both dynamic and static quantization:
    - Dynamic: amax and scale are computed from input tensor (default)
    - Static: amax or scale is provided (from calibration data)
    
    Args:
        x: Input tensor
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8'
        round: Rounding method
        amax: Optional pre-computed amax value for static quantization
        scale: Optional pre-computed scale value for static quantization
    Returns:
        Quantized tensor with gradient flow
    """
    x_temp = x.clone()
    x_quantized = quant_nvfp_core(x_temp.detach(), nvfp_format, round, amax, scale)
    
    # Straight-through estimator: preserve gradients
    out = x + (x_quantized - x.detach())
    return out


class NVFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, nvfp_format: str = 'nvfp4'):
        ctx.save_for_backward(A, B)
        ctx.nvfp_format = nvfp_format
        
        A_q = quant_nvfp(A, nvfp_format)
        B_q = quant_nvfp(B, nvfp_format)
        return torch.matmul(A_q, B_q)
    
    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        A_q = quant_nvfp(A, ctx.nvfp_format)
        B_q = quant_nvfp(B, ctx.nvfp_format)
        grad_output_q = quant_nvfp(grad_output, ctx.nvfp_format)
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output_q, B_q.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A_q.transpose(-2, -1), grad_output_q)
        return grad_A, grad_B, None


class NVFPBAddBmm(Function):
    @staticmethod
    def forward(ctx, input, batch1, batch2, beta=1.0, alpha=1.0, nvfp_format='nvfp4'):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta, ctx.alpha = beta, alpha
        ctx.nvfp_format = nvfp_format
        
        mm_out = NVFPMatMul.apply(batch1, batch2, nvfp_format)
        return beta * input + alpha * mm_out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        batch1_q = quant_nvfp(batch1, ctx.nvfp_format)
        batch2_q = quant_nvfp(batch2, ctx.nvfp_format)
        grad_output_q = quant_nvfp(grad_output, ctx.nvfp_format)
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output_q
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output_q
            grad_batch1 = torch.matmul(mm_grad, batch2_q.transpose(-2, -1))
            grad_batch2 = torch.matmul(batch1_q.transpose(-2, -1), mm_grad)
        
        return grad_input, grad_batch1, grad_batch2, None, None, None


def nvfp_matmul(A, B, nvfp_format='nvfp4'):
    """NVFP matrix multiplication"""
    return NVFPMatMul.apply(A, B, nvfp_format)


def nvfp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0, nvfp_format='nvfp4'):
    """NVFP batch matrix multiplication"""
    return NVFPBAddBmm.apply(input, batch1, batch2, beta, alpha, nvfp_format)


def quant_dequant_qkv(
    q, k, v, 
    nvfp_format='nvfp4',
    q_amax: Optional[Tensor] = None,
    q_scale: Optional[Tensor] = None,
    k_amax: Optional[Tensor] = None,
    k_scale: Optional[Tensor] = None,
    v_amax: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None
):
    """
    Quantize QKV tensors with NVFP
    
    Supports both dynamic and static quantization.
    For static quantization, provide amax or scale for each tensor (q, k, v).
    
    Args:
        q, k, v: Input QKV tensors
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8'
        q_amax, k_amax, v_amax: Optional pre-computed amax values for static quantization
        q_scale, k_scale, v_scale: Optional pre-computed scale values for static quantization
    Returns:
        Quantized QKV tensors with gradient flow
    """
    # quant_nvfp already handles straight-through estimator, so we can use it directly
    final_q = quant_nvfp(q, nvfp_format, amax=q_amax, scale=q_scale)
    final_k = quant_nvfp(k, nvfp_format, amax=k_amax, scale=k_scale)
    final_v = quant_nvfp(v, nvfp_format, amax=v_amax, scale=v_scale)
    return final_q, final_k, final_v


def quant_dequant_tensor(
    tensor, 
    nvfp_format='nvfp4',
    amax: Optional[Tensor] = None,
    scale: Optional[Tensor] = None
):
    """
    Quantize a tensor with NVFP
    
    Supports both dynamic and static quantization.
    
    Args:
        tensor: Input tensor
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8'
        amax: Optional pre-computed amax value for static quantization
        scale: Optional pre-computed scale value for static quantization
    Returns:
        Quantized tensor with gradient flow
    """
    # quant_nvfp already handles straight-through estimator, so we can use it directly
    return quant_nvfp(tensor, nvfp_format, amax=amax, scale=scale)


def compute_amax_from_calibration(
    x: Tensor,
    calibration_amax: Optional[Union[float, Tensor]] = None
) -> Tensor:
    """
    Compute amax for static quantization.
    
    If calibration_amax is provided, use it; otherwise compute dynamically.
    This function helps bridge calibration data to quantization.
    
    Args:
        x: Input tensor
        calibration_amax: Optional pre-computed amax from calibration data
    Returns:
        amax value (scalar tensor)
    """
    if calibration_amax is not None:
        if isinstance(calibration_amax, (int, float)):
            return torch.tensor(calibration_amax, device=x.device, dtype=x.dtype)
        elif isinstance(calibration_amax, torch.Tensor):
            return calibration_amax.to(device=x.device, dtype=x.dtype)
        else:
            return torch.tensor(float(calibration_amax), device=x.device, dtype=x.dtype)
    else:
        # Dynamic: compute from input
        return torch.amax(torch.abs(x))


def compute_scale_from_amax(
    amax: Tensor,
    nvfp_format: str = 'nvfp4',
    eps: Optional[float] = None
) -> Tensor:
    """
    Compute scale from amax for NVFP quantization.
    
    This is useful for static quantization where amax is known from calibration.
    
    Args:
        amax: Maximum absolute value (amax)
        nvfp_format: 'nvfp4', 'nvfp4_e2m1', 'nvfp8', 'nvfp8_e4m3', 'nvfp8_e5m2'
        eps: Optional epsilon value (default: torch.finfo(torch.float32).eps)
    Returns:
        Scale value (scalar tensor)
    """
    _, _, _, max_norm, _ = _get_nvfp_params(nvfp_format)
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    
    if isinstance(amax, (int, float)):
        amax = torch.tensor(amax, dtype=torch.float32)
    elif isinstance(amax, torch.Tensor):
        amax = amax.float()
    
    scale = max_norm / (amax + eps)
    return scale


class NVFPStaticQuantConfig:
    """
    Configuration for static NVFP quantization.
    
    Stores calibration data (amax values) for each tensor/layer.
    This can be loaded from calibration files generated by calibration scripts.
    """
    
    def __init__(self, calibration_data: Optional[Dict[str, Union[float, Tensor]]] = None):
        """
        Initialize static quantization config.
        
        Args:
            calibration_data: Dictionary mapping tensor names to their amax values.
                             Keys should be in format like "layer.weight_amax" or "layer.activation_amax"
        """
        self.calibration_data = calibration_data or {}
    
    def get_amax(self, tensor_name: str) -> Optional[Union[float, Tensor]]:
        """
        Get amax value for a tensor by name.
        
        Args:
            tensor_name: Name of the tensor (e.g., "layer.weight_amax")
        Returns:
            amax value if found, None otherwise
        """
        return self.calibration_data.get(tensor_name)
    
    def set_amax(self, tensor_name: str, amax: Union[float, Tensor]):
        """
        Set amax value for a tensor.
        
        Args:
            tensor_name: Name of the tensor
            amax: amax value to store
        """
        self.calibration_data[tensor_name] = amax
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, list]]) -> 'NVFPStaticQuantConfig':
        """
        Create config from dictionary (e.g., loaded from JSON).
        
        Args:
            data: Dictionary with tensor names as keys and amax values (or lists) as values
        Returns:
            NVFPStaticQuantConfig instance
        """
        calibration_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                # Convert list to tensor
                calibration_data[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, (int, float)):
                calibration_data[key] = float(value)
            else:
                calibration_data[key] = value
        return cls(calibration_data)
    
    def to_dict(self) -> Dict[str, Union[float, list]]:
        """
        Convert config to dictionary (for JSON serialization).
        
        Returns:
            Dictionary with tensor names and amax values
        """
        result = {}
        for key, value in self.calibration_data.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result[key] = value.item()
                else:
                    result[key] = value.tolist()
            else:
                result[key] = value
        return result


if __name__ == '__main__':
    device = 'cuda'
    A = torch.randn(1024, 1024, device=device)
    B = torch.randn(1024, 1024, device=device)
    print(f"A_shape: {A.shape}, A_max: {torch.max(A):.4f}, A_min: {torch.min(A):.4f}")
    print(f"B_shape: {B.shape}, B_max: {torch.max(B):.4f}, B_min: {torch.min(B):.4f}")
    
    C_nvfp8 = nvfp_matmul(A.transpose(-2, -1), B, nvfp_format='nvfp8_e4m3')
    C_bf16 = torch.matmul(A.transpose(-2, -1), B).to(torch.bfloat16 if device == 'cuda' else torch.float32)
    loss_nvfp = torch.mean((C_bf16 - C_nvfp8) ** 2)
    
    print(f"C_shape: {C_nvfp8.shape}, output_max: {torch.max(C_nvfp8):.4f}, output_min: {torch.min(C_nvfp8):.4f}")
    print(f"loss_nvfp: {loss_nvfp.item():.6f}")

