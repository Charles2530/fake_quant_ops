import torch
import torch_npu
from torch import Tensor
from torch.autograd import Function
import re

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def _get_nvfp_params(nvfp_format: str):
    """
    Get NVFP format parameters
    Args:
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8', etc.
    Returns:
        ebits: exponent bits
        mbits: mantissa bits (including sign)
        emax: max exponent
        max_norm: max representable value
    """
    res = re.match(r"^nvfp([0-9]+)$", nvfp_format.lower())
    if res is None:
        raise ValueError(f"Invalid NVFP format: {nvfp_format}. Expected nvfp2, nvfp4, nvfp8, etc.")
    
    n_bits = int(res.group(1))
    
    if n_bits == 2:
        ebits, mbits = 0, 2  # 2-bit: 0 exp, 1 mantissa + 1 sign
        emax = 0
        max_norm = 1.0  # 2^0 * (2^1 - 1) / 2^0 = 1.0
    elif n_bits == 4:
        ebits, mbits = 2, 3  # 4-bit: 2 exp, 1 mantissa + 1 sign
        emax = 2**(ebits - 1)  # 2
        max_norm = 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)  # 2^2 * 1.0 = 4.0
    elif n_bits == 8:
        ebits, mbits = 4, 5  # 8-bit: 4 exp, 3 mantissa + 1 sign
        emax = 2**(ebits - 1)  # 8
        max_norm = 2**emax * 1.75  # Similar to fp8_e4m3
    else:
        raise ValueError(f"NVFP{n_bits} not supported. Only nvfp2, nvfp4, nvfp8 are supported.")
    
    return ebits, mbits, emax, max_norm


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


def _quantize_elemwise_nvfp(A, ebits, mbits, max_norm, round='nearest'):
    """
    Element-wise quantization for NVFP format
    Args:
        A: Input tensor
        ebits: exponent bits
        mbits: mantissa bits (including sign)
        max_norm: max representable value
        round: rounding method
    Returns:
        Quantized tensor
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_A = A.coalesce()
        A = sparse_A.values().clone()
    
    out = A.clone()
    
    # Handle zero
    zero_mask = (A == 0)
    
    if ebits != 0:
        # Calculate private exponent for each element
        abs_A = torch.abs(A)
        private_exp = torch.floor(torch.log2(
            abs_A + FP32_MIN_NORMAL * zero_mask.type(A.dtype)
        ))
        
        # Clip exponent range
        min_exp = -(2**(ebits-1)) + 2
        private_exp = private_exp.clamp(min=min_exp)
        
        # Scale up to integer portion
        scale_up = 2**(mbits - 2)
        out = out / (2**private_exp) * scale_up
        
        # Round mantissa
        out = _round_mantissa(out, mbits, round)
        
        # Scale back
        out = out / scale_up * (2**private_exp)
    else:
        # For ebits=0 (nvfp2), treat as fixed-point
        scale_up = 2**(mbits - 1)
        out = out * scale_up
        out = _round_mantissa(out, mbits, round)
        out = out / scale_up
    
    # Clamp to max_norm
    out = torch.clamp(out, min=-max_norm, max=max_norm)
    
    # Preserve zeros
    out = torch.where(zero_mask, torch.zeros_like(out), out)
    
    # Handle Inf/NaN
    out = torch.where(torch.isinf(A) | torch.isnan(A), A, out)
    
    if A_is_sparse:
        out = torch.sparse_coo_tensor(
            sparse_A.indices(), out,
            sparse_A.size(), dtype=sparse_A.dtype,
            device=sparse_A.device, requires_grad=sparse_A.requires_grad
        )
    
    return out


@torch.no_grad()
def quant_nvfp_core(x: Tensor, nvfp_format: str = 'nvfp4', round: str = 'nearest') -> Tensor:
    """
    Core NVFP quantization function (per-tensor)
    Args:
        x: Input tensor
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8'
        round: Rounding method ('nearest', 'floor', 'even')
    Returns:
        Quantized tensor
    """
    ebits, mbits, emax, max_norm = _get_nvfp_params(nvfp_format)
    
    # Per-tensor scaling: find max absolute value
    amax = torch.amax(torch.abs(x))
    eps = torch.finfo(torch.float32).eps
    
    # Calculate scale factor to fit into max_norm range
    scale = max_norm / (amax + eps)
    
    # Scale input
    x_scaled = x.float() * scale
    
    # Quantize
    x_quantized = _quantize_elemwise_nvfp(x_scaled, ebits, mbits, max_norm, round)
    
    # Scale back
    x_dequantized = x_quantized / scale
    
    return x_dequantized


def quant_nvfp(x: Tensor, nvfp_format: str = 'nvfp4', round: str = 'nearest') -> Tensor:
    """
    NVFP quantization with gradient support (per-tensor)
    Args:
        x: Input tensor
        nvfp_format: 'nvfp2', 'nvfp4', 'nvfp8'
        round: Rounding method
    Returns:
        Quantized tensor with gradient flow
    """
    x_temp = x.clone()
    x_quantized = quant_nvfp_core(x_temp.detach(), nvfp_format, round)
    
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
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
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
        
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output
            grad_batch1 = torch.matmul(mm_grad, batch2.transpose(-2, -1))
            grad_batch2 = torch.matmul(batch1.transpose(-2, -1), mm_grad)
        
        return grad_input, grad_batch1, grad_batch2, None, None, None


def nvfp_matmul(A, B, nvfp_format='nvfp4'):
    """NVFP matrix multiplication"""
    return NVFPMatMul.apply(A, B, nvfp_format)


def nvfp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0, nvfp_format='nvfp4'):
    """NVFP batch matrix multiplication"""
    return NVFPBAddBmm.apply(input, batch1, batch2, beta, alpha, nvfp_format)


def quant_dequant_qkv(q, k, v, nvfp_format='nvfp4'):
    """Quantize QKV tensors with NVFP"""
    q_temp, k_temp, v_temp = q.clone(), k.clone(), v.clone()
    q_temp = quant_nvfp(q_temp.detach(), nvfp_format)
    k_temp = quant_nvfp(k_temp.detach(), nvfp_format)
    v_temp = quant_nvfp(v_temp.detach(), nvfp_format)
    
    final_q = q + (q_temp - q.detach())
    final_k = k + (k_temp - k.detach())
    final_v = v + (v_temp - v.detach())
    return final_q, final_k, final_v


def quant_dequant_tensor(tensor, nvfp_format='nvfp4'):
    """Quantize a tensor with NVFP"""
    tensor_temp = tensor.clone()
    tensor_temp = quant_nvfp(tensor_temp.detach(), nvfp_format)
    final_tensor = tensor + (tensor_temp - tensor.detach())
    return final_tensor


if __name__ == "__main__":
    # Test NVFP quantization on NPU
    A = torch.randn(1024, 1024).npu()
    
    # Test different NVFP formats
    for fmt in ['nvfp2', 'nvfp4', 'nvfp8']:
        nvfp_quantized = quant_nvfp(A, nvfp_format=fmt)
        print(f"\n{fmt} quantization:")
        print(f"Original shape: {A.shape}, max: {torch.max(A):.4f}, min: {torch.min(A):.4f}")
        print(f"Quantized shape: {nvfp_quantized.shape}, max: {torch.max(nvfp_quantized):.4f}, min: {torch.min(nvfp_quantized):.4f}")
        mse = torch.mean((A - nvfp_quantized) ** 2)
        print(f"MSE: {mse.item():.6f}")
    
    # Test matrix multiplication
    B = torch.randn(1024, 1024).npu()
    C_nvfp4 = nvfp_matmul(A.transpose(-2, -1), B, nvfp_format='nvfp4')
    C_bf16 = torch.matmul(A.transpose(-2, -1), B).to(torch.bfloat16)
    loss_nvfp = torch.mean((C_bf16 - C_nvfp4) ** 2)
    print(f"\nMatrix multiplication MSE (nvfp4 vs bf16): {loss_nvfp.item():.6f}")


