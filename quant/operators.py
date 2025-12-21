import torch
from .ops.mxfp import _quantize_mx
from .ops.hifp import quant_hif8
from .ops.nvfp import quant_nvfp

def _convert_format_to_internal(forward_format):
    """
    Convert format names from external API (mxfp8_e4m3) to internal format (fp8_e4m3).
    
    Args:
        forward_format: External format string (e.g., 'mxfp8_e4m3', 'mxfp8_e5m2')
    
    Returns:
        Internal format string (e.g., 'fp8_e4m3', 'fp8_e5m2')
    """
    format_mapping = {
        'mxfp8_e4m3': 'fp8_e4m3',
        'mxfp8_e5m2': 'fp8_e5m2',
        'mxfp4_e2m1': 'fp4_e2m1',
    }
    return format_mapping.get(forward_format, forward_format)

class QuantDequantTensorWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, forward_format='mxfp8_e4m3', minus_exp=None, 
                backward_quantize=True, backward_format='mxfp8_e4m3'):
        scale_bits = 8
        tensor_temp = tensor.clone()     
        # Forward 量化
        if forward_format in ['mxfp8_e4m3', 'mxfp8_e5m2','mxfp4_e2m1']:
            # Convert format name from external API to internal format
            internal_format = _convert_format_to_internal(forward_format)
            tensor_temp = _quantize_mx(
                tensor_temp.detach(),
                scale_bits,
                internal_format,
                shared_exp_method="max",
                axes=-1,
                # adaptive block size
                block_size=32 if forward_format in ['mxfp8_e4m3', 'mxfp8_e5m2'] else 16,
                round="nearest",
                flush_fp32_subnorms=False,
                minus_exp=minus_exp
            )
        elif forward_format in ['hif8']:
            tensor_temp = quant_hif8(tensor_temp.detach())
        elif forward_format in ['nvfp8_e4m3', 'nvfp8_e5m2','nvfp4_e2m1']:
            tensor_temp = quant_nvfp(tensor_temp.detach(), forward_format)
        elif forward_format in ['bf16']:
            tensor_temp = tensor_temp.to(torch.bfloat16)
        else:
            raise ValueError(f"Unsupported forward format: {forward_format}")
        
        # 保存参数用于 backward
        ctx.backward_quantize = backward_quantize
        ctx.backward_format = backward_format 
        ctx.minus_exp = minus_exp
        
        # STE: 允许梯度流回原 tensor，但使用量化后的值进行计算
        final_tensor = tensor + (tensor_temp - tensor.detach())
        
        return final_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.backward_quantize and ctx.backward_format:
            # 量化梯度
            scale_bits = 8
            grad_temp = grad_output.clone()
            if ctx.backward_format in ['mxfp8_e4m3', 'mxfp8_e5m2','mxfp4_e2m1']:
                # Convert format name from external API to internal format
                internal_format = _convert_format_to_internal(ctx.backward_format)
                grad_temp = _quantize_mx(
                    grad_temp.detach(),
                    scale_bits,
                    internal_format,
                    shared_exp_method="max",
                    axes=-1,
                    # adaptive block size
                    block_size=32 if ctx.backward_format in ['mxfp8_e4m3', 'mxfp8_e5m2'] else 16,
                    round="nearest",
                    flush_fp32_subnorms=False,
                    minus_exp=ctx.minus_exp
                )
            elif ctx.backward_format in ['hif8']:
                grad_temp = quant_hif8(grad_temp.detach())
            elif ctx.backward_format in ['nvfp8_e4m3', 'nvfp8_e5m2','nvfp4_e2m1']:
                grad_temp = quant_nvfp(grad_temp.detach(), ctx.backward_format)
            elif ctx.backward_format in ['bf16']:
                grad_temp = grad_temp.to(torch.bfloat16)
            else:
                raise ValueError(f"Unsupported backward format: {ctx.backward_format}")
            # STE: 允许梯度继续传播，但使用量化后的值
            grad_input = grad_output + (grad_temp - grad_output.detach())
            
            return grad_input, None, None, None, None
        else:
            # 不量化梯度，直接返回
            return grad_output, None, None, None, None


def quant_dequant_tensor_with_backward(tensor, forward_format='mxfp8_e4m3', 
                                       minus_exp=None, 
                                       backward_quantize=True,
                                       backward_format='mxfp8_e4m3'):
    return QuantDequantTensorWithBackward.apply(
        tensor, forward_format, minus_exp, backward_quantize, backward_format
    )

def quant_dequant_qkv(q,k,v,minus_exp=None, forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3'):
    """
    Quantize and dequantize Q, K, V tensors with backward quantization support.
    Returns tensors converted to bfloat16, matching the original implementation.
    """
    q = quant_dequant_tensor_with_backward(q, forward_format, minus_exp, backward_quantize, backward_format)
    k = quant_dequant_tensor_with_backward(k, forward_format, minus_exp, backward_quantize, backward_format)
    v = quant_dequant_tensor_with_backward(v, forward_format, minus_exp, backward_quantize, backward_format)
    # Convert to bfloat16 to match original implementation behavior
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    return q,k,v


def quant_matmul(A,B,forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3'):
    A = quant_dequant_tensor_with_backward(A, forward_format, None, backward_quantize, backward_format)
    B = quant_dequant_tensor_with_backward(B, forward_format, None, backward_quantize, backward_format)
    return torch.matmul(A,B)

def quant_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0,forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3'):
    input = quant_dequant_tensor_with_backward(input, forward_format, None, backward_quantize, backward_format)
    batch1 = quant_dequant_tensor_with_backward(batch1, forward_format, None, backward_quantize, backward_format)
    batch2 = quant_dequant_tensor_with_backward(batch2, forward_format, None, backward_quantize, backward_format)
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)