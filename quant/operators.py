import torch
from .ops.mxfp import _quantize_mx
from .ops.hifp import quant_hif8
from .ops.nvfp import quant_nvfp
import os

# --- 全局配置 ---
class DebugSaverConfig:
    ENABLE = True                  # 总开关
    CURRENT_ITER = 0               # 当前 iter (需要在训练循环里手动更新)
    TARGET_ITERS = [501,701,1001]        # 需要保存的 iter
    SAVE_DIR = "./newtensors_Olmo7B"   # 相对路径
    SAVE_COUNTER = 0
    MIN_SIZE_MB = 7

def _simple_save(tensor, prefix, name=None):
    """
    修改点：
    1. 增加 name 参数，生成可读文件名
    2. 增加大小过滤，避免保存太小的 tensor
    """
    # 检查是否在目标 iter
    if not (DebugSaverConfig.ENABLE and (DebugSaverConfig.CURRENT_ITER in DebugSaverConfig.TARGET_ITERS)):
        return

    # 分布式 rank 检查
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank != 0:
        return

    # --- 过滤逻辑 ---
    # 计算大小 (MB)
    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    if size_mb < DebugSaverConfig.MIN_SIZE_MB:
        return  # 跳过小文件，瞬间减少 90% 的垃圾文件

    iter_dir = os.path.join(DebugSaverConfig.SAVE_DIR, str(DebugSaverConfig.CURRENT_ITER))
    os.makedirs(iter_dir, exist_ok=True)
    
    # 构造文件名：如果有 name 就用 name，否则用计数器
    # 格式: layer0_qproj_fwd_in.pt 或 fwd_in_10000.pt
    if name:
        filename = f"{name}_{prefix}.pt"
    else:
        filename = f"{prefix}_{DebugSaverConfig.SAVE_COUNTER}.pt"
        DebugSaverConfig.SAVE_COUNTER += 1
    
    filepath = os.path.join(iter_dir, filename)
    # 避免覆盖（如果重计算导致同名，可以覆盖或者跳过）
    if not os.path.exists(filepath): 
        print(f"Saving {filename} ({size_mb:.2f} MB)...") # 打印日志方便调试
        torch.save(tensor.clone(), filepath)

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
                backward_quantize=True, backward_format='mxfp8_e4m3', name=None):

        if tensor.requires_grad: 
            _simple_save(tensor, "fwd_in", name=name)

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
        ctx.name = name # 保存 name
        
        # STE: 允许梯度流回原 tensor，但使用量化后的值进行计算
        final_tensor = tensor + (tensor_temp - tensor.detach())
        return final_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        _simple_save(grad_output, "bwd_grad", name=ctx.name)
        if ctx.backward_quantize and ctx.backward_format:
            # 量化梯度
            scale_bits = 8
            # from utils.saver.tensor_saver import _simple_save
            # _simple_save(grad_output, "bwd_in")
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
            
            return grad_input, None, None, None, None, None
        else:
            # 不量化梯度，直接返回
            return grad_output, None, None, None, None, None


# 恢复原样，不需要 name 参数
def quant_dequant_tensor_with_backward(tensor, forward_format='mxfp8_e4m3', 
                                       minus_exp=None, 
                                       backward_quantize=True,
                                       backward_format='mxfp8_e4m3'):
    return QuantDequantTensorWithBackward.apply(
        tensor, forward_format, minus_exp, backward_quantize, backward_format
    )

# 恢复原样，不需要 name 参数
def quant_dequant_qkv(q,k,v,minus_exp=None, forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3'):
    q = quant_dequant_tensor_with_backward(q, forward_format, minus_exp, backward_quantize, backward_format)
    k = quant_dequant_tensor_with_backward(k, forward_format, minus_exp, backward_quantize, backward_format)
    v = quant_dequant_tensor_with_backward(v, forward_format, minus_exp, backward_quantize, backward_format)
    
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    return q,k,v

def quant_dequant_tensor_with_backward(tensor, forward_format='mxfp8_e4m3', 
                                       minus_exp=None, 
                                       backward_quantize=True,
                                       backward_format='mxfp8_e4m3',
                                       name=None): # 新增 name
    return QuantDequantTensorWithBackward.apply(
        tensor, forward_format, minus_exp, backward_quantize, backward_format, name
    )

def quant_matmul(A, B, forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3', name_prefix=None):
    
    name_A = f"{name_prefix}_A" if name_prefix else None
    name_B = f"{name_prefix}_B" if name_prefix else None

    # 调用带 name 的版本
    A = quant_dequant_tensor_with_backward(A, forward_format, None, backward_quantize, backward_format, name=name_A)
    B = quant_dequant_tensor_with_backward(B, forward_format, None, backward_quantize, backward_format, name=name_B)
    return torch.matmul(A, B)

def quant_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0,forward_format='mxfp8_e4m3', backward_quantize=True, backward_format='mxfp8_e4m3'):
    input = quant_dequant_tensor_with_backward(input, forward_format, None, backward_quantize, backward_format)
    batch1 = quant_dequant_tensor_with_backward(batch1, forward_format, None, backward_quantize, backward_format)
    batch2 = quant_dequant_tensor_with_backward(batch2, forward_format, None, backward_quantize, backward_format)
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)