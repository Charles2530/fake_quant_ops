import torch
from auto_round.data_type.nvfp import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    get_reciprocal,
    ref_nvfp4_quant,
)
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

def quant_nvfp4(tensor: torch.Tensor, group_size: int = 16, global_scale: torch.Tensor = None) -> torch.Tensor:
    """
    对输入的 Tensor 进行 NVFP4 量化和反量化 (QDQ)。
    
    Args:
        tensor: 输入的原始张量 (float16, bfloat16, float32)
        group_size: 量化组大小，默认为 16
        global_scale: 全局缩放因子。若为 None，则按 tensor 的 abs().max() 计算，与 nvfp.nv_fp4 / nv_fp4_with_static_gs 一致
        
    Returns:
        torch.Tensor: 量化并反量化后的张量，保持原始 dtype
    """
    orig_dtype = tensor.dtype
    orig_device = tensor.device
    
    # 1. 准备全局缩放因子 (与 nvfp.py 中 nv_fp4 / nv_fp4_with_static_gs 一致，保证 scale 落在 e4m3 范围)
    if global_scale is None:
        tensor_max = tensor.abs().max().to(torch.float32)
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
        global_scale = global_scale.to(orig_device)
    else:
        global_scale = global_scale.to(device=orig_device, dtype=torch.float32)

    # 2. 形状对齐与填充 (Padding)
    # NVFP4 要求输入维度能被 group_size 整除
    temp_tensor = tensor.to(torch.float32) # 计算过程中建议使用 fp32
    reshaped_tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(temp_tensor, group_size)

    # 3. 执行量化和反量化 (QDQ) 核心逻辑
    # ref_nvfp4_quant 返回 (qdq_results, scales)
    # v=0 是 auto_round 中定义的 NVFP4 映射模式
    qdq_res, _ = ref_nvfp4_quant(reshaped_tensor, global_scale, group_size, v=0)

    # 4. 恢复原始形状并去除填充
    final_output = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)

    return final_output.to(orig_dtype)

# --- 使用示例 ---
if __name__ == "__main__":
    # 创建测试数据
    input_data = torch.randn((1024, 1024), dtype=torch.bfloat16)
    
    # 执行 QDQ
    output_data = quant_nvfp4(input_data, group_size=16)
    
    print(f"原始形状: {input_data.shape}")
    
    # 计算误差
    mse = torch.mean((input_data - output_data)**2)
    print(f"量化均方误差 (MSE): {mse:.6f}")
