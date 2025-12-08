import math
import torch
import torch_npu
from typing import Tuple
from quant_cy_npu.quant_cy_npu import QType, quant_dequant_float
from quant_cy_npu.quant_cy_npu import delay_quant_dequant_float
# import struct
import numpy as np
import pandas as pd

# E4M3格式常量
MAX_VAL = 448.0
MAX_VAL_E4M3 = 448.0            # max representable value (finite) in E4M3
MIN_NORM_E4M3 = 2**-6           # smallest normal value in E4M3
MIN_SUBNORM_E4M3 = 2**-9        # smallest non-zero (subnormal) in E4M3
BIAS_E4M3 = 7                   # exponent bias for E4M3
MANT_SCOPE_E4M3 = 8
EXP_MIN_E4M3 = -6
EXP_MAX_E4M3 = 8


MAX_VAL_E5M2 = 57344.0         # max representable value (finite) in E5M2
MIN_NORM_E5M2 = 2**-14          # smallest normal value in E5M2
MIN_SUBNORM_E5M2 = 2**-16       # smallest non-zero (subnormal) in E5M2
BIAS_E5M2 = 15                  # exponent bias for E5M2
MANT_SCOPE_E5M2 = 4
EXP_MIN_E5M2 = -14
EXP_MAX_E5M2 = 16




    
def injectMatmulNora(A: torch.Tensor, B: torch.Tensor, mxfp8_mode: str = "e4m3", fb: str = "f", block_size: int = 32) -> torch.Tensor:
    shape_A = A.shape
    if len(shape_A) == 3:
        M, BS, K = shape_A
    else:
        M, K = shape_A
    N = B.shape[-1]
    
    # 量化&反量化A和B
    if mxfp8_mode == "e4m3":
        qtype_a = QType('mxfp8e4m3')
        qtype_b = QType('mxfp8e4m3')
    elif mxfp8_mode == "e5m2":
        qtype = QType('mxfp8e5m2')
    elif mxfp8_mode == "mxfp4":
        qtype = QType('mxfp4')
    elif mxfp8_mode == "hif8":
        qtype_a = QType('hif8')
        qtype_b = QType('hif8')
        

    elif mxfp8_mode == "hif4":
        qtype = QType('hifx4_v12')
    elif mxfp8_mode == "dhif8": #delay hif8
        qtype_a = QType('dhif8')
        qtype_b = QType('dhif8')
        qtype_a.dim_(-1)
        qtype_b.dim_(0)
        amax = torch.max(torch.abs(A)).float()
        scale_position = 8.0
        newA = delay_quant_dequant_float(A.clone(), amax.clone().npu(), scale_position, qtype_a, force_py=False)
        amax = torch.max(torch.abs(B)).float()
        newB = delay_quant_dequant_float(B.clone(), amax.clone().npu(), scale_position, qtype_b, force_py=False)
        C = torch.matmul(newA, newB)
        return C
    else:
        assert(f'Mxfp8 type {mxfp8_mode} not support now!')
    
    # print(f'qtype: {qtype}')
    
    qtype_a.dim_(-1)
    qtype_b.dim_(0)

  
    
    newA = quant_dequant_float(A.clone(), qtype_a, force_py=False)
    newB = quant_dequant_float(B.clone(), qtype_b, force_py=False)
    # newA = quant_dequant_float(A.clone(), qtype_a, force_py=True)
    # newB = quant_dequant_float(B.clone(), qtype_b, force_py=True)


    
    C = torch.matmul(newA, newB)
    # C = C.bfloat16()
    return C
    

#均方差
def _compute_mse(tensor1, tensor2)-> float:#tensor1是真值
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)

    #计算真值的RMS
    rms_value = torch.sqrt(torch.mean(tensor1 ** 2))
    #计算相对RMSE
    rel_rmse_pct = (rmse / rms_value) * 100
    #计算normalized MSE百分比
    NMSE = (mse / torch.mean(tensor1 ** 2)) * 100

    # print(f"MSE: {mse:.6f}")
    # print(f"RMSE: {rmse:.6f}")
    # print(f"RMS_TRUE: {rms_value:.6f}")
    # print(f"相对RMSE: {rel_rmse_pct:.2f}%")
    print(f"normalized MSE: {NMSE:.2f}%")
    
    return NMSE

if __name__ == '__main__':
    # A = torch.randn((4096, 1, 5504), dtype=torch.bfloat16).npu()
    # B = torch.randn((5504, 2048), dtype=torch.bfloat16).npu()
    # A = torch.rand((4096, 1, 1024), dtype=torch.bfloat16).npu()
    # B = torch.rand((1024, 2048), dtype=torch.bfloat16).npu()
    # # A = torch.rand((4, 1, 6), dtype=torch.bfloat16).npu()
    # # B = torch.rand((6, 8), dtype=torch.bfloat16).npu()
    
    
    import sys
    file_list = [
        '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/saved_tensors/Hif8_22320_Nan/step318/rank0/dump_tensor_data/Module.module.module.decoder.layers.21.mlp.linear_fc2.RowParallelLinear.forward.0.output.0.pt',
        '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/saved_tensors/Hif8_22320_Nan/step318/rank0/dump_tensor_data/Module.module.module.decoder.layers.21.mlp.linear_fc2.RowParallelLinear.forward.0.input.0.pt',
        '/home/ma-user/work/bucket-pangu-green-guiyang/wangxuefei/saved_tensors/Hif8_22320_Nan/step318/rank0/dump_tensor_data/Module.module.module.decoder.layers.21.mlp.linear_fc2.RowParallelLinear.forward.0.parameters.weight.pt'
    ]
    
                
   
    for pt_file in file_list:
        
        grad_output = torch.load(file_list[0]).squeeze(1)
        total_input = torch.load(file_list[1]).squeeze(1)
        weight = torch.load(file_list[2])
        
        # data_dict = torch.load(pt_file)
        # grad_output = data_dict["grad_output"].squeeze(1)
        # total_input = data_dict["total_input"].squeeze(1)
        # weight = data_dict["weight"]
        # data_b_0_0_0.pt Testing matrix shape: grad_outputtorch.Size([4096, 1, 4096]), total_inputtorch.Size([4096, 1, 2048]), weighttorch.Size([4096, 2048])
        # print(f"Testing matrix shape: grad_output {grad_output.shape}, total_input {total_input.shape}, weight {weight.shape}")
        

        print(f"Testing matrix shape: grad_output {grad_output.shape}, total_input {total_input.shape}, weight {weight.shape}")
        print(f'try to handle total_input x weight.t()')
        C_input = injectMatmulNora(total_input.clone().npu(), weight.t().clone().npu(), "dhif8")
        # C_input = injectMatmulNora(total_input, weight.t().clone(), "e4m3")
        C = torch.matmul(total_input.clone().npu(), weight.t().clone().npu())
        _compute_mse(C,C_input)
        
        print(f'try to handle grad_output x weight')
        C_input = injectMatmulNora(grad_output.clone().npu(), weight.clone().npu(), "dhif8")
        # C_input = injectMatmulNora(grad_output, weight, "e4m3")
        C = torch.matmul(grad_output.clone().npu(), weight.clone().npu())
        _compute_mse(C,C_input)

        print(f'try to handle grad_output.t x total_input')
        C_input = injectMatmulNora(grad_output.t().clone().npu(), total_input.clone().npu(), "dhif8")
        # C_input = injectMatmulNora(grad_output.t().clone(), total_input, "e4m3")
        C = torch.matmul(grad_output.t().clone().npu(), total_input.clone().npu())
        _compute_mse(C,C_input)
       

