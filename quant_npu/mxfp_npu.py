import torch
import torch_npu
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.quant_compute.quant_cy_npu.quant_cy_npu import quant_dequant_float,QType
from utils.quant_compute.quant_cy_npu.mxfp8 import injectMatmulNora

def quant_dequant_qkv(q,k,v):
    q_temp,k_temp,v_temp = q.clone(),k.clone(),v.clone()
    q_temp = quant_dequant_float(
        q_temp.detach(),
        QType('mxfp8e4m3'),
    )
    k_temp = quant_dequant_float(
        k_temp.detach(),
        QType('mxfp8e4m3'),
    )
    v_temp = quant_dequant_float(
        v_temp.detach(),
        QType('mxfp8e4m3'),
    )
    final_q = q + (q_temp - q.detach())
    final_k = k + (k_temp - k.detach())
    final_v = v + (v_temp - v.detach())
    return final_q,final_k,final_v
    
def quant_dequant_tensor(tensor):
    tensor_temp = tensor.clone()
    tensor_temp = quant_dequant_float(
        tensor_temp.detach(),
        QType('mxfp8e4m3'),
    )
    final_tensor = tensor + (tensor_temp - tensor.detach())
    return final_tensor

def mxfp_matmul(A,B):
    qtype_a = QType('mxfp8e4m3')
    qtype_b = QType('mxfp8e4m3')
    qtype_a.dim_(-1)
    qtype_b.dim_(0)
    newA = quant_dequant_float(A.clone(), qtype_a, force_py=False)
    newB = quant_dequant_float(B.clone(), qtype_b, force_py=False)
    C = torch.matmul(newA, newB)
    return C

if __name__ == '__main__':
    A = torch.randn(1024, 1024).npu()
    mxfp8 = quant_dequant_float(A, QType('mxfp8e4m3'))

    print("origin_A:", A)
    print("mxfp8_A:", mxfp8)
    
    print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
    B = torch.randn(1024, 1024).npu()
    print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")

    C_mxfp8 = mxfp_matmul(A.transpose(-2,-1),B)
    C_bf16 = torch.matmul(A.transpose(-2,-1),B).to(torch.bfloat16)
    loss_mxfp = torch.mean((C_bf16 - C_mxfp8) ** 2)
        
    print(f"C_shape:{C_mxfp8.shape},output_max:{torch.max(C_mxfp8)},output_min:{torch.min(C_mxfp8)}")
    print(f"loss_mxfp:{loss_mxfp}")