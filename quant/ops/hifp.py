import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fake_quant_ops.quant.ops.qtype import QType
from torch import Tensor
import torch
from torch.autograd import Function
@torch.no_grad()
def quant_hif8_core(x: Tensor, Q: QType=None, qdim: int=-1) -> Tensor:
    max_value = (2**15)*0.95
    min_value = 2**(-22)

    x_unsignedl = torch.abs(x)
    sign = torch.sign(x)

    x_unsigned = torch.clamp(x_unsignedl, min=min_value, max=max_value)

    if x.dtype == torch.float16:
        e = torch.floor(torch.log2(x_unsigned + 2**(-14)))
    else:
        e = torch.floor(torch.log2(x_unsigned + 2**(-45)))

    abse = e.abs()
    mant_bits = torch.zeros_like(abse)
    mant_bits[abse <= 15] = 1
    mant_bits[abse <= 7] = 2
    mant_bits[abse <= 3] = 3

    res = torch.floor(x_unsigned * 2.0**(-e + mant_bits) + 0.5) * 2.0**(e - mant_bits) * sign
    return res

def quant_hif8(x: Tensor, Q: QType=None, qdim: int=-1) -> Tensor:
    K = 8.0                                                                       # mannully assign the range, [-8.0, 8.0]
    qtype = QType('hif8')
    amax = torch.amax(torch.abs(x.detach()))
    eps = torch.finfo(torch.float32).eps
    scale = torch.tensor(K, dtype=torch.float32, device=x.device) / (amax + eps)
    x = x.float() * scale
    x_qdq = quant_hif8_core(x, qtype).to(torch.bfloat16)
    out = ((x + (x_qdq - x).detach()) / scale).to(torch.bfloat16)
    return out


class HIFPMatMul(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor,
                elem_format: str = 'fp8_e5m2', block_size: int = 32):
        ctx.save_for_backward(A, B)
        ctx.elem_format = elem_format
        ctx.block_size = block_size

        A_q = quant_hif8(A)
        B_q = quant_hif8(B)
        return torch.matmul(A_q, B_q)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        A_q = quant_hif8(A)
        B_q = quant_hif8(B)
        grad_output_q = quant_hif8(grad_output)
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output_q, B_q.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A_q.transpose(-2, -1), grad_output_q)
        return grad_A, grad_B, None, None  # None对应elem_format和block_size

class HIFPBAddBmm(Function):
    @staticmethod
    def forward(ctx, input, batch1, batch2, beta=1.0, alpha=1.0):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta, ctx.alpha = beta, alpha
        
        mm_out = HIFPMatMul.apply(batch1, batch2)
        return beta * input + alpha * mm_out

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        batch1_q = quant_hif8(batch1)
        batch2_q = quant_hif8(batch2)
        grad_output_q = quant_hif8(grad_output)
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output_q
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output_q
            grad_batch1 = torch.matmul(mm_grad, batch2_q.transpose(-2, -1))
            grad_batch2 = torch.matmul(batch1_q.transpose(-2, -1), mm_grad)
        
        return grad_input, grad_batch1, grad_batch2, None, None, None, None

def hifp_matmul(A, B):
    return HIFPMatMul.apply(A, B)

def hifp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    return HIFPBAddBmm.apply(input, batch1, batch2, beta, alpha)

if __name__ == "__main__":
    A = torch.randn(1024, 1024).cuda()
    hifp8 = quant_hif8(A)

    print("origin_A:", A)
    print("hif8_A:", hifp8)
    
    print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
    B = torch.randn(1024, 1024).cuda()
    print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")

    C_hifp8 = hifp_matmul(A.transpose(-2,-1),B)
    C_bf16 = torch.matmul(A.transpose(-2,-1),B).to(torch.bfloat16)
    loss_hif = torch.mean((C_bf16 - C_hifp8) ** 2)
        
    print(f"C_shape:{C_hifp8.shape},output_max:{torch.max(C_hifp8)},output_min:{torch.min(C_hifp8)}")
    print(f"loss_hifp: {loss_hif}")
