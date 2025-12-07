#!/usr/bin/env python3
"""
BF16 operators module
"""

import torch
from torch.autograd import Function
from typing import Optional, Dict, Any


class BF16MatMul(Function):
    
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor):
        ctx.save_for_backward(A, B)
        
        if A.dtype != torch.bfloat16:
            A = A.to(torch.bfloat16)
        if B.dtype != torch.bfloat16:
            B = B.to(torch.bfloat16)
        
        output = torch.matmul(A, B)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        
        return grad_A, grad_B


class BF16BAddBmm(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor,
                beta: float = 1.0, alpha: float = 1.0):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta = beta
        ctx.alpha = alpha       
        
        mm_out = torch.bmm(batch1, batch2)
        output = beta * input + alpha * mm_out
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        beta, alpha = ctx.beta, ctx.alpha
        
        grad_input = grad_batch1 = grad_batch2 = None
        
        if ctx.needs_input_grad[0]:
            grad_input = beta * grad_output
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            mm_grad = alpha * grad_output
            grad_batch1 = torch.bmm(mm_grad, batch2.transpose(-2, -1))
            grad_batch2 = torch.bmm(batch1.transpose(-2, -1), mm_grad)

        return grad_input, grad_batch1, grad_batch2
                    
def bf16_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return BF16MatMul.apply(A, B)

def bf16_baddbmm(input: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor, 
                 beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    return BF16BAddBmm.apply(input, batch1, batch2, beta, alpha)

if __name__ == "__main__":
    A = torch.randn(1024, 1024).cuda()
    B = torch.randn(1024, 1024).cuda()
    print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
    print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")

    from hifp import hifp_matmul
    from mxfp import mxfp_matmul
    from nvfp import nvfp_matmul
    C_hifp8 = hifp_matmul(A.transpose(-2,-1),B)
    C_mxfp8 = mxfp_matmul(A.transpose(-2,-1),B,"fp8_e4m3")
    C_nvfp8 = nvfp_matmul(A.transpose(-2, -1), B, nvfp_format='nvfp8_e4m3')
    C_bf16 = torch.matmul(A.transpose(-2,-1),B).to(torch.bfloat16)
    loss_hif = torch.mean((C_bf16 - C_hifp8) ** 2)
    print(f"loss_hifp: {loss_hif}")
    loss_mx = torch.mean((C_bf16 - C_mxfp8) ** 2)
    print(f"loss_mx: {loss_mx}")
    loss_nv = torch.mean((C_bf16 - C_nvfp8) ** 2)
    print(f"loss_nv: {loss_nv}")