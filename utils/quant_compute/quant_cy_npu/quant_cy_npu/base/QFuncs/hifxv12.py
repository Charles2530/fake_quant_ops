import torch 
from ..QType import QType 
from torch import Tensor 


@torch.no_grad()
def quant_hifxv12(x: Tensor, Q: QType, qdim: int): 
    # print('HIFxV12')
    # ---- This code is modified from quant_py ----
    # reshape x 
    x = x.unflatten(qdim, (-1, 8, 2, 4))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)
    exp_offset = 15

    assert Q.exp_bits==0
    # compute initial shared exp 
    max_inner = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    if max_inner.dtype==torch.float16:
        shared_exp_grp = torch.floor(torch.log2(max_inner + 2**-14))   # find max exp 
    else:
        shared_exp_grp = torch.floor(torch.log2(max_inner + 2**-45))   # find max exp 
    # shared_exp_grp = torch.clip(shared_exp_grp, -Q.k_max-exp_offset, Q.k_max-exp_offset)

    # if Q.do_carry:
    #     # if mantissa become 10.00, shift exp by 1, not saturation
    #     raw_mant = x_unsigned / torch.exp2(shared_exp_grp - Q.man_bits + 1)
    #     exp_bias = torch.floor(raw_mant + 0.5) - (2**Q.man_bits - 1)   # max: 1
    #     exp_bias = torch.clip(exp_bias, 0, 1)   # min: 0
    #     shared_exp_grp = shared_exp_grp + torch.max(exp_bias, dim=qdim, keepdim=True)[0]

    # print(shared_exp_grp.flatten())
    shared_exp_mid = shared_exp_grp.max(dim=qdim-1, keepdim=True)[0]
    # print(shared_exp_mid.flatten())
    shared_exp_all = shared_exp_mid.max(dim=qdim-2, keepdim=True)[0]
    # print('EMAX', shared_exp_all.flatten())

    shared_exp_blk = shared_exp_all - 2
    shared_exp_blk = shared_exp_blk.clip(-46, 16)
    shared_exp_subblk = (shared_exp_mid - shared_exp_blk - 1).clip(0, 1)
    # print(shared_exp_subblk.flatten())
    shared_exp_subsubblk = (shared_exp_grp - shared_exp_blk - shared_exp_subblk).clip(0, 1)
    # print(shared_exp_subsubblk.flatten())

    shared_exp = shared_exp_blk + shared_exp_subblk + shared_exp_subsubblk
    
    mant = x_unsigned / torch.exp2(shared_exp)
    mant = torch.floor(mant * 2**(Q.man_bits - 1) + 0.5) / 2**(Q.man_bits - 1)
    mant[mant==2] = 2 - 2**(-Q.man_bits+1)
    # print(shared_exp.flatten())

    out = sign * mant * (2 ** shared_exp)

    # # check underflow and overflow 
    # underflow_idx = (x_unsigned < (2-2**(-Q.man_bits))*(2**(-Q.k_max-exp_offset+1)))
    # out[underflow_idx] = 0

    # nan_threshold = 2**(Q.k_max-exp_offset+Q.exp_max)*(1.5)
    # nan_idx = torch.any(torch.isnan(x_unsigned), dim=qdim, keepdim=True) | torch.any(torch.isinf(x_unsigned), dim=qdim, keepdim=True) | (torch.max(x_unsigned>=nan_threshold, dim=qdim, keepdim=True)[0]).max(dim=qdim-1, keepdim=True)[0]

    # nan_idx = torch.broadcast_to(nan_idx, out.shape)
    # out[nan_idx] = torch.nan

    out = out.flatten(qdim-3, qdim)
    return out 

