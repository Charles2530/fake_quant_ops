import torch 
from ..QType import QType
from torch import Tensor 


@torch.no_grad()
def quant_tmx(x: Tensor, Q: QType, qdim: int) -> Tensor: 
    print('QTMX')
    # reshape x 
    x = x.unflatten(qdim, (-1, 8, 2))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    # compute initial shared exp 
    max_inner = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    if max_inner.dtype==torch.float16:
        shared_exp_grp = torch.floor(torch.log2(max_inner + 2**-14))   # find max exp 
    else:
        shared_exp_grp = torch.floor(torch.log2(max_inner + 2**-45))   # find max exp 
    # shared_exp_grp = torch.clip(shared_exp_grp, -Q.k_max-exp_offset, Q.k_max-exp_offset)

    shared_exp_blk = torch.max(shared_exp_grp, dim=qdim-1, keepdim=True)[0]
    e8 = shared_exp_blk - 1 
    e1x8 = (shared_exp_grp - e8).clamp_(0)
    e8g = e1x8 + e8 
    # print('E8G', e8g.flatten())

    mant = torch.floor(x_unsigned * 2**(-e8g+Q.man_bits-1) + 0.5) * 2**(-Q.man_bits+1)
    mant[mant==2] = 2 - 2**(-Q.man_bits+1)
    out = sign * mant * 2**e8g
    
    out = out.flatten(qdim-2, qdim)
    return out 
