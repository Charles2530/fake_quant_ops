import torch 
from ..QType import QType
from torch import Tensor


@torch.no_grad()
def quant_nf4(x: Tensor, Q: QType, qdim: int) -> Tensor:
    fp8_qt = QType('e4m3k1b%dc'%Q.blk_outer_size)
    fp8_qt.k_max = 0 
    fp8_qt.exp_offset = 0

    if qdim!=-1:
        x = x.transpose(qdim, -1)
    x_grouped = x.unflatten(-1, (-1, Q.blk_outer_size, Q.blk_size)).abs()
    group_max = x_grouped.max(dim=-1, keepdim=True)[0]
    x_int = torch.floor_(x_grouped / group_max * 7 + 0.5)
    x_int = x_int / 7

    groupmax_squeeze = group_max.squeeze(-1)
    groupmax_max = groupmax_squeeze.max(dim=-1, keepdim=True)[0]
    groupmax_scaled = groupmax_squeeze / groupmax_max * fp8_qt.fp_val_max 

    from ..QTensor import quant_dequant_float
    groupmax_fp8 = quant_dequant_float(groupmax_scaled, fp8_qt)
    groupmax_dequant = groupmax_fp8 / fp8_qt.fp_val_max * groupmax_max 
    
    out = x.sign() * (x_int * groupmax_dequant.unsqueeze(-1)).flatten(-3)
    if qdim!=-1:
        out = out.transpose(qdim, -1)
    return out 
