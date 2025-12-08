import torch 
from ..QType import QType 
from torch import Tensor 


@torch.no_grad()
def quant_hif8(x: Tensor, Q: QType, qdim: int): 
    max_value = (2**15)*0.95
    min_value = 2**(-22)
    
    x_unsigned1 = torch.abs(x)
    x[torch.abs(x) < 2**(-23)]=0
    sign = torch.sign(x)

    x_unsigned = torch.clamp(x_unsigned1, min= min_value, max=max_value)

    if x.dtype==torch.float16:
        e = torch.floor(torch.log2(x_unsigned + 2**-14))
    else:
        e = torch.floor(torch.log2(x_unsigned + 2**-45))
    
    abse = e.abs()
    mant_bits = torch.zeros_like(abse)
    mant_bits[abse<=15] = 1 
    mant_bits[abse<=7] = 2 
    mant_bits[abse<=3] = 3

    res = torch.floor(x_unsigned * 2.0**(-e + mant_bits) + 0.5) * 2.0**(e - mant_bits) * sign
    return res 

