import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_cy_npu import QType, quant_dequant_float
import torch.distributed as dist

def _remove_oldest_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)  # roll amax_history[0] to the end, it becomes amax_history_new[-1], amax_history[1] becomes amax_history_new[0]
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)                              # amax_history_new[0] is zeroed, to be replaced by the amax to be obtained in the next step
    return amax_history

class _DelayedScalingState:
    def __init__(self, history_len: int = 256, scale_update_interval: int = 4, hif8_max: float = 8.0, device: str = 'cuda'):
        self.scale = torch.ones(1, dtype=torch.float32, device=device)
        self.amax_history = torch.zeros(history_len, 1, dtype=torch.float32, device=device)
        self.hif8_max = float(hif8_max)
        self.scale_update_interval = int(scale_update_interval)
        self.forward_counter = 0
        self.fwd_seen = False

    def record_amax(self, x: torch.Tensor) -> None:                    # record amax of current tensor
        self.amax_history[0] = torch.amax(torch.abs(x.detach()))       # and place it at amax_history[0]

    def update_scale_no_reduce(self) -> None:
        # update scale starts
        if self.forward_counter % self.scale_update_interval == 0:
            amax = torch.max(self.amax_history, dim=0).values
            sf = self.hif8_max / amax
            sf = torch.where(amax > 0.0, sf, self.scale)
            sf = torch.where(torch.isfinite(amax), sf, self.scale)
            sf = torch.where(torch.isinf(sf), torch.full_like(sf, torch.finfo(torch.float32).max), sf)
            self.scale.copy_(sf)
        # update scale ends
        
        _remove_oldest_amax_history(self.amax_history)
        self.forward_counter += 1

    def reduce_amax0_inplace(self) -> None:
        buf = self.amax_history[0].clone()
        dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        self.amax_history[0].copy_(buf)

    def reduce_and_update_scale(self) -> None:                         # aggregate and update scale
        self.reduce_amax0_inplace()                                    # aggregate amax_history[0] from different cards
        self.update_scale_no_reduce()                                  # 1) calculate scale based on max(amax_history); 2) roll amax_history[0] to the end, clear out amax_history[1]

def _qdq_dtype(x: torch.Tensor, backend: str) -> torch.Tensor:
    """
    - this func employs ascend kernels from 'quant_cy_npu' and perform pseudo-quantization
    - after 'quant_dequant_float', x_qdq is at the assigned 'backend' precision, but its dtype remains the same 
    """
    if backend == "hif8_delayed_scale":

        ############ initialization ############
        key = getattr(x, "_ds_key", None)                                                                          # get tensor label ('key') from input x, e.g., "model.layers.0.mlp.down_proj::x"
        if key is None:
            raise RuntimeError("hif8_delayed_scale requires x._ds_key to be set (e.g., '<module_name>::x|w|g').")
        if not hasattr(_qdq_dtype, "_DS_STATES"):
            _qdq_dtype._DS_STATES = {}                                                                             # establish attr '_DS_STATES' for '_qdq_dtype'

        if key not in _qdq_dtype._DS_STATES:                                                                       # create a state for each tensor
            _qdq_dtype._DS_STATES[key] = _DelayedScalingState(history_len=256, scale_update_interval=4, hif8_max=8.0, device=x.device)
        st = _qdq_dtype._DS_STATES[key]                                                                            # st = st = _qdq_dtype._DS_STATES[key] for convenience

        ############ loop start ############
        update = bool(getattr(x, "_ds_update", False))                                                             # no x/w update in backward, no g update in forward; search 'x._ds_update = True' for more detail
        cold_start = (st.forward_counter == 0)
        role = key.rsplit("::", 1)[-1]   # 'x' | 'w' | 'g'
        
        if update:
            if cold_start:                            # [cold start] use in-time-scaling for cold start for one step, specifically
                st.record_amax(x)                     # [cold start] a. self.amax_history[0] = torch.amax(torch.abs(x.detach())), it will be used in this step and the current interval
                st.reduce_and_update_scale()          # [cold start] b. reduce and compute scale based on self.amax_history[0]
                if role in ("x", "w"):
                    st.fwd_seen = True
            else:
                if role in ("x", "w"):
                    if not st.fwd_seen:
                        st.reduce_and_update_scale()  # [normal loop] given amax from last step amax_history[0], 
                        st.record_amax(x)             # [normal loop] a. "reduce_amax0_inplace" (in "reduce_and_update_scale") reduces amax_history[0]
                    st.fwd_seen = True                # [normal loop] b. [when the interval starts] "update scale" (in "update_scale_no_reduce" in "reduce_and_update_scale") will update scale based on max(amax_history)
                else:  # role == 'g'                                     otherwise, the scale will not be updated
                    st.reduce_and_update_scale()      # [normal loop] c. "_remove_oldest_amax_history" (in "update_scale_no_reduce" in "reduce_and_update_scale") roll amax_history[0] to the end, replace amax_history[0] (previously amax_history[0]) with 0.0
                    st.record_amax(x)                 # [normal loop] finally, st.record_amax(x) place the new amax at amax_history[0], which will only be used next step
        
        scale = st.scale.item()                                                          # get the scale (calculated based on amax previous iterations, or a cold start) from the state, use it for quantization
        qtype = QType('hif8')
        x_scaled = x.float() * scale
        x_qdq = quant_dequant_float(x_scaled, qtype, force_py=False).to(torch.bfloat16)
        out = ((x + (x_qdq - x).detach()) / scale).to(torch.bfloat16)

        # TARGET = "model.layers.0.mlp.down_proj"
        # if key == f"{TARGET}::x":
        #     current_amax = torch.amax(torch.abs(x)).item()
        #     print(f"{key} step={st.forward_counter}; current amax = {current_amax:.6f}; amax_history = {st.amax_history.flatten().tolist()}; scale = {st.scale.item()}")
        
        return out                                                                       
                                                                                         
    if backend == "mxfp8e4m3_in_time_scale":
        K = 8.0                                                                       # mannully assign the range, [-8.0, 8.0]
        qtype = QType('mxfp8e4m3')
        amax = torch.amax(torch.abs(x.detach()))
        eps = torch.finfo(torch.float32).eps
        scale = torch.tensor(K, dtype=torch.float32, device=x.device) / (amax + eps)
        x = x.float() * scale
        x_qdq = quant_dequant_float(x, qtype, force_py=False).to(torch.bfloat16)
        out = ((x + (x_qdq - x).detach()) / scale).to(torch.bfloat16)
        return out

    if backend == "bf16":
        return x.to(torch.bfloat16)
        
    if backend == "hif8":
        qtype = QType('hif8')
        x_qdq = quant_dequant_float(x, qtype, force_py=False)
        return x_qdq

    if backend == "hif8_in_time_scale":
        K = 8.0                                                                       # mannully assign the range, [-8.0, 8.0]
        qtype = QType('hif8')
        amax = torch.amax(torch.abs(x.detach()))
        eps = torch.finfo(torch.float32).eps
        scale = torch.tensor(K, dtype=torch.float32, device=x.device) / (amax + eps)
        x = x.float() * scale
        x_qdq = quant_dequant_float(x, qtype, force_py=False).to(torch.bfloat16)
        out = ((x + (x_qdq - x).detach()) / scale).to(torch.bfloat16)
        return out

    if backend == "mxfp8e4m3":
        qtype = QType('mxfp8e4m3')
        x_qdq = quant_dequant_float(x, qtype, force_py=False)
        return x_qdq
    
    if backend == "mxfp8e5m2":
        qtype = QType('mxfp8e5m2')
        x_qdq = quant_dequant_float(x, qtype, force_py=False)
        return x_qdq


class dtypeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, backend_x: str, backend_w: str, backend_g: str, name: str):

        # [FWD] y = x @ w + b
        if backend_x == "hif8_delayed_scale" or backend_w == "hif8_delayed_scale":
            x._ds_key = f"{name}::x"
            weight._ds_key = f"{name}::w"
            x._ds_update = True                                         # forward pass requires x/w update
            weight._ds_update = True
        x_qdq = _qdq_dtype(x, backend=backend_x)
        w_qdq = _qdq_dtype(weight, backend=backend_w)
        y = F.linear(x_qdq.to(torch.bfloat16), w_qdq.to(torch.bfloat16), bias if bias is None else bias.to(torch.bfloat16))

        # Save tensors for backward
        ctx.save_for_backward(x, weight)
        ctx.backend_g = backend_g
        ctx.backend_w = backend_w
        ctx.backend_x = backend_x
        ctx.has_bias = bias is not None
        ctx.name = name

        # Log once for forward quantization
        _log_quant_once("FWD", name, backend_x, backend_w, backend_g)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        backend_g = ctx.backend_g
        backend_w = ctx.backend_w
        backend_x = ctx.backend_x
        name = ctx.name if hasattr(ctx, "name") else ""

        grad_x = None
        grad_w = None
        grad_b = None
        g_qdq = None

        # [DGRAD] G @ W

        if backend_g == "hif8_delayed_scale" or backend_w == "hif8_delayed_scale" or backend_x == "hif8_delayed_scale":
            grad_output._ds_key = f"{name}::g"
            weight._ds_key = f"{name}::w"
            x._ds_key = f"{name}::x"
            grad_output._ds_update = True
            weight._ds_update = False
            x._ds_update = False

        if ctx.needs_input_grad[0]:
            g_qdq = _qdq_dtype(grad_output, backend=backend_g)
            w_qdq = _qdq_dtype(weight, backend=backend_w)
            grad_x = F.linear(g_qdq.to(torch.bfloat16), w_qdq.to(torch.bfloat16).transpose(0, 1))  # [DGRAD] F.linear with transposed weight: grad_input = g @ W
            _log_quant_once("DGRAD", name, backend_x, backend_w, backend_g)

        # [WGRAD]：X^T @ G
        if ctx.needs_input_grad[1]:
            x_qdq = _qdq_dtype(x, backend=backend_x)
            if g_qdq is None:
                g_qdq = _qdq_dtype(grad_output, backend=backend_g)
            x2d = x_qdq.to(torch.bfloat16).reshape(-1, x_qdq.shape[-1])
            g2d = g_qdq.to(torch.bfloat16).reshape(-1, g_qdq.shape[-1])
            grad_w = g2d.transpose(0, 1) @ x2d                                                     # [WGRAD]
            grad_w = grad_w.to(weight.dtype)
            _log_quant_once("WGRAD", name, backend_x, backend_w, backend_g)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.dim() - 1))                                      # sum over all dimensions except the last one
            grad_b = grad_output.sum(dim=reduce_dims).to(torch.bfloat16)                           # bias gradient

        states = getattr(_qdq_dtype, "_DS_STATES", None)
        for k in (f"{name}::x", f"{name}::w"):
            st_xw = states.get(k, None)
            if st_xw is not None:
                st_xw.fwd_seen = False

        return grad_x, grad_w, grad_b, None, None, None, None


_PRINTED_ONCE: dict[str, set[str]] = {"fwd": set(), "dgrad": set(), "wgrad": set()}


def _log_quant_once(phase: str, name: str, backend_x: str, backend_w: str, backend_g: str):
    if os.getenv("DEBUG_DTYPE_LOG", "0") != "1":
        return
    if int(os.getenv("LOCAL_RANK", "0")) != 0:
        return
    tag = phase.upper()
    if "DGRAD" in tag:
        key = "dgrad"
    elif "WGRAD" in tag:
        key = "wgrad"
    elif "FWD" in tag:
        key = "fwd"
    else:
        key = phase.lower()
    seen = _PRINTED_ONCE.get(key, set())
    if name in seen:
        return
    _PRINTED_ONCE.setdefault(key, set()).add(name)
    print(f"[{phase}] {name} :: backend_x={backend_x}, backend_w={backend_w}, backend_g={backend_g}")


class dtypeLinearTorch(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, backend_x: str = "bf16", backend_w: str = "bf16", backend_g: str = "bf16"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.backend_x = backend_x
        self.backend_w = backend_w
        self.backend_g = backend_g
        self.name: str = ""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dtypeLinearFn.apply(x, self.weight, self.bias, self.backend_x, self.backend_w, self.backend_g, self.name)


def _should_replace_linear(full_name: str) -> bool:
    # only linear operators in transformer blocks are replaced, i.e., excluding embedding and lm_head
    in_blocks = (".blocks." in full_name) or (".layers." in full_name) or (".h." in full_name) or ("model.layers." in full_name) or ("transformer.blocks." in full_name)  # confirm it's in transformer blks
    not_embedding = not any(k in full_name for k in ("wte", "wpe", "emb", "embed", "lm_head"))                                                                            # confirm it's not embedding
    return in_blocks and not_embedding


def apply_dtype_linear_torch(model: nn.Module, backend_x: str = "bf16", backend_w: str = "bf16", backend_g: str = "bf16") -> None:
    # scan all modules and replace all nn.Linear with the implementations in dtypeLinearTorch
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _should_replace_linear(full_name):
                repl = dtypeLinearTorch(child.in_features, child.out_features, child.bias is not None, backend_x, backend_w, backend_g)
                with torch.no_grad():
                    repl.weight.copy_(child.weight)
                    if child.bias is not None:
                        repl.bias.copy_(child.bias)
                repl.name = full_name
                setattr(module, child_name, repl)