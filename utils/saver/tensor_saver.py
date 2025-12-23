#!/usr/bin/env python3
"""
Tensor saving utility module
Simple functions for saving tensors during model quantization
"""

import os
import torch
import time
from typing import Optional, Dict, Any
from pathlib import Path
import torch.distributed as dist
GLOBAL_STEP = 0

def update_global_step(step):
    global GLOBAL_STEP
    GLOBAL_STEP = step

def _simple_save(tensor, name_prefix):
    # 1. 检查步数 (只存 1 和 100)
    if GLOBAL_STEP not in [1, 100]:
        return
    # 2. 检查 Rank (只在主卡存)
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    try:
        # 建立目录: debug_tensors/iter_1/
        save_dir = f"debug_tensors/iter_{GLOBAL_STEP}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 转 BF16 保存
        path = f"{save_dir}/{name_prefix}_{id(tensor)}.pt"
        torch.save(tensor.to(torch.bfloat16), path)
    except Exception as e:
        print(f"Save error: {e}")

def _get_rank() -> Optional[int]:
    """Get current rank from distributed environment or environment variables"""
    # Try distributed environment
    try:
        if dist.is_initialized():
            return dist.get_rank()
    except:
        pass
    
    # Try environment variables
    rank_env = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env)
        except ValueError:
            pass
    
    return None


def _get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """Get basic tensor information"""
    if tensor.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    
    tensor_flat = tensor.float().flatten()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_flat.min().item()),
        "max": float(tensor_flat.max().item()),
        "mean": float(tensor_flat.mean().item()),
        "std": float(tensor_flat.std().item()),
    }


def _generate_filename(layer_type: str,
                      operation: str,
                      quant_type: str,
                      tensor_name: str,
                      layer_idx: Optional[int] = None,
                      phase: str = "unknown",
                      component: str = "unknown",
                      rank: Optional[int] = None,
                      iteration: int = 0) -> str:
    """Generate filename for tensor"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    parts = [timestamp, f"iter{iteration:03d}", layer_type]
    
    if layer_idx is not None:
        parts.append(f"L{layer_idx}")
    
    parts.extend([operation, phase, component, quant_type])
    
    if rank is not None:
        parts.append(f"rank{rank:02d}")
    
    parts.append(tensor_name)
    
    return "_".join(parts) + ".pt"


# Global configuration
_SAVE_DIR = Path(os.environ.get("TENSOR_SAVE_DIR", "./tensor_logs"))
_ENABLED = os.environ.get("TENSOR_SAVE_ENABLED", "false").lower() == "true"
_ITERATION = 0
_TENSOR_COUNTER = 0


def set_save_dir(save_dir: str):
    """Set the save directory"""
    global _SAVE_DIR
    _SAVE_DIR = Path(save_dir)
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)


def set_enabled(enabled: bool):
    """Enable or disable tensor saving"""
    global _ENABLED
    _ENABLED = enabled


def set_iteration(iteration: int):
    """Set current iteration"""
    global _ITERATION
    _ITERATION = iteration


def save_tensor(tensor: torch.Tensor,
                layer_type: str,
                operation: str,
                quant_type: str,
                tensor_name: str,
                layer_idx: Optional[int] = None,
                phase: str = "unknown",
                component: str = "unknown",
                rank: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Save a tensor to file
    
    Args:
        tensor: Tensor to save
        layer_type: Layer type ("attention" or "linear")
        operation: Operation type ("forward" or "backward")
        quant_type: Quantization type ("mxfp8", "mxfp4", "bf16", etc.)
        tensor_name: Tensor name ("input", "output", "grad_input", etc.)
        layer_idx: Layer index
        phase: Phase ("pre" or "post")
        component: Component type ("linear" or "FA")
        rank: GPU rank (auto-detected if None)
        metadata: Additional metadata
        
    Returns:
        Saved file path, or None if disabled
    """
    global _TENSOR_COUNTER
    
    if not _ENABLED:
        return None
    
    # Auto-detect rank if not provided
    if rank is None:
        rank = _get_rank()
    
    if rank is None:
        rank = 0  # Default to 0
    
    try:
        # Generate filename
        filename = _generate_filename(
            layer_type, operation, quant_type, tensor_name,
            layer_idx, phase, component, rank, _ITERATION
        )
        filepath = _SAVE_DIR / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get tensor info
        tensor_info = _get_tensor_info(tensor)
        
        # Prepare tensor for saving
        if tensor.is_cuda:
            tensor_cpu = tensor.detach().cpu()
        else:
            tensor_cpu = tensor.detach().clone()
        
        if not tensor_cpu.is_contiguous():
            tensor_cpu = tensor_cpu.contiguous()
        
        # Prepare save data
        save_data = {
            "tensor": tensor_cpu,
            "tensor_info": tensor_info,
            "metadata": {
                "layer_type": layer_type,
                "operation": operation,
                "quant_type": quant_type,
                "tensor_name": tensor_name,
                "layer_idx": layer_idx,
                "phase": phase,
                "component": component,
                "rank": rank,
                "iteration": _ITERATION,
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **(metadata or {})
            }
        }
        
        # Save to file
        torch.save(save_data, filepath)
        
        _TENSOR_COUNTER += 1
        print(f"[TensorSaver] Saved: {filename}")
        return str(filepath)
        
    except Exception as e:
        print(f"[TensorSaver] Failed to save tensor: {e}")
        return None


def save_attention_tensors(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          quant_type: str,
                          operation: str = "forward",
                          layer_idx: Optional[int] = None,
                          phase: str = "pre",
                          component: str = "FA",
                          rank: Optional[int] = None,
                          attention_weights: Optional[torch.Tensor] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """
    Save attention layer tensors
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        quant_type: Quantization type
        operation: Operation type
        layer_idx: Layer index
        phase: Phase
        component: Component type
        rank: GPU rank
        attention_weights: Attention weights matrix
        metadata: Additional metadata
        
    Returns:
        Dictionary of saved file paths
    """
    results = {}
    
    if query is not None:
        results["query"] = save_tensor(
            query, "attention", operation, quant_type, "query",
            layer_idx, phase, component, rank, metadata
        )
    
    if key is not None:
        results["key"] = save_tensor(
            key, "attention", operation, quant_type, "key",
            layer_idx, phase, component, rank, metadata
        )
    
    if value is not None:
        results["value"] = save_tensor(
            value, "attention", operation, quant_type, "value",
            layer_idx, phase, component, rank, metadata
        )
    
    if attention_weights is not None:
        results["attention_weights"] = save_tensor(
            attention_weights, "attention", operation, quant_type, "attention_weights",
            layer_idx, phase, component, rank, metadata
        )
    
    return results


def save_linear_tensors(input_tensor: torch.Tensor,
                       weight: torch.Tensor,
                       quant_type: str,
                       operation: str = "forward",
                       layer_idx: Optional[int] = None,
                       phase: str = "pre",
                       component: str = "linear",
                       rank: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """
    Save linear layer tensors
    
    Args:
        input_tensor: Input tensor
        weight: Weight tensor
        quant_type: Quantization type
        operation: Operation type
        layer_idx: Layer index
        phase: Phase
        component: Component type
        rank: GPU rank
        metadata: Additional metadata
        
    Returns:
        Dictionary of saved file paths
    """
    results = {}
    
    if input_tensor is not None:
        results["input"] = save_tensor(
            input_tensor, "linear", operation, quant_type, "input",
            layer_idx, phase, component, rank, metadata
        )
    
    if weight is not None:
        results["weight"] = save_tensor(
            weight, "linear", operation, quant_type, "weight",
            layer_idx, phase, component, rank, metadata
        )
    
    return results

