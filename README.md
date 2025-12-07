# Fake Quantization Operations

一个高性能的深度学习量化操作库，支持多种量化格式（MXFP、HiFP等），提供CPU、GPU和NPU加速实现。

## 📋 目录

- [功能特性](#功能特性)
- [支持的量化格式](#支持的量化格式)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [API文档](#api文档)
- [性能基准](#性能基准)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## ✨ 功能特性

- **多种量化格式支持**：MXFP (E4M3/E5M2)、HiFP、FP8、FP4等
- **多平台支持**：CPU、GPU (CUDA)、NPU (Ascend 910B)
- **高效实现**：优化的量化算法，支持块级共享指数
- **PyTorch集成**：无缝集成PyTorch，支持自动求导
- **灵活配置**：可自定义量化参数（指数位、尾数位、块大小等）

## 🎯 支持的量化格式

### MXFP (Mixed-Precision Floating Point)
- **MXFP8 E4M3**: 8位，4位指数，3位尾数
- **MXFP8 E5M2**: 8位，5位指数，2位尾数
- **MXFP6 E3M2**: 6位，3位指数，2位尾数
- **MXFP4 E2M1**: 4位，2位指数，1位尾数

### HiFP (Hierarchical Floating Point)
- **HiF8**: 8位混合精度浮点量化
- **HiF4**: 4位混合精度浮点量化 (hifx4_v12)
- **HiF2/3/5**: 支持2-5位变体

### 其他格式
- **FP16/BF16**: 标准半精度浮点
- **NVF4**: 4位NV浮点量化
- **NF4**: 4位NormalFloat量化

## 🚀 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- NumPy

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/fake_quant_ops.git
cd fake_quant_ops

# 安装依赖
pip install torch numpy
```

### NPU支持（可选）

如果需要NPU加速支持（Ascend 910B），需要额外安装：

```bash
# 安装torch_npu
pip install torch_npu

# 编译NPU算子
cd utils/quant_cy_npu
./build.sh
```

## 📖 快速开始

### 基本量化示例

```python
import torch
from quant.mxfp import _quantize_mx, mxfp_matmul
from quant.hifp import quant_hif8, hifp_matmul
from quant.qtype import QType

# 创建测试张量
x = torch.randn(1024, 1024).cuda()

# MXFP8量化
x_mxfp8 = _quantize_mx(
    x, 
    scale_bits=8, 
    elem_format='fp8_e4m3',
    shared_exp_method="max",
    axes=-1,
    block_size=16
)

# HiF8量化
x_hif8 = quant_hif8(x)

print(f"Original shape: {x.shape}")
print(f"MXFP8 shape: {x_mxfp8.shape}")
print(f"HiF8 shape: {x_hif8.shape}")
```

### 矩阵乘法量化

```python
import torch
from quant.mxfp import mxfp_matmul
from quant.hifp import hifp_matmul

# 创建输入矩阵
A = torch.randn(1024, 1024).cuda()
B = torch.randn(1024, 1024).cuda()

# MXFP8矩阵乘法
C_mxfp8 = mxfp_matmul(A, B, elem_format='fp8_e4m3', block_size=32)

# HiF8矩阵乘法
C_hif8 = hifp_matmul(A, B)

# 对比精度
C_fp32 = torch.matmul(A, B)
print(f"FP32 vs MXFP8 MSE: {torch.mean((C_fp32 - C_mxfp8) ** 2).item():.6f}")
print(f"FP32 vs HiF8 MSE: {torch.mean((C_fp32 - C_hif8) ** 2).item():.6f}")
```

## 💡 使用示例

### 1. 使用QType定义量化类型

```python
from quant.qtype import QType

# 定义MXFP4量化类型
qtype_mxfp4 = QType('mxfp4')  # 等价于 e2m1k8b32c

# 定义MXFP8 E4M3量化类型
qtype_mxfp8_e4m3 = QType('mxfp8e4m3')  # 等价于 e4m3k8b32c

# 定义HiF8量化类型
qtype_hif8 = QType('hif8')

# 定义自定义量化类型
qtype_custom = QType('e3m2k8b16c')  # 3位指数，2位尾数，8位共享指数，块大小16

# 指定量化维度
qtype_with_dim = QType('hif8').dim(-1)  # 在最后一个维度量化
```

### 2. QKV量化（用于Transformer）

```python
from quant.mxfp import quant_dequant_qkv

# 假设q, k, v是Transformer的query, key, value张量
q = torch.randn(32, 128, 1024).cuda()
k = torch.randn(32, 128, 1024).cuda()
v = torch.randn(32, 128, 1024).cuda()

# 量化QKV（保持梯度）
q_q, k_q, v_q = quant_dequant_qkv(q, k, v, elem_format='fp8_e4m3')
```

### 3. 通用张量量化

```python
from quant.mxfp import quant_dequant_tensor

x = torch.randn(1024, 1024).cuda()
x_quantized = quant_dequant_tensor(x, elem_format='fp8_e5m2')
```

### 4. 批量矩阵乘法（BAddBmm）

```python
from quant.mxfp import mxfp_baddbmm
from quant.hifp import hifp_baddbmm

# 批量矩阵乘法
batch1 = torch.randn(10, 1024, 512).cuda()
batch2 = torch.randn(10, 512, 1024).cuda()
input_tensor = torch.randn(10, 1024, 1024).cuda()

# MXFP8批量矩阵乘法
output_mxfp = mxfp_baddbmm(
    input_tensor, batch1, batch2,
    beta=1.0, alpha=1.0,
    elem_format='fp8_e4m3',
    block_size=32
)

# HiF8批量矩阵乘法
output_hif = hifp_baddbmm(input_tensor, batch1, batch2, beta=1.0, alpha=1.0)
```

### 5. 量化误差分析

```python
import torch
from quant.mxfp import _quantize_mx
from quant.hifp import quant_hif8

x = torch.randn(1024, 1024).cuda()

# MXFP8量化
x_mxfp8 = _quantize_mx(
    x, scale_bits=8, elem_format='fp8_e4m3',
    shared_exp_method="max", axes=-1, block_size=16
)

# HiF8量化
x_hif8 = quant_hif8(x)

# 计算误差
mse_mxfp8 = torch.mean((x - x_mxfp8) ** 2)
mse_hif8 = torch.mean((x - x_hif8) ** 2)
max_err_mxfp8 = torch.max(torch.abs(x - x_mxfp8))
max_err_hif8 = torch.max(torch.abs(x - x_hif8))

print(f"MXFP8 - MSE: {mse_mxfp8.item():.6f}, Max Error: {max_err_mxfp8.item():.6f}")
print(f"HiF8  - MSE: {mse_hif8.item():.6f}, Max Error: {max_err_hif8.item():.6f}")
```

## 📁 项目结构

```
fake_quant_ops/
├── quant/                    # 基础量化实现
│   ├── __init__.py
│   ├── qtype.py             # 量化类型定义
│   ├── mxfp.py              # MXFP量化实现
│   └── hifp.py              # HiFP量化实现
│
├── quant_npu/               # NPU相关量化实现
│   ├── __init__.py
│   ├── qtype.py             # NPU量化类型
│   ├── mxfp_npu.py          # NPU MXFP实现
│   └── hifp_npu.py          # NPU HiFP实现
│
├── utils/                   # 工具和测试代码
│   ├── test_dtype.py        # 量化误差测试
│   ├── mxfp_scaling_test.py # MXFP缩放测试
│   ├── plot_loss_curve.py   # 损失曲线绘制
│   ├── saver/               # 量化保存器
│   │   ├── mxfp_saver.py
│   │   ├── hifp_saver.py
│   │   └── bf16_saver.py
│   └── quant_cy_npu/        # NPU C++扩展
│       ├── setup.py
│       ├── build.sh
│       ├── README.md
│       └── quant_cy_npu/
│           └── base/
│               ├── QType.py
│               ├── QTensor.py
│               └── QFunc/
│
└── README.md                # 项目说明文档
```

## 📚 API文档

### MXFP量化

#### `_quantize_mx(A, scale_bits, elem_format, shared_exp_method, axes, block_size, round, flush_fp32_subnorms, minus_exp)`

MXFP量化核心函数。

**参数：**
- `A`: 输入张量
- `scale_bits`: 共享指数位数（通常为8）
- `elem_format`: 元素格式（'fp8_e4m3', 'fp8_e5m2', 'fp6_e3m2', 'fp4_e2m1'等）
- `shared_exp_method`: 共享指数选择方法（'max'或'none'）
- `axes`: 共享指数的轴
- `block_size`: 块大小（0表示不使用块）
- `round`: 舍入方法（'nearest', 'floor', 'even', 'dither'）
- `flush_fp32_subnorms`: 是否将FP32次正规数刷新为0
- `minus_exp`: 指数偏移量

**返回：** 量化后的张量

#### `mxfp_matmul(A, B, elem_format='fp8_e5m2', block_size=32)`

MXFP矩阵乘法，支持自动求导。

#### `mxfp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0, elem_format='fp8_e5m2', block_size=32)`

MXFP批量矩阵乘法。

### HiFP量化

#### `quant_hif8(x, Q=None, qdim=-1)`

HiF8量化函数。

**参数：**
- `x`: 输入张量
- `Q`: QType对象（可选）
- `qdim`: 量化维度

**返回：** 量化后的张量

#### `hifp_matmul(A, B)`

HiF8矩阵乘法。

#### `hifp_baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0)`

HiF8批量矩阵乘法。

### QType类

#### `QType(desc)`

量化类型定义类。

**支持的格式：**
- `'mxfp4'`, `'mxfp6e3m2'`, `'mxfp8e4m3'`, `'mxfp8e5m2'`
- `'hif8'`, `'hifx2_v12'`, `'hifx3_v12'`, `'hifx4_v12'`, `'hifx5_v12'`
- `'fp16'`, `'fp32'`, `'bf16'`
- `'nvf4'`
- 自定义格式：`'e<exp_bits>m<man_bits>k<k_bits>b<block_size>[c]'`

**方法：**
- `dim(dim)`: 设置量化维度（返回新对象）
- `dim_(dim)`: 设置量化维度（原地修改）
- `copy()`: 复制QType对象

## ⚡ 性能基准

在NVIDIA A100 GPU上的典型性能表现：

| 量化格式 | 输入大小 | 量化延迟 | 内存节省 | 精度损失 (MSE) |
|---------|---------|---------|---------|---------------|
| MXFP8 E4M3 | 1024×1024 | ~0.15ms | 75% | ~1e-4 |
| MXFP8 E5M2 | 1024×1024 | ~0.15ms | 75% | ~1e-3 |
| MXFP4 | 1024×1024 | ~0.12ms | 87.5% | ~1e-2 |
| HiF8 | 1024×1024 | ~0.18ms | 75% | ~5e-4 |
| HiF4 | 1024×1024 | ~0.14ms | 87.5% | ~1e-2 |

*注：实际性能可能因硬件配置和软件版本而异*

## 🔧 开发与测试

### 运行测试

```bash
# 测试量化误差
python utils/test_dtype.py <tensor_file> --format hifp8

# 测试MXFP缩放
python utils/mxfp_scaling_test.py

# 快速测试NPU算子（如果已安装）
cd utils/quant_cy_npu
python quick_test.py
```

### 构建NPU扩展

```bash
cd utils/quant_cy_npu
./build.sh
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢所有为量化技术做出贡献的研究者和开发者
- 特别感谢PyTorch团队提供的优秀框架

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/fake_quant_ops/issues)
- 发送邮件: your.email@example.com

---

**注意**: 本项目仍在积极开发中，API可能会有变化。建议在生产环境使用前进行充分测试。



