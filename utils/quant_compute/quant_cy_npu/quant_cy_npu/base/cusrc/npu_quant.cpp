#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


void run_hifxg_v12_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_hifxg_v12_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_mxfp4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_delay_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, float amax, float scale_position, int len);
void run_delay_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, float amax, float scale_position, int len);
void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp8e4m3_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp8e4m3_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);


void hifxgv12_quant(at::Tensor x, at::Tensor y, int mant_bit){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hifxg_v12_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N, mant_bit);
}

void hifxgv12_quant_bf16(at::Tensor x, at::Tensor y, int mant_bit){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hifxg_v12_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N, mant_bit);
}

void mxfp4_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp4_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void mxfp4_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp4_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void hif8_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    // int M = x.numel() / x.size(-1);
    // int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
    // run_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void hif8_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    // int M = x.numel() / x.size(-1);
    // int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
    // run_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}


void delay_hif8_quant(at::Tensor x, at::Tensor y, float amax, float scale_position){
    int devidx = x.device().index();
    int len = x.numel();
    // int M = x.numel() / x.size(-1);
    // int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_delay_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), amax, scale_position, len);
    // run_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void delay_hif8_quant_bf16(at::Tensor x, at::Tensor y, float amax, float scale_position){
    int devidx = x.device().index();
    int len = x.numel();
    // int M = x.numel() / x.size(-1);
    // int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_delay_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), amax, scale_position, len);
    // run_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}


void mxfp8e4m3_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp8e4m3_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void mxfp8e4m3_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp8e4m3_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void nvf4_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_nvf4_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void nvf4_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_nvf4_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hifxgv12_quant", &hifxgv12_quant, "hifxgv12_quant");
    m.def("hifxgv12_quant_bf16", &hifxgv12_quant_bf16, "hifxgv12_quant_bf16");
    m.def("mxfp4_quant", &mxfp4_quant, "mxfp4_quant");
    m.def("mxfp4_quant_bf16", &mxfp4_quant_bf16, "mxfp4_quant_bf16");
    m.def("hif8_quant", &hif8_quant, "hif8_quant");
    m.def("hif8_quant_bf16", &hif8_quant_bf16, "hif8_quant_bf16");
    m.def("delay_hif8_quant", &delay_hif8_quant, "delay_hif8_quant");
    m.def("delay_hif8_quant_bf16", &delay_hif8_quant_bf16, "delay_hif8_quant_bf16");
    m.def("mxfp8e4m3_quant", &mxfp8e4m3_quant, "mxfp8e4m3_quant");
    m.def("mxfp8e4m3_quant_bf16", &mxfp8e4m3_quant_bf16, "mxfp8e4m3_quant_bf16");
    m.def("nvf4_quant", &nvf4_quant, "nvf4_quant");
    m.def("nvf4_quant_bf16", &nvf4_quant_bf16, "nvf4_quant_bf16");
}
