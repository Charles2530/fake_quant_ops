import torch
import time

def benchmark_bf16_matmul(matrix_size: int = 4096, num_repeats: int = 100):
    """
    Benchmarks the performance of bfloat16 (BF16) matrix multiplication on a CUDA device.

    Args:
        matrix_size (int): The dimension N for the N x N matrices to be tested.
        num_repeats (int): The number of repetitions to get a stable average time.

    Returns:
        float: The computed performance in TFLOP/s (TeraFLOPs per second).
               Returns 0.0 if no compatible CUDA device is available or if bfloat16 is not supported.
    """
    # 1. Check for a compatible CUDA device and bfloat16 support
    if not torch.cuda.is_available():
        print("Error: A CUDA-enabled device is required for this benchmark.")
        return 0.0
    
    device = torch.device("cuda")
    
    if not torch.cuda.is_bf16_supported():
        print(f"Error: Your current GPU ({torch.cuda.get_device_name(0)}) does not support bfloat16.")
        print("This feature typically requires an NVIDIA Ampere (A100) or newer architecture GPU.")
        return 0.0

    print(f"Running BF16 benchmark on device: {torch.cuda.get_device_name(0)}...")
    print(f"Matrix dimensions: {matrix_size}x{matrix_size}")
    print(f"Number of repeats: {num_repeats}")

    # 2. Create test matrices with bfloat16 data type
    try:
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
    except Exception as e:
        print(f"Error creating bfloat16 tensors: {e}")
        print("This may be due to insufficient GPU memory.")
        return 0.0

    # 3. Warm-up: Perform a few operations to ensure the GPU reaches a stable clock speed
    print("Warming up...")
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize() # Wait for all CUDA cores to finish

    # 4. Precise timing
    start_time = time.perf_counter()
    for _ in range(num_repeats):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize() # Ensure all matmul operations are complete
    end_time = time.perf_counter()

    # 5. Calculate performance
    total_time = end_time - start_time
    avg_time_per_op = total_time / num_repeats
    
    # Floating-point operations (FLOPs) for matrix multiplication = 2 * N^3
    flops_per_op = 2 * (matrix_size ** 3)
    
    # Calculate TFLOP/s (TeraFLOPs per second)
    # TFLOP/s = FLOPs / time / 10^12
    tflops = (flops_per_op / avg_time_per_op) / 1e12

    print("\n--- Benchmark Results ---")
    print(f"Average time per operation: {avg_time_per_op * 1000:.3f} ms")
    print(f"BF16 MatMul Performance: {tflops:.2f} TFLOP/s")
    
    return tflops

if __name__ == "__main__":
    # Benchmark using 4096x4096 matrices, a standard size for such tests.
    # If you encounter an "Out of Memory" error, try a smaller matrix_size, e.g., 2048.
    benchmark_bf16_matmul(matrix_size=4096, num_repeats=100)
