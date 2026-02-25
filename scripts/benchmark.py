import io
import sys

from tinygrad import Context, Device, Tensor
from tinygrad.helpers import GlobalCounters


def benchmark_operation(name, setup_fn, operation_fn, warmup=5, iterations=20):
  """
  Benchmark a GPU operation with proper methodology.

  Args:
    name: Name for display
    setup_fn: Function that returns input tensors (called once)
    operation_fn: Function that takes inputs and returns output tensor
    warmup: Number of warmup iterations
    iterations: Number of timed iterations
  """
  dev = Device[Device.DEFAULT]

  # Setup: create inputs and ensure they're on GPU
  inputs = setup_fn()
  if isinstance(inputs, Tensor):
    inputs = (inputs,)
  for t in inputs:
    t.realize()
  dev.synchronize()

  # Warmup: run several times to ensure kernels are compiled
  print(f"Warming up {name}...", end=" ", flush=True)
  for _ in range(warmup):
    operation_fn(*inputs).realize()
    dev.synchronize()
  print("done")

  # Get kernel time from multiple runs with DEBUG=2
  GlobalCounters.reset()
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  try:
    with Context(DEBUG=2):
      for _ in range(iterations):
        operation_fn(*inputs).realize()
  finally:
    sys.stdout = old_stdout

  kernel_avg_ms = GlobalCounters.time_sum_s / iterations * 1e3

  # Report
  print(f"\n{'=' * 60}")
  print(f"Benchmark: {name}")
  print(f"{'=' * 60}")
  print(f"GPU kernel time: {kernel_avg_ms:8.3f} ms")
  print()

  return kernel_avg_ms


# Compare memory-bound vs compute-bound operations
n = 4096

# Memory-bound: element-wise operations
elementwise_ms = benchmark_operation("Element-wise ops", lambda: (Tensor.randn(n, n), Tensor.randn(n, n)), lambda a, b: a + b * 2.0 + 1.0)

# Compute-bound: matrix multiplication
matmul_ms = benchmark_operation("Matrix multiply", lambda: (Tensor.randn(n, n), Tensor.randn(n, n)), lambda a, b: a @ b)

# Calculate effective bandwidth and FLOPS
elementwise_bytes = 4 * n * n * 4  # 3 reads + 1 write, float32
elementwise_bandwidth = elementwise_bytes / (elementwise_ms * 1e-3) / 1e9
print(f"Element-wise effective bandwidth: {elementwise_bandwidth:.1f} GB/s")

matmul_flops = 2 * n * n * n
matmul_gflops = matmul_flops / (matmul_ms * 1e-3) / 1e9
print(f"Matrix multiply effective GFLOPS: {matmul_gflops:.1f}")
