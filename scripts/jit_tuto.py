from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Timing


def make_chain(n):
  @TinyJit
  def f(x):
    # Create a chain of n simple kernels
    for _ in range(n):
      x = (x + 1).realize()
    return x
  return f

a = Tensor.empty(4, 4).realize()

for n_kernels in [10, 30, 50, 100]:
  f = make_chain(n_kernels)
  # Warmup + capture + graph build
  for _ in range(5):
    f(a)
  # Measure steady state
  with Timing(f"{n_kernels} kernels: "):
    f(a)
