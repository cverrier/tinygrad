import time

from tinygrad import Tensor, TinyJit

weight = Tensor.empty(4, 4)


@TinyJit
def forward(x: Tensor):
  c = (x * weight).contiguous()
  return c.sum(0).realize()


for i in range(4):
  start = time.time()
  x = Tensor.empty(4, 4).realize()
  _ = forward(x)
  end = time.time()
  print(f"Iteration {i} took {(end - start) * 1000:.2f}ms")
