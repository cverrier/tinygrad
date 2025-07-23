import math
import time

from tinygrad import Tensor, TinyJit, UOp


@TinyJit
def replace_if_zero(tokens: Tensor):
  tokens = (tokens == 0).where(-math.inf, tokens).contiguous()
  _ = (tokens == 0).all()
  return tokens, _


ctx = Tensor([[0, 0]])

_a = [
  [1, 7],
  [2, 0],
  [3, 2],
  [4, 0],
]

for i in range(0, len(_a)):
  a = Tensor(_a[i]).reshape((1, -1))
  a, _ = replace_if_zero(a)
  ctx = ctx.cat(a, dim=0)
  ctx.realize()

print(ctx.numpy())

# weight = Tensor.empty(4, 4)
#
#
# @TinyJit
# def forward(x: Tensor):
#   return weight.matmul(x).contiguous().sum().realize()
#
#
# for i in range(4):
#   dim = UOp.variable("dim", 1, 4).bind(i + 1)
#   start = time.time()
#   x = Tensor.empty(4, dim)
#   forward(x)
#   end = time.time()
#   print(f"Iteration {i} took {(end - start) * 1000:.2f}ms")


# @TinyJit
# def forward(x: Tensor):
#   c = (x * weight).contiguous()
#   c.sum(0).realize()
#
#
# for i in range(7):
#   start = time.time()
#   x = Tensor.empty(4, 4)
#   forward(x)
#   end = time.time()
#   print(f"Iteration {i} took {(end - start) * 1000:.2f}ms")
