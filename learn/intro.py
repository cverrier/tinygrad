from tinygrad import Tensor

# a = Tensor.empty(4, 4)
# b = Tensor.empty(4, 4)
# print(a.sum(1).tolist())

from tinygrad.renderer.cstyle import MetalRenderer, CUDARenderer
from tinygrad.ops import UOp, Ops
from tinygrad import dtypes

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

print(add)
print(MetalRenderer().render("example", [
  const,
  add
]))
print(CUDARenderer("sm_50").render("example", [
  const,
  add
]))
print(
  MetalRenderer().render("example", [
    UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16))
  ])
)
