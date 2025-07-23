from tinygrad import UOp, dtypes
from tinygrad.renderer.cstyle import CUDARenderer, MetalRenderer
from tinygrad.uop import Ops

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

print(add)
# print(CUDARenderer(arch="sm_50").render([const, add]))
# print(MetalRenderer().render([const, add]))
# print(MetalRenderer().render([UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16))]))
# print(CUDARenderer(arch="sm_50").render([UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16))]))
print(CUDARenderer(arch="sm_50").render([UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16)), UOp(Ops.SPECIAL, dtypes.int, arg=("gidx1", 16))]))
