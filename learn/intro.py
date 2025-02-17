from tinygrad import dtypes
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import CUDARenderer

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

# print(CUDARenderer("sm_50").render("example", [
#   const,
#   add
# ]))

# Render thread position.
print(
  CUDARenderer("sm_50").render(
    "example",
    [
      UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16)),
      UOp(Ops.SPECIAL, dtypes.int, arg=("gidx1", 16)),
      UOp(Ops.SPECIAL, dtypes.int, arg=("gidx2", 16)),
    ],
  )
)
