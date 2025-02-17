from tinygrad import dtypes
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import MetalRenderer

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

print(add)
print(MetalRenderer().render("example", [const, add]))
