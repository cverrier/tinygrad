from tinygrad import dtypes
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import CUDARenderer

renderer = CUDARenderer(arch="sm_50")

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16), src=())
added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const))
uops = [const, define_global, special, added, store]

rendered = renderer.render("rendered", uops)
print(rendered)
