from tinygrad import UOp, dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

pattern_matcher = PatternMatcher([(UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2)))])

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
const_rewritten = pattern_matcher.rewrite(const)

define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16), src=())
indexed = UOp(Ops.INDEX, dtypes.float.ptr(), arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(indexed, const_rewritten))
uops = [const_1, const_2, const_rewritten, define_global, special, indexed, store]

metal_renderer = MetalRenderer()
rendered = metal_renderer.render(uops)
print(rendered)
