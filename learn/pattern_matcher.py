from tinygrad import Tensor, dtypes
from tinygrad.ops import Ops, PatternMatcher, UOp, UPat

a = Tensor.empty(4, 4)
b = a + 1
b.realize()


# const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
# const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)
#
# matcher = PatternMatcher([
#   (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2))),
# ])
#
# const = UOp(Ops.CONST, dtypes.float, arg=1.0)
# const_rewritten = matcher.rewrite(const)
# print(const_rewritten)
