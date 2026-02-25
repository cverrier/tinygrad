from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite

def count_consts(ctx, x):
  """Count constants seen so far."""
  ctx["count"] = ctx.get("count", 0) + 1
  print(f"Saw constant {x.arg}, total: {ctx['count']}")

pm = PatternMatcher([
  (UPat(Ops.CONST, name="x"), count_consts),
])

one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
three = UOp.const(dtypes.int, 3)
expr = (one + two) + three

ctx = {}
graph_rewrite(expr, pm, ctx=ctx)
print(f"\nFinal count: {ctx['count']}")
# Saw constant 1, total: 1
# Saw constant 2, total: 2
# Saw constant 3, total: 3
#
# Final count: 3
