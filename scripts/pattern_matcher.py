from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite

# pm = PatternMatcher([
#   # (UPat(Ops.CONST, arg=0), lambda: UOp.const(dtypes.int, 999)),
#   # (UPat(Ops.CONST, name='x'), lambda x: f'Matched: {x}'),
#   # (UPat(Ops.ADD, src=(UPat(name='x'), UPat(name='x'))), lambda x: x * UOp.const(x.dtype, 2)),
#   # (UPat(Ops.ADD, src=(UPat(name='x'), UPat(name='x'))), lambda x: x * 2),
#   (UPat(Ops.ADD, src=(pat:=UPat(name='x'), pat)), lambda x: x * 2),
#   # (UPat((Ops.MUL, Ops.ADD), name='x'), lambda x: f'Matched {x.op}'),
# ])

# a = UOp.const(dtypes.int, 5)
# add_same = a + a
# print('(a+a) before rewriting:', add_same, sep='\n')
# print("*" * 80)
# print('(a+a) after rewriting:', pm.rewrite(add_same), sep='\n')
#
# print("=" * 80)
#
# add_diff = a + UOp.const(dtypes.int, 6)
# print('(a+6) before rewriting:', add_diff, sep='\n')
# print("*" * 80)
# print('(a+6) after rewriting:', pm.rewrite(add_diff), sep='\n')

# zero = UOp.const(dtypes.int, 0)
# one = UOp.const(dtypes.int, 1)
# print(pm.rewrite(zero))
# print(pm.rewrite(one))

# a = UOp.const(dtypes.int, 2)
# b = UOp.const(dtypes.int, 3)
# print('Testing op matching:')
# print(a - b)
# print(pm.rewrite(a + b))
# print(pm.rewrite(a * b))
# print(pm.rewrite(a - b))

# Constant folding: ADD(CONST, CONST) -> CONST
pm = PatternMatcher([
   (UPat(Ops.ADD, src=(UPat.cvar('a'), UPat.cvar('b'))),
    lambda a, b: UOp.const(a.dtype, a.arg + b.arg)),
])

# Build a graph: (1 + 2) + (3 + 4)
one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
three = UOp.const(dtypes.int, 3)
# four = UOp.const(dtypes.int, 4)
# expr = (one + two) + (three + four)
expr = (one + two) + three

print(f'Before:\n{expr}')
print()

# graph_rewrite applies the pattern recursively
# result = graph_rewrite(expr, pm, bottom_up=True)
# print(f'After:\n{result}')
