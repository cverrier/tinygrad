from tinygrad import dtypes
from tinygrad.ops import Ops, PatternMatcher, UOp, UPat
from tinygrad.renderer.cstyle import CUDARenderer


def get_uops_without_pattern_matcher():
  const = UOp(Ops.CONST, dtypes.float, arg=1.0)
  define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
  special = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16), src=())
  added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
  store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const))
  uops = [const, define_global, special, added, store]
  return uops


# TODO: Understand why having the pattern matcher defined inside this function leads to
# an error that has to do with closures.
def get_uops_with_pattern_matcher():
  const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
  const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

  pattern_matcher = PatternMatcher([
    (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2))),
  ])

  const = UOp(Ops.CONST, dtypes.float, arg=1.0)
  const_rewritten = pattern_matcher.rewrite(const)
  define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
  special = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16), src=())
  added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
  store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const_rewritten))
  uops = [const_1, const_2, const_rewritten, define_global, special, added, store]
  return uops


if __name__ == "__main__":
  renderer = CUDARenderer(arch="sm_50")
  # uops_without_matcher = get_uops_without_pattern_matcher()
  # uops_with_matcher = get_uops_with_pattern_matcher()
  # rendered_without_matcher = renderer.render("rendered", uops_without_matcher)
  const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
  const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

  pattern_matcher = PatternMatcher([
    (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2))),
  ])

  const = UOp(Ops.CONST, dtypes.float, arg=1.0)
  const_rewritten = pattern_matcher.rewrite(const)
  define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
  special = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16), src=())
  added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
  store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const_rewritten))
  uops_with_matcher = [const_1, const_2, const_rewritten, define_global, special, added, store]
  rendered_with_matcher = renderer.render("rendered", uops_with_matcher)
  # print("*" * 80)
  # print(rendered_without_matcher)
  print("*" * 80)
  print(rendered_with_matcher)
