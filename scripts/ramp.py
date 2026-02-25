from tinygrad import Tensor

t = Tensor([1,2,3,4])
# print(t.uop)
"""
UOp(Ops.COPY, dtypes.int, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""

t.realize()
# Ops.COPY was actually run after realizing the tensor, and now the UOp of the tensor
# is just a BUFFER.
# print(t.uop)
"""
UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
  UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""

# The BUFFER from above is being multiplied by a CONST (which is 2).
# This CONST is RESHAPEd then EXPANDed to broadcast to the BUFFER.
# print((t * 2).uop)
"""
UOp(Ops.MUL, dtypes.int, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
    x2:=UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),
  UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
    UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
      UOp(Ops.CONST, dtypes.int, arg=2, src=(
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
           x2,)),)),)),)),))
"""

t_times_4_try_1 = t * 4
t_times_4_try_2 = t * 4
# UOps are both immutable and globally unique.
# Even though the two tensors from above do the exact same computation:
# - the Tensor objects themselves are different from each other.
# - their specification is the same Python object (not just the same string literal, but
#   also the same object in memory).
assert t_times_4_try_1 is not t_times_4_try_2
assert t_times_4_try_1.uop is t_times_4_try_2.uop

# If we realize `t_times_4_try_1`...
t_times_4_try_1.realize()
# ... `t_times_4_try_2` also becomes the same BUFFER.
assert t_times_4_try_1.uop is t_times_4_try_2.uop
# print(t_times_4_try_2.uop)
"""
UOp(Ops.BUFFER, dtypes.int, arg=4, src=(
  UOp(Ops.UNIQUE, dtypes.void, arg=2, src=()),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
"""

# So the following print doesn't require any computation: it just needs a copy back to
# the CPU so we can print it (to confirm, run this script with `DEBUG=2`).
# print("** Only the COPY starts")
# print(t_times_4_try_2.tolist())
# print("** Only the COPY ends")

t_exp = t.exp()
t_exp_grad = t_exp.gradient(t)
print(t_exp_grad.tolist())
# print(t_exp_grad.uop)
