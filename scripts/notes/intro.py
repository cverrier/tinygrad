from tinygrad import Tensor
# a = Tensor.empty(1)
# b = Tensor.empty(1)
# print((a+b).uop)
# print((a+b).numpy())

a = Tensor([3])
b = Tensor([7])
# print((a+b).uop)
print((a+b).numpy())



# from tinygrad.uop.ops import UOp, Ops
# from tinygrad import dtypes
# buffer = UOp(Ops.BUFFER, dtypes.float, src=())
# from tinygrad.renderer.cstyle import CUDARenderer
# print(CUDARenderer("sm_50").render([buffer]))

"""
UOp(Ops.ADD, dtypes.float, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.float, arg=1, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),
    x2:=UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),
  UOp(Ops.BUFFER, dtypes.float, arg=1, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
     x2,)),))
"""

# print((a.sum(0)).numpy())

# from tinygrad.uop.ops import UOp, Ops
# from tinygrad import dtypes
# const = UOp(Ops.CONST, dtype=dtypes.float, arg=1.0)
# add = UOp(Ops.ADD, dtype=dtypes.float, src=(const, const), arg=None)
# print(add)
#
# # from tinygrad.renderer.cstyle import CUDARenderer
# # print(CUDARenderer("sm_50").render([const, add]))
#
# # from tinygrad.renderer.cstyle import MetalRenderer
# # print(MetalRenderer().render([const, add]))
#
# from tinygrad.renderer.cstyle import MetalRenderer
# print(MetalRenderer().render([
#   UOp(Ops.SPECIAL, dtypes.index, src=(UOp.const(dtypes.int, 16),), arg="g0"),
#   # UOp.special(16, "g0")
# ]))
#
