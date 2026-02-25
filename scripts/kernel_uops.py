from tinygrad import Tensor
from tinygrad.uop.ops import print_uops

# x = Tensor.ones(4).contiguous()
# schedule, _ = x.schedule_with_vars()
# print(schedule[0].ast)

# *****
x = Tensor.rand(4).realize()
tmp = (x+1).contiguous()
y = (tmp * 2).contiguous().realize()
# y = ((x+1)*2).contiguous().realize()
# print(y.uop)
# y = ((x + 1)*2).contiguous()#.realize()
# schedule, _ = y.schedule_with_vars()
# print(len(schedule))
# print(schedule[0].ast)
# print("*****")
# print(schedule[1].ast)

# *****
# x = Tensor.rand(4).realize()
# y = x.sum()#.realize()
# schedule, _ = y.schedule_with_vars()
# print(schedule[0].ast)
# print(x.uop)
# print(x.uop.render())
# print_uops(x.uop.toposort())

# *****
# x = Tensor.rand(3, 4).contiguous().realize()
# y = x.sum(axis=1)#.realize()  # Sum along axis 1: (3,4) -> (3,)
# schedule, _ = y.schedule_with_vars()
# print(schedule[0].ast)
