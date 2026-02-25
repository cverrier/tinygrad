from tinygrad import Tensor

x = Tensor.rand(4, 4).realize()
y = ((x + 1) * 2).sum()
schedule, _ = y.schedule_with_vars()
print(f"Kernels to be executed: {len(schedule)}")
