from tinygrad import Tensor

# Recreate what happens in the kernel generated for `Tensor.arange()`.
N = 15
a = Tensor.zeros(N-1).cat(Tensor.ones(N))
print(a._pool(k_=(N,), stride=1, dilation=1).sum(0).numpy())
