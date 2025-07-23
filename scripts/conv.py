from tinygrad import Tensor

a = Tensor.arange(16).reshape(1, 1, 4, 4)
print(a.numpy())

weights = Tensor.ones(1, 1, 3, 3)


def conv2d_manual(x: Tensor, weights: Tensor, stride: int = 1, dilation: int = 1) -> Tensor:
  pooled = x._pool(k_=weights.squeeze().shape, stride=stride, dilation=dilation)
  multiplied = pooled * weights
  return multiplied.sum(axis=(-2, -1))


stride, dilation = 1, 1
out = a.conv2d(weights, stride=stride, dilation=dilation)
print(out.shape)
print(out.numpy())

out_manual = conv2d_manual(a, weights, stride=stride, dilation=dilation)
print(out_manual.shape)
print(out_manual.numpy())

assert (out == out_manual).all().numpy(), "The outputs of the two methods do not match!"
