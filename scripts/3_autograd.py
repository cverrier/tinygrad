from tinygrad import Tensor

x = Tensor([3.0], requires_grad=True)
y = x * 2
y[0].backward()  # Must be scalar to call backward without gradient
print(x.grad.tolist())  # [2.0]
