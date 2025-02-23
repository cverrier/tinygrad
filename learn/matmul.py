from tinygrad import Tensor

M, N, P = 1853, 354, 8835

a = Tensor.randn(M, P)
b = Tensor.randn(P, N)

expected = a.matmul(b)

aa = a.unsqueeze(1)
bb = b.unsqueeze(0).permute(0, 2, 1)

result = (aa * bb).sum(-1)

assert result.isclose(expected).all().numpy()
