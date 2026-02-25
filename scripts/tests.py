from tinygrad import Tensor

a = Tensor.randn(2, 5, 3, 4)

assert (out := a.sum(axis=0)).shape == (5, 3, 4), f"Expected shape (5, 3, 4), got {out.shape}"
assert (out := a.sum(axis=1)).shape == (2, 3, 4), f"Expected shape (2, 3, 4), got {out.shape}"
assert (out := a.sum(axis=2)).shape == (2, 5, 4), f"Expected shape (2, 5, 4), got {out.shape}"
assert (out := a.sum(axis=3)).shape == (2, 5, 3), f"Expected shape (2, 5, 3), got {out.shape}"
assert (out := a.sum(axis=(0, 1))).shape == (3, 4), f"Expected shape (3, 4), got {out.shape}"
assert (out := a.sum(axis=2, keepdim=True)).shape == (2, 5, 1, 4), f"Expected shape (2, 5, 1, 4), got {out.shape}"
assert (out := a.sum(axis=(0, 1), keepdim=True)).shape == (1, 1, 3, 4), f"Expected shape (1, 1, 3, 4), got {out.shape}"
