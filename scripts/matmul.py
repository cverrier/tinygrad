from tinygrad import Tensor

# # Tiny 2x2 matrices - easy to verify by hand
# a = Tensor([[1, 2], [3, 4]]).realize()
# b = Tensor([[5, 6], [7, 8]]).realize()
#
# # Matrix multiply
# c = a @ b
# c.realize()

# # Larger matrices: 4x8 @ 8x4 = 4x4
# a = Tensor.rand(4, 8).realize()
# b = Tensor.rand(8, 4).realize()
#
# c = a @ b
# c.realize()

# # K=128 - too large to fully unroll
# a = Tensor.rand(16, 128).realize()
# b = Tensor.rand(128, 16).realize()
#
# c = a @ b
# c.realize()

# a = Tensor.rand(4, 32).realize()
# b = Tensor.rand(32, 4).realize()
#
# c = a @ b
# c.realize()

# # Simple 2x3 @ 3x4 = 2x4
# a = Tensor([[1, 2, 3],
#             [4, 5, 6]]).realize()  # Shape: (2, 3)
#
# b = Tensor([[1, 2, 3, 4],
#             [5, 6, 7, 8],
#             [9, 10, 11, 12]]).realize()  # Shape: (3, 4)
#
# # Create matmul WITHOUT realizing - keep it lazy
# c = a @ b  # Shape: (2, 4)
#
# # Print the UOp graph
# print("=== UOp Graph for Matmul ===")
# print(c.uop)

a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # (2, 3)

b = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])  # (3, 4)

# Manual matmul decomposition - exactly what tinygrad does internally
a_expanded = a.reshape(2, 1, 3).expand(2, 4, 3)
b_expanded = b.permute(1, 0).reshape(1, 4, 3).expand(2, 4, 3)

print("A expanded shape:", a_expanded.shape)  # (2, 4, 3)
print("B expanded shape:", b_expanded.shape)  # (2, 4, 3)

# Multiply and sum
manual_matmul = (a_expanded * b_expanded).sum(axis=2)
builtin_matmul = a @ b

print("\nManual result:")
print(manual_matmul.numpy())
print("\nBuiltin matmul result:")
print(builtin_matmul.numpy())
print("\nResults match:", (manual_matmul.numpy() == builtin_matmul.numpy()).all())
