from tinygrad import Tensor

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# a.realize()
# print(a.uop)
# schedule, _ = a.schedule_with_vars()
# print("*"*80)
# print(schedule[0].ast)
# print(a.uop)
# b.realize()
# print(b.uop)

a.realize()
b.realize()

c = a + b
# c.realize()
print(c.uop)
schedule, _ = c.schedule_with_vars()
print("*"*80)
print(schedule[0].ast)

# a = Tensor([[1, 2, 3],
#             [4, 5, 6]])  # (2, 3)
#
# b = Tensor([[1, 2, 3, 4],
#             [5, 6, 7, 8],
#             [9, 10, 11, 12]])  # (3, 4)

# a.realize()
# b.realize()

# c = a @ b
# print(c.uop)
# c.realize()


# ***
# # Manual matmul decomposition - exactly what tinygrad does internally
# a_expanded = a.reshape(2, 1, 3).expand(2, 4, 3)
# b_expanded = b.reshape(1, 3, 4).permute(0, 2, 1).expand(2, 4, 3)
#
# # Multiply and sum over K dimension
# manual_matmul = (a_expanded * b_expanded).sum(axis=2)
# builtin_matmul = a @ b
#
# print("Manual result:")
# print(manual_matmul.numpy())
# # [[ 38  44  50  56]
# #  [ 83  98 113 128]]
#
# print("Results match:", (manual_matmul == builtin_matmul).all().item())
# # True
