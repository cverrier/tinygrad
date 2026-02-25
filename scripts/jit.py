# from tinygrad import Tensor, TinyJit
#
# multiplier = 10
#
# @TinyJit
# def f(x):
#     return (x * multiplier).realize()
#
# print("multiplier=10:", f(Tensor([5])).item())  # warmup
# print("multiplier=10:", f(Tensor([5])).item())  # capture
#
# multiplier = 99
#
# print("multiplier=99:", f(Tensor([5])).item())  # replay

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(x, multiplier):
#     return (x * multiplier).realize()
#
# print(f(Tensor([5]), Tensor([10])).item())  # warmup
# print(f(Tensor([5]), Tensor([10])).item())  # capture
# print(f(Tensor([5]), Tensor([99])).item())  # replay with new multiplier

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(x, use_square):
#     if use_square:
#         return (x * x).realize()
#     else:
#         return (x * 2).realize()
#
# print("use_square=True: ", f(Tensor([5]), True).item())   # warmup
# print("use_square=False:", f(Tensor([5]), False).item())  # capture
#
# print("use_square=True: ", f(Tensor([5]), True).item())   # replay

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f_square(x):
#     print("Executing f_square")
#     return (x * x).realize()
#
# @TinyJit
# def f_double(x):
#     print("Executing f_double")
#     return (x * 2).realize()
#
# # Choose which to call in Python (outside JIT)
# use_square = True
# result = f_square(Tensor([5])) if use_square else f_double(Tensor([5]))
# print(result.numpy())
#
# use_square = False
# result = f_square(Tensor([5])) if use_square else f_double(Tensor([5]))
# print(result.numpy())
#
# use_square = True
# result = f_square(Tensor([5])) if use_square else f_double(Tensor([5]))
# print(result.numpy())
#
# use_square = False
# result = f_square(Tensor([5])) if use_square else f_double(Tensor([5]))
# print(result.numpy())
#
# use_square = False
# result = f_square(Tensor([5])) if use_square else f_double(Tensor([5]))
# print(result.numpy())

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(a, tensor_list):
#     return (a + tensor_list[0]).realize()
#
# a = Tensor([1, 1, 1])
#
# print(f(a, [Tensor([10, 10, 10])]).numpy())  # warmup
# print(f(a, [Tensor([20, 20, 20])]).numpy())  # capture
# print(f(a, [Tensor([99, 99, 99])]).numpy())  # replay

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(a, b):
#     return (a + b).realize()
#
# x = Tensor([1, 2, 3])
# f(x, x.clone())

# from tinygrad import Tensor, TinyJit
# from tinygrad.helpers import Timing
#
# class SimpleModel:
#     def __init__(self):
#         self.w1 = Tensor.randn(100, 50)
#         self.w2 = Tensor.randn(50, 10)
#
#     @TinyJit
#     def forward(self, x):
#         x = (x @ self.w1).relu()
#         x = x @ self.w2
#         return x.realize()
#
# model = SimpleModel()
#
# for i in range(10):
#     x = Tensor.randn(32, 100)
#     with Timing(f"Call {i+1}: "):
#         out = model.forward(x)

# from tinygrad import Tensor, TinyJit
#
# counter = Tensor([0])#.realize()
#
# @TinyJit
# def increment():
#     counter.assign(counter + 1)#.realize()
#     return counter
#
# for i in range(7):
#     result = increment()
#     print(f"Call {i+1}: counter = {result.realize().item()}")

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(x):
#     a = (x + 1)           # Not realized - will fuse with next op
#     b = (a * 2).realize() # Force kernel boundary here
#     c = (b + 3)           # Separate kernel
#     return c
#
# for _ in range(3):
#     f(Tensor.randn(10))
#
# print(f"Kernels captured: {len(f.captured.jit_cache)}")

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def g(x):
#     a = (x + 1)    # All operations
#     b = (a * 2)    # fuse into
#     c = (b + 3)    # one kernel
#     return c
#
# for _ in range(3):
#     g(Tensor.randn(10))
#
# print(f"Kernels captured: {len(g.captured.jit_cache)}")

# from tinygrad import Tensor, TinyJit
#
# counter = Tensor([0])#.realize()
#
# @TinyJit
# def increment_broken():
#     global counter
#     counter = (counter + 1).realize()  # Creates NEW tensor!
#     return counter
#
# for i in range(5):
#     result = increment_broken()
#     print(f"Call {i+1}: counter = {result.item()}")

# from tinygrad import Tensor, TinyJit, Variable
#
# @TinyJit
# def add_jit(a, b):
#     return (a + b).realize()
#
# n = Variable("n", 1, 10).bind(3)
# # n.bind(3)
#
# add_jit(Tensor.ones(n, n), Tensor.ones(n, n))
# add_jit(Tensor.ones(n, n), Tensor.ones(n, n))  # Add second call
# add_jit(Tensor.randn(n, n), Tensor.randn(n, n))
# add_jit(Tensor.ones(n, n), Tensor.ones(n, n)) # 2
# add_jit(Tensor.ones(n, n), Tensor.ones(n, n))

# from tinygrad import Tensor, Variable
#
# n = Variable("n", 1, 10).bind(5)
#
# # This should work:
# a = Tensor.ones(n, 3)
# print(f"ones shape: {a.shape}")
#
# # This should work:
# b = Tensor.zeros(n, 3)
# print(f"zeros shape: {b.shape}")
#
# # This fails:
# c = Tensor.rand(n, 3)

# add_jit(Tensor.randn(200, 200), Tensor.randn(200, 200))

# from tinygrad import Tensor, TinyJit, Variable
#
# @TinyJit
# def add_jit(a, b):
#     return (a + b).realize()
#
# # Create base tensors with MAXIMUM size (10x10)
# base_a = Tensor.randn(10, 10)
# base_b = Tensor.randn(10, 10)
#
# for i in range(1, 5):
#     n = Variable("n", 1, 10).bind(i)
#
#     # Slice to get symbolic shape (n, n)
#     result = add_jit(base_a[:n, :n], base_b[:n, :n])
#     print(f"n={i}: result shape = {result.shape}")

# from tinygrad import Tensor, TinyJit, Variable
#
# @TinyJit
# def add_jit(a, b):
#     return (a + b).realize()
#
# base_a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# base_b = Tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
#
# for i in range(1, 4):
#     n = Variable("n", 1, 3).bind(i)
#     result = add_jit(base_a[:n, :n], base_b[:n, :n])
#     print(f"n={i}:")
#     print(result[:i, :i].numpy())
#     print()

# from tinygrad import Tensor, Variable
#
# n = Variable("n", 1, 10).bind(3)
# a = Tensor.ones(n, n)
#
# print(f"Is realized: {a.uop.is_realized}")
# print(f"Base buffer: {a.uop.base.realized}")

# from tinygrad import Tensor, TinyJit, Variable
# from tinygrad.helpers import Timing
#
# @TinyJit
# def add_jit(a, b):
#     return (a + b).realize()
#
# n = Variable("n", 1, 10).bind(3)
#
# # .contiguous() forces a real buffer allocation
# with Timing("First call: "):
#   add_jit(Tensor.ones(n, n).contiguous(), Tensor.ones(n, n).contiguous())
# with Timing("Second call: "):
#   add_jit(Tensor.ones(n, n).contiguous(), Tensor.ones(n, n).contiguous())
# with Timing("Third call: "):
#   add_jit(Tensor.ones(n, n).contiguous(), Tensor.ones(n, n).contiguous())
# with Timing("Fourth call: "):
#   add_jit(Tensor.ones(n, n).contiguous(), Tensor.ones(n, n).contiguous())
# with Timing("Fifth call: "):
#   add_jit(Tensor.ones(n, n).contiguous(), Tensor.ones(n, n).contiguous())
#
# print("Success!")

# from tinygrad import Tensor, TinyJit, Variable
#
# @TinyJit
# def add_jit(x):
#   return (x + 1).realize()
#
# n = Variable("n", 1, 10).bind(3)
#
# print(add_jit(Tensor.ones(n)).numpy())

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(x):
#     return (x @ x.T).realize()  # x @ x^T, output shape depends on input
#
# # Use with 10x5
# for i in range(4):
#     out = f(Tensor.randn(10, 5))
# print(f"Shape after first capture: {out.shape}")
#
# # Reset and use with 20x8
# # f.reset()
#
# for i in range(4):
#     out = f(Tensor.randn(20, 8))
# print(f"Shape after reset: {out.shape}")

# from tinygrad import Tensor, TinyJit, Variable
#
# @TinyJit
# def process_sequence(x):
#     # Sum over sequence dimension
#     return x.sum(axis=1)
#
# # Base data: batch=2, max_seq=100, features=8
# data = Tensor.randn(2, 100, 8)
#
# for seq_len in [10, 25, 50, 75]:
#     # Create variable for current sequence length
#     s = Variable("s", 1, 100).bind(seq_len)
#
#     # Slice to current length
#     result = process_sequence(data[:, :s, :])
#     print(f"seq_len={seq_len}: output shape = {result.shape}")
#
# print(f"\nKernels captured: {len(process_sequence.captured.jit_cache)}")

# from tinygrad import Tensor
#
# # First run: compiles kernel
# a = Tensor.randn(100, 100).realize()
# b = Tensor.randn(100, 100).realize()
# (a + b).realize()

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(a, b):
#     x = (a + b).realize()    # Kernel 1
#     y = (x * a).realize()    # Kernel 2
#     z = (y - b).realize()    # Kernel 3
#     return z
#
# # Warmup + capture
# f(Tensor.empty(4, 4), Tensor.empty(4, 4))
# f(Tensor.empty(4, 4), Tensor.empty(4, 4))
#
# # Check what was captured
# print(f"Kernels captured: {len(f.captured.jit_cache)}")

# from tinygrad import Tensor, TinyJit
#
# @TinyJit
# def f(a, b):
#     x = (a + b).realize()    # Kernel 1
#     y = (x * a).realize()    # Kernel 2
#     z = (y - b).realize()    # Kernel 3
#     return z
#
# a = Tensor.empty(4, 4).realize()
# b = Tensor.empty(4, 4).realize()
#
# print("Call 1 (warmup):")
# f(a, b)
#
# print("\nCall 2 (capture):")
# f(a, b)
#
# print("\nCall 3 (first replay - builds GPU graph):")
# f(a, b)
#
# print("\nCall 4 (replay with GPU graph):")
# f(a, b)

# from tinygrad import Tensor, TinyJit, Device
# from tinygrad.helpers import Timing
#
# # Create 10 separate kernels (forced with .realize())
# def ten_kernels(x):
#     for _ in range(10):
#         x = (x + 1).realize()
#     return x
#
# a = Tensor.empty(4, 4).realize()
#
# # Without JIT - 10 separate commands
# print("Without JIT:")
# for i in range(3):
#     with Timing(f"  Run {i+1}: "):
#         ten_kernels(a)

# from tinygrad import Tensor, TinyJit
# from tinygrad.helpers import Timing
#
# @TinyJit
# def ten_kernels_jit(x):
#     for _ in range(10):
#         x = (x + 1).realize()
#     return x
#
# a = Tensor.empty(4, 4).realize()
#
# print("With JIT:")
# for i in range(6):
#     with Timing(f"  Run {i+1}: "):
#         ten_kernels_jit(a)

# from tinygrad import Tensor, TinyJit
# from tinygrad.helpers import Timing
#
# @TinyJit
# def ten_kernels_jit(x):
#     for _ in range(10):
#         x = (x + 1).realize()
#     return x
#
# a = Tensor.empty(4, 4).realize()
#
# print("With JIT + DEBUG:")
# for i in range(6):
#     print(f"\n--- Run {i+1} ---")
#     with Timing("  Time: "):
#         ten_kernels_jit(a)

from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Timing

def make_chain(n):
    @TinyJit
    def f(x):
        for _ in range(n):
            x = (x + 1).realize()
        return x
    return f

a = Tensor.empty(4, 4).realize()

for n_kernels in [10, 30, 50, 100, 500]:
    f = make_chain(n_kernels)

    # Warmup + capture + graph build
    for _ in range(5):
        f(a)

    # Measure steady state
    with Timing(f"{n_kernels} kernels: "):
        f(a)
