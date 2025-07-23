from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UPat

a = Tensor.empty(4, 4)
# b = Tensor.empty(4, 4)
# c = a + b
# c.realize()
b = a + 1
b.realize()

# Create the UPat to match STORE operations
# store_pattern = UPat(
#   Ops.STORE,  # Match STORE operation
#   dtype=dtypes.void,  # With void return type
#   src=(
#     UPat(Ops.INDEX, name="ptr"),  # First source: memory pointer (INDEX)
#     UPat(Ops.ADD, name="value"),  # Second source: computed value (ADD)
#   ),
#   name="store_op",  # Name the matched STORE operation
# )
#
# print(store_pattern)
