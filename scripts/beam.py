from tinygrad import Tensor
from tinygrad.codegen import full_rewrite

# Your original code
out = Tensor.empty(4, 4).sum(1).realize()

# Get the UOp from the realized tensor
uop = out.uop

# Extract the kernel AST (this is what DEBUG=6 shows you)
if uop.op.name == "ASSIGN":
  kernel_ast = uop.src[1].arg.ast  # Get the kernel AST
elif uop.op.name == "KERNEL":
  kernel_ast = uop.arg.ast
else:
  kernel_ast = uop

# Convert to linearized UOps (this is what DEBUG=6 prints)
uops = full_rewrite(kernel_ast)

# Print the final UOp in nested format - this is exactly what you want!
print("# The root UOp (usually SINK):")
print(repr(uops[-1]))

print("\n# Individual UOps from the list:")
for i, u in enumerate(uops):
  print(f"# UOp {i}:")
  print(repr(u))
  print()
