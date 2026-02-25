# Understanding tinygrad Internals: A First Principles Guide

This guide walks through tinygrad's internals step by step, using simple examples to show how tensor operations become GPU kernels.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Lazy Evaluation](#lazy-evaluation)
3. [UOps: The Universal IR](#uops-the-universal-ir)
4. [Triggering Execution with realize()](#triggering-execution-with-realize)
5. [Seeing Generated Code with DEBUG=4](#seeing-generated-code-with-debug4)
6. [Kernel Fusion](#kernel-fusion)
7. [Reductions](#reductions)
8. [Optimizations](#optimizations)
9. [Quick Reference](#quick-reference)

---

## Core Philosophy

tinygrad sits "between PyTorch and micrograd":

- **PyTorch-like API**: Familiar tensor operations with autograd
- **Fully transparent**: The entire compilation pipeline is visible Python code
- **Lazy by default**: Operations build graphs; nothing executes until you ask
- **Automatic fusion**: Multiple operations become single optimized kernels

Unlike PyTorch (C++ internals) or JAX (XLA black box), you can see exactly what tinygrad does at every step.

---

## Lazy Evaluation

Creating a tensor doesn't compute anything - it builds a graph:

```python
from tinygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
print("Tensor created - nothing computed yet!")
print(f"x.uop.op = {x.uop.op}")  # Shows: Ops.COPY
```

The tensor holds a **UOp** (Universal Operation) representing a plan: "copy this data from Python to GPU."

---

## UOps: The Universal IR

Everything in tinygrad is a UOp. View the full graph:

```python
from tinygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
print(x.uop)
```

Output:
```
UOp(Ops.COPY, dtypes.float, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.float, arg=3, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),
  UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),))
```

Reading this as a tree:
```
COPY (float)                    <- "Copy this data..."
+-- BUFFER (float, arg=3)       <- "...from this 3-element buffer..."
|   +-- UNIQUE (arg=0)          <- "...with unique ID 0..."
|   +-- DEVICE ('PYTHON')       <- "...which lives in Python memory..."
+-- DEVICE ('METAL')            <- "...to the METAL GPU"
```

**Key insight**: This is just a plan. No data has moved yet.

---

## Triggering Execution with realize()

Call `.realize()` to execute the graph:

```python
from tinygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
print(f"Before: x.uop.op = {x.uop.op}")  # Ops.COPY

x.realize()

print(f"After: x.uop.op = {x.uop.op}")   # Ops.RESHAPE
print(x.uop)
```

After realize:
```
UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
  UOp(Ops.BUFFER, dtypes.float, arg=3, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),
    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),
  UOp(Ops.CONST, dtypes.index, arg=3, src=()),))
```

The `COPY` was executed and replaced with a realized `BUFFER` on the GPU.

---

## Seeing Generated Code with DEBUG=4

This is where tinygrad shines. Set `DEBUG=4` to see actual GPU code:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor([1.0, 2.0, 3.0]).realize()
y = x + 1
y.realize()
"
```

Output (simplified):
```metal
kernel void E_3(device float* data0_3, device float* data1_3, ...) {
  int lidx0 = lid.x; /* 3 */
  float val0 = (*(data1_3+lidx0));
  *(data0_3+lidx0) = (val0+1.0f);
}
```

This is real Metal shader code! Breaking it down:

| Line | Meaning |
|------|---------|
| `data0_3` | Output buffer (3 floats) |
| `data1_3` | Input buffer (3 floats) |
| `lid.x` | Thread index (0, 1, or 2) |
| `*(data1_3+lidx0)` | Load element at thread's index |
| `(val0+1.0f)` | Add 1.0 |
| `*(data0_3+lidx0) = ...` | Store result |

Three GPU threads run in parallel, each handling one element.

---

## Kernel Fusion

Multiple operations become ONE kernel:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor([1.0, 2.0, 3.0]).realize()
y = (x + 1) * 2  # Two operations
y.realize()
"
```

Generated kernel:
```metal
kernel void E_3(...) {
  int lidx0 = lid.x;
  float val0 = (*(data1_3+lidx0));
  *(data0_3+lidx0) = ((val0+1.0f)*2.0f);  // Both ops fused!
}
```

Both `+1` and `*2` are in a single expression. This matters because:

**Without fusion** (hypothetical):
1. Kernel 1: Load -> Add 1 -> Store to temp
2. Kernel 2: Load from temp -> Multiply 2 -> Store

**With fusion**:
1. Kernel 1: Load -> Add 1 -> Multiply 2 -> Store

One kernel = one memory round-trip. Memory bandwidth is often the GPU bottleneck.

---

## Reductions

Reductions (sum, max, etc.) are fundamentally different - they combine elements:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor([1.0, 2.0, 3.0]).realize()
y = x.sum()
y.realize()
"
```

Generated kernel (for 3 elements, fully unrolled):
```metal
kernel void r_3(device float* data0_1, device float* data1_3, ...) {
  float val0 = (*(data1_3+2));
  float2 val1 = (*((device float2*)((data1_3+0))));
  *(data0_1+0) = (val1.x+val1.y+val0);
}
```

Notice:
- Kernel name is `r_3` (r = reduce) vs `E_3` (E = elementwise)
- Loads data as `float2` vector (2 elements) + 1 scalar (this is an optimization done by tinygrad, if you want to see the actual `for` loop, run the same code snippet with `NOOPT=1`)
- Sums in one expression: `val1.x + val1.y + val0`

### Fusion with Reduction

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor([1.0, 2.0, 3.0]).realize()
y = (x + 1).sum()
y.realize()
"
```

Generated kernel:
```metal
*(data0_1+0) = (val0+val1.y+val1.x+3.0f);
```

tinygrad algebraically simplified `(x[0]+1) + (x[1]+1) + (x[2]+1)` to `x[0] + x[1] + x[2] + 3`!

### Larger Reductions: Parallel Strategy

For 256 elements:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor.rand(256).realize()
y = (x * 2).sum()
y.realize()
"
```

The reduction kernel uses a two-stage parallel strategy:

```cpp
#include <metal_stdlib>
using namespace metal;
kernel void r_16_16(device float* data0_1, device float* data1_256, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float temp0[16]; // Shared memory for partial sums
  float acc0[1];
  float acc1[1];
  int lidx0 = lid.x; /* 16 */
  // Stage 1: each of 16 threads sums 16 elements
  *(acc0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    float val0 = (*(data1_256+((lidx0<<4)+Ridx0)));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(temp0+lidx0) = (*(acc0+0)); // Store partial sum
  threadgroup_barrier(mem_flags::mem_threadgroup); // Sync all threads
  // Stage 2: each thread sums the 16 partial sums
  // Although only one thread needs to do this, all do it for simplicity
  // and avoid divergent control flow in the kernel
  *(acc1+0) = 0.0f;
  for (int Ridx101 = 0; Ridx101 < 16; Ridx101++) {
    float val1 = (*(temp0+Ridx101));
    *(acc1+0) = ((*(acc1+0))+val1);
  }
  // Only thread 0 writes the final result
  bool alu8 = (((bool)(lidx0))!=1);
  if (alu8) {
    *(data0_1+0) = ((*(acc1+0))*2.0f);
  }
}
```

The `*2` operation is fused into the final store.

---

## Optimizations

### Constant Folding

tinygrad can compute constant expressions at compile time:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
y = (Tensor.ones(256) * 2).sum()
y.realize()
"
```

Output:
```
scheduled    0 kernels
```

**Zero kernels!** tinygrad computed `256 * 2 = 512` during graph construction.

### Vectorized Loads

tinygrad uses vector types (`float4`) to load multiple elements at once:

```bash
DEBUG=4 python3 -c "
from tinygrad import Tensor
x = Tensor.arange(8).float().realize()
y = x.sum()
y.realize()
"
```

Arange kernel (2 threads, each writes 4 values):
```metal
kernel void E_2_4(device float* data0_8, ...) {
  int lidx0 = lid.x; /* 2 */
  int alu0 = (lidx0<<2);
  *((device float4*)((data0_8+alu0))) = float4(
    ((float)(alu0)),
    ((float)((alu0+1))),
    ((float)((alu0+2))),
    ((float)((alu0+3)))
  );
}
```

Sum kernel (fully unrolled with vector loads):
```metal
kernel void r_8(device float* data0_1, device float* data1_8, ...) {
  float4 val0 = (*((device float4*)((data1_8+0))));   // Load [0,1,2,3]
  float4 val1 = (*((device float4*)((data1_8+4))));   // Load [4,5,6,7]
  *(data0_1+0) = (val0.x+val0.y+val0.z+val0.w+val1.x+val1.y+val1.z+val1.w);
}
```

---

## Quick Reference

### Debug Levels

```bash
DEBUG=1  # Basic info
DEBUG=2  # More details
DEBUG=3  # Kernel-level operations
DEBUG=4  # Generated source code
```

### Kernel Naming Convention

| Prefix | Meaning |
|--------|---------|
| `E_N` | **E**lementwise, N elements |
| `r_N` | **r**educe, N elements |

### Key Concepts Summary

| Concept | What It Means |
|---------|---------------|
| **Lazy evaluation** | Operations build UOp graphs; `.realize()` executes |
| **UOp** | Universal Operation - the IR for everything |
| **Kernel fusion** | Multiple ops become one kernel |
| **Constant folding** | Known values computed at compile time |
| **Vectorization** | `float4` loads/stores 4 elements at once |
| **Unrolling** | Small loops become direct expressions |
| **Parallel reduction** | Large sums use shared memory + barriers |

### When to Use realize()

```python
# In production: let tinygrad fuse everything
x = Tensor.rand(1000, 1000)
y = ((x + 1) * 2).sqrt().sum()
y.realize()  # One optimized kernel (or minimal kernels)

# For debugging: separate to see each kernel
x = Tensor.rand(1000, 1000).realize()
y = (x + 1).realize()
z = (y * 2).realize()
```

---

## Next Steps

- **Matrix operations**: 2D tensors, matmul, convolutions
- **Autograd**: How gradients flow through the graph
- **Scheduling**: How tinygrad breaks graphs into kernels
- **Device backends**: How the same UOps become CUDA/Metal/OpenCL

Run any operation with `DEBUG=4` to see exactly what tinygrad generates!
