# Matrix Multiplication in tinygrad: A First Principles Walkthrough

This document explains how tinygrad implements matrix multiplication, from the mathematical definition to generated GPU code.

## Prerequisites

Before reading this, you should understand:
- Lazy evaluation: operations build UOp graphs, nothing executes until `.realize()`
- UOp structure: everything is a UOp with (op, dtype, src, arg, tag)
- Basic UOp types: `BUFFER`, `RESHAPE`, `EXPAND`, `MUL`, `REDUCE_AXIS`
- Kernel fusion: multiple operations become one kernel

## The Mathematical Definition

Matrix multiplication for A (M×K) and B (K×N) producing C (M×N):

```
C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
```

Each output element is a **dot product** - multiply corresponding elements, then sum.

### General Rule for `A @ B`

- **Contract over**: last axis of A and second-to-last axis of B
- These dimensions must match (both equal K)

For 2D matrices:
- Last axis of A = axis 1 (size K)
- Second-to-last axis of B = axis 0 (size K)

This extends naturally to batched matmul:
```
A: (batch, M, K)  # last axis = K
B: (batch, K, N)  # second-to-last axis = K
C: (batch, M, N)  # K is summed out
```

## Key Insight: Matmul = Reshape + Expand + Multiply + Reduce

Matmul is **not** a primitive operation in tinygrad. It decomposes into simpler operations:

1. **Reshape** A and B to align dimensions
2. **Expand** (broadcast) so they have the same shape
3. **Multiply** elementwise
4. **Reduce** (sum) over the K dimension

This decomposition is elegant because tinygrad already knows how to optimize each of these operations individually.

## Concrete Example: 2×3 @ 3×4

Let's trace through a specific example:

```python
from tinygrad import Tensor

a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # Shape: (2, 3), M=2, K=3

b = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])  # Shape: (3, 4), K=3, N=4

c = a @ b  # Shape: (2, 4)
```

### Shape Transformations

**Matrix A: (2, 3) → ready for broadcast**
```
Original:     (2, 3)      "2 rows, 3 cols"
Reshape:      (2, 1, 3)   "2 rows, 1 output col (placeholder), 3 K values"
Expand:       (2, 4, 3)   "broadcast to all 4 output columns"
```

**Matrix B: (3, 4) → ready for broadcast**
```
Original:     (3, 4)      "3 rows (K), 4 cols"
Reshape:      (1, 3, 4)   "1 row (placeholder), 3 K values, 4 cols"
Permute:      (1, 4, 3)   "swap last two dims to align K as last axis"
Expand:       (2, 4, 3)   "broadcast to all 2 output rows"
```

**Combined**
```
A:            (2, 4, 3)   element [i, j, k] = A[i, k]
B:            (2, 4, 3)   element [i, j, k] = B[k, j]
MUL:          (2, 4, 3)   element [i, j, k] = A[i,k] * B[k,j]
REDUCE(k):    (2, 4)      element [i, j] = Σk A[i,k] * B[k,j]  ← This is matmul!
```

### Manual Verification

You can manually decompose matmul to verify:

```python
from tinygrad import Tensor

a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # (2, 3)

b = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])  # (3, 4)

# Manual matmul decomposition - exactly what tinygrad does internally
a_expanded = a.reshape(2, 1, 3).expand(2, 4, 3)
b_expanded = b.permute(1, 0).reshape(1, 4, 3).expand(2, 4, 3)

# Multiply and sum over K dimension
manual_matmul = (a_expanded * b_expanded).sum(axis=2)
builtin_matmul = a @ b

print("Manual result:")
print(manual_matmul.numpy())
# [[ 38  44  50  56]
#  [ 83  98 113 128]]

print("Results match:", (manual_matmul.numpy() == builtin_matmul.numpy()).all())
# True
```

## The UOp Graph

When you create a matmul without realizing it, tinygrad builds a UOp graph:

```python
c = a @ b
print(c.uop)
```

The graph structure (simplified):

```
A: BUFFER(6)                         B: BUFFER(12)
     │                                     │
     ▼                                     ▼
RESHAPE (2, 3)                       RESHAPE (3, 4)
     │                                     │
     ▼                                     ▼
RESHAPE (2, 1, 3)                    RESHAPE (1, 3, 4)
     │                                     │
     ▼                                     ▼
EXPAND (2, 4, 3)                     PERMUTE (0, 2, 1) → (1, 4, 3)
     │                                     │
     │                                     ▼
     │                               EXPAND (2, 4, 3)
     │                                     │
     └──────────────┬──────────────────────┘
                    ▼
              MUL (2, 4, 3)
                    │
                    ▼
         REDUCE_AXIS sum axis=2
                    │
                    ▼
             RESHAPE (2, 4)
                    │
                    ▼
               Output C
```

Key UOp types in this graph:
- `BUFFER`: The actual data storage
- `RESHAPE`: Change shape without moving data
- `PERMUTE`: Reorder axes without moving data
- `EXPAND`: Broadcast to larger shape (no data copy)
- `MUL`: Elementwise multiplication
- `REDUCE_AXIS`: Sum over specified axis

## Generated Kernel Code

tinygrad generates different kernels based on matrix size and available optimizations.

### Tiny Matrices: Fully Unrolled (2×2 @ 2×2)

```python
import os
os.environ['DEBUG'] = '4'

from tinygrad import Tensor
a = Tensor([[1, 2], [3, 4]]).realize()
b = Tensor([[5, 6], [7, 8]]).realize()
c = (a @ b).realize()
```

Generated kernel (`r_2_2_2`):

```c
kernel void r_2_2_2(device int* data0, device int* data1, device int* data2,
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 2 - output row */
  int lidx1 = lid.y; /* 2 - output col */
  int alu0 = (lidx0<<1);

  // Load both elements from row lidx0 of A
  int val0 = (*(data1+(alu0+1)));  // A[lidx0, 1]
  int val1 = (*(data1+alu0));       // A[lidx0, 0]

  // Load both elements from column lidx1 of B
  int val2 = (*(data2+lidx1));      // B[0, lidx1]
  int val3 = (*(data2+(lidx1+2)));  // B[1, lidx1]

  // Dot product - fully unrolled, no loop!
  *(data0+(lidx1+alu0)) = ((val1*val2)+(val0*val3));
}
```

Key observations:
- **4 threads** (2×2), one per output element
- **No loop** - K=2 is fully unrolled
- Each thread computes one dot product

### Medium Matrices: Vectorized Loads (4×8 @ 8×4)

```python
a = Tensor.rand(4, 8).realize()
b = Tensor.rand(8, 4).realize()
c = (a @ b).realize()
```

Generated kernel (`r_4_4_8`):

```c
kernel void r_4_4_8(device float* data0, device float* data1, device float* data2, ...) {
  int lidx0 = lid.x; /* 4 - output row */
  int lidx1 = lid.y; /* 4 - output col */
  int alu0 = (lidx0<<3);

  // Vectorized loads - 4 floats at once using float4
  float4 val8 = (*((device float4*)((data1+(alu0+4)))));  // A[row, 4:8]
  float4 val9 = (*((device float4*)((data1+alu0))));       // A[row, 0:4]

  // Individual loads from B column
  float val0 = (*(data2+lidx1));       // B[0, col]
  float val1 = (*(data2+(lidx1+4)));   // B[1, col]
  // ... val2 through val7 ...

  // Unrolled dot product using vector components
  *(data0+(lidx1+(lidx0<<2))) =
    (val9.x*val0)+(val9.y*val1)+(val9.z*val2)+(val9.w*val3)+
    (val8.x*val4)+(val8.y*val5)+(val8.z*val6)+(val8.w*val7);
}
```

Key observations:
- **Vectorized loads**: `float4` loads 4 values in one memory transaction
- **Still unrolled**: K=8 small enough to avoid loops
- `.x`, `.y`, `.z`, `.w` access vector components

### Large Matrices: Tensor Cores (16×128 @ 128×16)

```python
a = Tensor.rand(16, 128).realize()
b = Tensor.rand(128, 16).realize()
c = (a @ b).realize()
```

Generated kernel uses Apple's SIMD group matrix operations:

```c
float2 __WMMA_8_8_8_float_float(float2 a, float2 b, float2 c){
  simdgroup_float8x8 mat_a, mat_b, mat_c;
  // ... load matrices ...
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);
}

kernel void r_32_2_2_2_16(device float* data0, device float* data1, device float* data2, ...) {
  float acc0[8];  // Each thread accumulates 8 values

  // Initialize accumulators
  *(acc0+0) = 0.0f;
  // ...

  // Reduction loop - K=128 processed in chunks of 8
  for (int Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    // Load tiles from A and B
    float2 val0 = (*((device float2*)((data1+...))));
    float2 val1 = ...;

    // Tensor core matrix multiply-accumulate
    float2 wmma0 = __WMMA_8_8_8_float_float(val0, val2, float2(acc0[6], acc0[7]));
    // ... more WMMA calls ...

    // Update accumulators
    *(acc0+0) = wmma3.x;
    // ...
  }

  // Write results
  *((device float2*)((data0+...))) = float2(acc0[0], acc0[1]);
}
```

Key observations:
- **WMMA**: Warp Matrix Multiply-Accumulate using hardware tensor cores
- **`simdgroup_float8x8`**: Apple Silicon's 8×8 matrix unit
- **Reduction loop**: K=128 ÷ 8 = 16 iterations
- **Accumulator pattern**: initialize → loop → write

### Naive Implementation (NOOPT=1)

To see the simplest possible matmul, disable optimizations:

```python
import os
os.environ['DEBUG'] = '4'
os.environ['NOOPT'] = '1'

from tinygrad import Tensor
a = Tensor.rand(4, 32).realize()
b = Tensor.rand(32, 4).realize()
c = (a @ b).realize()
```

Generated kernel (`r_4_4_32`):

```c
kernel void r_4_4_32(device float* data0, device float* data1, device float* data2, ...) {
  float acc0[1];                    // Single accumulator
  int gidx0 = gid.x; /* 4 */        // Output column
  int gidx1 = gid.y; /* 4 */        // Output row

  *(acc0+0) = 0.0f;                 // Initialize to 0

  for (int Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    // Load A[row, k]
    float val0 = (*(data1+((gidx1<<5)+Ridx0)));
    // Load B[k, col]
    float val1 = (*(data2+(gidx0+(Ridx0<<2))));
    // Accumulate
    *(acc0+0) = ((*(acc0+0))+(val0*val1));
  }

  // Write C[row, col]
  *(data0+(gidx0+(gidx1<<2))) = (*(acc0+0));
}
```

This is the **textbook implementation** - directly implements the mathematical formula:

```
C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
```

## Kernel Naming Convention

Kernel names encode their structure:
- `E_N`: Elementwise kernel, N elements
- `r_M_N_K`: Reduction kernel, output M×N, reduce over K

Examples:
- `r_2_2_2`: 2×2 output, K=2 reduction
- `r_4_4_32`: 4×4 output, K=32 reduction
- `r_32_2_2_2_16`: More complex tiling structure

## Optimization Levels Summary

| Matrix Size | K | Optimization | Code Pattern |
|-------------|---|--------------|--------------|
| 2×2 @ 2×2 | 2 | UNROLL + LOCAL | No loop, fully unrolled |
| 4×8 @ 8×4 | 8 | UNROLL + LOCAL | No loop, vectorized float4 |
| 16×128 @ 128×16 | 128 | Tensor Cores | Loop with SIMD matrix ops |
| 4×32 @ 32×4 (NOOPT) | 32 | None | Simple accumulator loop |

## Memory Access Patterns

In the naive implementation:
- **A access**: `A[gidx1, Ridx0]` → Sequential along row (cache-friendly)
- **B access**: `B[Ridx0, gidx0]` → Strided down column (cache-unfriendly)

This strided access pattern is why naive matmul is slow. Optimized implementations use:
- **Tiling**: Process small blocks that fit in cache
- **Shared memory**: Load tiles cooperatively
- **Tensor cores**: Hardware-accelerated matrix units

## Environment Variables

Useful for exploring matmul:

```bash
DEBUG=4 python script.py      # Show generated kernel code
NOOPT=1 python script.py      # Disable optimizations
VIZ=1 python script.py        # Enable graph visualization
```

## Key Takeaways

1. **Matmul is decomposed**: reshape → expand → multiply → reduce
2. **All movement ops are free**: reshape, permute, expand don't copy data
3. **tinygrad auto-optimizes**: Chooses unrolling, vectorization, or tensor cores based on size
4. **The UOp graph captures intent**: Scheduling and codegen handle optimization
5. **Same algorithm, different code**: The mathematical operation is constant; only the implementation varies
