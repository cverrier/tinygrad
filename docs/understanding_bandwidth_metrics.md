# Understanding tinygrad's Bandwidth Metrics: A Tutorial

When running tinygrad with `DEBUG=2` or higher, you'll see output lines like this:

```
*** METAL      1 r_64_16_32_4_2_4_4_256    arg  3 mem  0.05 GB tm  5049.83us/  5.05ms (3402 GFLOPS  10|428  GB/s)
*** METAL      1 E_78125_32_4n1            arg  3 mem  0.12 GB tm   767.25us/  0.77ms (  13 GFLOPS 158|158  GB/s)
```

Notice the pattern `10|428 GB/s` and `158|158 GB/s`. What do these two numbers mean? Why are they sometimes different and sometimes the same?

This tutorial explains these metrics from first principles.

---

## Table of Contents

1. [GPU Memory Hierarchy](#1-gpu-memory-hierarchy)
2. [What Are Load and Store Instructions?](#2-what-are-load-and-store-instructions)
3. [A Simple Example: Vector Addition](#3-a-simple-example-vector-addition)
4. [A More Complex Example: Matrix Multiplication](#4-a-more-complex-example-matrix-multiplication)
5. [Defining the Two Bandwidth Metrics](#5-defining-the-two-bandwidth-metrics)
6. [Why the Numbers Differ: Data Reuse and Caching](#6-why-the-numbers-differ-data-reuse-and-caching)
7. [Interpreting Real Benchmark Results](#7-interpreting-real-benchmark-results)
8. [Summary](#8-summary)

---

## 1. GPU Memory Hierarchy

Before understanding the bandwidth metrics, you need to understand how GPU memory is organized.

A GPU has a **memory hierarchy** with different levels of storage:

```
┌─────────────────────────────────────────────────────┐
│                    GPU Chip                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Thread 1   │  │  Thread 2   │  │  Thread N   │  │
│  │ [Registers] │  │ [Registers] │  │ [Registers] │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│                         │                           │
│                         ▼                           │
│              ┌─────────────────────┐                │
│              │   Cache / Shared    │  ← Fast        │
│              │      Memory         │    (on-chip)   │
│              └─────────────────────┘                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │    Global Memory    │  ← Slow (off-chip)
              │   (Main GPU RAM)    │    e.g., M1 Pro: 205 GB/s max
              └─────────────────────┘
```

### Key Points

- **Registers**: Fastest storage, private to each thread, very limited capacity.
- **Cache / Shared Memory**: Fast on-chip memory shared among threads. When data is loaded from global memory, it may be cached here for faster subsequent access.
- **Global Memory**: The main GPU RAM. Large capacity but slow to access. Has a maximum bandwidth (e.g., 205 GB/s for Apple M1 Pro).

When a GPU thread needs data:
1. It first checks the cache.
2. If the data is there (**cache hit**), access is fast.
3. If not (**cache miss**), it must fetch from global memory (slow).

---

## 2. What Are Load and Store Instructions?

GPU kernels are programs that run on the GPU. They consist of **instructions**. Two important types are:

- **LOAD**: Read a value from memory into a register.
- **STORE**: Write a value from a register to memory.

Consider this simple kernel that adds two arrays element-wise:

```python
# Pseudocode for what each GPU thread executes
def add_kernel(a, b, c, i):
    x = a[i]      # LOAD instruction: read from address a+i
    y = b[i]      # LOAD instruction: read from address b+i
    z = x + y     # ALU instruction: compute in registers
    c[i] = z      # STORE instruction: write to address c+i
```

Each thread executes:
- 2 LOAD instructions (reading `a[i]` and `b[i]`)
- 1 STORE instruction (writing `c[i]`)

---

## 3. A Simple Example: Vector Addition

Let's trace through a concrete example with small numbers.

### Setup

We have two input arrays and one output array, each with 4 elements:
- `a = [1, 2, 3, 4]`
- `b = [5, 6, 7, 8]`
- `c = a + b = [6, 8, 10, 12]`

Each element is a 32-bit float (4 bytes).

### Kernel Execution

The kernel launches **4 threads**, one per element:

| Thread | Operations | Bytes Transferred |
|--------|------------|-------------------|
| 0 | Load `a[0]`, Load `b[0]`, Store `c[0]` | 3 × 4 = 12 bytes |
| 1 | Load `a[1]`, Load `b[1]`, Store `c[1]` | 3 × 4 = 12 bytes |
| 2 | Load `a[2]`, Load `b[2]`, Store `c[2]` | 3 × 4 = 12 bytes |
| 3 | Load `a[3]`, Load `b[3]`, Store `c[3]` | 3 × 4 = 12 bytes |

### Calculating the Metrics

**Total bytes in load/store instructions (`lds`)**:
```
lds = 4 threads × 12 bytes/thread = 48 bytes
```

**Unique memory accessed (`mem`)**:
```
Array a: 4 elements × 4 bytes = 16 bytes
Array b: 4 elements × 4 bytes = 16 bytes
Array c: 4 elements × 4 bytes = 16 bytes
Total mem = 48 bytes
```

**Result**: For vector addition, `lds = mem = 48 bytes`.

This is because **each memory location is accessed exactly once**. No element is read or written more than once.

---

## 4. A More Complex Example: Matrix Multiplication

Now consider matrix multiplication, where data reuse occurs.

### Setup

We have two 2×2 input matrices and one 2×2 output matrix:

```
A = [a00  a01]    B = [b00  b01]    C = A @ B = [c00  c01]
    [a10  a11]        [b10  b11]                [c10  c11]
```

The formula for each output element:
```
c00 = a00×b00 + a01×b10
c01 = a00×b01 + a01×b11
c10 = a10×b00 + a11×b10
c11 = a10×b01 + a11×b11
```

### Counting Unique Memory (`mem`)

Each matrix has 4 elements × 4 bytes = 16 bytes:
```
mem = 16 (A) + 16 (B) + 16 (C) = 48 bytes
```

### Counting Load/Store Instructions (`lds`)

Let's count how many times each element is loaded:

**For output `c00`**: Load `a00`, `a01`, `b00`, `b10` → 4 loads
**For output `c01`**: Load `a00`, `a01`, `b01`, `b11` → 4 loads
**For output `c10`**: Load `a10`, `a11`, `b00`, `b10` → 4 loads
**For output `c11`**: Load `a10`, `a11`, `b01`, `b11` → 4 loads

Now count how many times each element appears across all outputs:

| Element | Used in | Times Loaded |
|---------|---------|--------------|
| `a00` | c00, c01 | 2 |
| `a01` | c00, c01 | 2 |
| `a10` | c10, c11 | 2 |
| `a11` | c10, c11 | 2 |
| `b00` | c00, c10 | 2 |
| `b01` | c01, c11 | 2 |
| `b10` | c00, c10 | 2 |
| `b11` | c01, c11 | 2 |

**Total loads**: 8 elements × 2 loads each = 16 loads × 4 bytes = 64 bytes
**Total stores**: 4 elements × 4 bytes = 16 bytes
**Total `lds`**: 64 + 16 = **80 bytes**

### Result

For this 2×2 matrix multiplication:
- `mem` = 48 bytes (unique data)
- `lds` = 80 bytes (all load/store instructions)
- **Ratio**: 80 / 48 ≈ 1.67×

### Scaling to Larger Matrices

For n×n matrix multiplication:
- `mem` = 3 × n² × 4 bytes (three matrices)
- `lds` ≈ 2 × n³ × 4 bytes (each element is loaded ~n times)
- **Ratio** ≈ n / 3

For a 2048×2048 matrix multiplication:
- Ratio ≈ 2048 / 3 ≈ **683×** (theoretical maximum)

This massive ratio reflects the inherent **data reuse** in matrix multiplication: each input element contributes to many output elements.

---

## 5. Defining the Two Bandwidth Metrics

Now we can precisely define what the two numbers in `N1|N2 GB/s` mean.

### The Definitions

tinygrad computes two estimates (see `tinygrad/renderer/__init__.py`):

1. **`lds` (loads/stores)**: The total bytes accessed by all load and store instructions in the kernel. If the same memory location is accessed 1000 times, it counts 1000 times.

2. **`mem` (memory)**: The total bytes of unique memory accessed. Each buffer is counted only once, regardless of how many times its elements are accessed.

The bandwidth metrics are then:

```python
membw = mem / execution_time    # First number (N1)
ldsbw = lds / execution_time    # Second number (N2)
```

### In Plain English

- **`membw` (first number)**: "If each byte of data were transferred from main memory exactly once, what bandwidth would be required to achieve this execution time?"

- **`ldsbw` (second number)**: "If we count every single load/store instruction the kernel executes, what bandwidth would be required?"

---

## 6. Why the Numbers Differ: Data Reuse and Caching

### When N1 = N2: No Data Reuse

For operations like vector addition:
- Each memory location is accessed exactly once.
- `lds = mem`, so `ldsbw = membw`.
- Example: `158|158 GB/s`

### When N2 > N1: Data Reuse

For operations like matrix multiplication:
- The same memory locations are accessed multiple times.
- `lds > mem`, so `ldsbw > membw`.
- Example: `10|428 GB/s`

### The Role of Caching

Here's a crucial insight: your hardware has a **physical bandwidth limit**. For example, an Apple M1 Pro has a maximum memory bandwidth of 205 GB/s.

When you see `ldsbw = 428 GB/s`, which exceeds 205 GB/s, how is this possible?

**The answer is caching.**

The `ldsbw` metric counts *all* load instructions, but not all of them actually access main memory:

```
First access to a00:   Global Memory → Cache → Register  (slow, uses main memory bandwidth)
Second access to a00:  Cache → Register                   (fast, no main memory access)
```

When `ldsbw` exceeds your hardware's bandwidth limit, it proves that caching is working effectively. Many loads are satisfied from cache (fast) rather than global memory (slow).

### The Ratio as a Reuse Indicator

The ratio `ldsbw / membw` indicates the **data reuse factor**:

| Operation | Typical Ratio | Meaning |
|-----------|---------------|---------|
| Vector addition | 1× | No reuse, each element accessed once |
| Matrix multiplication | 10-100× | High reuse, elements accessed many times |
| Convolution | 10-1000× | Very high reuse depending on kernel size |

---

## 7. Interpreting Real Benchmark Results

Let's analyze actual benchmark output from an Apple M1 Pro GPU (205 GB/s memory bandwidth, ~5.3 TFLOPS compute).

### Vector Addition

```
*** METAL  E_78125_32_4n1  tm 728.35us (13 GFLOPS 158|158 GB/s)
```

**Analysis**:
```
mem = 120 MB (three arrays of 10M floats each)
lds = 120 MB (each element accessed once)
time = 0.728 ms

membw = 120 MB / 0.728 ms = 165 GB/s
ldsbw = 120 MB / 0.728 ms = 165 GB/s
```

**Interpretation**:
- Both numbers are equal → no data reuse.
- Achieving 158-165 GB/s out of 205 GB/s → **77-80% memory efficiency**.
- This operation is **memory-bound**: performance is limited by how fast we can move data.

### Matrix Multiplication

```
*** METAL  r_64_16_32_4_2_4_4_256  tm 5079.34us (3402 GFLOPS 10|428 GB/s)
```

**Analysis**:
```
mem = 50.3 MB (three 2048×2048 matrices)
lds ≈ 2.17 GB (accounting for all loads in nested computation)
time = 5.079 ms

membw = 50.3 MB / 5.079 ms = 10 GB/s
ldsbw = 2.17 GB / 5.079 ms = 428 GB/s
```

**Interpretation**:
- `ldsbw / membw` ≈ 43× → significant data reuse.
- `ldsbw = 428 GB/s` exceeds the 205 GB/s hardware limit → caching is effective.
- Achieving 3402 GFLOPS out of ~5300 GFLOPS → **64% compute efficiency**.
- This operation is **compute-bound**: performance is limited by arithmetic throughput, not memory.

### Summary Table

| Metric | Vector Addition | Matrix Multiplication |
|--------|-----------------|----------------------|
| `mem` | 120 MB | 50.3 MB |
| `lds` | 120 MB | ~2.17 GB |
| `membw` | 158 GB/s | 10 GB/s |
| `ldsbw` | 158 GB/s | 428 GB/s |
| Reuse ratio | 1× | ~43× |
| GFLOPS | 13 | 3402 |
| Bottleneck | Memory bandwidth | Compute throughput |

---

## 8. Summary

### The Two Numbers Explained

In tinygrad's debug output `N1|N2 GB/s`:

- **N1 (`membw`)**: Bandwidth calculated from unique memory accessed. Answers: "What's the minimum data transfer needed?"

- **N2 (`ldsbw`)**: Bandwidth calculated from all load/store instructions. Answers: "How much data does the kernel logically request?"

### Key Takeaways

1. **When N1 = N2**: The operation has no data reuse. Each memory location is accessed exactly once. These operations are typically memory-bound.

2. **When N2 > N1**: The operation reuses data. The same memory locations are accessed multiple times. The ratio N2/N1 indicates the reuse factor.

3. **When N2 exceeds hardware bandwidth**: Caching is working. Many loads are served from cache rather than main memory.

4. **For memory-bound operations** (like vector addition): Look at N1 relative to your hardware's bandwidth limit to assess efficiency.

5. **For compute-bound operations** (like matrix multiplication): Look at GFLOPS relative to your hardware's compute limit to assess efficiency.

### Where This Is Computed

The metrics are computed in `tinygrad/renderer/__init__.py` in the `Estimates` class:
- `lds`: Counts every LOAD/STORE instruction, multiplied by loop iterations and thread count.
- `mem`: Counts each buffer once based on its total size.

The bandwidth calculation happens in `tinygrad/engine/realize.py`:
```python
membw = mem_est / execution_time
ldsbw = lds_est / execution_time
```

---

## Further Reading

- Run benchmarks with `DEBUG=2` to see these metrics for your own kernels.
- Use `VIZ=1` to visualize the UOp graphs and understand kernel structure.
- Examine `tinygrad/renderer/__init__.py` to see the exact counting logic.
