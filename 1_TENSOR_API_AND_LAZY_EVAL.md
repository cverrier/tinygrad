# Step 1: The Tensor API and Lazy Evaluation

## Table of Contents
1. [Core Concept: Lazy Evaluation](#1-core-concept-lazy-evaluation)
2. [The Tensor Class](#2-the-tensor-class)
3. [UOp: The Building Block](#3-uop-the-building-block)
4. [How Operations Build Graphs](#4-how-operations-build-graphs)
5. [The 6 Movement Operations](#5-the-6-movement-operations)
6. [Realize: When Computation Happens](#6-realize-when-computation-happens)
7. [Exercises and Outputs](#7-exercises-and-outputs)
8. [Key Takeaways](#8-key-takeaways)

---

## 1. Core Concept: Lazy Evaluation

Most frameworks (NumPy, eager-mode PyTorch) execute operations immediately:

```
# NumPy: each line executes right away
x = np.array([1, 2, 3])    # allocates memory, stores data
y = x + 1                   # allocates new array, computes [2, 3, 4]
z = y * 2                   # allocates new array, computes [4, 6, 8]
```

tinygrad does the opposite. **Nothing executes until you explicitly ask for a result.**

```
# tinygrad: no computation happens here
x = Tensor([1, 2, 3])      # builds a UOp graph node (no kernel runs)
y = x + 1                   # adds an ADD node to the graph
z = y * 2                   # adds a MUL node on top

z.realize()                 # NOW everything runs: schedule -> compile -> execute
```

**Why?** Because lazy evaluation lets tinygrad see the *entire* computation before
executing anything. This enables:
- **Kernel fusion**: `x + 1` and `* 2` get fused into a single GPU kernel
- **Dead code elimination**: operations whose results are never used get dropped
- **Optimized scheduling**: tinygrad can reorder and batch operations

---

## 2. The Tensor Class

The `Tensor` class lives in `tinygrad/tensor.py` and is remarkably minimal:

```python
class Tensor(ElementwiseMixin, MovementMixin, ReduceMixin, CreationMixin, ...):
    __slots__ = "uop", "requires_grad", "grad"
```

Only **3 fields**:

```
+---------------------------------------------+
|                  Tensor                      |
+---------------------------------------------+
| uop: UOp           <- the computation graph  |
| requires_grad: bool <- track gradients?      |
| grad: Tensor|None   <- gradient (after bwd)  |
+---------------------------------------------+
```

A Tensor is just a **thin wrapper** around a `UOp` computation graph. All the
intelligence lives in the UOp graph, not in the Tensor object itself.

### Tensor Construction (`__init__`)

When you write `Tensor([1, 2, 3])`, the `__init__` method:

1. Detects the input is a Python list
2. Converts it to bytes via `_frompy()` (raw memory representation)
3. Creates a `UOp.new_buffer(...)` node representing data on a device
4. Creates a `UOp(Ops.COPY, ...)` node to move data to the target device
5. Stores the final UOp node in `self.uop`

Even construction is lazy -- `COPY` just records the *intent* to copy.

---

## 3. UOp: The Building Block

**UOp** (Universal Operation) is the single IR node type in tinygrad. Every
operation -- math, movement, buffer, device -- is a UOp node.

A UOp has exactly **4 fields**:

```
+-------------------------------------------+
|                   UOp                      |
+-------------------------------------------+
| op:    Ops enum    (ADD, MUL, RESHAPE...)  |
| dtype: DType       (float32, int32...)     |
| src:   tuple[UOp]  (input nodes)          |
| arg:   Any         (extra data: shape...) |
+-------------------------------------------+
```

UOps form a **DAG** (Directed Acyclic Graph). Each node points to its inputs
via `src`, building a tree-like computation graph:

```
        MUL
       /   \
     ADD    CONST(2)
    /   \
  DATA   CONST(1)
```

### UOp Deduplication

Identical UOps are **shared in memory**. If two operations reference
`DEVICE('METAL')`, there's only one node in memory, referenced by both.
In printed output, you'll see `x5:=` meaning "this node is defined here
and referenced elsewhere by id x5."

---

## 4. How Operations Build Graphs

Here's the exact call chain when you write `x + 1`:

```
x + 1
  |
  v
Tensor.__add__(1)                     # Python dunder method
  |
  v
Tensor.add(1, reverse=False)          # elementwise.py
  |
  v
Tensor._binop(Ops.ADD, 1, reverse)    # wraps 1 into a Tensor via ufix()
  |                                    # ufix: scalar -> Tensor(CONST) + broadcast
  v
Tensor._apply_uop(UOp.alu, other, op=Ops.ADD)   # tensor.py
  |
  v
UOp.alu(Ops.ADD, other_uop)           # ops.py: creates the ADD node
  |
  v
UOp(Ops.ADD, dtype, (self_uop, other_uop))   # the actual UOp constructor
  |
  v
Returns new Tensor(uop=<the new ADD UOp>)
```

### Broadcasting Constants

When you do `x + 1`, the scalar `1` must match `x`'s shape. tinygrad handles
this by wrapping the constant in movement ops:

```
CONST(1)          # scalar value, shape ()
    |
    v
RESHAPE(1,)       # reshape to rank-1: shape (1,)
    |
    v
EXPAND(3,)        # broadcast to match x: shape (3,)
```

This is the same broadcasting semantics as NumPy, but represented as
explicit graph nodes.

---

## 5. The 6 Movement Operations

Movement ops are **virtual** -- they don't move data. They add metadata nodes
to the graph that the scheduler later converts into index math inside kernels.

```
+----------+-------------------------------+------------------------+
| Op       | What it does                  | Example                |
+----------+-------------------------------+------------------------+
| RESHAPE  | Change shape, same data       | (6,) -> (2,3)          |
| PERMUTE  | Reorder axes (transpose)      | (2,3) -> (3,2)         |
| EXPAND   | Broadcast dim (size 1 -> N)   | (1,3) -> (4,3)         |
| PAD      | Add zeros around edges        | (3,) -> (5,) with pad  |
| SHRINK   | Slice/crop a region           | (10,) -> (5,)          |
| FLIP     | Reverse along an axis         | [1,2,3] -> [3,2,1]    |
+----------+-------------------------------+------------------------+
```

These 6 ops are **complete** -- any data layout transformation can be
expressed as a composition of these operations. They are the "movement
algebra" of tinygrad.

### How Movement Ops Compose

```python
a = Tensor([1, 2, 3])       # shape (3,)
b = a.reshape(3, 1)          # shape (3, 1) -- no data moves
c = b.expand(3, 4)           # shape (3, 4) -- no data moves, just broadcasting

# The UOp graph:
#
#   EXPAND(3,4)      <-- virtual: repeat columns 4x
#       |
#   RESHAPE(3,1)     <-- virtual: add a dimension
#       |
#   COPY             <-- the actual data [1, 2, 3]
#       |
#   BUFFER           <-- raw memory on device
```

Key insight: `expand` doesn't allocate a 3x4 array. It records "when you
read column j, just read column 0." The actual index remapping happens
during kernel compilation.

---

## 6. Realize: When Computation Happens

`.realize()` is the trigger that collapses the lazy graph into actual
computed results.

```python
def realize(self, *lst, do_update_stats=True):
    if len(to_realize := [x for x in (self,)+lst if not x.uop.has_buffer_identity()]):
        run_schedule(*Tensor.schedule_with_vars(*to_realize), do_update_stats=do_update_stats)
    return self
```

The pipeline:

```
  Lazy UOp Graph                After realize()
  ==============                ================

      MUL                          BUFFER
     /   \             -->        (contains [4, 6, 8])
   ADD    CONST(2)
  /   \
DATA   CONST(1)

15 nodes in graph          -->   3 nodes (BUFFER + metadata)
```

### The 3-step process:

```
+------------------+     +------------------+     +------------------+
|   1. SCHEDULE    | --> |   2. COMPILE     | --> |   3. EXECUTE     |
+------------------+     +------------------+     +------------------+
| Walk the UOp     |     | Generate kernel  |     | Run compiled     |
| graph, fuse ops  |     | code (Metal/     |     | kernel on device |
| into kernels,    |     | CUDA/OpenCL),    |     | (GPU/CPU), write |
| determine what   |     | compile to       |     | results into     |
| to compute       |     | binary           |     | buffer           |
+------------------+     +------------------+     +------------------+

After execution, tensor.uop is REPLACED with a simple BUFFER reference.
The entire computation graph is garbage collected.
```

---

## 7. Exercises and Outputs

### Exercise 1: Create a Tensor

```python
from tinygrad import Tensor
x = Tensor([1, 2, 3])
print(x.uop)
```

**Output structure:**
```
UOp(Ops.COPY, dtypes.int, arg=False, src=(
  UOp(Ops.BUFFER, dtypes.int, arg=3, src=(
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG'),)),)),    # <-- source device
  UOp(Ops.DEVICE, dtypes.void, arg='METAL'),))             # <-- target device
```

**Explanation:**
- `DEVICE('METAL')` -- target device
- `UNIQUE` -- gives each buffer a unique identity
- `BUFFER` -- represents raw allocated memory (3 ints)
- `COPY` -- intent to copy data to the target device

### Exercise 2: Add Without Realize

```python
y = x + 1
print(y.uop)
```

**Graph structure:**
```
          ADD
         /   \
      COPY    EXPAND(3,)
       |         |
    BUFFER    RESHAPE(1,)
       |         |
    UNIQUE    CONST(1, dtypes.int)
       |         |
    DEVICE    DEVICE('METAL')
   ('METAL')
```

The graph grew with an `ADD` node. The constant `1` was wrapped in
`CONST -> RESHAPE -> EXPAND` to match the shape `(3,)`. No math happened.

### Exercise 3: Chain Operations

```python
z = (x + 1) * 2
print(z.uop)
```

**Graph structure:**
```
              MUL
             /   \
          ADD     EXPAND(3,)
         /   \        |
      COPY   EXPAND  RESHAPE(1,)
       |       |        |
    BUFFER  RESHAPE  CONST(2)
       |       |        |
    UNIQUE  CONST(1)  DEVICE
       |       |
    DEVICE  DEVICE
```

Operations stack: MUL wraps ADD, which wraps COPY. The graph records the
full computation `(x + 1) * 2` without executing it.

### Exercise 4: Before and After Realize

```python
# BEFORE realize:
print(len(z.uop.toposort()))  # ~15 nodes
print(z.uop.op)               # Ops.MUL

z.realize()

# AFTER realize:
print(len(z.uop.toposort()))  # 3 nodes
print(z.uop.op)               # Ops.BUFFER
print(z.numpy())              # [4, 6, 8]
```

**Before:** 15 UOp nodes forming the full computation tree.
**After:** 3 nodes -- just `BUFFER -> UNIQUE -> DEVICE`. The entire lazy
graph was consumed, a fused kernel was compiled and executed, and the
result `[4, 6, 8]` lives in the buffer.

### Exercise 5: Movement Ops

```python
a = Tensor([1, 2, 3])
b = a.reshape(3, 1)
c = b.expand(3, 4)
print(c.uop)
```

**Graph structure:**
```
  EXPAND(3, 4)
      |
  RESHAPE(3, 1)
      |
    COPY
      |
   BUFFER
      |
   UNIQUE
      |
   DEVICE
```

- `RESHAPE(3,1)` changes the logical shape from `(3,)` to `(3,1)` -- no data moves
- `EXPAND(3,4)` broadcasts from `(3,1)` to `(3,4)` -- no data moves
- If realized, the result would be:
  ```
  [[1, 1, 1, 1],
   [2, 2, 2, 2],
   [3, 3, 3, 3]]
  ```

---

## 8. Key Takeaways

### Why `Tensor([1,2,3]) + 1` doesn't execute anything
Operations build UOp graph nodes. The `+` operator creates an `ADD` UOp that
points to its inputs. No kernel is compiled, no GPU work is dispatched. The
graph just grows.

### What a UOp node contains
Four fields: `op` (what operation), `dtype` (data type), `src` (input nodes),
`arg` (extra metadata like shape or constant value).

### How `x.uop` changes before vs after `.realize()`
Before: a deep tree of operation nodes (ADD, MUL, RESHAPE, etc.).
After: a flat 3-node chain (`BUFFER -> UNIQUE -> DEVICE`) containing the
computed result.

### The 6 movement ops
`RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`. They are virtual
(no data moves), composable, and complete (any layout transform can be
expressed with them).

---

## Source Files Referenced

| File | Key Contents |
|------|-------------|
| `tinygrad/tensor.py:103-114` | `class Tensor` with 3 slots |
| `tinygrad/tensor.py:117-175` | `__init__` -- Tensor construction |
| `tinygrad/tensor.py:180-191` | `_apply_uop` -- bridge to UOp graph |
| `tinygrad/tensor.py:259-300` | `schedule`, `realize` |
| `tinygrad/mixin/elementwise.py:21-71` | `_binop`, `add`, `mul` |
| `tinygrad/mixin/movement.py:18-170` | `_mop`, `reshape`, `permute`, `shrink`, `flip` |
| `tinygrad/uop/ops.py:445-448` | `UOp.alu()` |
| `tinygrad/uop/ops.py:587-602` | `UOp._mop()` |
