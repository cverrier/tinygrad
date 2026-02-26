# Step 2: UOp — The Universal IR

## Table of Contents
1. [The Big Idea: One Node Type](#1-the-big-idea-one-node-type)
2. [The 4 Fields of a UOp](#2-the-4-fields-of-a-uop)
3. [The Ops Enum: ~100 Operations in 6 Sections](#3-the-ops-enum-100-operations-in-6-sections)
4. [UOp Deduplication: Identity via Caching](#4-uop-deduplication-identity-via-caching)
5. [Key Methods](#5-key-methods)
6. [GroupOp: Categorizing Operations](#6-groupop-categorizing-operations)
7. [PatternMatcher and UPat (Preview)](#7-patternmatcher-and-upat-preview)
8. [The Two Worlds: Tensor Graph vs Kernel](#8-the-two-worlds-tensor-graph-vs-kernel)
9. [Exercises and Outputs](#9-exercises-and-outputs)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. The Big Idea: One Node Type

Most compilers have many different IR types: a high-level AST, a mid-level loop IR, a low-level instruction IR. tinygrad takes a radically simpler approach: **there is only one node type — `UOp`** (Universal Operation).

Every single thing in tinygrad's compilation pipeline — from user-facing `Tensor.add` all the way down to GPU thread assignments — is a `UOp` node connected to other `UOp` nodes by edges, forming a DAG (Directed Acyclic Graph).

```
Traditional compiler:          tinygrad:

  High-level AST               UOp graph (tensor level)
       ↓                            ↓
  Mid-level loop IR            UOp graph (kernel level)
       ↓                            ↓
  Low-level instruction IR     UOp graph (device level)
       ↓                            ↓
  Machine code                 GPU source code
```

Same node type at every stage. The difference is **which `Ops`** appear at each stage — high-level ops like `RESHAPE` and `REDUCE_AXIS` exist only in the tensor graph, while low-level ops like `RANGE`, `LOAD`, and `STORE` exist only in compiled kernels.

---

## 2. The 4 Fields of a UOp

A UOp has exactly **4 fields** (plus an optional 5th `tag`):

```
┌──────────────────────────────────────────────────────┐
│                       UOp                            │
├──────────────┬───────────────────────────────────────┤
│ op: Ops      │ What operation (ADD, MUL, BUFFER...)  │
│ dtype: DType │ Data type (float32, int32, bool...)   │
│ src: tuple   │ Input UOps — the edges of the graph   │
│ arg: Any     │ Extra data (shape, constant value...) │
└──────────────┴───────────────────────────────────────┘
```

Defined at `tinygrad/uop/ops.py:121-127`:

```python
@dataclass(eq=False, slots=True)
class UOp(OpMixin, metaclass=UOpMetaClass):
  op:Ops
  dtype:DType = dtypes.void
  src:tuple[UOp, ...] = tuple()
  arg:Any = None
  tag:Any = None
```

What each field carries depends on the `op`:

| `op` | `dtype` | `src` | `arg` |
|------|---------|-------|-------|
| `CONST` | the constant's type (float32, int...) | `()` or `(DEVICE,)` | the constant value (1.0, 42, ...) |
| `ADD` | result type | `(left_operand, right_operand)` | `None` |
| `BUFFER` | element dtype | `(UNIQUE, DEVICE)` | buffer size (int) |
| `RESHAPE` | same as input dtype | `(input, shape_arg)` | `None` |
| `RANGE` | `dtypes.index` | `(end_value,)` | `(axis_id, axis_type)` |
| `LOAD` | data type being loaded | `(indexed_ptr,)` | `None` |

The key insight: **`src` defines the graph structure** (edges pointing to dependencies), while **`arg` carries metadata** that doesn't reference other UOps.

---

## 3. The Ops Enum: ~100 Operations in 6 Sections

The `Ops` enum at `tinygrad/uop/__init__.py:13` defines every operation type. The order of entries controls toposort ordering.

### Section 1: Defines / Special (kernel boundary ops)
```python
DEFINE_VAR = auto(); BIND = auto()   # symbolic variables (ptrs to outside the kernel)
SPECIAL = auto()                      # GPU dimensions (threadIdx, blockIdx)
DEFINE_LOCAL = auto(); DEFINE_REG     # allocate local/register memory
```
These are the **interface** between a kernel and the outside world. `DEFINE_VAR` represents a symbolic variable whose value is only known at runtime. `SPECIAL` maps to GPU thread/block indices. `DEFINE_LOCAL` allocates shared memory within a workgroup.

### Section 2: Non-op UOps (structural / meta)
```python
NOOP = auto()                  # no operation
PARAM = auto(); CALL = auto()  # function parameters and calls
PROGRAM = auto(); LINEAR = auto(); SOURCE = auto(); BINARY = auto()  # renderer pipeline
SINK = auto(); AFTER = auto(); GROUP = auto()  # ordering/grouping
GEP = auto(); VECTORIZE = auto()               # vector creation/extraction
```
These don't represent computation — they're structural. `SINK` collects multiple outputs into one node (like a "gather" for the graph). `GEP` (Get Element Pointer, borrowed from LLVM terminology) extracts one element from a vector. `VECTORIZE` packs scalars into a vector.

### Section 3: Load / Store (memory access)
```python
INDEX = auto()                 # pointer arithmetic (like &buf[i])
LOAD = auto(); STORE = auto()  # read from / write to memory
```
Just 3 ops for all memory access. `INDEX` takes a buffer pointer and an offset, producing an indexed pointer. `LOAD` reads from that pointer. `STORE` writes to it.

### Section 4: Math (the actual computation)
```python
# Unary (7 ops)
CAST; BITCAST; EXP2; LOG2; SIN; SQRT; RECIPROCAL; NEG; TRUNC

# Binary (17 ops)
ADD; MUL; SHL; SHR; IDIV; MAX; MOD;
CMPLT; CMPNE; CMPEQ;
XOR; OR; AND;
THREEFRY; SUB; FDIV; POW

# Ternary (2 ops)
WHERE; MULACC
```
These are the actual math operations that end up in GPU kernels. Note how minimal the **core** set is:
- No `SUB` natively in kernels — it decomposes to `ADD(a, NEG(b))`
- No `DIV` natively — decomposes to `MUL(a, RECIPROCAL(b))`
- `SUB`, `FDIV`, `POW` exist in the Ops enum but get decomposed before code generation

`THREEFRY` is for random number generation. `MULACC` (multiply-accumulate) is a fused `a*b+c` that maps to hardware FMA instructions.

### Section 5: Control Flow / Consts
```python
BARRIER; RANGE; IF; END; ENDIF   # loops and conditionals
CONST; VCONST                     # scalar and vector constants
```
`RANGE` is the universal loop construct — both GPU grid dimensions and regular for-loops become `RANGE` nodes. `END` marks the end of a `RANGE` scope. `BARRIER` is a GPU synchronization primitive (all threads in a workgroup wait). `VCONST` is a vectorized constant (e.g., `float4(1.0, 2.0, 3.0, 4.0)`).

### Section 6: Tensor Graph Ops (high-level, never in kernels)
```python
# Identity / device
UNIQUE; DEVICE; ASSIGN

# Buffer management
BUFFERIZE; COPY; BUFFER; BUFFER_VIEW; MSELECT; MSTACK; ENCDEC

# The 6 movement ops (from Step 1)
RESHAPE; PERMUTE; EXPAND; PAD; SHRINK; FLIP
MULTI  # for multi-device sharding

# Reductions
REDUCE_AXIS; REDUCE; ALLREDUCE
```
These **only exist in the tensor-level graph** — before scheduling. They get lowered into Section 3-5 ops during compilation:

```
Tensor graph world         Kernel world
─────────────────         ────────────────
RESHAPE, EXPAND     →     RANGE + index math
REDUCE_AXIS         →     RANGE + LOAD + ADD + STORE
BUFFER + COPY       →     LOAD + STORE
```

---

## 4. UOp Deduplication: Identity via Caching

This is one of tinygrad's cleverest design decisions. The `UOpMetaClass` at `tinygrad/uop/ops.py:84` intercepts every UOp construction:

```python
class UOpMetaClass(type):
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op, dtype, src, arg, tag, ...):
    # Check if identical UOp already exists
    if (wret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg, tag))) is not None \
       and (ret:=wret()) is not None:
      return ret    # ← Return the existing object!
    # Otherwise create new and cache it
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(*key))
    return created
```

**Every time you construct a UOp, Python checks if an identical one already exists.** If so, it returns the existing object instead of creating a new one:

```python
a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=1.0)
assert a is b  # True! Same object in memory
```

### Why this matters

1. **Memory efficiency** — shared subexpressions aren't duplicated in memory
2. **Graph identity** — you can check `a is b` (pointer comparison) instead of expensive deep comparison
3. **Automatic CSE (Common Subexpression Elimination)** — if two parts of your computation produce the same intermediate result, they automatically share the same UOp node

For example, `(x+1) * (x+1)` automatically shares the `x+1` subexpression:

```python
x = UOp(Ops.CONST, dtypes.float, arg=3.0)
one = UOp(Ops.CONST, dtypes.float, arg=1.0)
xp1_a = UOp(Ops.ADD, dtypes.float, (x, one))
xp1_b = UOp(Ops.ADD, dtypes.float, (x, one))
assert xp1_a is xp1_b  # Same object — automatic CSE!
```

The cache uses **weak references** (`weakref.ref`) so UOps that are no longer referenced anywhere get garbage collected. When a UOp is deleted, its `__del__` method removes it from the cache:

```python
def __del__(self):
    try: del UOpMetaClass.ucache[(self.op, self.dtype, self.src, self.arg, self.tag)]
    except AttributeError: pass
```

---

## 5. Key Methods

### `toposort()` — Walk the graph in dependency order

```python
def toposort(self, gate=None) -> dict[UOp, None]:
```

Returns all nodes reachable from `self`, ordered so that every node appears **after** all its dependencies. Uses an iterative DFS (not recursive, to avoid stack overflow on deep graphs).

The return type `dict[UOp, None]` is used as an **ordered set** — Python dicts preserve insertion order since 3.7, and using a dict gives O(1) membership testing.

```python
z = UOp(Ops.ADD, dtypes.float, (
    UOp(Ops.CONST, dtypes.float, arg=1.0),
    UOp(Ops.CONST, dtypes.float, arg=2.0)))

for node in z.toposort():
    print(node.op, node.arg)
# Ops.CONST 1.0     ← dependencies first
# Ops.CONST 2.0
# Ops.ADD   None    ← consumer last
```

Shared nodes are visited only once — in the `(x+1) * (x+1)` example, toposort produces 4 nodes not 5.

### `replace()` — Create a modified copy

```python
def replace(self, **kwargs) -> UOp:
```

Since UOps are effectively immutable (via dedup), you never mutate them. `replace` creates a new UOp with some fields changed. If the resulting fields are identical to the original, dedup returns `self`:

```python
a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = a.replace(arg=2.0)   # new UOp with arg changed
c = a.replace(arg=1.0)   # identical to a, so returns a
assert c is a             # True — dedup kicks in
```

### `simplify()` — Algebraic simplification

```python
def simplify(self):
    from tinygrad.uop.symbolic import symbolic
    return graph_rewrite(self, symbolic)
```

Runs the symbolic PatternMatcher (covered in Step 3) on this UOp graph. Applies algebraic identities like:
- `x + 0` → `x`
- `x * 1` → `x`
- `x * 0` → `0`
- `(x + c1) + c2` → `x + (c1+c2)` (constant folding)

### `substitute()` — Replace nodes in a graph

```python
def substitute(self, dvars:dict[UOp, UOp], ...):
```

Walks the graph and replaces every occurrence of a key UOp with its corresponding value. Useful for plugging in concrete values for symbolic variables.

### `UOp.const()` — Create constant values

```python
@staticmethod
def const(dtype, b, device=None, shape=None):
    ret = UOp(Ops.CONST, dtype, arg=dtypes.as_const(b, dtype), ...)
    return ret.reshape((1,)*len(shape)).expand(shape) if shape else ret
```

Creates a `CONST` UOp. If a `shape` is given, it broadcasts the constant to that shape using `RESHAPE → EXPAND` — this is exactly what happens when you write `Tensor([1,2,3]) + 1`: the scalar `1` gets wrapped in `CONST → RESHAPE(1,) → EXPAND(3,)`.

### `alu()` — Create arithmetic nodes

```python
def alu(self, op, *src):
    out_dtype = (self, *src)[-1].dtype
    if op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}: out_dtype = dtypes.bool
    return UOp(op, out_dtype, (self,)+src)
```

Convenience method for creating math operations. Automatically handles dtype: comparison ops always return `bool`, other ops propagate the input dtype.

---

## 6. GroupOp: Categorizing Operations

At `tinygrad/uop/__init__.py:106`, `GroupOp` creates useful sets of ops for use throughout the codebase:

```python
class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, ...}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = Unary | Binary | Ternary              # 26 ops: all arithmetic/logic

  Elementwise = ALU | {Ops.CAST, Ops.BITCAST}  # everything that fuses elementwise
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

  Commutative = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.CMPEQ, ...}  # a+b == b+a
  Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}             # (a+b)+c == a+(b+c)
  Idempotent = {Ops.OR, Ops.AND, Ops.MAX}                                 # f(x,x) == x
```

### Where each group is used

| Group | Used for |
|-------|----------|
| `ALU` | Scheduler: all ALU ops can fuse into one kernel |
| `Elementwise` | Shape tracking: elementwise ops preserve input shape |
| `Movement` | Identifying tensor-graph-only ops (never in kernels) |
| `Commutative` | PatternMatcher tries both orderings `(a,b)` and `(b,a)` |
| `Associative` | Enables reassociation optimizations |
| `Idempotent` | Enables `f(x,x) → x` simplification |
| `UnsafePad` | Ops where padding with zeros would give wrong results (e.g., `log2(0)`) |

The clean separation between `ALU` and `Movement` reflects the two-world design: ALU ops exist in both tensor graphs and compiled kernels, Movement ops only exist in tensor graphs.

---

## 7. PatternMatcher and UPat (Preview)

These are covered in depth in Step 3, but here's the core idea since they're defined alongside UOp.

**`UPat`** (at `tinygrad/uop/ops.py:954`) is a *pattern* that matches UOp nodes — think regex for graphs:

```python
UPat(Ops.ADD)                                    # matches any ADD node
UPat(Ops.ADD, src=(UPat.var("x"), UPat.var("y"))) # matches ADD, captures inputs as x, y
UPat.cvar("c")                                    # matches any CONST or VCONST
UPat.var("x")                                     # matches any UOp, captures as x
```

**`PatternMatcher`** (at `tinygrad/uop/ops.py:1085`) takes a list of `(pattern, replacement_function)` pairs:

```python
pm = PatternMatcher([
    # x + 0 → x
    (UPat(Ops.ADD, src=(UPat.var("x"), UPat.const(dtypes.int, 0))),
     lambda x: x),
])
```

**`graph_rewrite(sink, pm)`** (at `tinygrad/uop/ops.py:1360`) walks a UOp graph and repeatedly applies the PatternMatcher until no more rules match (fixed-point iteration). This single mechanism powers:

- **Gradient computation** (`tinygrad/gradient.py`)
- **Symbolic simplification** (`tinygrad/uop/symbolic.py`)
- **Scheduling** (`tinygrad/engine/schedule.py`)
- **Codegen lowering** (`tinygrad/codegen/__init__.py`)
- **Rendering** (`tinygrad/renderer/cstyle.py`)

Understanding PatternMatcher is the key to understanding all of tinygrad's internals.

---

## 8. The Two Worlds: Tensor Graph vs Kernel

The most important conceptual insight about UOps is that they serve **two different purposes** depending on where they are in the pipeline:

```
┌──────────────────────────────────────────────────────────┐
│                   TENSOR GRAPH WORLD                      │
│  (before scheduling — what you see with tensor.uop)       │
│                                                           │
│  Ops: BUFFER, COPY, DEVICE, UNIQUE,                       │
│       RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP,        │
│       ADD, MUL, SIN, ..., REDUCE_AXIS                     │
│                                                           │
│  Every UOp has a shape (tracked by the _shape property)    │
│  Purpose: represent the lazy computation graph             │
├───────────────────────────────────────────────────────────┤
│                                                           │
│           .realize() → scheduling → lowering               │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                   KERNEL WORLD                            │
│  (after scheduling — what becomes GPU code)                │
│                                                           │
│  Ops: RANGE, LOAD, STORE, INDEX,                          │
│       ADD, MUL, SIN, ...,                                 │
│       SPECIAL, DEFINE_LOCAL, BARRIER, END                  │
│                                                           │
│  UOps don't have shapes (they have loop indices)           │
│  Purpose: represent the compiled kernel                    │
└───────────────────────────────────────────────────────────┘
```

The math ops (ADD, MUL, SIN, etc.) exist in **both** worlds — they pass straight through from the tensor graph into compiled kernels. The movement ops (RESHAPE, EXPAND, etc.) exist only in the tensor graph and get converted into index arithmetic during lowering. The kernel ops (RANGE, LOAD, STORE, INDEX, SPECIAL) exist only in the kernel world.

### Concrete example of the transformation

A simple `(x + 1).sum()` in the tensor graph:

```
REDUCE_AXIS(op=ADD, axis=(0,))     ← "sum along axis 0"
    │
   ADD                              ← "add 1 to each element"
  /   \
COPY    EXPAND(3,)                  ← "broadcast scalar to shape (3,)"
  │       │
BUFFER  RESHAPE(1,)
  │       │
UNIQUE  CONST(1)
  │       │
DEVICE  DEVICE
```

Becomes in the kernel world:

```
STORE(result_ptr, accumulated_value)
    │
   END(range_loop)
    │
   ADD(accumulator, loaded_value + 1)     ← fused: add + reduce in one kernel
    │
   LOAD(input_ptr[range_idx])
    │
   RANGE(0..3)                            ← the loop that replaced REDUCE_AXIS
    │
   INDEX(input_buffer, range_idx)
```

Movement ops disappeared entirely — they became index math inside the `INDEX` computation.

---

## 9. Exercises and Outputs

### Exercise 1: Deduplication

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

a = UOp(Ops.CONST, dtypes.int, arg=5)
b = UOp(Ops.CONST, dtypes.int, arg=5)
print(f'a is b: {a is b}')   # True — same object!
print(f'id(a) == id(b): {id(a) == id(b)}')  # True

c = UOp(Ops.CONST, dtypes.int, arg=7)
print(f'a is c: {a is c}')   # False — different arg
```

**Output:**
```
a is b: True
id(a) == id(b): True
a is c: False
```

Constructing `UOp(Ops.CONST, dtypes.int, arg=5)` twice returns the **exact same Python object**. The metaclass checks a global cache keyed by `(op, dtype, src, arg, tag)`. Change any field and you get a different object.

### Exercise 2: Build a UOp graph by hand

```python
x = UOp(Ops.CONST, dtypes.float, arg=1.0)
y = UOp(Ops.CONST, dtypes.float, arg=2.0)
z = UOp(Ops.ADD, dtypes.float, (x, y))

print(f'x: op={x.op}, dtype={x.dtype}, arg={x.arg}, src={x.src}')
print(f'y: op={y.op}, dtype={y.dtype}, arg={y.arg}, src={y.src}')
print(f'z: op={z.op}, dtype={z.dtype}, arg={z.arg}')
print(f'z.src[0] is x: {z.src[0] is x}')
print(f'z.src[1] is y: {z.src[1] is y}')
print(z)
```

**Output:**
```
x: op=Ops.CONST, dtype=dtypes.float, arg=1.0, src=()
y: op=Ops.CONST, dtype=dtypes.float, arg=2.0, src=()
z: op=Ops.ADD, dtype=dtypes.float, arg=None
z.src[0] is x: True
z.src[1] is y: True

UOp(Ops.ADD, dtypes.float, arg=None, src=(
  UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),
  UOp(Ops.CONST, dtypes.float, arg=2.0, src=()),))
```

We manually built `1.0 + 2.0` as three UOps. The `ADD` node has `src=(x, y)` pointing to its two inputs. The `arg` for ADD is `None` — math ops don't need extra metadata, their behavior is fully determined by `op`.

### Exercise 3: toposort

```python
topo = z.toposort()
for i, node in enumerate(topo):
    print(f'  [{i}] op={node.op}, arg={node.arg}')
```

**Output:**
```
  [0] op=Ops.CONST, arg=1.0
  [1] op=Ops.CONST, arg=2.0
  [2] op=Ops.ADD, arg=None
```

Dependencies first (`CONST` nodes), then consumers (`ADD`). Every node appears **after** all nodes it depends on. This ordering is critical for code generation — you must compute inputs before using them.

### Exercise 4: GroupOp exploration

```python
from tinygrad.uop import GroupOp

print(f'ALU ({len(GroupOp.ALU)} ops)')
print(f'Movement ({len(GroupOp.Movement)} ops)')

print(f'Ops.ADD in ALU: {Ops.ADD in GroupOp.ALU}')         # True
print(f'Ops.RESHAPE in ALU: {Ops.RESHAPE in GroupOp.ALU}') # False
print(f'Ops.RESHAPE in Movement: {Ops.RESHAPE in GroupOp.Movement}') # True
print(f'Ops.ADD in Movement: {Ops.ADD in GroupOp.Movement}')         # False
```

**Output:**
```
ALU (26 ops)
Movement (6 ops)
Ops.ADD in ALU: True
Ops.RESHAPE in ALU: False
Ops.RESHAPE in Movement: True
Ops.ADD in Movement: False
```

Clean separation: **ALU** (26 math ops) is what goes inside GPU kernels. **Movement** (6 ops) only exists in the tensor graph before scheduling. The two sets don't overlap.

### Exercise 5: Automatic CSE with dedup

```python
x = UOp(Ops.CONST, dtypes.float, arg=3.0)
one = UOp(Ops.CONST, dtypes.float, arg=1.0)
xp1_a = UOp(Ops.ADD, dtypes.float, (x, one))
xp1_b = UOp(Ops.ADD, dtypes.float, (x, one))
print(f'xp1_a is xp1_b: {xp1_a is xp1_b}')

result = UOp(Ops.MUL, dtypes.float, (xp1_a, xp1_b))
print(f'result.src[0] is result.src[1]: {result.src[0] is result.src[1]}')

topo = result.toposort()
print(f'Toposort has {len(topo)} nodes (not 5, because x+1 is shared)')
print(result)
```

**Output:**
```
xp1_a is xp1_b: True
result.src[0] is result.src[1]: True
Toposort has 4 nodes (not 5, because x+1 is shared)

UOp(Ops.MUL, dtypes.float, arg=None, src=(
  x0:=UOp(Ops.ADD, dtypes.float, arg=None, src=(
    UOp(Ops.CONST, dtypes.float, arg=3.0, src=()),
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),)),
  x0,))
```

Building `(x+1) * (x+1)` — the `x+1` UOp is created twice but dedup gives us the same object. The toposort has **4 nodes, not 5**, because the ADD is shared. In the repr, `x0:=` marks a node defined once and referenced multiple times. This is **free Common Subexpression Elimination** — no optimization pass needed.

### Exercise 6: Connecting to the Tensor world

```python
from tinygrad import Tensor
t = Tensor([1, 2, 3])
u = t + 10

print(f'Tensor uop type: {u.uop.op}')
print(f'Graph size: {len(u.uop.toposort())} nodes')

for i, node in enumerate(u.uop.toposort()):
    shape_str = ''
    try: shape_str = f', shape={node.shape}'
    except: pass
    print(f'  [{i}] {node.op}, dtype={node.dtype}, arg={node.arg}{shape_str}')
```

**Output:**
```
Tensor uop type: Ops.ADD
Graph size: 11 nodes

  [0] Ops.UNIQUE, dtype=dtypes.void, arg=0
  [1] Ops.DEVICE, dtype=dtypes.void, arg=PYTHON
  [2] Ops.BUFFER, dtype=dtypes.int, arg=3, shape=(3,)
  [3] Ops.DEVICE, dtype=dtypes.void, arg=METAL
  [4] Ops.COPY, dtype=dtypes.int, arg=None, shape=(3,)
  [5] Ops.CONST, dtype=dtypes.int, arg=10, shape=()
  [6] Ops.CONST, dtype=dtypes.index, arg=1, shape=()
  [7] Ops.RESHAPE, dtype=dtypes.int, arg=None, shape=(1,)
  [8] Ops.CONST, dtype=dtypes.index, arg=3, shape=()
  [9] Ops.EXPAND, dtype=dtypes.int, arg=None, shape=(3,)
  [10] Ops.ADD, dtype=dtypes.int, arg=None, shape=(3,)
```

You can trace the full story through the node types:

- **[0-2]** `UNIQUE → DEVICE('PYTHON') → BUFFER(size=3)` — the raw data `[1,2,3]` on CPU
- **[3-4]** `DEVICE('METAL') → COPY` — intent to copy data to GPU
- **[5-9]** `CONST(10) → RESHAPE(1,) → EXPAND(3,)` — the scalar 10 broadcast to shape (3,)
- **[10]** `ADD` — the final operation, shape (3,)

Every node tracks shape correctly: `CONST` has shape `()`, after `RESHAPE` it's `(1,)`, after `EXPAND` it's `(3,)`, and `ADD` preserves the `(3,)` shape of its inputs.

---

## 10. Key Takeaways

### UOps are deduplicated (memory efficiency, graph identity)
Identical UOps are the **same object in memory**. This is achieved by a global weak-reference cache in the metaclass. Benefits: memory efficiency, O(1) identity checks via `is`, and automatic Common Subexpression Elimination.

### The 4 fields of a UOp
- **`op`** (Ops enum) — what operation this node represents
- **`dtype`** (DType) — the data type of this node's output
- **`src`** (tuple of UOps) — the input edges, forming the DAG structure
- **`arg`** (Any) — extra metadata: constant values, shapes, axis info, etc.

### Tensor-graph UOps vs kernel UOps
- **Tensor graph**: BUFFER, COPY, RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP, REDUCE_AXIS + all math ops. Every node has a shape.
- **Kernel**: RANGE, LOAD, STORE, INDEX, SPECIAL, DEFINE_LOCAL, BARRIER, END + all math ops. Nodes have loop indices instead of shapes.
- **Shared**: math ops (ADD, MUL, SIN, etc.) exist in both worlds and pass through unchanged.

### PatternMatcher does graph rewriting via pattern rules
`PatternMatcher` takes `(UPat_pattern, replacement_function)` pairs and applies them to a UOp graph until convergence. This single mechanism powers gradient computation, algebraic simplification, scheduling, lowering, and rendering. Covered in depth in Step 3.

### GroupOp organizes Ops into useful categories
`GroupOp.ALU` (26 math ops), `GroupOp.Movement` (6 shape ops), `GroupOp.Commutative`, `GroupOp.Associative`, `GroupOp.Idempotent` — used throughout the codebase for pattern matching, scheduling decisions, and optimization rules.

---

## Source Files Referenced

| File | Key Contents |
|------|-------------|
| `tinygrad/uop/__init__.py:13-105` | `Ops` enum — all ~100 operations in 6 sections |
| `tinygrad/uop/__init__.py:106-138` | `GroupOp` — op categorization (ALU, Movement, Commutative, etc.) |
| `tinygrad/uop/ops.py:84-100` | `UOpMetaClass` — deduplication cache with weak references |
| `tinygrad/uop/ops.py:121-127` | `UOp` class definition — the 4 fields |
| `tinygrad/uop/ops.py:136-141` | `UOp.replace()` — create modified copies |
| `tinygrad/uop/ops.py:166-177` | `UOp.toposort()` — iterative DFS topological sort |
| `tinygrad/uop/ops.py:359-364` | `UOp.simplify()` — algebraic simplification via graph_rewrite |
| `tinygrad/uop/ops.py:376-381` | `UOp.substitute()` — node replacement in a graph |
| `tinygrad/uop/ops.py:445-448` | `UOp.alu()` — create arithmetic nodes |
| `tinygrad/uop/ops.py:449-458` | `UOp.const()` — create constants with optional broadcasting |
| `tinygrad/uop/ops.py:954-1052` | `UPat` class — pattern matching for UOps |
| `tinygrad/uop/ops.py:1085-1109` | `PatternMatcher` class — graph rewriting engine |
| `tinygrad/uop/ops.py:1360-1362` | `graph_rewrite()` — the main rewrite driver |
