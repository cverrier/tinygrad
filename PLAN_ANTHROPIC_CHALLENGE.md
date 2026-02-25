# Anthropic Performance Challenge - Comprehensive Learning Plan

## Goal
Optimize the tinygrad VLIWRenderer to achieve < 1,340 cycles (beat George Hotz's score).

---

## Phase 0: Problem Understanding (Prerequisite)

### What You're Optimizing
A **tree traversal kernel** that:
- Walks a binary forest (height=10) for 16 rounds
- Hashes values at each node using XOR/ADD/SHL operations
- Processes 256 items in batch

### The Target Machine: VLIW Simulator
VLIW (Very Long Instruction Word) = **compiler-scheduled parallelism**.

**Key insight**: Multiple independent operations can execute **in the same cycle** if packed into one instruction.

| Engine | Slots/Cycle | Operations |
|--------|-------------|------------|
| ALU | 12 | Scalar arithmetic (+, *, ^, <<, etc.) |
| VALU | 6 | Vector ops (8-wide) |
| Load | 2 | Memory reads, constants |
| Store | 2 | Memory writes |
| Flow | 1 | Select, halt |

**Constraints**:
- Scratch space (registers): **1536 words maximum**
- Vector width: 8 elements

### Current State of tinygrad Implementation
```
examples/anthropic_challenge.py
├── tree_traversal()     # Algorithm in tinygrad Tensors
├── VLIWRenderer         # Compiles UOps → VLIW instructions
└── vliw_prepare         # Graph transformations (loop unrolling, etc.)
```

**Two problems to solve**:
1. **Register allocation**: Currently uses ~3000+ registers (limit: 1536)
2. **VLIW packing**: Currently 1 op/cycle (should be many ops/cycle)

### Action Items
- [ ] Run: `python examples/anthropic_challenge.py`
- [ ] Note the register count and cycle count in output
- [ ] Read lines 89-140 of `examples/anthropic_challenge.py` (VLIWRenderer.render)

---

## Phase 1: tinygrad UOp System

### Why This Matters
The VLIWRenderer receives a list of UOps and must convert them to VLIW instructions. You need to understand what UOps are and how they flow through the system.

### Key Concepts

**UOp (Universal Operation)**: Everything in tinygrad is a UOp - a node in a computation graph.

```python
class UOp:
    op: Ops      # Operation type (ADD, MUL, LOAD, STORE, etc.)
    dtype: DType # Data type (uint32, uint32.vec(8), etc.)
    src: tuple   # Input UOps (forms a DAG)
    arg: Any     # Operation-specific argument
```

**Important UOps for this challenge**:
| UOp | Purpose | Example |
|-----|---------|---------|
| CONST | Constant value | `UOp(CONST, uint32, arg=42)` |
| LOAD | Read from memory | `UOp(LOAD, uint32, (addr_uop,))` |
| STORE | Write to memory | `UOp(STORE, None, (addr_uop, val_uop))` |
| ADD/MUL/XOR | Arithmetic | `UOp(ADD, uint32, (a, b))` |
| VECTORIZE | Combine scalars | `UOp(VECTORIZE, uint32.vec(8), (s0,s1,...))` |
| GEP | Extract from vector | `UOp(GEP, uint32, (vec,), arg=(idx,))` |
| WHERE | Ternary select | `UOp(WHERE, dtype, (cond, true, false))` |

**The UOp list**: By the time `render()` is called, the UOp graph has been linearized into a list. The order respects dependencies (uses come after definitions).

### Key Files to Read
- `tinygrad/uop/ops.py`: UOp class definition (search for `class UOp`)
- `examples/anthropic_challenge.py:89-140`: How render() processes UOps

### Action Items
- [ ] Understand: UOp is a node with op, dtype, src, arg
- [ ] Understand: The UOp list is already topologically sorted
- [ ] Understand: `u.dtype.count` tells you if it's scalar (1) or vector (8)

---

## Phase 2: Register Allocation Fundamentals

### Why This Matters
Current implementation: each UOp gets fresh registers → uses ~3000+ registers
Required: fit within 1536 registers by **reusing** registers for non-overlapping lifetimes

### Key Concepts

**Live Range**: The span from when a value is defined to when it's last used.

```
UOp 0: a = CONST 5        # a is defined here
UOp 1: b = CONST 3        # b is defined here
UOp 2: c = ADD a, b       # a,b are used here (last use for both)
UOp 3: d = MUL c, c       # c is used here (last use for c)
```

Live ranges:
- `a`: [0, 2]
- `b`: [1, 2]
- `c`: [2, 3]
- `d`: [3, ...]

**Interference**: Two values interfere if their live ranges overlap → cannot share a register.

**Linear Scan Algorithm** (simplest practical allocator):
1. Compute live intervals for each value
2. Sort intervals by start position
3. Walk through, assigning registers
4. When a value's interval ends, its register becomes free
5. If no register available, **spill** to memory (not needed if algorithm is good)

### Key Insight for This Challenge
The UOp list is already in topological order. For each UOp:
- **Definition point**: The UOp's position in the list
- **Last use point**: The latest position where this UOp appears in another's `src`

### Pseudocode for Linear Scan
```python
def allocate_registers(uops):
    live_end = {}  # uop -> last position where it's used

    # Pass 1: Find last use of each UOp
    for i, u in enumerate(uops):
        for src in u.src:
            live_end[src] = i  # Update last use position

    # Pass 2: Allocate registers
    free_regs = []
    reg_map = {}
    for i, u in enumerate(uops):
        # Free registers whose values are dead
        for v, end in list(live_end.items()):
            if end < i and v in reg_map:
                free_regs.append(reg_map[v])
                del reg_map[v]

        # Allocate register for u
        if needs_register(u):
            reg = free_regs.pop() if free_regs else next_new_reg()
            reg_map[u] = reg

    return reg_map
```

### Complication: Vector Registers
UOps with `dtype.count == 8` need 8 contiguous registers. Your allocator must handle this.

### Action Items
- [ ] Understand: Live ranges and interference
- [ ] Understand: Linear scan algorithm
- [ ] Understand: Why reusing registers reduces total count

---

## Phase 3: Dependency Analysis for VLIW

### Why This Matters
VLIW packing requires knowing which operations are **independent** (no data dependencies) and can execute in parallel.

### Key Concepts

**Data Dependencies** (3 types):
| Type | Pattern | Example | Can Reorder? |
|------|---------|---------|--------------|
| RAW (Read After Write) | B reads what A wrote | `a = x + y; b = a * 2` | No |
| WAR (Write After Read) | B writes what A read | `a = x + y; x = 5` | With renaming |
| WAW (Write After Write) | B writes same as A | `a = 5; a = 7` | No |

In the UOp list, **all dependencies are RAW** because UOps are SSA (each UOp is defined exactly once).

**Dependency Graph**: Build a DAG where edge A→B means B depends on A.

```python
def build_dependency_graph(uops):
    deps = {u: set() for u in uops}
    for u in uops:
        for src in u.src:
            deps[u].add(src)  # u depends on src
    return deps
```

**Independent Operations**: Two ops are independent if neither is in the other's dependency chain.

### Key Insight for VLIW Packing
Operations that:
1. Have **no dependencies** between them
2. Use **different engines** (or same engine with available slots)

...can be packed into the **same VLIW instruction**.

### Action Items
- [ ] Understand: RAW dependencies (the only kind in UOp lists)
- [ ] Understand: How to build a dependency graph from UOp.src
- [ ] Understand: Independence = no path in dependency graph

---

## Phase 4: VLIW Instruction Scheduling

### Why This Matters
Even with dependency info, you need a **strategy** to decide which operations to pack together.

### Key Concepts

**List Scheduling Algorithm**:
1. Build dependency graph
2. Compute **priority** for each operation (e.g., critical path length)
3. Maintain "ready list" of ops whose dependencies are satisfied
4. Each cycle: pick highest-priority ready ops that fit in VLIW slots
5. Mark scheduled ops as complete, add newly-ready ops

**Critical Path**: Longest chain of dependent operations. Operations on the critical path should be prioritized.

**Resource Constraints**: Each engine has limited slots per cycle:
```python
ENGINE_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
```

### Pseudocode for List Scheduling
```python
def schedule(uops, deps):
    ready = [u for u in uops if not deps[u]]  # No dependencies
    scheduled = []

    while ready:
        # Build one VLIW instruction
        instruction = {"alu": [], "valu": [], "load": [], "store": [], "flow": []}
        used = []

        for u in sorted(ready, key=priority, reverse=True):
            engine = get_engine(u)
            if len(instruction[engine]) < ENGINE_LIMITS[engine]:
                instruction[engine].append(u)
                used.append(u)

        scheduled.append(instruction)

        # Update ready list
        for u in used:
            ready.remove(u)
        for u in uops:
            if u not in ready and all(d in used or d in scheduled for d in deps[u]):
                ready.append(u)

    return scheduled
```

### Action Items
- [ ] Understand: List scheduling algorithm
- [ ] Understand: Priority heuristics (critical path)
- [ ] Understand: Resource constraints (engine slot limits)

---

## Phase 5: Implementation

### Order of Implementation

**Step 1: Register Allocator** (Required for correctness)
- Implement liveness analysis
- Implement linear scan allocation
- Handle vector registers (8 contiguous)
- Verify: register count <= 1536

**Step 2: VLIW Packing** (Required for performance)
- Build dependency graph
- Implement list scheduling
- Pack operations respecting engine limits
- Verify: significant cycle reduction

**Step 3: Tuning** (To beat 1,340)
- Better priority heuristics
- Instruction reordering optimizations
- Profile and iterate

### Key Files to Modify
```
examples/anthropic_challenge.py
└── VLIWRenderer.render()  # Lines 89-140
    ├── Add: liveness analysis
    ├── Add: register allocation
    ├── Add: dependency analysis
    └── Add: VLIW packing/scheduling
```

### Testing
```bash
# Run and verify correctness
python examples/anthropic_challenge.py

# Expected output (after optimization):
# - "XXX regs used" (must be <= 1536)
# - "ran for XXX cycles" (goal: < 1340)
# - "compare passed!" (correctness)
```

---

## Summary: Learning Checklist

| Phase | Topic | Key Takeaway |
|-------|-------|--------------|
| 0 | Problem Understanding | VLIW = compiler-scheduled parallelism |
| 1 | UOp System | UOps form a linearized DAG with op/dtype/src/arg |
| 2 | Register Allocation | Reuse registers based on live ranges |
| 3 | Dependency Analysis | Build DAG from UOp.src to find independent ops |
| 4 | VLIW Scheduling | List scheduling with priority and resource constraints |
| 5 | Implementation | Allocator first, then packing |

---

## Quick Reference: Current VLIWRenderer.render() Flow

```python
def render(self, uops):
    reg, inst = 0, []
    r = {}  # UOp -> register mapping

    for u in uops:
        # 1. Allocate register (DUMB: just increment)
        if u.op not in {STORE, SINK, GEP}:
            r[u] = reg
            reg += u.dtype.count  # <-- PROBLEM: never reuses

        # 2. Emit instruction (ONE OP PER INSTRUCTION)
        match u.op:
            case Ops.ADD:
                inst.append({"valu": [(...)]})  # <-- PROBLEM: no packing
            ...

    return repr(inst)
```

**Your task**: Replace the "DUMB" allocation and add packing logic.
