# Tinygrad Learning Plan: From User to Expert Contributor

This plan is designed to build deep understanding of tinygrad from first principles. Each step is a self-contained session you can start with a coding assistant in a fresh conversation. Steps build on each other sequentially.

**How to use**: Copy the relevant step into a new conversation. Each step includes context, files to read, exercises, and verification criteria so the assistant knows exactly what to cover.

---

## Step 1: The Tensor API and Lazy Evaluation

**Goal**: Understand how tinygrad's user-facing API works and why nothing executes until `.realize()`.

**Context for the assistant**: I'm learning tinygrad's internals from first principles. Start with the absolute basics of how tensor operations build lazy computation graphs. Use simple examples I can run interactively.

**Files to read (in order)**:
1. `tinygrad/tensor.py` — focus on `class Tensor`, the `__init__` method, and how basic operations are defined
2. `tinygrad/mixin/elementwise.py` — how element-wise ops (add, mul, sin, relu, etc.) are implemented as UOp graph construction
3. `tinygrad/mixin/movement.py` — how reshape, permute, expand, pad, shrink work

**Exercises** (run each in a Python REPL):
1. Create `x = Tensor([1, 2, 3])` and print `x.uop`. Draw the UOp tree on paper. Explain what each node means.
2. Do `y = x + 1` (without realize). Print `y.uop`. How did the graph grow? What new ops appeared?
3. Do `z = (x + 1) * 2`. Print `z.uop`. Trace the full graph.
4. Now do `z.realize()`. Print `z.uop` again. What changed? Where did the computation graph go?
5. Try `a = Tensor([1,2,3]).reshape(3,1).expand(3,4)`. Print `a.uop` at each step. Identify which Ops correspond to reshape and expand.

**Verification**: You should be able to explain:
- Why `Tensor([1,2,3]) + 1` doesn't execute anything
- What a UOp node contains (op, dtype, src, arg)
- How `x.uop` changes before vs after `.realize()`
- What the 6 movement ops are (RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP)

---

## Step 2: UOp — The Universal IR

**Goal**: Deeply understand the UOp class and the Ops enum — tinygrad's single, universal intermediate representation for everything.

**Context for the assistant**: I've already learned about lazy evaluation and how tensor ops build UOp graphs (Step 1). Now I need to understand UOp itself in depth — the class, the Ops enum, how deduplication works, and the key methods.

**Files to read (in order)**:
1. `tinygrad/uop/__init__.py` — the `Ops` enum (all ~100 ops). Read every op and its comment. Note the 6 sections: defines/special, non-op uops, load/store, math, control flow, tensor graph ops. Also read `GroupOp` at the bottom.
2. `tinygrad/uop/ops.py` — the `UOp` class definition. Focus on:
   - `UOpMetaClass` and `ucache` (lines ~84-121) — how UOps are deduplicated via weak references
   - `class UOp` (line ~122) — the 4 fields: `op`, `dtype`, `src`, `arg`
   - Key methods: `.toposort()`, `.simplify()`, `.const()`, `.substitute()`, `.replace()`
   - `PatternMatcher` class and `UPat` class — the graph rewriting engine
3. `tinygrad/uop/upat.py` — `UPat` pattern matching primitives

**Exercises**:
1. In a REPL, create two identical UOps and verify they are the same object (deduplication): `a = UOp(Ops.CONST, dtypes.int, arg=5); b = UOp(Ops.CONST, dtypes.int, arg=5); assert a is b`
2. Build a simple UOp graph by hand: `x = UOp(Ops.CONST, dtypes.float, arg=1.0); y = UOp(Ops.CONST, dtypes.float, arg=2.0); z = UOp(Ops.ADD, dtypes.float, (x, y))`. Print and inspect `z`.
3. Use `.toposort()` on the graph from exercise 2. Verify the ordering.
4. Categorize the Ops enum into: tensor-graph-only ops (RESHAPE, BUFFER, etc.), kernel ops (RANGE, LOAD, STORE, etc.), and math ops (ADD, MUL, etc.). Explain why this distinction matters.
5. Study `GroupOp` — what is `GroupOp.ALU`? What is `GroupOp.Movement`? Why are these groupings useful?

**Verification**: You should be able to:
- Explain why UOps are deduplicated (memory efficiency, graph identity)
- List the 4 fields of a UOp and what each contains
- Explain the difference between tensor-graph UOps and kernel UOps
- Describe what `PatternMatcher` does at a high level (graph rewriting via pattern rules)

---

## Step 3: PatternMatcher — The Rewriting Engine

**Goal**: Master tinygrad's graph rewriting engine, which powers everything from gradient computation to codegen.

**Context for the assistant**: I understand UOp and the Ops enum (Step 2). Now I need to understand PatternMatcher and UPat deeply — this is the most important mechanism in tinygrad, used in gradient computation, scheduling, lowering, and optimization.

**Files to read (in order)**:
1. `tinygrad/uop/upat.py` — the `UPat` class. Understand how patterns are constructed: `UPat(Ops.ADD)`, `UPat(Ops.ADD, src=(UPat.var("x"), UPat.var("y")))`, `UPat.cvar()`, etc.
2. `tinygrad/uop/ops.py` — search for `class PatternMatcher`. Understand how it takes a list of `(pattern, replacement_fn)` tuples and applies them to a UOp graph.
3. `tinygrad/gradient.py` — **the simplest real example** of PatternMatcher. Read `pm_gradient` (line ~26). For each rule, trace what pattern it matches and what it returns. Start with `Ops.ADD` (returns `(ctx, ctx)` — gradient of addition is identity for both inputs).
4. `tinygrad/uop/symbolic.py` — symbolic simplification rules. Look at the first 20-30 rules to see how algebraic identities are expressed as pattern matching.

**Exercises**:
1. Read `pm_gradient` line by line. For each rule, write in plain English what it does. Example: "The CAST rule: when computing gradient through a cast, cast the gradient back to the source dtype."
2. Create a minimal PatternMatcher yourself:
   ```python
   from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops
   from tinygrad.dtype import dtypes
   # Rule: x + 0 → x
   pm = PatternMatcher([(UPat(Ops.ADD, src=(UPat.var("x"), UPat(Ops.CONST, arg=0))), lambda x: x)])
   ```
   Apply it to a test graph and verify it simplifies.
3. Study 5 rules from `symbolic.py`. For each, identify: (a) the algebraic identity, (b) the UPat pattern, (c) the replacement.
4. Explain why PatternMatcher is used instead of traditional visitor/AST-walk patterns.

**Verification**: You should be able to:
- Write a simple PatternMatcher rule from scratch
- Read any rule in `pm_gradient` and explain what gradient it computes
- Explain why `UPat.var("x")` captures a node and `UPat.cvar()` matches constants
- Describe the match-and-replace cycle

---

## Step 4: Scheduling — From Lazy Graphs to Kernels

**Goal**: Understand how tinygrad's scheduler breaks a lazy UOp graph into executable kernel boundaries and movement operations.

**Context for the assistant**: I understand UOp, PatternMatcher, and the tensor API (Steps 1-3). Now I need to understand scheduling — when `.realize()` is called, how does tinygrad decide which operations go into which kernels? This is the bridge between the user's lazy graph and actual GPU execution.

**Files to read (in order)**:
1. `docs/abstractions3.py` — a front-to-back walkthrough. Run it and study the output: how many schedule items are generated, what types they are.
2. `tinygrad/engine/schedule.py` — the main scheduling logic. Focus on:
   - `class ExecItem` — what a scheduled item looks like
   - How the scheduler walks the UOp graph and decides kernel boundaries
   - The topological sort of kernels
3. `tinygrad/schedule/rangeify.py` — how movement ops (RESHAPE/PERMUTE/EXPAND/PAD/SHRINK) are converted into explicit RANGE loops and index computations
4. `tinygrad/engine/memory.py` — how buffers are planned (reuse, allocation)

**Exercises**:
1. Run `docs/abstractions3.py` and study the output. How many schedule items? What types (kernel, copy, etc.)?
2. Simple fusion example:
   ```python
   from tinygrad import Tensor
   x = Tensor.rand(100).realize()
   y = (x + 1) * 2 - 3
   schedule = Tensor.schedule(y)
   print(f"{len(schedule)} items")
   for si in schedule: print(si)
   ```
   Verify it produces 1 kernel (fused elementwise).
3. Reduction boundary example:
   ```python
   x = Tensor.rand(100).realize()
   y = (x + 1).sum() * 2
   schedule = Tensor.schedule(y)
   print(f"{len(schedule)} items")
   ```
   Explain why this might produce 2 kernels (reduction + elementwise) or 1 kernel (if the `*2` is fused into the reduction store).
4. Multi-realize example:
   ```python
   x = Tensor.rand(100).realize()
   y = x.sum()
   z = x.max()
   schedule = Tensor.schedule(y, z)
   ```
   How many kernels? Why can't sum and max always be fused?
5. Run `DEBUG=1 python3 -c "from tinygrad import Tensor; (Tensor.rand(100)+1).sum().realize()"` and read the scheduling output.

**Verification**: You should be able to explain:
- What triggers scheduling (`.realize()`, `.numpy()`, `Tensor.schedule()`)
- Why elementwise ops fuse into one kernel but reductions create boundaries
- What an `ExecItem` contains
- How movement ops become index computations (not separate kernels)

---

## Step 5: Lowering — The 10+ Rewrite Passes

**Goal**: Understand `full_rewrite_to_sink()` — the orchestrator that transforms a high-level kernel UOp graph into low-level, device-ready IR through sequential PatternMatcher passes.

**Context for the assistant**: I understand scheduling (Step 4) and PatternMatcher (Step 3). Now I need to understand the lowering pipeline — the series of graph rewrites that transform a scheduled kernel's UOp AST into something a renderer can turn into GPU code. This is `codegen/__init__.py`'s `full_rewrite_to_sink()`.

**Files to read (in order)**:
1. `tinygrad/codegen/__init__.py` — the main orchestrator. Read `full_rewrite_to_sink()` top to bottom. Identify each rewrite pass and its purpose. Key passes:
   - Movement op cleanup
   - Load collapse
   - Range splitting/flattening
   - Symbolic simplification
   - Post-range optimization (BEAM search)
   - Expander (UNROLL/UPCAST expansion)
   - GPU dims assignment
   - Devectorization
   - Index dtype lowering
   - Decompositions to device primitives
   - Control flow injection
2. `tinygrad/codegen/opt/postrange.py` — BEAM search optimization. How tinygrad explores different optimization strategies (UPCAST, UNROLL, LOCAL, THREAD, GROUP).
3. `tinygrad/codegen/late/linearizer.py` — topological sort with priority ordering (loads early, stores late)
4. `tinygrad/codegen/late/expander.py` — UNROLL/UPCAST expansion

**Exercises**:
1. Use VIZ=1 to visualize the rewrite passes:
   ```bash
   VIZ=1 python3 -c "from tinygrad import Tensor; (Tensor.rand(256)+1).realize()"
   ```
   Open the visualization and step through each rewrite pass. Note how the graph changes.
2. Trace a simple elementwise kernel through the passes:
   ```bash
   DEBUG=4 python3 -c "from tinygrad import Tensor; (Tensor.rand(16).realize()+1).realize()"
   ```
   Read the generated code. Map each line back to a lowering decision.
3. Trace a reduction kernel:
   ```bash
   DEBUG=4 python3 -c "from tinygrad import Tensor; Tensor.rand(256).realize().sum().realize()"
   ```
   Identify: RANGE loops, accumulator, barrier, local memory (if any).
4. Compare with NOOPT=1:
   ```bash
   NOOPT=1 DEBUG=4 python3 -c "from tinygrad import Tensor; Tensor.rand(256).realize().sum().realize()"
   ```
   What optimizations were disabled?
5. Try BEAM search:
   ```bash
   BEAM=2 DEBUG=4 python3 -c "from tinygrad import Tensor; Tensor.rand(256,256).realize().sum().realize()"
   ```
   See how BEAM explores different optimization configurations.

**Verification**: You should be able to:
- List the major rewrite passes in order and explain each one's purpose in 1 sentence
- Explain what UPCAST, UNROLL, LOCAL, THREAD, GROUP optimizations do
- Read generated GPU code (Metal/CUDA/OpenCL) and trace it back to the UOp graph
- Explain the difference between NOOPT=1 and default optimized output

---

## Step 6: Rendering — From UOps to GPU Code

**Goal**: Understand how the final lowered UOp graph is rendered into actual source code (Metal, CUDA, OpenCL, etc.).

**Context for the assistant**: I understand the full pipeline from tensor → schedule → lowered UOps (Steps 1-5). Now I need to understand the final rendering step — how UOps become actual GPU shader/kernel source code strings.

**Files to read (in order)**:
1. `tinygrad/renderer/__init__.py` — base `Renderer` class, `ProgramSpec` dataclass (what a rendered program looks like: source code, global_size, local_size, variable bindings)
2. `tinygrad/renderer/cstyle.py` — the C-style renderer (used for Metal, OpenCL, CUDA). This is the most important renderer. Focus on:
   - How each Ops maps to a C expression (e.g., `Ops.ADD → "({a}+{b})"`)
   - How RANGE becomes `for` loops
   - How LOAD/STORE become pointer dereferences
   - How vectorized types (float4) are handled
   - Buffer argument generation
3. `tinygrad/renderer/ptx.py` — NVIDIA PTX renderer (lower-level, register-based)
4. `docs/developer/runtime.md` — how renderers connect to device runtimes

**Exercises**:
1. Generate Metal code for a simple kernel and annotate every line:
   ```bash
   DEBUG=4 python3 -c "
   from tinygrad import Tensor
   x = Tensor.rand(64).realize()
   y = (x * 2 + 1).sin()
   y.realize()
   "
   ```
   For each line of the generated Metal kernel, identify which UOp produced it.
2. Generate the same kernel for a different backend (if available):
   ```bash
   CPU=1 DEBUG=4 python3 -c "
   from tinygrad import Tensor
   x = Tensor.rand(64).realize()
   y = (x * 2 + 1).sin()
   y.realize()
   "
   ```
   Compare the C code vs Metal code. What's similar? What differs?
3. Study how `ProgramSpec` is constructed. What does `global_size` and `local_size` mean in GPU terms? How do they relate to the RANGE loops in the UOp graph?
4. Find in `cstyle.py` how each of these is rendered:
   - `Ops.ADD`, `Ops.MUL`, `Ops.WHERE`
   - `Ops.LOAD`, `Ops.STORE`
   - `Ops.RANGE`, `Ops.END`
   - `Ops.CAST`, `Ops.BITCAST`

**Verification**: You should be able to:
- Read generated GPU code and map every expression back to a UOp
- Explain what `ProgramSpec` contains and how it's used by the runtime
- Describe the difference between C-style rendering and PTX rendering
- Explain how GPU thread/block dimensions map to RANGE loops

---

## Step 7: Device & Runtime — Execution and Memory

**Goal**: Understand how compiled kernels are dispatched to actual hardware and how buffers are managed.

**Context for the assistant**: I understand the full compilation pipeline (Steps 1-6). Now I need to understand the execution layer — devices, allocators, runtimes, and the JIT.

**Files to read (in order)**:
1. `tinygrad/device.py` — the core device abstractions:
   - `class Buffer` — device memory handle
   - `class Compiled` — base class for all backends (allocator + compiler + runtime)
   - `class Allocator` — memory allocation/deallocation
   - Device singleton and how `Device["METAL"]` works
2. `tinygrad/engine/realize.py` — how schedule items are executed:
   - `CompiledRunner` — compiles and runs a kernel
   - `ViewOp`, `BufferCopy`, `BufferXfer` — non-kernel operations
   - LRU buffer caching
3. `tinygrad/engine/jit.py` — `TinyJit` decorator and graph execution:
   - How repeated calls capture a graph and replay it
   - Graph-level optimizations
4. `tinygrad/runtime/ops_metal.py` (or whatever backend you have) — one concrete backend implementation
5. `docs/developer/hcq.md` — Hardware Command Queue architecture for modern accelerators

**Exercises**:
1. Inspect the device:
   ```python
   from tinygrad.device import Device
   d = Device["METAL"]  # or your backend
   print(type(d))
   print(dir(d))
   ```
2. Trace a kernel execution with `DEBUG=5`:
   ```bash
   DEBUG=5 python3 -c "from tinygrad import Tensor; (Tensor.rand(100)+1).realize()"
   ```
   What happens between compilation and kernel dispatch?
3. Study buffer lifecycle:
   ```python
   from tinygrad import Tensor
   x = Tensor.rand(100).realize()
   print(x.uop)  # Find the BUFFER node
   # The BUFFER's arg is the size, and it has a DEVICE child
   ```
4. Benchmark with the JIT:
   ```python
   from tinygrad import Tensor, TinyJit
   @TinyJit
   def f(x): return (x + 1) * 2
   x = Tensor.rand(1000)
   for i in range(5):
       y = f(x).realize()
       # First call compiles, subsequent calls replay the graph
   ```
5. Run with `PROFILE=1` to see execution timing:
   ```bash
   PROFILE=1 python3 -c "from tinygrad import Tensor; (Tensor.rand(1000,1000).realize() @ Tensor.rand(1000,1000).realize()).realize()"
   ```

**Verification**: You should be able to:
- Explain the `Buffer` → `Allocator` → device memory relationship
- Describe what `TinyJit` does and when to use it
- Trace a kernel from `ProgramSpec` to actual GPU dispatch
- Explain buffer caching and reuse

---

## Step 8: DType System and Decompositions

**Goal**: Understand tinygrad's type system and how high-level ops are decomposed into device-supported primitives.

**Context for the assistant**: I understand the full pipeline (Steps 1-7). Now I need to understand two cross-cutting concerns: (1) the dtype system that governs types throughout the pipeline, and (2) decompositions that convert ops the device doesn't support into sequences of ops it does.

**Files to read (in order)**:
1. `tinygrad/dtype.py` — the DType system:
   - All dtypes: float16/32/64, bfloat16, int8-64, uint8-64, bool, fp8 variants
   - `PtrDType` — pointer types for buffers
   - `ImageDType` — texture memory types
   - Vector types (e.g., `dtypes.float.vec(4)` = float4)
   - `truncate()`, `least_upper_dtype()`, casting rules
2. `tinygrad/uop/decompositions.py` — how high-level ops become low-level sequences:
   - Example: `SUB(a,b)` → `ADD(a, NEG(b))`
   - Example: `FDIV(a,b)` → `MUL(a, RECIPROCAL(b))`
   - Device-specific decompositions (some devices lack certain ops)
3. `tinygrad/uop/spec.py` — type verification rules that catch invalid UOp graphs

**Exercises**:
1. Explore the dtype hierarchy:
   ```python
   from tinygrad.dtype import dtypes
   print(dtypes.float, dtypes.float.itemsize)
   print(dtypes.float.vec(4))  # float4
   print(dtypes.half, dtypes.bfloat16)
   ```
2. Look at decomposition rules. Find:
   - How `Ops.SUB` is decomposed
   - How `Ops.POW` is decomposed
   - What happens with ops the device doesn't support
3. Test casting behavior:
   ```python
   from tinygrad import Tensor
   x = Tensor([1.5, 2.7, 3.1])
   print(x.dtype)
   y = x.half()
   print(y.dtype)
   ```
4. Generate code for a half-precision kernel and compare with float32:
   ```bash
   DEBUG=4 python3 -c "from tinygrad import Tensor; (Tensor.rand(16, dtype='half').realize()+1).realize()"
   ```

**Verification**: You should be able to:
- List the main dtype categories and their sizes
- Explain how vector types work (float4 = `dtypes.float.vec(4)`)
- Trace how an unsupported op gets decomposed
- Explain when and why decompositions happen in the pipeline

---

## Step 9: Testing and Debugging Workflows

**Goal**: Learn tinygrad's testing infrastructure and debugging techniques — essential for contributing.

**Context for the assistant**: I understand tinygrad's architecture (Steps 1-8). Now I need to learn the practical skills for contributing: running tests, debugging failures, and understanding the test organization.

**Files/directories to explore**:
1. Test organization:
   - `test/backend/` — cross-backend tests (test behavior is same across devices)
   - `test/unit/` — single-backend tests (test specific backend features)
   - `test/null/` — no-device/compiler tests (pure logic tests, fastest to run)
2. Key test files:
   - `test/backend/test_tensor.py` — comprehensive tensor operation tests
   - `test/null/test_graph_rewrite.py` — PatternMatcher tests
   - `test/null/test_linearizer_rewrite.py` — codegen rewrite tests
   - `test/null/test_gradient.py` — autograd tests
3. Contributing guidelines: read the top of `CONTRIBUTING.md` or the GitHub PR template

**Exercises**:
1. Run a small test suite:
   ```bash
   python -m pytest test/null/test_const_folding.py -v
   ```
2. Run a single test:
   ```bash
   python -m pytest test/backend/test_tensor.py::TestTensor::test_add -v
   ```
3. Run with debug output:
   ```bash
   DEBUG=4 python -m pytest test/backend/test_tensor.py::TestTensor::test_add -v -s
   ```
4. Try the visualization tool:
   ```bash
   VIZ=1 python -m pytest test/backend/test_tensor.py::TestTensor::test_add -v -s
   ```
5. Write a minimal test that would catch a bug. Example: write a test for `(x + 0).realize()` producing the same result as `x.realize()`.
6. Study how process replay works by searching for `[pr]` in recent commit messages.

**Debug command reference**:
```bash
DEBUG=1  # scheduling info (kernel count, shapes)
DEBUG=2  # more detail
DEBUG=3  # kernel AST
DEBUG=4  # generated source code (MOST USEFUL for codegen work)
DEBUG=5  # execution details
DEBUG=7  # buffer allocations
NOOPT=1  # disable optimizations (useful to isolate codegen vs opt bugs)
VIZ=1    # interactive web visualization
BEAM=N   # try N BEAM search configurations
PROFILE=1 # execution timing
```

**Verification**: You should be able to:
- Run any test file or individual test
- Use DEBUG=4 to inspect generated code
- Identify whether a bug is in scheduling, codegen, or runtime
- Write a regression test for a simple bug fix

---

## Step 10: Contribution Practice — Read and Reproduce Real PRs

**Goal**: Study real merged PRs to understand contribution patterns, then practice making a small change.

**Context for the assistant**: I've completed the tinygrad learning path (Steps 1-9). Now I want to practice contributing by studying real PRs and attempting a small change myself.

**Exercises**:
1. Browse recent merged PRs on `https://github.com/tinygrad/tinygrad/pulls?q=is%3Apr+is%3Amerged`. Find 3 small PRs (labeled "bug fix" or "cleanup") and for each:
   - Read the diff
   - Understand what was broken and how it was fixed
   - Identify which test was added
   - Identify which pipeline stage (scheduling, codegen, rendering, runtime) was affected
2. Find an open issue labeled "good first issue" or "help wanted"
3. Practice the contribution workflow:
   ```bash
   # Install pre-commit hooks
   pre-commit install
   # Create a branch
   git checkout -b my-fix
   # Make changes...
   # Run relevant tests
   python -m pytest test/null/test_relevant.py -v
   # Run linting
   pre-commit run --all-files
   # Commit with [pr] in title for process replay
   git commit -m "[pr] fix: description of change"
   ```
4. Try a concrete mini-task: add a new algebraic simplification rule to `tinygrad/uop/symbolic.py`. For example, if there's a missing identity like `x * 1 → x` (check if it already exists first). Write the UPat rule, add a test in `test/null/`, and run the test suite.

**Verification**: You should be able to:
- Read a PR diff and explain what it changes and why
- Identify which tests cover a given code change
- Run the full pre-commit and test workflow
- Make a small, clean change that passes all checks

---

## Appendix: Quick Reference

### The Pipeline in One Line
```
Tensor ops → UOp DAG → .realize() → Schedule (ExecItems) → Lowering (10+ rewrite passes) → Linearize → Render → Compile → Execute
```

### Most Important Files for Contributors
| File | Why |
|------|-----|
| `tinygrad/uop/__init__.py` | All Ops defined here |
| `tinygrad/uop/ops.py` | UOp class + PatternMatcher |
| `tinygrad/codegen/__init__.py` | Full lowering orchestrator |
| `tinygrad/uop/symbolic.py` | Algebraic simplification rules |
| `tinygrad/renderer/cstyle.py` | GPU code generation |
| `tinygrad/engine/schedule.py` | Kernel boundary decisions |

### Environment Variables
| Variable | Use |
|----------|-----|
| `DEBUG=1-7` | Increasing verbosity |
| `DEBUG=4` | Show generated GPU code |
| `NOOPT=1` | Disable optimizations |
| `VIZ=1` | Web-based visualization |
| `BEAM=N` | BEAM search optimization |
| `PROFILE=1` | Execution timing |
| `CPU=1` | Force CPU backend |
| `METAL=1` / `CUDA=1` | Force specific backend |
