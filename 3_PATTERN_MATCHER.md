# Step 3: PatternMatcher — The Rewriting Engine

## Table of Contents
1. [The Core Idea: Find-and-Replace for Graphs](#1-the-core-idea-find-and-replace-for-graphs)
2. [UPat — The Pattern Language](#2-upat--the-pattern-language)
3. [PatternMatcher — The Rule Engine](#3-patternmatcher--the-rule-engine)
4. [graph_rewrite — The Driver Loop](#4-graph_rewrite--the-driver-loop)
5. [Real Example: Gradient Rules (pm_gradient)](#5-real-example-gradient-rules-pm_gradient)
6. [Real Example: Symbolic Simplification](#6-real-example-symbolic-simplification)
7. [Why PatternMatcher Instead of Visitors](#7-why-patternmatcher-instead-of-visitors)
8. [Exercises and Outputs](#8-exercises-and-outputs)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. The Core Idea: Find-and-Replace for Graphs

Most compilers transform code using the **visitor pattern** — you write a class with methods like `visit_Add()`, `visit_Mul()`, etc. that walk the AST and produce transformed output. This works, but it spreads transformation logic across many methods, makes composition hard, and requires explicit traversal code.

tinygrad takes a radically different approach. The **entire compiler** — from gradient computation to GPU code generation — is built on a single mechanism: **pattern-match-and-replace on UOp graphs**.

Think of it like find-and-replace in a text editor, but for computation graphs:

```
Text editor:           tinygrad:
─────────────          ──────────
Find:    "colour"      Pattern:  ADD(x, CONST(0))
Replace: "color"       Replace:  x

Applied to:            Applied to:
  "The colour is..."     ADD(MUL(a, b), CONST(0))
  → "The color is..."    → MUL(a, b)
```

You define **rules** as `(pattern, replacement)` pairs. The engine walks the graph, tries every pattern on every node, and applies replacements until nothing changes (a **fixed point**). That's the whole mechanism.

### Where it's used

This single mechanism powers nearly everything in tinygrad:

| System | File | What it does |
|--------|------|-------------|
| Gradient computation | `tinygrad/gradient.py` | Computes derivatives via chain rule |
| Algebraic simplification | `tinygrad/uop/symbolic.py` | `x+0 → x`, `x*1 → x`, constant folding |
| Scheduling | `tinygrad/engine/schedule.py` | Decides kernel boundaries |
| Kernel lowering | `tinygrad/codegen/__init__.py` | 10+ rewrite passes from high-level to device IR |
| Rendering | `tinygrad/renderer/cstyle.py` | UOps → GPU source code strings |

Learn PatternMatcher once, understand the whole compiler.

---

## 2. UPat — The Pattern Language

`UPat` (defined at `tinygrad/uop/ops.py:954`) is the pattern class. It mirrors `UOp` — just as a `UOp` describes a computation node, a `UPat` describes **what a computation node should look like** in order to match.

### The 5 matching fields

```
┌──────────────────────────────────────────────────────────────┐
│                         UPat                                  │
├──────────────┬───────────────────────────────────────────────┤
│ op           │ Which Ops to match (None = any)                │
│ match_dtype  │ Which dtypes to match (None = any)             │
│ src          │ Child patterns to match (None = any children)  │
│ arg          │ Specific arg value to match (None = any)       │
│ name         │ Capture name — binds matched node to a name    │
└──────────────┴───────────────────────────────────────────────┘
```

Each field acts as a **filter**. If a field is `None`, it matches anything. If specified, it must match exactly. All fields must match simultaneously for the whole pattern to match.

### Building patterns — from simple to complex

**Match by op type:**
```python
UPat(Ops.ADD)                    # matches any ADD node (any dtype, any src, any arg)
UPat(Ops.CONST)                  # matches any CONST node
UPat((Ops.ADD, Ops.MUL))         # matches ADD or MUL
UPat(GroupOp.Unary)              # matches any unary op (EXP2, LOG2, SIN, etc.)
```

**Match by op + dtype:**
```python
UPat(Ops.CONST, dtype=dtypes.float)    # matches float CONST only
UPat(Ops.ADD, dtype=dtypes.bool)       # matches boolean ADD only
```

**Match by op + arg:**
```python
UPat(Ops.CONST, arg=0)                # matches CONST with value 0
UPat(Ops.CONST, arg=1.0)              # matches CONST with value 1.0
```

**Match by op + children (src):**
```python
# Match ADD whose children are both CONSTs
UPat(Ops.ADD, src=(UPat(Ops.CONST), UPat(Ops.CONST)))

# Match ADD whose left child is anything, right child is CONST(0)
UPat(Ops.ADD, src=(UPat(), UPat(Ops.CONST, arg=0)))
```

**Capture matched nodes with `name`:**
```python
UPat(Ops.ADD, name="the_add")           # matches any ADD, binds it to "the_add"
UPat(Ops.ADD, src=(
    UPat(name="left"),                   # captures left child as "left"
    UPat(name="right")))                 # captures right child as "right"
```

When a pattern matches, all named sub-patterns are collected into a dict and passed as keyword arguments to the replacement function.

### The three essential shortcuts

These are used everywhere in the codebase and you'll see them constantly:

#### `UPat.var(name)` — Wildcard ("match anything")

```python
UPat.var("x")                           # matches ANY UOp, captures as "x"
UPat.var("x", dtype=dtypes.float)       # matches any float UOp, captures as "x"
UPat.var("x", dtype=dtypes.bool)        # matches any bool UOp
```

Defined at `tinygrad/uop/ops.py:1001`:
```python
@staticmethod
def var(name=None, dtype=None): return UPat(dtype=dtype, name=name)
```

It's just `UPat(dtype=dtype, name=name)` — no `op` filter, no `src` filter, no `arg` filter. Matches literally anything.

#### `UPat.cvar(name)` — Constant wildcard ("match any constant")

```python
UPat.cvar("c")                          # matches any CONST or VCONST, captures as "c"
UPat.cvar("c", vec=False)               # matches only scalar CONST (not vector VCONST)
```

Defined at `tinygrad/uop/ops.py:1004`:
```python
@staticmethod
def cvar(name=None, dtype=None, vec=True, arg=None):
    return UPat((Ops.CONST, Ops.VCONST) if vec else Ops.CONST, dtype, name=name, arg=arg)
```

Like `var` but restricted to constant nodes. Essential for constant folding rules.

#### `UPat.const(dtype, value)` — Exact constant match

```python
UPat.const(dtypes.float, 0)             # matches CONST(0) with float dtype
UPat.const(dtypes.bool, True)           # matches CONST(True) with bool dtype
```

Unlike `cvar`, this matches a **specific** constant value.

### Name-based identity matching

When the same name appears multiple times in a pattern, the matched UOps must be the **same object** (identity check via `is`). This is critical for self-referencing rules:

```python
# This pattern:
UPat.var("x") + UPat.var("x")

# Matches:  a + a  (same node on both sides)
# Doesn't match:  a + b  (different nodes, even if same value)
```

This works because of `UPat.match()` at line 1039:
```python
if self.name is not None and store.setdefault(self.name, uop) is not uop: return []
```

`store.setdefault("x", uop)` returns the existing value if "x" is already in the dict. If it's not the same object (`is not uop`), the match fails.

### Commutative matching via list src

A subtle but important detail. When `src` is a **tuple**, children are matched in exact order. When `src` is a **list**, UPat tries **all permutations**:

```python
# Tuple: matches exactly (x, y) in that order
UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c")))
# Matches: ADD(anything, CONST)
# Doesn't match: ADD(CONST, anything)

# List: tries all permutations
UPat(Ops.ADD, src=[UPat.var("x"), UPat.cvar("c")])
# Matches: ADD(anything, CONST)  ← tried first
# Matches: ADD(CONST, anything)  ← tried second
```

This is used automatically for commutative ops. When you write `UPat.var("x") + 0`, the `+` operator on UPat calls `.alu(Ops.ADD, ...)`, which checks if the op is in `GroupOp.Commutative` and uses a list accordingly (line 1032):

```python
def alu(self, op, *src):
    asrc = (self,)+src
    return UPat(op, ..., list(asrc) if op in GroupOp.Commutative else asrc)
```

So `UPat.var("x") + 0` automatically matches both `x + 0` and `0 + x`.

### The operator overloads

`UPat` inherits from `OpMixin`, giving it the same operator overloads as `UOp`. This lets you write patterns that look like math:

```python
# These are equivalent:
UPat.var("x") + 0
UPat(Ops.ADD, src=[UPat.var("x"), UPat.const(None, 0)])

# More examples:
UPat.var("x") * 1                      # matches x * 1 or 1 * x
UPat.var("x") * UPat.var("x")          # matches x * x (same node both sides)
UPat.var("x") + UPat.var("x") * UPat.cvar("c")  # matches x + x*c
UPat.var("x").cast(name="c")           # matches CAST(x), captures cast as "c"
```

This syntactic sugar is why tinygrad's rules read like mathematical identities.

### How UPat.match() works internally

The matching algorithm at `tinygrad/uop/ops.py:1034` is a recursive descent with backtracking:

```python
def match(self, uop, store):
    # Step 1: Check op type
    if self.op is not None and uop.op not in self.op: return []

    # Step 2: Check name (identity matching)
    if self.name is not None and store.setdefault(self.name, uop) is not uop: return []

    # Step 3: Check dtype
    if self.match_dtype is not None and uop.dtype not in self.match_dtype: return []

    # Step 4: Check arg
    if self.arg is not None and self.arg != uop.arg: return []

    # Step 5: Check child count
    if len(uop.src) < self.required_len: return []

    # Step 6: If no src patterns, we're done — match succeeds
    if self.src is None: return [store]

    # Step 7: Recursively match children
    # For each permutation of child patterns (list = try all, tuple = one order):
    for vp in self.src:
        # For each child pair (uop_child, pattern_child):
        for uu, vv in zip(uop.src, vp):
            # Recursively match, accumulating name bindings in store
            new_stores = vv.match(uu, store_copy)
        ...
    return results
```

The return type is `list[dict[str, UOp]]` — a list of possible match results. Each dict maps capture names to matched UOps. Multiple results happen when commutative patterns have multiple valid matchings.

---

## 3. PatternMatcher — The Rule Engine

`PatternMatcher` (at `tinygrad/uop/ops.py:1085`) is the container for a set of rewrite rules. It takes a list of `(UPat, replacement_function)` pairs and applies them to individual UOp nodes.

### Construction

```python
pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat.var("x"), UPat.const(None, 0))),  # pattern
     lambda x: x),                                               # replacement

    (UPat(Ops.MUL, src=(UPat.var("x"), UPat.const(None, 1))),  # pattern
     lambda x: x),                                               # replacement
])
```

Internally, the constructor indexes patterns by op type in `self.pdict`:

```python
def __init__(self, patterns):
    self.pdict: dict[Ops, list[list]] = {}
    for p, fxn in self.patterns:
        entry = [p, compiled_match_fn, p.early_reject]
        for uop in p.op:
            self.pdict.setdefault(uop, []).append(entry)
```

So `pdict[Ops.ADD]` contains only the patterns that can match ADD nodes, `pdict[Ops.MUL]` only MUL patterns, etc. This avoids trying every rule on every node.

### The rewrite method

The core method is `rewrite()` at line 1103:

```python
def rewrite(self, uop, ctx=None):
    # 1. Look up patterns for this op type
    pats = self.pdict.get(uop.op, [])
    if not pats: return None

    # 2. Cache the set of child op types for early rejection
    ler = {u.op for u in uop.src}

    # 3. Try each pattern in order
    for _, match, early_reject in pats:
        # Quick check: does this node have children with the right ops?
        if not early_reject.issubset(ler): continue

        # Full match + replacement
        if (ret := match(uop, ctx)) is not None and ret is not uop:
            return ret

    return None  # no pattern matched
```

Three key design choices:

1. **Indexed by op**: O(1) lookup instead of scanning all patterns.
2. **Early rejection**: Before expensive recursive matching, check a cheap set operation. `early_reject` is the set of child op types the pattern requires. If the node's children don't have those ops, skip immediately.
3. **First match wins**: Patterns are tried in the order they were defined. The first successful match is applied.

### The ctx parameter

Some PatternMatchers need external context beyond what's in the graph. The `ctx` parameter is passed through to replacement functions that accept it:

```python
# In gradient computation, ctx is the gradient flowing backward (dL/d_output)
pm_gradient = PatternMatcher([
    (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
    #                       ^^^ ctx = the upstream gradient
])

# Usage:
result = pm_gradient.rewrite(some_add_node, ctx=upstream_gradient)
```

The replacement function's signature determines whether it receives `ctx`. If the function has a parameter named `ctx`, it gets the context; otherwise, it doesn't.

### Composing PatternMatchers with `+`

PatternMatchers can be combined:

```python
pm_a = PatternMatcher([(pat1, fn1), (pat2, fn2)])
pm_b = PatternMatcher([(pat3, fn3)])
pm_combined = pm_a + pm_b  # has all three rules
```

This is used throughout the codebase to build up complex rule sets from simpler ones. For example in `symbolic.py`:

```python
symbolic_simple = propagate_invalid + PatternMatcher([...])  # phase 1
symbolic = symbolic_simple + commutative + PatternMatcher([...])  # phase 2
```

---

## 4. graph_rewrite — The Driver Loop

`PatternMatcher.rewrite()` only transforms a **single node**. To transform an **entire graph**, you use `graph_rewrite()` (at `tinygrad/uop/ops.py:1360`):

```python
def graph_rewrite(sink, pm, ctx=None, bottom_up=False, name=None, bpm=None, walk=False):
    rewrite_ctx = RewriteContext(pm if not bottom_up else None, pm if bottom_up else bpm, ctx)
    return rewrite_ctx.unified_rewrite(sink) if not walk else rewrite_ctx.walk_rewrite(sink)
```

### How the driver loop works

The `unified_rewrite` method at line 1293 is the main rewrite loop. The algorithm has three stages per node, managed via a stack:

```
Stage 0: Process a node
  ├─ If bottom_up pm exists: apply it repeatedly until fixed point
  └─ Push children onto the stack, then push self at stage 1

Stage 1: Rebuild after children are done
  ├─ Collect rewritten children
  ├─ If children changed: construct new UOp with new children
  ├─ If top-down pm exists: try to rewrite the rebuilt node
  └─ If rewritten: push result for another round (stage 2)

Stage 2: Link result
  └─ Map original node → final rewritten result
```

The key insight: **it applies rules until a fixed point**. If rewriting a node produces a new node, that new node goes back through the pipeline and may trigger more rewrites. This continues until nothing changes.

```
Example: simplifying (x + 0) * 1

  Start:    MUL(ADD(x, 0), 1)

  Pass 1:   MUL(x, 1)          ← rule "x + 0 → x" applied to ADD

  Pass 2:   x                   ← rule "x * 1 → x" applied to MUL

  Pass 3:   x                   ← nothing matches, fixed point reached
```

Without fixed-point iteration, you'd need to carefully order your rules or make multiple explicit passes. With it, you write simple local rules and they compose into complex global transformations.

### Top-down vs bottom-up

```python
# Top-down (default): rewrite children first, then try to rewrite parent
graph_rewrite(sink, pm)

# Bottom-up: rewrite nodes before descending into their children
graph_rewrite(sink, pm, bottom_up=True)
```

- **Top-down** is the default. Children are rewritten first, then the parent is rebuilt with the new children, and the parent is tried against the rules. This is natural for simplification (simplify subexpressions first).
- **Bottom-up** rewrites a node before processing its children. This is useful when you want to transform a high-level node into a different structure before descending. Used in scheduling where you want to handle high-level ops before looking at their internals.

### walk_rewrite (single-pass)

There's also `walk_rewrite` (line 1268) — an MLIR-style single-pass driver that doesn't re-traverse into rewritten subtrees. It's faster but less powerful (no fixed-point convergence). Used when a single pass is sufficient.

---

## 5. Real Example: Gradient Rules (pm_gradient)

`tinygrad/gradient.py` is the cleanest, most readable example of PatternMatcher in action. The entire automatic differentiation system is **~30 pattern-match rules**.

### How it works

`compute_gradient()` walks the forward computation graph in reverse topological order (backpropagation). For each node, it calls `pm_gradient.rewrite(node, ctx=gradient)` where `ctx` is the gradient flowing backward (`dL/d_output`).

Each rule returns a **tuple of gradients**, one for each input of the matched node. `None` means "no gradient for this input."

```
Forward:   y = f(x₁, x₂, ..., xₙ)
Rule returns: (dL/dx₁, dL/dx₂, ..., dL/dxₙ)
              where each is computed from ctx = dL/dy
```

### Rule-by-rule breakdown

#### Simple unary ops (single input → single gradient)

```python
# CAST: gradient through a type cast just casts the gradient back
(UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),))
# Forward:  y = cast(x, float32)
# Backward: dx = cast(dy, x.dtype)
# Intuition: casting doesn't change the mathematical value, so gradient passes through
```

```python
# RECIPROCAL: d/dx(1/x) = -1/x²
(UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,))
# Forward:  ret = 1/x
# Backward: dx = -dy * (1/x)² = -dy * ret²
# Note: reuses `ret` (the forward output) to avoid recomputing 1/x
```

```python
# SIN: d/dx(sin(x)) = cos(x)
(UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,))
# Forward:  ret = sin(x)           where x = ret.src[0]
# Backward: dx = cos(x) * dy = sin(π/2 - x) * dy
# Note: tinygrad has no COS op, so it uses the identity cos(x) = sin(π/2 - x)
```

```python
# LOG2: d/dx(log₂(x)) = 1/(x · ln(2))
(UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),))
# Forward:  ret = log₂(x)          where x = ret.src[0]
# Backward: dx = dy / (x · ln(2))
# Note: chain rule — derivative of log₂(x) is 1/(x·ln(2))
```

```python
# EXP2: d/dx(2ˣ) = 2ˣ · ln(2)
(UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),))
# Forward:  ret = 2ˣ
# Backward: dx = ret · ln(2) · dy = 2ˣ · ln(2) · dy
# Note: reuses `ret` (= 2ˣ) from the forward pass
```

```python
# SQRT: d/dx(√x) = 1/(2√x)
(UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret*2),))
# Forward:  ret = √x
# Backward: dx = dy / (2·√x) = dy / (2·ret)
```

#### Binary ops (two inputs → two gradients)

```python
# ADD: d/dx(x+y) = 1, d/dy(x+y) = 1
(UPat(Ops.ADD), lambda ctx: (ctx, ctx))
# Both inputs get the full upstream gradient unchanged.
# This is the simplest gradient rule — addition distributes gradient equally.
```

```python
# MUL: d/dx(x·y) = y, d/dy(x·y) = x  (product rule)
(UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx))
# Forward:  ret = x · y          where x = ret.src[0], y = ret.src[1]
# Backward: dx = y · dy          (ret.src[1] * ctx)
#           dy_input = x · dy    (ret.src[0] * ctx)
```

```python
# Comparisons: not differentiable → gradient is None
(UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None))
# x < y and x != y have no meaningful gradient
```

```python
# MAX: gradient goes to the larger input (with tie-breaking at 0.5)
(UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("y"))), lambda ctx, x, y:
    ((x>y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0)),
     (x<y).where(ctx, (x.eq(y)).where(ctx * 0.5, 0))))
# If x > y: dx = dy_upstream, dy = 0
# If x < y: dx = 0, dy = dy_upstream
# If x == y: dx = dy = 0.5 * dy_upstream  (split evenly)
```

#### Ternary ops

```python
# WHERE: gradient flows through the chosen branch only
(UPat(Ops.WHERE, name="ret"), lambda ctx, ret:
    (None,                                      # no gradient for condition
     ret.src[0].where(ctx, ctx.const_like(0)),  # gradient for true branch
     ret.src[0].where(ctx.const_like(0), ctx))) # gradient for false branch
# Forward:  ret = cond ? true_val : false_val
# Backward: d_cond = None (not differentiable)
#           d_true = cond ? dy : 0   (gradient only where cond was True)
#           d_false = cond ? 0 : dy  (gradient only where cond was False)
```

#### Reduction ops

```python
# REDUCE_AXIS: delegates to reduce_gradient helper
(UPat(Ops.REDUCE_AXIS, name="ret"), lambda ctx, ret: reduce_gradient(ctx, ret, ret.arg[0]))
```

The helper `reduce_gradient` handles three reduction types:

```python
def reduce_gradient(ctx, ret, op):
    # For SUM (Ops.ADD): broadcast gradient back to input shape
    if op == Ops.ADD: return (broadcast_to_input(ctx),)
    # sum([a,b,c]) = a+b+c, so d/da = d/db = d/dc = 1 → just broadcast ctx

    # For MAX: gradient goes to the max element(s)
    if op == Ops.MAX:
        mask = ret.src[0].eq(broadcast_to_input(ret))  # which elements are max?
        count = mask.sum(...)                            # how many maxes?
        return (mask / count * broadcast_to_input(ctx),) # split evenly among maxes

    # For PROD (Ops.MUL): d/dx_i(∏x) = (∏x)/x_i
    if op == Ops.MUL: return (broadcast_to_input(ctx * ret) / ret.src[0],)
```

#### Movement ops (gradient reverses the movement)

Each movement op's gradient is its **inverse operation**:

```python
# RESHAPE: reshape gradient back to the original shape
(UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None))
# Forward: y = reshape(x, new_shape)
# Backward: dx = reshape(dy, x.shape)   ← undo the reshape

# EXPAND: sum over expanded dimensions (reverse of broadcast)
(UPat(Ops.EXPAND, name="ret"), lambda ctx, ret:
    (ctx.r(Ops.ADD, tuple(i for i,(s,n) in enumerate(zip(ret.src[0].shape, ret.shape)) if s!=n)), None))
# Forward: y = expand(x, bigger_shape)      broadcast: (3,1) → (3,4)
# Backward: dx = sum(dy, expanded_axes)     reduce:    (3,4) → (3,1)

# PAD: shrink gradient (remove the padding region)
(UPat(Ops.PAD, name="ret"), lambda ctx, ret:
    (ctx.shrink(tuple([(p[0], s+p[0]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None))
# Forward: y = pad(x, padding)
# Backward: dx = shrink(dy, original_region)  ← extract the non-padded part

# SHRINK: pad gradient (reverse of shrink)
(UPat(Ops.SHRINK, name="ret"), lambda ctx, ret:
    (ctx.pad(tuple([(p[0], s-p[1]) for s,p in zip(ret.src[0].shape, ret.marg)])), None, None))
# Forward: y = shrink(x, region)
# Backward: dx = pad(dy, inverse_padding)  ← put zeros where shrink removed data

# PERMUTE: permute gradient with inverse permutation
(UPat(Ops.PERMUTE, name="ret"), lambda ctx, ret: (ctx.permute(argsort(ret.marg)),))
# Forward: y = permute(x, perm)
# Backward: dx = permute(dy, inverse(perm))  ← argsort gives the inverse permutation

# FLIP: flip gradient back (flip is its own inverse)
(UPat(Ops.FLIP, name="ret"), lambda ctx, ret: (ctx.flip([i for i,x in enumerate(ret.marg) if x]),))
# Forward: y = flip(x, axes)
# Backward: dx = flip(dy, axes)  ← flipping twice restores original order
```

### The elegance

The entire autograd engine is **~30 declarative rules** in one list. No class hierarchy, no visitor boilerplate, no manual graph walking. Adding a gradient for a new op means adding one tuple to the list.

---

## 6. Real Example: Symbolic Simplification

`tinygrad/uop/symbolic.py` uses PatternMatcher for algebraic simplification. These rules run during lowering to simplify index computations and constant-fold expressions.

### Phase 1: Basic folding (`symbolic_simple`)

#### Self-folding rules (identity elements)

```python
(UPat.var("x") + 0, lambda x: x)      # x + 0 → x   (additive identity)
(UPat.var("x") * 1, lambda x: x)      # x * 1 → x   (multiplicative identity)
(UPat.var("x") // 1, lambda x: x)     # x // 1 → x  (division by 1)
(UPat.var("x") // -1, lambda x: -x)   # x // -1 → -x
```

Remember: `UPat.var("x") + 0` is commutative (uses list src), so it also matches `0 + x`.

#### Zero-folding rules

```python
(UPat.var("x") < UPat.var("x"), lambda x: x.const_like(False).cast(dtypes.bool.vec(x.dtype.count)))
# x < x → False (nothing is less than itself)

(UPat.var("x") % UPat.var("x"), lambda x: x.const_like(0))   # x % x → 0
(UPat.var("x") ^ UPat.var("x"), lambda x: x.const_like(0))   # x ^ x → 0
(UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1))   # x // x → 1
```

These use the same-name identity trick: `UPat.var("x")` appearing twice requires the exact same UOp node in both positions.

#### Constant folding

```python
# Unary: if input is a constant, just compute the result
(UPat(GroupOp.Unary, src=(UPat((Ops.VCONST, Ops.CONST)),), name="a"),
 lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg], False)))

# Binary: if both inputs are constants, compute the result
(UPat(GroupOp.Binary-{Ops.THREEFRY}, src=(UPat((Ops.VCONST, Ops.CONST)),)*2, name="a"),
 lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg, a.src[1].arg], False)))

# Ternary: all three inputs constant → fold
(UPat(GroupOp.Ternary, src=(UPat((Ops.VCONST, Ops.CONST)),)*3, name="a"),
 lambda a: a.const_like(exec_alu(a.op, a.dtype, [a.src[0].arg, a.src[1].arg, a.src[2].arg], False)))
```

The pattern `(UPat((Ops.VCONST, Ops.CONST)),)*2` creates a tuple of two patterns, each matching either CONST or VCONST. `exec_alu` evaluates the ALU operation in Python, so `ADD(CONST(3), CONST(4))` becomes `CONST(7)` at compile time.

#### Multiply-by-zero with NaN handling

```python
(UPat.var("x") * 0, lambda x: x.const_like(
    float("nan") if x.op is Ops.CONST
    and isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0))
```

`x * 0` is usually `0`, but per IEEE 754: `NaN * 0 = NaN` and `inf * 0 = NaN`. The rule handles this correctly.

#### Boolean algebra

```python
(UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool), lambda x,y: x&y)
# bool multiplication is AND

(UPat.var('x', dtype=dtypes.bool) + UPat.var('y', dtype=dtypes.bool), lambda x,y: x|y)
# bool addition is OR
```

These prevent other numeric rules from incorrectly transforming boolean operations.

#### WHERE folding

```python
# where(cond, val, val) → val (same value both ways)
(UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val)

# where(True, a, b) → a; where(False, a, b) → b
(UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")),
 lambda gate, c0, c1: c0 if gate.arg else c1)
```

#### Cast folding

```python
# cast(const, dtype) → const with new dtype
(UPat(Ops.CAST, name="root", src=(UPat.cvar("c"),)),
 lambda root, c: root.const_like(c.arg))

# cast to same dtype → remove cast
(UPat((Ops.CAST, Ops.BITCAST), name="root"),
 lambda root: root.src[0] if root.dtype == root.src[0].dtype else None)

# x.cast(a).cast(b) → x if a preserves all values in b
(UPat.var('x').cast(name="a").cast(name="b"),
 lambda x,a,b: x if x.dtype == b.dtype and can_lossless_cast(b.dtype, a.dtype) else None)
```

### Phase 2: Deeper rules (`symbolic`)

Phase 2 builds on phase 1 with more complex patterns:

```python
# Combine like terms: (x*c0) + (x*c1) → x*(c0+c1)
(UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"),
 lambda x,c0,c1: x*(c0+c1))

# x + x*c → x*(c+1)
(UPat.var("x") + UPat.var("x") * UPat.cvar("c"),
 lambda x,c: x*(c+1))

# x|!x → True
(UPat.var("x", dtype=dtypes.bool) | UPat.var("x").logical_not(),
 lambda x: x.const_like(True))
```

### How rules compose via fixed-point iteration

Consider simplifying `(a + 0) * 1`:

```
Step 1: graph_rewrite encounters ADD(a, CONST(0))
        Rule "x + 0 → x" fires → becomes just `a`

Step 2: Now MUL(a, CONST(1)) has been rebuilt with the simplified child
        Rule "x * 1 → x" fires → becomes just `a`

Step 3: Nothing more to rewrite → fixed point reached
```

No explicit multi-pass scheduling was needed. The fixed-point driver handles it automatically.

---

## 7. Why PatternMatcher Instead of Visitors

Traditional compilers use the visitor pattern:

```python
# Traditional approach (NOT tinygrad)
class Simplifier(Visitor):
    def visit_Add(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(right, Const) and right.value == 0:
            return left
        return Add(left, right)

    def visit_Mul(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(right, Const) and right.value == 1:
            return left
        return Mul(left, right)

    # ... dozens more methods ...
```

PatternMatcher has several advantages:

### 1. Declarative rules read like math

```python
# tinygrad: rules ARE the mathematical identities
(UPat.var("x") + 0, lambda x: x)
(UPat.var("x") * 1, lambda x: x)
```

vs.

```python
# visitor: identities buried in traversal boilerplate
def visit_Add(self, node):
    left = self.visit(node.left)    # boilerplate
    right = self.visit(node.right)  # boilerplate
    if isinstance(right, Const) and right.value == 0:  # the actual identity
        return left
    return Add(left, right)         # boilerplate
```

### 2. Composable — merge rule sets with `+`

```python
basic_rules = PatternMatcher([(pat1, fn1), (pat2, fn2)])
advanced_rules = PatternMatcher([(pat3, fn3)])
all_rules = basic_rules + advanced_rules
```

With visitors, you'd need class inheritance and method resolution order.

### 3. Fixed-point semantics for free

Rules compose automatically through the driver loop. Rule A might enable rule B, which enables rule C. You don't have to think about ordering.

### 4. Separation of "what" from "how"

The pattern (UPat) specifies **what** to match. The lambda specifies **what** to produce. The driver loop handles **how** to traverse. Three separate concerns, cleanly separated.

### 5. Same mechanism everywhere

Gradient rules, simplification rules, lowering rules, optimization rules — all use the same `(UPat, lambda)` format. Learn it once, read any part of the compiler.

### 6. Easy to add new rules

Adding a new optimization is adding one line:

```python
# Before: 30 rules
pm = PatternMatcher([...30 rules...])

# After: 31 rules (added one new algebraic identity)
pm = PatternMatcher([...30 rules...,
    (UPat.var("x") - UPat.var("x"), lambda x: x.const_like(0)),  # x - x → 0
])
```

No new classes, no new methods, no modification of existing code.

---

## 8. Exercises and Outputs

### Exercise 1: Build a minimal PatternMatcher

```python
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, Ops, graph_rewrite
from tinygrad.dtype import dtypes

# Rule: x + 0 → x
pm = PatternMatcher([
    (UPat.var("x") + 0, lambda x: x),
])

# Build a test graph: 5 + 0
five = UOp(Ops.CONST, dtypes.int, arg=5)
zero = UOp(Ops.CONST, dtypes.int, arg=0)
add = UOp(Ops.ADD, dtypes.int, (five, zero))
print(f'Before: {add}')

# Apply the rule to the ADD node
result = pm.rewrite(add)
print(f'After:  {result}')
print(f'Result is five: {result is five}')
```

**Output:**
```
Before: UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.CONST, dtypes.int, arg=5, src=()),
  UOp(Ops.CONST, dtypes.int, arg=0, src=()),))
After:  UOp(Ops.CONST, dtypes.int, arg=5, src=())
Result is five: True
```

The rule matched `ADD(CONST(5), CONST(0))`, captured `CONST(5)` as `x`, and returned it. The ADD node was eliminated.

### Exercise 2: PatternMatcher with multiple rules

```python
# Three simplification rules
pm = PatternMatcher([
    (UPat.var("x") + 0, lambda x: x),     # x + 0 → x
    (UPat.var("x") * 1, lambda x: x),     # x * 1 → x
    (UPat.var("x") * 0, lambda x: x.const_like(0)),  # x * 0 → 0
])

a = UOp(Ops.CONST, dtypes.int, arg=42)
one = UOp(Ops.CONST, dtypes.int, arg=1)
zero = UOp(Ops.CONST, dtypes.int, arg=0)

# Test each rule
print(pm.rewrite(UOp(Ops.ADD, dtypes.int, (a, zero))))  # 42 + 0 → 42
print(pm.rewrite(UOp(Ops.MUL, dtypes.int, (a, one))))   # 42 * 1 → 42
print(pm.rewrite(UOp(Ops.MUL, dtypes.int, (a, zero))))  # 42 * 0 → 0
print(pm.rewrite(UOp(Ops.ADD, dtypes.int, (a, one))))   # 42 + 1 → None (no rule)
```

**Output:**
```
UOp(Ops.CONST, dtypes.int, arg=42, src=())
UOp(Ops.CONST, dtypes.int, arg=42, src=())
UOp(Ops.CONST, dtypes.int, arg=0, src=())
None
```

`42 + 1` returns `None` because no rule matches — the right operand is `1`, not `0`.

### Exercise 3: graph_rewrite for full-graph simplification

```python
# Build: (x + 0) * 1
x = UOp(Ops.CONST, dtypes.int, arg=7)
zero = UOp(Ops.CONST, dtypes.int, arg=0)
one = UOp(Ops.CONST, dtypes.int, arg=1)

expr = UOp(Ops.MUL, dtypes.int, (
    UOp(Ops.ADD, dtypes.int, (x, zero)),  # x + 0
    one                                     # * 1
))

pm = PatternMatcher([
    (UPat.var("x") + 0, lambda x: x),
    (UPat.var("x") * 1, lambda x: x),
])

print(f'Before: {expr}')
print(f'Nodes before: {len(expr.toposort())}')

# pm.rewrite only transforms ONE node — it won't simplify the nested ADD
single = pm.rewrite(expr)
print(f'After pm.rewrite: {single}')  # only MUL * 1 is simplified, ADD+0 remains

# graph_rewrite transforms the ENTIRE graph
result = graph_rewrite(expr, pm)
print(f'After graph_rewrite: {result}')
print(f'Result is x: {result is x}')
```

**Output:**
```
Before: UOp(Ops.MUL, dtypes.int, arg=None, src=(
  UOp(Ops.ADD, dtypes.int, arg=None, src=(
    UOp(Ops.CONST, dtypes.int, arg=7, src=()),
    UOp(Ops.CONST, dtypes.int, arg=0, src=()),)),
  UOp(Ops.CONST, dtypes.int, arg=1, src=()),))
Nodes before: 4

After pm.rewrite: UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.CONST, dtypes.int, arg=7, src=()),
  UOp(Ops.CONST, dtypes.int, arg=0, src=()),))

After graph_rewrite: UOp(Ops.CONST, dtypes.int, arg=7, src=())
Result is x: True
```

`pm.rewrite(expr)` only applied one rule (MUL * 1 → the ADD subexpression). `graph_rewrite` walked the whole graph and applied both rules, collapsing `(7 + 0) * 1` all the way down to `7`.

### Exercise 4: Identity matching (same name = same node)

```python
# Rule: x + x → x * 2
pm = PatternMatcher([
    (UPat.var("x") + UPat.var("x"),
     lambda x: x * UOp(Ops.CONST, x.dtype, arg=2)),
])

a = UOp(Ops.CONST, dtypes.int, arg=5)
b = UOp(Ops.CONST, dtypes.int, arg=5)

# a and b are the SAME object (deduplication!)
print(f'a is b: {a is b}')

# So ADD(a, b) matches "x + x" — both sides are the same node
result = pm.rewrite(UOp(Ops.ADD, dtypes.int, (a, b)))
print(f'Match (same node): {result}')  # matches!

# But ADD(CONST(5), CONST(7)) doesn't match — different nodes
c = UOp(Ops.CONST, dtypes.int, arg=7)
result2 = pm.rewrite(UOp(Ops.ADD, dtypes.int, (a, c)))
print(f'Match (diff nodes): {result2}')  # None — no match
```

**Output:**
```
a is b: True
Match (same node): UOp(Ops.MUL, dtypes.int, arg=None, src=(
  UOp(Ops.CONST, dtypes.int, arg=5, src=()),
  UOp(Ops.CONST, dtypes.int, arg=2, src=()),))
Match (diff nodes): None
```

### Exercise 5: Using ctx for gradient-style rules

```python
# A mini gradient system: ctx is the upstream gradient
pm = PatternMatcher([
    # d/dx(x + y) = (dy_upstream, dy_upstream) — addition distributes gradient
    (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
    # d/dx(x * y) = (y * dy_upstream, x * dy_upstream) — product rule
    (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1] * ctx, ret.src[0] * ctx)),
])

# Forward: z = a * b
a = UOp(Ops.CONST, dtypes.float, arg=3.0)
b = UOp(Ops.CONST, dtypes.float, arg=4.0)
z = UOp(Ops.MUL, dtypes.float, (a, b))

# Backward: dz = 1.0 (upstream gradient)
dz = UOp(Ops.CONST, dtypes.float, arg=1.0)

# Apply the MUL gradient rule with ctx=dz
grads = pm.rewrite(z, ctx=dz)
print(f'da = b * dz: {grads[0]}')  # should be b * 1.0
print(f'db = a * dz: {grads[1]}')  # should be a * 1.0
```

**Output:**
```
da = b * dz: UOp(Ops.MUL, dtypes.float, arg=None, src=(
  UOp(Ops.CONST, dtypes.float, arg=4.0, src=()),
  UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),))
db = a * dz: UOp(Ops.MUL, dtypes.float, arg=None, src=(
  UOp(Ops.CONST, dtypes.float, arg=3.0, src=()),
  UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),))
```

`da = 4.0 * 1.0` (= b * dz) and `db = 3.0 * 1.0` (= a * dz). This is exactly the product rule.

### Exercise 6: Constant folding with graph_rewrite

```python
from tinygrad.uop.symbolic import symbolic_simple

# Build: (3 + 4) * (2 + 0)
three = UOp(Ops.CONST, dtypes.int, arg=3)
four = UOp(Ops.CONST, dtypes.int, arg=4)
two = UOp(Ops.CONST, dtypes.int, arg=2)
zero = UOp(Ops.CONST, dtypes.int, arg=0)

expr = UOp(Ops.MUL, dtypes.int, (
    UOp(Ops.ADD, dtypes.int, (three, four)),    # 3 + 4
    UOp(Ops.ADD, dtypes.int, (two, zero)),      # 2 + 0
))

print(f'Before: {len(expr.toposort())} nodes')
result = graph_rewrite(expr, symbolic_simple)
print(f'After:  {result}')
```

**Output:**
```
Before: 6 nodes
After:  UOp(Ops.CONST, dtypes.int, arg=14, src=())
```

The real `symbolic_simple` PatternMatcher combines identity folding (`2 + 0 → 2`), constant folding (`3 + 4 → 7`, `7 * 2 → 14`), and dozens of other rules. Applied via `graph_rewrite`, the entire expression collapses to `CONST(14)`.

---

## 9. Key Takeaways

### Writing a PatternMatcher rule
A rule is a `(UPat, function)` pair. The UPat describes what to match, the function produces the replacement. Named captures in the UPat become keyword arguments to the function. Return `None` from the function to signal "no rewrite."

### UPat.var("x") captures any node, UPat.cvar("c") captures constants
`var` is the universal wildcard. `cvar` restricts to CONST/VCONST. Same-name captures require the same UOp object (identity matching).

### PatternMatcher.rewrite() transforms one node, graph_rewrite() transforms a whole graph
`rewrite()` tries patterns on a single UOp and returns the replacement (or None). `graph_rewrite()` walks the entire DAG and applies rewrites until a fixed point — no more rules match anywhere.

### Reading any rule in pm_gradient
Each rule takes `ctx` (the upstream gradient dL/dy) and returns a tuple of gradients for each input. The math follows standard calculus: chain rule for compositions, product rule for multiplication, identity for addition.

### The match-and-replace cycle
1. The driver walks the graph (top-down or bottom-up)
2. For each node, it looks up matching patterns by op type
3. Early rejection checks child op types (cheap set operation)
4. Full recursive matching tries to bind the pattern to the node
5. If matched, the replacement function is called with captured bindings
6. The result replaces the original node in the graph
7. New nodes may trigger further rewrites until convergence

### Why PatternMatcher
Declarative (rules read like math), composable (`+` merges rule sets), self-ordering (fixed-point handles rule interactions), uniform (same mechanism everywhere in the compiler), and easy to extend (one line per new rule).

---

## Source Files Referenced

| File | Key Contents |
|------|-------------|
| `tinygrad/uop/ops.py:954-1052` | `UPat` class — pattern construction, matching, operators |
| `tinygrad/uop/ops.py:1000-1008` | `UPat.var()`, `UPat.cvar()`, `UPat.const()` — shortcut constructors |
| `tinygrad/uop/ops.py:1034-1052` | `UPat.match()` — recursive matching algorithm |
| `tinygrad/uop/ops.py:1085-1109` | `PatternMatcher` class — rule indexing and rewrite method |
| `tinygrad/uop/ops.py:1252-1357` | `RewriteContext` — unified_rewrite and walk_rewrite drivers |
| `tinygrad/uop/ops.py:1360-1362` | `graph_rewrite()` — the main entry point |
| `tinygrad/gradient.py:32-64` | `pm_gradient` — all gradient rules (~30 rules) |
| `tinygrad/gradient.py:73-93` | `compute_gradient()` — the backpropagation loop using pm_gradient |
| `tinygrad/uop/symbolic.py:40-125` | `symbolic_simple` — phase 1 simplification rules |
| `tinygrad/uop/symbolic.py:192+` | `symbolic` — phase 2 deeper rules (combine terms, boolean algebra) |
| `tinygrad/uop/upat.py` | UPat compilation — compiles patterns to Python code for speed |
