"""Benchmark toposort. Standalone script, run before/after code changes to compare."""
import time
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# --- Build test graphs ---

def build_linear_chain(n: int) -> UOp:
  node = UOp(Ops.DEFINE_VAR, dtypes.int, (), arg=("v0", 0, 1))
  for i in range(1, n):
    node = UOp(Ops.ADD, dtypes.int, (node, UOp(Ops.CONST, dtypes.int, (), arg=i)))
  return node

def build_diamond_dag(n: int) -> UOp:
  nodes = [UOp(Ops.CONST, dtypes.int, (), arg=i) for i in range(n)]
  for _ in range(n):
    new_nodes = []
    for i in range(len(nodes) - 1):
      new_nodes.append(UOp(Ops.ADD, dtypes.int, (nodes[i], nodes[i + 1])))
    if not new_nodes: break
    nodes = new_nodes
  return nodes[0]

# --- Benchmark runner ---

def run_benchmark(name: str, graph: UOp, n_runs: int = 500, n_warmup: int = 50):
  # warmup
  for _ in range(n_warmup): graph.toposort()

  times = []
  for _ in range(n_runs):
    st = time.perf_counter_ns()
    graph.toposort()
    times.append(time.perf_counter_ns() - st)

  times.sort()
  med = times[len(times) // 2] / 1e3
  mn = times[0] / 1e3
  node_count = len(graph.toposort())
  print(f"  {name:20s} | {node_count:5d} nodes | median: {med:8.1f} us | min: {mn:8.1f} us")

if __name__ == "__main__":
  graphs = [
    ("chain_50", build_linear_chain(50)),
    ("chain_500", build_linear_chain(500)),
    ("diamond_20", build_diamond_dag(20)),
    ("diamond_40", build_diamond_dag(40)),
  ]
  for name, graph in graphs:
    run_benchmark(name, graph)
