# Embedding Phylogenetics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Bayesian phylogenetic inference engine that reconstructs evolutionary trees from protein embedding vectors using Brownian motion, inspired by ExaBayes architecture.

**Architecture:** Single experiment file `experiments/35_embedding_phylogenetics.py` with all classes (Tree, BMLikelihood, Proposals, MCMC, MC3, Diagnostics). Tests in `tests/test_embedding_phylo.py`. Parallelism via ProcessPoolExecutor across independent runs; MC3 chains run sequentially within each run process. Benchmark against IQ-TREE on 40 conotoxin proteins.

**Tech Stack:** numpy (vectorized BM likelihood, NJ), concurrent.futures (parallel runs), h5py (embedding I/O), matplotlib (diagnostics/tanglegram)

**Review fixes applied:** (1) Added `resolve_polytomies()` for trifurcating IQ-TREE tree, (2) Replaced eSPR with symmetric fixed-radius SPR for correct Hastings ratio, (3) Fixed sigma proposal shared state, (4) Changed ASDSF `min_freq` to 0.05, (5) Use pre-existing SCOPe 5K corpus stats for ABTT3

**Spec:** `docs/superpowers/specs/2026-03-17-embedding-phylogenetics-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `experiments/35_embedding_phylogenetics.py` | All classes + experiment main. Self-contained, later extractable to tools/ |
| `tests/test_embedding_phylo.py` | Unit + integration tests for tree, likelihood, proposals, NJ, RF, MCMC |

---

## Chunk 1: Tree Data Structures + Newick I/O + NJ Builder

### Task 1: TreeNode and Tree dataclasses

**Files:**
- Create: `experiments/35_embedding_phylogenetics.py`
- Create: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write test file scaffold + TreeNode/Tree tests**

```python
# tests/test_embedding_phylo.py
"""Tests for Experiment 35: Bayesian phylogenetics from protein embeddings."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# We import from the experiment file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from embedding_phylogenetics_35 import TreeNode, Tree


class TestTreeNode:
    def test_leaf_creation(self):
        node = TreeNode(id=0, name="taxon_A")
        assert node.is_leaf()
        assert node.name == "taxon_A"
        assert node.branch_length == 0.0
        assert node.children == []

    def test_internal_node(self):
        left = TreeNode(id=0, name="A")
        right = TreeNode(id=1, name="B")
        parent = TreeNode(id=2, children=[left, right])
        left.parent = parent
        right.parent = parent
        assert not parent.is_leaf()
        assert len(parent.children) == 2


class TestTree:
    def test_three_taxon_tree(self):
        """Build ((A:1,B:2):0.5,C:3) and verify structure."""
        a = TreeNode(id=0, name="A", branch_length=1.0)
        b = TreeNode(id=1, name="B", branch_length=2.0)
        internal = TreeNode(id=3, children=[a, b], branch_length=0.5)
        a.parent = internal
        b.parent = internal
        c = TreeNode(id=2, name="C", branch_length=3.0)
        root = TreeNode(id=4, children=[internal, c])
        internal.parent = root
        c.parent = root
        tree = Tree(root=root)

        assert tree.n_leaves == 3
        assert set(tree.leaf_names()) == {"A", "B", "C"}
        assert tree.n_internal == 2  # root + one internal

    def test_copy_is_independent(self):
        a = TreeNode(id=0, name="A", branch_length=1.0)
        b = TreeNode(id=1, name="B", branch_length=2.0)
        root = TreeNode(id=2, children=[a, b])
        a.parent = root
        b.parent = root
        tree = Tree(root=root)
        tree_copy = tree.copy()
        # Modify copy, original unchanged
        tree_copy.root.children[0].branch_length = 99.0
        assert tree.root.children[0].branch_length == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestTreeNode -v 2>&1 | head -20`
Expected: FAIL (import error — module doesn't exist yet)

- [ ] **Step 3: Write TreeNode and Tree classes**

```python
# experiments/35_embedding_phylogenetics.py
#!/usr/bin/env python3
"""Experiment 35 — Bayesian phylogenetics from protein embeddings.

Re-implements ExaBayes core MCMC in Python with a Brownian motion likelihood
for continuous embedding data. No existing Bayesian phylo software handles
512-dimensional continuous characters.

Spec: docs/superpowers/specs/2026-03-17-embedding-phylogenetics-design.md
"""

# Make importable as a module name without the leading number
# (tests import as `from embedding_phylogenetics_35 import ...`)
__all__ = [
    "TreeNode", "Tree", "NJBuilder", "BMLikelihood",
    "StochasticNNI", "SubtreePruneRegraft", "BranchLengthMultiplier",
    "TreeLengthMultiplier", "SigmaMultiplier", "ProposalMixer",
    "MCMCChain", "MC3Runner", "MultiRunOrchestrator",
    "Diagnostics", "ConsensusBuilder",
    "parse_newick", "write_newick", "random_tree", "simulate_bm",
    "robinson_foulds",
]

import copy
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Tree Data Structures ─────────────────────────────────────────────────


@dataclass
class TreeNode:
    """A node in a rooted binary phylogenetic tree."""
    id: int
    name: str = ""
    branch_length: float = 0.0
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


class Tree:
    """Rooted binary phylogenetic tree with Newick I/O."""

    def __init__(self, root: TreeNode):
        self.root = root
        self._reindex()

    def _reindex(self):
        """Rebuild node lists and leaf lookup from root."""
        self.nodes: List[TreeNode] = []
        self.leaves: List[TreeNode] = []
        self.internals: List[TreeNode] = []
        self._name_to_leaf: Dict[str, TreeNode] = {}
        stack = [self.root]
        while stack:
            node = stack.pop()
            self.nodes.append(node)
            if node.is_leaf():
                self.leaves.append(node)
                if node.name:
                    self._name_to_leaf[node.name] = node
            else:
                self.internals.append(node)
            for child in node.children:
                stack.append(child)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def n_internal(self) -> int:
        return len(self.internals)

    def leaf_names(self) -> List[str]:
        return [leaf.name for leaf in self.leaves]

    def copy(self) -> "Tree":
        """Deep copy the tree (topology + branch lengths)."""
        node_map: Dict[int, TreeNode] = {}

        def _copy_subtree(node: TreeNode, parent: Optional[TreeNode] = None) -> TreeNode:
            new_node = TreeNode(
                id=node.id, name=node.name,
                branch_length=node.branch_length, parent=parent,
            )
            node_map[node.id] = new_node
            new_node.children = [_copy_subtree(c, new_node) for c in node.children]
            return new_node

        new_root = _copy_subtree(self.root)
        return Tree(root=new_root)

    def total_branch_length(self) -> float:
        return sum(n.branch_length for n in self.nodes if not n.is_root())

    def postorder(self) -> List[TreeNode]:
        """Return nodes in postorder (leaves first, root last)."""
        result = []
        stack = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited or node.is_leaf():
                result.append(node)
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))
        return result

    def resolve_polytomies(self):
        """Convert multifurcating nodes to binary cascades with zero-length branches.

        Standard in phylogenetics when reading unrooted IQ-TREE trees (trifurcating root).
        """
        max_id = max(n.id for n in self.nodes)
        next_id = [max_id + 1]
        def _next_id():
            nid = next_id[0]; next_id[0] += 1; return nid

        for node in list(self.nodes):
            while len(node.children) > 2:
                # Take last two children, group under new internal node
                c1 = node.children.pop()
                c2 = node.children.pop()
                new_internal = TreeNode(id=_next_id(), children=[c1, c2],
                                        branch_length=0.0, parent=node)
                c1.parent = new_internal
                c2.parent = new_internal
                node.children.append(new_internal)
        self._reindex()
```

Note: the file is named `35_embedding_phylogenetics.py` but to import it in tests we need a symlink or rename. The standard project pattern uses `sys.path.insert`. We'll create a small importable alias — or the test can import via `importlib`. Actually, looking at the test, let's use an import helper:

Update the test import to:
```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "embedding_phylogenetics_35",
    str(Path(__file__).resolve().parent.parent / "experiments" / "35_embedding_phylogenetics.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TreeNode = mod.TreeNode
Tree = mod.Tree
```

Actually this gets messy with many imports. Simpler: the test file just does:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
# Python can't import a module starting with a digit, so:
import importlib
_exp35 = importlib.import_module("35_embedding_phylogenetics")
TreeNode = _exp35.TreeNode
Tree = _exp35.Tree
```

We'll use this pattern and add classes to the import as we build them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestTreeNode tests/test_embedding_phylo.py::TestTree -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add Tree and TreeNode data structures for embedding phylogenetics"
```

---

### Task 2: Newick parser and writer

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write Newick I/O tests**

```python
# Add to tests/test_embedding_phylo.py
# Also import: parse_newick, write_newick from _exp35

class TestNewickIO:
    def test_simple_roundtrip(self):
        """Parse a Newick string, write it back, parse again — same topology."""
        nwk = "((A:1.0,B:2.0):0.5,C:3.0);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 3
        assert set(tree.leaf_names()) == {"A", "B", "C"}
        # Roundtrip
        nwk2 = write_newick(tree)
        tree2 = parse_newick(nwk2)
        assert tree2.n_leaves == 3
        assert set(tree2.leaf_names()) == {"A", "B", "C"}

    def test_branch_lengths_preserved(self):
        nwk = "((A:1.5,B:2.3):0.7,C:4.1);"
        tree = parse_newick(nwk)
        # Find leaf A and check its branch length
        a_node = tree._name_to_leaf["A"]
        assert abs(a_node.branch_length - 1.5) < 1e-6

    def test_pipe_in_names(self):
        """IQ-TREE uses names like 'A0A1P8NVR5|org101286'."""
        nwk = "('A0A1P8NVR5|org101286':0.48,'P58917|org101291':0.41);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 2
        assert "A0A1P8NVR5|org101286" in tree.leaf_names()

    def test_unquoted_simple_names(self):
        nwk = "(A:1,B:2,C:3);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 3

    def test_trifurcating_resolved(self):
        """Trifurcating root (IQ-TREE style) resolved to binary."""
        nwk = "(A:1,B:2,C:3);"
        tree = parse_newick(nwk)
        assert len(tree.root.children) == 3  # trifurcating
        tree.resolve_polytomies()
        # Now binary: all internal nodes have exactly 2 children
        for node in tree.internals:
            assert len(node.children) == 2
        assert tree.n_leaves == 3
        assert set(tree.leaf_names()) == {"A", "B", "C"}

    def test_iqtree_real_tree(self):
        """Parse the actual IQ-TREE conotoxin tree (40 taxa), resolve trifurcation."""
        treefile = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding/results/iqtree_conotoxin.treefile")
        if not treefile.exists():
            pytest.skip("IQ-TREE reference tree not available")
        nwk = treefile.read_text().strip()
        tree = parse_newick(nwk)
        tree.resolve_polytomies()
        assert tree.n_leaves == 40
        for node in tree.internals:
            assert len(node.children) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestNewickIO -v 2>&1 | head -15`
Expected: FAIL (parse_newick not defined)

- [ ] **Step 3: Implement Newick parser and writer**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Newick I/O ────────────────────────────────────────────────────────────

def parse_newick(newick_str: str) -> Tree:
    """Parse a Newick string into a Tree. Handles quoted names (pipes etc)."""
    s = newick_str.strip().rstrip(";")
    node_id = [0]

    def _next_id():
        nid = node_id[0]
        node_id[0] += 1
        return nid

    def _parse(s: str, parent: Optional[TreeNode] = None) -> TreeNode:
        # Find children block if any
        children_part = ""
        label_part = s

        if s.startswith("("):
            # Find matching closing paren
            depth = 0
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        children_part = s[1:i]
                        label_part = s[i + 1:]
                        break

        # Parse label:branch_length
        name, branch_length = "", 0.0
        if ":" in label_part:
            name_part, bl_str = label_part.rsplit(":", 1)
            name = name_part.strip("'\"")
            try:
                branch_length = float(bl_str)
            except ValueError:
                branch_length = 0.0
        else:
            name = label_part.strip("'\"")

        node = TreeNode(id=_next_id(), name=name,
                        branch_length=branch_length, parent=parent)

        if children_part:
            # Split children on commas at depth 0
            child_strings = _split_top_level(children_part)
            for cs in child_strings:
                child = _parse(cs.strip(), parent=node)
                node.children.append(child)

        return node

    root = _parse(s)
    return Tree(root=root)


def _split_top_level(s: str) -> List[str]:
    """Split a string on commas that are not inside parentheses or quotes."""
    parts = []
    depth = 0
    in_quote = False
    current = []
    for ch in s:
        if ch in ("'", '"') and depth == 0:
            in_quote = not in_quote
            current.append(ch)
        elif ch == "(" and not in_quote:
            depth += 1
            current.append(ch)
        elif ch == ")" and not in_quote:
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0 and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def write_newick(tree: Tree) -> str:
    """Write tree to Newick string. Quotes names containing special chars."""
    def _needs_quoting(name: str) -> bool:
        return bool(re.search(r"[|,;:()\[\]'\s]", name))

    def _fmt_name(name: str) -> str:
        if not name:
            return ""
        if _needs_quoting(name):
            return f"'{name}'"
        return name

    def _write(node: TreeNode) -> str:
        if node.is_leaf():
            s = _fmt_name(node.name)
        else:
            children_str = ",".join(_write(c) for c in node.children)
            s = f"({children_str}){_fmt_name(node.name)}"
        if node.branch_length > 0 and not node.is_root():
            s += f":{node.branch_length:.10f}"
        return s

    return _write(tree.root) + ";"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestNewickIO -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add Newick parser and writer with quoted name support"
```

---

### Task 3: Neighbor-Joining builder

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write NJ tests**

```python
# Add to tests/test_embedding_phylo.py
# Also import: NJBuilder from _exp35

class TestNJBuilder:
    def test_four_taxon_textbook(self):
        """Saitou-Nei 1987 textbook example: known 4-taxon distance matrix."""
        # Distance matrix for ((A,B),(C,D))
        #     A    B    C    D
        D = np.array([
            [0,   5,   9,   9],
            [5,   0,   10,  10],
            [9,   10,  0,   8],
            [9,   10,  8,   0],
        ], dtype=np.float64)
        names = ["A", "B", "C", "D"]
        tree = NJBuilder.build(D, names)
        assert tree.n_leaves == 4
        assert set(tree.leaf_names()) == {"A", "B", "C", "D"}
        # A and B should be siblings (closest pair)
        a = tree._name_to_leaf["A"]
        b = tree._name_to_leaf["B"]
        assert a.parent is b.parent, "A and B should be siblings"

    def test_three_taxon(self):
        D = np.array([
            [0, 2, 4],
            [2, 0, 4],
            [4, 4, 0],
        ], dtype=np.float64)
        names = ["X", "Y", "Z"]
        tree = NJBuilder.build(D, names)
        assert tree.n_leaves == 3
        assert tree.total_branch_length() > 0

    def test_from_embeddings(self):
        """Build NJ tree from random embedding vectors."""
        rng = np.random.RandomState(42)
        embeddings = {"A": rng.randn(512), "B": rng.randn(512),
                      "C": rng.randn(512), "D": rng.randn(512)}
        tree = NJBuilder.from_embeddings(embeddings)
        assert tree.n_leaves == 4
        assert set(tree.leaf_names()) == set(embeddings.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestNJBuilder -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement NJBuilder**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Neighbor Joining ──────────────────────────────────────────────────────

class NJBuilder:
    """Saitou-Nei Neighbor-Joining tree from a distance matrix."""

    @staticmethod
    def build(dist_matrix: np.ndarray, names: List[str]) -> Tree:
        """Build NJ tree from N×N distance matrix and taxon names."""
        N = len(names)
        assert dist_matrix.shape == (N, N)
        D = dist_matrix.copy().astype(np.float64)

        node_id = [0]
        def _next_id():
            nid = node_id[0]; node_id[0] += 1; return nid

        # Active nodes
        nodes: List[TreeNode] = []
        for i, name in enumerate(names):
            nodes.append(TreeNode(id=_next_id(), name=name))

        active = list(range(N))

        while len(active) > 2:
            n = len(active)
            # Row sums of active submatrix
            row_sums = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    row_sums[i] += D[active[i], active[j]]

            # Q matrix
            best_q = float("inf")
            best_i, best_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    q = (n - 2) * D[active[i], active[j]] - row_sums[i] - row_sums[j]
                    if q < best_q:
                        best_q = q
                        best_i, best_j = i, j

            ai, aj = active[best_i], active[best_j]

            # Branch lengths to new node
            d_ij = D[ai, aj]
            if n > 2:
                bl_i = 0.5 * d_ij + (row_sums[best_i] - row_sums[best_j]) / (2 * (n - 2))
                bl_j = d_ij - bl_i
            else:
                bl_i = d_ij / 2
                bl_j = d_ij / 2
            # Clamp to positive
            bl_i = max(bl_i, 1e-10)
            bl_j = max(bl_j, 1e-10)

            nodes[ai].branch_length = bl_i
            nodes[aj].branch_length = bl_j

            new_node = TreeNode(id=_next_id(), children=[nodes[ai], nodes[aj]])
            nodes[ai].parent = new_node
            nodes[aj].parent = new_node

            # Update distance matrix: distance from new node to all others
            new_idx = len(nodes)
            nodes.append(new_node)

            # Expand D to accommodate new node
            old_size = D.shape[0]
            new_D = np.zeros((old_size + 1, old_size + 1))
            new_D[:old_size, :old_size] = D
            for k_idx in range(len(active)):
                if k_idx == best_i or k_idx == best_j:
                    continue
                ak = active[k_idx]
                d_new = 0.5 * (D[ai, ak] + D[aj, ak] - d_ij)
                new_D[old_size, ak] = d_new
                new_D[ak, old_size] = d_new
            D = new_D

            # Update active list
            new_active = [a for idx, a in enumerate(active) if idx != best_i and idx != best_j]
            new_active.append(old_size)
            active = new_active

        # Final two: join under root
        ai, aj = active[0], active[1]
        d_final = D[ai, aj] / 2
        nodes[ai].branch_length = max(d_final, 1e-10)
        nodes[aj].branch_length = max(d_final, 1e-10)

        root = TreeNode(id=_next_id(), children=[nodes[ai], nodes[aj]])
        nodes[ai].parent = root
        nodes[aj].parent = root

        return Tree(root=root)

    @staticmethod
    def from_embeddings(embeddings: Dict[str, np.ndarray]) -> Tree:
        """Build NJ tree from {name: vector} dict using Euclidean distances."""
        names = sorted(embeddings.keys())
        vecs = np.array([embeddings[n] for n in names])
        # Pairwise Euclidean distance matrix
        diff = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=-1))
        return NJBuilder.build(D, names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestNJBuilder -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add Neighbor-Joining tree builder (Saitou-Nei)"
```

---

### Task 4: Random tree generator

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write random tree tests**

```python
# Add to tests/test_embedding_phylo.py
# Also import: random_tree from _exp35

class TestRandomTree:
    def test_correct_n_leaves(self):
        names = ["A", "B", "C", "D", "E"]
        tree = random_tree(names, seed=42)
        assert tree.n_leaves == 5
        assert set(tree.leaf_names()) == set(names)

    def test_binary(self):
        """All internal nodes should have exactly 2 children."""
        tree = random_tree([f"t{i}" for i in range(10)], seed=123)
        for node in tree.internals:
            assert len(node.children) == 2

    def test_positive_branch_lengths(self):
        tree = random_tree(["A", "B", "C", "D"], seed=0)
        for node in tree.nodes:
            if not node.is_root():
                assert node.branch_length > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestRandomTree -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement random_tree**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Random Tree ───────────────────────────────────────────────────────────

def random_tree(names: List[str], seed: int = 42,
                bl_mean: float = 0.1) -> Tree:
    """Generate a random binary tree by random taxon addition.

    Branch lengths drawn from Exponential(1/bl_mean).
    """
    rng = np.random.RandomState(seed)
    node_id = [0]
    def _next_id():
        nid = node_id[0]; node_id[0] += 1; return nid

    shuffled = list(names)
    rng.shuffle(shuffled)

    # Start with first two taxa
    a = TreeNode(id=_next_id(), name=shuffled[0],
                 branch_length=rng.exponential(bl_mean))
    b = TreeNode(id=_next_id(), name=shuffled[1],
                 branch_length=rng.exponential(bl_mean))
    root = TreeNode(id=_next_id(), children=[a, b])
    a.parent = root
    b.parent = root
    tree = Tree(root=root)

    # Add remaining taxa one by one
    for name in shuffled[2:]:
        # Pick a random non-root edge to break
        candidates = [n for n in tree.nodes if not n.is_root()]
        target = candidates[rng.randint(len(candidates))]

        # Insert new internal node on this edge
        old_parent = target.parent
        new_internal = TreeNode(id=_next_id(),
                                branch_length=target.branch_length * rng.uniform(0.1, 0.9))
        target.branch_length = max(target.branch_length - new_internal.branch_length, 1e-10)

        new_leaf = TreeNode(id=_next_id(), name=name,
                            branch_length=rng.exponential(bl_mean))

        # Rewire
        old_parent.children = [new_internal if c is target else c for c in old_parent.children]
        new_internal.parent = old_parent
        new_internal.children = [target, new_leaf]
        target.parent = new_internal
        new_leaf.parent = new_internal

        tree._reindex()

    return tree
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestRandomTree -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add random binary tree generator"
```

---

## Chunk 2: Brownian Motion Likelihood

### Task 5: BM likelihood — vectorized pruning

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write BM likelihood tests**

```python
# Add to tests/test_embedding_phylo.py
# Also import: BMLikelihood from _exp35

class TestBMLikelihood:
    def test_three_taxon_analytical(self):
        """3-taxon tree, 1 dimension — compare to hand-calculated value.

        Tree: ((A:1,B:1):0,C:2)  (root has zero-length internal branch)
        Data: A=0, B=2, C=1, sigma2=1.0

        Internal node:
          s_L = 0 + 1 = 1 (v_A + t_A)
          s_R = 0 + 1 = 1 (v_B + t_B)
          logL_node1 = -0.5 * 1 * log(2*pi*1.0*(1+1)) - 0.5 * (0-2)^2 / (1.0*(1+1))
                     = -0.5 * log(4*pi) - 0.5 * 4/2
                     = -0.5 * log(4*pi) - 1.0

        Root node:
          v_internal = 1*1/(1+1) = 0.5
          s_L = 0.5 + 0 = 0.5 (v_internal + t_internal, t_internal=0 to root)
          s_R = 0 + 2 = 2 (v_C + t_C)
          mu_internal = (0*1 + 2*1) / (1+1) = 1.0
          logL_root = -0.5 * 1 * log(2*pi*1.0*(0.5+2)) - 0.5 * (1.0-1)^2 / (1.0*2.5)
                    = -0.5 * log(2.5*2*pi) - 0
                    = -0.5 * log(5*pi)

        Total = -0.5*log(4*pi) - 1.0 - 0.5*log(5*pi)
        """
        a = TreeNode(id=0, name="A", branch_length=1.0)
        b = TreeNode(id=1, name="B", branch_length=1.0)
        internal = TreeNode(id=3, children=[a, b], branch_length=0.0)
        a.parent = internal
        b.parent = internal
        c = TreeNode(id=2, name="C", branch_length=2.0)
        root = TreeNode(id=4, children=[internal, c])
        internal.parent = root
        c.parent = root
        tree = Tree(root=root)

        data = {"A": np.array([0.0]), "B": np.array([2.0]), "C": np.array([1.0])}
        bm = BMLikelihood()
        logL = bm.log_likelihood(tree, data, sigma2=1.0)

        expected = -0.5 * np.log(4 * np.pi) - 1.0 - 0.5 * np.log(5 * np.pi)
        assert abs(logL - expected) < 1e-10, f"Expected {expected}, got {logL}"

    def test_vectorized_matches_loop(self):
        """Multi-dim likelihood should equal sum of per-dim likelihoods."""
        rng = np.random.RandomState(42)
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        D = 64
        data = {name: rng.randn(D) for name in tree.leaf_names()}
        bm = BMLikelihood()
        sigma2 = 0.5

        logL_vectorized = bm.log_likelihood(tree, data, sigma2)

        # Compute per-dimension naively
        logL_loop = 0.0
        for d in range(D):
            data_1d = {name: np.array([vec[d]]) for name, vec in data.items()}
            logL_loop += bm.log_likelihood(tree, data_1d, sigma2)

        assert abs(logL_vectorized - logL_loop) < 1e-6

    def test_higher_sigma_higher_likelihood_for_spread_data(self):
        """If data is spread out, higher σ² should give higher likelihood."""
        tree = parse_newick("((A:1,B:1):1,C:1);")
        data = {"A": np.array([0.0]), "B": np.array([10.0]), "C": np.array([20.0])}
        bm = BMLikelihood()
        logL_low = bm.log_likelihood(tree, data, sigma2=0.01)
        logL_high = bm.log_likelihood(tree, data, sigma2=100.0)
        assert logL_high > logL_low
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestBMLikelihood -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement BMLikelihood**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Brownian Motion Likelihood ────────────────────────────────────────────

class BMLikelihood:
    """Felsenstein pruning for continuous characters under Brownian motion.

    Vectorized across D dimensions using numpy broadcasting.
    Flat (improper) prior on root state — root is integrated out analytically.
    """

    def log_likelihood(self, tree: Tree, data: Dict[str, np.ndarray],
                       sigma2: float) -> float:
        """Compute log P(data | tree, σ²) under independent BM per dimension.

        Args:
            tree: Rooted binary Tree
            data: {taxon_name: (D,) array} for each leaf
            sigma2: BM rate (shared across all dimensions)

        Returns:
            Total log-likelihood (float)
        """
        D = next(iter(data.values())).shape[0]  # dimensionality
        log_2pi = np.log(2.0 * np.pi)

        # Partial likelihoods stored per node: (mu, v, logL_contrib)
        # mu: (D,) array — weighted mean
        # v: scalar — accumulated variance
        # logL_contrib: scalar — log-likelihood contribution at this node
        partials: Dict[int, Tuple[np.ndarray, float, float]] = {}

        total_logL = 0.0

        for node in tree.postorder():
            if node.is_leaf():
                mu = data[node.name].astype(np.float64)
                partials[node.id] = (mu, 0.0, 0.0)
            else:
                assert len(node.children) == 2
                left, right = node.children

                mu_L, v_L, _ = partials[left.id]
                mu_R, v_R, _ = partials[right.id]

                s_L = v_L + left.branch_length
                s_R = v_R + right.branch_length
                s_total = s_L + s_R

                # Weighted mean (vectorized across D dims)
                mu = (mu_L * s_R + mu_R * s_L) / s_total

                # Combined partial variance (propagated up)
                v = (s_L * s_R) / s_total

                # Log-likelihood contribution at this node
                diff_sq = np.sum((mu_L - mu_R) ** 2)
                logL_node = (
                    -0.5 * D * (log_2pi + np.log(sigma2 * s_total))
                    - 0.5 * diff_sq / (sigma2 * s_total)
                )

                total_logL += logL_node
                partials[node.id] = (mu, v, logL_node)

        return total_logL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestBMLikelihood -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add vectorized Brownian motion likelihood (Felsenstein pruning)"
```

---

## Chunk 3: MCMC Proposals

### Task 6: Stochastic NNI proposal

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write proposal base class + stNNI tests**

```python
# Add to tests/test_embedding_phylo.py
# Also import: StochasticNNI from _exp35

class TestStochasticNNI:
    def test_preserves_leaf_set(self):
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        nni = StochasticNNI(seed=0)
        new_tree, log_hr = nni.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_symmetric_hastings_ratio(self):
        """NNI is symmetric, so log Hastings ratio should be 0."""
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        nni = StochasticNNI(seed=1)
        _, log_hr = nni.propose(tree)
        assert log_hr == 0.0

    def test_topology_changes(self):
        """At least some NNI proposals should change the topology."""
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        nni = StochasticNNI(seed=0)
        changed = 0
        for i in range(20):
            nni.rng = np.random.RandomState(i)
            new_tree, _ = nni.propose(tree)
            if write_newick(new_tree) != write_newick(tree):
                changed += 1
        assert changed > 0, "NNI should change topology sometimes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestStochasticNNI -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement StochasticNNI**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── MCMC Proposals ────────────────────────────────────────────────────────

class StochasticNNI:
    """Stochastic Nearest-Neighbor Interchange.

    Pick a random internal edge, swap one subtree from each side.
    Symmetric proposal → Hastings ratio = 1 (log HR = 0).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        """Return (new_tree, log_hastings_ratio)."""
        new_tree = tree.copy()

        # Find internal edges (edges where BOTH endpoints are internal)
        internal_edges = []
        for node in new_tree.internals:
            if not node.is_root() and not node.parent.is_root():
                internal_edges.append(node)
            elif not node.is_root():
                # Edge to root: only valid if node has 2 children
                # and root has 2 children (always true for binary)
                internal_edges.append(node)

        if not internal_edges:
            return new_tree, 0.0

        # Pick random internal node (the "lower" end of the edge)
        node = internal_edges[self.rng.randint(len(internal_edges))]
        parent = node.parent

        if len(node.children) < 2 or len(parent.children) < 2:
            return new_tree, 0.0

        # Pick one child of node and one sibling (other child of parent)
        child_idx = self.rng.randint(len(node.children))
        sibling_idx = [i for i, c in enumerate(parent.children) if c is not node]
        if not sibling_idx:
            return new_tree, 0.0
        sib_idx = sibling_idx[self.rng.randint(len(sibling_idx))]

        # Swap
        child = node.children[child_idx]
        sibling = parent.children[sib_idx]

        node.children[child_idx] = sibling
        parent.children[sib_idx] = child

        sibling.parent = node
        child.parent = parent

        new_tree._reindex()
        return new_tree, 0.0  # symmetric
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestStochasticNNI -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add stochastic NNI topology proposal"
```

---

### Task 7: Branch length and σ² proposals

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write branch length + sigma + tree length proposal tests**

```python
# Also import: BranchLengthMultiplier, TreeLengthMultiplier, SigmaMultiplier

class TestBranchLengthMultiplier:
    def test_positive_branch_length(self):
        tree = random_tree(["A", "B", "C"], seed=42)
        prop = BranchLengthMultiplier(seed=0)
        for _ in range(50):
            new_tree, _ = prop.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length > 0

    def test_hastings_ratio_nonzero(self):
        tree = random_tree(["A", "B", "C"], seed=42)
        prop = BranchLengthMultiplier(seed=0)
        _, log_hr = prop.propose(tree)
        # Log HR for multiplier proposal = log(new_bl / old_bl) = log(multiplier)
        assert log_hr != 0.0  # very unlikely to be exactly 0

class TestTreeLengthMultiplier:
    def test_scales_all_branches(self):
        tree = random_tree(["A", "B", "C", "D"], seed=42)
        prop = TreeLengthMultiplier(seed=0, lambda_=0.1)
        old_total = tree.total_branch_length()
        new_tree, _ = prop.propose(tree)
        new_total = new_tree.total_branch_length()
        assert new_total != old_total  # should change
        # All branches should scale by same factor
        n_branches = sum(1 for n in tree.nodes if not n.is_root())
        ratio = new_total / old_total
        for old_n, new_n in zip(tree.postorder(), new_tree.postorder()):
            if not old_n.is_root():
                assert abs(new_n.branch_length / old_n.branch_length - ratio) < 1e-10

class TestSigmaMultiplier:
    def test_positive(self):
        prop = SigmaMultiplier(seed=0)
        for _ in range(50):
            new_sigma2, log_hr = prop.propose_sigma(1.0)
            assert new_sigma2 > 0

    def test_hastings_ratio(self):
        prop = SigmaMultiplier(seed=0)
        sigma2 = 1.0
        new_sigma2, log_hr = prop.propose_sigma(sigma2)
        expected_hr = np.log(new_sigma2 / sigma2)
        assert abs(log_hr - expected_hr) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestBranchLengthMultiplier tests/test_embedding_phylo.py::TestTreeLengthMultiplier tests/test_embedding_phylo.py::TestSigmaMultiplier -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement proposals**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
class BranchLengthMultiplier:
    """Log-uniform multiplier proposal for a single random branch length.

    t' = t * exp(lambda * (U - 0.5)), U ~ Uniform(0,1)
    Hastings ratio = t'/t = exp(lambda * (U - 0.5))
    """

    def __init__(self, seed: int = 42, lambda_: float = 2.0):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        # Pick random non-root edge
        candidates = [n for n in new_tree.nodes if not n.is_root()]
        if not candidates:
            return new_tree, 0.0
        node = candidates[self.rng.randint(len(candidates))]
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        node.branch_length *= multiplier
        log_hr = math.log(multiplier)  # Jacobian
        return new_tree, log_hr


class TreeLengthMultiplier:
    """Scale ALL branch lengths by the same multiplier."""

    def __init__(self, seed: int = 42, lambda_: float = 0.5):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        n_branches = 0
        for node in new_tree.nodes:
            if not node.is_root():
                node.branch_length *= multiplier
                n_branches += 1
        log_hr = n_branches * math.log(multiplier)
        return new_tree, log_hr


class SigmaMultiplier:
    """Log-uniform multiplier for BM rate σ²."""

    def __init__(self, seed: int = 42, lambda_: float = 1.0):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose_sigma(self, sigma2: float) -> Tuple[float, float]:
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        new_sigma2 = sigma2 * multiplier
        log_hr = math.log(multiplier)
        return new_sigma2, log_hr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestBranchLengthMultiplier tests/test_embedding_phylo.py::TestTreeLengthMultiplier tests/test_embedding_phylo.py::TestSigmaMultiplier -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add branch length, tree length, and sigma2 multiplier proposals"
```

---

### Task 8: Fixed-radius SPR proposal

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write SPR tests**

```python
# Also import: SubtreePruneRegraft from _exp35

class TestSubtreePruneRegraft:
    def test_preserves_leaf_set(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        spr = SubtreePruneRegraft(seed=0)
        new_tree, _ = spr.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_binary_tree_preserved(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        spr = SubtreePruneRegraft(seed=1)
        new_tree, _ = spr.propose(tree)
        for node in new_tree.internals:
            assert len(node.children) == 2, f"Node {node.id} has {len(node.children)} children"

    def test_some_proposals_change_topology(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        changed = 0
        for i in range(30):
            spr = SubtreePruneRegraft(seed=i)
            new_tree, _ = spr.propose(tree)
            if write_newick(new_tree) != write_newick(tree):
                changed += 1
        assert changed > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestExtendingSPR -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement SubtreePruneRegraft**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
class SubtreePruneRegraft:
    """Fixed-radius Subtree Prune-and-Regraft.

    1. Pick a random subtree to prune
    2. Pick a random regraft edge uniformly from all valid edges
    3. Hastings ratio = 1 (symmetric: uniform selection in both directions)

    Uses uniform edge selection (not extending walk) to guarantee
    correct detailed balance without complex HR calculation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()

        # Pick a random non-root node to prune
        prune_candidates = [n for n in new_tree.nodes
                            if not n.is_root() and not n.parent.is_root()
                            or (not n.is_root() and n.parent.is_root()
                                and len(n.parent.children) == 2)]
        if len(prune_candidates) < 2:
            return new_tree, 0.0

        prune_node = prune_candidates[self.rng.randint(len(prune_candidates))]
        prune_parent = prune_node.parent

        # Detach pruned subtree: sibling takes prune_parent's place
        siblings = [c for c in prune_parent.children if c is not prune_node]
        if not siblings:
            return new_tree, 0.0
        sibling = siblings[0]

        grandparent = prune_parent.parent
        if grandparent is None:
            sibling.parent = None
            sibling.branch_length += prune_parent.branch_length
            new_tree.root = sibling
        else:
            sibling.branch_length += prune_parent.branch_length
            sibling.parent = grandparent
            grandparent.children = [sibling if c is prune_parent else c
                                    for c in grandparent.children]
        new_tree._reindex()

        # Collect pruned subtree IDs (to exclude from regraft targets)
        pruned_ids = set()
        stack = [prune_node]
        while stack:
            n = stack.pop()
            pruned_ids.add(n.id)
            stack.extend(n.children)

        # Pick uniform random regraft edge (any non-root node not in pruned subtree)
        regraft_candidates = [n for n in new_tree.nodes
                              if not n.is_root() and n.id not in pruned_ids]
        if not regraft_candidates:
            return tree.copy(), 0.0

        regraft_target = regraft_candidates[self.rng.randint(len(regraft_candidates))]
        regraft_parent = regraft_target.parent

        # Insert prune_parent on the regraft edge
        split_frac = self.rng.uniform(0.1, 0.9)
        old_bl = regraft_target.branch_length
        prune_parent.branch_length = old_bl * split_frac
        regraft_target.branch_length = old_bl * (1 - split_frac)

        regraft_parent.children = [prune_parent if c is regraft_target else c
                                   for c in regraft_parent.children]
        prune_parent.parent = regraft_parent
        prune_parent.children = [regraft_target, prune_node]
        regraft_target.parent = prune_parent
        prune_node.parent = prune_parent

        new_tree._reindex()

        # Symmetric: uniform choice of prune node and regraft edge
        # Forward and reverse have same number of candidates → log HR = 0
        return new_tree, 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestSubtreePruneRegraft -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add subtree prune-regraft topology proposal"
```

---

### Task 9: ProposalMixer with auto-tuning

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write ProposalMixer tests**

```python
# Also import: ProposalMixer from _exp35

class TestProposalMixer:
    def test_weighted_selection(self):
        """Proposals selected roughly proportional to weights."""
        mixer = ProposalMixer(
            proposal_names=["nni", "bl", "sigma"],
            weights=[6.0, 9.0, 1.0],
            seed=42,
        )
        counts = {"nni": 0, "bl": 0, "sigma": 0}
        for _ in range(10000):
            name = mixer.select()
            counts[name] += 1
        # bl should be most common (~56%), nni ~38%, sigma ~6%
        assert counts["bl"] > counts["nni"] > counts["sigma"]

    def test_acceptance_tracking(self):
        mixer = ProposalMixer(
            proposal_names=["nni", "bl"],
            weights=[1.0, 1.0],
            seed=42,
        )
        mixer.record_acceptance("nni", True)
        mixer.record_acceptance("nni", True)
        mixer.record_acceptance("nni", False)
        assert abs(mixer.acceptance_rate("nni") - 2.0 / 3.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestProposalMixer -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement ProposalMixer**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Proposal Mixer ────────────────────────────────────────────────────────

class ProposalMixer:
    """Weighted proposal selection with acceptance tracking and auto-tuning."""

    def __init__(self, proposal_names: List[str], weights: List[float],
                 seed: int = 42):
        self.names = proposal_names
        self.weights = np.array(weights, dtype=np.float64)
        self.probs = self.weights / self.weights.sum()
        self.rng = np.random.RandomState(seed)

        self._accepted: Dict[str, int] = {n: 0 for n in self.names}
        self._total: Dict[str, int] = {n: 0 for n in self.names}

    def select(self) -> str:
        return self.names[self.rng.choice(len(self.names), p=self.probs)]

    def record_acceptance(self, name: str, accepted: bool):
        self._total[name] += 1
        if accepted:
            self._accepted[name] += 1

    def acceptance_rate(self, name: str) -> float:
        if self._total[name] == 0:
            return 0.0
        return self._accepted[name] / self._total[name]

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "total": self._total[name],
                "accepted": self._accepted[name],
                "rate": self.acceptance_rate(name),
            }
            for name in self.names
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestProposalMixer -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add proposal mixer with weighted selection and acceptance tracking"
```

---

## Chunk 4: MCMC Engine + MC3 + Multi-Run

### Task 10: Single-chain MCMC

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write MCMCChain tests**

```python
# Also import: MCMCChain from _exp35

class TestMCMCChain:
    def test_synthetic_convergence(self):
        """Run MCMC on synthetic 5-taxon BM data. Should find reasonable tree."""
        rng = np.random.RandomState(42)
        # True tree
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,(C:0.4,(D:0.2,E:0.2):0.2):0.3);")
        # Simulate BM data on true tree
        sigma2_true = 1.0
        D = 32
        data = simulate_bm(true_tree, sigma2_true, D, seed=42)

        chain = MCMCChain(
            data=data,
            start_tree=random_tree(list(data.keys()), seed=99),
            sigma2_init=1.0,
            n_generations=5000,
            sample_freq=100,
            seed=42,
        )
        chain.run()

        # Chain should have samples
        assert len(chain.sampled_trees) > 0
        assert len(chain.sampled_logL) > 0
        # Log-likelihood should improve from start
        assert chain.sampled_logL[-1] > chain.sampled_logL[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestMCMCChain -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement simulate_bm and MCMCChain**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── BM Simulation ─────────────────────────────────────────────────────────

def simulate_bm(tree: Tree, sigma2: float, D: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Simulate Brownian motion data on a tree. Returns {leaf_name: (D,) array}."""
    rng = np.random.RandomState(seed)
    node_values: Dict[int, np.ndarray] = {}

    # Root value
    node_values[tree.root.id] = rng.randn(D) * 0.1  # near zero

    # Preorder traversal: root → leaves
    stack = [tree.root]
    visited = set()
    while stack:
        node = stack[-1]
        if node.id in visited:
            stack.pop()
            continue
        # Ensure parent is computed
        if not node.is_root() and node.parent.id not in node_values:
            stack.pop()
            continue
        visited.add(node.id)
        stack.pop()

        if not node.is_root():
            parent_val = node_values[node.parent.id]
            noise = rng.randn(D) * math.sqrt(sigma2 * node.branch_length)
            node_values[node.id] = parent_val + noise

        for child in reversed(node.children):
            if child.id not in visited:
                stack.append(child)

    return {node.name: node_values[node.id] for node in tree.leaves}


# ── MCMC Chain ────────────────────────────────────────────────────────────

class MCMCChain:
    """Single Metropolis-Hastings MCMC chain for BM phylogenetics."""

    def __init__(self, data: Dict[str, np.ndarray],
                 start_tree: Tree, sigma2_init: float = 1.0,
                 n_generations: int = 100_000, sample_freq: int = 500,
                 beta: float = 1.0, seed: int = 42):
        self.data = data
        self.tree = start_tree
        self.sigma2 = sigma2_init
        self.n_generations = n_generations
        self.sample_freq = sample_freq
        self.beta = beta  # heating parameter (1.0 = cold chain)

        self.bm = BMLikelihood()
        self.logL = self.bm.log_likelihood(self.tree, self.data, self.sigma2)

        # Log prior
        self.log_prior = self._compute_log_prior(self.tree, self.sigma2)

        # Proposals
        self.nni = StochasticNNI(seed=seed)
        self.spr = SubtreePruneRegraft(seed=seed + 1)
        self.bl_mult = BranchLengthMultiplier(seed=seed + 2)
        self.tl_mult = TreeLengthMultiplier(seed=seed + 3)
        self.sigma_mult = SigmaMultiplier(seed=seed + 4)

        self.mixer = ProposalMixer(
            proposal_names=["nni", "spr", "bl", "tl", "sigma"],
            weights=[6.0, 6.0, 9.0, 1.0, 1.0],
            seed=seed + 5,
        )
        self.accept_rng = np.random.RandomState(seed + 6)

        # Samples (from cold chain only)
        self.sampled_trees: List[str] = []  # Newick strings
        self.sampled_logL: List[float] = []
        self.sampled_sigma2: List[float] = []
        self.sampled_tree_length: List[float] = []

    @staticmethod
    def _compute_log_prior(tree: Tree, sigma2: float) -> float:
        """Log prior: Exp(10) on branch lengths, LogNormal(0,1) on σ²."""
        log_p = 0.0
        rate = 10.0
        for node in tree.nodes:
            if not node.is_root():
                log_p += math.log(rate) - rate * node.branch_length
        # σ² ~ LogNormal(0, 1)
        log_p += -0.5 * (math.log(sigma2)) ** 2 - math.log(sigma2)
        return log_p

    def _step(self):
        proposal_name = self.mixer.select()

        if proposal_name == "nni":
            new_tree, log_hr = self.nni.propose(self.tree)
            new_sigma2 = self.sigma2
        elif proposal_name == "spr":
            new_tree, log_hr = self.spr.propose(self.tree)
            new_sigma2 = self.sigma2
        elif proposal_name == "bl":
            new_tree, log_hr = self.bl_mult.propose(self.tree)
            new_sigma2 = self.sigma2
        elif proposal_name == "tl":
            new_tree, log_hr = self.tl_mult.propose(self.tree)
            new_sigma2 = self.sigma2
        elif proposal_name == "sigma":
            new_tree = self.tree
            new_sigma2, log_hr = self.sigma_mult.propose_sigma(self.sigma2)
        else:
            return

        new_logL = self.bm.log_likelihood(new_tree, self.data, new_sigma2)

        # Compute new prior (static method — no shared state issues)
        old_log_prior = self.log_prior
        new_log_prior = self._compute_log_prior(new_tree, new_sigma2)

        # Metropolis-Hastings acceptance
        log_alpha = self.beta * (new_logL - self.logL) + (new_log_prior - old_log_prior) + log_hr

        accepted = False
        if log_alpha >= 0 or math.log(self.accept_rng.uniform()) < log_alpha:
            self.tree = new_tree
            self.sigma2 = new_sigma2
            self.logL = new_logL
            self.log_prior = new_log_prior
            accepted = True

        self.mixer.record_acceptance(proposal_name, accepted)

    def run(self):
        for gen in range(self.n_generations):
            self._step()
            if (gen + 1) % self.sample_freq == 0 and self.beta == 1.0:
                self.sampled_trees.append(write_newick(self.tree))
                self.sampled_logL.append(self.logL)
                self.sampled_sigma2.append(self.sigma2)
                self.sampled_tree_length.append(self.tree.total_branch_length())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestMCMCChain -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add single-chain MCMC with BM likelihood and all proposals"
```

---

### Task 11: MC3 runner + multi-run orchestrator

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write MC3 and multi-run tests**

```python
# Also import: MC3Runner, MultiRunOrchestrator from _exp35

class TestMC3Runner:
    def test_heated_chains(self):
        """MC3 with 2 chains should run without error."""
        rng = np.random.RandomState(42)
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(true_tree, 1.0, 16, seed=42)

        runner = MC3Runner(
            data=data,
            n_chains=2,
            n_generations=1000,
            sample_freq=100,
            swap_freq=50,
            delta=0.1,
            start_tree=random_tree(list(data.keys()), seed=0),
            seed=42,
        )
        runner.run()
        # Cold chain should have samples
        assert len(runner.cold_chain.sampled_trees) > 0

class TestMultiRunOrchestrator:
    def test_two_runs(self):
        """Two independent runs should both produce samples."""
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(true_tree, 1.0, 16, seed=42)

        orch = MultiRunOrchestrator(
            data=data,
            n_runs=2,
            n_chains=2,
            n_generations=1000,
            sample_freq=100,
            seed=42,
        )
        results = orch.run()
        assert len(results) == 2
        for r in results:
            assert len(r["sampled_trees"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestMC3Runner tests/test_embedding_phylo.py::TestMultiRunOrchestrator -v 2>&1 | head -10`
Expected: FAIL

- [ ] **Step 3: Implement MC3Runner and MultiRunOrchestrator**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── MC3 Runner ────────────────────────────────────────────────────────────

class MC3Runner:
    """Metropolis-Coupled MCMC: N heated chains with periodic swap attempts."""

    def __init__(self, data: Dict[str, np.ndarray],
                 n_chains: int = 4, n_generations: int = 100_000,
                 sample_freq: int = 500, swap_freq: int = 100,
                 delta: float = 0.1,
                 start_tree: Optional[Tree] = None,
                 seed: int = 42):
        self.n_chains = n_chains
        self.n_generations = n_generations
        self.swap_freq = swap_freq
        self.swap_rng = np.random.RandomState(seed + 100)

        # Heating: β_i = 1 / (1 + i * δ)
        betas = [1.0 / (1.0 + i * delta) for i in range(n_chains)]

        if start_tree is None:
            start_tree = random_tree(sorted(data.keys()), seed=seed)

        self.chains: List[MCMCChain] = []
        for i in range(n_chains):
            chain = MCMCChain(
                data=data,
                start_tree=start_tree.copy(),
                sigma2_init=1.0,
                n_generations=0,  # we drive generations externally
                sample_freq=sample_freq,
                beta=betas[i],
                seed=seed + i * 10,
            )
            self.chains.append(chain)

        self.cold_chain = self.chains[0]
        self.n_swaps_attempted = 0
        self.n_swaps_accepted = 0

    def run(self):
        for gen in range(self.n_generations):
            # Step all chains
            for chain in self.chains:
                chain._step()

            # Sample from cold chain
            if (gen + 1) % self.cold_chain.sample_freq == 0:
                self.cold_chain.sampled_trees.append(write_newick(self.cold_chain.tree))
                self.cold_chain.sampled_logL.append(self.cold_chain.logL)
                self.cold_chain.sampled_sigma2.append(self.cold_chain.sigma2)
                self.cold_chain.sampled_tree_length.append(
                    self.cold_chain.tree.total_branch_length())

            # MC3 swap attempt
            if (gen + 1) % self.swap_freq == 0 and self.n_chains > 1:
                self._attempt_swap()

    def _attempt_swap(self):
        """Attempt to swap adjacent-temperature chains."""
        i = self.swap_rng.randint(self.n_chains - 1)
        j = i + 1

        chain_i = self.chains[i]
        chain_j = self.chains[j]

        log_alpha = (chain_i.beta - chain_j.beta) * (chain_j.logL - chain_i.logL)
        self.n_swaps_attempted += 1

        if log_alpha >= 0 or math.log(self.swap_rng.uniform()) < log_alpha:
            # Swap states
            chain_i.tree, chain_j.tree = chain_j.tree, chain_i.tree
            chain_i.sigma2, chain_j.sigma2 = chain_j.sigma2, chain_i.sigma2
            chain_i.logL, chain_j.logL = chain_j.logL, chain_i.logL
            chain_i.log_prior, chain_j.log_prior = chain_j.log_prior, chain_i.log_prior
            self.n_swaps_accepted += 1


# ── Multi-Run Orchestrator ────────────────────────────────────────────────

def _run_single_mc3(args: dict) -> dict:
    """Worker function for ProcessPoolExecutor. Runs one MC3 analysis."""
    runner = MC3Runner(
        data=args["data"],
        n_chains=args["n_chains"],
        n_generations=args["n_generations"],
        sample_freq=args["sample_freq"],
        swap_freq=args["swap_freq"],
        delta=args["delta"],
        start_tree=args["start_tree"],
        seed=args["seed"],
    )
    runner.run()
    return {
        "sampled_trees": runner.cold_chain.sampled_trees,
        "sampled_logL": runner.cold_chain.sampled_logL,
        "sampled_sigma2": runner.cold_chain.sampled_sigma2,
        "sampled_tree_length": runner.cold_chain.sampled_tree_length,
        "acceptance_summary": runner.cold_chain.mixer.summary(),
        "n_swaps_attempted": runner.n_swaps_attempted,
        "n_swaps_accepted": runner.n_swaps_accepted,
    }


class MultiRunOrchestrator:
    """Run M independent MC3 analyses in parallel."""

    def __init__(self, data: Dict[str, np.ndarray],
                 n_runs: int = 2, n_chains: int = 4,
                 n_generations: int = 100_000, sample_freq: int = 500,
                 swap_freq: int = 100, delta: float = 0.1,
                 seed: int = 42, max_workers: Optional[int] = None):
        self.data = data
        self.n_runs = n_runs
        self.n_chains = n_chains
        self.n_generations = n_generations
        self.sample_freq = sample_freq
        self.swap_freq = swap_freq
        self.delta = delta
        self.seed = seed
        self.max_workers = max_workers or n_runs

        # Build starting trees: run 0 = NJ, rest = random
        names = sorted(data.keys())
        self.start_trees: List[Tree] = []
        nj_tree = NJBuilder.from_embeddings(data)
        self.start_trees.append(nj_tree)
        for i in range(1, n_runs):
            self.start_trees.append(random_tree(names, seed=seed + i * 1000))

    def run(self) -> List[dict]:
        args_list = []
        for i in range(self.n_runs):
            args_list.append({
                "data": self.data,
                "n_chains": self.n_chains,
                "n_generations": self.n_generations,
                "sample_freq": self.sample_freq,
                "swap_freq": self.swap_freq,
                "delta": self.delta,
                "start_tree": self.start_trees[i],
                "seed": self.seed + i * 100,
            })

        if self.n_runs == 1:
            return [_run_single_mc3(args_list[0])]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_run_single_mc3, args_list))
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestMC3Runner tests/test_embedding_phylo.py::TestMultiRunOrchestrator -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add MC3 heated chains and multi-run parallel orchestrator"
```

---

### Task 12: Convergence diagnostics (ASDSF, ESS, PSRF)

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write diagnostics tests**

```python
# Also import: Diagnostics from _exp35

class TestDiagnostics:
    def test_ess_perfect_iid(self):
        """ESS of i.i.d. samples should equal sample size."""
        rng = np.random.RandomState(42)
        samples = rng.randn(1000)
        ess = Diagnostics.effective_sample_size(samples)
        # Should be close to 1000 (within 20%)
        assert ess > 800

    def test_ess_correlated(self):
        """ESS of highly correlated samples should be much less than N."""
        rng = np.random.RandomState(42)
        x = np.cumsum(rng.randn(1000))  # random walk — highly correlated
        ess = Diagnostics.effective_sample_size(x)
        assert ess < 100  # much less than 1000

    def test_psrf_identical_chains(self):
        """PSRF of identical chains should be ~1.0."""
        chain1 = np.random.RandomState(42).randn(500)
        chain2 = np.random.RandomState(43).randn(500)
        psrf = Diagnostics.psrf([chain1, chain2])
        assert psrf < 1.2

    def test_asdsf_identical_trees(self):
        """ASDSF of identical tree sets should be 0."""
        trees1 = ["((A,B),C);"] * 100
        trees2 = ["((A,B),C);"] * 100
        asdsf = Diagnostics.asdsf([trees1, trees2])
        assert asdsf == 0.0

    def test_asdsf_different_trees(self):
        """ASDSF of different tree sets should be > 0."""
        trees1 = ["((A,B),C);"] * 100
        trees2 = ["((A,C),B);"] * 100
        asdsf = Diagnostics.asdsf([trees1, trees2])
        assert asdsf > 0

    def test_get_splits(self):
        """Verify bipartition extraction from a tree."""
        tree = parse_newick("((A,B),(C,D));")
        splits = Diagnostics._get_splits(tree)
        # Should have splits for each internal edge
        assert len(splits) > 0
        # AB|CD should be a split
        assert frozenset({"A", "B"}) in splits or frozenset({"C", "D"}) in splits
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestDiagnostics -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement Diagnostics**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Convergence Diagnostics ───────────────────────────────────────────────

class Diagnostics:
    """ASDSF, ESS, PSRF convergence diagnostics (from ExaBayes)."""

    @staticmethod
    def effective_sample_size(samples: np.ndarray) -> float:
        """ESS via autocorrelation (Geyer's initial monotone sequence)."""
        n = len(samples)
        if n < 10:
            return float(n)
        x = samples - np.mean(samples)
        var = np.var(x, ddof=1)
        if var < 1e-30:
            return float(n)

        # Autocorrelation via FFT
        fft = np.fft.fft(x, n=2 * n)
        acf = np.real(np.fft.ifft(fft * np.conj(fft)))[:n] / (var * n)

        # Sum autocorrelations in pairs until sum goes negative
        tau = 1.0
        for i in range(1, n // 2):
            pair_sum = acf[2 * i - 1] + acf[2 * i]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum
        return n / tau

    @staticmethod
    def psrf(chains: List[np.ndarray]) -> float:
        """Potential Scale Reduction Factor (Gelman-Rubin R-hat)."""
        m = len(chains)
        n = min(len(c) for c in chains)
        if n < 2 or m < 2:
            return float("inf")

        chains = [c[:n] for c in chains]
        chain_means = [np.mean(c) for c in chains]
        chain_vars = [np.var(c, ddof=1) for c in chains]

        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)  # between-chain
        W = np.mean(chain_vars)  # within-chain

        if W < 1e-30:
            return 1.0

        var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B
        return math.sqrt(var_hat / W)

    @staticmethod
    def _get_splits(tree: Tree) -> set:
        """Get bipartition set (splits) for an unrooted tree.

        Each split is a frozenset of the smaller partition's leaf names.
        """
        all_leaves = frozenset(tree.leaf_names())
        splits = set()

        for node in tree.postorder():
            if node.is_leaf() or node.is_root():
                continue
            # Leaves below this node
            below = set()
            stack = [node]
            while stack:
                n = stack.pop()
                if n.is_leaf():
                    below.add(n.name)
                else:
                    stack.extend(n.children)
            below = frozenset(below)
            complement = all_leaves - below
            # Use the smaller side as canonical
            split = min(below, complement, key=len)
            if len(split) > 0 and len(split) < len(all_leaves):
                splits.add(split)
        return splits

    @staticmethod
    def asdsf(tree_sets: List[List[str]], burnin_frac: float = 0.25,
              min_freq: float = 0.05) -> float:
        """Average Standard Deviation of Split Frequencies across runs.

        Args:
            tree_sets: List of [Newick strings] per run
            burnin_frac: fraction of samples to discard
            min_freq: ignore splits below this frequency
        """
        # Parse trees and collect split frequencies per run
        split_freqs_per_run: List[Dict[frozenset, float]] = []

        for trees_nwk in tree_sets:
            n = len(trees_nwk)
            start = int(n * burnin_frac)
            trees_post_burnin = trees_nwk[start:]
            n_post = len(trees_post_burnin)
            if n_post == 0:
                split_freqs_per_run.append({})
                continue

            split_counts: Dict[frozenset, int] = {}
            for nwk in trees_post_burnin:
                tree = parse_newick(nwk)
                for split in Diagnostics._get_splits(tree):
                    split_counts[split] = split_counts.get(split, 0) + 1

            split_freqs = {s: c / n_post for s, c in split_counts.items()}
            split_freqs_per_run.append(split_freqs)

        # Collect all splits
        all_splits = set()
        for sf in split_freqs_per_run:
            all_splits.update(sf.keys())

        if not all_splits:
            return 0.0

        # ASDSF
        sds = []
        for split in all_splits:
            freqs = [sf.get(split, 0.0) for sf in split_freqs_per_run]
            mean_freq = np.mean(freqs)
            if mean_freq < min_freq:
                continue
            sd = np.std(freqs, ddof=0)
            sds.append(sd)

        return float(np.mean(sds)) if sds else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestDiagnostics -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add convergence diagnostics — ASDSF, ESS, PSRF"
```

---

## Chunk 5: Consensus, RF Distance, Benchmark, Main

### Task 13: Consensus tree builder + Robinson-Foulds distance

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`
- Modify: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write consensus and RF tests**

```python
# Also import: ConsensusBuilder, robinson_foulds from _exp35

class TestConsensusBuilder:
    def test_unanimous_topology(self):
        """If all sampled trees have the same topology, consensus = that topology."""
        trees = ["((A,B),(C,D));"] * 50
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        assert consensus.n_leaves == 4
        splits = Diagnostics._get_splits(consensus)
        expected = Diagnostics._get_splits(parse_newick("((A,B),(C,D));"))
        assert splits == expected

    def test_majority_wins(self):
        """The more frequent topology should dominate the consensus."""
        trees = ["((A,B),(C,D));"] * 70 + ["((A,C),(B,D));"] * 30
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        splits = Diagnostics._get_splits(consensus)
        # AB|CD should be in consensus (70% support)
        ab = frozenset({"A", "B"})
        cd = frozenset({"C", "D"})
        assert ab in splits or cd in splits


class TestRobinsonFoulds:
    def test_identical_trees(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,B),(C,D));")
        rf = robinson_foulds(t1, t2)
        assert rf == 0

    def test_maximally_different(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,C),(B,D));")
        rf = robinson_foulds(t1, t2)
        assert rf > 0

    def test_symmetric(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,C),(B,D));")
        assert robinson_foulds(t1, t2) == robinson_foulds(t2, t1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestConsensusBuilder tests/test_embedding_phylo.py::TestRobinsonFoulds -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement ConsensusBuilder and robinson_foulds**

Add to `experiments/35_embedding_phylogenetics.py`:

```python
# ── Consensus Tree ────────────────────────────────────────────────────────

class ConsensusBuilder:
    """Majority-rule consensus tree from posterior sample."""

    @staticmethod
    def majority_rule(trees_nwk: List[str], burnin_frac: float = 0.25,
                      threshold: float = 0.5) -> Tree:
        """Build majority-rule consensus tree.

        Args:
            trees_nwk: Newick strings from posterior sample
            burnin_frac: fraction to discard
            threshold: minimum split frequency to include (default 0.5 = majority)
        """
        n = len(trees_nwk)
        start = int(n * burnin_frac)
        post_trees = trees_nwk[start:]
        n_post = len(post_trees)

        if n_post == 0:
            return parse_newick(trees_nwk[-1])

        # Count split frequencies
        split_counts: Dict[frozenset, int] = {}
        leaf_names = None

        for nwk in post_trees:
            tree = parse_newick(nwk)
            if leaf_names is None:
                leaf_names = set(tree.leaf_names())
            for split in Diagnostics._get_splits(tree):
                split_counts[split] = split_counts.get(split, 0) + 1

        # Filter to majority splits
        majority_splits = []
        for split, count in split_counts.items():
            freq = count / n_post
            if freq >= threshold:
                majority_splits.append((split, freq))

        # Sort by size (largest first) for greedy tree building
        majority_splits.sort(key=lambda x: len(x[0]), reverse=True)

        # Build tree greedily: add compatible splits
        all_leaves = sorted(leaf_names)
        node_id = [0]
        def _next_id():
            nid = node_id[0]; node_id[0] += 1; return nid

        # Start with star tree
        leaf_nodes = {}
        for name in all_leaves:
            leaf_nodes[name] = TreeNode(id=_next_id(), name=name, branch_length=0.01)

        # Greedily add compatible splits
        groups = [{name} for name in all_leaves]  # each leaf is its own group
        parent_map: Dict[str, TreeNode] = {n: leaf_nodes[n] for n in all_leaves}

        for split, freq in majority_splits:
            split_set = set(split)
            # Check compatibility: split should be a union of existing groups
            matching_groups = [g for g in groups if g & split_set]
            if all(g <= split_set for g in matching_groups) and len(matching_groups) >= 2:
                # Create internal node grouping these
                children = []
                for g in matching_groups:
                    # Find representative node for this group
                    rep = list(g)[0]
                    children.append(parent_map[rep])

                new_node = TreeNode(id=_next_id(), children=children,
                                    branch_length=0.01)
                for child in children:
                    child.parent = new_node

                # Merge groups
                merged = set()
                new_groups = []
                for g in groups:
                    if g <= split_set:
                        merged.update(g)
                    else:
                        new_groups.append(g)
                new_groups.append(merged)
                groups = new_groups

                # Update parent map
                for name in merged:
                    parent_map[name] = new_node

        # Final root: join remaining groups
        remaining_nodes = []
        seen = set()
        for g in groups:
            rep = list(g)[0]
            node = parent_map[rep]
            if id(node) not in seen:
                remaining_nodes.append(node)
                seen.add(id(node))

        if len(remaining_nodes) == 1:
            root = remaining_nodes[0]
        else:
            root = TreeNode(id=_next_id(), children=remaining_nodes)
            for child in remaining_nodes:
                child.parent = root

        return Tree(root=root)


# ── Robinson-Foulds Distance ─────────────────────────────────────────────

def robinson_foulds(tree1: Tree, tree2: Tree) -> int:
    """Symmetric (unweighted) Robinson-Foulds distance between two trees.

    RF = |splits(T1) △ splits(T2)| = |splits(T1) - splits(T2)| + |splits(T2) - splits(T1)|
    """
    splits1 = Diagnostics._get_splits(tree1)
    splits2 = Diagnostics._get_splits(tree2)
    return len(splits1.symmetric_difference(splits2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestConsensusBuilder tests/test_embedding_phylo.py::TestRobinsonFoulds -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add majority-rule consensus tree and Robinson-Foulds distance"
```

---

### Task 14: Experiment main — conotoxin benchmark

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py`

- [ ] **Step 1: Write the main experiment runner**

Add to bottom of `experiments/35_embedding_phylogenetics.py`:

```python
# ── Experiment Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PROJ_ROOT = Path(__file__).resolve().parent.parent
    SPECIES_ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding")
    sys.path.insert(0, str(PROJ_ROOT))

    import h5py

    DATA_DIR = PROJ_ROOT / "data"
    RESULTS_DIR = PROJ_ROOT / "results" / "embed_phylo"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_PATH = DATA_DIR / "benchmarks" / "embedding_phylo_results.json"
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)

    EMB_PATH = SPECIES_ROOT / "data" / "conotoxin_embeddings.h5"
    IQTREE_PATH = SPECIES_ROOT / "results" / "iqtree_conotoxin.treefile"

    results = {}

    # ── Step 1: Load and preprocess embeddings ────────────────────────
    print("=" * 60)
    print("Step 1: Load conotoxin embeddings")
    print("=" * 60)

    embeddings_raw = {}
    with h5py.File(EMB_PATH, "r") as f:
        for key in f.keys():
            emb = np.array(f[key], dtype=np.float32)
            # Mean pool: (L, 1024) → (1024,)
            embeddings_raw[key] = emb.mean(axis=0)
    print(f"  Loaded {len(embeddings_raw)} proteins, dim={next(iter(embeddings_raw.values())).shape[0]}")

    # Apply ABTT3 + RP512
    from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
    from src.one_embedding.universal_transforms import random_orthogonal_project
    from src.utils.h5_store import load_residue_embeddings

    # Compute ABTT3 stats from the 5K SCOPe corpus (stable PC estimates)
    CORPUS_H5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if CORPUS_H5.exists():
        print("  Computing ABTT3 stats from 5K SCOPe corpus...")
        corpus_embs = load_residue_embeddings(CORPUS_H5)
        stats = compute_corpus_stats(corpus_embs, n_sample=50_000, n_pcs=5, seed=42)
    else:
        print("  WARNING: 5K corpus not found, using conotoxin set (less stable)")
        stats = compute_corpus_stats(
            {k: v.reshape(1, -1) for k, v in embeddings_raw.items()},
            n_sample=len(embeddings_raw), n_pcs=5, seed=42,
        )
    top3 = stats["top_pcs"][:3]

    data = {}
    for pid, vec in embeddings_raw.items():
        vec_abtt = all_but_the_top(vec.reshape(1, -1), top3).flatten()
        vec_rp = random_orthogonal_project(vec_abtt.reshape(1, -1), d_out=512, seed=42).flatten()
        data[pid] = vec_rp.astype(np.float64)

    print(f"  Preprocessed: ABTT3+RP512 → {next(iter(data.values())).shape[0]}d")
    results["n_taxa"] = len(data)
    results["n_dims"] = next(iter(data.values())).shape[0]

    # ── Step 2: NJ starting tree ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Build NJ tree from embedding distances")
    print("=" * 60)

    t0 = time.time()
    nj_tree = NJBuilder.from_embeddings(data)
    t_nj = time.time() - t0
    print(f"  NJ tree: {nj_tree.n_leaves} leaves, "
          f"total BL={nj_tree.total_branch_length():.2f}, built in {t_nj:.3f}s")

    nj_nwk = write_newick(nj_tree)
    (RESULTS_DIR / "conotoxin_nj.nwk").write_text(nj_nwk)

    # ── Step 3: MCMC ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Run Bayesian MCMC (2 runs × 4 chains)")
    print("=" * 60)

    N_RUNS = 2
    N_CHAINS = 4
    N_GEN = 200_000
    SAMPLE_FREQ = 200
    SWAP_FREQ = 50

    t0 = time.time()
    orch = MultiRunOrchestrator(
        data=data, n_runs=N_RUNS, n_chains=N_CHAINS,
        n_generations=N_GEN, sample_freq=SAMPLE_FREQ,
        swap_freq=SWAP_FREQ, seed=42,
    )
    run_results = orch.run()
    t_mcmc = time.time() - t0

    print(f"  MCMC completed in {t_mcmc:.1f}s")
    for i, rr in enumerate(run_results):
        n_samples = len(rr["sampled_trees"])
        final_logL = rr["sampled_logL"][-1] if rr["sampled_logL"] else float("nan")
        print(f"  Run {i}: {n_samples} samples, final logL={final_logL:.1f}")
        print(f"    Proposals: {rr['acceptance_summary']}")
        if rr["n_swaps_attempted"] > 0:
            swap_rate = rr["n_swaps_accepted"] / rr["n_swaps_attempted"]
            print(f"    Swaps: {rr['n_swaps_accepted']}/{rr['n_swaps_attempted']} ({swap_rate:.2%})")

    # Save sampled trees
    for i, rr in enumerate(run_results):
        path = RESULTS_DIR / f"conotoxin_trees_run{i}.nwk"
        path.write_text("\n".join(rr["sampled_trees"]))

    results["n_generations"] = N_GEN
    results["n_runs"] = N_RUNS
    results["n_chains"] = N_CHAINS
    results["mcmc_time_s"] = t_mcmc

    # ── Step 4: Convergence diagnostics ───────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Convergence diagnostics")
    print("=" * 60)

    tree_sets = [rr["sampled_trees"] for rr in run_results]
    asdsf = Diagnostics.asdsf(tree_sets)
    print(f"  ASDSF: {asdsf:.4f} ({'CONVERGED' if asdsf < 0.05 else 'NOT CONVERGED'})")

    for param_name in ["sampled_logL", "sampled_sigma2", "sampled_tree_length"]:
        chains = [np.array(rr[param_name]) for rr in run_results]
        ess_vals = [Diagnostics.effective_sample_size(c[len(c)//4:]) for c in chains]
        psrf_val = Diagnostics.psrf([c[len(c)//4:] for c in chains])
        print(f"  {param_name}: ESS={[f'{e:.0f}' for e in ess_vals]}, PSRF={psrf_val:.3f}")

    results["asdsf"] = asdsf
    results["converged"] = asdsf < 0.05

    diag_out = {
        "asdsf": asdsf,
        "logL_ess": [Diagnostics.effective_sample_size(np.array(rr["sampled_logL"])[len(rr["sampled_logL"])//4:])
                     for rr in run_results],
    }
    with open(RESULTS_DIR / "conotoxin_diagnostics.json", "w") as f:
        json.dump(diag_out, f, indent=2, default=float)

    # ── Step 5: Consensus tree ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Build consensus tree")
    print("=" * 60)

    all_trees = []
    for rr in run_results:
        all_trees.extend(rr["sampled_trees"])
    consensus = ConsensusBuilder.majority_rule(all_trees, burnin_frac=0.25)
    consensus_nwk = write_newick(consensus)
    (RESULTS_DIR / "conotoxin_consensus.nwk").write_text(consensus_nwk)
    print(f"  Consensus tree: {consensus.n_leaves} leaves")

    # ── Step 6: Benchmark against IQ-TREE ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Compare to IQ-TREE reference")
    print("=" * 60)

    if IQTREE_PATH.exists():
        iqtree_nwk = IQTREE_PATH.read_text().strip()
        iqtree_tree = parse_newick(iqtree_nwk)
        iqtree_tree.resolve_polytomies()  # IQ-TREE uses trifurcating root

        rf_consensus = robinson_foulds(consensus, iqtree_tree)
        rf_nj = robinson_foulds(nj_tree, iqtree_tree)

        # Max possible RF
        max_rf = 2 * (consensus.n_leaves - 3)

        # Clade recovery: what fraction of IQ-TREE splits are in our consensus
        iq_splits = Diagnostics._get_splits(iqtree_tree)
        our_splits = Diagnostics._get_splits(consensus)
        if iq_splits:
            clade_recovery = len(iq_splits & our_splits) / len(iq_splits)
        else:
            clade_recovery = 0.0

        print(f"  RF distance (consensus vs IQ-TREE): {rf_consensus} / {max_rf}")
        print(f"  RF distance (NJ vs IQ-TREE): {rf_nj} / {max_rf}")
        print(f"  Normalized RF: {rf_consensus / max_rf:.3f}")
        print(f"  Clade recovery: {clade_recovery:.3f}")

        results["rf_consensus_vs_iqtree"] = rf_consensus
        results["rf_nj_vs_iqtree"] = rf_nj
        results["max_rf"] = max_rf
        results["normalized_rf"] = rf_consensus / max_rf
        results["clade_recovery"] = clade_recovery
    else:
        print("  IQ-TREE reference tree not found, skipping comparison")

    # ── Step 7: Diagnostic plots ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7: Diagnostic plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Trace plots
    for i, rr in enumerate(run_results):
        axes[0, 0].plot(rr["sampled_logL"], alpha=0.7, label=f"Run {i}")
        axes[0, 1].plot(rr["sampled_sigma2"], alpha=0.7, label=f"Run {i}")
        axes[1, 0].plot(rr["sampled_tree_length"], alpha=0.7, label=f"Run {i}")

    axes[0, 0].set_ylabel("Log-likelihood")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].legend()
    axes[0, 0].set_title("Log-likelihood trace")

    axes[0, 1].set_ylabel("σ²")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].legend()
    axes[0, 1].set_title("BM rate trace")

    axes[1, 0].set_ylabel("Tree length")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].legend()
    axes[1, 0].set_title("Tree length trace")

    # LogL histogram (post-burnin)
    for i, rr in enumerate(run_results):
        n = len(rr["sampled_logL"])
        post = rr["sampled_logL"][n // 4:]
        axes[1, 1].hist(post, bins=30, alpha=0.5, label=f"Run {i}")
    axes[1, 1].set_xlabel("Log-likelihood")
    axes[1, 1].set_title("Post-burnin logL")
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = RESULTS_DIR / "conotoxin_diagnostics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Save results ──────────────────────────────────────────────────
    with open(BENCH_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {BENCH_PATH}")
    print("\nDone!")
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All tests PASS (should be ~30 tests)

- [ ] **Step 3: Run the experiment (short test run first)**

Temporarily reduce N_GEN to 10000 for a smoke test:
Run: `uv run python experiments/35_embedding_phylogenetics.py`
Expected: Completes without error, produces output in `results/embed_phylo/`

- [ ] **Step 4: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py
git commit -m "feat(exp35): add conotoxin benchmark main — MCMC + IQ-TREE comparison"
```

---

### Task 15: Full run + results analysis

- [ ] **Step 1: Run full experiment (200K generations)**

Restore N_GEN to 200_000 and run:
Run: `uv run python experiments/35_embedding_phylogenetics.py`
Expected: ~1-5 minutes, produces consensus tree, diagnostics, RF comparison

- [ ] **Step 2: Check convergence**

Read `results/embed_phylo/conotoxin_diagnostics.json` and verify:
- ASDSF < 0.05 (ideally < 0.01)
- ESS > 200
- If not converged, increase N_GEN or N_RUNS

- [ ] **Step 3: Review results and commit**

```bash
git add data/benchmarks/embedding_phylo_results.json results/embed_phylo/
git commit -m "results(exp35): conotoxin embedding phylogenetics — first run"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | TreeNode + Tree | 4 |
| 2 | Newick I/O | 5 |
| 3 | NJ Builder | 3 |
| 4 | Random tree | 3 |
| 5 | BM Likelihood | 3 |
| 6 | stNNI proposal | 3 |
| 7 | Branch/tree/sigma proposals | 5 |
| 8 | SPR proposal | 3 |
| 9 | ProposalMixer | 2 |
| 10 | MCMCChain | 1 |
| 11 | MC3 + MultiRun | 2 |
| 12 | Diagnostics | 6 |
| 13 | Consensus + RF | 5 |
| 14 | Main experiment | — |
| 15 | Full run | — |
| **Total** | | **~45 tests** |
