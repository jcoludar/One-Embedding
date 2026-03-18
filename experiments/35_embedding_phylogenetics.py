#!/usr/bin/env python3
"""Experiment 35 — Bayesian phylogenetics from protein embeddings.

Re-implements ExaBayes core MCMC in Python with a Brownian motion likelihood
for continuous embedding data. No existing Bayesian phylo software handles
512-dimensional continuous characters.

Spec: docs/superpowers/specs/2026-03-17-embedding-phylogenetics-design.md
"""

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


# ---------------------------------------------------------------------------
# Branch length bounds (ExaBayes BoundsChecker)
# ---------------------------------------------------------------------------

BL_MIN = 1e-10
BL_MAX = 100.0


def clamp_branch_length(bl: float) -> float:
    """Clamp branch length to valid range [BL_MIN, BL_MAX]."""
    return max(BL_MIN, min(bl, BL_MAX))


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in a phylogenetic tree."""
    id: int
    name: str = ""
    branch_length: float = 0.0
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Return True if this node has no parent."""
        return self.parent is None


class Tree:
    """A rooted phylogenetic tree."""

    def __init__(self, root: TreeNode):
        self.root = root
        self._reindex()

    def _reindex(self):
        """Rebuild internal indices by traversing from root."""
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
        """Number of leaf nodes."""
        return len(self.leaves)

    @property
    def n_internal(self) -> int:
        """Number of internal nodes (including root)."""
        return len(self.internals)

    def leaf_names(self) -> List[str]:
        """Return list of leaf names."""
        return [leaf.name for leaf in self.leaves]

    def _copy_subtree(self, node: TreeNode, parent: Optional[TreeNode] = None) -> TreeNode:
        """Recursively deep-copy a subtree."""
        new_node = TreeNode(
            id=node.id,
            name=node.name,
            branch_length=node.branch_length,
            parent=parent,
        )
        for child in node.children:
            new_child = self._copy_subtree(child, parent=new_node)
            new_node.children.append(new_child)
        return new_node

    def copy(self) -> "Tree":
        """Return a deep copy of this tree."""
        new_root = self._copy_subtree(self.root)
        return Tree(new_root)

    def total_branch_length(self) -> float:
        """Sum of all non-root branch lengths."""
        total = 0.0
        for node in self.nodes:
            if not node.is_root():
                total += node.branch_length
        return total

    def postorder(self) -> List[TreeNode]:
        """Return nodes in postorder traversal (leaves first, root last)."""
        result = []
        stack = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                result.append(node)
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))
        return result

    def resolve_polytomies(self):
        """Convert multifurcating nodes to binary cascades with zero-length branches."""
        # We need a counter for new node IDs. Find current max.
        max_id = max(n.id for n in self.nodes)
        next_id = max_id + 1

        # Process nodes in postorder so children are resolved before parents.
        for node in self.postorder():
            while len(node.children) > 2:
                # Pop the last two children.
                child_b = node.children.pop()
                child_a = node.children.pop()
                # Create a new internal node to group them.
                new_internal = TreeNode(
                    id=next_id,
                    name="",
                    branch_length=0.0,
                    parent=node,
                )
                next_id += 1
                child_a.parent = new_internal
                child_b.parent = new_internal
                new_internal.children = [child_a, child_b]
                node.children.append(new_internal)

        self._reindex()


# ---------------------------------------------------------------------------
# Newick I/O
# ---------------------------------------------------------------------------

def _split_top_level(s: str) -> List[str]:
    """Split string on commas not inside parentheses or quotes."""
    parts = []
    depth = 0
    in_quote = False
    quote_char = None
    current = []

    for ch in s:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
            continue

        if ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))

    return parts


def parse_newick(newick_str: str) -> Tree:
    """Parse a Newick-format string into a Tree.

    Handles quoted names (for pipe chars like A0A1P8NVR5|org101286),
    bootstrap values, and multifurcating trees.
    """
    newick_str = newick_str.strip()
    # Remove trailing semicolons.
    while newick_str.endswith(';'):
        newick_str = newick_str[:-1]
    newick_str = newick_str.strip()

    counter = [0]  # mutable counter for node IDs

    def _parse(s: str, parent: Optional[TreeNode] = None) -> TreeNode:
        s = s.strip()
        node_id = counter[0]
        counter[0] += 1

        if s.startswith('('):
            # Find the matching close paren.
            depth = 0
            match_pos = -1
            in_q = False
            q_ch = None
            for i, ch in enumerate(s):
                if in_q:
                    if ch == q_ch:
                        in_q = False
                    continue
                if ch in ("'", '"'):
                    in_q = True
                    q_ch = ch
                    continue
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        match_pos = i
                        break

            children_str = s[1:match_pos]
            label_str = s[match_pos + 1:]

            # Parse label:branch_length from the suffix after ')'.
            name, bl = _parse_label(label_str)

            node = TreeNode(id=node_id, name=name, branch_length=bl, parent=parent)

            # Parse children.
            child_strs = _split_top_level(children_str)
            for cs in child_strs:
                cs = cs.strip()
                if cs:
                    child = _parse(cs, parent=node)
                    node.children.append(child)

            return node
        else:
            # Leaf node: name:branch_length
            name, bl = _parse_label(s)
            return TreeNode(id=node_id, name=name, branch_length=bl, parent=parent)

    def _parse_label(s: str) -> Tuple[str, float]:
        """Parse 'name:branch_length' from a label string.

        Name may be quoted. Bootstrap values (numbers without quotes) on
        internal nodes are treated as names (standard Newick convention).
        """
        s = s.strip()
        if not s:
            return "", 0.0

        # Check if name is quoted.
        if s and s[0] in ("'", '"'):
            quote_char = s[0]
            end_quote = s.index(quote_char, 1)
            name = s[1:end_quote]
            rest = s[end_quote + 1:]
            if rest.startswith(':'):
                try:
                    bl = float(rest[1:])
                except ValueError:
                    bl = 0.0
            else:
                bl = 0.0
            return name, bl

        # Unquoted: find last colon (to handle names with colons, though rare).
        colon_pos = s.rfind(':')
        if colon_pos == -1:
            return s, 0.0

        name = s[:colon_pos]
        try:
            bl = float(s[colon_pos + 1:])
        except ValueError:
            return s, 0.0

        return name, bl

    root = _parse(newick_str)
    root.parent = None
    return Tree(root)


def write_newick(tree: Tree) -> str:
    """Write a Tree to Newick format string.

    Quotes names containing special characters: |,;:()[]' and whitespace.
    """
    _special = re.compile(r"[|,;:()\[\]' \t]")

    def _quote_name(name: str) -> str:
        if not name:
            return ""
        if _special.search(name):
            return "'" + name + "'"
        return name

    def _write_node(node: TreeNode) -> str:
        if node.is_leaf():
            s = _quote_name(node.name)
        else:
            child_strs = [_write_node(c) for c in node.children]
            s = "(" + ",".join(child_strs) + ")"
            if node.name:
                s += _quote_name(node.name)

        if not node.is_root():
            s += ":%.10f" % node.branch_length

        return s

    return _write_node(tree.root) + ";"


# ---------------------------------------------------------------------------
# Neighbor-Joining
# ---------------------------------------------------------------------------

class NJBuilder:
    """Neighbor-Joining tree construction."""

    @staticmethod
    def build(dist_matrix: np.ndarray, names: List[str]) -> Tree:
        """Build a tree using the Saitou-Nei Neighbor-Joining algorithm.

        Parameters
        ----------
        dist_matrix : np.ndarray
            Symmetric distance matrix of shape (n, n).
        names : list of str
            Taxon names corresponding to rows/columns.

        Returns
        -------
        Tree
            The constructed NJ tree.
        """
        n = len(names)
        D = dist_matrix.astype(float).copy()
        node_id_counter = 0

        # Create initial leaf nodes.
        active_nodes: List[TreeNode] = []
        for name in names:
            active_nodes.append(TreeNode(id=node_id_counter, name=name))
            node_id_counter += 1

        # Iteratively join closest pair.
        while len(active_nodes) > 2:
            r = len(active_nodes)
            # Compute row sums.
            row_sums = D[:r, :r].sum(axis=1)

            # Compute Q matrix.
            Q = np.zeros((r, r))
            for i in range(r):
                for j in range(i + 1, r):
                    Q[i, j] = (r - 2) * D[i, j] - row_sums[i] - row_sums[j]
                    Q[j, i] = Q[i, j]

            # Find minimum Q (upper triangle).
            np.fill_diagonal(Q, np.inf)
            min_idx = np.unravel_index(np.argmin(Q), Q.shape)
            i, j = min(min_idx), max(min_idx)

            # Compute branch lengths to new node.
            bl_i = 0.5 * D[i, j] + (row_sums[i] - row_sums[j]) / (2.0 * (r - 2))
            bl_j = D[i, j] - bl_i

            # Clamp to minimum.
            bl_i = max(bl_i, 1e-10)
            bl_j = max(bl_j, 1e-10)

            # Create new internal node.
            new_node = TreeNode(id=node_id_counter, name="")
            node_id_counter += 1

            node_i = active_nodes[i]
            node_j = active_nodes[j]
            node_i.branch_length = bl_i
            node_j.branch_length = bl_j
            node_i.parent = new_node
            node_j.parent = new_node
            new_node.children = [node_i, node_j]

            # Compute distances from new node to remaining taxa.
            new_dists = np.zeros(r)
            for k in range(r):
                if k != i and k != j:
                    new_dists[k] = 0.5 * (D[i, k] + D[j, k] - D[i, j])

            # Remove i and j, add new node.
            # Build new distance matrix.
            keep = [k for k in range(r) if k != i and k != j]
            new_r = len(keep) + 1
            new_D = np.zeros((new_r, new_r))

            # Fill in distances between kept nodes.
            for a_idx, a in enumerate(keep):
                for b_idx, b in enumerate(keep):
                    new_D[a_idx, b_idx] = D[a, b]

            # Fill in distances from new node (last index).
            new_idx = new_r - 1
            for a_idx, a in enumerate(keep):
                new_D[a_idx, new_idx] = new_dists[a]
                new_D[new_idx, a_idx] = new_dists[a]

            D = new_D
            new_active = [active_nodes[k] for k in keep] + [new_node]
            active_nodes = new_active

        # Connect the last two nodes.
        assert len(active_nodes) == 2
        node_a, node_b = active_nodes

        # Create root.
        root = TreeNode(id=node_id_counter, name="")
        bl_final = D[0, 1]
        bl_a = max(bl_final / 2.0, 1e-10)
        bl_b = max(bl_final / 2.0, 1e-10)

        node_a.branch_length = bl_a
        node_b.branch_length = bl_b
        node_a.parent = root
        node_b.parent = root
        root.children = [node_a, node_b]

        return Tree(root)

    @staticmethod
    def from_embeddings(embeddings: Dict[str, np.ndarray]) -> Tree:
        """Build an NJ tree from protein embeddings using Euclidean distances.

        Parameters
        ----------
        embeddings : dict
            Mapping from protein name to embedding vector.

        Returns
        -------
        Tree
            The constructed NJ tree.
        """
        names = sorted(embeddings.keys())
        n = len(names)
        vecs = np.array([embeddings[name] for name in names])

        # Compute pairwise Euclidean distances.
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(vecs[i] - vecs[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return NJBuilder.build(dist_matrix, names)


# ---------------------------------------------------------------------------
# Random tree generation
# ---------------------------------------------------------------------------

def random_tree(names: List[str], seed: int = 42, bl_mean: float = 0.1) -> Tree:
    """Generate a random binary tree by random taxon addition.

    Parameters
    ----------
    names : list of str
        Taxon names for the leaves.
    seed : int
        Random seed for reproducibility.
    bl_mean : float
        Mean branch length (exponential distribution).

    Returns
    -------
    Tree
        A random binary tree.
    """
    rng = np.random.default_rng(seed)
    names = list(names)
    rng.shuffle(names)

    node_id = 0

    if len(names) < 2:
        if len(names) == 1:
            leaf = TreeNode(id=node_id, name=names[0], branch_length=0.0)
            return Tree(leaf)
        else:
            root = TreeNode(id=node_id)
            return Tree(root)

    # Start with first two taxa.
    root = TreeNode(id=node_id, name="")
    node_id += 1

    leaf_a = TreeNode(
        id=node_id, name=names[0],
        branch_length=rng.exponential(bl_mean),
        parent=root,
    )
    node_id += 1

    leaf_b = TreeNode(
        id=node_id, name=names[1],
        branch_length=rng.exponential(bl_mean),
        parent=root,
    )
    node_id += 1

    root.children = [leaf_a, leaf_b]
    tree = Tree(root)

    # Add remaining taxa one at a time.
    for name in names[2:]:
        # Collect all edges (each non-root node represents the edge to its parent).
        edges = [n for n in tree.nodes if not n.is_root()]
        # Pick a random edge to break.
        target = edges[rng.integers(len(edges))]

        # Create new internal node on the chosen edge.
        new_internal = TreeNode(id=node_id, name="")
        node_id += 1

        # Create new leaf.
        new_leaf = TreeNode(
            id=node_id, name=name,
            branch_length=rng.exponential(bl_mean),
            parent=new_internal,
        )
        node_id += 1

        # Insert new_internal between target and target.parent.
        old_parent = target.parent

        # Remove target from old parent's children.
        old_parent.children.remove(target)

        # Set up new_internal.
        new_internal.parent = old_parent
        new_internal.branch_length = target.branch_length / 2.0
        old_parent.children.append(new_internal)

        # Reattach target under new_internal.
        target.parent = new_internal
        target.branch_length = target.branch_length / 2.0
        new_internal.children = [target, new_leaf]

        # Ensure branch lengths are positive.
        if new_internal.branch_length <= 0:
            new_internal.branch_length = 1e-10
        if target.branch_length <= 0:
            target.branch_length = 1e-10

        tree._reindex()

    return tree


# ---------------------------------------------------------------------------
# Brownian Motion Likelihood
# ---------------------------------------------------------------------------

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
        D = next(iter(data.values())).shape[0]
        log_2pi = np.log(2.0 * np.pi)

        partials: Dict[int, Tuple[np.ndarray, float]] = {}  # node_id -> (mu, v)
        total_logL = 0.0

        for node in tree.postorder():
            if node.is_leaf():
                mu = data[node.name].astype(np.float64)
                partials[node.id] = (mu, 0.0)
            else:
                assert len(node.children) == 2
                left, right = node.children

                mu_L, v_L = partials[left.id]
                mu_R, v_R = partials[right.id]

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
                partials[node.id] = (mu, v)

        return total_logL


# ---------------------------------------------------------------------------
# Main (placeholder for future MCMC)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MCMC Proposals — Topology Moves
# ---------------------------------------------------------------------------

class StochasticNNI:
    """Stochastic Nearest-Neighbor Interchange.
    Pick random internal edge, swap one subtree from each side.
    Symmetric → log Hastings ratio = 0.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        # Find internal edges: nodes that are not root and not leaf
        internal_edges = [n for n in new_tree.internals if not n.is_root()]
        if not internal_edges:
            return new_tree, 0.0

        node = internal_edges[self.rng.randint(len(internal_edges))]
        parent = node.parent

        if len(node.children) < 2 or len(parent.children) < 2:
            return new_tree, 0.0

        # Pick one child of node and one sibling (other child of parent)
        child_idx = self.rng.randint(len(node.children))
        sibling_indices = [i for i, c in enumerate(parent.children) if c is not node]
        if not sibling_indices:
            return new_tree, 0.0
        sib_idx = sibling_indices[self.rng.randint(len(sibling_indices))]

        # Swap
        child = node.children[child_idx]
        sibling = parent.children[sib_idx]
        node.children[child_idx] = sibling
        parent.children[sib_idx] = child
        sibling.parent = node
        child.parent = parent

        new_tree._reindex()
        return new_tree, 0.0


class SubtreePruneRegraft:
    """Fixed-radius Subtree Prune-and-Regraft.
    Uniform edge selection → symmetric → log HR = 0.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()

        # Pick random non-root node to prune
        prune_candidates = [n for n in new_tree.nodes
                           if not n.is_root() and not n.parent.is_root()
                           or (not n.is_root() and n.parent.is_root()
                               and len(n.parent.children) == 2)]
        if len(prune_candidates) < 2:
            return new_tree, 0.0

        prune_node = prune_candidates[self.rng.randint(len(prune_candidates))]
        prune_parent = prune_node.parent

        # Detach: sibling takes prune_parent's place
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

        # Collect pruned subtree IDs
        pruned_ids = set()
        stack = [prune_node]
        while stack:
            n = stack.pop()
            pruned_ids.add(n.id)
            stack.extend(n.children)

        # Pick uniform random regraft edge
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
        return new_tree, 0.0


# ---------------------------------------------------------------------------
# MCMC Proposals — Continuous Parameters
# ---------------------------------------------------------------------------

class BranchLengthMultiplier:
    """Log-uniform multiplier for a single random branch.
    t' = t * exp(lambda * (U - 0.5)). HR = log(multiplier).
    """
    def __init__(self, seed: int = 42, lambda_: float = 2.0):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        candidates = [n for n in new_tree.nodes if not n.is_root()]
        if not candidates:
            return new_tree, 0.0
        node = candidates[self.rng.randint(len(candidates))]
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        node.branch_length *= multiplier
        node.branch_length = clamp_branch_length(node.branch_length)
        return new_tree, math.log(multiplier)

    def tune(self, acceptance_rate: float, batch: int = 0):
        """ExaBayes-style adaptive tuning: delta = 1/sqrt(batch+1)."""
        delta = 1.0 / math.sqrt(batch + 1)
        log_lambda = math.log(self.lambda_)
        if acceptance_rate > 0.25:  # ExaBayes TARGET_RATIO = 0.25
            log_lambda += delta  # too many accepts → bolder moves
        else:
            log_lambda -= delta  # too few accepts → more conservative
        new_lambda = math.exp(log_lambda)
        # ExaBayes bounds: [0.0001, 100]
        self.lambda_ = max(0.0001, min(new_lambda, 100.0))


class TreeLengthMultiplier:
    """Scale ALL branch lengths by same multiplier."""
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
                node.branch_length = clamp_branch_length(node.branch_length)
                n_branches += 1
        return new_tree, n_branches * math.log(multiplier)

    def tune(self, acceptance_rate: float, batch: int = 0):
        """ExaBayes-style adaptive tuning: delta = 1/sqrt(batch+1)."""
        delta = 1.0 / math.sqrt(batch + 1)
        log_lambda = math.log(self.lambda_)
        if acceptance_rate > 0.25:  # ExaBayes TARGET_RATIO = 0.25
            log_lambda += delta  # too many accepts → bolder moves
        else:
            log_lambda -= delta  # too few accepts → more conservative
        new_lambda = math.exp(log_lambda)
        # ExaBayes bounds: [0.0001, 100]
        self.lambda_ = max(0.0001, min(new_lambda, 100.0))


class SigmaMultiplier:
    """Log-uniform multiplier for BM rate σ²."""
    def __init__(self, seed: int = 42, lambda_: float = 1.0):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose_sigma(self, sigma2: float) -> Tuple[float, float]:
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        new_sigma2 = sigma2 * multiplier
        return new_sigma2, math.log(multiplier)

    def tune(self, acceptance_rate: float, batch: int = 0):
        """ExaBayes-style adaptive tuning: delta = 1/sqrt(batch+1)."""
        delta = 1.0 / math.sqrt(batch + 1)
        log_lambda = math.log(self.lambda_)
        if acceptance_rate > 0.25:  # ExaBayes TARGET_RATIO = 0.25
            log_lambda += delta  # too many accepts → bolder moves
        else:
            log_lambda -= delta  # too few accepts → more conservative
        new_lambda = math.exp(log_lambda)
        # ExaBayes bounds: [0.0001, 100]
        self.lambda_ = max(0.0001, min(new_lambda, 100.0))


class NodeSlider:
    """Slide a node by redistributing two adjacent branch lengths.

    Pick a random internal non-root node, take its branch to parent and one
    child's branch. Scale their product by a multiplier, redistribute.
    Topology unchanged. Hastings ratio = 2 * log(multiplier).
    Matches ExaBayes NodeSlider (weight=5.0, init lambda=0.191).
    """
    def __init__(self, seed: int = 42, lambda_: float = 0.191):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        internal_edges = [n for n in new_tree.internals if not n.is_root()]
        if not internal_edges:
            return new_tree, 0.0
        node = internal_edges[self.rng.randint(len(internal_edges))]
        if not node.children:
            return new_tree, 0.0
        child = node.children[self.rng.randint(len(node.children))]

        old_a = node.branch_length
        old_b = child.branch_length

        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        new_product = old_a * old_b * multiplier

        split_frac = self.rng.uniform(0.1, 0.9)
        new_a = new_product ** split_frac
        new_b = new_product ** (1.0 - split_frac)

        node.branch_length = clamp_branch_length(new_a)
        child.branch_length = clamp_branch_length(new_b)

        log_hr = 2.0 * math.log(multiplier)
        return new_tree, log_hr

    def tune(self, acceptance_rate: float, batch: int = 0):
        """ExaBayes-style adaptive tuning."""
        delta = 1.0 / math.sqrt(batch + 1)
        log_lambda = math.log(self.lambda_)
        if acceptance_rate > 0.25:
            log_lambda += delta
        else:
            log_lambda -= delta
        new_lambda = math.exp(log_lambda)
        self.lambda_ = max(0.0001, min(new_lambda, 100.0))


# ---------------------------------------------------------------------------
# Proposal Mixer
# ---------------------------------------------------------------------------

class ProposalMixer:
    """Weighted proposal selection with acceptance tracking."""
    def __init__(self, proposal_names: List[str], weights: List[float], seed: int = 42):
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
            name: {"total": self._total[name], "accepted": self._accepted[name],
                   "rate": self.acceptance_rate(name)}
            for name in self.names
        }


# ---------------------------------------------------------------------------
# BM Simulation
# ---------------------------------------------------------------------------

def simulate_bm(tree: Tree, sigma2: float, D: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Simulate Brownian motion data on a tree. Returns {leaf_name: (D,) array}."""
    rng = np.random.RandomState(seed)
    node_values: Dict[int, np.ndarray] = {}
    # Root
    node_values[tree.root.id] = rng.randn(D) * 0.1
    # Preorder traversal
    for node in tree.postorder()[::-1]:  # reverse postorder = preorder
        if node.is_root():
            continue
        parent_val = node_values[node.parent.id]
        noise = rng.randn(D) * math.sqrt(sigma2 * node.branch_length)
        node_values[node.id] = parent_val + noise
    return {node.name: node_values[node.id] for node in tree.leaves}


def estimate_sigma2(data: Dict[str, np.ndarray], tree: Tree) -> float:
    """Estimate σ² from data and tree using method-of-moments.

    Under BM, Var(X_i - X_j) = σ² * (t_i + t_j) for siblings.
    We approximate by using overall data variance / total tree length.
    """
    vecs = np.array(list(data.values()))
    data_var = np.var(vecs)
    total_bl = tree.total_branch_length()
    if total_bl < 1e-10:
        return 1.0
    return float(data_var / total_bl)


# ---------------------------------------------------------------------------
# MCMC Chain
# ---------------------------------------------------------------------------

class MCMCChain:
    """Single Metropolis-Hastings MCMC chain for BM phylogenetics."""

    def __init__(self, data: Dict[str, np.ndarray],
                 start_tree: Tree, sigma2_init: float = 1.0,
                 n_generations: int = 100_000, sample_freq: int = 500,
                 beta: float = 1.0, seed: int = 42, bl_prior_rate: float = 1.0):
        self.data = data
        self.tree = start_tree
        self.sigma2 = sigma2_init
        self.n_generations = n_generations
        self.sample_freq = sample_freq
        self.beta = beta  # 1.0 = cold chain
        self.bl_prior_rate = bl_prior_rate

        self.bm = BMLikelihood()
        self.logL = self.bm.log_likelihood(self.tree, self.data, self.sigma2)
        self.log_prior = self._compute_log_prior(self.tree, self.sigma2, self.bl_prior_rate)

        # Proposals
        self.nni = StochasticNNI(seed=seed)
        self.spr = SubtreePruneRegraft(seed=seed + 1)
        self.bl_mult = BranchLengthMultiplier(seed=seed + 2)
        self.tl_mult = TreeLengthMultiplier(seed=seed + 3)
        self.sigma_mult = SigmaMultiplier(seed=seed + 4)
        self.node_slider = NodeSlider(seed=seed + 7)

        self.mixer = ProposalMixer(
            proposal_names=["nni", "spr", "bl", "tl", "sigma", "ns"],
            weights=[6.0, 6.0, 9.0, 1.0, 1.0, 5.0],
            seed=seed + 8,
        )
        self.accept_rng = np.random.RandomState(seed + 6)

        self._tune_batch = 0

        # Samples (cold chain only)
        self.sampled_trees: List[str] = []
        self.sampled_logL: List[float] = []
        self.sampled_sigma2: List[float] = []
        self.sampled_tree_length: List[float] = []

    @staticmethod
    def _compute_log_prior(tree: Tree, sigma2: float, bl_prior_rate: float = 1.0) -> float:
        """Exp(bl_prior_rate) on branch lengths, LogNormal(0,1) on sigma2."""
        log_p = 0.0
        rate = bl_prior_rate
        for node in tree.nodes:
            if not node.is_root():
                log_p += math.log(rate) - rate * node.branch_length
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
            new_tree = self.tree.copy()
            new_sigma2, log_hr = self.sigma_mult.propose_sigma(self.sigma2)
        elif proposal_name == "ns":
            new_tree, log_hr = self.node_slider.propose(self.tree)
            new_sigma2 = self.sigma2
        else:
            return

        new_logL = self.bm.log_likelihood(new_tree, self.data, new_sigma2)
        old_log_prior = self.log_prior
        new_log_prior = self._compute_log_prior(new_tree, new_sigma2, self.bl_prior_rate)

        log_alpha = self.beta * (new_logL - self.logL) + (new_log_prior - old_log_prior) + log_hr

        accepted = False
        if log_alpha >= 0 or math.log(self.accept_rng.uniform()) < log_alpha:
            self.tree = new_tree
            self.sigma2 = new_sigma2
            self.logL = new_logL
            self.log_prior = new_log_prior
            accepted = True

        self.mixer.record_acceptance(proposal_name, accepted)

        # ExaBayes-style auto-tuning with decreasing step size
        gen_count = sum(self.mixer._total.values())
        if gen_count > 0 and gen_count % 200 == 0:
            for name, prop in [("bl", self.bl_mult), ("tl", self.tl_mult), ("sigma", self.sigma_mult), ("ns", self.node_slider)]:
                rate = self.mixer.acceptance_rate(name)
                if self.mixer._total[name] > 20:
                    prop.tune(rate, batch=self._tune_batch)
            self._tune_batch += 1

    def run(self):
        for gen in range(self.n_generations):
            self._step()
            if (gen + 1) % self.sample_freq == 0 and self.beta == 1.0:
                self.sampled_trees.append(write_newick(self.tree))
                self.sampled_logL.append(self.logL)
                self.sampled_sigma2.append(self.sigma2)
                self.sampled_tree_length.append(self.tree.total_branch_length())


# ---------------------------------------------------------------------------
# MC3 Runner (Metropolis-Coupled MCMC)
# ---------------------------------------------------------------------------

class MC3Runner:
    """Metropolis-Coupled MCMC: N heated chains with periodic swap."""

    def __init__(self, data: Dict[str, np.ndarray],
                 n_chains: int = 4, n_generations: int = 100_000,
                 sample_freq: int = 500, swap_freq: int = 100,
                 delta: float = 0.1, start_tree: Optional[Tree] = None,
                 seed: int = 42, bl_prior_rate: float = 1.0):
        self.n_chains = n_chains
        self.n_generations = n_generations
        self.swap_freq = swap_freq
        self.swap_rng = np.random.RandomState(seed + 100)

        betas = [1.0 / (1.0 + i * delta) for i in range(n_chains)]

        if start_tree is None:
            start_tree = random_tree(sorted(data.keys()), seed=seed)

        sigma2_init = estimate_sigma2(data, start_tree)

        self.chains: List[MCMCChain] = []
        for i in range(n_chains):
            chain = MCMCChain(
                data=data, start_tree=start_tree.copy(),
                sigma2_init=sigma2_init, n_generations=0,
                sample_freq=sample_freq, beta=betas[i],
                seed=seed + i * 10, bl_prior_rate=bl_prior_rate,
            )
            self.chains.append(chain)

        self.cold_chain = self.chains[0]
        self.n_swaps_attempted = 0
        self.n_swaps_accepted = 0

    def run(self):
        report_interval = max(self.n_generations // 10, 1000)
        for gen in range(self.n_generations):
            for chain in self.chains:
                chain._step()

            if (gen + 1) % self.cold_chain.sample_freq == 0:
                self.cold_chain.sampled_trees.append(write_newick(self.cold_chain.tree))
                self.cold_chain.sampled_logL.append(self.cold_chain.logL)
                self.cold_chain.sampled_sigma2.append(self.cold_chain.sigma2)
                self.cold_chain.sampled_tree_length.append(
                    self.cold_chain.tree.total_branch_length())

            if (gen + 1) % self.swap_freq == 0 and self.n_chains > 1:
                self._attempt_swap()

            if (gen + 1) % report_interval == 0:
                pct = 100 * (gen + 1) / self.n_generations
                logL = self.cold_chain.logL
                sigma2 = self.cold_chain.sigma2
                print(f"    [{pct:5.1f}%] gen={gen+1:>8d}  logL={logL:.1f}  \u03c3\u00b2={sigma2:.4f}", flush=True)

    def _attempt_swap(self):
        i = self.swap_rng.randint(self.n_chains - 1)
        j = i + 1
        ci, cj = self.chains[i], self.chains[j]

        log_alpha = (ci.beta - cj.beta) * (cj.logL - ci.logL)
        self.n_swaps_attempted += 1

        if log_alpha >= 0 or math.log(self.swap_rng.uniform()) < log_alpha:
            ci.tree, cj.tree = cj.tree, ci.tree
            ci.sigma2, cj.sigma2 = cj.sigma2, ci.sigma2
            ci.logL, cj.logL = cj.logL, ci.logL
            ci.log_prior, cj.log_prior = cj.log_prior, ci.log_prior
            self.n_swaps_accepted += 1


# ---------------------------------------------------------------------------
# Multi-run Orchestrator
# ---------------------------------------------------------------------------

def _run_single_mc3(args: dict) -> dict:
    """Worker function for ProcessPoolExecutor."""
    runner = MC3Runner(
        data=args["data"], n_chains=args["n_chains"],
        n_generations=args["n_generations"],
        sample_freq=args["sample_freq"],
        swap_freq=args["swap_freq"],
        delta=args["delta"],
        start_tree=args["start_tree"],
        seed=args["seed"],
        bl_prior_rate=args.get("bl_prior_rate", 1.0),
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
                 seed: int = 42, max_workers: Optional[int] = None,
                 bl_prior_rate: float = 1.0):
        self.data = data
        self.n_runs = n_runs
        self.n_chains = n_chains
        self.n_generations = n_generations
        self.sample_freq = sample_freq
        self.swap_freq = swap_freq
        self.delta = delta
        self.seed = seed
        self.max_workers = max_workers or n_runs
        self.bl_prior_rate = bl_prior_rate

        names = sorted(data.keys())
        self.start_trees: List[Tree] = [NJBuilder.from_embeddings(data)]
        for i in range(1, n_runs):
            self.start_trees.append(random_tree(names, seed=seed + i * 1000))

    def run(self) -> List[dict]:
        args_list = [{
            "data": self.data, "n_chains": self.n_chains,
            "n_generations": self.n_generations,
            "sample_freq": self.sample_freq,
            "swap_freq": self.swap_freq,
            "delta": self.delta,
            "start_tree": self.start_trees[i],
            "seed": self.seed + i * 100,
            "bl_prior_rate": self.bl_prior_rate,
        } for i in range(self.n_runs)]

        if self.n_runs == 1:
            return [_run_single_mc3(args_list[0])]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_run_single_mc3, args_list))
        return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class Diagnostics:
    """ASDSF, ESS, PSRF convergence diagnostics."""

    @staticmethod
    def effective_sample_size(samples: np.ndarray) -> float:
        """ESS via autocorrelation."""
        n = len(samples)
        if n < 10:
            return float(n)
        x = samples - np.mean(samples)
        var = np.var(x, ddof=1)
        if var < 1e-30:
            return float(n)
        fft = np.fft.fft(x, n=2 * n)
        acf = np.real(np.fft.ifft(fft * np.conj(fft)))[:n] / (var * n)
        tau = 1.0
        for i in range(1, n // 2):
            pair_sum = acf[2 * i - 1] + acf[2 * i]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum
        return n / tau

    @staticmethod
    def psrf(chains: List[np.ndarray]) -> float:
        """Potential Scale Reduction Factor (Gelman-Rubin)."""
        m = len(chains)
        n = min(len(c) for c in chains)
        if n < 2 or m < 2:
            return float("inf")
        chains = [c[:n] for c in chains]
        chain_means = [np.mean(c) for c in chains]
        chain_vars = [np.var(c, ddof=1) for c in chains]
        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        if W < 1e-30:
            return 1.0
        var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B
        return math.sqrt(var_hat / W)

    @staticmethod
    def _get_splits(tree: Tree) -> set:
        """Get bipartition set for an unrooted tree."""
        all_leaves = frozenset(tree.leaf_names())
        splits = set()
        for node in tree.postorder():
            if node.is_leaf() or node.is_root():
                continue
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
            split = min(below, complement, key=len)
            if 0 < len(split) < len(all_leaves):
                splits.add(split)
        return splits

    @staticmethod
    def _get_splits_with_lengths(tree: Tree) -> Dict[frozenset, float]:
        """Get bipartition set with branch lengths for an unrooted tree."""
        all_leaves = frozenset(tree.leaf_names())
        splits = {}
        for node in tree.postorder():
            if node.is_leaf() or node.is_root():
                continue
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
            split = min(below, complement, key=len)
            if 0 < len(split) < len(all_leaves):
                splits[split] = node.branch_length
        return splits

    @staticmethod
    def asdsf(tree_sets: List[List[str]], burnin_frac: float = 0.25,
              min_freq: float = 0.05) -> float:
        """Average Standard Deviation of Split Frequencies across runs."""
        split_freqs_per_run: List[Dict[frozenset, float]] = []
        for trees_nwk in tree_sets:
            n = len(trees_nwk)
            start = int(n * burnin_frac)
            post = trees_nwk[start:]
            n_post = len(post)
            if n_post == 0:
                split_freqs_per_run.append({})
                continue
            split_counts: Dict[frozenset, int] = {}
            for nwk in post:
                tree = parse_newick(nwk)
                for split in Diagnostics._get_splits(tree):
                    split_counts[split] = split_counts.get(split, 0) + 1
            split_freqs_per_run.append({s: c / n_post for s, c in split_counts.items()})

        all_splits = set()
        for sf in split_freqs_per_run:
            all_splits.update(sf.keys())
        if not all_splits:
            return 0.0

        sds = []
        for split in all_splits:
            freqs = [sf.get(split, 0.0) for sf in split_freqs_per_run]
            if np.mean(freqs) < min_freq:
                continue
            sds.append(np.std(freqs, ddof=0))
        return float(np.mean(sds)) if sds else 0.0


# ── Consensus Tree ────────────────────────────────────────────────────────

class ConsensusBuilder:
    """Majority-rule consensus tree from posterior sample.

    Uses the standard Build algorithm: sort compatible splits by size
    ascending (smallest/most specific first), then merge clusters
    bottom-up. All majority-rule splits (>50%) are guaranteed mutually
    compatible, so they all get included. This matches ExaBayes consense
    behaviour.
    """

    @staticmethod
    def majority_rule(trees_nwk: List[str], burnin_frac: float = 0.25,
                      threshold: float = 0.5) -> Tree:
        n = len(trees_nwk)
        start = int(n * burnin_frac)
        post_trees = trees_nwk[start:]
        n_post = len(post_trees)
        if n_post == 0:
            return parse_newick(trees_nwk[-1])

        split_counts: Dict[frozenset, int] = {}
        split_branch_lengths: Dict[frozenset, List[float]] = {}
        leaf_branch_lengths: Dict[str, List[float]] = {}
        leaf_names_set: Optional[set] = None
        for nwk in post_trees:
            tree = parse_newick(nwk)
            if leaf_names_set is None:
                leaf_names_set = set(tree.leaf_names())
            for split in Diagnostics._get_splits(tree):
                split_counts[split] = split_counts.get(split, 0) + 1
            splits_bl = Diagnostics._get_splits_with_lengths(tree)
            for split, bl in splits_bl.items():
                if split not in split_branch_lengths:
                    split_branch_lengths[split] = []
                split_branch_lengths[split].append(bl)
            for leaf in tree.leaves:
                if leaf.name not in leaf_branch_lengths:
                    leaf_branch_lengths[leaf.name] = []
                leaf_branch_lengths[leaf.name].append(leaf.branch_length)

        # Filter to majority splits, sort by SIZE ASCENDING (build bottom-up)
        majority_splits = [
            (s, c / n_post)
            for s, c in split_counts.items()
            if c / n_post >= threshold
        ]
        majority_splits.sort(key=lambda x: len(x[0]))  # smallest first

        # Build tree bottom-up using cluster merging
        node_id = [0]
        def _next_id():
            nid = node_id[0]; node_id[0] += 1; return nid

        # Start: each leaf is its own cluster
        clusters: Dict[frozenset, TreeNode] = {}
        for name in sorted(leaf_names_set):
            bl = float(np.median(leaf_branch_lengths.get(name, [0.01])))
            clusters[frozenset({name})] = TreeNode(
                id=_next_id(), name=name, branch_length=bl,
            )

        for split, freq in majority_splits:
            # Find clusters that are subsets of this split
            children_items = []
            remaining = {}
            for cluster_set, cluster_node in clusters.items():
                if cluster_set <= split:
                    children_items.append((cluster_set, cluster_node))
                else:
                    remaining[cluster_set] = cluster_node

            if len(children_items) >= 2:
                bl = float(np.median(split_branch_lengths.get(split, [0.01])))
                new_node = TreeNode(id=_next_id(), branch_length=bl)
                merged_set = frozenset()
                for cs, cn in children_items:
                    cn.parent = new_node
                    new_node.children.append(cn)
                    merged_set = merged_set | cs
                remaining[merged_set] = new_node
                clusters = remaining

        # MRE: add compatible minority splits (ExaBayes extended consensus)
        minority_splits = [
            (s, c / n_post)
            for s, c in split_counts.items()
            if c / n_post < threshold
        ]
        minority_splits.sort(key=lambda x: -x[1])  # highest frequency first

        # Max possible splits for fully resolved tree
        max_splits = len(leaf_names_set) - 3

        for split, freq in minority_splits:
            # Stop if fully resolved
            if len(clusters) <= 2:
                break
            # Check compatibility with current tree
            compatible = True
            for existing_set in clusters:
                # Two splits are compatible if one is subset of other,
                # or they don't overlap
                overlap = existing_set & split
                if overlap and overlap != existing_set and overlap != split:
                    compatible = False
                    break
            if not compatible:
                continue
            # Apply this split (same logic as majority)
            children_items = []
            remaining = {}
            for cluster_set, cluster_node in clusters.items():
                if cluster_set <= split:
                    children_items.append((cluster_set, cluster_node))
                else:
                    remaining[cluster_set] = cluster_node
            if len(children_items) >= 2:
                bl = float(np.median(split_branch_lengths.get(split, [0.01])))
                new_node = TreeNode(id=_next_id(), branch_length=bl)
                merged_set = frozenset()
                for cs, cn in children_items:
                    cn.parent = new_node
                    new_node.children.append(cn)
                    merged_set = merged_set | cs
                remaining[merged_set] = new_node
                clusters = remaining

        # Join remaining top-level clusters under root
        remaining_nodes = list(clusters.values())
        if len(remaining_nodes) == 1:
            root = remaining_nodes[0]
        else:
            root = TreeNode(id=_next_id(), children=remaining_nodes)
            for child in remaining_nodes:
                child.parent = root
        return Tree(root=root)


# ── Robinson-Foulds Distance ─────────────────────────────────────────────

def robinson_foulds(tree1: Tree, tree2: Tree) -> int:
    """Symmetric Robinson-Foulds distance."""
    splits1 = Diagnostics._get_splits(tree1)
    splits2 = Diagnostics._get_splits(tree2)
    return len(splits1.symmetric_difference(splits2))


def normalize_leaf_names(tree: Tree) -> Tree:
    """Strip |orgXXX suffixes from leaf names (IQ-TREE convention)."""
    for leaf in tree.leaves:
        if '|' in leaf.name:
            leaf.name = leaf.name.split('|')[0]
    tree._reindex()
    return tree


def load_msa(fasta_path: str) -> Dict[str, str]:
    """Load aligned FASTA into {name: aligned_sequence} dict."""
    msa = {}
    current_name = None
    current_seq: List[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    msa[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                if "|" in current_name:
                    current_name = current_name.split("|")[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        msa[current_name] = "".join(current_seq)
    return msa


def extract_aligned_embeddings(
    msa: Dict[str, str],
    embeddings: Dict[str, np.ndarray],
    fill_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Map per-residue embeddings to MSA-aligned positions.

    For each protein, walk the aligned sequence. Non-gap positions map to
    successive residue embeddings. Gap positions get fill_value.

    Args:
        msa: {name: aligned_sequence} with gaps as '-'
        embeddings: {name: (L, D) array} of per-residue embeddings
        fill_value: value for gap positions

    Returns:
        {name: (aligned_length, D) array}
    """
    aligned_len = len(next(iter(msa.values())))
    D = next(iter(embeddings.values())).shape[1]
    result = {}
    for name, aligned_seq in msa.items():
        if name not in embeddings:
            continue
        emb = embeddings[name]
        out = np.full((aligned_len, D), fill_value, dtype=np.float32)
        residue_idx = 0
        for col_idx, char in enumerate(aligned_seq):
            if char != "-":
                if residue_idx < emb.shape[0]:
                    out[col_idx] = emb[residue_idx]
                residue_idx += 1
        result[name] = out
    return result


# ── Experiment Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 35: Bayesian phylogenetics from embeddings")
    parser.add_argument("--n-gen", type=int, default=200_000, help="Generations per chain")
    parser.add_argument("--n-runs", type=int, default=2, help="Independent MCMC runs")
    parser.add_argument("--n-chains", type=int, default=4, help="Chains per run (MC3)")
    parser.add_argument("--sample-freq", type=int, default=200, help="Sample every N generations")
    parser.add_argument("--swap-freq", type=int, default=50, help="MC3 swap attempt frequency")
    parser.add_argument("--bl-prior-rate", type=float, default=1.0,
                        help="Exponential prior rate on branch lengths (default: 1.0)")
    parser.add_argument("--per-residue", action="store_true",
                        help="Use per-residue embeddings aligned by MSA")
    parser.add_argument("--msa", type=str, default=None,
                        help="Path to aligned FASTA (default: auto-detect)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py

    PROJ_ROOT = Path(__file__).resolve().parent.parent
    SPECIES_ROOT = Path("/Users/jcoludar/CascadeProjects/SpeciesEmbedding")
    sys.path.insert(0, str(PROJ_ROOT))

    DATA_DIR = PROJ_ROOT / "data"
    RESULTS_DIR = PROJ_ROOT / "results" / "embed_phylo"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_PATH = DATA_DIR / "benchmarks" / "embedding_phylo_results.json"
    BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)

    EMB_PATH = SPECIES_ROOT / "data" / "conotoxin_embeddings.h5"
    IQTREE_PATH = SPECIES_ROOT / "results" / "iqtree_conotoxin.treefile"

    results = {}

    print(f"  Config: {args.n_gen:,} generations, {args.n_runs} runs × {args.n_chains} chains")

    # ── Step 1: Load and preprocess embeddings ────────────────────────
    print("=" * 60)
    print("Step 1: Load conotoxin embeddings")
    print("=" * 60)

    embeddings_raw = {}
    with h5py.File(EMB_PATH, "r") as f:
        for key in f.keys():
            emb = np.array(f[key], dtype=np.float32)
            embeddings_raw[key] = emb.mean(axis=0)
    print(f"  Loaded {len(embeddings_raw)} proteins, dim={next(iter(embeddings_raw.values())).shape[0]}")

    # Apply ABTT3 + RP512
    from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
    from src.one_embedding.universal_transforms import random_orthogonal_project
    from src.utils.h5_store import load_residue_embeddings

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

    if args.per_residue:
        # Load per-residue embeddings
        embeddings_per_res = {}
        with h5py.File(EMB_PATH, "r") as f:
            for key in f.keys():
                embeddings_per_res[key] = np.array(f[key], dtype=np.float32)

        # Load MSA
        msa_path = args.msa or str(SPECIES_ROOT / "data" / "conotoxin_aligned.fasta")
        print(f"  Loading MSA from {msa_path}")
        msa = load_msa(msa_path)
        print(f"  MSA: {len(msa)} sequences, {len(next(iter(msa.values())))} columns")

        # Align embeddings to MSA positions
        aligned_embs = extract_aligned_embeddings(msa, embeddings_per_res)

        # Apply ABTT3 + RP per residue position, then flatten
        data = {}
        for pid, emb_matrix in aligned_embs.items():
            emb_abtt = all_but_the_top(emb_matrix, top3)
            emb_rp = random_orthogonal_project(emb_abtt, d_out=512, seed=42)
            data[pid] = emb_rp.flatten().astype(np.float64)

        n_cols = len(next(iter(msa.values())))
        dim = next(iter(data.values())).shape[0]
        print(f"  Per-residue mode: {dim}d ({n_cols} cols x 512)")
    else:
        data = {}
        for pid, vec in embeddings_raw.items():
            vec_abtt = all_but_the_top(vec.reshape(1, -1), top3).flatten()
            vec_rp = random_orthogonal_project(vec_abtt.reshape(1, -1), d_out=512, seed=42).flatten()
            data[pid] = vec_rp.astype(np.float64)

        print(f"  Preprocessed: ABTT3+RP512 → {next(iter(data.values())).shape[0]}d")

    results["n_taxa"] = len(data)
    results["n_dims"] = next(iter(data.values())).shape[0]
    results["mode"] = "per_residue" if args.per_residue else "per_protein"

    # ── Step 2: NJ starting tree ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Build NJ tree from embedding distances")
    print("=" * 60)

    t0 = time.time()
    nj_tree = NJBuilder.from_embeddings(data)
    t_nj = time.time() - t0
    print(f"  NJ tree: {nj_tree.n_leaves} leaves, "
          f"total BL={nj_tree.total_branch_length():.2f}, built in {t_nj:.3f}s")
    (RESULTS_DIR / "conotoxin_nj.nwk").write_text(write_newick(nj_tree))

    # ── Step 3: MCMC ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 3: Run Bayesian MCMC ({args.n_runs} runs × {args.n_chains} chains)")
    print("=" * 60)

    t0 = time.time()
    orch = MultiRunOrchestrator(
        data=data, n_runs=args.n_runs, n_chains=args.n_chains,
        n_generations=args.n_gen, sample_freq=args.sample_freq,
        swap_freq=args.swap_freq, seed=42,
        bl_prior_rate=args.bl_prior_rate,
    )
    run_results = orch.run()
    t_mcmc = time.time() - t0

    print(f"  MCMC completed in {t_mcmc:.1f}s")
    for i, rr in enumerate(run_results):
        n_samples = len(rr["sampled_trees"])
        final_logL = rr["sampled_logL"][-1] if rr["sampled_logL"] else float("nan")
        print(f"  Run {i}: {n_samples} samples, final logL={final_logL:.1f}")
        acc = rr["acceptance_summary"]
        for pname, pdata in acc.items():
            print(f"    {pname}: {pdata['accepted']}/{pdata['total']} ({pdata['rate']:.2%})")
        if rr["n_swaps_attempted"] > 0:
            sr = rr["n_swaps_accepted"] / rr["n_swaps_attempted"]
            print(f"    swaps: {rr['n_swaps_accepted']}/{rr['n_swaps_attempted']} ({sr:.2%})")

    for i, rr in enumerate(run_results):
        (RESULTS_DIR / f"conotoxin_trees_run{i}.nwk").write_text("\n".join(rr["sampled_trees"]))

    results["n_generations"] = args.n_gen
    results["n_runs"] = args.n_runs
    results["n_chains"] = args.n_chains
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

    diag_out = {"asdsf": asdsf}
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
        iqtree_tree.resolve_polytomies()

        iqtree_tree = normalize_leaf_names(iqtree_tree)

        rf_consensus = robinson_foulds(consensus, iqtree_tree)
        rf_nj = robinson_foulds(nj_tree, iqtree_tree)
        max_rf = 2 * (consensus.n_leaves - 3)

        iq_splits = Diagnostics._get_splits(iqtree_tree)
        our_splits = Diagnostics._get_splits(consensus)
        clade_recovery = len(iq_splits & our_splits) / len(iq_splits) if iq_splits else 0.0

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
    for i, rr in enumerate(run_results):
        axes[0, 0].plot(rr["sampled_logL"], alpha=0.7, label=f"Run {i}")
        axes[0, 1].plot(rr["sampled_sigma2"], alpha=0.7, label=f"Run {i}")
        axes[1, 0].plot(rr["sampled_tree_length"], alpha=0.7, label=f"Run {i}")

    axes[0, 0].set_ylabel("Log-likelihood"); axes[0, 0].set_xlabel("Sample")
    axes[0, 0].legend(); axes[0, 0].set_title("Log-likelihood trace")
    axes[0, 1].set_ylabel("σ²"); axes[0, 1].set_xlabel("Sample")
    axes[0, 1].legend(); axes[0, 1].set_title("BM rate trace")
    axes[1, 0].set_ylabel("Tree length"); axes[1, 0].set_xlabel("Sample")
    axes[1, 0].legend(); axes[1, 0].set_title("Tree length trace")

    for i, rr in enumerate(run_results):
        n = len(rr["sampled_logL"])
        axes[1, 1].hist(rr["sampled_logL"][n // 4:], bins=30, alpha=0.5, label=f"Run {i}")
    axes[1, 1].set_xlabel("Log-likelihood"); axes[1, 1].set_title("Post-burnin logL")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "conotoxin_diagnostics.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'conotoxin_diagnostics.png'}")

    with open(BENCH_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {BENCH_PATH}")
    print("\nDone!")
