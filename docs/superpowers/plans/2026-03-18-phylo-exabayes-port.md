# ExaBayes Port & Per-Residue Phylogenetics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port critical ExaBayes algorithms to our Python Bayesian phylogenetics (experiment 35) and add per-residue embedding mode using MSA-aligned positions.

**Architecture:** All changes are in `experiments/35_embedding_phylogenetics.py` and `tests/test_embedding_phylo.py`. The experiment is a self-contained single-file Bayesian MCMC engine with Brownian motion likelihood. We fix consensus branch lengths (median per bipartition, matching ExaBayes), add branch length bounds, add NodeSlider proposal, make the prior configurable, and add a `--per-residue` mode that uses MSA-aligned positions instead of mean-pooled vectors.

**Tech Stack:** Python 3.12, NumPy, h5py, matplotlib. MAFFT at `/opt/homebrew/bin/mafft` for alignment. No new dependencies.

**Spec:** Based on ExaBayes source at `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/tools/exabayes-src/`

---

### Task 1: Consensus branch lengths from posterior (median per bipartition)

**Context:** ExaBayes consensus (`BipartitionExtractor.cpp:393-401`) uses the **median** branch length across all posterior trees for each bipartition. Our `ConsensusBuilder.majority_rule()` hardcodes `branch_length=0.01` for all nodes (lines 1333 and 1347).

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — `ConsensusBuilder.majority_rule()`
- Test: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write failing test for consensus branch lengths**

```python
class TestConsensusBranchLengths:
    def test_median_branch_lengths(self):
        """Consensus should use median branch lengths from posterior, not hardcoded."""
        # 50 identical trees with known branch lengths
        trees = ["((A:0.5,B:0.3):0.2,C:0.7);"] * 50
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        # Find leaf A
        leaf_a = [n for n in consensus.leaves if n.name == "A"][0]
        # Should be close to 0.5, not 0.01
        assert abs(leaf_a.branch_length - 0.5) < 0.1, (
            f"Expected ~0.5, got {leaf_a.branch_length}"
        )

    def test_varied_branch_lengths_uses_median(self):
        """When branch lengths vary across trees, consensus should use median."""
        # 80 trees with BL=0.1, 20 trees with BL=10.0 for leaf A
        trees_short = ["((A:0.1,B:0.3):0.2,C:0.7);"] * 80
        trees_long = ["((A:10.0,B:0.3):0.2,C:0.7);"] * 20
        consensus = ConsensusBuilder.majority_rule(
            trees_short + trees_long, burnin_frac=0.0
        )
        leaf_a = [n for n in consensus.leaves if n.name == "A"][0]
        # Median should be 0.1 (80% of values), not mean (~2.08)
        assert abs(leaf_a.branch_length - 0.1) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestConsensusBranchLengths -v`
Expected: FAIL — branch lengths are 0.01

- [ ] **Step 3: Implement median branch lengths in ConsensusBuilder**

Modify `ConsensusBuilder.majority_rule()` in `experiments/35_embedding_phylogenetics.py`:

1. During the tree-parsing loop (line 1309-1314), also collect branch lengths per split:
```python
split_counts: Dict[frozenset, int] = {}
split_branch_lengths: Dict[frozenset, List[float]] = {}  # NEW
leaf_branch_lengths: Dict[str, List[float]] = {}  # NEW for leaves
```

2. For each parsed tree, extract both splits AND the branch length associated with each split. Also track leaf branch lengths by name.

3. When building the consensus tree, instead of `branch_length=0.01`, use `np.median(split_branch_lengths[split])` for internal nodes and `np.median(leaf_branch_lengths[name])` for leaves.

Key implementation detail: `Diagnostics._get_splits()` returns frozenset splits but not branch lengths. Add a new method `_get_splits_with_lengths()` that returns `Dict[frozenset, float]` mapping each split to its branch length in that tree. Also extract leaf branch lengths.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding_phylo.py::TestConsensusBranchLengths -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All 49+ tests pass (including new ones)

- [ ] **Step 6: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "fix(exp35): consensus uses median branch lengths per bipartition

Port from ExaBayes BipartitionExtractor: consensus tree now uses
median branch length from posterior samples for each bipartition,
instead of hardcoded 0.01. Matches ExaBayes consense behavior."
```

---

### Task 2: Branch length bounds checking

**Context:** ExaBayes (`BoundsChecker.cpp`) clamps all branch lengths to `[zMin=1e-15, zMax=1-1e-6]` (in z-space, equivalent to `[1e-6, 34.5]` in branch-length space). Our proposals have no bounds — branches can go to 0 or infinity, causing numerical issues.

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — `BranchLengthMultiplier.propose()`, `TreeLengthMultiplier.propose()`, `NodeSlider.propose()` (Task 3)
- Test: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write failing test**

```python
class TestBranchBounds:
    def test_branch_length_minimum(self):
        """Proposed branch lengths should never go below BL_MIN."""
        tree = parse_newick("((A:0.0000001,B:0.001):0.0001,C:0.001);")
        prop = BranchLengthMultiplier(seed=0, lambda_=10.0)  # aggressive
        for i in range(100):
            prop_instance = BranchLengthMultiplier(seed=i, lambda_=10.0)
            new_tree, _ = prop_instance.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length >= 1e-10, (
                        f"Branch {n.name or n.id} too small: {n.branch_length}"
                    )

    def test_branch_length_maximum(self):
        """Proposed branch lengths should never exceed BL_MAX."""
        tree = parse_newick("((A:50.0,B:80.0):30.0,C:90.0);")
        prop = BranchLengthMultiplier(seed=0, lambda_=10.0)
        for i in range(100):
            prop_instance = BranchLengthMultiplier(seed=i, lambda_=10.0)
            new_tree, _ = prop_instance.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length <= 100.0, (
                        f"Branch {n.name or n.id} too large: {n.branch_length}"
                    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestBranchBounds -v`
Expected: FAIL

- [ ] **Step 3: Add BL_MIN and BL_MAX constants and clamping**

Add constants at module level:
```python
BL_MIN = 1e-10
BL_MAX = 100.0
```

Add a helper function:
```python
def clamp_branch_length(bl: float) -> float:
    """Clamp branch length to valid range [BL_MIN, BL_MAX]."""
    return max(BL_MIN, min(bl, BL_MAX))
```

Modify `BranchLengthMultiplier.propose()`: after `node.branch_length *= multiplier`, add:
```python
node.branch_length = clamp_branch_length(node.branch_length)
```

Modify `TreeLengthMultiplier.propose()`: after scaling each branch, add the same clamp.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "fix(exp35): add branch length bounds [1e-10, 100]

Port from ExaBayes BoundsChecker: clamp proposed branch lengths to
[BL_MIN, BL_MAX] to prevent numerical underflow/overflow."
```

---

### Task 3: NodeSlider proposal

**Context:** ExaBayes `NodeSlider` (weight 5.0) picks an internal branch and one descendant, jointly rescales them. This redistributes branch lengths locally without changing topology, helping with mixing. Uses `pow(bothZ, m)` then splits via `pow(newZ, u)` / `pow(newZ, 1-u)`. Hastings ratio = `drawnMultiplier^2`.

In our branch-length space (not z-space), the equivalent is: pick two adjacent branches with lengths a and b, compute `total = a + b`, draw `u ~ Uniform(0,1)`, set `new_a = total * u`, `new_b = total * (1-u)`. The Hastings ratio for this uniform redistribution on a fixed total is 0 (symmetric). If we also scale the total with a multiplier, HR = `log(multiplier)`.

Simpler approach matching our existing style: pick an internal node, multiply `a*b` by a multiplier to get new product, then split uniformly. This is closer to ExaBayes.

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — add `NodeSlider` class, update `MCMCChain.__init__`
- Test: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write failing tests**

```python
class TestNodeSlider:
    def test_preserves_leaf_set(self):
        """NodeSlider should not change topology or leaf set."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        slider = NodeSlider(seed=0)
        new_tree, _ = slider.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_preserves_topology(self):
        """NodeSlider only changes branch lengths, not topology."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        slider = NodeSlider(seed=0)
        old_splits = Diagnostics._get_splits(tree)
        new_tree, _ = slider.propose(tree)
        new_splits = Diagnostics._get_splits(new_tree)
        assert old_splits == new_splits

    def test_changes_branch_lengths(self):
        """NodeSlider should change at least some branch lengths."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        changed = False
        for i in range(20):
            slider = NodeSlider(seed=i)
            new_tree, _ = slider.propose(tree)
            if abs(new_tree.total_branch_length() - tree.total_branch_length()) > 1e-10:
                changed = True
                break
        assert changed

    def test_positive_branch_lengths(self):
        """All proposed branch lengths should be positive."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        for i in range(50):
            slider = NodeSlider(seed=i)
            new_tree, _ = slider.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length >= BL_MIN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestNodeSlider -v`
Expected: FAIL — NodeSlider not defined

- [ ] **Step 3: Implement NodeSlider**

```python
class NodeSlider:
    """Slide a node along a path by redistributing two adjacent branch lengths.

    Pick a random internal (non-root) node, pick one child branch and the
    parent branch. Multiply their product by a log-uniform multiplier,
    then split the new product uniformly. Hastings ratio = multiplier^2.
    Matches ExaBayes NodeSlider (weight=5.0).
    """
    def __init__(self, seed: int = 42, lambda_: float = 0.191):
        self.rng = np.random.RandomState(seed)
        self.lambda_ = lambda_

    def propose(self, tree: Tree) -> Tuple[Tree, float]:
        new_tree = tree.copy()
        # Pick a random internal non-root node
        internal_edges = [n for n in new_tree.internals if not n.is_root()]
        if not internal_edges:
            return new_tree, 0.0
        node = internal_edges[self.rng.randint(len(internal_edges))]

        # Pick one of its children at random
        if not node.children:
            return new_tree, 0.0
        child = node.children[self.rng.randint(len(node.children))]

        # Get the two branch lengths: node->parent and node->child
        old_a = node.branch_length  # node to its parent
        old_b = child.branch_length  # child to node

        # Multiply product by a log-uniform multiplier
        u = self.rng.uniform()
        multiplier = math.exp(self.lambda_ * (u - 0.5))
        new_product = old_a * old_b * multiplier

        # Split uniformly
        split = self.rng.uniform(0.1, 0.9)
        new_a = new_product ** split
        new_b = new_product ** (1 - split)

        # Clamp
        node.branch_length = clamp_branch_length(new_a)
        child.branch_length = clamp_branch_length(new_b)

        # Hastings ratio: multiplier^2 (ExaBayes NodeSlider.cpp line 138)
        log_hr = 2.0 * math.log(multiplier)
        return new_tree, log_hr

    def tune(self, acceptance_rate: float, batch: int = 0):
        delta = 1.0 / math.sqrt(batch + 1)
        log_lambda = math.log(self.lambda_)
        if acceptance_rate > 0.25:
            log_lambda += delta
        else:
            log_lambda -= delta
        new_lambda = math.exp(log_lambda)
        self.lambda_ = max(0.0001, min(new_lambda, 100.0))
```

Also update `MCMCChain.__init__` to include NodeSlider in the proposal set with weight 5.0:
```python
self.node_slider = NodeSlider(seed=seed + 7)
# Update mixer to include "ns" (node slider)
self.mixer = ProposalMixer(
    proposal_names=["nni", "spr", "bl", "tl", "sigma", "ns"],
    weights=[6.0, 6.0, 9.0, 1.0, 1.0, 5.0],
    seed=seed + 5,
)
```

And add the "ns" case in `_step()`:
```python
elif proposal_name == "ns":
    new_tree, log_hr = self.node_slider.propose(self.tree)
    new_sigma2 = self.sigma2
```

Add NodeSlider to the tuning block too.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): add NodeSlider proposal (ExaBayes port, weight=5)

NodeSlider redistributes branch lengths between adjacent edges without
changing topology. Improves MCMC mixing for branch length estimation.
ExaBayes default initial lambda=0.191, target acceptance=0.25."
```

---

### Task 4: Configurable prior and Exp(1) default for embedding data

**Context:** ExaBayes uses `Exponential(10)` as default prior on branch lengths (in substitution units, typical values 0.001–1.0). Our embedding distances are larger (posterior mean ~0.57), so `Exp(10)` is too strong. Our σ² prior is `LogNormal(0,1)` which is fine. Make the BL prior rate configurable and default to `Exp(1)` for embedding data.

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — `MCMCChain._compute_log_prior()`, `MCMCChain.__init__()`, argparse
- Test: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write failing test**

```python
class TestConfigurablePrior:
    def test_exp1_different_from_exp10(self):
        """Exp(1) and Exp(10) priors should give different log-prior values."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        lp_1 = MCMCChain._compute_log_prior(tree, 1.0, bl_prior_rate=1.0)
        lp_10 = MCMCChain._compute_log_prior(tree, 1.0, bl_prior_rate=10.0)
        assert lp_1 != lp_10
        # Exp(1) should be less penalizing for longer branches
        assert lp_1 > lp_10

    def test_chain_accepts_bl_prior_rate(self):
        """MCMCChain should accept bl_prior_rate parameter."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(tree, 1.0, 16, seed=42)
        chain = MCMCChain(
            data=data, start_tree=tree.copy(),
            sigma2_init=1.0, n_generations=100, sample_freq=50,
            seed=42, bl_prior_rate=1.0,
        )
        chain.run()
        assert len(chain.sampled_trees) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestConfigurablePrior -v`
Expected: FAIL — `_compute_log_prior` doesn't accept `bl_prior_rate`

- [ ] **Step 3: Implement configurable prior**

1. Add `bl_prior_rate` parameter to `MCMCChain.__init__()` (default=1.0):
```python
def __init__(self, ..., bl_prior_rate: float = 1.0):
    self.bl_prior_rate = bl_prior_rate
    ...
    self.log_prior = self._compute_log_prior(self.tree, self.sigma2, self.bl_prior_rate)
```

2. Update `_compute_log_prior` to accept and use `bl_prior_rate`:
```python
@staticmethod
def _compute_log_prior(tree: Tree, sigma2: float, bl_prior_rate: float = 1.0) -> float:
    log_p = 0.0
    rate = bl_prior_rate
    for node in tree.nodes:
        if not node.is_root():
            log_p += math.log(rate) - rate * node.branch_length
    log_p += -0.5 * (math.log(sigma2)) ** 2 - math.log(sigma2)
    return log_p
```

3. Update all calls to `_compute_log_prior` in `_step()` to pass `self.bl_prior_rate`.

4. Update `MC3Runner` and `MultiRunOrchestrator` to pass through `bl_prior_rate`.

5. Add `--bl-prior-rate` to argparse (default=1.0).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): configurable BL prior rate, default Exp(1)

ExaBayes uses Exp(10) for substitution-scale branch lengths. Our
embedding distances are ~5x larger, so Exp(1) is more appropriate.
Add --bl-prior-rate CLI arg. Exp(1) is less penalizing for the
0.1-1.0 branch lengths typical in embedding space."
```

---

### Task 5: Per-residue embedding phylogenetics mode

**Context:** Currently experiment 35 mean-pools per-residue embeddings to a single vector per protein (line 1472: `emb.mean(axis=0)`). The user believes per-residue embeddings should carry phylogenetic signal comparable to AA sequences. An MSA already exists at `SpeciesEmbedding/data/conotoxin_aligned.fasta` (created with MAFFT). We'll extract per-residue embeddings at aligned positions, treating each MSA column as independent BM characters. MAFFT is installed at `/opt/homebrew/bin/mafft`.

**Design:** Add `--per-residue` flag. When set:
1. Load MSA from FASTA (or run MAFFT on unaligned sequences)
2. For each protein, extract the embedding vector at each non-gap MSA position
3. After ABTT3+RP per column, each aligned position gives a 512d vector
4. Stack all positions: data is now `{protein: (n_aligned_cols, 512)}` matrix
5. The BM likelihood treats each column×dimension as an independent character
6. This gives ~60×512 = 30,720 characters vs current 512

The BMLikelihood already handles multi-dimensional data — we just need to reshape the data from `(512,)` to `(n_cols * 512,)` per protein. The BM model assumes independence across dimensions, which holds for our RP-projected data.

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — add MSA loading, per-residue data prep, `--per-residue` flag
- Test: `tests/test_embedding_phylo.py`

- [ ] **Step 1: Write failing tests**

```python
class TestPerResidueMode:
    def test_load_msa(self):
        """load_msa should parse aligned FASTA into {name: sequence} dict."""
        load_msa = _exp35.load_msa
        # Create a tiny test MSA
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">A\nACDE-FG\n>B\nAC-EEFG\n>C\nACDEEFG\n")
            tmp_path = f.name
        msa = load_msa(tmp_path)
        assert len(msa) == 3
        assert len(msa["A"]) == 7  # aligned length
        assert msa["A"][4] == "-"  # gap
        import os; os.unlink(tmp_path)

    def test_extract_aligned_embeddings(self):
        """extract_aligned_embeddings should map embeddings to aligned positions."""
        extract_aligned_embeddings = _exp35.extract_aligned_embeddings
        # Mock: protein "A" has 6 residues, aligned to 7 columns with 1 gap
        msa = {"A": "ACDE-FG", "B": "AC-EEFG"}
        embeddings = {
            "A": np.random.randn(6, 4),  # 6 residues, 4 dims
            "B": np.random.randn(6, 4),  # 6 residues, 4 dims
        }
        aligned = extract_aligned_embeddings(msa, embeddings, fill_value=0.0)
        # Both should have shape (7, 4) — aligned length
        assert aligned["A"].shape == (7, 4)
        assert aligned["B"].shape == (7, 4)
        # Gap positions should be fill_value (0.0)
        assert np.all(aligned["A"][4] == 0.0)  # column 4 is gap for A
        assert np.all(aligned["B"][2] == 0.0)  # column 2 is gap for B

    def test_bm_with_per_residue_data(self):
        """BM likelihood should work with higher-dimensional per-residue data."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        # 5 aligned columns × 4 dims = 20d vectors per protein
        rng = np.random.default_rng(42)
        data = {
            "A": rng.standard_normal(20),
            "B": rng.standard_normal(20),
            "C": rng.standard_normal(20),
        }
        bm = BMLikelihood()
        logL = bm.log_likelihood(tree, data, sigma2=1.0)
        assert np.isfinite(logL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding_phylo.py::TestPerResidueMode -v`
Expected: FAIL — `load_msa` and `extract_aligned_embeddings` not defined

- [ ] **Step 3: Implement MSA loading and per-residue extraction**

Add two functions to `experiments/35_embedding_phylogenetics.py`:

```python
def load_msa(fasta_path: str) -> Dict[str, str]:
    """Load aligned FASTA into {name: aligned_sequence} dict."""
    msa = {}
    current_name = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    msa[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                # Strip |orgXXX suffixes if present
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
        emb = embeddings[name]  # (L, D)
        out = np.full((aligned_len, D), fill_value, dtype=np.float32)
        residue_idx = 0
        for col_idx, char in enumerate(aligned_seq):
            if char != "-":
                if residue_idx < emb.shape[0]:
                    out[col_idx] = emb[residue_idx]
                residue_idx += 1
        result[name] = out
    return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embedding_phylo.py::TestPerResidueMode -v`
Expected: All 3 pass

- [ ] **Step 5: Add `--per-residue` and `--msa` CLI flags and wire into experiment main**

In the `if __name__ == "__main__"` block, add:
```python
parser.add_argument("--per-residue", action="store_true",
                    help="Use per-residue embeddings aligned by MSA")
parser.add_argument("--msa", type=str, default=None,
                    help="Path to aligned FASTA (default: auto-detect or run MAFFT)")
```

In Step 1 (data loading), add a branch for `--per-residue` mode:
```python
if args.per_residue:
    # Load per-residue embeddings (don't mean pool)
    embeddings_per_res = {}
    with h5py.File(EMB_PATH, "r") as f:
        for key in f.keys():
            embeddings_per_res[key] = np.array(f[key], dtype=np.float32)

    # Load or find MSA
    msa_path = args.msa or (SPECIES_ROOT / "data" / "conotoxin_aligned.fasta")
    msa = load_msa(str(msa_path))

    # Align embeddings to MSA
    aligned_embs = extract_aligned_embeddings(msa, embeddings_per_res)

    # Apply ABTT3 + RP per column, then flatten
    data = {}
    for pid, emb_matrix in aligned_embs.items():
        # ABTT3 on each column independently
        emb_abtt = all_but_the_top(emb_matrix, top3)
        # RP to 512d
        emb_rp = random_orthogonal_project(emb_abtt, d_out=512, seed=42)
        # Flatten: (n_cols, 512) -> (n_cols * 512,)
        data[pid] = emb_rp.flatten().astype(np.float64)

    print(f"  Per-residue mode: {next(iter(data.values())).shape[0]}d "
          f"({aligned_embs[list(aligned_embs.keys())[0]].shape[0]} cols × 512)")
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/test_embedding_phylo.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add experiments/35_embedding_phylogenetics.py tests/test_embedding_phylo.py
git commit -m "feat(exp35): per-residue embedding phylogenetics via MSA alignment

Add --per-residue mode: loads MSA (MAFFT-aligned FASTA), maps per-residue
embeddings to aligned positions, applies ABTT3+RP per column, then
flattens to (n_cols × 512)d BM characters. Each aligned position becomes
independent Brownian motion characters, analogous to amino acid columns
in traditional phylogenetics.

Adds load_msa() and extract_aligned_embeddings() functions."
```

---

### Task 6: Run experiment and compare per-protein vs per-residue

**Context:** Run both modes on the conotoxin dataset and compare RF distances to IQ-TREE reference.

**Files:**
- Modify: `experiments/35_embedding_phylogenetics.py` — add comparison output
- No new tests (this is the experiment itself)

- [ ] **Step 1: Run per-protein mode with new fixes (shorter run for quick test)**

```bash
uv run python experiments/35_embedding_phylogenetics.py \
    --n-gen 200000 --bl-prior-rate 1.0
```

Expected: Completes in ~5min, prints RF distance and clade recovery.

- [ ] **Step 2: Run per-residue mode**

```bash
uv run python experiments/35_embedding_phylogenetics.py \
    --n-gen 200000 --bl-prior-rate 1.0 --per-residue
```

Expected: Completes (slower due to higher dimensionality), prints RF distance.

- [ ] **Step 3: Compare results**

Check the benchmark JSON for both runs. Compare:
- RF distance to IQ-TREE
- Clade recovery
- Convergence (ASDSF)
- Runtime

- [ ] **Step 4: Commit results**

```bash
git add data/benchmarks/embedding_phylo_results.json results/embed_phylo/
git commit -m "data(exp35): per-protein vs per-residue phylogenetics comparison"
```
