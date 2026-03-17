# Experiment 35: Bayesian Phylogenetics from Protein Embeddings

**Date:** 2026-03-17
**Status:** Design
**Approach:** Re-implement ExaBayes core in Python with Brownian motion likelihood for continuous embedding data

## Motivation

Protein Language Model embeddings encode evolutionary information in continuous vector space. If embeddings reflect evolutionary relationships, we should be able to reconstruct phylogenetic trees directly from them — without sequence alignment. This is a novel approach: no existing phylogenetic software supports Bayesian inference on high-dimensional continuous character data at our scale (512 dimensions).

### Why Not Existing Tools?

- **MrBayes**: `datatype=continuous` was never implemented ([Issue #135](https://github.com/NBISweden/MrBayes/issues/135))
- **ExaBayes**: C++, discrete characters only (DNA/protein substitution models), no continuous support
- **RevBayes**: Has multivariate BM but scales O(D²) — 512 dims = 124,750 correlation parameters
- **BEAST 2 contraband**: Java, closest to what we need but not designed for 512d embeddings

### Why ExaBayes as Blueprint?

ExaBayes has a clean, well-documented MCMC architecture: proposal mixtures with auto-tuning, MC3 heated chains, convergence diagnostics (ASDSF/ESS/PSRF), and checkpointing. We port the algorithmic design to Python, replacing the sequence likelihood with a vectorized Brownian motion model.

## Model

### Data

Each taxon is represented by a single D-dimensional vector (mean pool of per-residue embeddings). The BM model operates on these protein-level vectors; per-residue structure is not used during tree inference.

- N taxa × D dimensions (e.g., 40 conotoxins × 512d)
- Preprocessing: ABTT3 + random projection to 512d (from One Embedding V1/V2 pipeline)
  - ABTT3 corpus statistics come from the pre-existing 5K SCOPe corpus (not the small per-dataset set), ensuring stable PC estimates
- Can also work on raw mean-pool vectors (1024d or 2048d)

### Data Preparation (Conotoxin Test Case)

Conotoxin data lives in SpeciesEmbedding (cross-project reference):
- Embeddings: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/data/conotoxin_embeddings.h5` (40 proteins, shape (L, 1024) each)
- MSA: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/data/conotoxin_aligned.fasta`
- IQ-TREE reference: `/Users/jcoludar/CascadeProjects/SpeciesEmbedding/results/iqtree_conotoxin.treefile`

The experiment script uses `sys.path.insert` to reference SpeciesEmbedding paths (standard pattern for this project). Mean-pool each protein's (L, 1024) embeddings to (1024,), then apply ABTT3+RP512 → (512,) per taxon.

### Evolution Model

Independent Brownian motion per embedding dimension. Justified because:
1. Random projection decorrelates dimensions (by construction — JL lemma preserves inner products)
2. ABTT3 removes dominant PCs that would create inter-dimension dependence
3. Independence assumption reduces parameters from O(D²) to O(D) or O(1)

A single shared σ² is used across all dimensions. This is justified by the approximate isotropy of the RP-projected space (all dimensions have similar variance by construction). Per-dimension σ²_d is a future extension if diagnostics reveal rate heterogeneity across dimensions.

Under BM, the embedding value at each tip is normally distributed:
- `X_tip ~ N(X_root, σ² · t_root→tip)`
- Covariance between tips i,j: `Cov(X_i, X_j) = σ² · t_shared(i,j)`

### Parameters Sampled

| Parameter | Prior | Proposal |
|-----------|-------|----------|
| Tree topology τ | Uniform over topologies | stNNI (weight 6), eSPR (weight 6) |
| Branch lengths {t_e} | Exponential(10) per branch | Gamma multiplier (weight 9) |
| Tree length (sum of branches) | — | Joint multiplier (weight 1) |
| BM rate σ² | LogNormal(0, 1) | Log-normal multiplier (weight 1) |

### Likelihood

Felsenstein pruning for continuous characters, vectorized across all D dimensions:

At each leaf node:
- `μ_d = observed_embedding[d]` for each dimension d
- `v = 0`

At each internal node with children L, R (where `s_L = v_L + t_L` and `s_R = v_R + t_R`):
- `v = s_L · s_R / (s_L + s_R)` (combined partial variance, propagated up)
- `μ_d = (μ_L_d · s_R + μ_R_d · s_L) / (s_L + s_R)` (weighted mean, per dim)
- Per-node log-likelihood contribution (summed over D dimensions):
  `logL += -0.5 · D · log(2π · σ² · (s_L + s_R)) - 0.5 · Σ_d (μ_L_d - μ_R_d)² / (σ² · (s_L + s_R))`

The `Σ_d (μ_L_d - μ_R_d)²` term is just `||μ_L - μ_R||²` — a single numpy vector norm.

**Root treatment (flat prior):** The root state is integrated out analytically (improper flat prior). Under this treatment, the pruning algorithm already yields the marginal likelihood — no additional term is needed at the root. The final `(μ, v)` at the root are unused for the likelihood; only the per-node contributions matter. This is the standard Felsenstein (1973) approach.

Total log-likelihood = sum over all N-1 internal nodes. With numpy broadcasting, the inner loop over 512 dimensions disappears into a single vector operation per node.

**Complexity**: O(N · D) per likelihood evaluation, where N = number of taxa and D = embedding dimensionality. For 40 taxa × 512 dims, each evaluation is ~20K multiply-adds — microseconds on modern hardware.

## Architecture

Everything lives in `experiments/35_embedding_phylogenetics.py` as a self-contained experiment. Classes are designed for later extraction to `tools/pipelines/embed_phylo/`.

### Classes

```
TreeNode          — Dataclass: id, children, parent, branch_length, name, leaf data
Tree              — Rooted binary tree: nodes list, leaf lookup, newick I/O, copy
NJBuilder         — Neighbor-joining from distance matrix (numpy, no external dep)
BMLikelihood      — Vectorized pruning: log_likelihood(tree, data, sigma2)
Proposal (ABC)    — Base: propose(tree) → (new_tree, log_hastings_ratio)
  ├─ StochasticNNI
  ├─ ExtendingSPR
  ├─ BranchLengthMultiplier
  ├─ TreeLengthMultiplier
  └─ SigmaMultiplier
ProposalMixer     — Weighted selection, acceptance tracking, auto-tuning
MCMCChain         — Single chain: state, likelihood, generation counter
MC3Runner         — Metropolis-coupled: N heated chains (sequential in-process), swap proposals
MultiRunOrchestrator — M independent MC3 runs via ProcessPoolExecutor (1 process per run)
Diagnostics       — ASDSF, ESS, PSRF computation
TreeSampler       — Write sampled trees + parameters to files
ConsensusBuilder  — Majority-rule consensus from posterior sample
```

### Parallelism

Two-level parallelism, ported from ExaBayes MC3 design:

**Level 1 — Across runs:** `ProcessPoolExecutor(max_workers=n_runs)`. Each run is fully independent — no shared state, no synchronization. This is trivially parallel.

**Level 2 — Within a run (MC3):** Chains run sequentially within a single process. Since each likelihood evaluation is ~4 microseconds (40 taxa × 512d), the bottleneck is generation count, not per-chain compute. Running 4 chains sequentially within a process is fast and avoids inter-process synchronization complexity for MC3 swaps.

```
ProcessPoolExecutor(max_workers=n_runs)  # e.g., 2-4 workers

Run 0 (1 process):
  for gen in range(n_generations):
    for chain in [Cold β=1.00, Hot β=0.91, Hot β=0.83, Hot β=0.77]:
      propose + accept/reject
    if gen % swap_freq == 0:
      attempt_adjacent_swap()  # in-process, no IPC needed

Run 1 (1 process):  # same, independent
  ...
```

```
Heating: β_i = 1 / (1 + i · δ),  δ = 0.1
→ β = [1.00, 0.91, 0.83, 0.77] for 4 chains
```

- Swap acceptance: `min(1, exp((β_i - β_j) · (logL_i - logL_j)))`
- Only cold chains (β=1) contribute posterior samples
- Run 0 starts from NJ tree; remaining runs start from random trees

Default: 2 runs × 4 chains = 2 processes, 8 chains total. Can scale to 4-10 runs on M3 Max (14 cores) for better ASDSF convergence.

### Proposal Details (from ExaBayes)

**stNNI (Stochastic Nearest-Neighbor Interchange):**
1. Pick random internal edge
2. Swap one subtree from each side
3. Hastings ratio = 1 (symmetric)

**eSPR (Extending Subtree Prune-Regraft):**
1. Pick random subtree to prune
2. Walk along the tree from prune point with stopping probability p=0.5 per step
3. Regraft at landing point
4. Hastings ratio accounts for path length asymmetry

**Branch Length Multiplier:**
1. Pick random branch
2. Propose `t' = t · exp(λ · (U - 0.5))` where U ~ Uniform(0,1)
3. Hastings ratio = t'/t (Jacobian of log transform)
4. λ auto-tuned for ~30% acceptance

**Tree Length Multiplier:**
1. Scale all branch lengths by `c = exp(λ · (U - 0.5))`
2. Hastings ratio = c^(n_branches)

**σ² Multiplier:**
1. `σ'² = σ² · exp(λ · (U - 0.5))`
2. Hastings ratio = σ'²/σ²

### Convergence Diagnostics (from ExaBayes)

- **ASDSF**: Average standard deviation of split frequencies across runs. Target < 0.01 (strict) or < 0.05 (acceptable).
- **ESS**: Effective sample size for σ², tree length, log-likelihood. Target > 200.
- **PSRF**: Potential scale reduction factor (Gelman-Rubin). Target < 1.1.
- Checked every `diag_freq` generations (default: 5000).
- Burn-in: 25% of samples discarded.

### Auto-Tuning (from ExaBayes)

After each proposal type has been drawn 100 times:
- If acceptance rate > target + 0.05: increase step size (bolder moves)
- If acceptance rate < target - 0.05: decrease step size (more conservative)
- Target acceptance: 25% for topology, 30% for branch lengths, 35% for σ²

## Starting Trees

- **Run 0**: Neighbor-Joining tree from pairwise embedding distances (euclidean on protein_vec or mean-pool). Provides a reasonable starting topology for faster convergence.
- **Runs 1+**: Random tree (uniform random topology with exponential branch lengths). Ensures exploration from diverse starting points.

NJ implementation: pure numpy, Saitou-Nei algorithm. No external dependency.

## Benchmark Against IQ-TREE

To validate that our Bayesian embedding trees are biologically meaningful:

1. **Reference tree**: IQ-TREE ML tree from sequence alignment (standard phylogenetics)
2. **Our tree**: EmbedTree consensus from embedding data (no alignment needed)
3. **Comparison metrics**:
   - Robinson-Foulds distance (topology similarity)
   - Weighted RF distance (branch length similarity)
   - Clade recovery: fraction of IQ-TREE clades found in our consensus
   - Visual comparison: tanglegram (two trees side by side)

We already have `IQTreeRunner` in `tools/pipelines/phylo_scaffold.py`.

### Test Dataset: Conotoxins

- 40 conotoxin proteins with known superfamily classification
- ProtT5 embeddings already extracted
- MSA already available (`data/conotoxin_aligned.fasta`)
- NEXUS files already generated (from previous `embed_to_nexus.py` work)
- IQ-TREE results available (`results/iqtree_conotoxin.splits.nex`)

## Output Files

```
data/benchmarks/embedding_phylo_results.json     — benchmark metrics
results/embed_phylo/
├── conotoxin_consensus.nwk                       — majority-rule consensus tree
├── conotoxin_trees_run{0,1}.nwk                  — sampled trees per run
├── conotoxin_params_run{0,1}.tsv                 — parameter traces
├── conotoxin_diagnostics.json                    — ASDSF, ESS, PSRF
└── conotoxin_tanglegram.png                      — IQ-TREE vs EmbedTree comparison
```

## Implementation Milestones

### M1: BM Likelihood on Fixed Tree
- Implement Tree, TreeNode, BMLikelihood
- Load conotoxin embeddings
- Compute log-likelihood on NJ tree
- Verify: vectorized matches naive loop, sensible values

### M2: Single-Chain MCMC
- Implement stNNI, eSPR, branch length proposals
- Single Metropolis-Hastings chain
- Verify: chain mixes, acceptance rates reasonable, trace plots show convergence

### M3: MC3 + Multi-Run
- Add heated chains with ProcessPoolExecutor
- Multiple independent runs
- Convergence diagnostics (ASDSF, ESS, PSRF)
- Auto-tuning

### M4: Consensus + Benchmark
- Majority-rule consensus tree builder
- Robinson-Foulds comparison against IQ-TREE
- Tanglegram visualization
- Full benchmark results

## Dependencies

All already in the project's environment:
- `numpy` — vectorized likelihood, NJ, linear algebra
- `scipy.stats` — proposal distributions
- `concurrent.futures` — ProcessPoolExecutor for parallelism
- `h5py` — embedding I/O
- `sklearn.decomposition` — optional PCA
- `matplotlib` — diagnostic plots, tanglegram

Robinson-Foulds distance is implemented from scratch (compare bipartition sets of two unrooted trees — straightforward, ~30 lines). No `ete3` or `dendropy` dependency needed. Newick parser handles pipe characters in taxon names (e.g., `A0A1P8NVR5|org101286`) by quoting names.

No new dependencies required.

## Testing

Consistent with project conventions (318 tests across 13 modules), key correctness tests:

### Unit Tests
- **BM likelihood**: 3-taxon tree with known analytical solution — verify vectorized implementation matches hand calculation
- **BM likelihood**: Vectorized (numpy) matches naive per-node loop, per-dimension loop
- **NJ builder**: Known 4-taxon distance matrix → verify correct topology (textbook example)
- **Newick I/O**: Round-trip parse → write → parse preserves topology, branch lengths, and names with special characters
- **Proposals**: Each proposal is reversible — propose then reverse yields original tree (test stNNI, eSPR)
- **Hastings ratios**: Verify detailed balance for branch length multiplier on a simple case
- **RF distance**: Known tree pair → verify correct symmetric difference count

### Integration Tests
- **Synthetic recovery**: Generate data from a known 8-taxon tree under BM, run MCMC, verify the true topology is in the 95% credible set
- **Convergence**: Two independent runs on same data → ASDSF < 0.05 within budget

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| BM model too simple for embedding evolution | Start with BM, can extend to OU (mean-reverting) later |
| MCMC doesn't converge in 512d | Independence assumption makes it 512 × 1d problems, not one 512d problem |
| Trees don't match sequence-based trees | This IS the experiment — negative result is still publishable |
| Thermal throttling on M3 Max with 8 processes | Monitor with pmset, can reduce to 2 runs × 2 chains |

## Future Extensions (Not in This Experiment)

- Ornstein-Uhlenbeck model (stabilizing selection toward optimal embedding)
- Per-clade rate variation (relaxed clock analog)
- Multiple PLMs as partitions (like gene partitions in molecular phylogenetics)
- Marginal likelihood estimation for model comparison (stepping-stone sampling)
