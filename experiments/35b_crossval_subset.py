#!/usr/bin/env python3
"""Cross-validation: run our BM MCMC on the same 14-taxa × 10d subset as RevBayes.

Uses n_runs=1 to avoid multiprocessing pickle issues, then runs a second
independent chain sequentially for ASDSF comparison.
"""

import importlib.util
import numpy as np
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# Import classes from experiment 35 (single-file module)
spec = importlib.util.spec_from_file_location(
    "exp35", PROJ_ROOT / "experiments" / "35_embedding_phylogenetics.py"
)
exp35 = importlib.util.module_from_spec(spec)
# Only exec up to the __main__ guard
import types
source = (PROJ_ROOT / "experiments" / "35_embedding_phylogenetics.py").read_text()
code_block = source.split("if __name__")[0]
exec(compile(code_block, str(PROJ_ROOT / "experiments" / "35_embedding_phylogenetics.py"), "exec"),
     exp35.__dict__)

NJBuilder = exp35.NJBuilder
write_newick = exp35.write_newick
_run_single_mc3 = exp35._run_single_mc3
ConsensusBuilder = exp35.ConsensusBuilder
Diagnostics = exp35.Diagnostics
random_tree = exp35.random_tree

# Read the subset data (same as RevBayes input)
taxa_data = {}
with open(PROJ_ROOT / "data/phylo_benchmark/revbayes_subset.tsv") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        pid = parts[0]
        vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        taxa_data[pid] = vals

print(f"Loaded {len(taxa_data)} taxa, {next(iter(taxa_data.values())).shape[0]}d")

# NJ tree
nj = NJBuilder.from_embeddings(taxa_data)
nj_nwk = write_newick(nj)
print(f"NJ: {nj_nwk[:80]}...")

# Run 2 independent MCMC chains sequentially
names = sorted(taxa_data.keys())
run_results = []
for run_idx in range(2):
    print(f"\n--- Run {run_idx} ---")
    start_tree = nj if run_idx == 0 else random_tree(names, seed=42 + run_idx * 1000)
    t0 = time.time()
    rr = _run_single_mc3({
        "data": taxa_data, "n_chains": 4,
        "n_generations": 50000, "sample_freq": 100,
        "swap_freq": 50, "delta": 0.1,
        "start_tree": start_tree,
        "seed": 42 + run_idx * 100,
        "bl_prior_rate": 1.0,
    })
    print(f"  Run {run_idx}: {time.time() - t0:.1f}s, "
          f"{len(rr['sampled_trees'])} samples, final logL={rr['sampled_logL'][-1]:.1f}")
    run_results.append(rr)

# Diagnostics
tree_sets = [rr["sampled_trees"] for rr in run_results]
asdsf = Diagnostics.asdsf(tree_sets)
print(f"\nASDSF: {asdsf:.4f}")

for pname in ["sampled_logL", "sampled_sigma2", "sampled_tree_length"]:
    chains = [np.array(rr[pname]) for rr in run_results]
    ess = [Diagnostics.effective_sample_size(c[len(c) // 4 :]) for c in chains]
    psrf = Diagnostics.psrf([c[len(c) // 4 :] for c in chains])
    print(f"  {pname}: ESS={[f'{e:.0f}' for e in ess]}, PSRF={psrf:.3f}")

# Consensus
all_trees = []
for rr in run_results:
    all_trees.extend(rr["sampled_trees"])
consensus = ConsensusBuilder.majority_rule(all_trees, burnin_frac=0.25)
nwk = write_newick(consensus)

RESULTS = PROJ_ROOT / "results" / "embed_phylo"
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "crossval_python_consensus.nwk").write_text(nwk)
(RESULTS / "crossval_python_nj.nwk").write_text(nj_nwk)
print(f"\nConsensus: {nwk[:100]}...")

# sigma2 posterior
sigma2_post = []
for rr in run_results:
    s = rr["sampled_sigma2"]
    sigma2_post.extend(s[len(s) // 4 :])
print(f"sigma2: mean={np.mean(sigma2_post):.4f}, median={np.median(sigma2_post):.4f}, "
      f"95%CI=[{np.percentile(sigma2_post, 2.5):.4f}, {np.percentile(sigma2_post, 97.5):.4f}]")
