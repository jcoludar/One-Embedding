#!/usr/bin/env python3
"""Experiment 47: Codec Configuration Sweep.

Tests multiple compression configs per PLM on the rigorous benchmark suite.
Raw benchmarks are computed once per PLM and cached; only the compressed
side is re-run per config.

Config sets:
  --configs standard  6 tiers: lossless, fp16-896, int4-896, pq224, pq128, binary
  --configs useful    10 configs: standard + old default (ABTT3+768d) + no-RP (1024d PQ)
  --configs retest    3 configs: VQ/RVQ retests from bugged Exp 33 (confirmed bad)
  --configs all       13 configs: everything

Each config is a FittedCodec: corpus stats + codebook fitted ONCE on SCOPe train,
then applied to CB513 (SS3/SS8), SCOPe (retrieval), CheZOD (disorder).

Usage:
    uv run python experiments/47_codec_sweep.py --plm prot_t5_full --smoke-test
    uv run python experiments/47_codec_sweep.py --plm prot_t5_full esm2_650m --configs useful
    uv run python experiments/47_codec_sweep.py  # all PLMs, useful configs

    # Overnight full run:
    nohup uv run python -u experiments/47_codec_sweep.py > results/exp47_full.log 2>&1 &
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "43_rigorous_benchmark"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

DATA = ROOT / "data"
RESULTS_DIR = DATA / "benchmarks" / "rigorous_v1"

SEEDS = [42, 123, 456]
BOOTSTRAP_N = 10_000
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CV_FOLDS = 3


# ═══════════════════════════════════════════════════════════════════════
# Codec Configurations
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CodecConfig:
    name: str
    d_out: int
    quantization: str  # "none", "int4", "pq", "binary", "vq", "rvq"
    pq_m: int = 0
    abtt_k: int = 0
    vq_k: int = 0      # for VQ: number of centroids
    rvq_levels: int = 0  # for RVQ: number of residual levels
    description: str = ""

    @property
    def compression_ratio(self) -> str:
        """Approximate compression ratio vs raw fp32."""
        raw = 1024 * 4  # fp32 baseline
        if self.quantization == "none":
            return f"{raw / (self.d_out * 2):.0f}x"  # fp16
        elif self.quantization == "int4":
            return f"{raw / (self.d_out // 2):.0f}x"
        elif self.quantization == "pq":
            return f"{raw / self.pq_m:.0f}x"
        elif self.quantization == "binary":
            return f"{raw / (self.d_out // 8):.0f}x"
        elif self.quantization == "vq":
            # VQ: 2 bytes per residue (codebook index)
            return f"{raw / 2:.0f}x"
        elif self.quantization == "rvq":
            return f"{raw / (2 * self.rvq_levels):.0f}x"
        return "?"


# Standard compression tiers (new default pipeline: center only, no ABTT)
STANDARD_CONFIGS = [
    CodecConfig("lossless-1024", 1024, "none", abtt_k=0,
                description="No RP, fp16, lossless"),
    CodecConfig("fp16-896", 896, "none", abtt_k=0,
                description="RP 896d, fp16"),
    CodecConfig("int4-896", 896, "int4", abtt_k=0,
                description="RP 896d, int4"),
    CodecConfig("pq224-896", 896, "pq", pq_m=224, abtt_k=0,
                description="DEFAULT: RP 896d, PQ M=224 (~18x)"),
    CodecConfig("pq128-896", 896, "pq", pq_m=128, abtt_k=0,
                description="RP 896d, PQ M=128 (~32x)"),
    CodecConfig("binary-896", 896, "binary", abtt_k=0,
                description="RP 896d, binary (~37x)"),
]

# Old default for comparison
OLD_DEFAULT_CONFIGS = [
    CodecConfig("old-pq192-768-abtt3", 768, "pq", pq_m=192, abtt_k=3,
                description="OLD DEFAULT: ABTT3 + RP 768d + PQ M=192 (~21x)"),
    CodecConfig("old-fp16-768-abtt3", 768, "none", abtt_k=3,
                description="Old: ABTT3 + RP 768d + fp16"),
]

# No-RP configs (keep full dimensionality, PQ directly on 1024d)
NO_RP_CONFIGS = [
    CodecConfig("pq256-1024", 1024, "pq", pq_m=256, abtt_k=0,
                description="No RP, PQ M=256 on 1024d (~16x)"),
    CodecConfig("pq128-1024", 1024, "pq", pq_m=128, abtt_k=0,
                description="No RP, PQ M=128 on 1024d (~32x)"),
]

# Retested "failures" from earlier experiments (affected by ABTT centering bug)
RETEST_CONFIGS = [
    CodecConfig("vq4096-896", 896, "vq", vq_k=4096, abtt_k=0,
                description="RETEST: VQ K=4096 on centered 896d (was bugged in Exp 33)"),
    CodecConfig("vq16384-896", 896, "vq", vq_k=16384, abtt_k=0,
                description="RETEST: VQ K=16384 on centered 896d (was bugged in Exp 33)"),
    CodecConfig("rvq2-896", 896, "rvq", vq_k=4096, rvq_levels=2, abtt_k=0,
                description="RETEST: RVQ 2-level on centered 896d"),
]

USEFUL_CONFIGS = STANDARD_CONFIGS + OLD_DEFAULT_CONFIGS + NO_RP_CONFIGS
ALL_CONFIGS = USEFUL_CONFIGS + RETEST_CONFIGS


# ═══════════════════════════════════════════════════════════════════════
# Import PLM registry from Exp 46
# ═══════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(ROOT / "experiments"))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("exp46", str(ROOT / "experiments" / "46_multi_plm_benchmark.py"))
exp46 = module_from_spec(_spec)
_spec.loader.exec_module(exp46)

PLMS = exp46.PLMS
emb_path = exp46.emb_path
load_h5 = exp46.load_h5


# ═══════════════════════════════════════════════════════════════════════
# Compression functions
# ═══════════════════════════════════════════════════════════════════════

class FittedCodec:
    """A fitted codec config — corpus stats, projection, and codebook ready to apply."""

    def __init__(self, config: CodecConfig, train_embs: dict, plm_dim: int):
        from src.one_embedding.preprocessing import compute_corpus_stats
        from src.one_embedding.quantization import pq_fit, rvq_fit
        from sklearn.cluster import MiniBatchKMeans

        self.config = config
        self.plm_dim = plm_dim

        # Corpus stats from training data
        self.cs = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)

        # RP projection matrix
        self.proj = None
        if config.d_out < plm_dim:
            rng = np.random.RandomState(42)
            R = rng.randn(plm_dim, config.d_out).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            self.proj = Q * np.sqrt(plm_dim / config.d_out)

        # Preprocess training data for codebook fitting
        proc_train = self._preprocess_dict(train_embs)

        # Fit quantization codebook
        self.pq_model = None
        self.vq_codebook = None
        self.vq_km = None
        self.rvq_model = None

        if config.quantization == "pq":
            self.pq_model = pq_fit(proc_train, M=config.pq_m, n_centroids=256,
                                   max_residues=500_000, seed=42)
        elif config.quantization == "vq":
            train_stack = np.concatenate(
                [proc_train[pid][:512] for pid in list(proc_train.keys())[:500]],
                axis=0,
            )
            if len(train_stack) > 500_000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(train_stack), 500_000, replace=False)
                train_stack = train_stack[idx]
            self.vq_km = MiniBatchKMeans(
                n_clusters=config.vq_k, batch_size=4096,
                random_state=42, n_init=3, max_iter=100,
            )
            self.vq_km.fit(train_stack)
            self.vq_codebook = self.vq_km.cluster_centers_.astype(np.float32)
        elif config.quantization == "rvq":
            self.rvq_model = rvq_fit(proc_train, n_levels=config.rvq_levels,
                                     n_centroids=config.vq_k,
                                     max_residues=500_000, seed=42)

    def _preprocess_one(self, emb: np.ndarray) -> np.ndarray:
        from src.one_embedding.preprocessing import center_embeddings, all_but_the_top
        e = center_embeddings(emb, self.cs["mean_vec"])
        if self.config.abtt_k > 0:
            e = all_but_the_top(e, self.cs["top_pcs"][:self.config.abtt_k])
        if self.proj is not None:
            e = e @ self.proj
        return e

    def _preprocess_dict(self, embs: dict) -> dict:
        return {pid: self._preprocess_one(e) for pid, e in embs.items()}

    def compress(self, embs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Preprocess + quantize + dequantize a dict of embeddings."""
        from src.one_embedding.quantization import (
            quantize_int4, dequantize_int4,
            quantize_binary, dequantize_binary,
            pq_encode, pq_decode,
            rvq_encode, rvq_decode,
        )
        proc = self._preprocess_dict(embs)
        q = self.config.quantization

        if q == "none":
            return {pid: e.astype(np.float16).astype(np.float32) for pid, e in proc.items()}
        elif q == "int4":
            return {pid: dequantize_int4(quantize_int4(e)) for pid, e in proc.items()}
        elif q == "binary":
            return {pid: dequantize_binary(quantize_binary(e)) for pid, e in proc.items()}
        elif q == "pq":
            return {pid: pq_decode(pq_encode(e, self.pq_model), self.pq_model)
                    for pid, e in proc.items()}
        elif q == "vq":
            return {pid: self.vq_codebook[self.vq_km.predict(e)]
                    for pid, e in proc.items()}
        elif q == "rvq":
            return {pid: rvq_decode(rvq_encode(e, self.rvq_model), self.rvq_model)
                    for pid, e in proc.items()}
        raise ValueError(f"Unknown quantization: {q}")


# ═══════════════════════════════════════════════════════════════════════
# Sweep runner
# ═══════════════════════════════════════════════════════════════════════

def run_sweep(plm_name: str, configs: list[CodecConfig],
              bootstrap_n: int = BOOTSTRAP_N) -> list[dict]:
    """Run codec sweep for one PLM. Computes raw benchmarks once, then
    runs each config's compressed benchmarks."""
    from src.evaluation.per_residue_tasks import load_cb513_csv, load_chezod_seth
    from src.extraction.data_loader import load_metadata_csv
    from runners.per_residue import (
        run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark, pooled_spearman,
    )
    from runners.protein_level import run_retrieval_benchmark, compute_protein_vectors
    from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention

    plm = PLMS[plm_name]

    # ── Load all data once ──
    print(f"  Loading data...", flush=True)
    _, ss3_labels, ss8_labels, _ = load_cb513_csv(DATA / "per_residue_benchmarks" / "CB513.csv")
    with open(DATA / "benchmark_suite" / "splits" / "cb513_80_20.json") as f:
        cb_split = json.load(f)
    with open(DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json") as f:
        sc_split = json.load(f)
    metadata = load_metadata_csv(DATA / "proteins" / "metadata_5k.csv")
    _, cz_scores, cz_train_ids, cz_test_ids = load_chezod_seth(DATA / "per_residue_benchmarks")

    cb_embs = load_h5(emb_path(plm_name, "cb513"))
    sc_embs = load_h5(emb_path(plm_name, "scope_5k"))
    cz_embs = load_h5(emb_path(plm_name, "chezod"))

    cb_train = [p for p in cb_split["train_ids"] if p in cb_embs and p in ss3_labels]
    cb_test = [p for p in cb_split["test_ids"] if p in cb_embs and p in ss3_labels]
    sc_train = [k for k in sc_split["train_ids"] if k in sc_embs]
    sc_test = [k for k in sc_split["test_ids"] if k in sc_embs]
    cz_train = [p for p in cz_train_ids if p in cz_embs and p in cz_scores]
    cz_test = [p for p in cz_test_ids if p in cz_embs and p in cz_scores]

    sc_train_embs = {k: sc_embs[k] for k in sc_train}

    # ── Raw benchmarks (computed once) ──
    print(f"  Raw benchmarks...", flush=True)
    ss3_raw = run_ss3_benchmark(cb_embs, ss3_labels, cb_train, cb_test,
                                 C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
    ss8_raw = run_ss8_benchmark(cb_embs, ss8_labels, cb_train, cb_test,
                                 C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
    raw_vecs = compute_protein_vectors(sc_embs, method="dct_k4")
    ret_raw = run_retrieval_benchmark(raw_vecs, metadata, label_key="family",
                                       n_bootstrap=bootstrap_n, seed=SEEDS[0])
    dis_raw = run_disorder_benchmark(cz_embs, cz_scores, cz_train, cz_test,
                                      ALPHA_GRID, CV_FOLDS, SEEDS, bootstrap_n)
    print(f"  Raw: SS3={ss3_raw['q3'].value:.4f} SS8={ss8_raw['q8'].value:.4f} "
          f"Ret={ret_raw['ret1_cosine'].value:.4f} Dis={dis_raw['pooled_spearman_rho'].value:.4f}",
          flush=True)

    # ── Per-config compressed benchmarks ──
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {cfg.name} ({cfg.description})", flush=True)
        t0 = time.time()

        try:
            # Fit codec once, apply to all datasets
            fitted = FittedCodec(cfg, sc_train_embs, plm.dim)
            comp_cb = fitted.compress(cb_embs)
            comp_sc = fitted.compress(sc_embs)
            comp_cz = fitted.compress(cz_embs)

            # SS3
            ss3_comp = run_ss3_benchmark(comp_cb, ss3_labels, cb_train, cb_test,
                                          C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
            ss3_ret = paired_bootstrap_retention(
                ss3_raw["per_protein_scores"], ss3_comp["per_protein_scores"],
                n_bootstrap=bootstrap_n, seed=SEEDS[0])

            # SS8
            ss8_comp = run_ss8_benchmark(comp_cb, ss8_labels, cb_train, cb_test,
                                          C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
            ss8_ret = paired_bootstrap_retention(
                ss8_raw["per_protein_scores"], ss8_comp["per_protein_scores"],
                n_bootstrap=bootstrap_n, seed=SEEDS[0])

            # Retrieval
            comp_vecs = compute_protein_vectors(comp_sc, method="dct_k4")
            ret_comp = run_retrieval_benchmark(comp_vecs, metadata, label_key="family",
                                                n_bootstrap=bootstrap_n, seed=SEEDS[0])
            ret_ret = paired_bootstrap_retention(
                ret_raw["per_query_cosine"], ret_comp["per_query_cosine"],
                n_bootstrap=bootstrap_n, seed=SEEDS[0])

            # Disorder
            dis_comp = run_disorder_benchmark(comp_cz, cz_scores, cz_train, cz_test,
                                               ALPHA_GRID, CV_FOLDS, SEEDS, bootstrap_n)
            dis_ret = paired_cluster_bootstrap_retention(
                dis_raw["per_protein_predictions"], dis_comp["per_protein_predictions"],
                pooled_spearman, n_bootstrap=bootstrap_n, seed=SEEDS[0])

            r = {
                "config": cfg.name, "description": cfg.description,
                "d_out": cfg.d_out, "quantization": cfg.quantization,
                "pq_m": cfg.pq_m, "abtt_k": cfg.abtt_k,
                "vq_k": cfg.vq_k, "rvq_levels": cfg.rvq_levels,
                "compression": cfg.compression_ratio,
                "ss3": {"raw": ss3_raw["q3"].value, "comp": ss3_comp["q3"].value,
                        "retention": ss3_ret.value, "ci": [ss3_ret.ci_lower, ss3_ret.ci_upper]},
                "ss8": {"raw": ss8_raw["q8"].value, "comp": ss8_comp["q8"].value,
                        "retention": ss8_ret.value, "ci": [ss8_ret.ci_lower, ss8_ret.ci_upper]},
                "ret1": {"raw": ret_raw["ret1_cosine"].value, "comp": ret_comp["ret1_cosine"].value,
                         "retention": ret_ret.value, "ci": [ret_ret.ci_lower, ret_ret.ci_upper]},
                "disorder": {"raw": dis_raw["pooled_spearman_rho"].value,
                             "comp": dis_comp["pooled_spearman_rho"].value,
                             "retention": dis_ret.value,
                             "ci": [dis_ret.ci_lower, dis_ret.ci_upper]},
                "time_s": time.time() - t0,
            }
            results.append(r)

            print(f"    SS3={ss3_ret.value:.1f}% SS8={ss8_ret.value:.1f}% "
                  f"Ret={ret_ret.value:.1f}% Dis={dis_ret.value:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            results.append({"config": cfg.name, "error": str(e)})

    return results


# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════

def print_sweep_table(plm_name: str, results: list[dict]):
    """Print codec sweep results for one PLM."""
    plm = PLMS[plm_name]
    print(f"\n{'='*100}")
    print(f"CODEC SWEEP: {plm.display} ({plm.dim}d)")
    print(f"{'='*100}")
    print()
    print("Column definitions:")
    print("  Config    — Codec configuration: preprocessing + dimensionality + quantization")
    print("  Comp      — Approximate compression ratio vs raw fp32 (1024d × 4 bytes)")
    print("  SS3 ret   — Retention of 3-class secondary structure accuracy (Q3: H/E/C)")
    print("  SS8 ret   — Retention of 8-class secondary structure accuracy (Q8)")
    print("  Ret ret   — Retention of family retrieval Ret@1 (cosine kNN on DCT K=4 vectors)")
    print("  Dis ret   — Retention of disorder pooled Spearman rho (CheZOD Z-scores)")
    print()
    print("Retention = compressed metric / raw metric × 100%. All with paired BCa bootstrap")
    print("CIs (B=10,000, 3-seed averaged predictions). Codec fitted on external SCOPe 5K.")
    print("SS3/SS8: CV-tuned LogReg. Disorder: CV-tuned Ridge, pooled residue-level rho.")
    print("See: experiments/43_rigorous_benchmark/ for full methodology.")
    print()

    header = (f"{'Config':<25} {'Comp':>5} {'SS3 ret':>10} {'SS8 ret':>10} "
              f"{'Ret ret':>10} {'Dis ret':>10} {'Time':>6}")
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['config']:<25} {'ERR':>5} {r['error'][:50]}")
            continue
        def _f(task):
            d = r[task]
            half = (d["ci"][1] - d["ci"][0]) / 2
            return f"{d['retention']:.1f}±{half:.1f}%"
        print(f"{r['config']:<25} {r['compression']:>5} "
              f"{_f('ss3'):>10} {_f('ss8'):>10} "
              f"{_f('ret1'):>10} {_f('disorder'):>10} "
              f"{r['time_s']:>5.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Codec Configuration Sweep")
    parser.add_argument("--plm", nargs="+", help="PLM names (default: all)")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--configs", choices=["standard", "useful", "all", "retest"],
                        default="useful", help="Which configs to sweep")
    args = parser.parse_args()

    plm_names = args.plm or list(PLMS.keys())
    for name in plm_names:
        if name not in PLMS:
            print(f"Unknown PLM: {name}")
            sys.exit(1)

    if args.smoke_test:
        global BOOTSTRAP_N, SEEDS
        BOOTSTRAP_N = 100
        SEEDS = [42]
        print("SMOKE TEST MODE: B=100, 1 seed\n")

    if args.configs == "standard":
        configs = STANDARD_CONFIGS
    elif args.configs == "useful":
        configs = USEFUL_CONFIGS
    elif args.configs == "retest":
        configs = RETEST_CONFIGS
    elif args.configs == "all":
        configs = ALL_CONFIGS

    print(f"Configs: {len(configs)} ({args.configs})")
    for c in configs:
        print(f"  {c.name:<25} {c.compression_ratio:>5} — {c.description}")

    t0 = time.time()
    all_results = {}

    for plm_name in plm_names:
        plm = PLMS[plm_name]
        print(f"\n{'='*60}")
        print(f"SWEEP: {plm.display} — {len(configs)} configs")
        print(f"{'='*60}")

        results = run_sweep(plm_name, configs, bootstrap_n=BOOTSTRAP_N)
        all_results[plm_name] = results

        # Save per-PLM
        out_path = RESULTS_DIR / f"exp47_sweep_{plm_name}.json"
        with open(out_path, "w") as f:
            json.dump({"plm": plm.display, "dim": plm.dim, "configs": results},
                      f, indent=2, default=float)
        print(f"\n  Saved: {out_path}")

        print_sweep_table(plm_name, results)

    # Combined
    combined_path = RESULTS_DIR / "exp47_codec_sweep.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nCombined: {combined_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
