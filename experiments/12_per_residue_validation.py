#!/usr/bin/env python3
"""Phase 9A: Per-Residue Validation on CB513.

Validates that compressed per-residue embeddings retain enough structural
information for standard per-residue prediction tasks (SS3, SS8, disorder).

Steps:
  V1: Data preparation + embedding extraction (ESM2-650M, ProtT5-XL for CB513)
  V2: Baseline evaluation (original full-dim, PCA at d64/d128/d256)
  V3: ChannelCompressor evaluation (ESM2-650M checkpoints)
  V4: ChannelCompressor evaluation (ProtT5-XL checkpoints)

Usage:
  uv run python experiments/12_per_residue_validation.py --step V1
  uv run python experiments/12_per_residue_validation.py --step V2
  uv run python experiments/12_per_residue_validation.py --step V3
  uv run python experiments/12_per_residue_validation.py --step V4
  uv run python experiments/12_per_residue_validation.py          # run all

Success criterion: Q3_compressed / Q3_original > 0.90 at d256 unsupervised.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    evaluate_disorder_probe,
    load_cb513_csv,
)
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings, save_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CB513_CSV = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "per_residue_validation_results.json"
SPLITS_PATH = DATA_DIR / "splits" / "cb513_probe_splits.json"

PROBE_SEEDS = [42, 123, 456]
ESM2_EMBED_DIM = 1280
PROTT5_EMBED_DIM = 1024


# ── Helpers ──────────────────────────────────────────────────────


def monitor():
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  System load: {load1:.1f} / {load5:.1f} / {load15:.1f}")
    except OSError:
        pass


def load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {RESULTS_PATH}")


def is_done(results: list[dict], name: str) -> bool:
    return any(r["name"] == name for r in results)


def make_splits(protein_ids: list[str], train_fraction: float = 0.8) -> dict:
    """Create 80/20 splits for 3 seeds. Returns {seed: (train_ids, test_ids)}."""
    splits = {}
    for seed in PROBE_SEEDS:
        rng = random.Random(seed)
        ids = list(protein_ids)
        rng.shuffle(ids)
        n_train = int(len(ids) * train_fraction)
        splits[seed] = (ids[:n_train], ids[n_train:])
    return splits


def get_splits(protein_ids: list[str]) -> dict:
    """Load or create CB513 probe splits."""
    if SPLITS_PATH.exists():
        with open(SPLITS_PATH) as f:
            data = json.load(f)
        splits = {}
        for seed_str, (train, test) in data.items():
            splits[int(seed_str)] = (train, test)
        return splits

    splits = make_splits(protein_ids)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_data = {str(s): (tr, te) for s, (tr, te) in splits.items()}
    with open(SPLITS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved CB513 splits to {SPLITS_PATH}")
    return splits


def run_probes(
    embeddings: dict[str, np.ndarray],
    ss3_labels: dict[str, str],
    ss8_labels: dict[str, str],
    disorder_labels: dict[str, np.ndarray],
    splits: dict,
) -> dict:
    """Run SS3, SS8, disorder probes across all seeds. Returns averaged metrics."""
    q3_all, q8_all = [], []
    q3_per_class_all = []
    q8_per_class_all = []

    for seed, (train_ids, test_ids) in splits.items():
        ss3_result = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids)
        q3_all.append(ss3_result["q3"])
        if "per_class_acc" in ss3_result:
            q3_per_class_all.append(ss3_result["per_class_acc"])

        ss8_result = evaluate_ss8_probe(embeddings, ss8_labels, train_ids, test_ids)
        q8_all.append(ss8_result["q8"])
        if "per_class_acc" in ss8_result:
            q8_per_class_all.append(ss8_result["per_class_acc"])

    metrics = {
        "q3_mean": float(np.mean(q3_all)),
        "q3_std": float(np.std(q3_all)),
        "q3_per_seed": q3_all,
        "q8_mean": float(np.mean(q8_all)),
        "q8_std": float(np.std(q8_all)),
        "q8_per_seed": q8_all,
    }

    # Average per-class accuracies
    if q3_per_class_all:
        avg_pc = {}
        for cls in q3_per_class_all[0]:
            vals = [d[cls] for d in q3_per_class_all if cls in d]
            avg_pc[cls] = float(np.mean(vals))
        metrics["q3_per_class"] = avg_pc

    if q8_per_class_all:
        avg_pc = {}
        for cls in q8_per_class_all[0]:
            vals = [d[cls] for d in q8_per_class_all if cls in d]
            avg_pc[cls] = float(np.mean(vals))
        metrics["q8_per_class"] = avg_pc

    # Disorder (binary classification via regression probe + threshold)
    if disorder_labels:
        dis_rho_all = []
        for seed, (train_ids, test_ids) in splits.items():
            dis_result = evaluate_disorder_probe(
                embeddings, disorder_labels, train_ids, test_ids
            )
            dis_rho_all.append(dis_result["spearman_rho"])
        metrics["disorder_rho_mean"] = float(np.mean(dis_rho_all))
        metrics["disorder_rho_std"] = float(np.std(dis_rho_all))

    return metrics


def compress_embeddings(
    model: ChannelCompressor,
    embeddings: dict[str, np.ndarray],
    device: torch.device,
    max_len: int = 512,
) -> dict[str, np.ndarray]:
    """Compress all embeddings through a ChannelCompressor."""
    model.eval()
    compressed = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)
            compressed[pid] = latent[0, :L].cpu().numpy()
    return compressed


# ── V1: Data Preparation + Embedding Extraction ──────────────────


def step_v1(device):
    print(f"\n{'='*60}")
    print("V1: Data Preparation + Embedding Extraction")
    print(f"{'='*60}")

    # Parse CB513
    print("\n  Parsing CB513.csv...")
    sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(CB513_CSV)
    n = len(sequences)
    print(f"  Parsed {n} proteins from CB513")
    if n == 0:
        print(f"  ERROR: CB513.csv not found at {CB513_CSV}")
        return

    # Show some stats
    total_residues = sum(len(s) for s in sequences.values())
    print(f"  Total residues: {total_residues:,}")
    max_len = max(len(s) for s in sequences.values())
    print(f"  Max sequence length: {max_len}")

    # Create splits
    splits = get_splits(list(sequences.keys()))
    for seed, (train, test) in splits.items():
        print(f"  Split seed={seed}: {len(train)} train, {len(test)} test")

    # Extract ESM2-650M embeddings
    esm2_h5 = DATA_DIR / "residue_embeddings" / "esm2_650m_cb513.h5"
    if esm2_h5.exists():
        print(f"\n  ESM2-650M CB513 embeddings already exist at {esm2_h5}")
    else:
        print(f"\n  Extracting ESM2-650M embeddings for {n} CB513 proteins...")
        monitor()
        from src.extraction.esm_extractor import extract_residue_embeddings
        esm2_emb = extract_residue_embeddings(
            sequences, model_name="esm2_t33_650M_UR50D", batch_size=4, device=device
        )
        save_residue_embeddings(esm2_emb, esm2_h5)
        print(f"  ESM2-650M: {len(esm2_emb)} proteins, dim={next(iter(esm2_emb.values())).shape[-1]}")
        # Free GPU memory
        del esm2_emb
        torch.mps.empty_cache() if device.type == "mps" else None

    # Extract ProtT5-XL embeddings
    t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    if t5_h5.exists():
        print(f"\n  ProtT5-XL CB513 embeddings already exist at {t5_h5}")
    else:
        print(f"\n  Extracting ProtT5-XL embeddings for {n} CB513 proteins...")
        monitor()
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
        t5_emb = extract_prot_t5_embeddings(
            sequences, batch_size=4, device=device
        )
        save_residue_embeddings(t5_emb, t5_h5)
        print(f"  ProtT5-XL: {len(t5_emb)} proteins, dim={next(iter(t5_emb.values())).shape[-1]}")
        del t5_emb
        torch.mps.empty_cache() if device.type == "mps" else None

    print("\n  V1 complete.")


# ── V2: Baseline Evaluation ──────────────────────────────────────


def step_v2():
    print(f"\n{'='*60}")
    print("V2: Baseline Evaluation (original + PCA)")
    print(f"{'='*60}")

    all_results = load_results()

    # Load data
    sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(CB513_CSV)
    splits = get_splits(list(sequences.keys()))

    for plm_name, h5_name, embed_dim in [
        ("ESM2-650M", "esm2_650m_cb513.h5", ESM2_EMBED_DIM),
        ("ProtT5-XL", "prot_t5_xl_cb513.h5", PROTT5_EMBED_DIM),
    ]:
        h5_path = DATA_DIR / "residue_embeddings" / h5_name
        if not h5_path.exists():
            print(f"\n  {plm_name} embeddings not found, run V1 first.")
            continue

        embeddings = load_residue_embeddings(h5_path)

        # V2a: Original full-dim
        name = f"v2_{plm_name}_original_d{embed_dim}"
        if not is_done(all_results, name):
            print(f"\n  {name}: Original {embed_dim}d embeddings...")
            metrics = run_probes(embeddings, ss3_labels, ss8_labels, disorder_labels, splits)
            result = {"name": name, "plm": plm_name, "method": "original",
                       "dim": embed_dim, **metrics}
            all_results.append(result)
            save_results(all_results)
            print(f"  >> Q3={metrics['q3_mean']:.3f}+-{metrics['q3_std']:.3f}, "
                  f"Q8={metrics['q8_mean']:.3f}+-{metrics['q8_std']:.3f}")
        else:
            print(f"  {name} already done")

        # V2b: PCA baselines
        # Fit PCA on first split's train set
        train_ids_for_pca = splits[42][0]
        all_residues = []
        for pid in train_ids_for_pca:
            if pid in embeddings:
                all_residues.append(embeddings[pid][:512])
        all_residues = np.concatenate(all_residues, axis=0)
        print(f"\n  PCA fitting on {len(all_residues):,} residue vectors ({plm_name})...")

        for d_prime in [64, 128, 256]:
            name = f"v2_{plm_name}_pca_d{d_prime}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            pca = PCA(n_components=d_prime, random_state=42)
            pca.fit(all_residues)
            explained = pca.explained_variance_ratio_.sum()
            print(f"  {name}: PCA → {d_prime}d (explained var: {explained:.3f})")

            pca_emb = {}
            for pid, emb in embeddings.items():
                pca_emb[pid] = pca.transform(emb).astype(np.float32)

            metrics = run_probes(pca_emb, ss3_labels, ss8_labels, disorder_labels, splits)
            result = {"name": name, "plm": plm_name, "method": "pca",
                       "dim": d_prime, "explained_variance": float(explained), **metrics}
            all_results.append(result)
            save_results(all_results)
            print(f"  >> Q3={metrics['q3_mean']:.3f}+-{metrics['q3_std']:.3f}, "
                  f"Q8={metrics['q8_mean']:.3f}+-{metrics['q8_std']:.3f}")

        del embeddings

    print("\n  V2 complete.")


# ── V3: ChannelCompressor Evaluation (ESM2) ──────────────────────


def step_v3(device):
    print(f"\n{'='*60}")
    print("V3: ChannelCompressor Evaluation (ESM2-650M)")
    print(f"{'='*60}")

    all_results = load_results()
    sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(CB513_CSV)
    splits = get_splits(list(sequences.keys()))

    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_cb513.h5"
    if not h5_path.exists():
        print("  ESM2 CB513 embeddings not found, run V1 first.")
        return

    embeddings = load_residue_embeddings(h5_path)

    # ESM2 checkpoints
    configs = [
        ("channel_unsup_d64_s42", 64),
        ("channel_unsup_d64_s123", 64),
        ("channel_unsup_d128_s42", 128),
        ("channel_unsup_d128_s123", 128),
        ("channel_unsup_d256_s42", 256),
        ("channel_unsup_d256_s123", 256),
        ("channel_contrastive_d128_s42", 128),
        ("channel_contrastive_d128_s123", 128),
        ("channel_contrastive_d256_s42", 256),
        ("channel_contrastive_d256_s123", 256),
    ]

    for ckpt_name, d_prime in configs:
        name = f"v3_esm2_{ckpt_name}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  {name}: checkpoint not found, skipping")
            continue

        print(f"\n  {name}: D'={d_prime}...")
        model = ChannelCompressor(
            input_dim=ESM2_EMBED_DIM, latent_dim=d_prime,
            dropout=0.1, use_residual=True,
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        compressed = compress_embeddings(model, embeddings, device)
        metrics = run_probes(compressed, ss3_labels, ss8_labels, disorder_labels, splits)

        method = "channel_contrastive" if "contrastive" in ckpt_name else "channel_unsup"
        seed = int(ckpt_name.split("_s")[-1])
        result = {"name": name, "plm": "ESM2-650M", "method": method,
                   "dim": d_prime, "model_seed": seed, **metrics}
        all_results.append(result)
        save_results(all_results)
        print(f"  >> Q3={metrics['q3_mean']:.3f}+-{metrics['q3_std']:.3f}, "
              f"Q8={metrics['q8_mean']:.3f}+-{metrics['q8_std']:.3f}")

        del model, compressed

    print("\n  V3 complete.")


# ── V4: ChannelCompressor Evaluation (ProtT5) ────────────────────


def step_v4(device):
    print(f"\n{'='*60}")
    print("V4: ChannelCompressor Evaluation (ProtT5-XL)")
    print(f"{'='*60}")

    all_results = load_results()
    sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(CB513_CSV)
    splits = get_splits(list(sequences.keys()))

    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    if not h5_path.exists():
        print("  ProtT5 CB513 embeddings not found, run V1 first.")
        return

    embeddings = load_residue_embeddings(h5_path)

    configs = [
        ("channel_prot_t5_unsup_d256_s42", 256),
        ("channel_prot_t5_contrastive_d256_s42", 256),
    ]

    for ckpt_name, d_prime in configs:
        name = f"v4_prott5_{ckpt_name}"
        if is_done(all_results, name):
            print(f"  {name} already done")
            continue

        ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  {name}: checkpoint not found, skipping")
            continue

        print(f"\n  {name}: D'={d_prime}...")
        model = ChannelCompressor(
            input_dim=PROTT5_EMBED_DIM, latent_dim=d_prime,
            dropout=0.1, use_residual=True,
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        compressed = compress_embeddings(model, embeddings, device)
        metrics = run_probes(compressed, ss3_labels, ss8_labels, disorder_labels, splits)

        method = "channel_contrastive" if "contrastive" in ckpt_name else "channel_unsup"
        result = {"name": name, "plm": "ProtT5-XL", "method": method,
                   "dim": d_prime, **metrics}
        all_results.append(result)
        save_results(all_results)
        print(f"  >> Q3={metrics['q3_mean']:.3f}+-{metrics['q3_std']:.3f}, "
              f"Q8={metrics['q8_mean']:.3f}+-{metrics['q8_std']:.3f}")

        del model, compressed

    print("\n  V4 complete.")


# ── Summary ──────────────────────────────────────────────────────


def print_summary():
    all_results = load_results()
    if not all_results:
        print("No results yet.")
        return

    print(f"\n{'='*80}")
    print("PHASE 9A: Per-Residue Validation Summary")
    print(f"{'='*80}")
    print(f"{'Name':<48} {'D':>5} {'Q3':>7} {'Q8':>7} {'Q3/Orig':>8}")
    print(f"{'-'*80}")

    # Find original Q3 for ratio computation
    orig_q3 = {}
    for r in all_results:
        if r.get("method") == "original":
            orig_q3[r["plm"]] = r["q3_mean"]

    for r in all_results:
        name = r.get("name", "?")
        dim = r.get("dim", "?")
        q3 = r.get("q3_mean", float("nan"))
        q8 = r.get("q8_mean", float("nan"))
        plm = r.get("plm", "")
        ratio = q3 / orig_q3[plm] if plm in orig_q3 and orig_q3[plm] > 0 else float("nan")
        marker = " *" if ratio >= 0.90 and r.get("method") != "original" else ""
        print(f"{name:<48} {dim:>5} {q3:>7.3f} {q8:>7.3f} {ratio:>8.3f}{marker}")

    print(f"\n  * = meets success criterion (Q3/Q3_orig >= 0.90)")

    # Key takeaways
    if orig_q3:
        print(f"\n  Baselines:")
        for plm, q3_val in orig_q3.items():
            print(f"    {plm} original Q3 = {q3_val:.3f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 9A: Per-Residue Validation")
    parser.add_argument("--step", type=str, default=None,
                        help="Run specific step: V1, V2, V3, V4")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    steps = {
        "V1": lambda: step_v1(device),
        "V2": lambda: step_v2(),
        "V3": lambda: step_v3(device),
        "V4": lambda: step_v4(device),
    }

    if args.step:
        step_key = args.step.upper()
        if step_key in steps:
            steps[step_key]()
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
            return
    else:
        for step_key in ["V1", "V2", "V3", "V4"]:
            steps[step_key]()

    print_summary()


if __name__ == "__main__":
    main()
