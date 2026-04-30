"""
Experiment 15: External Validation
===================================
Validates compressed ProtT5 d256 embeddings on:
  A) ToxFam binary toxicity classification (balanced 7K subset)
  B) TMbed PCA-256 baseline comparison

Usage:
  uv run python experiments/15_external_validation.py --step S1   # Select balanced subset
  uv run python experiments/15_external_validation.py --step S2   # Extract + compress embeddings
  uv run python experiments/15_external_validation.py --step S3   # TMbed PCA baseline
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
TOXFAM_ROOT = Path(os.environ.get("TOXFAM_ROOT", "../students/ToxFam")).expanduser()
TOXFAM_CSV = TOXFAM_ROOT / "data" / "processed" / "training_data.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_validation" / "toxfam"
RESULTS_FILE = PROJECT_ROOT / "data" / "benchmarks" / "external_validation_results.json"

# ToxFam label conventions
NONTOXIN_LABELS = {"nontox", "nontoxic"}


def is_toxic(label: str) -> bool:
    return str(label).strip().lower() not in NONTOXIN_LABELS


def step_s1_select_subset(seed: int = 42):
    """Select balanced subset: all toxic + matched non-toxic, stratified by split."""
    print("=" * 60)
    print("Step S1: Select balanced ToxFam subset")
    print("=" * 60)

    df = pd.read_csv(TOXFAM_CSV)
    print(f"Total proteins: {len(df)}")

    df["is_toxic"] = df["Protein families"].apply(is_toxic)
    toxic = df[df["is_toxic"]]
    nontoxic = df[~df["is_toxic"]]
    print(f"Toxic: {len(toxic)}, Non-toxic: {len(nontoxic)}")

    # Sample non-toxic to match toxic count, stratified by split
    rng = np.random.RandomState(seed)
    sampled_parts = []
    for split_name in ["train", "val", "test"]:
        nt_split = nontoxic[nontoxic["Split"] == split_name]
        t_split = toxic[toxic["Split"] == split_name]
        n_sample = min(len(t_split), len(nt_split))
        sampled = nt_split.sample(n=n_sample, random_state=rng)
        sampled_parts.append(sampled)
        print(f"  {split_name}: {len(t_split)} toxic + {n_sample} non-toxic sampled")

    nontoxic_sampled = pd.concat(sampled_parts)
    subset = pd.concat([toxic, nontoxic_sampled]).sort_index()
    subset["binary_label"] = subset["is_toxic"].map({True: "toxin", False: "nontoxin"})

    print(f"\nSubset total: {len(subset)}")
    for split_name in ["train", "val", "test"]:
        s = subset[subset["Split"] == split_name]
        print(f"  {split_name}: {len(s)} ({s['is_toxic'].sum()} toxic, "
              f"{(~s['is_toxic']).sum()} non-toxic)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = OUTPUT_DIR / "subset.csv"
    subset.to_csv(subset_path, index=False)
    print(f"\nSaved subset CSV: {subset_path}")
    print(f"Columns: {list(subset.columns)}")
    return subset


def step_s2_extract_and_compress(seed: int = 42):
    """Extract ProtT5 per-residue embeddings, compress, save mean-pooled H5 files."""
    import shutil
    print("=" * 60)
    print("Step S2: Streaming extraction + compression")
    print("=" * 60)

    # Disk space safety check
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    print(f"Free disk: {free_gb:.1f} GB")
    if free_gb < 1.0:
        print("ERROR: Less than 1 GB free, aborting")
        return

    from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
    from src.compressors.channel_compressor import ChannelCompressor
    from src.utils.device import get_device

    device = get_device()
    print(f"Device: {device}")

    # Load subset
    subset_path = OUTPUT_DIR / "subset.csv"
    if not subset_path.exists():
        print("Subset CSV not found — running S1 first")
        step_s1_select_subset(seed=seed)
    subset = pd.read_csv(subset_path)
    print(f"Subset: {len(subset)} proteins")

    # Load ChannelCompressor checkpoint
    ckpt_path = (PROJECT_ROOT / "data" / "checkpoints" / "channel" /
                 "channel_prot_t5_contrastive_d256_s42" / "best_model.pt")
    model = ChannelCompressor(input_dim=1024, latent_dim=256, dropout=0.1,
                              use_residual=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    model = model.to(device).eval()
    print(f"Loaded ChannelCompressor from {ckpt_path}")

    # Build fasta dict from CSV sequences
    fasta_dict = dict(zip(subset["identifier"], subset["Sequence"]))
    print(f"Sequences to extract: {len(fasta_dict)}")

    # Extract ALL per-residue ProtT5 embeddings in one call
    # (avoids reloading the 21 GB model per chunk — internal batching handles memory)
    print(f"\nExtracting ProtT5 per-residue embeddings for {len(fasta_dict)} proteins...")
    print("This takes ~20-30 min. Monitor heat: pmset -g therm")
    per_residue = extract_prot_t5_embeddings(
        fasta_dict,
        model_name="Rostlab/prot_t5_xl_uniref50",
        batch_size=4,
        device=device,
    )
    print(f"Extracted {len(per_residue)} protein embeddings")

    # Compress and save — iterate once, write both H5 files
    baseline_h5_path = OUTPUT_DIR / "embeddings_baseline_1024.h5"
    compressed_h5_path = OUTPUT_DIR / "embeddings_compressed_256.h5"

    with h5py.File(baseline_h5_path, "w") as h5_base, \
         h5py.File(compressed_h5_path, "w") as h5_comp:

        with torch.no_grad():
            for i, (pid, emb) in enumerate(per_residue.items()):
                # emb shape: (L, 1024) — may be <seq_len due to 512-token truncation
                # Baseline: mean-pool to 1024d
                baseline_vec = emb.mean(axis=0)  # (1024,)
                h5_base.create_dataset(pid, data=baseline_vec.astype(np.float32))

                # Compressed: run through ChannelCompressor
                L = emb.shape[0]
                states = torch.from_numpy(emb).unsqueeze(0).to(device)  # (1, L, 1024)
                mask = torch.ones(1, L, device=device)
                latent = model.compress(states, mask)  # (1, L, 256)
                pooled = model.get_pooled(latent, strategy="mean")  # (1, 256)
                comp_vec = pooled[0].cpu().numpy()  # (256,)
                h5_comp.create_dataset(pid, data=comp_vec.astype(np.float32))

                if (i + 1) % 1000 == 0:
                    print(f"  Compressed {i + 1}/{len(per_residue)}")
                    if device.type == "mps":
                        torch.mps.empty_cache()

    # Free per-residue embeddings from memory
    del per_residue
    if device.type == "mps":
        torch.mps.empty_cache()

    base_size = baseline_h5_path.stat().st_size / 1e6
    comp_size = compressed_h5_path.stat().st_size / 1e6
    print(f"\nBaseline H5: {baseline_h5_path} ({base_size:.1f} MB)")
    print(f"Compressed H5: {compressed_h5_path} ({comp_size:.1f} MB)")
    print("Done. No per-residue embeddings saved to disk.")


def step_s3_tmbed_pca():
    """Compare PCA-256 with ChannelCompressor on TMbed topology prediction."""
    print("=" * 60)
    print("Step S3: TMbed PCA-256 baseline comparison")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # --- Load TMbed data ---
    tmbed_fasta = PROJECT_ROOT / "data" / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
    if not tmbed_fasta.exists():
        print(f"TMbed data not found at {tmbed_fasta}")
        return

    # Parse TMbed FASTA (3-line format: header, sequence, topology)
    # Headers like >Q9BEC7|EUKARYA|NO_SP|1 or >5mzs_A|Q05098|PFEA_PSEAE
    # First field before | is the protein ID
    proteins = []
    with open(tmbed_fasta) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            pid = lines[i][1:].split("|")[0]
            seq = lines[i + 1]
            topo = lines[i + 2]
            proteins.append((pid, seq, topo))
            i += 3
        else:
            i += 1

    print(f"TMbed proteins: {len(proteins)}")

    # Normalize topology labels: H/h->H, B/b->B, S->S, everything else->O
    def normalize_topo(char):
        if char in "Hh":
            return "H"
        elif char in "Bb":
            return "B"
        elif char == "S":
            return "S"
        else:
            return "O"

    # --- Load per-residue embeddings ---
    # Try both ESM2 and ProtT5 validation H5 files
    results = {}

    for plm_name, h5_name, input_dim in [
        ("ESM2-650M", "esm2_650m_validation.h5", 1280),
        ("ProtT5-XL", "prot_t5_xl_validation.h5", 1024),
    ]:
        h5_path = PROJECT_ROOT / "data" / "residue_embeddings" / h5_name
        if not h5_path.exists():
            print(f"  {plm_name} validation H5 not found, skipping")
            continue

        print(f"\n--- {plm_name} (dim={input_dim}) ---")

        # Collect all residue embeddings + labels
        all_embeddings = []
        all_labels = []

        with h5py.File(h5_path, "r") as h5f:
            matched = 0
            for pid, seq, topo in proteins:
                # H5 keys use tmbed_ prefix (from experiment 13 extraction)
                h5_key = f"tmbed_{pid}"
                if h5_key not in h5f:
                    continue
                emb = h5f[h5_key][:]  # (L, D)
                L = min(len(topo), emb.shape[0])
                for j in range(L):
                    all_embeddings.append(emb[j])
                    all_labels.append(normalize_topo(topo[j]))
                matched += 1

        print(f"  Matched proteins: {matched}/{len(proteins)}")
        if matched == 0:
            continue

        X = np.array(all_embeddings, dtype=np.float32)
        y_str = np.array(all_labels)
        le = LabelEncoder()
        y = le.fit_transform(y_str)
        print(f"  Total residues: {len(y)}")
        print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")

        # --- Evaluate: Original, PCA-256, PCA-128 ---
        seeds = [42, 123, 456]
        for method_name, X_method in [
            (f"original_{input_dim}", X),
            ("PCA-256", None),
            ("PCA-128", None),
        ]:
            dim = int(method_name.split("-")[1]) if "PCA" in method_name else input_dim

            accs, f1s = [], []
            for s in seeds:
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(y))
                split_pt = int(0.8 * len(idx))
                train_idx, test_idx = idx[:split_pt], idx[split_pt:]

                if "PCA" in method_name:
                    pca = PCA(n_components=dim, random_state=s)
                    X_train = pca.fit_transform(X[train_idx])
                    X_test = pca.transform(X[test_idx])
                else:
                    X_train = X_method[train_idx]
                    X_test = X_method[test_idx]

                clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                         random_state=s)
                clf.fit(X_train, y[train_idx])
                y_pred = clf.predict(X_test)
                accs.append(accuracy_score(y[test_idx], y_pred))
                f1s.append(f1_score(y[test_idx], y_pred, average="macro"))

            acc_mean, acc_std = np.mean(accs), np.std(accs)
            f1_mean, f1_std = np.mean(f1s), np.std(f1s)
            print(f"  {method_name}: Acc={acc_mean:.4f}+/-{acc_std:.4f}, "
                  f"F1={f1_mean:.4f}+/-{f1_std:.4f}")

            results[f"{plm_name}_{method_name}"] = {
                "accuracy_mean": round(float(acc_mean), 4),
                "accuracy_std": round(float(acc_std), 4),
                "f1_mean": round(float(f1_mean), 4),
                "f1_std": round(float(f1_std), 4),
            }

    # Load ChannelCompressor TMbed results from experiment 13 for comparison
    robust_results_path = PROJECT_ROOT / "data" / "benchmarks" / "robust_validation_results.json"
    if robust_results_path.exists():
        with open(robust_results_path) as f:
            robust = json.load(f)
        print("\n--- Comparison with ChannelCompressor d256 (from Experiment 13) ---")
        # robust_validation_results.json is a list of result dicts
        if isinstance(robust, list):
            for entry in robust:
                if isinstance(entry, dict) and "tmbed" in str(entry.get("name", "")).lower():
                    print(f"  {entry.get('name')}: F1={entry.get('f1_mean', 'N/A')}, "
                          f"Acc={entry.get('accuracy_mean', 'N/A')}")
        elif isinstance(robust, dict):
            for key, val in robust.items():
                if "tmbed" in str(key).lower():
                    print(f"  {key}: {val}")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing["tmbed_pca_comparison"] = results
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 15: External Validation")
    parser.add_argument("--step", required=True, choices=["S1", "S2", "S3"],
                        help="Which step to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.step == "S1":
        step_s1_select_subset(seed=args.seed)
    elif args.step == "S2":
        step_s2_extract_and_compress(seed=args.seed)
    elif args.step == "S3":
        step_s3_tmbed_pca()


if __name__ == "__main__":
    main()
