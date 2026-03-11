#!/usr/bin/env python3
"""Phase 9B: Two-Head Joint Training.

Trains ChannelCompressor with both reconstruction and retrieval heads
simultaneously, eliminating the two-checkpoint problem.

The key insight: contrastive gradient flows through a separate retrieval
projection head, NOT directly through the per-residue latent space.

Steps:
  T1: Joint training (ProtT5-XL, d256, 3 seeds)
  T2: Evaluate on all benchmarks (Ret@1, Q3, Q8, CheZOD, TMbed, TS115)

Usage:
  uv run python experiments/14_two_head_training.py --step T1
  uv run python experiments/14_two_head_training.py --step T2
  uv run python experiments/14_two_head_training.py          # run all

Success: Ret@1 >= 0.78 AND Q3/Q3_orig >= 0.98 from a single checkpoint.
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
import torch.nn as nn

from src.compressors.channel_compressor import ChannelCompressor
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    evaluate_disorder_probe,
    evaluate_tm_probe,
    load_cb513_csv,
    load_chezod_seth,
    load_tmbed_annotated,
)
from src.evaluation.reconstruction import evaluate_reconstruction
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.splitting import load_split
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.training.objectives import InfoNCEFamilyLoss, ReconstructionLoss
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "channel"
RESULTS_PATH = DATA_DIR / "benchmarks" / "two_head_results.json"
SPLIT_DIR = DATA_DIR / "splits"

CB513_CSV = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
TS115_CSV = DATA_DIR / "per_residue_benchmarks" / "TS115.csv"
TMBED_FASTA = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
CHEZOD_DIR = DATA_DIR / "per_residue_benchmarks"

SEEDS = [42, 123, 456]
D_PRIME = 256
RETRIEVAL_HEAD_DIM = 256
T5_DIM = 1024
PROBE_SEEDS = [42, 123, 456]


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
        json.dump(results, f, indent=2, default=str)


def is_done(results: list[dict], name: str) -> bool:
    return any(r["name"] == name for r in results)


def compress_embeddings(model, embeddings, device, max_len=512):
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


def get_retrieval_embeddings(model, embeddings, device, max_len=512):
    """Get retrieval embeddings from the retrieval head (pooled + projected)."""
    model.eval()
    retrieval_embs = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
            mask = torch.ones(1, L, device=device)
            output = model(states, mask)
            ret_emb = output["retrieval_embedding"]  # (1, D_proj)
            retrieval_embs[pid] = ret_emb[0].cpu().numpy().reshape(1, -1)
    return retrieval_embs


# ── T1: Joint Training ──────────────────────────────────────────


def step_t1(device):
    print(f"\n{'='*60}")
    print("T1: Two-Head Joint Training (ProtT5-XL, d256)")
    print(f"{'='*60}")

    all_results = load_results()

    # Load data
    h5_path = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if not h5_path.exists():
        print("  ProtT5 5K embeddings not found.")
        return

    prot_t5_emb = load_residue_embeddings(h5_path)
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    prot_t5_emb = {k: v for k, v in prot_t5_emb.items() if k in kept_ids}

    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    train_ids, test_ids, eval_ids = load_split(sf_split_path)

    id_to_fam = {m["id"]: m["family"] for m in filtered_meta}
    train_pids = [pid for pid in train_ids if pid in id_to_fam and pid in prot_t5_emb]
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}

    test_set = set(test_ids)
    val_emb = {k: v for k, v in prot_t5_emb.items() if k in test_set}

    for seed in SEEDS:
        name = f"two_head_prot_t5_d{D_PRIME}_s{seed}"
        ckpt_dir = CHECKPOINTS_DIR / name

        if (ckpt_dir / "best_model.pt").exists():
            print(f"\n  {name} checkpoint exists, skipping training")
            continue

        print(f"\n  Training {name}...")
        monitor()

        model = ChannelCompressor(
            input_dim=T5_DIM, latent_dim=D_PRIME, dropout=0.1,
            use_residual=True, retrieval_head_dim=RETRIEVAL_HEAD_DIM,
        )
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,} (includes retrieval head)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        warmup_epochs = 10
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 - warmup_epochs),
            ],
            milestones=[warmup_epochs],
        )
        infonce_fn = InfoNCEFamilyLoss(temperature=0.07)
        recon_loss_fn = ReconstructionLoss().to(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        max_len = 512
        batch_size = 32
        recon_weight = 1.0
        contrastive_weight = 0.5

        best_val_loss = float("inf")
        best_epoch = 0

        start = time.time()
        for epoch in range(1, 201):
            model.train()
            perm = np.random.permutation(len(train_pids))
            epoch_recon_loss = 0
            epoch_cl_loss = 0
            n_batches = 0

            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                batch_pids = [train_pids[j] for j in idx]

                batch_embs, batch_masks, batch_labels = [], [], []
                for pid in batch_pids:
                    emb = prot_t5_emb[pid]
                    L = min(emb.shape[0], max_len)
                    padded = np.zeros((max_len, T5_DIM), dtype=np.float32)
                    padded[:L] = emb[:L]
                    mask = np.zeros(max_len, dtype=np.float32)
                    mask[:L] = 1.0
                    batch_embs.append(padded)
                    batch_masks.append(mask)
                    batch_labels.append(fam_to_idx[id_to_fam[pid]])

                states = torch.from_numpy(np.array(batch_embs)).to(device)
                masks = torch.from_numpy(np.array(batch_masks)).to(device)
                labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

                optimizer.zero_grad()
                output = model(states, masks)

                # Reconstruction loss (per-residue, through decoder)
                recon_result = recon_loss_fn(output["reconstructed"], states, masks)

                # Contrastive loss (through retrieval head projection)
                retrieval_emb = output["retrieval_embedding"]
                cl = infonce_fn(retrieval_emb, labels)

                loss = recon_weight * recon_result["loss"] + contrastive_weight * cl["loss"]
                loss.backward()
                # Sanitize inf/nan gradients before clipping (0 * inf = NaN in clip_grad_norm_)
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_recon_loss += recon_result["loss"].item()
                epoch_cl_loss += cl["loss"].item()
                n_batches += 1

            scheduler.step()

            # Validation
            if epoch % 10 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    recon = evaluate_reconstruction(model, val_emb, device)
                val_loss = recon["mse"]
                elapsed = time.time() - start

                improved = ""
                if not np.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                    improved = " *"

                if epoch % 25 == 0 or epoch == 1:
                    print(f"    Epoch {epoch:3d}/200 | Recon={epoch_recon_loss/n_batches:.4f} "
                          f"| CL={epoch_cl_loss/n_batches:.4f} | CosSim={recon['cosine_sim']:.3f}"
                          f"{improved} | {elapsed:.0f}s")
                model.train()

        elapsed = time.time() - start
        print(f"  Training done in {elapsed:.0f}s (best epoch: {best_epoch})")

        # Load best checkpoint for evaluation
        model.load_state_dict(
            torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True)
        )
        model = model.to(device)
        model.eval()

        # Quick eval: retrieval + reconstruction
        recon = evaluate_reconstruction(model, val_emb, device)
        retrieval_embs = get_retrieval_embeddings(model, prot_t5_emb, device)

        ret = evaluate_retrieval(
            None, retrieval_embs, filtered_meta, label_key="family",
            query_ids=eval_ids, database_ids=test_ids,
        )

        cls_split_path = SPLIT_DIR / "esm2_650m_5k_cls_split.json"
        with open(cls_split_path) as f:
            cls_data = json.load(f)
        cls = evaluate_linear_probe(
            None, retrieval_embs, filtered_meta, label_key="family",
            train_ids=cls_data["cls_train_ids"], test_ids=cls_data["cls_test_ids"],
        )

        result = {
            "name": name, "plm": "ProtT5-XL", "method": "two_head",
            "latent_dim": D_PRIME, "retrieval_head_dim": RETRIEVAL_HEAD_DIM,
            "seed": seed, "best_epoch": best_epoch,
            "reconstruction": recon,
            "retrieval_family": ret,
            "classification_family": cls,
            "training_time_s": elapsed,
        }
        all_results.append(result)
        save_results(all_results)

        p1 = ret.get("precision@1", 0)
        mrr = ret.get("mrr", 0)
        cos = recon.get("cosine_sim", 0)
        cls_acc = cls.get("accuracy_mean", 0)
        print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, CosSim={cos:.3f}, Cls={cls_acc:.3f}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n  T1 complete.")


# ── T2: Full Evaluation ─────────────────────────────────────────


def step_t2(device):
    print(f"\n{'='*60}")
    print("T2: Two-Head Full Evaluation")
    print(f"{'='*60}")

    all_results = load_results()

    # Load all per-residue data
    cb513_seqs, cb513_ss3, cb513_ss8, cb513_dis = load_cb513_csv(CB513_CSV)
    cb513_splits_path = SPLIT_DIR / "cb513_probe_splits.json"
    with open(cb513_splits_path) as f:
        cb513_splits = {int(k): (tr, te) for k, (tr, te) in json.load(f).items()}

    cb513_t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    cb513_emb = load_residue_embeddings(cb513_t5_h5) if cb513_t5_h5.exists() else {}

    # CheZOD
    chezod_seqs, chezod_scores, chezod_train, chezod_test = load_chezod_seth(CHEZOD_DIR)
    chezod_t5_h5 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"
    chezod_emb = {}
    if chezod_t5_h5.exists():
        all_val_emb = load_residue_embeddings(chezod_t5_h5)
        chezod_emb = {k[len("chezod_"):]: v for k, v in all_val_emb.items() if k.startswith("chezod_")}

    # TMbed
    tmbed_seqs, tmbed_topo = load_tmbed_annotated(TMBED_FASTA)
    tmbed_emb = {}
    if chezod_t5_h5.exists():
        tmbed_emb = {k[len("tmbed_"):]: v for k, v in all_val_emb.items() if k.startswith("tmbed_")}

    # TS115
    ts115_seqs, ts115_ss3, ts115_ss8, _ = load_cb513_csv(TS115_CSV)
    ts115_emb = {}
    if chezod_t5_h5.exists():
        ts115_emb = {k[len("ts115_"):]: v for k, v in all_val_emb.items() if k.startswith("ts115_")}

    tmbed_splits = {}
    if tmbed_seqs:
        for seed in PROBE_SEEDS:
            rng = random.Random(seed)
            ids = list(tmbed_seqs.keys())
            rng.shuffle(ids)
            n = int(len(ids) * 0.8)
            tmbed_splits[seed] = (ids[:n], ids[n:])

    ts115_splits = {}
    if ts115_seqs:
        for seed in PROBE_SEEDS:
            rng = random.Random(seed)
            ids = list(ts115_seqs.keys())
            rng.shuffle(ids)
            n = int(len(ids) * 0.8)
            ts115_splits[seed] = (ids[:n], ids[n:])

    for seed in SEEDS:
        name = f"two_head_prot_t5_d{D_PRIME}_s{seed}"
        eval_name = f"t2_eval_{name}"

        if is_done(all_results, eval_name):
            print(f"  {eval_name} already done")
            continue

        ckpt_path = CHECKPOINTS_DIR / name / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  {name}: checkpoint not found, run T1 first")
            continue

        print(f"\n  Evaluating {name}...")
        model = ChannelCompressor(
            input_dim=T5_DIM, latent_dim=D_PRIME, dropout=0.1,
            use_residual=True, retrieval_head_dim=RETRIEVAL_HEAD_DIM,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)

        result = {"name": eval_name, "plm": "ProtT5-XL", "method": "two_head",
                   "seed": seed, "dim": D_PRIME}

        # CB513 SS3/SS8
        if cb513_emb:
            compressed_cb513 = compress_embeddings(model, cb513_emb, device)
            q3_all, q8_all = [], []
            for s, (tr, te) in cb513_splits.items():
                r3 = evaluate_ss3_probe(compressed_cb513, cb513_ss3, tr, te)
                q3_all.append(r3["q3"])
                r8 = evaluate_ss8_probe(compressed_cb513, cb513_ss8, tr, te)
                q8_all.append(r8["q8"])
            result["cb513_q3_mean"] = float(np.mean(q3_all))
            result["cb513_q3_std"] = float(np.std(q3_all))
            result["cb513_q8_mean"] = float(np.mean(q8_all))
            result["cb513_q8_std"] = float(np.std(q8_all))
            print(f"    CB513: Q3={result['cb513_q3_mean']:.3f}, Q8={result['cb513_q8_mean']:.3f}")

        # CheZOD disorder
        if chezod_emb:
            compressed_chezod = compress_embeddings(model, chezod_emb, device)
            dis = evaluate_disorder_probe(compressed_chezod, chezod_scores, chezod_train, chezod_test)
            result["chezod_spearman_rho"] = dis["spearman_rho"]
            print(f"    CheZOD: rho={dis['spearman_rho']:.3f}")

        # TMbed topology
        if tmbed_emb and tmbed_splits:
            compressed_tmbed = compress_embeddings(model, tmbed_emb, device)
            accs, f1s = [], []
            for s, (tr, te) in tmbed_splits.items():
                tm = evaluate_tm_probe(compressed_tmbed, tmbed_topo, tr, te)
                accs.append(tm["accuracy"])
                f1s.append(tm["macro_f1"])
            result["tmbed_accuracy_mean"] = float(np.mean(accs))
            result["tmbed_f1_mean"] = float(np.mean(f1s))
            print(f"    TMbed: Acc={result['tmbed_accuracy_mean']:.3f}, F1={result['tmbed_f1_mean']:.3f}")

        # TS115 SS3/SS8
        if ts115_emb and ts115_splits:
            compressed_ts115 = compress_embeddings(model, ts115_emb, device)
            q3_all, q8_all = [], []
            for s, (tr, te) in ts115_splits.items():
                r3 = evaluate_ss3_probe(compressed_ts115, ts115_ss3, tr, te)
                q3_all.append(r3["q3"])
                r8 = evaluate_ss8_probe(compressed_ts115, ts115_ss8, tr, te)
                q8_all.append(r8["q8"])
            result["ts115_q3_mean"] = float(np.mean(q3_all))
            result["ts115_q8_mean"] = float(np.mean(q8_all))
            print(f"    TS115: Q3={result['ts115_q3_mean']:.3f}, Q8={result['ts115_q8_mean']:.3f}")

        all_results.append(result)
        save_results(all_results)
        del model

    # Print summary
    print(f"\n{'='*80}")
    print("TWO-HEAD RESULTS SUMMARY")
    print(f"{'='*80}")

    for r in all_results:
        if "retrieval_family" in r:
            p1 = r["retrieval_family"].get("precision@1", "?")
            mrr = r["retrieval_family"].get("mrr", "?")
            cos = r.get("reconstruction", {}).get("cosine_sim", "?")
            print(f"  {r['name']}: Ret@1={p1}, MRR={mrr}, CosSim={cos}")

    for r in all_results:
        if r["name"].startswith("t2_eval_"):
            q3 = r.get("cb513_q3_mean", "?")
            rho = r.get("chezod_spearman_rho", "?")
            tm = r.get("tmbed_accuracy_mean", "?")
            ts_q3 = r.get("ts115_q3_mean", "?")
            print(f"  {r['name']}: Q3={q3}, CheZOD_rho={rho}, TMbed_acc={tm}, TS115_Q3={ts_q3}")

    print("\n  T2 complete.")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Two-Head Joint Training")
    parser.add_argument("--step", type=str, default=None, help="Run specific step: T1, T2")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    steps = {
        "T1": lambda: step_t1(device),
        "T2": lambda: step_t2(device),
    }

    if args.step:
        step_key = args.step.upper()
        if step_key in steps:
            steps[step_key]()
        else:
            print(f"Unknown step: {args.step}. Available: {', '.join(steps.keys())}")
    else:
        for step_key in ["T1", "T2"]:
            steps[step_key]()


if __name__ == "__main__":
    main()
