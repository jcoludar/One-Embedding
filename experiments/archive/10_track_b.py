#!/usr/bin/env python3
"""Phase 7, Track B: Per-residue approaches that avoid reconstruction.

B-5. DeepSets attention pooling + MLP encoder
B-6. Pool-first, compress-second (mean + attention concat)
B-7. Multi-scale pooling (mean, std, max -> MLP)

Usage:
  uv run python experiments/10_track_b.py                   # run all
  uv run python experiments/10_track_b.py --step B5         # attention pooling
  uv run python experiments/10_track_b.py --step B6         # dual-pool concat
  uv run python experiments/10_track_b.py --step B7         # multi-scale
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compressors.attention_pool_simple import (
    DeepSetsAttentionCompressor,
    MultiScalePoolCompressor,
)
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.classification import evaluate_linear_probe
from src.evaluation.splitting import (
    family_stratified_split,
    load_split,
    save_split,
    split_statistics,
    superfamily_aware_split,
)
from src.extraction.data_loader import (
    filter_by_family_size,
    load_metadata_csv,
    read_fasta,
)
from src.training.objectives import InfoNCEFamilyLoss, MeanPoolReconLoss, VICRegLoss
from src.training.trainer import ResidueEmbeddingDataset
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "track_b"
RESULTS_PATH = DATA_DIR / "benchmarks" / "track_b_results.json"
SPLIT_DIR = DATA_DIR / "splits"

SEEDS = [42, 123]


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
    save_benchmark_results(results, RESULTS_PATH)


def is_done(results: list[dict], name: str) -> bool:
    return any(r["name"] == name for r in results)


def load_5k_data():
    """Load 5K dataset, filter to families >= 3 members."""
    fasta_path = DATA_DIR / "proteins" / "medium_diverse_5k.fasta"
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    h5_path = DATA_DIR / "residue_embeddings" / "esm2_650m_medium5k.h5"

    embeddings = load_residue_embeddings(h5_path)
    metadata = load_metadata_csv(meta_path)
    sequences = read_fasta(fasta_path)

    filtered_meta, kept_ids = filter_by_family_size(metadata, min_members=3)
    filt_emb = {k: v for k, v in embeddings.items() if k in kept_ids}
    filt_seq = {k: v for k, v in sequences.items() if k in kept_ids}

    n_fam = len(set(m["family"] for m in filtered_meta))
    print(f"  Loaded: {len(filt_emb)}/{len(embeddings)} proteins, {n_fam} families")
    return filt_emb, filtered_meta, filt_seq


def get_splits(metadata):
    """Get both superfamily-aware (retrieval) and family-stratified (classification) splits."""
    sf_split_path = SPLIT_DIR / "esm2_650m_5k_split.json"
    if sf_split_path.exists():
        train_ids, test_ids, eval_ids = load_split(sf_split_path)
    else:
        train_ids, test_ids, eval_ids = superfamily_aware_split(
            metadata, test_fraction=0.3, seed=42
        )
        stats = split_statistics(metadata, train_ids, test_ids, eval_ids)
        save_split(train_ids, test_ids, eval_ids, sf_split_path, stats=stats)

    cls_split_path = SPLIT_DIR / "esm2_650m_5k_cls_split.json"
    if cls_split_path.exists():
        with open(cls_split_path) as f:
            cls_data = json.load(f)
        cls_train_ids = cls_data["cls_train_ids"]
        cls_test_ids = cls_data["cls_test_ids"]
    else:
        cls_train_ids, cls_test_ids = family_stratified_split(
            metadata, test_fraction=0.3, min_family_size=2, seed=42
        )
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(cls_split_path, "w") as f:
            json.dump({"cls_train_ids": cls_train_ids, "cls_test_ids": cls_test_ids}, f, indent=2)

    return train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids


def train_per_residue_model(
    model, embeddings, sequences, train_ids, test_ids,
    device, seed=42, epochs=100, batch_size=16, lr=1e-3,
    vicreg_weight=0.0, contrastive_weight=0.0,
    metadata=None, max_len=512,
):
    """Train a per-residue model using mean-pooled reconstruction + optional losses."""
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    train_set = set(train_ids)
    test_set = set(test_ids)

    train_emb = {k: v for k, v in embeddings.items() if k in train_set}
    train_seq = {k: v for k, v in sequences.items() if k in train_set}
    val_emb = {k: v for k, v in embeddings.items() if k in test_set}

    g = torch.Generator().manual_seed(seed)
    train_ds = ResidueEmbeddingDataset(train_emb, train_seq, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    val_ds = ResidueEmbeddingDataset(val_emb, max_len=max_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    vicreg_fn = VICRegLoss() if vicreg_weight > 0 else None

    # Contrastive setup
    infonce_fn = None
    id_to_label_idx = None
    if contrastive_weight > 0 and metadata is not None:
        id_to_fam = {m["id"]: m["family"] for m in metadata}
        unique_fams = sorted(set(id_to_fam.get(pid, "?") for pid in train_ids if pid in id_to_fam))
        fam_to_idx = {f: i for i, f in enumerate(unique_fams)}
        id_to_label_idx = {pid: fam_to_idx[id_to_fam[pid]] for pid in train_ids if pid in id_to_fam}
        infonce_fn = InfoNCEFamilyLoss(temperature=0.07)

    best_val_loss = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            states = batch["states"].to(device)
            mask = batch["mask"].to(device)
            pids = batch["id"]

            optimizer.zero_grad()
            output = model(states, mask)

            # Mean-pooled reconstruction loss
            pooled_pred = output["pooled_recon"]
            pooled_target = output["pooled_input"]
            mse = F.mse_loss(pooled_pred, pooled_target)
            cos_loss = 1.0 - F.cosine_similarity(pooled_pred, pooled_target, dim=-1).mean()
            loss = mse + 0.5 * cos_loss

            # VICReg
            if vicreg_fn is not None:
                z = output["latent"].squeeze(1)  # (B, D')
                vr = vicreg_fn(z)
                loss = loss + vicreg_weight * vr["loss"]

            # Contrastive
            if infonce_fn is not None and id_to_label_idx is not None:
                z = output["latent"].squeeze(1)
                labels = torch.tensor(
                    [id_to_label_idx.get(pid, -1) for pid in pids],
                    dtype=torch.long, device=device,
                )
                valid = labels >= 0
                if valid.sum() > 1:
                    cl = infonce_fn(z[valid], labels[valid])
                    loss = loss + contrastive_weight * cl["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device)
                mask = batch["mask"].to(device)
                output = model(states, mask)
                val_mse = F.mse_loss(output["pooled_recon"], output["pooled_input"])
                val_losses.append(val_mse.item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"    Epoch {epoch:3d}/{epochs} | Loss={avg_loss:.4f} | Val={val_loss:.6f} | {elapsed:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, time.time() - start


def evaluate_per_residue_model(model, embeddings, metadata,
                               eval_ids, test_ids, cls_train_ids, cls_test_ids, device):
    """Evaluate a per-residue model using the standard benchmark flow."""
    ret = evaluate_retrieval(
        model, embeddings, metadata, label_key="family",
        query_ids=eval_ids, database_ids=test_ids, device=device,
    )
    cls_result = evaluate_linear_probe(
        model, embeddings, metadata, label_key="family",
        train_ids=cls_train_ids, test_ids=cls_test_ids, device=device,
    )
    ret_sf = evaluate_retrieval(
        model, embeddings, metadata, label_key="superfamily",
        query_ids=eval_ids, database_ids=test_ids, device=device,
    )
    return ret, cls_result, ret_sf


# ── B-5: DeepSets Attention Pooling ──────────────────────────────


def step_b5(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B-5: DeepSets attention pooling + MLP encoder")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    configs = [
        # (suffix, latent_dim, hidden_dims, vicreg_w, contrastive_w)
        ("base_d128", 128, (512,), 0.0, 0.0),
        ("base_d256", 256, (512,), 0.0, 0.0),
        ("deep_d128", 128, (512, 256), 0.1, 0.0),
        ("deep_d256", 256, (512, 256), 0.1, 0.0),
        ("contrastive_d128", 128, (512, 256), 0.1, 0.5),
        ("contrastive_d256", 256, (512, 256), 0.1, 0.5),
    ]

    for suffix, latent_dim, hidden_dims, vicreg_w, contrastive_w in configs:
        for seed in SEEDS:
            name = f"deepsets_attn_{suffix}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = DeepSetsAttentionCompressor(
                embed_dim, latent_dim=latent_dim,
                attn_hidden=256,
                hidden_dims=hidden_dims,
                dropout=0.1,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  D={embed_dim} -> attn -> {hidden_dims} -> {latent_dim}d, params={n_params:,}")

            model, elapsed = train_per_residue_model(
                model, embeddings, sequences, train_ids, test_ids,
                device, seed=seed, epochs=100, batch_size=16,
                vicreg_weight=vicreg_w, contrastive_weight=contrastive_w,
                metadata=metadata if contrastive_w > 0 else None,
            )
            print(f"  Training done in {elapsed:.0f}s")

            ret, cls_result, ret_sf = evaluate_per_residue_model(
                model, embeddings, metadata, eval_ids, test_ids,
                cls_train_ids, cls_test_ids, device,
            )

            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": latent_dim,
                "hidden_dims": list(hidden_dims),
                "vicreg_weight": vicreg_w,
                "contrastive_weight": contrastive_w,
                "n_params": n_params,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls_result,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_acc = cls_result.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── B-6: Dual-Pool Concat ────────────────────────────────────────


class DualPoolCompressor(nn.Module):
    """Concatenate mean-pool + attention-pool, then MLP encode."""

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 128,
        attn_hidden: int = 256,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._latent_dim = latent_dim
        self._n_tokens = 1

        # DeepSets attention
        self.attn_V = nn.Linear(embed_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1, bias=False)

        # MLP encoder (input: 2*embed_dim from concat)
        enc_layers = []
        in_dim = embed_dim * 2
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (back to embed_dim, not 2*embed_dim)
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, embed_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

    @property
    def num_tokens(self):
        return 1

    @property
    def latent_dim(self):
        return self._latent_dim

    def attention_pool(self, residue_states, mask):
        scores = self.attn_w(torch.tanh(self.attn_V(residue_states))).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return (residue_states * weights.unsqueeze(-1)).sum(dim=1)

    def mean_pool(self, residue_states, mask):
        mask_f = mask.unsqueeze(-1).float()
        return (residue_states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def encode(self, x):
        return self.encoder(x)

    def decode_latent(self, z):
        return self.decoder_net(z)

    def compress(self, residue_states, mask):
        mean_p = self.mean_pool(residue_states, mask)
        attn_p = self.attention_pool(residue_states, mask)
        concat = torch.cat([mean_p, attn_p], dim=-1)
        z = self.encode(concat)
        return z.unsqueeze(1)

    def get_pooled(self, latent, strategy="mean"):
        return latent.squeeze(1)

    def forward(self, residue_states, mask):
        mean_p = self.mean_pool(residue_states, mask)
        attn_p = self.attention_pool(residue_states, mask)
        concat = torch.cat([mean_p, attn_p], dim=-1)
        z = self.encode(concat)
        recon = self.decode_latent(z)
        target_length = residue_states.shape[1]
        return {
            "latent": z.unsqueeze(1),
            "reconstructed": recon.unsqueeze(1).expand(-1, target_length, -1),
            "pooled_input": mean_p,
            "pooled_recon": recon,
        }


def step_b6(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B-6: Dual-pool concat (mean + attention)")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    configs = [
        ("d128", 128, (1024, 512), 0.1, 0.0),
        ("d256", 256, (1024, 512), 0.1, 0.0),
        ("contrastive_d128", 128, (1024, 512), 0.1, 0.5),
    ]

    for suffix, latent_dim, hidden_dims, vicreg_w, contrastive_w in configs:
        for seed in SEEDS:
            name = f"dual_pool_{suffix}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = DualPoolCompressor(
                embed_dim, latent_dim=latent_dim,
                attn_hidden=256, hidden_dims=hidden_dims, dropout=0.1,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  D={embed_dim}*2 -> {hidden_dims} -> {latent_dim}d, params={n_params:,}")

            model, elapsed = train_per_residue_model(
                model, embeddings, sequences, train_ids, test_ids,
                device, seed=seed, epochs=100, batch_size=16,
                vicreg_weight=vicreg_w, contrastive_weight=contrastive_w,
                metadata=metadata if contrastive_w > 0 else None,
            )
            print(f"  Training done in {elapsed:.0f}s")

            # Evaluate
            ret = evaluate_retrieval(
                model, embeddings, metadata, label_key="family",
                query_ids=eval_ids, database_ids=test_ids, device=device,
            )
            cls_result = evaluate_linear_probe(
                model, embeddings, metadata, label_key="family",
                train_ids=cls_train_ids, test_ids=cls_test_ids, device=device,
            )
            ret_sf = evaluate_retrieval(
                model, embeddings, metadata, label_key="superfamily",
                query_ids=eval_ids, database_ids=test_ids, device=device,
            )

            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": latent_dim,
                "hidden_dims": list(hidden_dims),
                "vicreg_weight": vicreg_w,
                "contrastive_weight": contrastive_w,
                "n_params": n_params,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls_result,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_acc = cls_result.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── B-7: Multi-Scale Pooling ─────────────────────────────────────


def step_b7(all_results, device, embeddings, metadata, sequences,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("B-7: Multi-scale pooling (mean + std + max)")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]

    configs = [
        ("mean_std_d128", 128, ("mean", "std"), (1024, 512), 0.1),
        ("mean_std_max_d128", 128, ("mean", "std", "max"), (1024, 512), 0.1),
        ("mean_std_d256", 256, ("mean", "std"), (1024, 512), 0.1),
        ("mean_std_max_d256", 256, ("mean", "std", "max"), (1024, 512), 0.1),
    ]

    for suffix, latent_dim, stats, hidden_dims, vicreg_w in configs:
        for seed in SEEDS:
            name = f"multiscale_{suffix}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = MultiScalePoolCompressor(
                embed_dim, latent_dim=latent_dim,
                stats=stats, hidden_dims=hidden_dims, dropout=0.1,
            )
            n_params = sum(p.numel() for p in model.parameters())
            n_stats = len(stats)
            print(f"  D={embed_dim}*{n_stats} -> {hidden_dims} -> {latent_dim}d, params={n_params:,}")

            model, elapsed = train_per_residue_model(
                model, embeddings, sequences, train_ids, test_ids,
                device, seed=seed, epochs=100, batch_size=16,
                vicreg_weight=vicreg_w,
            )
            print(f"  Training done in {elapsed:.0f}s")

            ret, cls_result, ret_sf = evaluate_per_residue_model(
                model, embeddings, metadata, eval_ids, test_ids,
                cls_train_ids, cls_test_ids, device,
            )

            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": latent_dim,
                "stats": list(stats),
                "hidden_dims": list(hidden_dims),
                "vicreg_weight": vicreg_w,
                "n_params": n_params,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls_result,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_acc = cls_result.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    print(f"\n{'='*100}")
    print("TRACK B RESULTS SUMMARY")
    print(f"{'='*100}")

    print(f"\n{'Name':<50} {'Ret@1':<8} {'MRR':<8} {'MAP':<8} {'Cls':<8} {'Dim'}")
    print("-" * 92)
    print(f"{'[ref] mean-pool (1280d)':<50} {'0.618':<8} {'-':<8} {'-':<8} {'-':<8} 1280")
    print(f"{'[ref] PCA-128':<50} {'0.454':<8} {'-':<8} {'-':<8} {'-':<8} 128")
    print(f"{'[ref] MLP-AE d256 (Phase 6)':<50} {'0.600':<8} {'-':<8} {'-':<8} {'0.729':<8} 256")
    print(f"{'[ref] AttnPool pool_recon (Phase 6)':<50} {'0.462':<8} {'-':<8} {'-':<8} {'0.411':<8} 128")
    print("-" * 92)

    ret_results = [r for r in all_results if "retrieval_family" in r]
    for r in ret_results:
        ret = r["retrieval_family"]
        p1 = ret.get("precision@1", 0)
        mrr = ret.get("mrr", "-")
        map_val = ret.get("map", "-")
        cls = r.get("classification_family", {}).get("accuracy_mean", "-")
        dim = r.get("latent_dim", "?")

        if isinstance(mrr, float):
            mrr = f"{mrr:.3f}"
        if isinstance(map_val, float):
            map_val = f"{map_val:.3f}"
        if isinstance(cls, float):
            cls = f"{cls:.3f}"

        print(f"{r['name']:<50} {p1:<8.3f} {mrr:<8} {map_val:<8} {cls:<8} {dim}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Track B: Per-residue approaches")
    parser.add_argument(
        "--step", nargs="*", default=None,
        help="Run specific step(s): B5 B6 B7. Default: all.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Results: {RESULTS_PATH}")

    all_results = load_results()
    print(f"Loaded {len(all_results)} existing results")

    filt_emb, filt_meta, filt_seq = load_5k_data()
    train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids = get_splits(filt_meta)

    stats = split_statistics(filt_meta, train_ids, test_ids, eval_ids)
    print(f"  Retrieval split: {stats['n_train']} train / {stats['n_test']} test")
    print(f"  Classification split: {len(cls_train_ids)} train / {len(cls_test_ids)} test")

    steps = args.step or ["B5", "B6", "B7"]
    steps = [s.upper() for s in steps]

    if "B5" in steps:
        all_results = step_b5(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B6" in steps:
        all_results = step_b6(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "B7" in steps:
        all_results = step_b7(
            all_results, device, filt_emb, filt_meta, filt_seq,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
