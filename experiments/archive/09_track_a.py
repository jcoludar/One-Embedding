#!/usr/bin/env python3
"""Phase 7, Track A: Enhanced mean-pool compressors.

A-1. Deeper MLP AE with residual connections + VICReg
A-2. Supervised contrastive fine-tuning (InfoNCE with family labels)
A-3. Hyperbolic + Euclidean product manifold

Usage:
  uv run python experiments/09_track_a.py                   # run all
  uv run python experiments/09_track_a.py --step A1         # deeper MLP AE
  uv run python experiments/09_track_a.py --step A2         # contrastive fine-tuning
  uv run python experiments/09_track_a.py --step A3         # hyperbolic component
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

from src.compressors.mlp_ae import MLPAutoencoder
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
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints" / "track_a"
RESULTS_PATH = DATA_DIR / "benchmarks" / "track_a_results.json"
SPLIT_DIR = DATA_DIR / "splits"

SEEDS = [42, 123]
SEEDS_EXTENDED = [42, 123, 456, 789]  # More seeds for contrastive variance estimation


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


def prepare_meanpool_data(embeddings, train_ids, test_ids):
    """Prepare mean-pooled vectors for training."""
    train_set = set(train_ids)
    test_set = set(test_ids)

    all_vecs = {}
    train_vecs, train_pids = [], []
    val_vecs, val_pids = [], []

    for pid, emb in embeddings.items():
        mp = emb.mean(axis=0)
        all_vecs[pid] = mp
        if pid in train_set:
            train_vecs.append(mp)
            train_pids.append(pid)
        elif pid in test_set:
            val_vecs.append(mp)
            val_pids.append(pid)

    X_train = np.array(train_vecs, dtype=np.float32)
    X_val = np.array(val_vecs, dtype=np.float32) if val_vecs else None

    return all_vecs, X_train, train_pids, X_val, val_pids


def encode_all_and_evaluate(model, all_vecs, metadata, eval_ids, test_ids,
                            cls_train_ids, cls_test_ids, device):
    """Encode all proteins with trained model and evaluate."""
    model.eval()
    encoded_vecs = {}
    with torch.no_grad():
        for pid, mp in all_vecs.items():
            x = torch.from_numpy(mp).unsqueeze(0).float().to(device)
            z = model.encode(x)
            encoded_vecs[pid] = z[0].cpu().numpy()

    # Mock embeddings for evaluation (shape (1, D') for mean-pool path)
    mock_emb = {pid: vec.reshape(1, -1) for pid, vec in encoded_vecs.items()}

    ret_results = evaluate_retrieval(
        None, mock_emb, metadata, label_key="family",
        query_ids=eval_ids, database_ids=test_ids,
    )
    cls_results = evaluate_linear_probe(
        None, mock_emb, metadata, label_key="family",
        train_ids=cls_train_ids, test_ids=cls_test_ids,
    )

    # Also evaluate retrieval by superfamily
    ret_sf_results = evaluate_retrieval(
        None, mock_emb, metadata, label_key="superfamily",
        query_ids=eval_ids, database_ids=test_ids,
    )

    return ret_results, cls_results, ret_sf_results


def train_mlp_ae(model, X_train, device, epochs=200, batch_size=64, lr=1e-3,
                 vicreg_weight=0.0, X_val=None, seed=42):
    """Train MLP AE on mean-pooled vectors with optional VICReg."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device) if X_val is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    vicreg_fn = VICRegLoss() if vicreg_weight > 0 else None

    best_val_loss = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            batch = X_train_t[perm[i:i + batch_size]]
            optimizer.zero_grad()

            z = model.encode(batch)
            recon = model.decode_latent(z)

            loss = F.mse_loss(recon, batch)
            cos_loss = 1.0 - F.cosine_similarity(recon, batch, dim=-1).mean()
            loss = loss + 0.5 * cos_loss

            if vicreg_fn is not None:
                vr = vicreg_fn(z)
                loss = loss + vicreg_weight * vr["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Validation
        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                z_val = model.encode(X_val_t)
                recon_val = model.decode_latent(z_val)
                val_loss = F.mse_loss(recon_val, X_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start
            val_str = f" | Val={val_loss:.6f}" if X_val_t is not None else ""
            print(f"    Epoch {epoch:3d}/{epochs} | Loss={avg_loss:.6f}{val_str} | {elapsed:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, time.time() - start


# ── A-1: Deeper MLP AE ──────────────────────────────────────────


def step_a1(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("A-1: Deeper MLP AE with residual connections + VICReg")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    all_vecs, X_train, train_pids, X_val, val_pids = prepare_meanpool_data(
        embeddings, train_ids, test_ids
    )

    configs = [
        # (name_suffix, latent_dim, hidden_dims, use_residual, vicreg_weight)
        ("deep_d128", 128, (512, 256), False, 0.0),
        ("deep_d256", 256, (512, 256), False, 0.0),
        ("deep_res_d128", 128, (512, 256), True, 0.0),
        ("deep_res_d256", 256, (512, 256), True, 0.0),
        ("deep_res_vicreg_d128", 128, (512, 256), True, 0.1),
        ("deep_res_vicreg_d256", 256, (512, 256), True, 0.1),
    ]

    for suffix, latent_dim, hidden_dims, use_residual, vicreg_w in configs:
        # Use extended seeds for the key config that A2 loads from
        seeds_for_config = SEEDS_EXTENDED if suffix == "deep_res_d256" else SEEDS
        for seed in seeds_for_config:
            name = f"mlp_ae_{suffix}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = MLPAutoencoder(
                embed_dim, latent_dim,
                hidden_dims=hidden_dims,
                use_residual=use_residual,
                dropout=0.1,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  D={embed_dim} -> {hidden_dims} -> {latent_dim}, params={n_params:,}, residual={use_residual}")

            model, elapsed = train_mlp_ae(
                model, X_train, device, epochs=200, batch_size=64,
                vicreg_weight=vicreg_w, X_val=X_val, seed=seed,
            )
            print(f"  Training done in {elapsed:.0f}s")

            ret, cls, ret_sf = encode_all_and_evaluate(
                model, all_vecs, metadata, eval_ids, test_ids,
                cls_train_ids, cls_test_ids, device,
            )

            # Save checkpoint
            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": latent_dim,
                "hidden_dims": list(hidden_dims),
                "use_residual": use_residual,
                "vicreg_weight": vicreg_w,
                "n_params": n_params,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_acc = cls.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── A-2: Supervised Contrastive Fine-tuning ──────────────────────


def step_a2(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("A-2: Supervised contrastive fine-tuning")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    all_vecs, X_train, train_pids, X_val, val_pids = prepare_meanpool_data(
        embeddings, train_ids, test_ids
    )

    # Build family label mapping for training proteins
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids if pid in id_to_fam))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}
    train_labels = np.array([fam_to_idx[id_to_fam[pid]] for pid in train_pids], dtype=np.int64)

    for seed in SEEDS_EXTENDED:
        # First check if we have a pre-trained model from A-1
        # Use the best A-1 config (deep_res_d256 preferred over vicreg variant)
        for pretrained_suffix in ["deep_res_d256", "deep_res_vicreg_d256", "deep_d256"]:
            pretrained_name = f"mlp_ae_{pretrained_suffix}_s{seed}"
            pretrained_ckpt = CHECKPOINTS_DIR / pretrained_name / "best_model.pt"
            if pretrained_ckpt.exists():
                break
        else:
            pretrained_ckpt = None

        for latent_dim in [128, 256]:
            name = f"mlp_ae_contrastive_d{latent_dim}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            model = MLPAutoencoder(
                embed_dim, latent_dim,
                hidden_dims=(512, 256),
                use_residual=True,
                dropout=0.1,
            )

            # Phase 1: Pre-train with reconstruction (or load checkpoint)
            if pretrained_ckpt is not None and latent_dim == 256:
                print(f"  Loading pre-trained weights from {pretrained_name}")
                model.load_state_dict(
                    torch.load(pretrained_ckpt, map_location=device, weights_only=True)
                )
                model = model.to(device)
            else:
                print(f"  Phase 1: Reconstruction pre-training...")
                model, elapsed_pretrain = train_mlp_ae(
                    model, X_train, device, epochs=100, batch_size=64,
                    vicreg_weight=0.1, X_val=X_val, seed=seed,
                )
                print(f"  Pre-training done in {elapsed_pretrain:.0f}s")

            # Phase 2: Fine-tune encoder with InfoNCE
            print(f"  Phase 2: Contrastive fine-tuning...")
            torch.manual_seed(seed)

            model = model.to(device)
            X_train_t = torch.from_numpy(X_train).to(device)
            labels_t = torch.from_numpy(train_labels).to(device)

            # Only fine-tune encoder, freeze decoder
            for p in model.dec_layers.parameters():
                p.requires_grad = False
            for p in model.dec_norms.parameters():
                p.requires_grad = False
            model.dec_proj.requires_grad_(False)

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            infonce_fn = InfoNCEFamilyLoss(temperature=0.07)

            start = time.time()
            batch_size = 128  # Larger batch for more negatives

            for epoch in range(1, 101):
                model.train()
                perm = torch.randperm(len(X_train_t))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i + batch_size]
                    batch = X_train_t[idx]
                    batch_labels = labels_t[idx]

                    optimizer.zero_grad()
                    z = model.encode(batch)
                    cl = infonce_fn(z, batch_labels)

                    # Small reconstruction regularization to prevent drift
                    recon = model.decode_latent(z.detach())
                    recon_loss = F.mse_loss(recon, batch)

                    loss = cl["loss"] + 0.1 * recon_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                if epoch % 25 == 0 or epoch == 1:
                    elapsed = time.time() - start
                    print(f"    Epoch {epoch:3d}/100 | Loss={epoch_loss/n_batches:.4f} | {elapsed:.0f}s")

            elapsed = time.time() - start
            print(f"  Contrastive fine-tuning done in {elapsed:.0f}s")

            # Unfreeze for evaluation
            for p in model.parameters():
                p.requires_grad = True

            ret, cls, ret_sf = encode_all_and_evaluate(
                model, all_vecs, metadata, eval_ids, test_ids,
                cls_train_ids, cls_test_ids, device,
            )

            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": latent_dim,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_acc = cls.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_acc:.3f}")

    return all_results


# ── A-3: Hyperbolic Component ────────────────────────────────────


class PoincareBallEncoder(nn.Module):
    """Project Euclidean embeddings onto the Poincaré ball model of hyperbolic space.

    Uses exponential map at the origin for projection.
    Hyperbolic distance better preserves hierarchical relationships.
    """

    def __init__(self, input_dim: int, hyp_dim: int, curvature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hyp_dim)
        self.c = curvature
        self.eps = 1e-5

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at the origin of the Poincaré ball."""
        sqrt_c = self.c ** 0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball. Input (B, D), output (B, hyp_dim)."""
        v = self.proj(x)
        return self.expmap0(v)


class ProductManifoldEncoder(nn.Module):
    """Product manifold: hyperbolic (Poincaré) + Euclidean components.

    Hyperbolic space for hierarchical structure (family ⊂ superfamily ⊂ fold).
    Euclidean space for content/function similarity.
    """

    def __init__(
        self,
        input_dim: int,
        hyp_dim: int = 64,
        euc_dim: int = 64,
        hidden_dim: int = 512,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.euc_dim = euc_dim
        self.total_dim = hyp_dim + euc_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Euclidean head
        self.euc_head = nn.Linear(hidden_dim, euc_dim)

        # Hyperbolic head
        self.hyp_encoder = PoincareBallEncoder(hidden_dim, hyp_dim, curvature)

        # Decoder (from concatenated back to input)
        self.decoder = nn.Sequential(
            nn.Linear(hyp_dim + euc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to hyperbolic + Euclidean components."""
        h = self.shared(x)
        z_hyp = self.hyp_encoder(h)
        z_euc = self.euc_head(h)
        return z_hyp, z_euc

    def decode(self, z_hyp: torch.Tensor, z_euc: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_hyp, z_euc], dim=-1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_hyp, z_euc = self.encode(x)
        recon = self.decode(z_hyp, z_euc)
        z_combined = torch.cat([z_hyp, z_euc], dim=-1)
        return {
            "z_hyp": z_hyp,
            "z_euc": z_euc,
            "z_combined": z_combined,
            "recon": recon,
        }


def hyperbolic_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5):
    """Poincaré ball distance between points u and v."""
    sqrt_c = c ** 0.5
    diff = u - v
    u_norm_sq = u.pow(2).sum(dim=-1).clamp(max=1 - eps)
    v_norm_sq = v.pow(2).sum(dim=-1).clamp(max=1 - eps)
    diff_norm_sq = diff.pow(2).sum(dim=-1)
    num = 2 * diff_norm_sq
    denom = (1 - u_norm_sq) * (1 - v_norm_sq)
    return (1 / sqrt_c) * torch.acosh(1 + num / denom.clamp(min=eps))


def step_a3(all_results, device, embeddings, metadata,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids):
    print(f"\n{'='*60}")
    print("A-3: Hyperbolic + Euclidean product manifold")
    print(f"{'='*60}")

    embed_dim = next(iter(embeddings.values())).shape[-1]
    all_vecs, X_train, train_pids, X_val, val_pids = prepare_meanpool_data(
        embeddings, train_ids, test_ids
    )

    # Build hierarchy labels for training proteins
    id_to_fam = {m["id"]: m["family"] for m in metadata}
    id_to_sf = {m["id"]: m["superfamily"] for m in metadata}
    id_to_fold = {m["id"]: m.get("fold", m.get("class_name", "")) for m in metadata}

    for hyp_dim, euc_dim in [(64, 64), (64, 192)]:
        total_dim = hyp_dim + euc_dim
        for seed in SEEDS:
            name = f"product_manifold_h{hyp_dim}_e{euc_dim}_s{seed}"
            if is_done(all_results, name):
                print(f"  {name} already done")
                continue

            print(f"\n  Training {name}...")
            monitor()

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ProductManifoldEncoder(
                embed_dim, hyp_dim=hyp_dim, euc_dim=euc_dim,
                hidden_dim=512, curvature=1.0,
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  D={embed_dim} -> hyp={hyp_dim} + euc={euc_dim} = {total_dim}d, params={n_params:,}")

            X_train_t = torch.from_numpy(X_train).to(device)
            X_val_t = torch.from_numpy(X_val).to(device) if X_val is not None else None

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            # Build family labels for contrastive loss
            unique_fams = sorted(set(id_to_fam[pid] for pid in train_pids if pid in id_to_fam))
            fam_to_idx = {f: i for i, f in enumerate(unique_fams)}
            train_labels = torch.tensor(
                [fam_to_idx[id_to_fam[pid]] for pid in train_pids],
                dtype=torch.long, device=device,
            )

            infonce_fn = InfoNCEFamilyLoss(temperature=0.1)
            vicreg_fn = VICRegLoss()

            best_val_loss = float("inf")
            best_state = None
            start = time.time()
            batch_size = 128

            for epoch in range(1, 201):
                model.train()
                perm = torch.randperm(len(X_train_t))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i + batch_size]
                    batch = X_train_t[idx]
                    batch_labels = train_labels[idx]

                    optimizer.zero_grad()
                    out = model(batch)

                    # Reconstruction loss
                    recon_loss = F.mse_loss(out["recon"], batch)
                    cos_loss = 1.0 - F.cosine_similarity(out["recon"], batch, dim=-1).mean()

                    # Contrastive on combined embedding
                    cl = infonce_fn(out["z_combined"], batch_labels)

                    # VICReg on Euclidean component
                    vr = vicreg_fn(out["z_euc"])

                    loss = recon_loss + 0.5 * cos_loss + 0.5 * cl["loss"] + 0.05 * vr["loss"]
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                # Validation
                if X_val_t is not None:
                    model.eval()
                    with torch.no_grad():
                        out_val = model(X_val_t)
                        val_loss = F.mse_loss(out_val["recon"], X_val_t).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if epoch % 50 == 0 or epoch == 1:
                    elapsed = time.time() - start
                    val_str = f" | Val={val_loss:.6f}" if X_val_t is not None else ""
                    print(f"    Epoch {epoch:3d}/200 | Loss={epoch_loss/n_batches:.4f}{val_str} | {elapsed:.0f}s")

            if best_state is not None:
                model.load_state_dict(best_state)

            elapsed = time.time() - start
            print(f"  Training done in {elapsed:.0f}s")

            # Encode all proteins
            model.eval()
            encoded_vecs = {}
            with torch.no_grad():
                for pid, mp in all_vecs.items():
                    x = torch.from_numpy(mp).unsqueeze(0).float().to(device)
                    out = model(x)
                    encoded_vecs[pid] = out["z_combined"][0].cpu().numpy()

            mock_emb = {pid: vec.reshape(1, -1) for pid, vec in encoded_vecs.items()}

            ret = evaluate_retrieval(
                None, mock_emb, metadata, label_key="family",
                query_ids=eval_ids, database_ids=test_ids,
            )
            cls_acc_result = evaluate_linear_probe(
                None, mock_emb, metadata, label_key="family",
                train_ids=cls_train_ids, test_ids=cls_test_ids,
            )
            ret_sf = evaluate_retrieval(
                None, mock_emb, metadata, label_key="superfamily",
                query_ids=eval_ids, database_ids=test_ids,
            )

            ckpt_dir = CHECKPOINTS_DIR / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            result = {
                "name": name,
                "split_mode": "held_out",
                "latent_dim": total_dim,
                "hyp_dim": hyp_dim,
                "euc_dim": euc_dim,
                "n_params": n_params,
                "seed": seed,
                "retrieval_family": ret,
                "retrieval_superfamily": ret_sf,
                "classification_family": cls_acc_result,
                "training_time_s": elapsed,
            }
            all_results.append(result)
            save_results(all_results)

            p1 = ret.get("precision@1", 0)
            mrr = ret.get("mrr", 0)
            cls_a = cls_acc_result.get("accuracy_mean", 0)
            print(f"  >> {name}: Ret@1={p1:.3f}, MRR={mrr:.3f}, Cls={cls_a:.3f}")

    return all_results


# ── Summary ──────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    print(f"\n{'='*100}")
    print("TRACK A RESULTS SUMMARY")
    print(f"{'='*100}")

    # Reference baselines
    print(f"\n{'Name':<50} {'Ret@1':<8} {'MRR':<8} {'MAP':<8} {'Cls':<8} {'Dim'}")
    print("-" * 92)
    print(f"{'[ref] mean-pool (1280d)':<50} {'0.618':<8} {'-':<8} {'-':<8} {'-':<8} 1280")
    print(f"{'[ref] PCA-128':<50} {'0.454':<8} {'-':<8} {'-':<8} {'-':<8} 128")
    print(f"{'[ref] MLP-AE d128 (Phase 6)':<50} {'0.588':<8} {'-':<8} {'-':<8} {'0.719':<8} 128")
    print(f"{'[ref] MLP-AE d256 (Phase 6)':<50} {'0.600':<8} {'-':<8} {'-':<8} {'0.729':<8} 256")
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
    parser = argparse.ArgumentParser(description="Phase 7 Track A: Enhanced MLP compressors")
    parser.add_argument(
        "--step", nargs="*", default=None,
        help="Run specific step(s): A1 A2 A3. Default: all.",
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

    steps = args.step or ["A1", "A2", "A3"]
    steps = [s.upper() for s in steps]

    if "A1" in steps:
        all_results = step_a1(
            all_results, device, filt_emb, filt_meta,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "A2" in steps:
        all_results = step_a2(
            all_results, device, filt_emb, filt_meta,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    if "A3" in steps:
        all_results = step_a3(
            all_results, device, filt_emb, filt_meta,
            train_ids, test_ids, eval_ids, cls_train_ids, cls_test_ids,
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
