#!/usr/bin/env python3
"""Exp 55b: Extract ProtT5 embeddings for WT + variant sequences.

Reads ProteinGym DMS substitution CSVs (diversity subset) and ClinVar clinical
CSVs, embeds the WT sequence once and each variant sequence individually, then
writes per-assay/parent H5 groups.

H5 layout (per assay or per ClinVar parent):
    wt                  — (L, 1024) float16, gzip-compressed
    v_<idx>             — (L, 1024) float16 per variant, gzip-compressed
    attrs["variant_meta"] — JSON list of metadata dicts
    attrs["wt_sequence"]  — full WT amino acid string

Usage:
    uv run python experiments/55b_extract_variant_embeddings.py --smoke-test --skip-clinvar
    uv run python experiments/55b_extract_variant_embeddings.py --max-variants-per-assay 200
    uv run python experiments/55b_extract_variant_embeddings.py --skip-dms
    uv run python experiments/55b_extract_variant_embeddings.py  # full run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.one_embedding.vep import (
    load_clinvar_split,
    load_dms_assay_single_subs,
    prepare_reference_df,
    select_diversity_subset,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA = ROOT / "data"
PROTEINGYM_DIR = DATA / "proteingym"
REF_CSV = PROTEINGYM_DIR / "DMS_substitutions.csv"
DMS_DIR = PROTEINGYM_DIR / "DMS_substitutions" / "DMS_ProteinGym_substitutions"
CLINVAR_DIR = PROTEINGYM_DIR / "clinical_substitutions"
EMB_DIR = DATA / "residue_embeddings"

OUT_DMS = EMB_DIR / "prot_t5_xl_proteingym_diversity.h5"
OUT_CLINVAR = EMB_DIR / "prot_t5_xl_proteingym_clinvar.h5"

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
BATCH_SIZE = 1  # one protein at a time to avoid OOM; variants share WT length


# ---------------------------------------------------------------------------
# ProtT5 loader & embed helper
# ---------------------------------------------------------------------------

def _load_prot_t5(device: torch.device):
    """Load ProtT5 tokenizer + encoder. Returns (tokenizer, model)."""
    from transformers import AutoTokenizer, T5EncoderModel

    print(f"Loading {MODEL_NAME} on {device} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  (dim={model.config.d_model})")
    return tokenizer, model


def _embed_one(seq: str, tokenizer, model, device: torch.device) -> np.ndarray:
    """Embed a single protein sequence. Returns (L, 1024) float16 array.

    Applies ProtT5 preprocessing: insert spaces between residues, replace
    [UZOB] with X.
    """
    seq_clean = re.sub(r"[UZOB]", "X", seq.upper())
    seq_spaced = " ".join(list(seq_clean))
    encoded = tokenizer(
        [seq_spaced],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,  # hard limit — skip longer proteins
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state: (1, L_tokens, 1024).  First L positions = residues.
    L = len(seq_clean)
    emb = out.last_hidden_state[0, :L].cpu().numpy().astype(np.float16)
    return emb


# ---------------------------------------------------------------------------
# DMS extraction
# ---------------------------------------------------------------------------

def extract_diversity(
    ref_df: pd.DataFrame,
    n_assays: int = 15,
    max_variants_per_assay: Optional[int] = None,
    out_h5: Path = OUT_DMS,
    seed: int = 42,
) -> None:
    """Extract WT + variant embeddings for the diversity subset.

    ``n_assays`` must be 15 (the fixed diversity selection) or <= 15 in which
    case we take the first n_assays from the full 15-assay selection.
    """
    from src.utils.device import get_device

    device = get_device()
    # select_diversity_subset only supports n=15 (hardcoded 4+7+4 buckets).
    # For smoke tests with n<15 we select the full 15 and slice.
    if n_assays == 15:
        subset = select_diversity_subset(ref_df, n=15, seed=seed)
    else:
        subset = select_diversity_subset(ref_df, n=15, seed=seed)[:n_assays]
    print(f"\nSelected {len(subset)} DMS assays:")
    for a in subset:
        print(f"  {a.dms_id}  len={a.seq_len}  family={a.family}  type={a.fitness_type}")

    tokenizer, model = _load_prot_t5(device)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as hf:
        for assay_info in subset:
            dms_id = assay_info.dms_id
            csv_path = DMS_DIR / f"{dms_id}.csv"
            if not csv_path.exists():
                print(f"  [SKIP] {dms_id}: CSV not found at {csv_path}")
                continue

            print(f"\n[{dms_id}]  family={assay_info.family}  type={assay_info.fitness_type}")
            try:
                assay = load_dms_assay_single_subs(csv_path, dms_id)
            except Exception as exc:
                print(f"  [SKIP] load failed: {exc}")
                continue

            # Skip single-residue assays / very long proteins
            if len(assay.wt_sequence) > 900:
                print(f"  [SKIP] WT length {len(assay.wt_sequence)} > 900; too long for single-GPU pass")
                continue

            # Cap variants
            variants = assay.variants
            if max_variants_per_assay is not None:
                variants = variants[:max_variants_per_assay]

            grp = hf.require_group(dms_id)
            grp.attrs["wt_sequence"] = assay.wt_sequence

            # Embed WT
            print(f"  WT len={len(assay.wt_sequence)}  embedding ...")
            wt_emb = _embed_one(assay.wt_sequence, tokenizer, model, device)
            grp.create_dataset("wt", data=wt_emb, compression="gzip", compression_opts=4)

            # Embed variants
            meta_list = []
            for idx, var in enumerate(tqdm(variants, desc=f"  variants", leave=False)):
                v_emb = _embed_one(var.mutated_sequence, tokenizer, model, device)
                grp.create_dataset(
                    f"v_{idx}", data=v_emb, compression="gzip", compression_opts=4
                )
                meta_list.append({
                    "idx": idx,
                    "mut_pos": var.mut_pos,
                    "wt_aa": var.wt_aa,
                    "mut_aa": var.mut_aa,
                    "score": var.score,
                    "label": None,
                })

            grp.attrs["variant_meta"] = json.dumps(meta_list)
            print(f"  Done: WT + {len(variants)} variants written.")

    print(f"\nDMS output: {out_h5}")


# ---------------------------------------------------------------------------
# ClinVar extraction
# ---------------------------------------------------------------------------

def extract_clinvar(
    max_proteins: Optional[int] = None,
    out_h5: Path = OUT_CLINVAR,
) -> None:
    """Extract WT + variant embeddings for ClinVar proteins."""
    from src.utils.device import get_device

    if not CLINVAR_DIR.exists():
        print(f"[SKIP] ClinVar dir not found: {CLINVAR_DIR}")
        return

    device = get_device()
    csv_files = sorted(CLINVAR_DIR.glob("*.csv"))
    if max_proteins is not None:
        csv_files = csv_files[:max_proteins]

    print(f"\nClinVar: {len(csv_files)} parent protein files")
    tokenizer, model = _load_prot_t5(device)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as hf:
        for csv_path in tqdm(csv_files, desc="ClinVar proteins"):
            pid = csv_path.stem
            try:
                variants = load_clinvar_split(str(csv_path), pid)
            except Exception as exc:
                print(f"  [SKIP] {pid}: {exc}")
                continue
            if not variants:
                continue

            wt_seq = variants[0].wt_seq
            if len(wt_seq) > 900:
                print(f"  [SKIP] {pid}: WT length {len(wt_seq)} > 900")
                continue

            grp = hf.require_group(pid)
            grp.attrs["wt_sequence"] = wt_seq

            # Embed WT once
            wt_emb = _embed_one(wt_seq, tokenizer, model, device)
            grp.create_dataset("wt", data=wt_emb, compression="gzip", compression_opts=4)

            # Embed each variant
            meta_list = []
            for idx, var in enumerate(variants):
                v_emb = _embed_one(var.mutated_sequence, tokenizer, model, device)
                grp.create_dataset(
                    f"v_{idx}", data=v_emb, compression="gzip", compression_opts=4
                )
                meta_list.append({
                    "idx": idx,
                    "mut_pos": var.mut_pos,
                    "wt_aa": var.wt_aa,
                    "mut_aa": var.mut_aa,
                    "score": None,
                    "label": var.label,
                })

            grp.attrs["variant_meta"] = json.dumps(meta_list)

    print(f"\nClinVar output: {out_h5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ProtT5 embeddings for ProteinGym DMS + ClinVar variants."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick validation: 2 assays, 50 variants each, skip ClinVar.",
    )
    parser.add_argument(
        "--max-variants-per-assay",
        type=int,
        default=None,
        metavar="N",
        help="Cap variants per DMS assay.",
    )
    parser.add_argument(
        "--max-clinvar-proteins",
        type=int,
        default=None,
        metavar="N",
        help="Cap number of ClinVar parent proteins.",
    )
    parser.add_argument("--skip-dms", action="store_true", help="Skip DMS extraction.")
    parser.add_argument("--skip-clinvar", action="store_true", help="Skip ClinVar extraction.")
    args = parser.parse_args()

    # Load and prepare reference DataFrame
    if not REF_CSV.exists():
        print(f"ERROR: reference CSV not found: {REF_CSV}")
        sys.exit(1)
    ref_df = pd.read_csv(REF_CSV)
    ref_df = prepare_reference_df(ref_df)

    # Apply smoke-test overrides
    n_assays = 15
    max_variants = args.max_variants_per_assay
    max_clinvar = args.max_clinvar_proteins
    skip_clinvar = args.skip_clinvar

    if args.smoke_test:
        n_assays = 2
        max_variants = 50
        skip_clinvar = True
        print("=== SMOKE TEST: 2 assays, 50 variants each, ClinVar skipped ===")

    t_start = time.time()

    if not args.skip_dms:
        if not DMS_DIR.exists():
            print(f"ERROR: DMS directory not found: {DMS_DIR}")
            print("Run: uv run python experiments/55a_download_proteingym.py")
            sys.exit(1)
        extract_diversity(
            ref_df=ref_df,
            n_assays=n_assays,
            max_variants_per_assay=max_variants,
            out_h5=OUT_DMS,
        )

    if not skip_clinvar:
        extract_clinvar(
            max_proteins=max_clinvar,
            out_h5=OUT_CLINVAR,
        )

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
