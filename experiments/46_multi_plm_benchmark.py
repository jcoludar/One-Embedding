#!/usr/bin/env python3
"""Experiment 46: Multi-PLM Benchmark Pipeline.

Extracts embeddings and runs the full benchmark suite for any registered PLM.
Produces a unified comparison table across all PLMs and codec configurations.

Uses OneEmbeddingCodec directly (not manual reimplementation) to ensure
benchmark results match the actual shipped codec behavior.

Usage:
    uv run python experiments/46_multi_plm_benchmark.py --list
    uv run python experiments/46_multi_plm_benchmark.py --plm prot_t5_full
    uv run python experiments/46_multi_plm_benchmark.py --plm prot_t5_full --smoke-test
    uv run python experiments/46_multi_plm_benchmark.py --extract-only --plm esm2_650m
    uv run python experiments/46_multi_plm_benchmark.py  # all PLMs, full benchmarks

Notes:
    - ESM-C requires the EvolutionaryScale `esm` package (NOT fair-esm):
      uv run --with esm python experiments/46_multi_plm_benchmark.py --plm esmc_300m
    - ESM-C runs on CPU only (MPS not supported)
    - ProtT5 half-precision uses Rostlab/prot_t5_xl_half_uniref50 (fp16 weights)
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "43_rigorous_benchmark"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

DATA = ROOT / "data"
EMB_DIR = DATA / "residue_embeddings"
RESULTS_DIR = DATA / "benchmarks" / "rigorous_v1"

# Benchmark config (matches Exp 43)
SEEDS = [42, 123, 456]
BOOTSTRAP_N = 10_000
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CV_FOLDS = 3


# ═══════════════════════════════════════════════════════════════════════
# PLM Registry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PLMConfig:
    """Configuration for a protein language model."""
    name: str
    display: str
    dim: int
    extractor: str      # key into EXTRACTORS
    model_id: str
    batch_size: int = 4
    max_len: int = 512  # max sequence length for extraction
    device: str = "auto"  # "auto", "cpu", "mps", "cuda"


# ── Extractors ────────────────────────────────────────────────────────

def _sanitize_sequences(fasta_dict: dict) -> dict:
    """Remove non-standard amino acid characters that crash tokenizers."""
    import re
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    clean = {}
    for pid, seq in fasta_dict.items():
        # Replace non-standard with X, remove stop codons
        seq = seq.replace("*", "").replace(".", "")
        seq = re.sub(r"[UZOB]", "X", seq)
        seq = "".join(c if c in standard_aa else "X" for c in seq.upper())
        if len(seq) > 0:
            clean[pid] = seq
    return clean


def _extract_esm2(fasta_dict, model_name, batch_size, device=None, **kw):
    """ESM2 extraction via fair-esm."""
    from src.extraction.esm_extractor import extract_residue_embeddings
    fasta_dict = _sanitize_sequences(fasta_dict)
    return extract_residue_embeddings(
        fasta_dict, model_name=model_name, batch_size=batch_size, device=device
    )


def _extract_prot_t5(fasta_dict, model_name, batch_size, device=None, **kw):
    """ProtT5 extraction via HuggingFace transformers."""
    from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
    fasta_dict = _sanitize_sequences(fasta_dict)
    return extract_prot_t5_embeddings(
        fasta_dict, model_name=model_name, batch_size=batch_size, device=device
    )


def _extract_prot_t5_half(fasta_dict, model_name, batch_size, device=None, **kw):
    """ProtT5 extraction in half precision — load full model, cast to fp16."""
    import re
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    from tqdm import tqdm
    from src.utils.device import get_device

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    embed_dim = model.config.d_model
    print(f"Loaded {model_name} (fp16, dim={embed_dim}) on {device}")

    ids = list(fasta_dict.keys())
    embeddings = {}

    for i in tqdm(range(0, len(ids), batch_size), desc="Extracting ProtT5 fp16"):
        batch_ids = ids[i:i + batch_size]
        batch_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", fasta_dict[sid])))
                      for sid in batch_ids]
        encoded = tokenizer(batch_seqs, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        for j, sid in enumerate(batch_ids):
            seq_len = len(fasta_dict[sid])
            emb = output.last_hidden_state[j, :seq_len].cpu().numpy().astype(np.float32)
            embeddings[sid] = emb

    print(f"Extracted {len(embeddings)} proteins, embed_dim={embed_dim}")
    return embeddings


def _extract_esmc(fasta_dict, model_name, batch_size, device=None, max_len=2000, **kw):
    """ESM-C extraction via EvolutionaryScale esm package.

    NOTE: Requires `esm` from EvolutionaryScale (NOT fair-esm).
    Install: pip install esm  (or uv run --with esm)
    ESM-C runs on CPU only — MPS is not supported.
    """
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError(
            "ESM-C requires the EvolutionaryScale esm package:\n"
            "  pip install esm\n"
            "  (This is NOT fair-esm — they conflict. Use: uv run --with esm ...)"
        )

    import torch
    from tqdm import tqdm

    # ESM-C: CPU only
    model = ESMC.from_pretrained(model_name).to("cpu")
    model.eval()
    print(f"Loaded {model_name} on cpu (ESM-C: CPU only)")

    ids = list(fasta_dict.keys())
    embeddings = {}

    for pid in tqdm(ids, desc=f"Extracting {model_name}"):
        seq = fasta_dict[pid]
        if len(seq) > max_len:
            seq = seq[:max_len]
        if not seq:
            continue

        try:
            with torch.no_grad():
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(return_embeddings=True)
                )
                emb = logits_output.embeddings
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy().astype(np.float32)
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb[0]
                # Strip BOS/EOS tokens
                if emb.shape[0] == len(seq) + 2:
                    emb = emb[1:-1]
                embeddings[pid] = emb
        except Exception as e:
            print(f"  ERROR {pid} (len={len(seq)}): {e}")

    dim = next(iter(embeddings.values())).shape[1] if embeddings else 0
    print(f"Extracted {len(embeddings)} proteins, embed_dim={dim}")
    return embeddings


def _extract_ankh(fasta_dict, model_name, batch_size, device=None, **kw):
    """ANKH extraction — T5-based, raw sequence input (NOT space-separated).

    ANKH tokenizer maps each AA to a single token. The raw sequence produces
    (L+2) tokens: BOS + L amino acids + EOS. We take positions 1..L+1.
    """
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    from tqdm import tqdm
    from src.utils.device import get_device

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    embed_dim = model.config.d_model
    print(f"Loaded {model_name} (dim={embed_dim}) on {device}")

    ids = list(fasta_dict.keys())
    embeddings = {}

    for i in tqdm(range(0, len(ids), batch_size), desc="Extracting ANKH"):
        batch_ids = ids[i:i + batch_size]
        # ANKH: raw sequence, no space separation
        batch_seqs = [fasta_dict[sid] for sid in batch_ids]
        encoded = tokenizer(batch_seqs, padding=True, truncation=True,
                            max_length=514, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        for j, sid in enumerate(batch_ids):
            seq_len = len(fasta_dict[sid])
            # Skip BOS token at position 0, take L residue positions
            emb = output.last_hidden_state[j, 1:seq_len + 1].cpu().numpy().astype(np.float32)
            embeddings[sid] = emb

    print(f"Extracted {len(embeddings)} proteins, embed_dim={embed_dim}")
    return embeddings


EXTRACTORS: dict[str, Callable] = {
    "esm2": _extract_esm2,
    "prot_t5": _extract_prot_t5,
    "prot_t5_half": _extract_prot_t5_half,
    "esmc": _extract_esmc,
    "ankh": _extract_ankh,
}

# Registered PLMs
PLMS: dict[str, PLMConfig] = {
    "prot_t5_full": PLMConfig(
        name="prot_t5_full", display="ProtT5-XL (fp32)",
        dim=1024, extractor="prot_t5",
        model_id="Rostlab/prot_t5_xl_uniref50",
    ),
    "prot_t5_half": PLMConfig(
        name="prot_t5_half", display="ProtT5-XL (fp16)",
        dim=1024, extractor="prot_t5_half",
        model_id="Rostlab/prot_t5_xl_uniref50",
    ),
    "esm2_650m": PLMConfig(
        name="esm2_650m", display="ESM2-650M",
        dim=1280, extractor="esm2",
        model_id="esm2_t33_650M_UR50D",
    ),
    "esmc_300m": PLMConfig(
        name="esmc_300m", display="ESM-C 300M",
        dim=960, extractor="esmc",
        model_id="esmc_300m", device="cpu", max_len=2000,
    ),
    "esmc_600m": PLMConfig(
        name="esmc_600m", display="ESM-C 600M",
        dim=1152, extractor="esmc",
        model_id="esmc_600m", device="cpu", max_len=2000,
    ),
    "prostt5": PLMConfig(
        name="prostt5", display="ProstT5",
        dim=1024, extractor="prot_t5",
        model_id="Rostlab/ProstT5",
    ),
    "ankh_large": PLMConfig(
        name="ankh_large", display="ANKH-large",
        dim=1536, extractor="ankh",
        model_id="ElnaggarLab/ankh-large",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Embedding file paths
# ═══════════════════════════════════════════════════════════════════════

# Map (plm, dataset) → existing filename on disk (legacy names)
_EXISTING = {
    ("prot_t5_full", "cb513"): "prot_t5_xl_cb513.h5",
    ("prot_t5_full", "ts115"): "prot_t5_xl_ts115.h5",
    ("prot_t5_full", "casp12"): "prot_t5_xl_casp12.h5",
    ("prot_t5_full", "chezod"): "prot_t5_xl_chezod.h5",
    ("prot_t5_full", "trizod"): "prot_t5_xl_trizod.h5",
    ("prot_t5_full", "scope_5k"): "prot_t5_xl_medium5k.h5",
    ("prot_t5_full", "cath20"): "prot_t5_xl_cath20.h5",
    ("prot_t5_full", "deeploc"): "prot_t5_xl_deeploc.h5",
    ("esm2_650m", "cb513"): "esm2_650m_cb513.h5",
    ("esm2_650m", "scope_5k"): "esm2_650m_medium5k.h5",
    ("esm2_650m", "chezod"): "esm2_650m_chezod.h5",
    ("esmc_300m", "cb513"): "esmc_300m_cb513.h5",
    ("esmc_300m", "scope_5k"): "esmc_300m_medium5k.h5",
    ("esmc_600m", "cb513"): "esmc_600m_cb513.h5",
    ("esmc_600m", "chezod"): "esmc_600m_chezod.h5",
    ("esmc_600m", "scope_5k"): "esmc_600m_scope_5k.h5",
}


def emb_path(plm_name: str, dataset: str) -> Path:
    """Embedding H5 file path for a PLM + dataset."""
    key = (plm_name, dataset)
    if key in _EXISTING:
        return EMB_DIR / _EXISTING[key]
    return EMB_DIR / f"{plm_name}_{dataset}.h5"


# ═══════════════════════════════════════════════════════════════════════
# Sequence loading (for extraction)
# ═══════════════════════════════════════════════════════════════════════

def load_sequences(dataset: str) -> dict[str, str]:
    """Load FASTA sequences for a benchmark dataset."""
    from src.extraction.data_loader import read_fasta
    from src.evaluation.per_residue_tasks import load_cb513_csv

    if dataset == "cb513":
        seqs, _, _, _ = load_cb513_csv(DATA / "per_residue_benchmarks" / "CB513.csv")
        return seqs
    elif dataset == "ts115":
        seqs, _, _, _ = load_cb513_csv(DATA / "per_residue_benchmarks" / "TS115.csv")
        return seqs
    elif dataset == "casp12":
        seqs, _, _, _ = load_cb513_csv(DATA / "per_residue_benchmarks" / "CASP12.csv")
        return seqs
    elif dataset == "chezod":
        train = read_fasta(str(DATA / "per_residue_benchmarks/SETH/CheZOD1174_training_set_sequences.fasta"))
        test = read_fasta(str(DATA / "per_residue_benchmarks/SETH/CheZOD117_test_set_sequences.fasta"))
        return {**train, **test}
    elif dataset == "trizod":
        rest = read_fasta(str(DATA / "per_residue_benchmarks/TriZOD/moderate_rest_set.fasta"))
        test = read_fasta(str(DATA / "per_residue_benchmarks/TriZOD/TriZOD_test_set.fasta"))
        return {**rest, **test}
    elif dataset == "scope_5k":
        path = DATA / "proteins" / "medium_diverse_5k.fasta"
        if path.exists():
            return read_fasta(str(path))
        raise FileNotFoundError(f"SCOPe FASTA not found: {path}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════

# Datasets needed for the full benchmark suite
NEEDED_DATASETS = ["cb513", "ts115", "casp12", "chezod", "trizod", "scope_5k"]


def extract_all(plm_names: list[str], datasets: list[str] = None):
    """Extract embeddings for PLMs × datasets. Skips existing files."""
    import torch
    from src.utils.h5_store import save_residue_embeddings

    if datasets is None:
        datasets = NEEDED_DATASETS

    for plm_name in plm_names:
        plm = PLMS[plm_name]
        print(f"\n{'='*60}")
        print(f"Extraction: {plm.display}")
        print(f"{'='*60}")

        for ds in datasets:
            path = emb_path(plm_name, ds)
            if path.exists():
                size_mb = path.stat().st_size / 1e6
                print(f"  {ds}: exists ({size_mb:.0f} MB)")
                continue

            try:
                seqs = load_sequences(ds)
            except FileNotFoundError as e:
                print(f"  {ds}: SKIP — {e}")
                continue

            print(f"  {ds}: extracting {len(seqs)} proteins...", flush=True)
            t0 = time.time()

            device = None if plm.device == "auto" else plm.device
            extractor_fn = EXTRACTORS[plm.extractor]
            embs = extractor_fn(seqs, model_name=plm.model_id,
                                batch_size=plm.batch_size, device=device,
                                max_len=plm.max_len)

            path.parent.mkdir(parents=True, exist_ok=True)
            save_residue_embeddings(embs, str(path))
            dt = time.time() - t0
            size_mb = path.stat().st_size / 1e6
            print(f"    Saved {len(embs)} proteins ({size_mb:.0f} MB, {dt:.0f}s)")

        # Free GPU memory between PLMs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmarking — uses OneEmbeddingCodec directly
# ═══════════════════════════════════════════════════════════════════════

def load_h5(path: Path) -> dict[str, np.ndarray]:
    import h5py
    with h5py.File(str(path), "r") as f:
        return {k: f[k][:].astype(np.float32) for k in f.keys()}


def run_benchmark_suite(
    plm_name: str,
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict:
    """Run the full benchmark suite for one PLM using OneEmbeddingCodec."""
    from src.one_embedding.codec_v2 import OneEmbeddingCodec
    from src.evaluation.per_residue_tasks import load_cb513_csv, load_chezod_seth
    from src.extraction.data_loader import load_metadata_csv
    from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark, pooled_spearman
    from runners.protein_level import run_retrieval_benchmark, compute_protein_vectors
    from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention

    plm = PLMS[plm_name]
    results = {
        "plm": plm.display, "plm_name": plm.name, "plm_dim": plm.dim,
        "codec": "center + RP896 + PQ224 (18x)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Load labels and splits ──
    _, ss3_labels, ss8_labels, _ = load_cb513_csv(DATA / "per_residue_benchmarks" / "CB513.csv")
    with open(DATA / "benchmark_suite" / "splits" / "cb513_80_20.json") as f:
        cb_split = json.load(f)
    with open(DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json") as f:
        sc_split = json.load(f)
    metadata = load_metadata_csv(DATA / "proteins" / "metadata_5k.csv")

    # ── Load embeddings ──
    cb_path = emb_path(plm_name, "cb513")
    sc_path = emb_path(plm_name, "scope_5k")
    cz_path = emb_path(plm_name, "chezod")

    # ── Fit codec on SCOPe train ──
    if not sc_path.exists():
        print(f"  No SCOPe embeddings for {plm.display} — cannot fit codec")
        return results

    sc_embs = load_h5(sc_path)
    sc_train = [k for k in sc_split["train_ids"] if k in sc_embs]
    sc_test = [k for k in sc_split["test_ids"] if k in sc_embs]

    print(f"  Fitting codec (d_out=896, PQ M=224) on SCOPe train...", flush=True)
    t0 = time.time()
    codec = OneEmbeddingCodec(d_out=896, quantization="pq", pq_m=224, abtt_k=0)
    codec.fit({k: sc_embs[k] for k in sc_train})
    results["fit_time_s"] = time.time() - t0

    def compress(embs_dict):
        """Encode + decode per-residue via the codec."""
        out = {}
        for pid, emb in embs_dict.items():
            encoded = codec.encode(emb)
            out[pid] = codec.decode_per_residue(encoded)
        return out

    # ── SS3 + SS8 on CB513 ──
    if cb_path.exists():
        cb_embs = load_h5(cb_path)
        cb_train = [p for p in cb_split["train_ids"] if p in cb_embs and p in ss3_labels]
        cb_test = [p for p in cb_split["test_ids"] if p in cb_embs and p in ss3_labels]
        comp_cb = compress(cb_embs)

        print(f"  SS3 on CB513 ({len(cb_test)} test)...", flush=True)
        ss3_raw = run_ss3_benchmark(cb_embs, ss3_labels, cb_train, cb_test,
                                     C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        ss3_comp = run_ss3_benchmark(comp_cb, ss3_labels, cb_train, cb_test,
                                      C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        ss3_ret = paired_bootstrap_retention(
            ss3_raw["per_protein_scores"], ss3_comp["per_protein_scores"],
            n_bootstrap=bootstrap_n, seed=SEEDS[0])
        results["ss3"] = _pack(ss3_raw["q3"], ss3_comp["q3"], ss3_ret)
        print(f"    {results['ss3']['raw']:.4f} -> {results['ss3']['comp']:.4f} "
              f"({results['ss3']['retention']:.1f}%)", flush=True)

        print(f"  SS8 on CB513...", flush=True)
        ss8_raw = run_ss8_benchmark(cb_embs, ss8_labels, cb_train, cb_test,
                                     C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        ss8_comp = run_ss8_benchmark(comp_cb, ss8_labels, cb_train, cb_test,
                                      C_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        ss8_ret = paired_bootstrap_retention(
            ss8_raw["per_protein_scores"], ss8_comp["per_protein_scores"],
            n_bootstrap=bootstrap_n, seed=SEEDS[0])
        results["ss8"] = _pack(ss8_raw["q8"], ss8_comp["q8"], ss8_ret)
        print(f"    {results['ss8']['raw']:.4f} -> {results['ss8']['comp']:.4f} "
              f"({results['ss8']['retention']:.1f}%)", flush=True)
    else:
        print(f"  SKIP SS3/SS8 — no CB513 embeddings")

    # ── Retrieval on SCOPe ──
    comp_sc = compress(sc_embs)
    print(f"  Retrieval on SCOPe ({len(sc_test)} test)...", flush=True)
    raw_vecs = compute_protein_vectors(sc_embs, method="dct_k4")
    comp_vecs = compute_protein_vectors(comp_sc, method="dct_k4")
    ret_raw = run_retrieval_benchmark(raw_vecs, metadata, label_key="family",
                                       n_bootstrap=bootstrap_n, seed=SEEDS[0])
    ret_comp = run_retrieval_benchmark(comp_vecs, metadata, label_key="family",
                                        n_bootstrap=bootstrap_n, seed=SEEDS[0])
    ret_ret = paired_bootstrap_retention(
        ret_raw["per_query_cosine"], ret_comp["per_query_cosine"],
        n_bootstrap=bootstrap_n, seed=SEEDS[0])
    results["ret1"] = _pack(ret_raw["ret1_cosine"], ret_comp["ret1_cosine"], ret_ret)
    print(f"    {results['ret1']['raw']:.4f} -> {results['ret1']['comp']:.4f} "
          f"({results['ret1']['retention']:.1f}%)", flush=True)

    # ── Disorder on CheZOD ──
    if cz_path.exists():
        cz_embs = load_h5(cz_path)
        _, cz_scores, cz_train, cz_test = load_chezod_seth(DATA / "per_residue_benchmarks")
        cz_train = [p for p in cz_train if p in cz_embs and p in cz_scores]
        cz_test = [p for p in cz_test if p in cz_embs and p in cz_scores]
        comp_cz = compress(cz_embs)

        print(f"  Disorder on CheZOD ({len(cz_test)} test)...", flush=True)
        dis_raw = run_disorder_benchmark(cz_embs, cz_scores, cz_train, cz_test,
                                          ALPHA_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        dis_comp = run_disorder_benchmark(comp_cz, cz_scores, cz_train, cz_test,
                                           ALPHA_GRID, CV_FOLDS, SEEDS, bootstrap_n)
        dis_ret = paired_cluster_bootstrap_retention(
            dis_raw["per_protein_predictions"], dis_comp["per_protein_predictions"],
            pooled_spearman, n_bootstrap=bootstrap_n, seed=SEEDS[0])
        results["disorder"] = _pack(dis_raw["pooled_spearman_rho"],
                                     dis_comp["pooled_spearman_rho"], dis_ret)
        print(f"    {results['disorder']['raw']:.4f} -> {results['disorder']['comp']:.4f} "
              f"({results['disorder']['retention']:.1f}%)", flush=True)
    else:
        print(f"  SKIP Disorder — no CheZOD embeddings")

    return results


def _pack(raw_metric, comp_metric, retention_metric) -> dict:
    """Pack MetricResult objects into a serializable dict."""
    return {
        "raw": raw_metric.value,
        "raw_ci": [raw_metric.ci_lower, raw_metric.ci_upper],
        "comp": comp_metric.value,
        "comp_ci": [comp_metric.ci_lower, comp_metric.ci_upper],
        "retention": retention_metric.value,
        "retention_ci": [retention_metric.ci_lower, retention_metric.ci_upper],
        "n": raw_metric.n,
    }


# ═══════════════════════════════════════════════════════════════════════
# Giant Table
# ═══════════════════════════════════════════════════════════════════════

def print_table(all_results: dict):
    """Print unified comparison table with legend."""
    print(f"\n{'='*100}")
    print(f"MULTI-PLM BENCHMARK: center + RP 896d + PQ M=224 (~18x)")
    print(f"{'='*100}")
    print()
    print("Column definitions:")
    print("  SS3 raw   — Secondary structure 3-class accuracy (Q3: H/E/C) on raw embeddings")
    print("  SS3 ret   — Retention: compressed Q3 / raw Q3 × 100% (paired bootstrap CI)")
    print("  SS8 raw   — Secondary structure 8-class accuracy (Q8: H/B/E/G/I/T/S/C) on raw")
    print("  SS8 ret   — Retention of Q8 after compression")
    print("  Ret raw   — Family retrieval Ret@1 (cosine kNN) on raw protein vectors")
    print("  Ret ret   — Retention of Ret@1 after compression")
    print("  Dis raw   — Disorder prediction: pooled residue-level Spearman rho vs CheZOD Z-scores")
    print("  Dis ret   — Retention of pooled Spearman rho after compression")
    print()
    print("Methodology: CV-tuned linear probes (LogReg C / Ridge alpha via GridSearchCV),")
    print("  3-seed averaged predictions (Bouthillier 2021), BCa bootstrap CIs (B=10,000),")
    print("  paired retention CIs, cluster bootstrap for disorder (Davison & Hinkley 1997).")
    print("  Codec fitted on external SCOPe 5K corpus. See experiments/43_rigorous_benchmark/")
    print()

    def _fmt(r, key):
        if key not in r:
            return "  —   ", "  —  "
        d = r[key]
        half_ci = (d["retention_ci"][1] - d["retention_ci"][0]) / 2
        return f"{d['raw']:.3f}", f"{d['retention']:.1f}±{half_ci:.1f}%"

    header = (f"{'PLM':<22} {'dim':>4} "
              f"{'SS3 raw':>8} {'SS3 ret':>10} "
              f"{'SS8 raw':>8} {'SS8 ret':>10} "
              f"{'Ret raw':>8} {'Ret ret':>10} "
              f"{'Dis raw':>8} {'Dis ret':>10}")
    print(header)
    print("-" * len(header))

    for plm_name, r in all_results.items():
        ss3_r, ss3_p = _fmt(r, "ss3")
        ss8_r, ss8_p = _fmt(r, "ss8")
        ret_r, ret_p = _fmt(r, "ret1")
        dis_r, dis_p = _fmt(r, "disorder")
        dim = r.get("plm_dim", "?")
        print(f"{r.get('plm', plm_name):<22} {dim:>4} "
              f"{ss3_r:>8} {ss3_p:>10} "
              f"{ss8_r:>8} {ss8_p:>10} "
              f"{ret_r:>8} {ret_p:>10} "
              f"{dis_r:>8} {dis_p:>10}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-PLM Benchmark Pipeline")
    parser.add_argument("--plm", nargs="+", help="PLM names (default: all)")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Fast mode: B=100, single seed, for dev iteration")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Registered PLMs:")
        for name, plm in PLMS.items():
            has_cb = "✓" if emb_path(name, "cb513").exists() else "✗"
            has_sc = "✓" if emb_path(name, "scope_5k").exists() else "✗"
            has_cz = "✓" if emb_path(name, "chezod").exists() else "✗"
            print(f"  {name:<20} {plm.display:<22} dim={plm.dim:>4}  "
                  f"CB513={has_cb} SCOPe={has_sc} CheZOD={has_cz}")
        return

    plm_names = args.plm or list(PLMS.keys())
    for name in plm_names:
        if name not in PLMS:
            print(f"Unknown PLM: {name}. Use --list to see options.")
            sys.exit(1)

    if args.smoke_test:
        global BOOTSTRAP_N, SEEDS
        BOOTSTRAP_N = 100
        SEEDS = [42]
        print("SMOKE TEST MODE: B=100, 1 seed")

    t0 = time.time()

    if not args.benchmark_only:
        extract_all(plm_names)

    if not args.extract_only:
        all_results = {}
        for plm_name in plm_names:
            plm = PLMS[plm_name]
            print(f"\n{'='*60}")
            print(f"Benchmarking: {plm.display}")
            print(f"{'='*60}")
            all_results[plm_name] = run_benchmark_suite(plm_name, bootstrap_n=BOOTSTRAP_N)

        # Save per-PLM results (never overwrite other PLMs)
        for plm_name, r in all_results.items():
            per_plm_path = RESULTS_DIR / f"exp46_{plm_name}.json"
            with open(per_plm_path, "w") as f:
                json.dump(r, f, indent=2, default=float)
            print(f"  Saved: {per_plm_path}")

        # Merge with existing combined results
        combined_path = RESULTS_DIR / "exp46_multi_plm_results.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if combined_path.exists():
            with open(combined_path) as f:
                existing = json.load(f)
        existing.update(all_results)
        with open(combined_path, "w") as f:
            json.dump(existing, f, indent=2, default=float)
        print(f"  Combined: {combined_path} ({len(existing)} PLMs)")

        print_table(all_results)

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
