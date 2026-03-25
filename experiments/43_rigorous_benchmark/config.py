"""Central configuration for the rigorous benchmark framework."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Raw embedding paths
RAW_EMBEDDINGS = {
    "prot_t5": DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5",
    "prot_t5_cb513": DATA / "residue_embeddings" / "prot_t5_xl_cb513.h5",
    "prot_t5_chezod": DATA / "residue_embeddings" / "prot_t5_xl_chezod.h5",
    "prot_t5_trizod": DATA / "residue_embeddings" / "prot_t5_xl_trizod.h5",
}

# Compressed embedding paths (may not exist yet — script compresses on-the-fly)
COMP_EMBEDDINGS = {
    "prot_t5_768d_cb513": DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "cb513.one.h5",
    "prot_t5_768d_chezod": DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "chezod.one.h5",
    "prot_t5_768d_scope": DATA / "benchmark_suite" / "compressed" / "prot_t5_768d" / "scope_5k.one.h5",
}

# Split paths
SPLITS = {
    "cb513": DATA / "benchmark_suite" / "splits" / "cb513_80_20.json",
    "chezod": DATA / "benchmark_suite" / "splits" / "chezod_seth.json",
    "scope_5k": DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json",
}

# Label paths
LABELS = {
    "cb513_csv": DATA / "per_residue_benchmarks" / "CB513.csv",
    "chezod_data_dir": DATA / "per_residue_benchmarks",  # Contains SETH/ subdir
    "tmbed_cv00": DATA / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta",
}

# Metadata
METADATA = {
    "scope_5k": DATA / "proteins" / "metadata_5k.csv",
}

# Results output
RESULTS_DIR = DATA / "benchmarks" / "rigorous_v1"

# Golden rule thresholds
SEEDS = [42, 123, 456]
BOOTSTRAP_N = 10_000
CV_FOLDS = 3
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CROSS_CHECK_WARN_PP = 3.0
CROSS_CHECK_BLOCK_PP = 5.0
