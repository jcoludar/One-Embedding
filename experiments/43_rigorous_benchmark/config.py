"""Central configuration for the rigorous benchmark framework."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Raw embedding paths
RAW_EMBEDDINGS = {

    "prot_t5":{
        "scope": DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5",
        "cath20": DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5",
        "cb513": DATA / "residue_embeddings" / "prot_t5_xl_cb513.h5",
        "ts115": DATA / "residue_embeddings" / "prot_t5_xl_ts115.h5",
        "casp12": DATA / "residue_embeddings" / "prot_t5_xl_casp12.h5",
        "chezod": DATA / "residue_embeddings" / "prot_t5_xl_chezod.h5",
        "trizod": DATA / "residue_embeddings" / "prot_t5_xl_trizod.h5",
        "deeploc": DATA / "residue_embeddings" / "prot_t5_xl_deeploc.h5",
    } 
    
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
    "ts115": DATA / "benchmark_suite" / "splits" / "ts115_80_20.json",
    "casp12": DATA / "benchmark_suite" / "splits" / "casp12_80_20.json",
    "chezod": DATA / "benchmark_suite" / "splits" / "chezod_seth.json",
    "trizod": DATA / "benchmark_suite" / "splits" / "trizod_split_strict.json",
    "scope": DATA / "benchmark_suite" / "splits" / "esm2_650m_5k_split.json",
}

# Label paths
LABELS = {
    "cb513": DATA / "per_residue_benchmarks" / "CB513.csv",
    "ts115": DATA / "per_residue_benchmarks" / "ts115.csv",
    "casp12": DATA / "per_residue_benchmarks" / "casp12.csv",
    "chezod_data_dir": DATA / "per_residue_benchmarks" / "SETH",
    "trizod_data_dir": DATA / "per_residue_benchmarks" / "trizod",
    "tmbed_cv00": DATA / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta",
}

# Metadata
METADATA = {
    "scope": DATA / "proteins" / "scope_all_metadata.csv",
    "cath20": DATA / "proteins" / "cath20_metadata.csv",
    "cb513": DATA / "per_residue_benchmarks"  / "metadata_cb513.csv",
    "ts115": DATA / "per_residue_benchmarks" / "metadata_ts115.csv",
    "casp12": DATA / "per_residue_benchmarks" / "metadata_casp12.csv",
    "chezod": DATA / "per_residue_benchmarks"  / "metadata_chezod.csv",
    "trizod": DATA / "per_residue_benchmarks"  / "metadata_trizod.csv",
    "deeploc":DATA / "per_residue_benchmarks"  / "deeploc1_metadata.csv"
}
#buckets
BUCKETS = {
    "scope": DATA / "benchmark_suite" / "buckets" / "scope5k_buckets.json",
    "cath20": DATA / "benchmark_suite" / "buckets" / "cath20_buckets.json",
    "cb513": DATA / "benchmark_suite" / "buckets" / "cb513_buckets.json",
    "ts115": DATA / "benchmark_suite" / "buckets" / "ts115_buckets.json",
    "casp12": DATA / "benchmark_suite" / "buckets" / "casp12_buckets.json",
    "chezod": DATA / "benchmark_suite" / "buckets" / "chezod_buckets.json",
    "trizod": DATA / "benchmark_suite" / "buckets" / "chezod_buckets.json",
    "deeploc": DATA / "benchmark_suite" / "buckets" / "deeploc1_buckets.json",
}
ENCODED_FOLDER = DATA / "encoded_h5s"
CHECKPOINT_FOLDER = DATA / "checkpoints" / "nn_weights"
# Results output
RESULTS_DIR = DATA / "benchmarks" / "rigorous_v3"

# Golden rule thresholds
SEEDS = [42, 123, 456]
BOOTSTRAP_N = 10_000
CV_FOLDS = 3
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CROSS_CHECK_WARN_PP = 3.0
CROSS_CHECK_BLOCK_PP = 5.0
