#!/bin/bash
#SBATCH --job-name=extract_emb
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# LRZ embedding extraction for large models / datasets
#
# Usage:
#   sbatch slurm/extract_embeddings.sh esm2_3b 5000
#   sbatch slurm/extract_embeddings.sh esm2_650m 50000

set -euo pipefail

MODEL=${1:-esm2_3b}
N_PROTEINS=${2:-5000}

echo "=== Embedding extraction ==="
echo "Model: $MODEL"
echo "N proteins: $N_PROTEINS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Activate environment (adjust to your LRZ setup)
module load python/3.12 cuda/12.4 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || source venv/bin/activate

python -c "
import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')

MODEL = '$MODEL'
N_PROTEINS = $N_PROTEINS

from src.utils.device import get_device
from src.extraction.data_loader import read_fasta, curate_scope_set, write_fasta, save_metadata_csv
from src.extraction.esm_extractor import extract_residue_embeddings
from src.utils.h5_store import save_residue_embeddings

device = get_device()
print(f'Device: {device}')

DATA_DIR = Path('data')

# Model name mapping
model_map = {
    'esm2_3b':   ('esm2_t36_3B_UR50D', 2),
    'esm2_650m': ('esm2_t33_650M_UR50D', 4),
    'esm2_35m':  ('esm2_t12_35M_UR50D', 8),
}
esm_model, batch_size = model_map[MODEL]

# Dataset
tag = f'{MODEL}_{N_PROTEINS // 1000}k' if N_PROTEINS >= 1000 else f'{MODEL}_{N_PROTEINS}'
fasta_path = DATA_DIR / 'proteins' / f'diverse_{tag}.fasta'
meta_path = DATA_DIR / 'proteins' / f'metadata_{tag}.csv'
h5_path = DATA_DIR / 'residue_embeddings' / f'{tag}.h5'

if not fasta_path.exists():
    print(f'Curating {N_PROTEINS} proteins...')
    scope_fasta = DATA_DIR / 'proteins' / 'astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa'
    fasta_dict, metadata = curate_scope_set(scope_fasta, n_proteins=N_PROTEINS, seed=789)
    write_fasta(fasta_dict, fasta_path)
    save_metadata_csv(metadata, meta_path)
    print(f'  Curated {len(fasta_dict)} proteins')
else:
    print(f'FASTA already exists: {fasta_path}')

if not h5_path.exists():
    fasta_dict = read_fasta(fasta_path)
    start = time.time()
    embeddings = extract_residue_embeddings(fasta_dict, model_name=esm_model, batch_size=batch_size, device=device)
    elapsed = time.time() - start
    save_residue_embeddings(embeddings, h5_path)
    print(f'Extraction done in {elapsed:.0f}s ({len(embeddings)} proteins)')
else:
    print(f'H5 already exists: {h5_path}')
"

echo "=== Done ==="
date
