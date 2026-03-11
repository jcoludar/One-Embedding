#!/bin/bash
#SBATCH --job-name=train_comp
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# LRZ compressor training for large models / datasets
#
# Usage:
#   sbatch slurm/train_compressor.sh esm2_650m_5k 8 200
#   sbatch slurm/train_compressor.sh esm2_3b_5k 8 300

set -euo pipefail

H5_TAG=${1:-esm2_650m_5k}
K=${2:-8}
EPOCHS=${3:-200}
RUN_NAME="attnpool_${H5_TAG}_K${K}_e${EPOCHS}"

echo "=== Compressor training ==="
echo "H5 tag: $H5_TAG"
echo "K: $K, Epochs: $EPOCHS"
echo "Run name: $RUN_NAME"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Activate environment
module load python/3.12 cuda/12.4 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || source venv/bin/activate

python -c "
import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')

H5_TAG = '$H5_TAG'
K = $K
EPOCHS = $EPOCHS
RUN_NAME = '$RUN_NAME'

import torch
from src.utils.device import get_device
from src.utils.h5_store import load_residue_embeddings
from src.extraction.data_loader import read_fasta, load_metadata_csv
from src.compressors.attention_pool import AttentionPoolCompressor
from src.training.trainer import train_compressor
from src.evaluation.benchmark_suite import run_benchmark_suite, save_benchmark_results

device = get_device()
print(f'Device: {device}')

DATA_DIR = Path('data')

# Find matching files
h5_path = list(DATA_DIR.glob(f'residue_embeddings/*{H5_TAG}*.h5'))
if not h5_path:
    print(f'ERROR: No H5 file matching {H5_TAG}')
    sys.exit(1)
h5_path = h5_path[0]
print(f'H5: {h5_path}')

# Find matching FASTA and metadata
fasta_files = list(DATA_DIR.glob('proteins/*.fasta'))
meta_files = list(DATA_DIR.glob('proteins/metadata*.csv'))

# Load data
embeddings = load_residue_embeddings(h5_path)
embed_dim = next(iter(embeddings.values())).shape[-1]
print(f'{len(embeddings)} proteins, embed_dim={embed_dim}')

# Find metadata with matching protein count
metadata = None
for mf in meta_files:
    meta = load_metadata_csv(mf)
    meta_ids = {m['id'] for m in meta}
    emb_ids = set(embeddings.keys())
    if meta_ids & emb_ids:
        metadata = meta
        print(f'Metadata: {mf} ({len(meta)} entries)')
        break

if metadata is None:
    print('ERROR: No matching metadata found')
    sys.exit(1)

# Find FASTA
sequences = {}
for ff in fasta_files:
    seqs = read_fasta(ff)
    if set(seqs.keys()) & set(embeddings.keys()):
        sequences = seqs
        print(f'FASTA: {ff}')
        break

# Train
model = AttentionPoolCompressor(embed_dim, latent_dim=128, n_tokens=K, n_heads=4)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params:,} parameters')

start = time.time()
history = train_compressor(
    model=model,
    embeddings=embeddings,
    sequences=sequences,
    epochs=EPOCHS,
    batch_size=8,
    lr=1e-3,
    recon_weight=1.0,
    masked_weight=0.1,
    contrastive_weight=0.1,
    device=device,
    checkpoint_dir=DATA_DIR / 'checkpoints' / RUN_NAME,
    log_every=25,
)
elapsed = time.time() - start
print(f'Training done in {elapsed:.0f}s (best epoch: {history[\"best_epoch\"]})')

# Benchmark
best_path = DATA_DIR / 'checkpoints' / RUN_NAME / 'best_model.pt'
if best_path.exists():
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

results = run_benchmark_suite(model, embeddings, metadata, name=RUN_NAME, device=device)
results['training_time_s'] = elapsed
results['embed_dim'] = embed_dim
results['K'] = K
results['epochs'] = EPOCHS
results['n_proteins'] = len(embeddings)
results['best_epoch'] = history['best_epoch']

out_path = DATA_DIR / 'benchmarks' / f'{RUN_NAME}.json'
save_benchmark_results(results, out_path)
print(f'Results saved to {out_path}')

ret = results.get('retrieval_family', {}).get('precision@1', 0)
cls = results.get('classification_family', {}).get('accuracy_mean', 0)
cos = results.get('reconstruction', {}).get('cosine_sim', 0)
print(f'Ret-Fam@1={ret:.3f}, Cls-Fam={cls:.3f}, Recon-Cos={cos:.3f}')
"

echo "=== Done ==="
date
