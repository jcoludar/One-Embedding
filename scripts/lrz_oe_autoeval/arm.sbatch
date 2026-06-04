#!/bin/bash
# One OE×autoeval arm. Submitted by submit_all.sh, which sets --partition/--qos on the CLI
# (CLI overrides #SBATCH) and exports EMBEDDER/MODE/NATIVE_D/PRECISION/LABEL via --export.
# pyxis container is declared via #SBATCH directives (the proven pattern on this cluster).
#
# Paths below are LITERAL (recon 2026-06-03) — #SBATCH lines are parsed before the shell runs,
# so they cannot use variables. CONFIRM $IMAGE / $HF_CACHE match config.sh before submitting.
#
#SBATCH --job-name=oe_arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz/logs/%x_%j.out
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz/logs/%x_%j.err
#SBATCH --container-image=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh
#SBATCH --container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz:/work:rw,/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/plm_choice_lrz/data/hf_cache:/hf_cache:ro
#SBATCH --container-workdir=/work
set -euo pipefail

export MKL_THREADING_LAYER=GNU          # load-bearing on this image (MKL vs libgomp abort)
export HF_HOME=/hf_cache                 # the REAL model cache, mounted read-only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BIOTRAINER_CACHE=/work/biotrainer_cache   # pre-staged PBC datasets (see 02_prestage_pbc.py)

source /work/venv/bin/activate
nvidia-smi -L || true

python /work/ProteEmbedExplorations/scripts/lrz_oe_autoeval/run_arm.py \
  --embedder "$EMBEDDER" --mode "$MODE" --native-d "$NATIVE_D" \
  --precision "$PRECISION" --label "$LABEL" \
  --reference /work/reference_2000.fasta --out "/work/out/$LABEL"
