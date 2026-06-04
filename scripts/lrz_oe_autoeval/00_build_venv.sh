#!/usr/bin/env bash
# G1: Build the biotrainer+OE venv INSIDE the container, then verify the autoeval API.
# Run via srun into the pyxis image (NOT on the login node — it has no python3.11/conda).
#
#   source config.sh
#   srun --partition=lrz-v100x2 --qos=gpu --gres=gpu:1 --time=00:30:00 \
#        --container-image="$IMAGE" --container-mounts="$WORK:/work:rw" --container-workdir=/work \
#        bash /work/scripts/00_build_venv.sh
#
# Prereq: 01_deploy_repo.sh has put the PEE checkout at $WORK/ProteEmbedExplorations.
set -euo pipefail
export MKL_THREADING_LAYER=GNU

# Build against the container's python with system-site-packages (inherit torch/CUDA).
# ISOLATED venv (NOT --system-site-packages): biotrainer 1.4.0 pulls torch 2.12 + transformers
# 5.9, which clash with the container's torch-2.4-built torchvision if system-site-packages is on
# (torchvision::nms missing -> T5EncoderModel import fails). Isolated = transformers sees no
# torchvision and skips it. torch 2.12 brings its own bundled CUDA libs. Wheels cached on DSS so
# rebuilds don't re-download.
export PIP_CACHE_DIR=/work/pip_cache
rm -rf /work/venv
/opt/conda/bin/python3.11 -m venv /work/venv
source /work/venv/bin/activate
pip install --upgrade pip
# Install torch for CUDA 12.x FIRST: biotrainer's default pulls torch+cu130 (CUDA 13), but the
# node driver is CUDA 12.2 -> "driver too old" (major-version mismatch). cu126 (CUDA 12.6) runs
# on a 12.2 driver via CUDA minor-version compatibility (driver r535 >= the r525 12.x baseline).
# Installing it first means biotrainer sees torch>=2.10 satisfied and won't pull cu130.
pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu126
pip install "biotrainer==1.4.0"
# NOTE: do NOT `pip install -e` the PEE repo — its pyproject pins requires-python>=3.12 but the
# container is py3.11. The codec only needs numpy/scipy/h5py and is imported via sys.path
# (run_arm.py / 05 / 06 already do sys.path.insert(0, "/work/ProteEmbedExplorations")).

# Match the ga38fak baseline probe: biotrainer 1.4.0's PBC config bank defaults to model_choice
# "LogReg"; the baseline (and the paper) used CNN. Patch the installed config bank to CNN.
python - <<'PY'
import pathlib
p = pathlib.Path("/work/venv/lib/python3.11/site-packages/biotrainer/autoeval/pbc/pbc_config_bank.py")
t = p.read_text()
if '"model_choice": "LogReg"' in t:
    t = t.replace('"model_choice": "LogReg"', '"model_choice": "CNN"')   # CNN for residue tasks
    # scl is sequence_to_class -> CNN is invalid there; use FNN (matches the ga38fak baseline).
    t = t.replace('"protocol": "sequence_to_class",',
                  '"protocol": "sequence_to_class",\n                "model_choice": "FNN",')
    p.write_text(t)
    print("patched PBC config bank -> CNN (residue tasks) + FNN (scl)")
else:
    print("PBC config bank already patched / layout changed — verify manually")
PY

# G1 gate: biotrainer API + the codec/oe_autoeval imports + torch/CUDA actually work on the node.
python - <<'PY'
import sys, inspect
sys.path.insert(0, "/work/ProteEmbedExplorations")

from biotrainer.autoeval import autoeval_pipeline
sig = inspect.signature(autoeval_pipeline)
need = ("embedder_name", "framework",
        "custom_embedding_function_per_residue", "custom_embedding_function_per_sequence")
missing = [k for k in need if k not in sig.parameters]
assert not missing, f"G1 FAIL — autoeval_pipeline missing kwargs: {missing}"

from biotrainer.embedders import get_embedding_service
gs = inspect.signature(get_embedding_service)
for k in ("embedder_name", "custom_tokenizer_config", "use_half_precision", "device"):
    assert k in gs.parameters, f"G1 FAIL — get_embedding_service missing {k}"

# codec + oe_autoeval import under the new env (numpy 2.x / scipy 1.17)
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.oe_autoeval import driver, wrappers, retention
assert driver.d_out_eff_for(1280) == 896

import numpy as np, torch
print("G1 OK — biotrainer autoeval + get_embedding_service kwargs present")
print("G1 OK — codec/oe_autoeval import; numpy", np.__version__)
print("G1 torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
assert torch.cuda.is_available(), "G1 FAIL — torch cannot see the GPU (CUDA-13 wheel vs node driver?)"
print("G1 PASS")
PY

pip freeze > /work/venv.freeze.txt
echo "venv built + frozen at /work/venv ; freeze -> /work/venv.freeze.txt"
