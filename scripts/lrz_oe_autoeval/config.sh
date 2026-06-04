#!/usr/bin/env bash
# Shared paths/config for the OE×autoeval LRZ sweep. Source this from the *.sh scripts.
# CONFIRM the paths marked CONFIRM on first login (recon'd 2026-06-03 but verify before use).
set -euo pipefail

# DSS workspace (recon: 707 GB free; keep setgid group pr63ci-dss-0004; NOT home).
export WORK=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz

# Existing pyxis image (recon: lives under taxembed_lrz). CONFIRM exact path on login.
export IMAGE=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/taxembed_lrz/pytorch-2.4.0-cuda12.1.sqsh

# Existing HF cache with all 6 models already staged (recon: plm_choice_lrz/data/hf_cache).
export HF_CACHE=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/plm_choice_lrz/data/hf_cache

# In-container python (3.11.9 in the pytorch-2.4.0 image). NOT a login-node interpreter.
export CONTAINER_PY=/opt/conda/bin/python3.11

# Biotrainer dataset cache (env var name UNVERIFIED — confirm at G3; must be the same physical
# dir pre-staged on the login node AND mounted into the container so it is read offline).
export BIOTRAINER_CACHE=$WORK/biotrainer_cache
