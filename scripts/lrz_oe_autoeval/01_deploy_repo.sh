#!/usr/bin/env bash
# Deploy the PEE repo (OE codec + oe_autoeval package) to LRZ. Run from your LOCAL machine.
#   bash scripts/lrz_oe_autoeval/01_deploy_repo.sh
# Requires the LRZ ssh alias `ai` and VPN up.
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_WORK=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/oe_autoeval_lrz

ssh ai "mkdir -p $REMOTE_WORK/logs $REMOTE_WORK/out $REMOTE_WORK/biotrainer_cache"

# rsync the repo — keep it LEAN: only what `pip install -e` + the run scripts need.
# tools/reference is gigabytes of PEbA/TMbed alignment data irrelevant to the codec; exclude it
# (and data/, slides/, figures, notebooks) so we don't push GBs onto DSS.
rsync -av --delete \
  --exclude '.git' --exclude '.venv' \
  --exclude 'data' --exclude 'tools/reference' --exclude 'slides' \
  --exclude 'docs/figures' --exclude 'docs/_audit' --exclude '*.ipynb' \
  --exclude '*.h5' --exclude '*.png' --exclude '*.jpg' \
  --exclude '**/__pycache__' --exclude 'scripts/tmp' \
  "$LOCAL_REPO/" "ai:$REMOTE_WORK/ProteEmbedExplorations/"

# Record the exact codec version used (reproducibility, PLAN §R2 invariant M1).
git -C "$LOCAL_REPO" rev-parse HEAD | ssh ai "cat > $REMOTE_WORK/PEE_SHA.txt"
echo "deployed to ai:$REMOTE_WORK/ProteEmbedExplorations ; SHA recorded."
