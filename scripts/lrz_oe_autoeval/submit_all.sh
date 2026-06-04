#!/usr/bin/env bash
# Submit the OE×autoeval arm matrix. Run on the LRZ login node after gates G1–G4 + C1/C2 pass.
#   source config.sh && bash submit_all.sh            # all arms
#   source config.sh && bash submit_all.sh --only ESM2-8M:raw   # one arm (calibration)
# qos=gpu caps concurrent jobs at 10 (MaxJobsPU) — the rest queue; that is expected.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${2:-}"   # used with --only

# arm matrix: LABEL  EMBEDDER  NATIVE_D  PRECISION  PARTITION  MODE
#   precision: small ESM2 fp32; 650M/3B/ProtT5 fp16 (3B fp32 OOMs even on H100-94GB).
#   oe_norp is the RP-vs-noRP control — only meaningful for native_d>896 (650M, ProtT5).
read -r -d '' MATRIX <<'EOF' || true
ESM2-8M       facebook/esm2_t6_8M_UR50D            320   fp32  lrz-v100x2          raw
ESM2-8M-ONE   facebook/esm2_t6_8M_UR50D            320   fp32  lrz-v100x2          oe
ESM2-35M      facebook/esm2_t12_35M_UR50D          480   fp32  lrz-v100x2          raw
ESM2-35M-ONE  facebook/esm2_t12_35M_UR50D          480   fp32  lrz-v100x2          oe
ESM2-150M     facebook/esm2_t30_150M_UR50D         640   fp32  lrz-hgx-a100-80x4   raw
ESM2-150M-ONE facebook/esm2_t30_150M_UR50D         640   fp32  lrz-hgx-a100-80x4   oe
ESM2-650M     facebook/esm2_t33_650M_UR50D         1280  fp16  lrz-hgx-a100-80x4   raw
ESM2-650M-ONE facebook/esm2_t33_650M_UR50D         1280  fp16  lrz-hgx-a100-80x4   oe
ESM2-650M-NORP facebook/esm2_t33_650M_UR50D        1280  fp16  lrz-hgx-a100-80x4   oe_norp
ESM2-3B       facebook/esm2_t36_3B_UR50D           2560  fp16  lrz-hgx-h100-94x4   raw
ESM2-3B-ONE   facebook/esm2_t36_3B_UR50D           2560  fp16  lrz-hgx-h100-94x4   oe
ProtT5        Rostlab/prot_t5_xl_half_uniref50-enc 1024  fp16  lrz-hgx-a100-80x4   raw
ProtT5-ONE    Rostlab/prot_t5_xl_half_uniref50-enc 1024  fp16  lrz-hgx-a100-80x4   oe
ProtT5-NORP   Rostlab/prot_t5_xl_half_uniref50-enc 1024  fp16  lrz-hgx-a100-80x4   oe_norp
EOF

while read -r LABEL EMBEDDER NATIVE_D PRECISION PARTITION MODE; do
  [ -z "${LABEL:-}" ] && continue
  if [ -n "$ONLY" ] && [ "$ONLY" != "$LABEL:$MODE" ]; then continue; fi
  echo "submit $LABEL ($MODE, $PRECISION) -> $PARTITION"
  sbatch --job-name="oe_$LABEL" --partition="$PARTITION" --qos=gpu \
    --export=ALL,EMBEDDER="$EMBEDDER",MODE="$MODE",NATIVE_D="$NATIVE_D",PRECISION="$PRECISION",LABEL="$LABEL" \
    "$HERE/arm.sbatch"
done <<< "$MATRIX"
