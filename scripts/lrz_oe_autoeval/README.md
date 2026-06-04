# OE × autoeval — LRZ AI run scripts

> **AUTHORED, NOT RUN.** These scripts were written from read-only recon (2026-06-03) and the
> §R-revised plan. **Do not execute any of them until Ivan greenlights the cluster**, and then
> only in gate order. Several paths/behaviors are marked `CONFIRM`/`G?` because they cannot be
> verified without biotrainer installed (which is gate G1 itself).

Account `ge94xik2`, project `pr63ci`, LRZ AI (`login.ai.lrz.de`). Design + plan:
`docs/oe_autoeval_lrz/{DESIGN.md,PLAN.md}` (read DESIGN §14 and PLAN §R first).

## Run order (each step is a gate)

| # | Script | Where | Gate |
|---|---|---|---|
| 0 | `01_deploy_repo.sh` | login | rsync PEE → `$WORK/ProteEmbedExplorations`, record SHA |
| 1 | `00_build_venv.sh` | **inside container** (srun) | **G1**: biotrainer 1.4.0 imports + 4 kwargs |
| 2 | `02_prestage_pbc.py` | login (has internet) | **G3**: PBC datasets downloaded, content hash recorded |
| 3 | `03_build_reference.py` | login | reference_2000.fasta built, disjoint-from-test asserted; **G4** L<4 scan |
| 4 | `submit_all.sh --only ESM2-8M:raw` (→ `arm.sbatch`) | compute | **C1**: calibration vs ga38fak 8M (after the §R3 CNN/10/30 harvest, step 7) |
| 5 | `submit_all.sh --only ESM2-8M:oe` | compute | **C2**: OE smoke (per-residue width 320, not packed 40) + produces embeddings for G2 |
| 6 | `04_probe_g2.py` (uses 8M-oe embeddings) | login (CPU) | **G2**: biotrainer-direct seed control + per-item harvest works on real embeddings |
| 7 | `submit_all.sh` | compute | full sweep (14 embedding arms) |
| 8 | `06_harvest_predictions.py` (per arm×task×seed) | compute/CPU | 5-seed CNN probes on cached embeddings, FIXED shared split → `predictions.json` |
| 9 | `05_analyze.py` | login/CPU | cluster-paired retention + tables → `docs/oe_autoeval_lrz/RESULTS.md` |

`run_arm.py` (called by `arm.sbatch`) produces the embeddings + a single-seed autoeval report;
the **5-seed** per-item predictions for the statistics come from `06_harvest_predictions.py`
(biotrainer-direct, since autoeval 1.4.0 can't vary the probe seed).

## Hard rules (from the reviews)

- **Build the venv INSIDE the container** with `/opt/conda/bin/python3.11 -m venv --system-site-packages`.
  The login node has only python3.10 and no conda (recon LRZ-C1).
- **HF_HOME must point at the existing cache** `$HF_CACHE` (all 6 models there). With
  `HF_HUB_OFFLINE=1`, an empty cache = guaranteed failure (recon LRZ-C2).
- **autoeval 1.4.0 cannot vary the probe seed or save per-item predictions** (PBC config bank
  hardcodes seed:42, LogReg, no save_split_ids). Multi-seed + per-item predictions go through
  the **biotrainer-direct** path (PLAN §R3): drive biotrainer per (task×seed) on cached
  embeddings with our config (CNN/10-epoch/30-bootstrap to MATCH the ga38fak baseline, seed:k,
  save_split_ids:true), harvest via `Inferencer.create_from_out_file` + `from_embeddings(...,
  split_name="test")`. `04_probe_g2.py` confirms the mechanics before the sweep.
- **Raw and OE arms must share the identical test split per (task, seed)** (PLAN §R2) — pass a
  pre-computed split-id file into every arm, or assert split-id identity post-hoc.
- **Do NOT force cuDNN-deterministic** for probe training — the 5-seed spread is the intended
  noise floor (PLAN §R8).
- MaxJobsPU=10 on qos=gpu → the sweep auto-queues; that is expected, not an error.
- Sidecar every run (args, PEE SHA, biotrainer ver, GPU model, CUDA/cuDNN, per-seed metrics).
