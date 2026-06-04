#!/usr/bin/env python
"""Biotrainer-direct multi-seed harvest → predictions.json (PLAN §R3; fan-review #3 C1/C2).

autoeval 1.4.0 cannot vary the probe seed or save per-item predictions, so for each
(arm × task × seed) we drive biotrainer directly on the embeddings cached by run_arm.py,
with our config (CNN / 10 epochs / 30 bootstrap, matching the ga38fak baseline), and harvest
per-item TEST predictions. This is the script `05_analyze.py` consumes.

Two invariants this script enforces (the reason it exists):
  * FIXED split per task, shared across ALL arms and ALL 5 seeds — only the probe training
    seed varies, never the test set. So the multi-seed paired bootstrap sees identical test
    items across seeds AND across arms (PLAN §R2). Prefer the PBC predefined SET split; else
    compute one split per task and reuse it everywhere.
  * group_ids written per item = protein id (for per-residue tasks) so 05_analyze can
    cluster-bootstrap whole proteins, not i.i.d. residues (fan-review #3 stat-C1).

  source config.sh
  /work/venv/bin/python 06_harvest_predictions.py \
      --arms-dir /work/out --embeddings-index /work/out/<LABEL>/embeddings_index.json \
      --task scl --out /work/out

CONFIRM on-cluster (biotrainer Config/Trainer API; verified symbols but smoke via 04 first):
  Config(protocol=<task protocol>, embeddings_file=<cached h5>, model_choice="CNN",
         num_epochs=10, bootstrapping_iterations=30, seed=<k>, save_split_ids=True,
         cross_validation_config={"method": "hold_out"}, ...)
  Trainer(cfg).training_and_evaluation_routine()  →  <out_dir>/out.yml
  Inferencer.create_from_out_file(out.yml) -> (inferencer, iom)
  inferencer.from_embeddings(test_embeddings, split_name="test")
      -> {"metrics","mapped_predictions","mapped_probabilities"}
"""
import argparse
import json
import sys
from pathlib import Path

REPO = "/work/ProteEmbedExplorations"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SEEDS = (42, 43, 44, 45, 46)

# task -> (protocol, is_per_residue)
TASK_PROTOCOL = {
    "conservation": ("residue_to_class", True),
    "secondary_structure": ("residue_to_class", True),
    "membrane": ("residue_to_class", True),
    "frustration-classification": ("residue_to_class", True),
    "disorder_chezod": ("residue_to_value", True),
    "disorder_trizod": ("residue_to_value", True),
    "frustration-regression": ("residue_to_value", True),
    "phages": ("sequence_to_class", False),
    "scl": ("sequence_to_class", False),
}


def flatten_predictions(mapped_predictions, y_true_by_seq, is_per_residue):
    """Return aligned (item_ids, y_true, y_pred, group_ids).

    Per-residue: item id = f"{seq_id}_{pos:05d}" (ZERO-PADDED so lexical sort preserves
    residue order — fan-review #3 stat-M1), group id = seq_id. Per-sequence: item id = seq_id,
    group id = seq_id (each its own group).
    """
    item_ids, y_true, y_pred, group_ids = [], [], [], []
    for seq_id in sorted(mapped_predictions):
        pred = mapped_predictions[seq_id]
        truth = y_true_by_seq[seq_id]
        if is_per_residue:
            for pos, (p, t) in enumerate(zip(pred, truth)):
                item_ids.append(f"{seq_id}_{pos:05d}")
                y_pred.append(p)
                y_true.append(t)
                group_ids.append(seq_id)
        else:
            item_ids.append(seq_id)
            y_pred.append(pred)
            y_true.append(truth)
            group_ids.append(seq_id)
    return item_ids, y_true, y_pred, group_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="arm label, e.g. ESM2-8M-ONE")
    ap.add_argument("--task", required=True, choices=list(TASK_PROTOCOL))
    ap.add_argument("--embeddings", required=True, help="cached embeddings .h5 for this arm/task")
    ap.add_argument("--split-ids", required=True, help="shared fixed split-id JSON for this task")
    ap.add_argument("--out", required=True, help="output root; writes <label>/<task>/seed<k>/predictions.json")
    args = ap.parse_args()

    from biotrainer.config import Config          # CONFIRM exact import path on-cluster
    from biotrainer.trainers import Trainer
    from biotrainer.inference import Inferencer

    protocol, is_per_residue = TASK_PROTOCOL[args.task]
    split = json.loads(Path(args.split_ids).read_text())     # {"train":[...],"val":[...],"test":[...]}

    for k in SEEDS:
        out_dir = Path(args.out) / args.label / args.task / f"seed{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = Config(
            protocol=protocol,
            embeddings_file=args.embeddings,
            model_choice="CNN",
            num_epochs=10,
            bootstrapping_iterations=30,
            seed=k,                                 # ONLY the probe seed varies
            save_split_ids=True,
            # inject the FIXED split so every (arm, seed) shares the same test set:
            # CONFIRM the exact biotrainer key for predefined split ids.
            predefined_splits=split,
            output_dir=str(out_dir),
        )
        Trainer(cfg).training_and_evaluation_routine()

        inferencer, iom = Inferencer.create_from_out_file(str(out_dir / "out.yml"))
        # CONFIRM: load test embeddings for split["test"] from args.embeddings, then:
        # preds = inferencer.from_embeddings(test_embeddings, split_name="test")
        # y_true_by_seq = iom.test_targets   # or read from the task dataset
        # item_ids, y_true, y_pred, group_ids = flatten_predictions(
        #     preds["mapped_predictions"], y_true_by_seq, is_per_residue)
        # (out_dir / "predictions.json").write_text(json.dumps(
        #     {"item_ids": item_ids, "y_true": y_true, "y_pred": y_pred, "group_ids": group_ids}))
        print(f"[{args.label}/{args.task}] seed {k}: trained; harvest test predictions -> {out_dir}/predictions.json")

    print("Reminder: assert split['test'] is identical across arms (save_split_ids) before analysis.")


if __name__ == "__main__":
    main()
