#!/usr/bin/env python
"""G2: confirm the biotrainer-direct path can (a) train a probe with a chosen seed on
PRECOMPUTED embeddings, (b) save split ids, and (c) harvest per-item TEST predictions.

autoeval 1.4.0 hardcodes seed:42/LogReg and never sets save_split_ids, so multi-seed +
per-item predictions (PLAN §R3) must go through biotrainer directly. This script proves the
mechanics on ONE small PBC task before the sweep. Run on the login node (CPU is fine).

  source config.sh
  /work/venv/bin/python 04_probe_g2.py --task scl --embeddings /work/out/ESM2-8M-ONE/<task>/embeddings.h5

CONFIRM on-cluster (the exact biotrainer config keys + Inferencer API are verified from v1.4.0
source but the precomputed-embeddings wiring should be smoke-tested here):
  * Config(protocol=..., embeddings_file=<h5>, model_choice="CNN", num_epochs=10,
    bootstrapping_iterations=30, seed=k, save_split_ids=True, ...).
  * Inferencer.create_from_out_file(out.yml) -> (inferencer, iom); then
    inferencer.from_embeddings(test_embeddings, split_name="test")
       -> {'metrics','mapped_predictions','mapped_probabilities'}.
"""
import argparse
import sys
from pathlib import Path

REPO = "/work/ProteEmbedExplorations"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="a PBC task name, e.g. scl")
    ap.add_argument("--embeddings", required=True, help="precomputed embeddings .h5")
    ap.add_argument("--workdir", default="/work/g2_probe")
    args = ap.parse_args()

    from biotrainer.protocols import Protocol  # noqa: F401  (CONFIRM symbol)
    from biotrainer.inference import Inferencer

    results = {}
    for seed in (42, 43):  # two seeds: prove the seed actually varies the trained probe
        out_dir = Path(args.workdir) / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # CONFIRM: assemble + run a biotrainer Config programmatically on the precomputed
        # embeddings with save_split_ids=True. (Left as the explicit on-cluster step — the
        # config schema is verified but the runner entrypoint should be confirmed live.)
        # from biotrainer.config import Config; from biotrainer.trainers import Trainer
        # cfg = Config(... embeddings_file=args.embeddings, seed=seed, save_split_ids=True,
        #              model_choice="CNN", num_epochs=10, bootstrapping_iterations=30, ...)
        # Trainer(cfg).training_and_evaluation_routine()
        out_yml = out_dir / "out.yml"
        assert out_yml.exists(), f"no out.yml at {out_yml} — biotrainer run did not complete"

        inferencer, iom = Inferencer.create_from_out_file(str(out_yml))
        # Re-run inference on the TEST split embeddings to recover per-item predictions.
        # test_embeddings = load_h5_subset(args.embeddings, split_ids=iom.test_split_ids)
        # preds = inferencer.from_embeddings(test_embeddings, split_name="test")
        # results[seed] = preds["mapped_predictions"]
        print(f"seed {seed}: out.yml present; harvest via from_embeddings(split_name='test')")

    print("G2 OK if the two seeds produced DIFFERENT trained probes and per-item predictions "
          "were recoverable. If identical, seed is not wired — escalate before the sweep.")


if __name__ == "__main__":
    main()
