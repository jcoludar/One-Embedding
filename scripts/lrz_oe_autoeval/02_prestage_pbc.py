#!/usr/bin/env python
"""G3: pre-stage the PBC datasets on the LOGIN node (has internet) + record a content hash.

autoeval 1.4.0 exposes no --download-only, so we trigger the download via the autoeval dataset
machinery directly and stop before any embedding. Run inside the container venv on the login
node (CPU). The resulting cache dir MUST be the same physical path mounted into compute jobs
(see config.sh BIOTRAINER_CACHE) so it is read offline.

  source config.sh
  /work/venv/bin/python 02_prestage_pbc.py --cache "$BIOTRAINER_CACHE"

CONFIRM on-cluster (cannot verify without biotrainer installed):
  * the exact import path that downloads PBC (the loader below is the best-effort target);
  * which env var biotrainer reads for its dataset cache (BIOTRAINER_CACHE assumed).
"""
import argparse
import hashlib
import os
from pathlib import Path


def hash_dir(root: Path) -> str:
    """sha256 over the sorted (relpath, size) listing — stable content fingerprint."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dataset cache dir (== compute-job mount)")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BIOTRAINER_CACHE", str(cache))

    # Best-effort: use autoeval's PBC framework to materialize datasets without embedding.
    # The PBC handler downloads the single unauthenticated TUM-Nextcloud archive on first use.
    from biotrainer.autoeval import get_framework  # CONFIRM exact symbol on-cluster

    framework = get_framework("PBC")
    # Most autoeval frameworks expose a dataset/setup step that downloads without training.
    # CONFIRM the method name; candidates: framework.download(), .setup(), .get_datasets().
    for attr in ("download", "setup", "prepare", "get_datasets"):
        fn = getattr(framework, attr, None)
        if callable(fn):
            print(f"PBC pre-stage via framework.{attr}() …")
            fn()
            break
    else:
        raise SystemExit("Could not find a PBC download/setup method — inspect get_framework('PBC') on-cluster")

    digest = hash_dir(cache)
    (cache.parent / "pbc_dataset.hash").write_text(digest + "\n")
    print(f"PBC staged at {cache}\nG3 content hash: {digest}")


if __name__ == "__main__":
    main()
