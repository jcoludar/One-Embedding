#!/usr/bin/env python
"""Introspect the installed biotrainer 1.4.0 API to resolve the CONFIRM markers in
02_prestage_pbc / 03_build_reference / 04_probe_g2 / 06_harvest. Run in-container:

  srun ... bash -lc 'source /work/venv/bin/activate; python /work/.../explore_biotrainer.py'
"""
import inspect
import importlib
import pkgutil


def show(title):
    print("\n=== " + title + " ===")


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except (ValueError, TypeError):
        return "<no signature>"


import biotrainer
print("biotrainer", getattr(biotrainer, "__version__", "?"), "at", biotrainer.__file__)

show("biotrainer.autoeval exports")
import biotrainer.autoeval as ae
print([x for x in dir(ae) if not x.startswith("_")])

show("autoeval submodules")
ae_pkg = importlib.import_module("biotrainer.autoeval")
for m in pkgutil.iter_modules(ae_pkg.__path__):
    print(" ", m.name)

show("autoeval_pipeline signature")
from biotrainer.autoeval import autoeval_pipeline
print(sig(autoeval_pipeline))

show("PBC framework / dataset API")
for name in ("get_framework", "AvailableFramework", "get_unique_framework_sequences"):
    obj = getattr(ae, name, None)
    print(f"{name}: {obj!r}")
    if obj is not None and callable(obj):
        print("   sig:", sig(obj))
try:
    fw = ae.get_framework("PBC")
    print("PBC framework obj:", type(fw))
    print("  members:", [x for x in dir(fw) if not x.startswith("_")])
    for meth in ("download", "setup", "prepare", "get_datasets", "datasets", "tasks", "get_tasks"):
        if hasattr(fw, meth):
            a = getattr(fw, meth)
            print(f"  has {meth}:", sig(a) if callable(a) else repr(a))
except Exception as e:
    print("get_framework('PBC') ->", repr(e))

show("PBC submodule (datasets / config bank)")
try:
    pbc = importlib.import_module("biotrainer.autoeval.pbc")
    print("pbc dir:", [x for x in dir(pbc) if not x.startswith("_")])
    for mm in pkgutil.iter_modules(pbc.__path__):
        print("  submod:", mm.name)
except Exception as e:
    print("pbc import ->", repr(e))

show("dataset cache location hints")
import os
for var in ("BIOTRAINER_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
    print(f"  {var}={os.environ.get(var)}")
# search biotrainer for cache-path helpers
try:
    from biotrainer.utilities import get_device  # noqa
    print("  get_device ok")
except Exception as e:
    print("  utilities ->", repr(e))

show("Config / Trainer / Inferencer (for the harvest)")
try:
    from biotrainer.config import Config
    print("Config:", Config, "sig:", sig(Config))
except Exception as e:
    print("Config ->", repr(e))
try:
    import biotrainer.trainers as tr
    print("trainers exports:", [x for x in dir(tr) if not x.startswith("_")])
except Exception as e:
    print("trainers ->", repr(e))
try:
    from biotrainer.inference import Inferencer
    print("Inferencer.create_from_out_file:", sig(Inferencer.create_from_out_file))
    print("Inferencer methods:", [x for x in dir(Inferencer) if not x.startswith("_")])
except Exception as e:
    print("Inferencer ->", repr(e))

print("\nDONE")
