"""Download ProteinGym DMS substitution + clinical splits.

Reference: https://github.com/OATML-Markslab/ProteinGym
The official URLs change occasionally; if the URL below 404s, update from
the README of the upstream repo.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "proteingym"

DMS_ZIP_URL = (
    "https://marks.hms.harvard.edu/proteingym/"
    "ProteinGym_substitutions_DMS.zip"
)
CLINICAL_ZIP_URL = (
    "https://marks.hms.harvard.edu/proteingym/"
    "ProteinGym_clinical_substitutions.zip"
)
REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
    "reference_files/DMS_substitutions.csv"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already present: {dest.name}")
        return
    print(f"  downloading {url} -> {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)


def _unzip(zip_path: Path, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"  already unzipped: {target_dir.name}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"  unzipped {zip_path.name} -> {target_dir.name}")


def main(skip_dms: bool, skip_clinical: bool) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Target dir: {DATA}")

    _download(REFERENCE_URL, DATA / "DMS_substitutions.csv")

    if not skip_dms:
        dms_zip = DATA / "ProteinGym_substitutions_DMS.zip"
        _download(DMS_ZIP_URL, dms_zip)
        _unzip(dms_zip, DATA / "DMS_substitutions")

    if not skip_clinical:
        cli_zip = DATA / "ProteinGym_clinical_substitutions.zip"
        _download(CLINICAL_ZIP_URL, cli_zip)
        _unzip(cli_zip, DATA / "clinical_substitutions")

    print("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dms", action="store_true")
    parser.add_argument("--skip-clinical", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.skip_dms, args.skip_clinical))
