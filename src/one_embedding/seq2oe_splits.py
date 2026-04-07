"""CATH-level cluster splits for Seq2OE experiments.

Loads the CATH20 labeled FASTA (headers of the form `>{pid}|{C}.{A}.{T}.{H}`)
and produces whole-cluster holdout splits at the Homologous-Superfamily (H)
or Topology (T) level with per-Class greedy stratification.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def parse_cath_fasta(path: Path | str) -> dict[str, dict]:
    """Parse a CATH-labeled FASTA.

    Expected header format: `>{pid}|{C}.{A}.{T}.{H}` where C/A/T/H are CATH
    Class / Architecture / Topology / Homologous-Superfamily codes, e.g.
    `>12asA00|3.30.930.10`.

    Returns a dict mapping protein id to:
        {
            "seq": str,
            "C": int,          # class integer
            "A": int,          # architecture integer
            "T": str,          # topology dotted code (e.g. "3.30.930")
            "H": str,          # homologous-superfamily dotted code (full)
        }

    Raises ValueError on malformed headers or codes.
    """
    path = Path(path)
    meta: dict[str, dict] = {}
    current_id: str | None = None
    current_info: dict | None = None
    seq_lines: list[str] = []

    def flush():
        if current_id is not None:
            current_info["seq"] = "".join(seq_lines)
            meta[current_id] = current_info

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                if "|" not in header:
                    raise ValueError(f"Header has no CATH code: {header!r}")
                pid, code = header.split("|", 1)
                parts = code.split(".")
                if len(parts) != 4:
                    raise ValueError(
                        f"CATH code {code!r} expected 4 dot-separated fields"
                    )
                try:
                    c_int = int(parts[0])
                    a_int = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"Non-integer C/A in {code!r}") from e
                current_id = pid
                current_info = {
                    "C": c_int,
                    "A": a_int,
                    "T": ".".join(parts[:3]),
                    "H": code,
                }
                seq_lines = []
            else:
                seq_lines.append(line)
        flush()

    return meta


# Stubs to be implemented in Tasks 2 and 3
def cath_cluster_split(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 2")

def save_split(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 3")

def load_split(*args, **kwargs):
    raise NotImplementedError("Implemented in Task 3")
