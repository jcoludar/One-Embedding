"""Tests for Seq2OE CATH-level cluster splits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.seq2oe_splits import (
    parse_cath_fasta,
    cath_cluster_split,
    save_split,
    load_split,
)


SAMPLE_FASTA = """>12asA00|3.30.930.10
MKTAYIAKQRQIS
>132lA00|1.10.530.10
XVFGRCELAAAM
>153lA00|1.10.530.10
RTDCYGNVNRIDT
>16pkA02|3.40.50.1260
YFAKVLGNPPRP
>16vpA00|1.10.1290.10
SRMPSPPMPVPP
>1914A00|3.30.720.10
MVLLESEQFL
"""


class TestParseCathFasta:
    def test_parses_ids_and_codes(self, tmp_path):
        f = tmp_path / "cath.fa"
        f.write_text(SAMPLE_FASTA)
        meta = parse_cath_fasta(f)
        assert set(meta.keys()) == {
            "12asA00", "132lA00", "153lA00",
            "16pkA02", "16vpA00", "1914A00",
        }
        assert meta["12asA00"]["seq"] == "MKTAYIAKQRQIS"
        assert meta["12asA00"]["C"] == 3
        assert meta["12asA00"]["A"] == 30
        assert meta["12asA00"]["T"] == "3.30.930"
        assert meta["12asA00"]["H"] == "3.30.930.10"

    def test_rejects_malformed_header(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">noCode\nAAAA\n")
        with pytest.raises(ValueError, match="no CATH code"):
            parse_cath_fasta(f)

    def test_rejects_malformed_code(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">pid|1.2.3\nAAAA\n")  # only 3 levels
        with pytest.raises(ValueError, match="expected 4 dot-separated"):
            parse_cath_fasta(f)
