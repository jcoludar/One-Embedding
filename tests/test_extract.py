import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFastaParser:
    def test_parse_basic(self):
        from src.one_embedding.extract._base import read_fasta
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">prot_a\nMKTLLIFALG\n>prot_b\nARNDCQ\n")
            f.flush()
            seqs = read_fasta(f.name)
        assert seqs == {"prot_a": "MKTLLIFALG", "prot_b": "ARNDCQ"}

    def test_parse_multiline(self):
        from src.one_embedding.extract._base import read_fasta
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">prot_a\nMKTL\nLIFA\nLG\n")
            f.flush()
            seqs = read_fasta(f.name)
        assert seqs["prot_a"] == "MKTLLIFALG"

    def test_parse_empty_file(self):
        from src.one_embedding.extract._base import read_fasta
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write("")
            f.flush()
            seqs = read_fasta(f.name)
        assert seqs == {}


class TestDispatcher:
    def test_unknown_model(self):
        from src.one_embedding.extract import extract_embeddings
        with pytest.raises(ValueError, match="Unknown model"):
            extract_embeddings("in.fasta", "out.h5", model="nonexistent")

    def test_known_models_listed(self):
        from src.one_embedding.extract import MODELS
        assert "prot_t5" in MODELS
        assert "esm2" in MODELS
