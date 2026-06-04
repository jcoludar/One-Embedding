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


# ---------------------------------------------------------------------------
# Regression guard: ProtT5 must NEVER silently truncate (it is a T5 encoder
# with relative position bias — no architectural length limit). The old
# `truncation=True, max_length=512` BERT boilerplate chopped every protein
# >511 aa to 512 rows while still slicing [:seq_len], returning short, wrong
# embeddings with no error. See sessions/2026-06-04 + SE tools/embeddings/prott5.py.
# ---------------------------------------------------------------------------

class _FakeProtT5Tokenizer:
    """Faithful stand-in for the ProtT5 HF tokenizer.

    Models ProtT5 tokenization: space-separated residues → one token each, plus
    a trailing EOS token. Honors HF ``truncation``/``max_length`` EXACTLY so the
    test detects any silent length cap. Records the last call's kwargs.
    """

    last_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, batch_seqs, padding=True, truncation=False,
                 max_length=None, return_tensors=None):
        import torch
        type(self).last_kwargs = {"truncation": truncation, "max_length": max_length}
        tok_lens = [len(s.split()) + 1 for s in batch_seqs]  # +1 EOS
        if truncation and max_length is not None:
            tok_lens = [min(n, max_length) for n in tok_lens]
        width = max(tok_lens)
        input_ids = torch.zeros((len(batch_seqs), width), dtype=torch.long)
        attn = torch.zeros((len(batch_seqs), width), dtype=torch.long)
        for i, n in enumerate(tok_lens):
            input_ids[i, :n] = 1
            attn[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


class _FakeT5EncoderModel:
    class _Config:
        d_model = 8

    config = _Config()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        import torch
        b, t = input_ids.shape
        hs = torch.zeros((b, t, self.config.d_model), dtype=torch.float32)
        out = type("_Out", (), {})()
        out.last_hidden_state = hs
        return out


@pytest.fixture
def patch_prott5(monkeypatch):
    # Patch `.from_pretrained` on the *real* classes rather than swapping the
    # module attribute: the extractor does an in-function `from transformers
    # import AutoTokenizer, T5EncoderModel`, and the transformers lazy loader
    # re-resolves that to the real classes regardless of a module-level setattr.
    # Patching the constructor is import-resolution-proof and never loads weights.
    from transformers import AutoTokenizer, T5EncoderModel
    _FakeProtT5Tokenizer.last_kwargs = None
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        lambda *a, **k: _FakeProtT5Tokenizer())
    monkeypatch.setattr(T5EncoderModel, "from_pretrained",
                        lambda *a, **k: _FakeT5EncoderModel())


class TestProtT5NoTruncation:
    def test_long_protein_not_truncated_to_512(self, patch_prott5):
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
        out = extract_prot_t5_embeddings({"p600": "A" * 600}, batch_size=1)
        assert out["p600"].shape[0] == 600, (
            f"expected full 600 residues, got {out['p600'].shape[0]} — silent "
            "512-truncation regression (ProtT5 has no architectural length limit)"
        )

    def test_over_max_residues_skipped_not_truncated(self, patch_prott5):
        from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
        out = extract_prot_t5_embeddings({"short": "A" * 100, "huge": "A" * 3000}, batch_size=2)
        assert out["short"].shape[0] == 100
        assert "huge" not in out, "over-max proteins must be SKIPPED loudly, not truncated"
