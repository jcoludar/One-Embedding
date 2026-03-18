import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py
import tempfile


class TestEncode:
    def test_h5_to_oemb(self):
        from src.one_embedding import encode
        with tempfile.TemporaryDirectory() as d:
            raw_path = Path(d) / "raw.h5"
            with h5py.File(raw_path, "w") as f:
                f.create_dataset("prot_a", data=np.random.randn(80, 1024).astype(np.float32))
                f.create_dataset("prot_b", data=np.random.randn(120, 1024).astype(np.float32))
            out_path = Path(d) / "output.oemb"
            encode(str(raw_path), str(out_path))
            assert out_path.exists()
            from src.one_embedding.io import read_oemb_batch
            data = read_oemb_batch(str(out_path))
            assert "prot_a" in data
            assert data["prot_a"]["per_residue"].shape[1] == 512
            assert data["prot_a"]["protein_vec"].shape == (2048,)


class TestDecode:
    def test_oemb_to_arrays(self):
        from src.one_embedding import decode
        from src.one_embedding.io import write_oemb
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.oemb"
            write_oemb(str(path), {
                "per_residue": np.random.randn(50, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
                "sequence": "M" * 50,
            })
            data = decode(str(path))
            assert data["per_residue"].shape == (50, 512)

    def test_decode_batch(self):
        from src.one_embedding import decode
        from src.one_embedding.io import write_oemb_batch
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "batch.oemb"
            write_oemb_batch(str(path), {
                "a": {"per_residue": np.random.randn(30, 512).astype(np.float16),
                      "protein_vec": np.random.randn(2048).astype(np.float16)},
                "b": {"per_residue": np.random.randn(40, 512).astype(np.float16),
                      "protein_vec": np.random.randn(2048).astype(np.float16)},
            })
            data = decode(str(path))
            assert len(data) == 2


class TestVersion:
    def test_version_exists(self):
        from src.one_embedding import __version__
        assert __version__ == "0.1.0"
