import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py
import tempfile


class TestEncode:
    def test_h5_to_one_h5(self):
        from src.one_embedding import encode
        with tempfile.TemporaryDirectory() as d:
            raw_path = Path(d) / "raw.h5"
            with h5py.File(raw_path, "w") as f:
                f.create_dataset("prot_a", data=np.random.randn(80, 1024).astype(np.float32))
                f.create_dataset("prot_b", data=np.random.randn(120, 1024).astype(np.float32))
            out_path = Path(d) / "output.one.h5"
            encode(str(raw_path), str(out_path))
            assert out_path.exists()
            # Verify .one.h5 format attributes
            with h5py.File(str(out_path), "r") as f:
                assert f.attrs["format"] == "one_embedding"
                assert f.attrs["version"] == "1.0"
                assert f.attrs["d_out"] == 768
                assert f.attrs["source_model"] == "unknown"
            from src.one_embedding.io import read_one_h5_batch
            data = read_one_h5_batch(str(out_path))
            assert "prot_a" in data
            assert data["prot_a"]["per_residue"].shape[1] == 768
            assert data["prot_a"]["protein_vec"].shape == (3072,)

    def test_h5_to_one_h5_custom_d_out(self):
        from src.one_embedding import encode
        with tempfile.TemporaryDirectory() as d:
            raw_path = Path(d) / "raw.h5"
            with h5py.File(raw_path, "w") as f:
                f.create_dataset("prot_a", data=np.random.randn(80, 1024).astype(np.float32))
            out_path = Path(d) / "output.one.h5"
            encode(str(raw_path), str(out_path), d_out=512)
            from src.one_embedding.io import read_one_h5_batch
            data = read_one_h5_batch(str(out_path))
            assert data["prot_a"]["per_residue"].shape[1] == 512
            assert data["prot_a"]["protein_vec"].shape == (2048,)


class TestDecode:
    def test_oemb_to_arrays(self):
        """Legacy .oemb single-protein decode still works."""
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
        """Legacy .oemb batch decode still works."""
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

    def test_decode_one_h5_single(self):
        """New .one.h5 single-protein decode."""
        from src.one_embedding import decode
        from src.one_embedding.io import write_one_h5
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.one.h5"
            write_one_h5(str(path), {
                "per_residue": np.random.randn(50, 768).astype(np.float16),
                "protein_vec": np.random.randn(3072).astype(np.float16),
                "sequence": "M" * 50,
            }, protein_id="test_prot")
            data = decode(str(path))
            assert data["per_residue"].shape == (50, 768)
            assert data["protein_vec"].shape == (3072,)
            assert data["protein_id"] == "test_prot"

    def test_decode_one_h5_batch(self):
        """New .one.h5 batch decode."""
        from src.one_embedding import decode
        from src.one_embedding.io import write_one_h5_batch
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "batch.one.h5"
            write_one_h5_batch(str(path), {
                "a": {"per_residue": np.random.randn(30, 768).astype(np.float16),
                      "protein_vec": np.random.randn(3072).astype(np.float16)},
                "b": {"per_residue": np.random.randn(40, 768).astype(np.float16),
                      "protein_vec": np.random.randn(3072).astype(np.float16)},
            })
            data = decode(str(path))
            assert len(data) == 2
            assert "a" in data and "b" in data

    def test_decode_one_h5_batch_specific_protein(self):
        """New .one.h5 batch decode with protein_id selection."""
        from src.one_embedding import decode
        from src.one_embedding.io import write_one_h5_batch
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "batch.one.h5"
            write_one_h5_batch(str(path), {
                "a": {"per_residue": np.random.randn(30, 768).astype(np.float16),
                      "protein_vec": np.random.randn(3072).astype(np.float16)},
                "b": {"per_residue": np.random.randn(40, 768).astype(np.float16),
                      "protein_vec": np.random.randn(3072).astype(np.float16)},
            })
            data = decode(str(path), protein_id="a")
            assert data["per_residue"].shape == (30, 768)


class TestVersion:
    def test_version_exists(self):
        from src.one_embedding import __version__
        assert __version__ == "1.0.0"
