import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile
from click.testing import CliRunner
from src.one_embedding.cli import main
from src.one_embedding.io import write_oemb_batch


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output
    assert "encode" in result.output
    assert "disorder" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.0" in result.output


def test_inspect():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.oemb"
        write_oemb_batch(str(path), {
            "prot_a": {
                "per_residue": np.random.randn(80, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
            },
        })
        result = runner.invoke(main, ["inspect", str(path)])
        assert result.exit_code == 0
        assert "n_proteins" in result.output


def test_disorder():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.oemb"
        write_oemb_batch(str(path), {
            "prot_a": {
                "per_residue": np.random.randn(50, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
            },
        })
        result = runner.invoke(main, ["disorder", str(path)])
        assert result.exit_code == 0
        assert "prot_a" in result.output
        assert "residues" in result.output


def test_search():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.oemb"
        write_oemb_batch(str(path), {
            f"prot_{i}": {
                "per_residue": np.random.randn(50, 512).astype(np.float16),
                "protein_vec": np.random.randn(2048).astype(np.float16),
            } for i in range(5)
        })
        result = runner.invoke(main, ["search", str(path), "-k", "2"])
        assert result.exit_code == 0
        assert "sim=" in result.output


def test_encode_command():
    import h5py
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.h5"
        with h5py.File(str(raw), "w") as f:
            f.create_dataset("prot_a", data=np.random.randn(50, 1024).astype(np.float32))
        out = Path(d) / "out.oemb"
        result = runner.invoke(main, ["encode", str(raw), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
