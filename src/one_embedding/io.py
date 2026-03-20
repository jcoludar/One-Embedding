"""H5 serialization for OneEmbedding collections."""

from pathlib import Path

import h5py
import numpy as np

from src.one_embedding.embedding import OneEmbedding


def save_one_embeddings(
    embeddings: dict[str, OneEmbedding],
    h5_path: Path | str,
) -> None:
    """Save OneEmbedding collection to H5.

    Each protein is stored as a group with two datasets:
        summary: (summary_dim,) — fixed-size protein-level vector
        residues: (L, d) — per-residue compressed matrix

    File-level attrs store format metadata.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Get common metadata from first embedding
    first = next(iter(embeddings.values()))

    with h5py.File(h5_path, "w") as f:
        f.attrs["format"] = "one_embedding"
        f.attrs["plm"] = first.plm
        f.attrs["latent_dim"] = first.latent_dim
        f.attrs["transform"] = first.transform
        f.attrs["n_proteins"] = len(embeddings)

        for pid, emb in embeddings.items():
            grp = f.create_group(pid)
            grp.create_dataset(
                "summary",
                data=emb.summary,
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset(
                "residues",
                data=emb.residues,
                compression="gzip",
                compression_opts=4,
            )
            grp.attrs["seq_len"] = emb.seq_len


def load_one_embeddings(h5_path: Path | str) -> dict[str, OneEmbedding]:
    """Load OneEmbedding collection from H5."""
    embeddings = {}

    with h5py.File(h5_path, "r") as f:
        plm = str(f.attrs["plm"])
        transform = str(f.attrs["transform"])

        for pid in f.keys():
            grp = f[pid]
            summary = np.array(grp["summary"], dtype=np.float32)
            residues = np.array(grp["residues"], dtype=np.float32)

            embeddings[pid] = OneEmbedding(
                protein_id=pid,
                plm=plm,
                latent_dim=residues.shape[1],
                seq_len=residues.shape[0],
                transform=transform,
                _summary=summary,
                _residues=residues,
            )

    return embeddings


# ── .oemb format ─────────────────────────────────────────────────────

OEMB_VERSION = "1.0"


def write_oemb(
    path: Path | str,
    data: dict,
    protein_id: str = "protein",
) -> None:
    """Write a single protein to a .oemb (H5) file.

    Args:
        path: Output file path (conventionally ends with .oemb).
        data: Dict with keys:
            per_residue (required): np.ndarray shape (L, D) — per-residue embeddings.
            protein_vec (required): np.ndarray shape (V,) — protein-level vector (fp16).
            sequence (optional): str — amino-acid sequence.
            source_model (optional): str — PLM name, e.g. "prot_t5_xl".
            codec (optional): str — codec description, e.g. "ABTT3+RP512+PQ128".
        protein_id: Identifier stored in attrs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    per_residue = np.asarray(data["per_residue"])
    protein_vec = np.asarray(data["protein_vec"], dtype=np.float16)

    with h5py.File(path, "w") as f:
        f.attrs["oemb_version"] = OEMB_VERSION
        f.attrs["protein_id"] = protein_id
        f.attrs["sequence"] = data.get("sequence", "")
        f.attrs["source_model"] = data.get("source_model", "")
        f.attrs["codec"] = data.get("codec", "")

        f.create_dataset(
            "per_residue",
            data=per_residue,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("protein_vec", data=protein_vec)


def read_oemb(path: Path | str) -> dict:
    """Read a single-protein .oemb file.

    Returns:
        Dict with keys: per_residue (float32), protein_vec (float16),
        protein_id, sequence, source_model, codec, oemb_version.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        result = {
            "oemb_version": str(f.attrs.get("oemb_version", "")),
            "protein_id": str(f.attrs.get("protein_id", "")),
            "sequence": str(f.attrs.get("sequence", "")),
            "source_model": str(f.attrs.get("source_model", "")),
            "codec": str(f.attrs.get("codec", "")),
            "per_residue": np.array(f["per_residue"], dtype=np.float32),
            "protein_vec": np.array(f["protein_vec"], dtype=np.float16),
        }

    return result


def write_oemb_batch(
    path: Path | str,
    proteins: dict[str, dict],
) -> None:
    """Write multiple proteins to a single .oemb file.

    Args:
        path: Output file path.
        proteins: Mapping of protein_id -> data dict, where each data dict
            has the same keys as write_oemb (per_residue, protein_vec, and
            optional sequence, source_model, codec).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["oemb_version"] = OEMB_VERSION
        f.attrs["n_proteins"] = len(proteins)

        for pid, data in proteins.items():
            per_residue = np.asarray(data["per_residue"])
            protein_vec = np.asarray(data["protein_vec"], dtype=np.float16)

            grp = f.create_group(pid)
            grp.attrs["sequence"] = data.get("sequence", "")

            grp.create_dataset(
                "per_residue",
                data=per_residue,
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("protein_vec", data=protein_vec)


def read_oemb_batch(
    path: Path | str,
    protein_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Read a batch .oemb file.

    Args:
        path: Path to a batch .oemb file written by write_oemb_batch.
        protein_ids: If provided, only load these protein IDs. If None, load all.

    Returns:
        Mapping of protein_id -> dict with keys: per_residue (float32),
        protein_vec (float16), sequence.
    """
    path = Path(path)
    result: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        ids_to_load = list(f.keys()) if protein_ids is None else protein_ids

        for pid in ids_to_load:
            if pid not in f:
                raise KeyError(f"Protein '{pid}' not found in {path}. Available: {list(f.keys())[:10]}")
            grp = f[pid]
            result[pid] = {
                "per_residue": np.array(grp["per_residue"], dtype=np.float32),
                "protein_vec": np.array(grp["protein_vec"], dtype=np.float16),
                "sequence": str(grp.attrs.get("sequence", "")),
            }

    return result


def inspect_oemb(path: Path | str) -> dict:
    """Quick summary of a .oemb file without loading embeddings.

    Returns:
        Dict with keys:
            file_type: "single" or "batch"
            n_proteins: number of proteins (1 for single)
            n_residues: total residue count (single) or None (batch)
            source_model: str (single files only, else "")
            codec: str (single files only, else "")
            oemb_version: str
            size_bytes: file size in bytes
            protein_ids: list of protein IDs (batch) or [protein_id] (single)
    """
    path = Path(path)
    size_bytes = path.stat().st_size

    with h5py.File(path, "r") as f:
        version = str(f.attrs.get("oemb_version", ""))

        # Distinguish single vs batch by presence of "n_proteins" root attr
        # and whether root-level keys are groups (batch) or datasets (single).
        if "n_proteins" in f.attrs:
            # Batch file
            protein_ids = list(f.keys())
            return {
                "file_type": "batch",
                "n_proteins": int(f.attrs["n_proteins"]),
                "n_residues": None,
                "source_model": "",
                "codec": "",
                "oemb_version": version,
                "size_bytes": size_bytes,
                "protein_ids": protein_ids,
            }
        else:
            # Single-protein file
            n_residues = int(f["per_residue"].shape[0])
            protein_id = str(f.attrs.get("protein_id", ""))
            return {
                "file_type": "single",
                "n_proteins": 1,
                "n_residues": n_residues,
                "source_model": str(f.attrs.get("source_model", "")),
                "codec": str(f.attrs.get("codec", "")),
                "oemb_version": version,
                "size_bytes": size_bytes,
                "protein_ids": [protein_id],
            }


# ── .one.h5 format ──────────────────────────────────────────────────

ONE_H5_FORMAT = "one_embedding"
ONE_H5_VERSION = "1.0"


def _decode_attr(v):
    """Convert HDF5 attribute to native Python type.

    - bytes -> str
    - np.generic (np.int64, np.float32, etc.) -> Python scalar via .item()
    - everything else returned as-is
    """
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_one_h5(
    path: Path | str,
    data: dict,
    protein_id: str = "protein",
    tags: dict | None = None,
) -> None:
    """Write a single protein to a .one.h5 file.

    Args:
        path: Output file path (conventionally ends with .one.h5).
        data: Dict with keys:
            per_residue (required): np.ndarray shape (L, D).
            protein_vec (required): np.ndarray shape (V,).
            sequence (optional): str.
            tags (optional): dict of per-protein freeform tags.
        protein_id: Identifier for the protein group.
        tags: Root-level freeform tags (dict of str/int/float values).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    per_residue = np.asarray(data["per_residue"])
    protein_vec = np.asarray(data["protein_vec"], dtype=np.float16)

    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["format"] = ONE_H5_FORMAT
        f.attrs["version"] = ONE_H5_VERSION
        f.attrs["n_proteins"] = 1

        # Root-level freeform tags
        if tags:
            for k, v in tags.items():
                f.attrs[k] = v

        # Protein group
        grp = f.create_group(protein_id)
        grp.create_dataset(
            "per_residue",
            data=per_residue,
            compression="gzip",
            compression_opts=4,
        )
        grp.create_dataset("protein_vec", data=protein_vec)
        grp.attrs["seq_len"] = per_residue.shape[0]

        # Optional sequence
        if "sequence" in data and data["sequence"]:
            grp.attrs["sequence"] = data["sequence"]

        # Per-protein freeform tags
        per_protein_tags = data.get("tags")
        if per_protein_tags:
            for k, v in per_protein_tags.items():
                grp.attrs[k] = v


def read_one_h5(path: Path | str) -> dict:
    """Read a single-protein .one.h5 file.

    Falls back to read_oemb() if the file lacks format="one_embedding".

    Returns:
        Dict with keys: per_residue (float32), protein_vec (float16),
        protein_id, format, version, tags.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        fmt = _decode_attr(f.attrs.get("format", ""))

        if fmt != ONE_H5_FORMAT:
            # Fall back to legacy .oemb reader
            return read_oemb(path)

        version = _decode_attr(f.attrs.get("version", ""))

        # Collect root-level tags (everything except reserved attrs)
        reserved_root = {"format", "version", "n_proteins"}
        root_tags = {}
        for k in f.attrs:
            if k not in reserved_root:
                root_tags[k] = _decode_attr(f.attrs[k])

        # Find the protein group (first group key)
        protein_id = None
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                protein_id = key
                break

        if protein_id is None:
            raise ValueError(f"No protein group found in {path}")

        grp = f[protein_id]
        per_residue = np.array(grp["per_residue"], dtype=np.float32)
        protein_vec = np.array(grp["protein_vec"], dtype=np.float16)

        # Collect per-protein tags
        reserved_protein = {"seq_len", "sequence"}
        protein_tags = {}
        for k in grp.attrs:
            if k not in reserved_protein:
                protein_tags[k] = _decode_attr(grp.attrs[k])

        result = {
            "per_residue": per_residue,
            "protein_vec": protein_vec,
            "protein_id": protein_id,
            "format": fmt,
            "version": version,
            "tags": {**root_tags, **protein_tags},
        }

        # Include sequence if present
        seq = _decode_attr(grp.attrs.get("sequence", ""))
        if seq:
            result["sequence"] = seq

        return result


def write_one_h5_batch(
    path: Path | str,
    proteins: dict[str, dict],
    tags: dict | None = None,
) -> None:
    """Write multiple proteins to a single .one.h5 file.

    Args:
        path: Output file path.
        proteins: Mapping of protein_id -> data dict. Each data dict has:
            per_residue (required): np.ndarray shape (L, D).
            protein_vec (required): np.ndarray shape (V,).
            sequence (optional): str.
            tags (optional): dict of per-protein freeform tags.
        tags: Root-level freeform tags.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["format"] = ONE_H5_FORMAT
        f.attrs["version"] = ONE_H5_VERSION
        f.attrs["n_proteins"] = len(proteins)

        # Root-level freeform tags
        if tags:
            for k, v in tags.items():
                f.attrs[k] = v

        for pid, data in proteins.items():
            per_residue = np.asarray(data["per_residue"])
            protein_vec = np.asarray(data["protein_vec"], dtype=np.float16)

            grp = f.create_group(pid)
            grp.create_dataset(
                "per_residue",
                data=per_residue,
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("protein_vec", data=protein_vec)
            grp.attrs["seq_len"] = per_residue.shape[0]

            # Optional sequence
            if "sequence" in data and data["sequence"]:
                grp.attrs["sequence"] = data["sequence"]

            # Per-protein freeform tags
            per_protein_tags = data.get("tags")
            if per_protein_tags:
                for k, v in per_protein_tags.items():
                    grp.attrs[k] = v


def read_one_h5_batch(
    path: Path | str,
    protein_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Read a batch .one.h5 file.

    Args:
        path: Path to a batch .one.h5 file.
        protein_ids: If provided, only load these protein IDs. If None, load all.

    Returns:
        Mapping of protein_id -> dict with keys: per_residue (float32),
        protein_vec (float16).
    """
    path = Path(path)
    result: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if protein_ids is None:
            ids_to_load = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        else:
            ids_to_load = protein_ids

        for pid in ids_to_load:
            if pid not in f:
                raise KeyError(
                    f"Protein '{pid}' not found in {path}. "
                    f"Available: {list(f.keys())[:10]}"
                )
            grp = f[pid]
            result[pid] = {
                "per_residue": np.array(grp["per_residue"], dtype=np.float32),
                "protein_vec": np.array(grp["protein_vec"], dtype=np.float16),
            }

    return result


def inspect_one_h5(path: Path | str) -> dict:
    """Quick summary of a .one.h5 file without loading embedding data.

    Returns:
        Dict with keys: format, version, n_proteins, d_out, protein_vec_dim,
        protein_ids, tags, size_bytes.
    """
    path = Path(path)
    size_bytes = path.stat().st_size

    with h5py.File(path, "r") as f:
        fmt = _decode_attr(f.attrs.get("format", ""))
        version = _decode_attr(f.attrs.get("version", ""))
        n_proteins = _decode_attr(f.attrs.get("n_proteins", 0))

        # Collect root-level tags
        reserved_root = {"format", "version", "n_proteins"}
        root_tags = {}
        for k in f.attrs:
            if k not in reserved_root:
                root_tags[k] = _decode_attr(f.attrs[k])

        # Enumerate protein groups
        protein_ids = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

        # Read d_out and protein_vec_dim from first protein (no data loading)
        d_out = None
        protein_vec_dim = None
        if protein_ids:
            first_grp = f[protein_ids[0]]
            if "per_residue" in first_grp:
                d_out = first_grp["per_residue"].shape[1]
            if "protein_vec" in first_grp:
                protein_vec_dim = first_grp["protein_vec"].shape[0]

        return {
            "format": fmt,
            "version": version,
            "n_proteins": n_proteins,
            "d_out": d_out,
            "protein_vec_dim": protein_vec_dim,
            "protein_ids": protein_ids,
            "tags": root_tags,
            "size_bytes": size_bytes,
        }
