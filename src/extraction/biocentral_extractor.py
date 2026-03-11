"""Biocentral API wrapper for per-residue embeddings (reduce=False)."""

import time
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_residue_embeddings_biocentral(
    fasta_dict: dict[str, str],
    embedder_name: str = "ProtT5",
    batch_size: int = 15,
    max_retries: int = 3,
) -> dict[str, np.ndarray]:
    """Extract per-residue embeddings via Biocentral API (reduce=False).

    Returns dict: {protein_id: np.ndarray of shape (L, D)}.
    """
    try:
        from biocentral_api import BiocentralAPI, CommonEmbedder
    except ImportError:
        raise ImportError("biocentral_api not installed. pip install biocentral-api")

    # Map string name to enum
    embedder_map = {
        "ProtT5": CommonEmbedder.ProtT5,
        "ESM2_650M": CommonEmbedder.ESM2_650M,
        "ESM_8M": CommonEmbedder.ESM_8M,
        "ESM2_3B": CommonEmbedder.ESM2_3B,
        "ProstT5": CommonEmbedder.ProstT5,
    }
    embedder = embedder_map.get(embedder_name)
    if embedder is None:
        raise ValueError(f"Unknown embedder: {embedder_name}. Choose from: {list(embedder_map.keys())}")

    # Deduplicate by sequence
    seq_to_ids: dict[str, list[str]] = {}
    for sid, seq in fasta_dict.items():
        seq_to_ids.setdefault(seq, []).append(sid)

    unique_seqs = {ids_list[0]: seq for seq, ids_list in seq_to_ids.items()}
    n_dupes = len(fasta_dict) - len(unique_seqs)
    if n_dupes > 0:
        logger.info(f"Deduplicated: {len(fasta_dict)} -> {len(unique_seqs)} unique sequences")

    api = BiocentralAPI()
    unique_embeddings: dict[str, np.ndarray] = {}
    unique_ids = list(unique_seqs.keys())

    for i in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[i : i + batch_size]
        batch = {sid: unique_seqs[sid] for sid in batch_ids}
        logger.info(f"Batch {i // batch_size + 1} ({len(batch)} seqs, {i + len(batch)}/{len(unique_ids)})")

        result = None
        for attempt in range(max_retries):
            try:
                result = api.embed(
                    embedder_name=embedder,
                    reduce=False,  # Per-residue!
                    sequence_data=batch,
                    use_half_precision=False,
                ).run()
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                logger.warning(f"Batch failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if result is None:
            logger.error(f"Failed after {max_retries} attempts for batch starting at {i}")
            continue

        for sid in batch_ids:
            if sid in result:
                unique_embeddings[sid] = np.array(result[sid], dtype=np.float32)

    # Expand back to all IDs
    all_embeddings: dict[str, np.ndarray] = {}
    for seq, ids_list in seq_to_ids.items():
        rep_id = ids_list[0]
        if rep_id in unique_embeddings:
            emb = unique_embeddings[rep_id]
            for sid in ids_list:
                all_embeddings[sid] = emb

    logger.info(f"Generated {len(all_embeddings)}/{len(fasta_dict)} per-residue embeddings")
    return all_embeddings
