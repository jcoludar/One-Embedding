"""Local ESM2 per-residue embedding extraction."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import get_device


def extract_residue_embeddings(
    fasta_dict: dict[str, str],
    model_name: str = "esm2_t6_8M_UR50D",
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-residue embeddings from ESM2 model.

    Returns dict: {protein_id: np.ndarray of shape (L, D)}.
    """
    import esm

    if device is None:
        device = get_device()

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    num_layers = model.num_layers
    embed_dim = model.embed_dim

    print(f"Loaded {model_name} (dim={embed_dim}) on {device}")

    ids = list(fasta_dict.keys())
    embeddings = {}

    for i in tqdm(range(0, len(ids), batch_size), desc="Extracting embeddings"):
        batch_ids = ids[i : i + batch_size]
        batch_data = [(sid, fasta_dict[sid]) for sid in batch_ids]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers])

        token_representations = results["representations"][num_layers]

        for j, (sid, seq) in enumerate(batch_data):
            seq_len = len(seq)
            # Exclude BOS token (index 0), take residue positions 1..seq_len
            residue_emb = token_representations[j, 1 : seq_len + 1].cpu().numpy().astype(np.float32)
            embeddings[sid] = residue_emb

    print(f"Extracted {len(embeddings)} proteins, embed_dim={embed_dim}")
    return embeddings
