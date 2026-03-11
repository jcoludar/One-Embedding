"""Local ProtT5 per-residue embedding extraction via transformers."""

import re

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import get_device


def extract_prot_t5_embeddings(
    fasta_dict: dict[str, str],
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-residue embeddings from ProtT5 encoder.

    Returns dict: {protein_id: np.ndarray of shape (L, 1024)}.
    """
    from transformers import AutoTokenizer, T5EncoderModel

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    embed_dim = model.config.d_model
    print(f"Loaded {model_name} (dim={embed_dim}) on {device}")

    ids = list(fasta_dict.keys())
    embeddings = {}

    for i in tqdm(range(0, len(ids), batch_size), desc="Extracting ProtT5"):
        batch_ids = ids[i : i + batch_size]
        # ProtT5 requires space-separated AAs; replace rare AAs with X
        batch_seqs = []
        for sid in batch_ids:
            seq = re.sub(r"[UZOB]", "X", fasta_dict[sid])
            batch_seqs.append(" ".join(list(seq)))

        encoded = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        # output.last_hidden_state: (B, L_tokens, d_model)
        # First seq_len positions correspond to amino acids (before EOS/padding)
        for j, sid in enumerate(batch_ids):
            seq_len = len(fasta_dict[sid])
            emb = output.last_hidden_state[j, :seq_len].cpu().numpy().astype(np.float32)
            embeddings[sid] = emb

    print(f"Extracted {len(embeddings)} proteins, embed_dim={embed_dim}")
    return embeddings
