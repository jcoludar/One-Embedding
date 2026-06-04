"""Local ProtT5 per-residue embedding extraction via transformers."""

import re
import sys

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import get_device


def extract_prot_t5_embeddings(
    fasta_dict: dict[str, str],
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 4,
    device: torch.device | None = None,
    max_residues: int = 2000,
) -> dict[str, np.ndarray]:
    """Extract per-residue embeddings from ProtT5 encoder.

    ProtT5 is a T5 encoder with *relative* position bias — it has NO
    architectural sequence-length limit, so we never truncate. ``max_residues``
    is only an out-of-memory guard: sequences above it are SKIPPED loudly, never
    silently chopped.

    History note: this extractor previously called the tokenizer with
    ``truncation=True, max_length=512`` — stock HuggingFace BERT-era boilerplate
    that does NOT apply to T5. It silently truncated every sequence > 511 aa to
    its first 511 residues while still slicing ``[:seq_len]`` with the full
    length, returning short, wrong embeddings with no error. Removed 2026-06-04.

    Returns dict: {protein_id: np.ndarray of shape (L, 1024)}. Sequences longer
    than ``max_residues`` are absent from the result.
    """
    from transformers import AutoTokenizer, T5EncoderModel

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    embed_dim = model.config.d_model
    print(f"Loaded {model_name} (dim={embed_dim}) on {device}")

    # Skip-don't-truncate: drop empty + over-length sequences up front, loudly.
    skipped = [sid for sid, s in fasta_dict.items() if len(s) > max_residues]
    for sid in skipped:
        print(
            f"⚠️  Skipping {sid} (len {len(fasta_dict[sid])} > max_residues "
            f"{max_residues}) — would risk OOM; NOT truncated.",
            file=sys.stderr,
        )
    ids = [sid for sid, s in fasta_dict.items() if 0 < len(s) <= max_residues]
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
            # Guard the old silent-truncation footgun: the hidden state must hold
            # at least seq_len positions. If any cap ever sneaks back into the
            # tokenizer call, fail LOUDLY here rather than return a short tensor.
            assert output.last_hidden_state.shape[1] >= seq_len, (
                f"{sid}: hidden state has {output.last_hidden_state.shape[1]} "
                f"positions but sequence is {seq_len} aa — truncation regression!"
            )
            emb = output.last_hidden_state[j, :seq_len].cpu().numpy().astype(np.float32)
            embeddings[sid] = emb

    msg = f"Extracted {len(embeddings)} proteins, embed_dim={embed_dim}"
    if skipped:
        msg += f" ({len(skipped)} skipped, len > {max_residues} aa)"
    print(msg)
    return embeddings
