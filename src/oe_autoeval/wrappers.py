"""OE / raw embedding-function wrappers for biotrainer autoeval custom embedders.

autoeval validates only the per-residue ROW count (L), never the feature width, so a
wrong-width array (e.g. the packed `per_residue_bits`, shape (L, ceil(D/8))) is stored as
silent garbage. These wrappers hard-assert dtype+shape at the boundary, and yield the
codec's canonical consumer representation: per-residue = decode_per_residue(encode(raw))
(dequantized float, matching src/one_embedding/vep.py); per-sequence = protein_vec.
"""
import numpy as np


def oe_per_residue(codec, embed_service, d_out_eff, sequences):
    """Yield (sequence_str, (L, d_out_eff) float32) = decode_per_residue(encode(raw))."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        raw = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        encoded = codec.encode(raw)
        arr = np.asarray(codec.decode_per_residue(encoded), dtype=np.float32)
        if arr.shape != (len(rec.seq), d_out_eff):
            raise ValueError(
                f"per-residue shape {arr.shape} != {(len(rec.seq), d_out_eff)} "
                f"(seq len {len(rec.seq)}) — wrong-width array would be stored silently"
            )
        yield rec.seq, arr


def oe_per_sequence(codec, embed_service, d_out_eff, sequences):
    """Yield (sequence_str, (4*d_out_eff,) float32) = encode(raw)['protein_vec'].

    NOTE: protein_vec is emitted by the codec as float16 (codec_v2.py); we upcast to float32
    here (the probe sees fp16-precision features regardless of the float32 label). For L<4 the
    DCT summary collapses to (min(4,L)*d_out_eff,), so we reject L<4 rather than store a
    ragged-width per-sequence vector that a fixed-width probe cannot consume.
    """
    expected = 4 * d_out_eff
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        raw = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        if raw.shape[0] < 4:
            raise ValueError(f"protein len {raw.shape[0]} shorter than dct_k=4 ({rec.seq[:8]}…)")
        encoded = codec.encode(raw)
        vec = np.asarray(encoded["protein_vec"], dtype=np.float32)
        if vec.shape != (expected,):
            raise ValueError(f"per-sequence shape {vec.shape} != {(expected,)}")
        yield rec.seq, vec


def raw_per_residue(embed_service, sequences):
    """Yield (sequence_str, (L, native_D) float32) — uncompressed baseline arm."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        yield rec.seq, np.asarray(emb.cpu().numpy(), dtype=np.float32)


def raw_per_sequence(embed_service, sequences):
    """Yield (sequence_str, (native_D,) float32) — mean-pooled baseline arm."""
    for rec, emb in embed_service.generate_embeddings(input_data=sequences, reduce=False):
        arr = np.asarray(emb.cpu().numpy(), dtype=np.float32)
        yield rec.seq, arr.mean(axis=0).astype(np.float32)
