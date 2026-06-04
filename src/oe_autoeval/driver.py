"""Parametrized OE×autoeval driver.

Pure wiring (d_out_eff, codec construction, function selection) is unit-testable off-cluster.
`run()` takes the embedding service + autoeval_pipeline as injected callables so it can be
tested without biotrainer; `build_service()` isolates the one biotrainer-API call (verified
against v1.4.0: get_embedding_service(embedder_name, custom_tokenizer_config, use_half_precision,
device)). On the cluster, run_arm.py wires build_service() + the real autoeval_pipeline into run().
"""
import numpy as np

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.oe_autoeval import wrappers

D_OUT = 896


def d_out_eff_for(native_d):
    """Effective OE per-residue dim: RP only engages when native_d > D_OUT."""
    return min(D_OUT, native_d)


def build_codec(mode, native_d):
    """raw -> None; oe -> default 896d; oe_norp -> d_out=native_d (forces RP off)."""
    if mode == "raw":
        return None
    if mode == "oe":
        return OneEmbeddingCodec()                 # d_out=896 default, binary
    if mode == "oe_norp":
        return OneEmbeddingCodec(d_out=native_d)    # RP-off control (== oe when native_d<=896)
    raise ValueError(f"unknown mode {mode!r}")


def select_embedding_functions(mode, codec, d_out_eff, embed_service=None):
    """Return {'per_residue', 'per_sequence'} closures bound to the embed service."""
    if mode == "raw":
        def raw_pr(sequences):
            return wrappers.raw_per_residue(embed_service, sequences)

        def raw_ps(sequences):
            return wrappers.raw_per_sequence(embed_service, sequences)

        return {"per_residue": raw_pr, "per_sequence": raw_ps}

    def oe_pr(sequences):
        return wrappers.oe_per_residue(codec, embed_service, d_out_eff, sequences)

    def oe_ps(sequences):
        return wrappers.oe_per_sequence(codec, embed_service, d_out_eff, sequences)

    return {"per_residue": oe_pr, "per_sequence": oe_ps}


def build_service(embedder_hf_id, precision, device=None):
    """Construct the biotrainer embedding service (deferred import; needs biotrainer + G1)."""
    from biotrainer.embedders import get_embedding_service
    from biotrainer.utilities import get_device

    return get_embedding_service(
        embedder_name=embedder_hf_id,
        custom_tokenizer_config=None,
        use_half_precision=(precision == "fp16"),
        device=device or get_device(),
    )


def run(*, embed_service, autoeval_pipeline, mode, native_d, precision, label,
        output_dir, reference_embeddings=None):
    """Fit the codec on the reference set, run autoeval PBC with the selected functions.

    `autoeval_pipeline` is injectable for testing. It is a GENERATOR — we drain it so the
    pipeline actually executes. Returns the label on completion.
    """
    codec = build_codec(mode, native_d)
    # Effective per-residue width = min(codec.d_out, native_d): RP is skipped whenever
    # d_out >= native_d. This is correct for BOTH oe (d_out=896) and oe_norp (d_out=native_d);
    # using d_out_eff_for(native_d)=min(896,native_d) would be wrong for oe_norp (the bug
    # fan-review #3 caught: oe_norp would feed 896 while the codec emits native_d).
    d_eff = min(codec.d_out, native_d) if codec is not None else d_out_eff_for(native_d)
    if codec is not None and reference_embeddings is not None:
        codec.fit(reference_embeddings)
    fns = select_embedding_functions(mode, codec, d_eff, embed_service)
    generator = autoeval_pipeline(
        embedder_name=label,
        framework="PBC",
        custom_embedding_function_per_residue=fns["per_residue"],
        custom_embedding_function_per_sequence=fns["per_sequence"],
        output_dir=output_dir,
        use_half_precision=(precision == "fp16"),
    )
    for _ in generator:   # drain — autoeval_pipeline does nothing until consumed
        pass
    return label
