"""Embedding extraction — connect any PLM."""

MODELS = {
    "prot_t5": "src.one_embedding.extract.prot_t5",
    "esm2": "src.one_embedding.extract.esm2",
}


def extract_embeddings(input_path, output_path, model="prot_t5", **kwargs):
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS.keys())}")
    import importlib
    mod = importlib.import_module(MODELS[model])
    mod.extract(input_path, output_path, **kwargs)
