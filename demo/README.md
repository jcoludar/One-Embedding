# Demo — three forms of the same 10 proteins

A side-by-side, look-with-your-eyes demo. The same 10 CASP12 proteins (ProtT5-XL
embeddings) sit in three shapes here so you can see the trade-offs.

## Build

```bash
uv run python demo/build.py     # ~2 seconds
uv run python demo/show.py      # prints sizes + sample shapes
```

`build.py` reads `data/residue_embeddings/prot_t5_xl_casp12.h5`, takes the
first 10 proteins, and writes the three forms below. The `.h5` outputs are
gitignored — re-build any time.

## What's in each form

| folder | what it is | what you get per protein | typical use |
|---|---|---|---|
| `raw/` | per-residue, fp16 | `(L, 1024)` array | the PLM's full output — what every downstream tool was originally trained on |
| `mean_pool/` | per-protein, fp32 | `(1024,)` vector | naive baseline summary; loses all per-residue resolution |
| `ours/` | per-residue + protein-vec, binary | `(L, 896)` 1-bit signs + `(D,)` means + `(D,)` scales + `(3584,)` protein vec (fp16) | the codec output — keeps per-residue **and** a pre-computed protein-level summary |

## Indicative sizes (10 proteins, mean L ≈ 220)

```
form                                      on-disk   sample shape
------------------------------------------------------------------------------
raw       (per-residue, fp16)            6032.9 KB   CASP12_0: (341, 1024)
mean_pool (per-protein, fp32)              46.0 KB   CASP12_0: (1024,)
ours      (binary 896d, no codebook)      551.9 KB   CASP12_0: (group of 4 datasets)
```

A few things to notice:

- **Mean-pool is small (46 KB) but throws away L.** You can't ask "is residue 137
  disordered?" anymore; it's gone. Useful only for retrieval-style protein-level
  tasks, and even there it underperforms a real per-residue summary.
- **Ours is 11× smaller than raw fp16, ~22× smaller than raw fp32**, and unlike
  mean-pool it keeps per-residue addressability.
- The headline "~37×" claim is asymptotic at long L: the per-protein overhead
  (means/scales/protein-vec) is fixed-cost, so for short proteins (CASP12 has
  several with L < 150) the ratio is closer to 20×. For UniProt-mean L ≈ 350 it
  approaches the headline.

## Inspecting the binary file directly

The decoder is the universal-codec receiver path — `numpy + h5py` (+ `json` for
the metadata blob) only, no `OneEmbeddingCodec`, no codebook:

```python
import json, h5py, numpy as np

with h5py.File("demo/ours/casp12_first10.one.h5") as f:
    meta   = json.loads(f.attrs["metadata"])    # codec settings (d_out, version, ...)
    g      = f["CASP12_0"]                       # one protein per group in batch files
    bits   = g["per_residue_bits"][:]            # uint8, shape (L, ceil(D/8))
    means  = g["per_residue_means"][:]           # (D,)
    scales = g["per_residue_scales"][:]          # (D,)
    L      = int(g.attrs["seq_len"])
    D      = int(meta["d_out"])

unpacked = np.unpackbits(bits, axis=1, bitorder="big")[:, :D]
signs    = unpacked.astype(np.float32) * 2 - 1   # {0,1} → {-1,+1}
per_res  = signs * scales + means                 # (L, D) reconstructed
print(per_res.shape, per_res.dtype)               # (341, 896) float32
```

That's the entire receiver. For files written by `codec.save()` (single
protein) the same code works without the `g = f["CASP12_0"]` step — the
datasets sit at file-level instead of in a per-protein group.
