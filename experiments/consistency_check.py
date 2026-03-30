"""
Comprehensive consistency check between the production Codec (one_embedding.core.codec)
and the unified OneEmbeddingCodec (src.one_embedding.codec_v2).

NOTE: The old V1 research codec (src.one_embedding.codec.OneEmbeddingCodec with dtype
parameter) is no longer the primary codec. Section 1 now uses the unified
OneEmbeddingCodec from codec_v2 (quantization=None, d_out=512) as the comparison
baseline against the core Codec.

Run with:
    uv run python3 experiments/consistency_check.py
"""
import sys
import os
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    marker = "✓" if condition else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {marker} {name}{suffix}")
    return condition


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Data paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_H5   = os.path.join(PROJECT_ROOT, "data/residue_embeddings/prot_t5_xl_medium5k.h5")
META_CSV = os.path.join(PROJECT_ROOT, "data/proteins/metadata_5k.csv")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Codec output consistency (Old vs New on 10 real proteins)
# ═════════════════════════════════════════════════════════════════════════════
section("1. Codec output consistency — Old vs New on 10 real proteins")

try:
    from src.one_embedding.codec_v2 import OneEmbeddingCodec
    from src.one_embedding.core.codec import Codec

    with h5py.File(EMB_H5, "r") as f:
        protein_ids = list(f.keys())[:10]
        raw_dict = {pid: f[pid][:].astype(np.float32) for pid in protein_ids}

    # Fit both codecs on the same 10-protein corpus
    # old_codec is now the unified OneEmbeddingCodec (quantization=None = fp16, d_out=512)
    old_codec = OneEmbeddingCodec(d_out=512, quantization=None, dct_k=4, seed=42)
    old_codec.fit(raw_dict)
    new_codec = Codec(d_out=512, dct_k=4, seed=42)
    new_codec.fit(raw_dict, k=3)

    shape_pr_ok = True
    shape_pv_ok = True
    cosine_pr_list = []
    cosine_pv_list = []

    for pid, raw in raw_dict.items():
        old_enc = old_codec.encode(raw)
        new_enc = new_codec.encode(raw)

        L = raw.shape[0]
        expected_pr = (L, 512)
        expected_pv = (2048,)

        # Unified codec (quantization=None) uses "per_residue_fp16" key
        old_pr_raw = old_enc.get("per_residue_fp16", old_enc.get("per_residue"))
        if old_pr_raw is None or old_pr_raw.shape != expected_pr:
            shape_pr_ok = False
        if new_enc["per_residue"].shape != expected_pr:
            shape_pr_ok = False
        if old_enc["protein_vec"].shape != expected_pv:
            shape_pv_ok = False
        if new_enc["protein_vec"].shape != expected_pv:
            shape_pv_ok = False

        # Cosine similarity between old and new per_residue (mean over residues)
        old_pr = old_pr_raw.astype(np.float32)
        new_pr = new_enc["per_residue"].astype(np.float32)
        # Per-residue row-wise cosine
        norms_old = np.linalg.norm(old_pr, axis=1, keepdims=True) + 1e-10
        norms_new = np.linalg.norm(new_pr, axis=1, keepdims=True) + 1e-10
        cos_pr = np.sum((old_pr / norms_old) * (new_pr / norms_new), axis=1).mean()
        cosine_pr_list.append(float(cos_pr))

        # Protein-vec cosine
        old_pv = old_enc["protein_vec"].astype(np.float32)
        new_pv = new_enc["protein_vec"].astype(np.float32)
        cos_pv = float(
            np.dot(old_pv, new_pv)
            / (np.linalg.norm(old_pv) * np.linalg.norm(new_pv) + 1e-10)
        )
        cosine_pv_list.append(cos_pv)

    mean_cos_pr = np.mean(cosine_pr_list)
    mean_cos_pv = np.mean(cosine_pv_list)

    check("Old OneEmbeddingCodec per_residue shape (L, 512)", shape_pr_ok)
    check("New Codec per_residue shape (L, 512)", shape_pr_ok)
    check("Old OneEmbeddingCodec protein_vec shape (2048,)", shape_pv_ok)
    check("New Codec protein_vec shape (2048,)", shape_pv_ok)
    check(
        f"per_residue cosine similarity Old vs New > 0.80",
        mean_cos_pr > 0.80,
        f"mean={mean_cos_pr:.4f}",
    )
    check(
        f"protein_vec cosine similarity Old vs New > 0.80",
        mean_cos_pv > 0.80,
        f"mean={mean_cos_pv:.4f}",
    )

    # Print individual protein cosines for transparency
    print(f"\n  Individual protein-vec cosines (old vs new):")
    for pid, c in zip(protein_ids, cosine_pv_list):
        print(f"    {pid[:30]:30s}  cos={c:.4f}")

except Exception as e:
    check("Section 1 (codec consistency)", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Retrieval quality preservation (100 proteins, SCOPe 5K)
# ═════════════════════════════════════════════════════════════════════════════
section("2. Retrieval quality — 100 proteins with new Codec vs raw baseline ~0.73")

try:
    import pandas as pd

    meta = pd.read_csv(META_CSV)
    # Normalise column names: the CSV uses 'id' and 'family'
    id_col = "protein_id" if "protein_id" in meta.columns else "id"
    fam_col = "family_id" if "family_id" in meta.columns else "family"
    meta = meta.rename(columns={id_col: "protein_id", fam_col: "family_id"})

    # Keep proteins present in the H5 file
    with h5py.File(EMB_H5, "r") as f:
        available_ids = set(f.keys())
    meta = meta[meta["protein_id"].isin(available_ids)].reset_index(drop=True)

    # Build a 100-protein sample with meaningful same-family pairs.
    # Sample up to 2 proteins per multi-member family until we reach ~100.
    rng_sample = np.random.RandomState(42)
    fam_sizes = meta.groupby("family_id")["protein_id"].count()
    multi_fams = fam_sizes[fam_sizes >= 2].index.tolist()
    rng_sample.shuffle(multi_fams)

    selected_pids = []
    for fam in multi_fams:
        fam_pids = meta[meta["family_id"] == fam]["protein_id"].tolist()
        rng_sample.shuffle(fam_pids)
        selected_pids.extend(fam_pids[:2])
        if len(selected_pids) >= 100:
            break
    pids100 = selected_pids[:100]
    meta100 = (
        meta[meta["protein_id"].isin(pids100)]
        .set_index("protein_id")
        .loc[pids100]
        .reset_index()
    )

    with h5py.File(EMB_H5, "r") as f:
        raws100 = {pid: f[pid][:].astype(np.float32) for pid in pids100}

    # Fit codec on the same 100-protein corpus
    codec_ret = Codec(d_out=512, dct_k=4, seed=42)
    codec_ret.fit(raws100, k=3)

    # Encode and collect protein_vecs
    pvecs = []
    for pid in pids100:
        enc = codec_ret.encode(raws100[pid])
        pvecs.append(enc["protein_vec"].astype(np.float32))
    pvecs = np.stack(pvecs, axis=0)  # (100, 2048)

    # Raw mean-pool protein vecs (baseline)
    raw_means = np.stack([raws100[pid].mean(axis=0) for pid in pids100], axis=0)

    def ret_at_1(vecs, labels):
        """Compute Ret@1 (nearest neighbor retrieval) given L2-normalised vecs."""
        vecs_n = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        sim = vecs_n @ vecs_n.T  # (N, N)
        np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(axis=1)
        return np.mean([labels[i] == labels[nn[i]] for i in range(len(labels))])

    labels = meta100["family_id"].tolist()
    n_positive = sum(1 for l in labels if labels.count(l) > 1)
    print(f"\n  Sampling: {len(pids100)} proteins, {n_positive} in multi-family pairs")

    ret_raw   = ret_at_1(raw_means, labels)
    ret_codec = ret_at_1(pvecs, labels)

    check(
        "Raw mean-pool Ret@1 in expected range [0.60, 0.90]",
        0.60 <= ret_raw <= 0.90,
        f"Ret@1={ret_raw:.4f}",
    )
    check(
        "New Codec Ret@1 > 0.50 (well above random)",
        ret_codec > 0.50,
        f"Ret@1={ret_codec:.4f}",
    )
    check(
        "New Codec Ret@1 within 0.15 of raw baseline (correlated)",
        abs(ret_codec - ret_raw) < 0.15,
        f"delta={ret_codec - ret_raw:+.4f}",
    )
    print(f"\n  Raw mean-pool Ret@1    : {ret_raw:.4f}")
    print(f"  New Codec Ret@1        : {ret_codec:.4f}  (known full-5K baseline ~0.73)")

except Exception as e:
    check("Section 2 (retrieval quality)", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — .oemb roundtrip with real data
# ═════════════════════════════════════════════════════════════════════════════
section("3. .oemb roundtrip — bit-exact and h5py-only readable")

try:
    from src.one_embedding.io import write_oemb_batch, read_oemb_batch

    # Use the already-encoded 10 proteins
    codec_rt = Codec(d_out=512, dct_k=4, seed=42)
    codec_rt.fit(raw_dict, k=3)
    proteins_enc = {pid: codec_rt.encode(raw_dict[pid]) for pid in protein_ids}

    with tempfile.NamedTemporaryFile(suffix=".oemb", delete=False) as tmp:
        oemb_path = tmp.name

    # Write
    write_oemb_batch(oemb_path, proteins_enc)

    # Read back via codec API
    loaded_api = read_oemb_batch(oemb_path)

    all_pr_exact = True
    all_pv_exact = True
    for pid in protein_ids:
        orig_pr = proteins_enc[pid]["per_residue"].astype(np.float32)
        orig_pv = proteins_enc[pid]["protein_vec"].astype(np.float16)

        loaded_pr = loaded_api[pid]["per_residue"].astype(np.float32)
        loaded_pv = loaded_api[pid]["protein_vec"].astype(np.float16)

        if not np.array_equal(orig_pr.astype(np.float16), loaded_pr.astype(np.float16)):
            all_pr_exact = False
        if not np.array_equal(orig_pv, loaded_pv):
            all_pv_exact = False

    check("per_residue bit-exact after .oemb roundtrip", all_pr_exact)
    check("protein_vec bit-exact after .oemb roundtrip", all_pv_exact)

    # Read back with ONLY h5py + numpy (no codec import)
    h5py_only_ok = True
    h5py_pr_shapes_ok = True
    h5py_pv_shapes_ok = True
    try:
        with h5py.File(oemb_path, "r") as f:
            pids_in_file = list(f.keys())
            for pid in pids_in_file[:3]:
                grp = f[pid]
                pr_arr = np.array(grp["per_residue"])
                pv_arr = np.array(grp["protein_vec"])
                if pr_arr.ndim != 2 or pr_arr.shape[1] != 512:
                    h5py_pr_shapes_ok = False
                if pv_arr.shape != (2048,):
                    h5py_pv_shapes_ok = False
    except Exception as e2:
        h5py_only_ok = False

    check(".oemb readable with h5py+numpy only (no codec import)", h5py_only_ok)
    check(".oemb per_residue shape (L,512) readable with h5py", h5py_pr_shapes_ok)
    check(".oemb protein_vec shape (2048,) readable with h5py", h5py_pv_shapes_ok)

    os.unlink(oemb_path)

except Exception as e:
    check("Section 3 (.oemb roundtrip)", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Import hygiene
# ═════════════════════════════════════════════════════════════════════════════
section("4. Import hygiene — all public APIs importable without circular imports")

def try_import(description, import_fn):
    try:
        import_fn()
        check(description, True)
        return True
    except Exception as e:
        check(description, False, str(e))
        return False

try_import(
    "from src.one_embedding.codec import OneEmbeddingCodec",
    lambda: __import__("src.one_embedding.codec", fromlist=["OneEmbeddingCodec"]),
)

try_import(
    "from src.one_embedding.core.codec import Codec",
    lambda: __import__("src.one_embedding.core.codec", fromlist=["Codec"]),
)

def _import_top_level():
    import importlib
    mod = importlib.import_module("src.one_embedding")
    for name in ("encode", "decode", "Codec", "__version__"):
        assert hasattr(mod, name), f"Missing: {name}"

try_import(
    "from src.one_embedding import encode, decode, Codec, __version__",
    _import_top_level,
)

def _import_tools():
    from src.one_embedding.tools import disorder, search, classify, align

try_import(
    "from src.one_embedding.tools import disorder, search, classify, align",
    _import_tools,
)

def _import_all():
    """Trigger a fresh import of everything in one_embedding to surface circulars."""
    import importlib
    # These are the key sub-modules
    modules = [
        "src.one_embedding.core",
        "src.one_embedding.core.codec",
        "src.one_embedding.core.preprocessing",
        "src.one_embedding.core.projection",
        "src.one_embedding.codec",
        "src.one_embedding.io",
        "src.one_embedding.transforms",
        "src.one_embedding.universal_transforms",
        "src.one_embedding.enriched_transforms",
        "src.one_embedding.path_transforms",
        "src.one_embedding.tools",
    ]
    for m in modules:
        importlib.import_module(m)

try_import(
    "No circular imports — all sub-modules importable",
    _import_all,
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Edge cases
# ═════════════════════════════════════════════════════════════════════════════
section("5. Edge cases — L=1, L=5000, all-zero, all-identical")

ec_codec = Codec(d_out=512, dct_k=4, seed=42)
D = 1024  # ProtT5 dimension

# Edge case: L=1
try:
    raw_L1 = np.random.RandomState(0).randn(1, D).astype(np.float32)
    enc_L1 = ec_codec.encode(raw_L1)
    check(
        "L=1: per_residue shape (1, 512)",
        enc_L1["per_residue"].shape == (1, 512),
        str(enc_L1["per_residue"].shape),
    )
    check(
        "L=1: protein_vec shape (2048,)",
        enc_L1["protein_vec"].shape == (2048,),
        str(enc_L1["protein_vec"].shape),
    )
    # When L=1, only 1 DCT coefficient exists; rest should be zero-padded
    pv_arr = enc_L1["protein_vec"].astype(np.float32)
    # Check that padding happened (last 512*3 entries == 0)
    padded_ok = np.allclose(pv_arr[512:], 0.0)
    check(
        "L=1: protein_vec correctly zero-padded for missing DCT coefficients",
        padded_ok,
    )
except Exception as e:
    check("L=1 edge case", False, str(e))
    traceback.print_exc()

# Edge case: L=5000
try:
    raw_L5000 = np.random.RandomState(1).randn(5000, D).astype(np.float32)
    enc_L5000 = ec_codec.encode(raw_L5000)
    check(
        "L=5000: per_residue shape (5000, 512)",
        enc_L5000["per_residue"].shape == (5000, 512),
        str(enc_L5000["per_residue"].shape),
    )
    check(
        "L=5000: protein_vec shape (2048,)",
        enc_L5000["protein_vec"].shape == (2048,),
        str(enc_L5000["protein_vec"].shape),
    )
    check(
        "L=5000: per_residue is finite (no NaN/Inf)",
        np.all(np.isfinite(enc_L5000["per_residue"].astype(np.float32))),
    )
except Exception as e:
    check("L=5000 edge case", False, str(e))
    traceback.print_exc()

# Edge case: all-zero embeddings
try:
    raw_zero = np.zeros((50, D), dtype=np.float32)
    enc_zero = ec_codec.encode(raw_zero)
    check(
        "All-zero: per_residue shape (50, 512)",
        enc_zero["per_residue"].shape == (50, 512),
        str(enc_zero["per_residue"].shape),
    )
    check(
        "All-zero: protein_vec finite (no NaN/Inf)",
        np.all(np.isfinite(enc_zero["protein_vec"].astype(np.float32))),
    )
    # All-zero input → all-zero output (projection is linear)
    check(
        "All-zero: per_residue is all zeros",
        np.allclose(enc_zero["per_residue"].astype(np.float32), 0.0),
    )
except Exception as e:
    check("All-zero edge case", False, str(e))
    traceback.print_exc()

# Edge case: all-identical embeddings (constant rows)
try:
    row = np.random.RandomState(3).randn(D).astype(np.float32)
    raw_ident = np.tile(row, (50, 1))
    enc_ident = ec_codec.encode(raw_ident)
    check(
        "All-identical: per_residue shape (50, 512)",
        enc_ident["per_residue"].shape == (50, 512),
        str(enc_ident["per_residue"].shape),
    )
    check(
        "All-identical: protein_vec finite",
        np.all(np.isfinite(enc_ident["protein_vec"].astype(np.float32))),
    )
    # All rows identical → per_residue rows should all be identical
    pr_ident = enc_ident["per_residue"].astype(np.float32)
    rows_identical = np.allclose(pr_ident, pr_ident[0:1], atol=1e-3)
    check("All-identical: per_residue rows are all identical", rows_identical)
except Exception as e:
    check("All-identical edge case", False, str(e))
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
total  = len(results)

print(f"\n  {'Check':<55} Status")
print(f"  {'-'*55} ------")
for name, status, detail in results:
    marker = "✓" if status == PASS else "✗"
    suffix = f" ({detail})" if detail else ""
    trunc = (name + suffix)[:72]
    print(f"  {marker} {trunc:<72}  {status}")

print(f"\n  Total: {total}  |  PASS: {n_pass}  |  FAIL: {n_fail}")
print(f"\n  Overall: {'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")

sys.exit(0 if n_fail == 0 else 1)
