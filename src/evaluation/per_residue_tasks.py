"""Per-residue evaluation tasks: SS3, SS8, disorder prediction, TM topology.

Evaluates how well compressed per-residue embeddings retain residue-level
information by training linear probes on compressed vs. original embeddings.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr


# ── FASTA helper (no BioPython dependency) ─────────────────────


def _parse_fasta_simple(path: Path | str) -> dict[str, str]:
    """Parse a FASTA file into {id: sequence} without BioPython.

    IDs are the first whitespace-delimited token after '>'.
    """
    sequences = {}
    current_id = None
    current_seq: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


# ── SS3 / SS8 Secondary Structure ─────────────────────────────


SS3_MAP = {"H": 0, "E": 1, "C": 2}  # Helix, Strand, Coil
SS8_MAP = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "C": 7}
SS8_TO_SS3 = {
    "H": "H", "G": "H", "I": "H",   # Helix types → H
    "E": "E", "B": "E",              # Strand types → E
    "T": "C", "S": "C", "C": "C", " ": "C", "-": "C",  # Coil/other → C
}


def load_cb513_csv(csv_path: Path | str) -> tuple[dict, dict, dict, dict]:
    """Load CB513 dataset from CSV file.

    Expected CSV columns: input, dssp3, dssp8, disorder, cb513_mask

    Returns:
        sequences: {protein_id: amino_acid_string}
        ss3_labels: {protein_id: SS3_label_string}  (H/E/C per residue)
        ss8_labels: {protein_id: SS8_label_string}  (H/B/E/G/I/T/S/C per residue)
        disorder_labels: {protein_id: ndarray}  (binary 0/1 per residue)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}, {}, {}, {}

    sequences = {}
    ss3_labels = {}
    ss8_labels = {}
    disorder_labels = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pid = f"cb513_{i}"
            seq = row["input"]
            ss3 = row["dssp3"]
            ss8 = row["dssp8"]

            if len(ss3) != len(seq) or len(ss8) != len(seq):
                continue

            sequences[pid] = seq
            ss3_labels[pid] = ss3
            ss8_labels[pid] = ss8

            disorder_vals = [float(x) for x in row["disorder"].split()]
            if len(disorder_vals) == len(seq):
                disorder_labels[pid] = np.array(disorder_vals, dtype=np.float32)

    return sequences, ss3_labels, ss8_labels, disorder_labels


def load_cb513(data_dir: Path | str) -> list[dict]:
    """Load CB513 dataset from FASTA + SS annotations.

    Expected files in data_dir:
        cb513_sequences.fasta  — protein sequences
        cb513_ss.txt           — SS3 labels, one line per protein (H/E/C chars)

    Returns list of {id: str, sequence: str, ss3: str}.
    """
    data_dir = Path(data_dir)
    seq_path = data_dir / "cb513_sequences.fasta"
    ss_path = data_dir / "cb513_ss.txt"

    if not seq_path.exists() or not ss_path.exists():
        return []

    from Bio import SeqIO
    sequences = {}
    for record in SeqIO.parse(str(seq_path), "fasta"):
        sequences[record.id] = str(record.seq)

    ss_labels = {}
    with open(ss_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ss_labels[parts[0]] = parts[1]

    proteins = []
    for pid, seq in sequences.items():
        if pid in ss_labels:
            ss3 = ss_labels[pid]
            if len(ss3) == len(seq):
                proteins.append({"id": pid, "sequence": seq, "ss3": ss3})

    return proteins


def evaluate_ss3_probe(
    embeddings: dict[str, np.ndarray],
    ss3_labels: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    max_len: int = 512,
) -> dict[str, float]:
    """Train a per-residue linear probe for SS3 prediction.

    Args:
        embeddings: {protein_id: (L, D) array} — original or compressed.
        ss3_labels: {protein_id: "HHECCC..."} — SS3 string per protein.
        train_ids: Protein IDs for training.
        test_ids: Protein IDs for testing.

    Returns: {q3: float, per_class_acc: dict, n_train_residues: int, n_test_residues: int}
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for pid in train_ids:
        if pid not in embeddings or pid not in ss3_labels:
            continue
        emb = embeddings[pid][:max_len]
        ss3 = ss3_labels[pid][:max_len]
        for i, (vec, label) in enumerate(zip(emb, ss3)):
            if label in SS3_MAP:
                X_train.append(vec)
                y_train.append(SS3_MAP[label])

    for pid in test_ids:
        if pid not in embeddings or pid not in ss3_labels:
            continue
        emb = embeddings[pid][:max_len]
        ss3 = ss3_labels[pid][:max_len]
        for i, (vec, label) in enumerate(zip(emb, ss3)):
            if label in SS3_MAP:
                X_test.append(vec)
                y_test.append(SS3_MAP[label])

    if len(X_train) < 10 or len(X_test) < 10:
        return {"q3": 0.0, "n_train_residues": len(X_train), "n_test_residues": len(X_test)}

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    q3 = accuracy_score(y_test, y_pred)
    per_class = {}
    for label_name, label_idx in SS3_MAP.items():
        mask = y_test == label_idx
        if mask.sum() > 0:
            per_class[label_name] = float(accuracy_score(y_test[mask], y_pred[mask]))

    return {
        "q3": float(q3),
        "per_class_acc": per_class,
        "n_train_residues": len(X_train),
        "n_test_residues": len(X_test),
    }


def evaluate_ss8_probe(
    embeddings: dict[str, np.ndarray],
    ss8_labels: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    max_len: int = 512,
) -> dict[str, float]:
    """Train a per-residue linear probe for SS8 prediction.

    Args:
        embeddings: {protein_id: (L, D) array} — original or compressed.
        ss8_labels: {protein_id: "HBEGITSC..."} — SS8 string per protein.
        train_ids: Protein IDs for training.
        test_ids: Protein IDs for testing.

    Returns: {q8: float, per_class_acc: dict, n_train_residues: int, n_test_residues: int}
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for pid in train_ids:
        if pid not in embeddings or pid not in ss8_labels:
            continue
        emb = embeddings[pid][:max_len]
        ss8 = ss8_labels[pid][:max_len]
        for vec, label in zip(emb, ss8):
            if label in SS8_MAP:
                X_train.append(vec)
                y_train.append(SS8_MAP[label])

    for pid in test_ids:
        if pid not in embeddings or pid not in ss8_labels:
            continue
        emb = embeddings[pid][:max_len]
        ss8 = ss8_labels[pid][:max_len]
        for vec, label in zip(emb, ss8):
            if label in SS8_MAP:
                X_test.append(vec)
                y_test.append(SS8_MAP[label])

    if len(X_train) < 10 or len(X_test) < 10:
        return {"q8": 0.0, "n_train_residues": len(X_train), "n_test_residues": len(X_test)}

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    q8 = accuracy_score(y_test, y_pred)
    per_class = {}
    for label_name, label_idx in SS8_MAP.items():
        mask = y_test == label_idx
        if mask.sum() > 0:
            per_class[label_name] = float(accuracy_score(y_test[mask], y_pred[mask]))

    return {
        "q8": float(q8),
        "per_class_acc": per_class,
        "n_train_residues": len(X_train),
        "n_test_residues": len(X_test),
    }


# ── Disorder Prediction (CheZOD) ──────────────────────────────


def load_chezod(data_dir: Path | str) -> list[dict]:
    """Load CheZOD dataset (disorder z-scores per residue).

    Expected files in data_dir:
        chezod_sequences.fasta  — protein sequences
        chezod_scores.txt       — tab-separated: protein_id \\t comma-separated z-scores

    Returns list of {id: str, sequence: str, scores: np.ndarray}.
    """
    data_dir = Path(data_dir)
    seq_path = data_dir / "chezod_sequences.fasta"
    scores_path = data_dir / "chezod_scores.txt"

    if not seq_path.exists() or not scores_path.exists():
        return []

    from Bio import SeqIO
    sequences = {}
    for record in SeqIO.parse(str(seq_path), "fasta"):
        sequences[record.id] = str(record.seq)

    scores = {}
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pid = parts[0]
                vals = [float(x) for x in parts[1].split(",") if x.strip()]
                scores[pid] = np.array(vals, dtype=np.float32)

    proteins = []
    for pid, seq in sequences.items():
        if pid in scores and len(scores[pid]) == len(seq):
            proteins.append({"id": pid, "sequence": seq, "scores": scores[pid]})

    return proteins


def load_chezod_seth(data_dir: Path | str) -> tuple[dict, dict, list, list]:
    """Load CheZOD dataset from SETH directory structure.

    Parses the SETH-provided CheZOD1174 training set and CheZOD117 test set.

    Training data:
        SETH/CheZOD1174_training_set_sequences.fasta
        SETH/CheZOD1174_training_set_CheZOD_scores.txt  (ID:\\tcomma-separated z-scores)
    Test data:
        SETH/CheZOD117_test_set_sequences.fasta
        SETH/CheZOD117_test_scores/zscores{ID}.txt  (residue_char index z_score per line)

    Args:
        data_dir: Directory containing the SETH/ subdirectory.

    Returns:
        sequences: {protein_id: amino_acid_string}
        disorder_scores: {protein_id: ndarray of z-scores}  (999 values replaced with NaN)
        train_ids: list of training protein IDs
        test_ids: list of test protein IDs
    """
    data_dir = Path(data_dir)
    seth_dir = data_dir / "SETH"

    sequences: dict[str, str] = {}
    disorder_scores: dict[str, np.ndarray] = {}
    train_ids: list[str] = []
    test_ids: list[str] = []

    # ── Training set ──────────────────────────────────────────
    train_fasta = seth_dir / "CheZOD1174_training_set_sequences.fasta"
    train_scores_file = seth_dir / "CheZOD1174_training_set_CheZOD_scores.txt"

    if train_fasta.exists() and train_scores_file.exists():
        train_seqs = _parse_fasta_simple(train_fasta)

        # Parse training scores: each line is "ID:\tcomma-separated z-scores"
        train_scores: dict[str, np.ndarray] = {}
        with open(train_scores_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "26672:\t0.123,0.456,999,..."
                if ":\t" in line:
                    pid, score_str = line.split(":\t", 1)
                elif ":" in line:
                    pid, score_str = line.split(":", 1)
                    score_str = score_str.lstrip("\t ")
                else:
                    continue
                pid = pid.strip()
                vals = []
                for v in score_str.split(","):
                    v = v.strip()
                    if v:
                        fv = float(v)
                        vals.append(np.nan if fv == 999 else fv)
                train_scores[pid] = np.array(vals, dtype=np.float32)

        for pid, seq in train_seqs.items():
            if pid in train_scores and len(train_scores[pid]) == len(seq):
                sequences[pid] = seq
                disorder_scores[pid] = train_scores[pid]
                train_ids.append(pid)

    # ── Test set ──────────────────────────────────────────────
    test_fasta = seth_dir / "CheZOD117_test_set_sequences.fasta"
    test_scores_dir = seth_dir / "CheZOD117_test_scores"

    if test_fasta.exists() and test_scores_dir.exists():
        test_seqs = _parse_fasta_simple(test_fasta)

        for pid, seq in test_seqs.items():
            score_file = test_scores_dir / f"zscores{pid}.txt"
            if not score_file.exists():
                continue

            # Parse per-residue score file: "residue_char  index  z_score" per line
            residue_scores: dict[int, float] = {}
            with open(score_file) as f:
                for sline in f:
                    sline = sline.strip()
                    if not sline:
                        continue
                    parts = sline.split()
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[1])
                            zscore = float(parts[2])
                            residue_scores[idx] = zscore
                        except ValueError:
                            continue

            # Build full-length score array; positions not in file get NaN
            scores_arr = np.full(len(seq), np.nan, dtype=np.float32)
            for idx, zscore in residue_scores.items():
                # Indices in test files are 1-based
                pos = idx - 1
                if 0 <= pos < len(seq):
                    scores_arr[pos] = np.nan if zscore == 999 else zscore

            sequences[pid] = seq
            disorder_scores[pid] = scores_arr
            test_ids.append(pid)

    return sequences, disorder_scores, train_ids, test_ids


def evaluate_disorder_probe(
    embeddings: dict[str, np.ndarray],
    disorder_scores: dict[str, np.ndarray],
    train_ids: list[str],
    test_ids: list[str],
    max_len: int = 512,
) -> dict[str, float]:
    """Train a per-residue linear regressor for disorder z-score prediction.

    Args:
        embeddings: {protein_id: (L, D) array}.
        disorder_scores: {protein_id: (L,) z-score array}.
        train_ids, test_ids: Protein IDs.

    Returns: {spearman_rho: float, mse: float, n_train_residues: int, n_test_residues: int}
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for pid in train_ids:
        if pid not in embeddings or pid not in disorder_scores:
            continue
        emb = embeddings[pid][:max_len]
        sc = disorder_scores[pid][:max_len]
        L = min(len(emb), len(sc))
        for i in range(L):
            if not np.isnan(sc[i]):
                X_train.append(emb[i])
                y_train.append(sc[i])

    for pid in test_ids:
        if pid not in embeddings or pid not in disorder_scores:
            continue
        emb = embeddings[pid][:max_len]
        sc = disorder_scores[pid][:max_len]
        L = min(len(emb), len(sc))
        for i in range(L):
            if not np.isnan(sc[i]):
                X_test.append(emb[i])
                y_test.append(sc[i])

    if len(X_train) < 10 or len(X_test) < 10:
        return {"spearman_rho": 0.0, "n_train_residues": len(X_train), "n_test_residues": len(X_test)}

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    rho, p_val = spearmanr(y_test, y_pred)
    mse = float(np.mean((y_test - y_pred) ** 2))

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "mse": mse,
        "n_train_residues": len(X_train),
        "n_test_residues": len(X_test),
    }


# ── TM Topology ───────────────────────────────────────────────


TM_LABELS = {"H": 0, "B": 1, "S": 2, "O": 3}  # TM-helix, TM-beta, signal, other


def load_tmbed(data_dir: Path | str) -> list[dict]:
    """Load TMbed topology annotations.

    Expected files in data_dir:
        tmbed_sequences.fasta  — protein sequences
        tmbed_topology.txt     — tab-separated: protein_id \\t topology_string (H/B/S/O chars)

    Returns list of {id: str, sequence: str, topology: str}.
    """
    data_dir = Path(data_dir)
    seq_path = data_dir / "tmbed_sequences.fasta"
    topo_path = data_dir / "tmbed_topology.txt"

    if not seq_path.exists() or not topo_path.exists():
        return []

    from Bio import SeqIO
    sequences = {}
    for record in SeqIO.parse(str(seq_path), "fasta"):
        sequences[record.id] = str(record.seq)

    topology = {}
    with open(topo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                topology[parts[0]] = parts[1]

    proteins = []
    for pid, seq in sequences.items():
        if pid in topology and len(topology[pid]) == len(seq):
            proteins.append({"id": pid, "sequence": seq, "topology": topology[pid]})

    return proteins


# Label normalization map for TMbed annotated FASTA
_TMBED_LABEL_NORM = {
    "H": "H",  # TM-helix (inside→outside)
    "h": "H",  # TM-helix (outside→inside)
    "B": "B",  # TM-beta (inside→outside)
    "b": "B",  # TM-beta (outside→inside)
    "S": "S",  # Signal peptide
    "1": "O",  # Non-TM (inside)
    "2": "O",  # Non-TM (outside)
    "U": "O",  # Non-TM (unknown side)
}


def load_tmbed_annotated(fasta_path: Path | str) -> tuple[dict, dict]:
    """Load TMbed topology from annotated FASTA (alternating seq/topology lines).

    The file has triplets of lines:
        >header
        AMINO_ACID_SEQUENCE
        TOPOLOGY_LABEL_STRING

    Topology chars: H/h=TM-helix, B/b=TM-beta, S=Signal, 1/2/U=Other.
    Labels are normalized to 4 classes matching TM_LABELS: H, B, S, O.

    Args:
        fasta_path: Path to the annotated FASTA file (e.g. cv_00_annotated.fasta).

    Returns:
        sequences: {protein_id: amino_acid_string}
        topology_labels: {protein_id: topology_string with normalized labels}
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        return {}, {}

    sequences: dict[str, str] = {}
    topology_labels: dict[str, str] = {}

    with open(fasta_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        # Expect a header line starting with '>'
        if not lines[i].startswith(">"):
            i += 1
            continue

        header = lines[i][1:].split()[0]  # full first token after '>'
        # Extract concise ID: first '|'-delimited field
        pid = header.split("|")[0]

        if i + 2 >= len(lines):
            break

        seq = lines[i + 1]
        topo_raw = lines[i + 2]

        # Validate: topology line should not be a header and lengths must match
        if topo_raw.startswith(">") or len(topo_raw) != len(seq):
            i += 1
            continue

        # Normalize topology labels
        topo_normalized = []
        valid = True
        for ch in topo_raw:
            norm = _TMBED_LABEL_NORM.get(ch)
            if norm is None:
                valid = False
                break
            topo_normalized.append(norm)

        if valid:
            sequences[pid] = seq
            topology_labels[pid] = "".join(topo_normalized)

        i += 3  # advance past this triplet

    return sequences, topology_labels


def evaluate_tm_probe(
    embeddings: dict[str, np.ndarray],
    topology_labels: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    max_len: int = 512,
) -> dict[str, float]:
    """Train a per-residue linear probe for TM topology prediction.

    Args:
        embeddings: {protein_id: (L, D) array}.
        topology_labels: {protein_id: "HHHHOOO..."} per residue.
        train_ids, test_ids: Protein IDs.

    Returns: {accuracy: float, macro_f1: float, per_class_f1: dict}
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for pid in train_ids:
        if pid not in embeddings or pid not in topology_labels:
            continue
        emb = embeddings[pid][:max_len]
        topo = topology_labels[pid][:max_len]
        for i, (vec, label) in enumerate(zip(emb, topo)):
            if label in TM_LABELS:
                X_train.append(vec)
                y_train.append(TM_LABELS[label])

    for pid in test_ids:
        if pid not in embeddings or pid not in topology_labels:
            continue
        emb = embeddings[pid][:max_len]
        topo = topology_labels[pid][:max_len]
        for i, (vec, label) in enumerate(zip(emb, topo)):
            if label in TM_LABELS:
                X_test.append(vec)
                y_test.append(TM_LABELS[label])

    if len(X_train) < 10 or len(X_test) < 10:
        return {"accuracy": 0.0, "n_train_residues": len(X_train), "n_test_residues": len(X_test)}

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    per_class = {}
    label_names = {v: k for k, v in TM_LABELS.items()}
    for idx in sorted(TM_LABELS.values()):
        mask = y_test == idx
        if mask.sum() > 0:
            class_f1 = f1_score(y_test == idx, y_pred == idx, zero_division=0)
            per_class[label_names[idx]] = float(class_f1)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class,
        "n_train_residues": len(X_train),
        "n_test_residues": len(X_test),
    }


# ── Unified Per-Residue Benchmark ─────────────────────────────


def run_per_residue_benchmarks(
    embeddings: dict[str, np.ndarray],
    data_dir: Path | str,
    name: str = "unnamed",
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict:
    """Run all available per-residue benchmarks on the given embeddings.

    Auto-detects which datasets are available in data_dir.

    Args:
        embeddings: {protein_id: (L, D) array} — original or compressed.
        data_dir: Directory containing benchmark datasets.
        name: Name for this configuration.
        train_fraction: Fraction of proteins to use for training probes.
        seed: Random seed for splits.

    Returns dict with results for each available task.
    """
    import random
    rng = random.Random(seed)
    data_dir = Path(data_dir)
    results = {"name": name}

    # SS3 (CB513)
    cb513_dir = data_dir / "cb513"
    cb513_data = load_cb513(cb513_dir)
    if cb513_data:
        avail_ids = [p["id"] for p in cb513_data if p["id"] in embeddings]
        rng_copy = random.Random(seed)
        rng_copy.shuffle(avail_ids)
        n_train = int(len(avail_ids) * train_fraction)
        train_ids = avail_ids[:n_train]
        test_ids = avail_ids[n_train:]

        ss3_labels = {p["id"]: p["ss3"] for p in cb513_data}
        print(f"  [{name}] SS3 probe: {len(train_ids)} train, {len(test_ids)} test proteins")
        results["ss3"] = evaluate_ss3_probe(embeddings, ss3_labels, train_ids, test_ids)
        print(f"  [{name}] SS3 Q3 = {results['ss3']['q3']:.3f}")
    else:
        print(f"  [{name}] CB513 data not found in {cb513_dir}, skipping SS3")

    # Disorder (CheZOD)
    chezod_dir = data_dir / "chezod"
    chezod_data = load_chezod(chezod_dir)
    if chezod_data:
        avail_ids = [p["id"] for p in chezod_data if p["id"] in embeddings]
        rng_copy = random.Random(seed)
        rng_copy.shuffle(avail_ids)
        n_train = int(len(avail_ids) * train_fraction)
        train_ids = avail_ids[:n_train]
        test_ids = avail_ids[n_train:]

        disorder_scores = {p["id"]: p["scores"] for p in chezod_data}
        print(f"  [{name}] Disorder probe: {len(train_ids)} train, {len(test_ids)} test")
        results["disorder"] = evaluate_disorder_probe(
            embeddings, disorder_scores, train_ids, test_ids
        )
        print(f"  [{name}] Disorder Spearman rho = {results['disorder']['spearman_rho']:.3f}")
    else:
        print(f"  [{name}] CheZOD data not found in {chezod_dir}, skipping disorder")

    # TM topology (TMbed)
    tmbed_dir = data_dir / "tmbed"
    tmbed_data = load_tmbed(tmbed_dir)
    if tmbed_data:
        avail_ids = [p["id"] for p in tmbed_data if p["id"] in embeddings]
        rng_copy = random.Random(seed)
        rng_copy.shuffle(avail_ids)
        n_train = int(len(avail_ids) * train_fraction)
        train_ids = avail_ids[:n_train]
        test_ids = avail_ids[n_train:]

        topology_labels = {p["id"]: p["topology"] for p in tmbed_data}
        print(f"  [{name}] TM probe: {len(train_ids)} train, {len(test_ids)} test")
        results["tm_topology"] = evaluate_tm_probe(
            embeddings, topology_labels, train_ids, test_ids
        )
        print(f"  [{name}] TM accuracy = {results['tm_topology']['accuracy']:.3f}")
    else:
        print(f"  [{name}] TMbed data not found in {tmbed_dir}, skipping TM topology")

    return results
