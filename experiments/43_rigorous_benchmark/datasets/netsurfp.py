"""NetSurfP cross-validation datasets: TS115 and CASP12.

Same CSV format as CB513 (input, dssp3, dssp8, disorder, cb513_mask).
These are independent test sets for cross-validating SS3/SS8 results.
"""
import csv
from pathlib import Path


def load_netsurfp_csv(csv_path):
    """Load TS115 or CASP12 CSV. Same format as CB513.

    Returns:
        (sequences, ss3_labels, ss8_labels) as dicts keyed by protein_id.
    """
    csv_path = Path(csv_path)
    sequences, ss3_labels, ss8_labels = {}, {}, {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pid = f"{csv_path.stem}_{i}"
            sequences[pid] = row["input"]
            ss3_labels[pid] = row["dssp3"]
            ss8_labels[pid] = row["dssp8"]
    return sequences, ss3_labels, ss8_labels
