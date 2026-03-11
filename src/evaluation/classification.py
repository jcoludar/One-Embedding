"""Linear probe classification benchmark."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.compressors.base import SequenceCompressor
from src.utils.device import get_device


def evaluate_linear_probe(
    model: SequenceCompressor | None,
    embeddings: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
    n_folds: int = 5,
    device=None,
    train_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    pooling_strategy: str = "mean",
) -> dict[str, float]:
    """Train a linear classifier on frozen latent vectors and evaluate accuracy.

    Args:
        train_ids: If provided (with test_ids), train on these and evaluate on test_ids.
            No cross-validation is used in this mode.
        test_ids: If provided (with train_ids), evaluate on these.
        pooling_strategy: Pooling strategy for get_pooled().

    Returns dict with: accuracy_mean, accuracy_std, n_classes.
    """
    import torch
    from sklearn.metrics import accuracy_score

    if device is None:
        device = get_device()

    # Get pooled vectors
    vectors = {}
    if model is None:
        for pid, emb in embeddings.items():
            vectors[pid] = emb.mean(axis=0)
    else:
        model = model.to(device)
        model.eval()
        max_len = 512
        with torch.no_grad():
            for pid, emb in embeddings.items():
                L = min(emb.shape[0], max_len)
                states = torch.from_numpy(emb[:L]).unsqueeze(0).to(device)
                mask = torch.ones(1, L, device=device)
                latent = model.compress(states, mask)
                # For channel compressors (num_tokens=-1), pass mask for aware pooling
                if model.num_tokens == -1:
                    pooled = model.get_pooled(latent, strategy=pooling_strategy, mask=mask)
                else:
                    pooled = model.get_pooled(latent, strategy=pooling_strategy)
                vectors[pid] = pooled[0].cpu().numpy()

    # Build label mapping
    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}

    # Held-out evaluation mode: train on train_ids, test on test_ids
    if train_ids is not None and test_ids is not None:
        tr_ids = [pid for pid in train_ids if pid in vectors and pid in id_to_label]
        te_ids = [pid for pid in test_ids if pid in vectors and pid in id_to_label]

        if len(tr_ids) < 5 or len(te_ids) < 5:
            return {"accuracy_mean": 0.0, "accuracy_std": 0.0, "n_classes": 0}

        X_train = np.array([vectors[pid] for pid in tr_ids])
        y_train_labels = [id_to_label[pid] for pid in tr_ids]
        X_test = np.array([vectors[pid] for pid in te_ids])
        y_test_labels = [id_to_label[pid] for pid in te_ids]

        le = LabelEncoder()
        le.fit(y_train_labels + y_test_labels)
        y_train = le.transform(y_train_labels)
        y_test = le.transform(y_test_labels)
        n_classes = len(le.classes_)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return {
            "accuracy_mean": float(acc),
            "accuracy_std": 0.0,
            "n_classes": n_classes,
            "n_train": len(tr_ids),
            "n_test": len(te_ids),
            "mode": "held_out",
        }

    # Legacy cross-validation mode (backward compatible)
    ids = [pid for pid in vectors if pid in id_to_label]

    if len(ids) < 10:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0, "n_classes": 0}

    X = np.array([vectors[pid] for pid in ids])
    labels = [id_to_label[pid] for pid in ids]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Need at least n_folds samples per class for stratified CV
    min_samples = min(np.bincount(y))
    actual_folds = min(n_folds, min_samples) if min_samples >= 2 else 2

    if actual_folds < 2:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0, "n_classes": n_classes}

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_classes": n_classes,
        "n_samples": len(ids),
        "mode": "cross_validation",
    }
