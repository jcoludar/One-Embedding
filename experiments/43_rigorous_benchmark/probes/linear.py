"""CV-tuned linear probes for per-residue evaluation (Rule 8).

LogisticRegression for classification (SS3, SS8, TM topology).
Ridge for regression (disorder).
C and alpha selected via k-fold CV on training set only.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


def train_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: list[float] = None,
    cv_folds: int = 3,
    seed: int = 42,
) -> dict:
    """Train LogReg with CV-tuned C, return predictions + metrics.

    Args:
        X_train: (N_train, D) residue-level embeddings.
        y_train: (N_train,) class labels.
        X_test: (N_test, D) residue-level embeddings.
        y_test: (N_test,) class labels.
        C_grid: Regularization values to search (default: [0.01, 0.1, 1.0, 10.0]).
        cv_folds: Number of CV folds for hyperparameter selection.
        seed: Random seed.

    Returns:
        dict with: predictions, best_C, accuracy, macro_f1, weighted_f1,
                   per_class_acc, n_train, n_test.
    """
    if C_grid is None:
        C_grid = [0.01, 0.1, 1.0, 10.0]

    base_model = LogisticRegression(
        max_iter=500, solver="lbfgs", random_state=seed,
    )
    grid = GridSearchCV(
        base_model, param_grid={"C": C_grid},
        cv=cv_folds, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_C = grid.best_params_["C"]
    predictions = grid.predict(X_test)

    classes = np.unique(np.concatenate([y_train, y_test]))
    per_class_acc = {}
    for c in classes:
        mask = y_test == c
        if mask.sum() > 0:
            per_class_acc[str(c)] = float((predictions[mask] == c).mean())

    return {
        "predictions": predictions,
        "best_C": best_C,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "per_class_acc": per_class_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def train_regression_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: list[float] = None,
    cv_folds: int = 3,
    seed: int = 42,
) -> dict:
    """Train Ridge with CV-tuned alpha, return predictions + metrics.

    Args:
        X_train: (N_train, D) residue-level embeddings.
        y_train: (N_train,) continuous target values.
        X_test: (N_test, D) residue-level embeddings.
        y_test: (N_test,) continuous target values.
        alpha_grid: Regularization values to search.
        cv_folds: Number of CV folds.
        seed: Random seed (unused by Ridge, kept for interface consistency).

    Returns:
        dict with: predictions, best_alpha, spearman_rho, p_value, mse,
                   n_train, n_test.
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    model = RidgeCV(alphas=alpha_grid, cv=cv_folds)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rho, p_value = spearmanr(y_test, predictions)
    mse = float(np.mean((y_test - predictions) ** 2))

    return {
        "predictions": predictions,
        "best_alpha": float(model.alpha_),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "mse": mse,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
