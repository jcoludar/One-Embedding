"""Per-residue prediction probes on compressed embeddings.

Three probes that operate on decoded (L, 512) per-residue embeddings:
1. DisorderProbe — predicts continuous disorder Z-scores (SETH/UdonPred-style)
2. TopologyProbe — predicts TM topology classes (TMbed-style)
3. BindingProbe — predicts binding site residues (bindEmbed21-style)

All probes are deliberately lightweight (<50K params). The embedding
quality is the bottleneck, not the probe complexity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ── Disorder Probe (SETH/UdonPred architecture) ──────────────

class DisorderProbe:
    """2-layer CNN predicting per-residue disorder from embeddings.

    Architecture: Conv(512→28, k=5) + Tanh + Conv(28→1, k=5)
    Output: continuous Z-scores (< 8 = disordered, > 8 = ordered)

    Based on SETH (Frontiers Bioinf 2022) and UdonPred (bioRxiv 2026).
    """

    def __init__(self, input_dim: int = 512, hidden: int = 28, kernel: int = 5):
        self.input_dim = input_dim
        self.hidden = hidden
        self.kernel = kernel
        self.pad = kernel // 2
        # Weights: Conv1 (input_dim, hidden, kernel) + bias + Conv2 (hidden, 1, kernel) + bias
        self.w1 = None  # (hidden, input_dim, kernel)
        self.b1 = None  # (hidden,)
        self.w2 = None  # (1, hidden, kernel)
        self.b2 = None  # (1,)

    def _init_weights(self, seed: int = 42):
        """Initialize weights with Kaiming normal."""
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / (self.input_dim * self.kernel))
        self.w1 = rng.randn(self.hidden, self.input_dim, self.kernel).astype(np.float32) * scale1
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        scale2 = np.sqrt(2.0 / (self.hidden * self.kernel))
        self.w2 = rng.randn(1, self.hidden, self.kernel).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict disorder Z-scores.

        Args:
            embeddings: (L, D) per-residue embeddings

        Returns:
            (L,) Z-scores (lower = more disordered)
        """
        if self.w1 is None:
            raise ValueError("Probe not fitted. Call fit() or load().")

        L, D = embeddings.shape
        # Pad sequence
        x = np.pad(embeddings, ((self.pad, self.pad), (0, 0)), mode='constant')

        # Conv1: (L, D) -> (L, hidden)
        out1 = np.zeros((L, self.hidden), dtype=np.float32)
        for i in range(L):
            window = x[i:i+self.kernel]  # (kernel, D)
            for h in range(self.hidden):
                out1[i, h] = np.sum(window * self.w1[h].T) + self.b1[h]

        # Tanh activation
        out1 = np.tanh(out1)

        # Pad for conv2
        out1_padded = np.pad(out1, ((self.pad, self.pad), (0, 0)), mode='constant')

        # Conv2: (L, hidden) -> (L, 1)
        out2 = np.zeros(L, dtype=np.float32)
        for i in range(L):
            window = out1_padded[i:i+self.kernel]  # (kernel, hidden)
            out2[i] = np.sum(window * self.w2[0].T) + self.b2[0]

        return out2

    def predict_binary(self, embeddings: np.ndarray, threshold: float = 8.0) -> np.ndarray:
        """Predict binary disorder (1 = disordered, 0 = ordered)."""
        zscores = self.predict(embeddings)
        return (zscores < threshold).astype(np.int8)

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray],
            lr: float = 0.001, epochs: int = 50, seed: int = 42):
        """Train the probe using simple gradient descent.

        Args:
            X_list: list of (L_i, D) embedding arrays
            y_list: list of (L_i,) Z-score arrays
            lr: learning rate
            epochs: number of training epochs
            seed: random seed for weight initialization
        """
        self._init_weights(seed)

        for epoch in range(epochs):
            total_loss = 0.0
            n_samples = 0
            for X, y in zip(X_list, y_list):
                pred = self.predict(X)
                # MSE loss gradient
                error = pred - y
                total_loss += np.sum(error ** 2)
                n_samples += len(y)

                # Simple weight update (approximate gradient)
                L = len(y)
                x_padded = np.pad(X, ((self.pad, self.pad), (0, 0)), mode='constant')

                # Backward through conv2
                out1 = np.zeros((L, self.hidden), dtype=np.float32)
                for i in range(L):
                    window = x_padded[i:i+self.kernel]
                    for h in range(self.hidden):
                        out1[i, h] = np.sum(window * self.w1[h].T) + self.b1[h]
                out1_act = np.tanh(out1)
                out1_padded = np.pad(out1_act, ((self.pad, self.pad), (0, 0)), mode='constant')

                # Gradient for w2, b2
                for i in range(L):
                    window = out1_padded[i:i+self.kernel]  # (kernel, hidden)
                    self.w2[0] -= lr * error[i] * window.T / L
                self.b2[0] -= lr * np.mean(error)

                # Gradient for w1, b1 (chain rule through tanh)
                d_out1 = np.zeros((L, self.hidden), dtype=np.float32)
                for i in range(L):
                    for h in range(self.hidden):
                        for k in range(self.kernel):
                            if i + k < L:
                                d_out1[i, h] += error[min(i+k, L-1)] * self.w2[0, h, k]
                d_out1 *= (1 - out1_act ** 2)  # tanh derivative

                for i in range(L):
                    window = x_padded[i:i+self.kernel]
                    for h in range(self.hidden):
                        self.w1[h] -= lr * d_out1[i, h] * window.T / L
                self.b1 -= lr * np.mean(d_out1, axis=0)

            if (epoch + 1) % 10 == 0:
                mse = total_loss / n_samples
                print(f"  Epoch {epoch+1}/{epochs}: MSE={mse:.4f}")

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 input_dim=self.input_dim, hidden=self.hidden, kernel=self.kernel)

    @classmethod
    def load(cls, path: str) -> "DisorderProbe":
        data = np.load(path)
        obj = cls(int(data["input_dim"]), int(data["hidden"]), int(data["kernel"]))
        obj.w1, obj.b1, obj.w2, obj.b2 = data["w1"], data["b1"], data["w2"], data["b2"]
        return obj


# ── Topology Probe (TMbed-style parallel convolution) ────────

class TopologyProbe:
    """Parallel-kernel CNN predicting TM topology from embeddings.

    Architecture: Conv(512→64, k=1) + parallel [DepthwiseConv(k=9), DepthwiseConv(k=21)]
                  → concat(64+64+64) → Conv(192→n_classes)
    Output: per-residue class logits (inside, outside, TM-helix, TM-beta, signal)

    Based on TMbed (BMC Bioinformatics 2022).
    """

    # TMbed topology labels
    CLASSES = ["i", "o", "H", "B", "S"]  # inside, outside, helix, beta, signal

    def __init__(self, input_dim: int = 512, hidden: int = 64, n_classes: int = 5):
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_classes = n_classes
        self.w_proj = None  # (hidden, input_dim) pointwise projection
        self.b_proj = None
        self.w_out = None   # (n_classes, hidden*3) output projection
        self.b_out = None

    def _init_weights(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / self.input_dim)
        self.w_proj = rng.randn(self.hidden, self.input_dim).astype(np.float32) * scale
        self.b_proj = np.zeros(self.hidden, dtype=np.float32)
        scale_out = np.sqrt(2.0 / (self.hidden * 3))
        self.w_out = rng.randn(self.n_classes, self.hidden * 3).astype(np.float32) * scale_out
        self.b_out = np.zeros(self.n_classes, dtype=np.float32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict topology class logits.

        Args:
            embeddings: (L, D) per-residue embeddings

        Returns:
            (L, n_classes) class logits
        """
        if self.w_proj is None:
            raise ValueError("Probe not fitted. Call fit() or load().")

        L = embeddings.shape[0]

        # Pointwise projection: (L, D) → (L, hidden)
        proj = np.maximum(0, embeddings @ self.w_proj.T + self.b_proj)  # ReLU

        # Depthwise conv k=9 (mean pool over window)
        pad9 = 4
        proj_pad9 = np.pad(proj, ((pad9, pad9), (0, 0)), mode='constant')
        dw9 = np.zeros_like(proj)
        for i in range(L):
            dw9[i] = proj_pad9[i:i+9].mean(axis=0)

        # Depthwise conv k=21 (mean pool over window)
        pad21 = 10
        proj_pad21 = np.pad(proj, ((pad21, pad21), (0, 0)), mode='constant')
        dw21 = np.zeros_like(proj)
        for i in range(L):
            dw21[i] = proj_pad21[i:i+21].mean(axis=0)

        # Concat: (L, hidden*3)
        concat = np.concatenate([proj, dw9, dw21], axis=1)

        # Output projection: (L, hidden*3) → (L, n_classes)
        logits = concat @ self.w_out.T + self.b_out

        return logits

    def predict_labels(self, embeddings: np.ndarray) -> str:
        """Predict topology label string."""
        logits = self.predict(embeddings)
        indices = np.argmax(logits, axis=1)
        return "".join(self.CLASSES[i] for i in indices)

    def save(self, path: str):
        np.savez(path, w_proj=self.w_proj, b_proj=self.b_proj,
                 w_out=self.w_out, b_out=self.b_out,
                 input_dim=self.input_dim, hidden=self.hidden, n_classes=self.n_classes)

    @classmethod
    def load(cls, path: str) -> "TopologyProbe":
        data = np.load(path)
        obj = cls(int(data["input_dim"]), int(data["hidden"]), int(data["n_classes"]))
        obj.w_proj, obj.b_proj = data["w_proj"], data["b_proj"]
        obj.w_out, obj.b_out = data["w_out"], data["b_out"]
        return obj


# ── Binding Site Probe (bindEmbed21-style CNN) ───────────────

class BindingProbe:
    """CNN predicting per-residue binding sites from embeddings.

    Architecture: Linear(512→128) + ELU + Linear(128→3)
    Output: per-residue probabilities for metal, nucleic acid, small molecule binding

    Simplified from bindEmbed21DL (Rostlab, Scientific Reports 2021).
    Uses linear layers instead of Conv1d for numpy-only implementation.
    """

    BINDING_TYPES = ["metal", "nucleic", "small_molecule"]

    def __init__(self, input_dim: int = 512, hidden: int = 128, n_types: int = 3):
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_types = n_types
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def _init_weights(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / self.input_dim)
        self.w1 = rng.randn(self.hidden, self.input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        scale2 = np.sqrt(2.0 / self.hidden)
        self.w2 = rng.randn(self.n_types, self.hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(self.n_types, dtype=np.float32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict binding probabilities.

        Args:
            embeddings: (L, D) per-residue embeddings

        Returns:
            (L, 3) probabilities for [metal, nucleic, small_molecule]
        """
        if self.w1 is None:
            raise ValueError("Probe not fitted. Call fit() or load().")

        # Layer 1: Linear + ELU
        h = embeddings @ self.w1.T + self.b1
        h = np.where(h > 0, h, np.exp(h) - 1)  # ELU

        # Layer 2: Linear + Sigmoid
        logits = h @ self.w2.T + self.b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))  # Sigmoid

        return probs

    def predict_binary(self, embeddings: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary binding (1 = binding, 0 = not)."""
        probs = self.predict(embeddings)
        return (probs > threshold).astype(np.int8)

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 input_dim=self.input_dim, hidden=self.hidden, n_types=self.n_types)

    @classmethod
    def load(cls, path: str) -> "BindingProbe":
        data = np.load(path)
        obj = cls(int(data["input_dim"]), int(data["hidden"]), int(data["n_types"]))
        obj.w1, obj.b1, obj.w2, obj.b2 = data["w1"], data["b1"], data["w2"], data["b2"]
        return obj
