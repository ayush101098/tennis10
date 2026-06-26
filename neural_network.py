"""
Symmetric Tennis Neural Network — v3.0
Key changes from v2:
  - Bias terms RESTORED (symmetry enforced by data augmentation, not architecture)
  - Deeper hidden layer option (100 or 128 units)
  - Dropout regularisation for better generalisation
  - Ensemble of 20 bootstrap replicates (bagging)
  - Brier-score early stopping in addition to accuracy
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TennisNet:
    """
    Single feedforward neural network for tennis match prediction.
    Architecture: Input(N) → Dense(hidden, tanh) → Dense(1, sigmoid)

    Symmetry is enforced by training on both (X, y=1) and (-X, y=0)
    for every match, NOT by removing bias terms. This allows the model
    to learn player-independent biases (e.g. higher-ranked players
    win more often even with equal serve stats) while preserving the
    anti-symmetry property at the feature level.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 100,
                 lr: float = 0.0004, momentum: float = 0.55,
                 weight_decay: float = 0.002, dropout: float = 0.10):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.momentum   = momentum
        self.weight_decay = weight_decay
        self.dropout    = dropout

        # Xavier initialisation — WITH bias terms
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))

        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, 1))
        self.b2 = np.zeros(1)

        # SGD with momentum — velocity terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    # ── forward / backward ────────────────────────────────────────────────────

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _tanh(self, x):
        return np.tanh(x)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._tanh(self.z1)

        # Dropout during training
        if training and self.dropout > 0:
            self._mask = (np.random.rand(*self.a1.shape) > self.dropout).astype(float)
            self.a1 = self.a1 * self._mask / (1.0 - self.dropout)
        else:
            self._mask = None

        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self._sigmoid(self.z2)
        return self.out

    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        n = len(X)
        y = y.reshape(-1, 1)
        pred = self.out

        # Binary cross-entropy gradient
        dz2 = (pred - y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        if self._mask is not None:
            da1 = da1 * self._mask / (1.0 - self.dropout)
        dz1 = da1 * (1.0 - self.a1**2)  # tanh derivative
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # Weight decay (L2 regularisation)
        dW1 += self.weight_decay * self.W1
        dW2 += self.weight_decay * self.W2

        # SGD with momentum
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2

        self.W1 += self.vW1; self.b1 += self.vb1
        self.W2 += self.vW2; self.b2 += self.vb2

        loss = float(-np.mean(y * np.log(pred + 1e-9) + (1-y) * np.log(1-pred + 1e-9)))
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False).flatten()

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            max_epochs: int = 50, patience: int = 10,
            batch_size: int = 64) -> List[float]:
        """
        Train with Brier-score early stopping on validation set.
        Data augmentation: add flipped versions to enforce symmetry.
        """
        # Symmetry augmentation: (−X, 1−y) pairs
        X_aug = np.vstack([X, -X])
        y_aug = np.concatenate([y, 1.0 - y])

        best_brier = float('inf')
        best_weights = self._get_weights()
        no_improve = 0
        history = []

        n = len(X_aug)
        for epoch in range(max_epochs):
            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X_aug[idx], y_aug[idx]

            # Mini-batch SGD
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                Xb = X_s[start:start+batch_size]
                yb = y_s[start:start+batch_size]
                self.forward(Xb, training=True)
                epoch_loss += self.backward(Xb, yb)

            history.append(epoch_loss)

            # Validation Brier score
            if X_val is not None and y_val is not None:
                pv  = self.predict_proba(X_val)
                brier = float(np.mean((pv - y_val)**2))
                if brier < best_brier - 1e-5:
                    best_brier = brier
                    best_weights = self._get_weights()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.debug(f"Early stop at epoch {epoch+1}, Brier={best_brier:.5f}")
                    break

        if X_val is not None:
            self._set_weights(best_weights)
        return history

    def _get_weights(self):
        return (self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy())

    def _set_weights(self, w):
        self.W1, self.b1, self.W2, self.b2 = [x.copy() for x in w]


# ── Bagged Ensemble ────────────────────────────────────────────────────────────

class SymmetricTennisNet:
    """
    Ensemble of N bootstrap-trained TennisNet models.
    Final prediction: mean of all individual predictions.
    """

    def __init__(self, n_models: int = 20, input_dim: int = 19,
                 hidden_dim: int = 100, **net_kwargs):
        self.n_models  = n_models
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net_kwargs = net_kwargs
        self.models: List[TennisNet] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            max_epochs: int = 50, patience: int = 10) -> None:
        self.models = []
        n = len(X)
        for i in range(self.n_models):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            Xb, yb = X[idx], y[idx]

            net = TennisNet(self.input_dim, self.hidden_dim, **self.net_kwargs)
            net.fit(Xb, yb, X_val, y_val, max_epochs=max_epochs, patience=patience)
            self.models.append(net)
            logger.info(f"  Model {i+1}/{self.n_models} trained")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model not fitted — call fit() first")
        preds = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'SymmetricTennisNet':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Feature normalisation ──────────────────────────────────────────────────────

class FeatureScaler:
    """
    Zero-mean unit-variance scaler fitted on training data only.
    Prevents test-set data leakage in normalisation.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        with open(path, 'rb') as f:
            return pickle.load(f)
