"""LSTM-based time-series predictor for direction classification + magnitude regression.

Designed for CPU training with small model size for fast iteration.
Does NOT inherit ModelWrapper to keep it simple — provides the same interface
(fit, predict_proba, predict_magnitude) but operates on sliding windows.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helper: create sliding-window sequences
# ---------------------------------------------------------------------------


def create_sequences(
    X_np: np.ndarray,
    y_np: np.ndarray | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert (N, F) flat array into (N-seq_len+1, seq_len, F) windows.

    Parameters
    ----------
    X_np : array of shape (N, n_features)
    y_np : array of shape (N,) or None
    seq_len : window length

    Returns
    -------
    X_seq : array of shape (N-seq_len+1, seq_len, n_features)
    y_seq : array of shape (N-seq_len+1,) — label for the *last* row of each window
    """
    N = len(X_np)
    if N < seq_len:
        raise ValueError(f"Not enough rows ({N}) for seq_len={seq_len}")

    n_windows = N - seq_len + 1
    n_features = X_np.shape[1]

    X_seq = np.empty((n_windows, seq_len, n_features), dtype=np.float32)
    for i in range(n_windows):
        X_seq[i] = X_np[i : i + seq_len]

    y_seq = None
    if y_np is not None:
        y_seq = y_np[seq_len - 1 :].astype(np.float32)

    return X_seq, y_seq


# ---------------------------------------------------------------------------
# PyTorch Module
# ---------------------------------------------------------------------------


class LSTMPredictor(nn.Module):
    """LSTM + Linear head for binary classification (direction)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        dir_proba : (batch,) — probability of direction=1
        mag_pred  : (batch,) — predicted magnitude
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        dir_proba = self.classifier(last_hidden).squeeze(-1)
        mag_pred = self.regressor(last_hidden).squeeze(-1)
        return dir_proba, mag_pred


# ---------------------------------------------------------------------------
# Pipeline wrapper (same interface as LGBMPipeline)
# ---------------------------------------------------------------------------


class LSTMPipeline:
    """LSTM pipeline with fit / predict_proba / predict_magnitude interface.

    Accepts flat DataFrames (like LGBMPipeline) and internally converts them
    to sliding windows of length ``seq_len``.
    """

    def __init__(
        self,
        n_features: int | None = None,
        seq_len: int = 20,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 7,
    ):
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.model: LSTMPredictor | None = None
        self.feature_names: list[str] = []
        self.version: str = ""

        # Feature normalization stats (per-feature mean/std from training)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ---- helpers -----------------------------------------------------------

    def _normalize(self, X_np: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score normalization (fit on training data only)."""
        if fit:
            self._mean = np.nanmean(X_np, axis=0)
            self._std = np.nanstd(X_np, axis=0)
            self._std[self._std < 1e-8] = 1.0  # avoid division by zero
        X_norm = (X_np - self._mean) / self._std
        # Replace any remaining NaN/inf with 0
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return X_norm

    def _to_numpy(self, X: pl.DataFrame) -> np.ndarray:
        """Convert Polars DataFrame to float32 numpy, handling inf/NaN."""
        X_clean = X.with_columns(
            [
                pl.when(pl.col(c).is_infinite())
                .then(None)
                .otherwise(pl.col(c))
                .fill_null(0.0)
                .alias(c)
                for c in X.columns
            ]
        )
        return X_clean.to_numpy().astype(np.float32)

    # ---- fit ---------------------------------------------------------------

    def fit(
        self,
        X: pl.DataFrame,
        y_direction: pl.Series,
        y_magnitude: pl.Series,
    ) -> None:
        """Train LSTM on sliding windows with early stopping."""
        self.feature_names = list(X.columns)
        n_feat = len(self.feature_names)
        self.n_features = n_feat

        X_np = self._to_numpy(X)
        y_dir_np = y_direction.to_numpy().astype(np.float32)
        y_mag_np = y_magnitude.to_numpy().astype(np.float32)

        # Normalize features
        X_np = self._normalize(X_np, fit=True)

        # Create sequences
        X_seq, y_dir_seq = create_sequences(X_np, y_dir_np, self.seq_len)
        _, y_mag_seq = create_sequences(X_np, y_mag_np, self.seq_len)

        # Train / validation split (last 20% for early stopping)
        n_total = len(X_seq)
        split_idx = int(n_total * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_dir_train, y_dir_val = y_dir_seq[:split_idx], y_dir_seq[split_idx:]
        y_mag_train, y_mag_val = y_mag_seq[:split_idx], y_mag_seq[split_idx:]

        # DataLoaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_dir_train),
            torch.from_numpy(y_mag_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_dir_val),
            torch.from_numpy(y_mag_val),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        self.model = LSTMPredictor(
            n_features=n_feat,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        logger.info(
            "lstm_training_start",
            n_train=len(X_train),
            n_val=len(X_val),
            n_features=n_feat,
            seq_len=self.seq_len,
            epochs=self.epochs,
        )

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for X_b, y_dir_b, y_mag_b in train_loader:
                optimizer.zero_grad()
                dir_pred, mag_pred = self.model(X_b)
                loss = bce_loss(dir_pred, y_dir_b) + 0.5 * mse_loss(mag_pred, y_mag_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for X_b, y_dir_b, y_mag_b in val_loader:
                    dir_pred, mag_pred = self.model(X_b)
                    loss = bce_loss(dir_pred, y_dir_b) + 0.5 * mse_loss(mag_pred, y_mag_b)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logger.info(
                    "lstm_early_stopped",
                    epoch=epoch + 1,
                    best_val_loss=round(best_val_loss, 5),
                )
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.version = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(
            "lstm_training_complete",
            version=self.version,
            best_val_loss=round(best_val_loss, 5),
            epochs_run=epoch + 1,
        )

    # ---- predict -----------------------------------------------------------

    def _predict_raw(self, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Internal: get raw direction probas and magnitude predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        X_np = self._to_numpy(X)
        X_np = self._normalize(X_np, fit=False)

        # Need at least seq_len rows
        if len(X_np) < self.seq_len:
            # Pad with zeros at the start
            pad = np.zeros((self.seq_len - len(X_np), X_np.shape[1]), dtype=np.float32)
            X_np = np.vstack([pad, X_np])

        X_seq, _ = create_sequences(X_np, None, self.seq_len)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq)
            dir_proba, mag_pred = self.model(X_t)

        dir_np = dir_proba.numpy()
        mag_np = mag_pred.numpy()

        # The sequences produce (N - seq_len + 1) outputs.
        # We need to return exactly X.height values.
        # For the first (seq_len - 1) rows, use the first prediction.
        original_len = X.height
        n_seq = len(dir_np)
        if n_seq < original_len:
            pad_len = original_len - n_seq
            dir_np = np.concatenate([np.full(pad_len, dir_np[0]), dir_np])
            mag_np = np.concatenate([np.full(pad_len, mag_np[0]), mag_np])

        return dir_np, mag_np

    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        """Return probability of direction=1 for each row."""
        dir_proba, _ = self._predict_raw(X)
        return dir_proba.tolist()

    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        """Return predicted magnitude for each row."""
        _, mag_pred = self._predict_raw(X)
        return mag_pred.tolist()

    def get_params(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
