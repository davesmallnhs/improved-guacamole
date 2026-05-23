"""ML-based trading signal model.

Uses a Random Forest classifier trained on technical indicators to produce
BUY / SELL / HOLD signals with a confidence score.

Training
--------
Call ``MLModel.train(df)`` with a labelled DataFrame (feature columns + a
``"label"`` column where 1=BUY, -1=SELL, 0=HOLD).

A convenience helper ``create_labels`` is provided to auto-label historical
data using a simple forward-return threshold, useful for bootstrapping.

Inference
---------
Call ``model.predict(bar)`` with a ``pd.Series`` of the latest bar (the same
columns produced by ``MarketDataFetcher.fetch()``).  Returns a
``(signal, confidence)`` tuple where *signal* is one of ``"BUY"``,
``"SELL"``, ``"HOLD"``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature columns expected by the model
_FEATURE_COLS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "sma_20",
    "sma_50",
    "ema_9",
    "atr",
    "close",
    "volume",
]

_LABEL_MAP = {1: "BUY", -1: "SELL", 0: "HOLD"}
_REVERSE_LABEL_MAP = {"BUY": 1, "SELL": -1, "HOLD": 0}


class MLModel:
    """Random Forest trading signal classifier.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    min_confidence:
        Minimum predicted probability for a BUY or SELL signal to be
        returned; otherwise ``"HOLD"`` is emitted.
    model_path:
        If provided the trained model is saved to / loaded from this path.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_confidence: float = 0.55,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._model_path = Path(model_path) if model_path else None
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._trained = False

        if self._model_path and self._model_path.exists():
            self.load(self._model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """Fit the classifier on a labelled DataFrame.

        Parameters
        ----------
        df:
            Must contain feature columns (see ``_FEATURE_COLS``) and a
            ``"label"`` column with integer values 1 / 0 / -1.
        """
        available = [c for c in _FEATURE_COLS if c in df.columns]
        if len(available) < 3:
            raise ValueError(
                "DataFrame must contain at least 3 of the expected feature columns"
            )

        df_clean = df[available + ["label"]].dropna()
        X = df_clean[available].values
        y = df_clean["label"].values

        X_scaled = self._scaler.fit_transform(X)
        self._clf.fit(X_scaled, y)
        self._trained = True
        self._feature_cols = available
        logger.info(
            "MLModel trained on %d samples (%d features)",
            len(df_clean),
            len(available),
        )

        if self._model_path:
            self.save(self._model_path)

    def train_on_history(
        self,
        df: pd.DataFrame,
        forward_return_days: int = 5,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
    ) -> None:
        """Auto-label *df* using forward returns, then train.

        Parameters
        ----------
        df:
            OHLCV + indicator DataFrame from ``MarketDataFetcher``.
        forward_return_days:
            Number of bars to look ahead when computing returns.
        buy_threshold:
            Forward return above which a bar is labelled BUY.
        sell_threshold:
            Forward return below which a bar is labelled SELL.
        """
        labelled = create_labels(df, forward_return_days, buy_threshold, sell_threshold)
        self.train(labelled)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, bar: pd.Series) -> tuple[str, float]:
        """Predict a trading signal for the given bar.

        Parameters
        ----------
        bar:
            A ``pd.Series`` produced by ``MarketDataFetcher.fetch_latest_bar()``.

        Returns
        -------
        tuple[str, float]
            ``(signal, confidence)`` where *signal* is ``"BUY"``, ``"SELL"``,
            or ``"HOLD"`` and *confidence* is in ``[0, 1]``.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet — call train() first")

        available = [c for c in self._feature_cols if c in bar.index]
        x = np.array([[bar.get(c, np.nan) for c in available]])

        if np.isnan(x).any():
            logger.warning("NaN values in input features — returning HOLD")
            return "HOLD", 0.0

        x_scaled = self._scaler.transform(x)
        proba = self._clf.predict_proba(x_scaled)[0]
        classes = self._clf.classes_

        best_idx = int(np.argmax(proba))
        best_class = int(classes[best_idx])
        confidence = float(proba[best_idx])

        signal = _LABEL_MAP.get(best_class, "HOLD")

        # Suppress weak signals
        if signal != "HOLD" and confidence < self._min_confidence:
            logger.debug(
                "Signal %s suppressed (confidence %.2f < %.2f)",
                signal,
                confidence,
                self._min_confidence,
            )
            signal = "HOLD"

        logger.info("MLModel signal: %s (confidence=%.3f)", signal, confidence)
        return signal, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the trained classifier and scaler to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "clf": self._clf,
                    "scaler": self._scaler,
                    "feature_cols": getattr(self, "_feature_cols", _FEATURE_COLS),
                    "min_confidence": self._min_confidence,
                },
                fh,
            )
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from *path*."""
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        self._clf = data["clf"]
        self._scaler = data["scaler"]
        self._feature_cols = data.get("feature_cols", _FEATURE_COLS)
        self._min_confidence = data.get("min_confidence", self._min_confidence)
        self._trained = True
        logger.info("Model loaded from %s", path)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def create_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pd.DataFrame:
    """Add a ``"label"`` column to *df* based on *n*-day forward returns.

    Labels: ``1`` = BUY, ``-1`` = SELL, ``0`` = HOLD.
    """
    df = df.copy()
    fwd_return = df["close"].shift(-forward_days) / df["close"] - 1
    df["label"] = 0
    df.loc[fwd_return >= buy_threshold, "label"] = 1
    df.loc[fwd_return <= sell_threshold, "label"] = -1
    return df.dropna(subset=["label"])
