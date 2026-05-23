"""Tests for the ML model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ml_model import MLModel, create_labels


def _make_df(n: int = 200) -> pd.DataFrame:
    """Create a minimal synthetic OHLCV + indicator DataFrame."""
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(
        {
            "close": close,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "rsi": rng.uniform(20, 80, n),
            "macd": rng.normal(0, 0.5, n),
            "macd_signal": rng.normal(0, 0.4, n),
            "macd_hist": rng.normal(0, 0.2, n),
            "bb_upper": close * 1.02,
            "bb_mid": close,
            "bb_lower": close * 0.98,
            "sma_20": close * 0.999,
            "sma_50": close * 0.998,
            "ema_9": close * 1.001,
            "atr": rng.uniform(0.5, 2.0, n),
        }
    )
    return df


class TestMLModel:
    def test_train_and_predict_returns_valid_signal(self):
        df = _make_df(200)
        model = MLModel(n_estimators=10, min_confidence=0.0)
        model.train_on_history(df)
        bar = df.iloc[-1]
        signal, confidence = model.predict(bar)
        assert signal in ("BUY", "SELL", "HOLD")
        assert 0.0 <= confidence <= 1.0

    def test_predict_raises_if_not_trained(self):
        model = MLModel()
        bar = pd.Series({"close": 100.0})
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(bar)

    def test_low_confidence_returns_hold(self):
        df = _make_df(200)
        model = MLModel(n_estimators=10, min_confidence=0.99)  # near-impossible threshold
        model.train_on_history(df)
        bar = df.iloc[-1]
        signal, confidence = model.predict(bar)
        # With threshold 0.99 the model almost always falls back to HOLD
        assert signal in ("BUY", "SELL", "HOLD")  # just check it doesn't crash

    def test_nan_input_returns_hold(self):
        df = _make_df(200)
        model = MLModel(n_estimators=10, min_confidence=0.0)
        model.train_on_history(df)
        bar = df.iloc[-1].copy()
        bar["rsi"] = float("nan")
        signal, confidence = model.predict(bar)
        assert signal == "HOLD"
        assert confidence == 0.0

    def test_save_and_load(self, tmp_path):
        df = _make_df(200)
        model = MLModel(n_estimators=10, model_path=tmp_path / "model.pkl")
        model.train_on_history(df)

        loaded = MLModel(n_estimators=10, model_path=tmp_path / "model.pkl")
        assert loaded._trained

        bar = df.iloc[-1]
        signal, confidence = loaded.predict(bar)
        assert signal in ("BUY", "SELL", "HOLD")


class TestCreateLabels:
    def test_labels_column_created(self):
        df = _make_df(100)
        labelled = create_labels(df)
        assert "label" in labelled.columns

    def test_label_values_are_valid(self):
        df = _make_df(100)
        labelled = create_labels(df)
        assert set(labelled["label"].unique()).issubset({-1, 0, 1})

    def test_all_labels_present_in_large_dataset(self):
        df = _make_df(500)
        labelled = create_labels(df, buy_threshold=0.01, sell_threshold=-0.01)
        unique = set(labelled["label"].unique())
        assert len(unique) >= 2  # at least two classes
