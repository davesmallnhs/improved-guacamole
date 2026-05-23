"""LLM-based trading signal model.

Sends a structured prompt to an OpenAI chat model that contains:
- Recent OHLCV data and key indicator values for the instrument
- A request for a BUY / SELL / HOLD recommendation with reasoning

The model returns ``(signal, confidence, reasoning)`` where *reasoning* is the
full LLM explanation logged for auditing.

Requirements
------------
Set ``OPENAI_API_KEY`` in your ``.env`` file.  The ``openai`` Python package
must be installed (``pip install openai``).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LLMModel:
    """OpenAI-powered trading signal generator.

    Parameters
    ----------
    model_name:
        OpenAI model to use (e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``).
    min_confidence:
        Minimum self-reported LLM confidence (0-1) to pass through a
        BUY/SELL signal; lower values are returned as HOLD.
    temperature:
        Sampling temperature — lower = more deterministic.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        min_confidence: float = 0.6,
        temperature: float = 0.2,
    ) -> None:
        self._model_name = model_name
        self._min_confidence = min_confidence
        self._temperature = temperature
        self._client: Optional[object] = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI()
            logger.info("LLMModel initialised with model=%s", self._model_name)
        except ImportError:
            logger.warning(
                "openai package not installed — LLMModel will not function. "
                "Install it with: pip install openai"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        bar: pd.Series,
        df: Optional[pd.DataFrame] = None,
    ) -> tuple[str, float, str]:
        """Generate a trading signal using the LLM.

        Parameters
        ----------
        ticker:
            Instrument ticker for context.
        bar:
            Latest price/indicator bar from ``MarketDataFetcher``.
        df:
            Optional recent history DataFrame (last N rows used as context).

        Returns
        -------
        tuple[str, float, str]
            ``(signal, confidence, reasoning)``
        """
        if self._client is None:
            return "HOLD", 0.0, "OpenAI client not available"

        prompt = self._build_prompt(ticker, bar, df)

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative trading analyst. "
                            "Analyse the provided market data and return a JSON "
                            'object with keys "signal" (one of BUY/SELL/HOLD), '
                            '"confidence" (float 0-1), and "reasoning" (string).'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
        except Exception:
            logger.exception("LLM request failed")
            return "HOLD", 0.0, "LLM request failed"

        signal = str(parsed.get("signal", "HOLD")).upper()
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = str(parsed.get("reasoning", ""))

        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"

        if signal != "HOLD" and confidence < self._min_confidence:
            logger.debug(
                "LLM signal %s suppressed (confidence %.2f < %.2f)",
                signal,
                confidence,
                self._min_confidence,
            )
            signal = "HOLD"

        logger.info(
            "LLMModel signal: %s (confidence=%.2f) — %s",
            signal,
            confidence,
            reasoning[:120],
        )
        return signal, confidence, reasoning

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ticker: str,
        bar: pd.Series,
        df: Optional[pd.DataFrame],
    ) -> str:
        lines = [f"Instrument: {ticker}", ""]
        lines.append("=== Latest bar ===")
        for key in ["close", "open", "high", "low", "volume"]:
            if key in bar.index:
                lines.append(f"  {key}: {bar[key]:.4f}" if key != "volume" else f"  {key}: {bar[key]:.0f}")

        lines.append("")
        lines.append("=== Technical indicators ===")
        for key in ["rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_mid",
                    "bb_lower", "sma_20", "sma_50", "ema_9", "atr"]:
            if key in bar.index and pd.notna(bar[key]):
                lines.append(f"  {key}: {bar[key]:.4f}")

        if df is not None and not df.empty:
            recent = df.tail(10)[["close"]].copy()
            recent.index = recent.index.strftime("%Y-%m-%d")
            lines.append("")
            lines.append("=== Recent close prices (last 10 bars) ===")
            for date, row in recent.iterrows():
                lines.append(f"  {date}: {row['close']:.4f}")

        lines.append("")
        lines.append(
            "Based on the above data, provide a trading recommendation. "
            "Return JSON only."
        )
        return "\n".join(lines)
