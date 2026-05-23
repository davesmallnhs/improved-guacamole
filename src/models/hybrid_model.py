"""Hybrid model: combines ML quantitative signal with LLM confirmation.

The ML model votes first.  If it produces a BUY or SELL signal with sufficient
confidence, the LLM is called to confirm.  The LLM must agree (same direction)
for the trade to proceed; otherwise HOLD is returned.  This reduces false
positives at the cost of slightly fewer trades.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .ml_model import MLModel
from .llm_model import LLMModel

logger = logging.getLogger(__name__)


class HybridModel:
    """Combines MLModel and LLMModel signals.

    Parameters
    ----------
    ml_model:
        A pre-trained (or freshly constructed) ``MLModel``.
    llm_model:
        An ``LLMModel`` instance.
    ml_min_confidence:
        ML confidence threshold before bothering to ask the LLM.
    require_llm_agreement:
        When *True* (default) both models must agree for a BUY/SELL to pass.
        When *False* the ML signal alone is sufficient (LLM is advisory).
    """

    def __init__(
        self,
        ml_model: Optional[MLModel] = None,
        llm_model: Optional[LLMModel] = None,
        ml_min_confidence: float = 0.55,
        require_llm_agreement: bool = True,
    ) -> None:
        self._ml = ml_model or MLModel()
        self._llm = llm_model or LLMModel()
        self._ml_min_confidence = ml_min_confidence
        self._require_llm_agreement = require_llm_agreement

    # ------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        bar: pd.Series,
        df: Optional[pd.DataFrame] = None,
    ) -> tuple[str, float, str]:
        """Generate a combined signal.

        Returns
        -------
        tuple[str, float, str]
            ``(signal, confidence, reasoning)``
        """
        ml_signal, ml_confidence = self._ml.predict(bar)

        if ml_signal == "HOLD":
            return "HOLD", ml_confidence, "ML model returned HOLD"

        if ml_confidence < self._ml_min_confidence:
            return "HOLD", ml_confidence, f"ML confidence {ml_confidence:.2f} below threshold"

        # Ask LLM for confirmation
        llm_signal, llm_confidence, llm_reasoning = self._llm.predict(ticker, bar, df)

        if self._require_llm_agreement and llm_signal != ml_signal:
            logger.info(
                "Hybrid: ML=%s but LLM=%s — returning HOLD",
                ml_signal,
                llm_signal,
            )
            return (
                "HOLD",
                min(ml_confidence, llm_confidence),
                f"ML and LLM disagree (ML={ml_signal}, LLM={llm_signal}). {llm_reasoning}",
            )

        # Both agree (or LLM agreement not required)
        combined_confidence = (ml_confidence + llm_confidence) / 2
        reasoning = (
            f"ML signal: {ml_signal} ({ml_confidence:.2f}). "
            f"LLM signal: {llm_signal} ({llm_confidence:.2f}). "
            f"LLM reasoning: {llm_reasoning}"
        )
        logger.info(
            "HybridModel signal: %s (combined confidence=%.3f)",
            ml_signal,
            combined_confidence,
        )
        return ml_signal, combined_confidence, reasoning
