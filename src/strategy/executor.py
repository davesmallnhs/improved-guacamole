"""Trade executor: translates approved signals into T212 orders.

Supports two modes:
* **paper** (default) — logs the decision but does NOT place real orders.
* **live** — places real market orders via the T212 client.

Always call ``RiskManager.check()`` before invoking ``TradeExecutor``.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Place (or simulate) orders and maintain a trade log.

    Parameters
    ----------
    t212_client:
        A ``Trading212Client`` instance.  May be *None* in paper-trading mode.
    risk_manager:
        The ``RiskManager`` to update after each fill.
    paper_trading:
        When *True* no real orders are placed.
    log_path:
        CSV file to append trade records to.
    """

    def __init__(
        self,
        t212_client: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        paper_trading: bool = True,
        log_path: str | Path = "trades.csv",
    ) -> None:
        self._client = t212_client
        self._risk = risk_manager
        self._paper = paper_trading
        self._log_path = Path(log_path)
        self._ensure_log_header()

        mode = "PAPER" if paper_trading else "LIVE"
        logger.info("TradeExecutor initialised in %s mode", mode)

    # ------------------------------------------------------------------

    def execute(
        self,
        signal: str,
        ticker: str,
        quantity: float,
        price: float,
        confidence: float,
        reasoning: str = "",
    ) -> Optional[dict[str, Any]]:
        """Execute a trade order.

        Parameters
        ----------
        signal:
            ``"BUY"`` or ``"SELL"``.
        ticker:
            T212 ticker string (e.g. ``"AAPL_US_EQ"``).
        quantity:
            Absolute quantity to trade.
        price:
            Current market price (used for logging and P&L calc).
        confidence:
            Model confidence score for this signal.
        reasoning:
            Human-readable explanation from the model.

        Returns
        -------
        dict or None
            T212 order response (or a synthetic dict in paper mode).
        """
        signed_qty = quantity if signal == "BUY" else -quantity

        if self._paper:
            result = {
                "paper": True,
                "ticker": ticker,
                "signal": signal,
                "quantity": signed_qty,
                "price": price,
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            }
            logger.info(
                "[PAPER] %s %s qty=%.6f @ %.4f (conf=%.2f)",
                signal,
                ticker,
                quantity,
                price,
                confidence,
            )
        else:
            if self._client is None:
                raise RuntimeError("No Trading212Client provided for live trading")
            result = self._client.place_market_order(ticker, signed_qty)
            logger.info(
                "[LIVE] %s %s qty=%.6f @ %.4f (conf=%.2f) → order_id=%s",
                signal,
                ticker,
                quantity,
                price,
                confidence,
                result.get("id"),
            )

        # Update risk manager state
        if self._risk:
            if signal == "BUY":
                self._risk.record_buy(ticker, price)
            else:
                # Approximate P&L using entry price recorded by RiskManager
                entry = self._risk._entry_prices.get(ticker, price)
                pnl = (price - entry) * quantity
                self._risk.record_sell(ticker, price, pnl)

        self._log_trade(
            signal=signal,
            ticker=ticker,
            quantity=signed_qty,
            price=price,
            confidence=confidence,
            reasoning=reasoning,
            paper=self._paper,
        )

        return result

    def check_stop_losses(
        self,
        portfolio: list[dict[str, Any]],
        current_prices: dict[str, float],
    ) -> None:
        """Scan the portfolio for stop-loss breaches and issue sell orders."""
        if not self._risk:
            return
        for position in portfolio:
            ticker = position.get("ticker", "")
            price = current_prices.get(ticker)
            if price and self._risk.should_stop_loss(ticker, price):
                qty = abs(float(position.get("quantity", 0)))
                if qty > 0:
                    logger.warning("Executing stop-loss sell for %s", ticker)
                    self.execute(
                        signal="SELL",
                        ticker=ticker,
                        quantity=qty,
                        price=price,
                        confidence=1.0,
                        reasoning="Stop-loss triggered",
                    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _ensure_log_header(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["timestamp", "signal", "ticker", "quantity", "price",
                     "confidence", "paper", "reasoning"]
                )

    def _log_trade(
        self,
        signal: str,
        ticker: str,
        quantity: float,
        price: float,
        confidence: float,
        reasoning: str,
        paper: bool,
    ) -> None:
        with open(self._log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    signal,
                    ticker,
                    quantity,
                    price,
                    f"{confidence:.4f}",
                    paper,
                    reasoning[:500],
                ]
            )
