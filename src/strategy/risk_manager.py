"""Risk management layer.

Enforces configurable hard limits before any order is placed:

* Maximum per-trade notional value
* Maximum number of open positions
* Minimum time between trades on the same instrument (cooldown)
* Daily loss limit — bot halts if breached
* Stop-loss tracking per position
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Stateful risk gate that sits between the signal and the executor.

    Parameters
    ----------
    max_trade_value:
        Maximum notional value (in account currency) for a single order.
    max_open_positions:
        Maximum number of simultaneously held instruments.
    max_daily_loss:
        When realised P&L for today falls below ``-max_daily_loss`` the bot
        is halted for the rest of the day.
    stop_loss_pct:
        Percentage drop from entry price that triggers an automatic sell.
        E.g. ``0.05`` = 5 %.
    cooldown_minutes:
        Minimum minutes between two trades on the same ticker.
    """

    def __init__(
        self,
        max_trade_value: float = 500.0,
        max_open_positions: int = 5,
        max_daily_loss: float = 100.0,
        stop_loss_pct: float = 0.05,
        cooldown_minutes: int = 60,
    ) -> None:
        self.max_trade_value = max_trade_value
        self.max_open_positions = max_open_positions
        self.max_daily_loss = max_daily_loss
        self.stop_loss_pct = stop_loss_pct
        self.cooldown_minutes = cooldown_minutes

        self._daily_pnl: dict[date, float] = defaultdict(float)
        self._last_trade: dict[str, datetime] = {}
        self._entry_prices: dict[str, float] = {}
        self._halted_until: Optional[date] = None

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def check(
        self,
        signal: str,
        ticker: str,
        price: float,
        quantity: float,
        open_positions: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for the proposed trade.

        Parameters
        ----------
        signal:
            ``"BUY"`` or ``"SELL"``.
        ticker:
            Instrument ticker.
        price:
            Current price per share/unit.
        quantity:
            Proposed trade quantity (absolute value).
        open_positions:
            Current portfolio from ``Trading212Client.get_portfolio()``.
        """
        today = date.today()

        # Daily loss halt
        if self._halted_until and today <= self._halted_until:
            return False, f"Bot halted until {self._halted_until} (daily loss limit reached)"

        # Daily P&L check
        daily_loss = self._daily_pnl.get(today, 0.0)
        if daily_loss <= -self.max_daily_loss:
            self._halted_until = today
            return False, f"Daily loss limit ${self.max_daily_loss:.2f} reached (loss=${-daily_loss:.2f})"

        # Cooldown
        if ticker in self._last_trade:
            elapsed = datetime.now(timezone.utc).replace(tzinfo=None) - self._last_trade[ticker]
            required = timedelta(minutes=self.cooldown_minutes)
            if elapsed < required:
                remaining = int((required - elapsed).total_seconds() / 60)
                return False, f"Cooldown active for {ticker} ({remaining}m remaining)"

        # Max open positions (only blocks new BUYs when at limit)
        held_tickers = {p["ticker"] for p in open_positions}
        if signal == "BUY" and ticker not in held_tickers:
            if len(held_tickers) >= self.max_open_positions:
                return False, f"Max open positions ({self.max_open_positions}) reached"

        # Max trade value
        trade_value = price * abs(quantity)
        if trade_value > self.max_trade_value:
            return False, (
                f"Trade value ${trade_value:.2f} exceeds limit ${self.max_trade_value:.2f}"
            )

        return True, "OK"

    def suggested_quantity(self, price: float, available_cash: float) -> float:
        """Return a sensible quantity given current price and available cash.

        Caps at ``max_trade_value`` and ensures we don't over-allocate cash.
        """
        max_by_limit = self.max_trade_value / price if price > 0 else 0
        max_by_cash = available_cash / price if price > 0 else 0
        qty = min(max_by_limit, max_by_cash)
        return round(qty, 6)

    # ------------------------------------------------------------------
    # Stop-loss tracking
    # ------------------------------------------------------------------

    def should_stop_loss(self, ticker: str, current_price: float) -> bool:
        """Return *True* if *ticker* has fallen beyond the stop-loss threshold."""
        entry = self._entry_prices.get(ticker)
        if entry is None:
            return False
        drop = (current_price - entry) / entry
        if drop <= -self.stop_loss_pct:
            logger.warning(
                "Stop-loss triggered for %s: entry=%.4f current=%.4f (drop=%.2f%%)",
                ticker,
                entry,
                current_price,
                drop * 100,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # State updates (called by TradeExecutor after a fill)
    # ------------------------------------------------------------------

    def record_buy(self, ticker: str, price: float) -> None:
        self._entry_prices[ticker] = price
        self._last_trade[ticker] = datetime.now(timezone.utc).replace(tzinfo=None)

    def record_sell(self, ticker: str, price: float, pnl: float = 0.0) -> None:
        self._entry_prices.pop(ticker, None)
        self._last_trade[ticker] = datetime.now(timezone.utc).replace(tzinfo=None)
        self._daily_pnl[date.today()] += pnl

    def get_daily_pnl(self) -> float:
        return self._daily_pnl.get(date.today(), 0.0)
