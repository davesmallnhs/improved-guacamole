"""Tests for the Risk Manager."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.strategy.risk_manager import RiskManager


@pytest.fixture
def risk():
    return RiskManager(
        max_trade_value=500,
        max_open_positions=3,
        max_daily_loss=100,
        stop_loss_pct=0.05,
        cooldown_minutes=60,
    )


class TestRiskManager:
    def test_allows_valid_buy(self, risk):
        allowed, reason = risk.check(
            signal="BUY",
            ticker="AAPL_US_EQ",
            price=100.0,
            quantity=2.0,
            open_positions=[],
        )
        assert allowed
        assert reason == "OK"

    def test_blocks_trade_exceeding_value(self, risk):
        allowed, reason = risk.check(
            signal="BUY",
            ticker="AAPL_US_EQ",
            price=300.0,
            quantity=2.0,  # 300 * 2 = 600 > 500
            open_positions=[],
        )
        assert not allowed
        assert "Trade value" in reason

    def test_blocks_when_max_positions_reached(self, risk):
        portfolio = [
            {"ticker": "AAPL_US_EQ"},
            {"ticker": "MSFT_US_EQ"},
            {"ticker": "NVDA_US_EQ"},
        ]
        allowed, reason = risk.check(
            signal="BUY",
            ticker="TSLA_US_EQ",  # new ticker = 4th position
            price=100.0,
            quantity=1.0,
            open_positions=portfolio,
        )
        assert not allowed
        assert "Max open positions" in reason

    def test_allows_sell_when_max_positions_reached(self, risk):
        """Selling an existing position should be allowed even at max positions."""
        portfolio = [
            {"ticker": "AAPL_US_EQ"},
            {"ticker": "MSFT_US_EQ"},
            {"ticker": "NVDA_US_EQ"},
        ]
        allowed, _ = risk.check(
            signal="SELL",
            ticker="AAPL_US_EQ",
            price=100.0,
            quantity=1.0,
            open_positions=portfolio,
        )
        assert allowed

    def test_cooldown_blocks_second_trade(self, risk):
        risk._last_trade["AAPL_US_EQ"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=30)
        allowed, reason = risk.check(
            signal="BUY",
            ticker="AAPL_US_EQ",
            price=100.0,
            quantity=1.0,
            open_positions=[],
        )
        assert not allowed
        assert "Cooldown" in reason

    def test_cooldown_passes_after_duration(self, risk):
        risk._last_trade["AAPL_US_EQ"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=90)
        allowed, _ = risk.check(
            signal="BUY",
            ticker="AAPL_US_EQ",
            price=100.0,
            quantity=1.0,
            open_positions=[],
        )
        assert allowed

    def test_daily_loss_halt(self, risk):
        risk._daily_pnl[date.today()] = -150.0  # exceeds max_daily_loss
        allowed, reason = risk.check(
            signal="BUY",
            ticker="AAPL_US_EQ",
            price=100.0,
            quantity=1.0,
            open_positions=[],
        )
        assert not allowed
        assert "loss limit" in reason.lower()

    def test_suggested_quantity_capped_by_max_value(self, risk):
        qty = risk.suggested_quantity(price=200.0, available_cash=10_000.0)
        assert qty * 200.0 <= risk.max_trade_value + 0.01

    def test_suggested_quantity_capped_by_cash(self, risk):
        qty = risk.suggested_quantity(price=100.0, available_cash=50.0)
        assert qty * 100.0 <= 50.0 + 0.01

    def test_stop_loss_triggered(self, risk):
        risk._entry_prices["AAPL_US_EQ"] = 100.0
        assert risk.should_stop_loss("AAPL_US_EQ", 93.0)  # 7% drop

    def test_stop_loss_not_triggered(self, risk):
        risk._entry_prices["AAPL_US_EQ"] = 100.0
        assert not risk.should_stop_loss("AAPL_US_EQ", 97.0)  # only 3% drop

    def test_stop_loss_no_entry(self, risk):
        assert not risk.should_stop_loss("UNKNOWN", 50.0)

    def test_record_buy_updates_entry(self, risk):
        risk.record_buy("AAPL_US_EQ", 150.0)
        assert risk._entry_prices["AAPL_US_EQ"] == 150.0

    def test_record_sell_removes_entry(self, risk):
        risk._entry_prices["AAPL_US_EQ"] = 150.0
        risk.record_sell("AAPL_US_EQ", 160.0, pnl=10.0)
        assert "AAPL_US_EQ" not in risk._entry_prices
        assert risk.get_daily_pnl() == 10.0
