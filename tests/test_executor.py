"""Tests for the Trade Executor."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.strategy.executor import TradeExecutor
from src.strategy.risk_manager import RiskManager


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "trades.csv"


@pytest.fixture
def executor(log_file):
    return TradeExecutor(
        t212_client=None,
        risk_manager=RiskManager(),
        paper_trading=True,
        log_path=log_file,
    )


class TestTradeExecutor:
    def test_paper_trade_returns_result(self, executor):
        result = executor.execute(
            signal="BUY",
            ticker="AAPL_US_EQ",
            quantity=2.0,
            price=150.0,
            confidence=0.75,
            reasoning="Test buy",
        )
        assert result is not None
        assert result["paper"] is True
        assert result["signal"] == "BUY"
        assert result["quantity"] == 2.0

    def test_sell_produces_negative_quantity_in_result(self, executor):
        result = executor.execute(
            signal="SELL",
            ticker="AAPL_US_EQ",
            quantity=1.0,
            price=160.0,
            confidence=0.65,
        )
        assert result["quantity"] == -1.0

    def test_trade_logged_to_csv(self, executor, log_file):
        executor.execute(
            signal="BUY",
            ticker="MSFT_US_EQ",
            quantity=1.0,
            price=300.0,
            confidence=0.8,
            reasoning="Logged trade",
        )
        with open(log_file) as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 2  # header + one row
        assert rows[1][1] == "BUY"
        assert rows[1][2] == "MSFT_US_EQ"

    def test_csv_header_created_on_init(self, log_file):
        TradeExecutor(paper_trading=True, log_path=log_file)
        with open(log_file) as fh:
            header = next(csv.reader(fh))
        assert "signal" in header
        assert "ticker" in header
        assert "confidence" in header

    def test_live_mode_calls_client(self, log_file):
        mock_client = MagicMock()
        mock_client.place_market_order.return_value = {"id": 99}
        exec_live = TradeExecutor(
            t212_client=mock_client,
            paper_trading=False,
            log_path=log_file,
        )
        exec_live.execute(
            signal="BUY",
            ticker="NVDA_US_EQ",
            quantity=0.5,
            price=500.0,
            confidence=0.9,
        )
        mock_client.place_market_order.assert_called_once_with("NVDA_US_EQ", 0.5)

    def test_live_mode_sell_sends_negative_quantity(self, log_file):
        mock_client = MagicMock()
        mock_client.place_market_order.return_value = {"id": 100}
        exec_live = TradeExecutor(
            t212_client=mock_client,
            paper_trading=False,
            log_path=log_file,
        )
        exec_live.execute(
            signal="SELL",
            ticker="NVDA_US_EQ",
            quantity=0.5,
            price=500.0,
            confidence=0.7,
        )
        mock_client.place_market_order.assert_called_once_with("NVDA_US_EQ", -0.5)

    def test_live_mode_raises_without_client(self, log_file):
        exec_live = TradeExecutor(
            t212_client=None,
            paper_trading=False,
            log_path=log_file,
        )
        with pytest.raises(RuntimeError, match="No Trading212Client"):
            exec_live.execute(
                signal="BUY",
                ticker="AAPL_US_EQ",
                quantity=1.0,
                price=100.0,
                confidence=0.8,
            )

    def test_risk_manager_updated_on_buy(self, log_file):
        risk = RiskManager()
        exc = TradeExecutor(paper_trading=True, risk_manager=risk, log_path=log_file)
        exc.execute(signal="BUY", ticker="AAPL_US_EQ", quantity=1, price=150, confidence=0.8)
        assert risk._entry_prices.get("AAPL_US_EQ") == 150

    def test_stop_loss_check_sells_position(self, log_file):
        risk = RiskManager(stop_loss_pct=0.05)
        risk.record_buy("AAPL_US_EQ", 100.0)  # entry at 100
        exc = TradeExecutor(paper_trading=True, risk_manager=risk, log_path=log_file)
        portfolio = [{"ticker": "AAPL_US_EQ", "quantity": 2.0}]
        exc.check_stop_losses(portfolio, current_prices={"AAPL_US_EQ": 90.0})  # 10% drop
        with open(log_file) as fh:
            rows = list(csv.reader(fh))
        assert any(r[1] == "SELL" and r[2] == "AAPL_US_EQ" for r in rows[1:])
