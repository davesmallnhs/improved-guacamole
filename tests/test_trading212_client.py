"""Tests for the Trading 212 API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.trading212.client import Trading212Client, Trading212Error


@pytest.fixture
def client():
    return Trading212Client(api_key="test-key", demo=True)


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.ok = status_code < 400
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = b"content"
    resp.text = str(json_data)
    return resp


class TestTrading212Client:
    def test_demo_base_url(self):
        c = Trading212Client(api_key="k", demo=True)
        assert "demo" in c._base_url

    def test_live_base_url(self):
        c = Trading212Client(api_key="k", demo=False)
        assert "live" in c._base_url

    def test_get_account_info(self, client):
        expected = {"id": "ACC123", "currencyCode": "GBP"}
        with patch.object(client, "_request", return_value=expected) as mock_req:
            result = client.get_account_info()
        mock_req.assert_called_once_with("GET", "/equity/account/info")
        assert result == expected

    def test_get_account_cash(self, client):
        expected = {"free": 1000.0, "invested": 500.0}
        with patch.object(client, "_request", return_value=expected):
            result = client.get_account_cash()
        assert result["free"] == 1000.0

    def test_get_portfolio(self, client):
        expected = [{"ticker": "AAPL_US_EQ", "quantity": 5}]
        with patch.object(client, "_request", return_value=expected):
            result = client.get_portfolio()
        assert len(result) == 1

    def test_place_market_order_buy(self, client):
        expected = {"id": 1, "ticker": "AAPL_US_EQ", "quantity": 2}
        with patch.object(client, "_request", return_value=expected) as mock_req:
            result = client.place_market_order("AAPL_US_EQ", 2)
        mock_req.assert_called_once_with(
            "POST", "/equity/orders/market", json={"ticker": "AAPL_US_EQ", "quantity": 2}
        )
        assert result["id"] == 1

    def test_place_market_order_sell(self, client):
        with patch.object(client, "_request", return_value={"id": 2}) as mock_req:
            client.place_market_order("AAPL_US_EQ", -1)
        mock_req.assert_called_once_with(
            "POST", "/equity/orders/market", json={"ticker": "AAPL_US_EQ", "quantity": -1}
        )

    def test_place_limit_order(self, client):
        with patch.object(client, "_request", return_value={"id": 3}) as mock_req:
            client.place_limit_order("AAPL_US_EQ", 1, 150.0, "GTC")
        _, kwargs = mock_req.call_args
        payload = mock_req.call_args[1]["json"]
        assert payload["limitPrice"] == 150.0
        assert payload["timeValidity"] == "GTC"

    def test_cancel_order(self, client):
        with patch.object(client, "_request", return_value=None) as mock_req:
            client.cancel_order(42)
        mock_req.assert_called_once_with("DELETE", "/equity/orders/42")

    def test_error_raised_on_bad_status(self, client):
        with patch.object(client._session, "request") as mock_req:
            resp = MagicMock()
            resp.ok = False
            resp.status_code = 401
            resp.text = "Unauthorized"
            mock_req.return_value = resp
            with pytest.raises(Trading212Error):
                client.get_account_info()

    def test_throttle_between_calls(self, client):
        """Two consecutive _request calls should not raise."""
        with patch.object(client, "_request", return_value={}):
            client.get_account_info()
            client.get_account_info()
