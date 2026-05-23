"""Trading 212 REST API client.

Wraps the Trading 212 API (https://t212public-api-docs.redoc.ly/).
Supports both **demo** (default) and **live** modes.
Rate limiting and automatic retries are handled transparently.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_DEMO_BASE = "https://demo.trading212.com/api/v0"
_LIVE_BASE = "https://live.trading212.com/api/v0"

# Maximum requests per minute enforced by T212 (conservative estimate)
_RATE_LIMIT_PAUSE = 0.5  # seconds between consecutive API calls


class Trading212Error(Exception):
    """Raised when the Trading 212 API returns an error response."""


class Trading212Client:
    """Thin wrapper around the Trading 212 REST API.

    Parameters
    ----------
    api_key:
        Your Trading 212 API key (found in the app under Settings → API).
    demo:
        When *True* (default) all requests go to the demo environment so no
        real money is at risk.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of times to retry on transient network errors (5xx, connection
        errors, timeouts).
    """

    def __init__(
        self,
        api_key: str,
        demo: bool = True,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = _DEMO_BASE if demo else _LIVE_BASE
        self._timeout = timeout
        self._demo = demo
        self._session = self._build_session(max_retries)
        self._last_call: float = 0.0

        mode = "DEMO" if demo else "LIVE"
        logger.info("Trading212Client initialised in %s mode", mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self, max_retries: int) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": self._api_key,
                "Content-Type": "application/json",
            }
        )
        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    def _throttle(self) -> None:
        """Enforce a minimum gap between API calls."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RATE_LIMIT_PAUSE:
            time.sleep(_RATE_LIMIT_PAUSE - elapsed)
        self._last_call = time.monotonic()

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        self._throttle()
        url = f"{self._base_url}{path}"
        logger.debug("%s %s", method.upper(), url)
        response = self._session.request(
            method, url, timeout=self._timeout, **kwargs
        )
        if not response.ok:
            raise Trading212Error(
                f"{method.upper()} {path} failed: "
                f"{response.status_code} {response.text}"
            )
        if response.content:
            return response.json()
        return None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict[str, Any]:
        """Return basic account metadata (ID, currency, type)."""
        return self._request("GET", "/equity/account/info")

    def get_account_cash(self) -> dict[str, Any]:
        """Return current cash balances (free, invested, total)."""
        return self._request("GET", "/equity/account/cash")

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_portfolio(self) -> list[dict[str, Any]]:
        """Return all open equity positions."""
        return self._request("GET", "/equity/portfolio")

    def get_position(self, ticker: str) -> dict[str, Any]:
        """Return the open position for *ticker*, or raise if not held."""
        return self._request("GET", f"/equity/portfolio/{ticker}")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_orders(self) -> list[dict[str, Any]]:
        """Return all equity orders (open and recently filled)."""
        return self._request("GET", "/equity/orders")

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a market order.

        Parameters
        ----------
        ticker:
            T212 instrument ticker (e.g. ``"AAPL_US_EQ"``).
        quantity:
            Fractional quantity to buy (positive) or sell (negative).
        """
        payload = {"ticker": ticker, "quantity": quantity}
        logger.info("Placing market order: %s qty=%s", ticker, quantity)
        return self._request("POST", "/equity/orders/market", json=payload)

    def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        time_validity: str = "DAY",
    ) -> dict[str, Any]:
        """Place a limit order.

        Parameters
        ----------
        ticker:
            T212 instrument ticker.
        quantity:
            Fractional quantity (positive = buy, negative = sell).
        limit_price:
            Limit price in the instrument's currency.
        time_validity:
            ``"DAY"`` or ``"GTC"`` (Good Till Cancelled).
        """
        payload = {
            "ticker": ticker,
            "quantity": quantity,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        }
        logger.info(
            "Placing limit order: %s qty=%s @ %s", ticker, quantity, limit_price
        )
        return self._request("POST", "/equity/orders/limit", json=payload)

    def cancel_order(self, order_id: int) -> None:
        """Cancel an open order by its ID."""
        logger.info("Cancelling order %s", order_id)
        self._request("DELETE", f"/equity/orders/{order_id}")

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def get_instruments(self) -> list[dict[str, Any]]:
        """Return all tradeable instruments available on this account."""
        return self._request("GET", "/equity/metadata/instruments")
