"""Market data fetching and technical indicator calculation.

Uses *yfinance* as the data source and *pandas-ta* for indicators.
Data can optionally be persisted to a local SQLite database so that
indicator history survives bot restarts.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Indicators calculated by default
_DEFAULT_INDICATORS = ["rsi", "macd", "bbands", "sma_20", "sma_50", "ema_9"]


class MarketDataFetcher:
    """Fetch OHLCV history and compute technical indicators.

    Parameters
    ----------
    db_path:
        Path to an SQLite file for persisting price data.  Pass *None*
        to keep everything in memory only.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MarketDataFetcher initialised (db=%s)",
            self._db_path or "in-memory",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV data and return a DataFrame with indicators.

        Parameters
        ----------
        ticker:
            Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
        period:
            Lookback period accepted by yfinance (``"1mo"``, ``"6mo"``,
            ``"1y"``, etc.).
        interval:
            Bar interval (``"1d"``, ``"1h"``, ``"15m"``, etc.).

        Returns
        -------
        pandas.DataFrame
            OHLCV columns plus computed indicator columns, index = datetime.
        """
        logger.info("Fetching %s (period=%s, interval=%s)", ticker, period, interval)
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            logger.warning("No data returned for %s", ticker)
            return raw

        # yfinance may return a MultiIndex when only one ticker is requested
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = self._add_indicators(df)

        if self._db_path:
            self._persist(ticker, df)

        return df

    def fetch_latest_bar(
        self,
        ticker: str,
        interval: str = "1d",
    ) -> pd.Series:
        """Return only the most recent completed bar with all indicators."""
        df = self.fetch(ticker, period="3mo", interval=interval)
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        return df.iloc[-1]

    def load_from_db(self, ticker: str) -> pd.DataFrame:
        """Load previously persisted data for *ticker* from SQLite."""
        if not self._db_path:
            raise RuntimeError("No db_path configured — cannot load from DB")
        table = _safe_table_name(ticker)
        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql(
                f"SELECT * FROM '{table}' ORDER BY date",  # noqa: S608
                conn,
                index_col="date",
                parse_dates=["date"],
            )
        return df

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators and append as new columns."""
        try:
            import pandas_ta as ta  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "pandas-ta not installed — skipping indicator calculation"
            )
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI (14-period)
        df["rsi"] = ta.rsi(close, length=14)

        # MACD
        macd = ta.macd(close)
        if macd is not None:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 2]
            df["macd_hist"] = macd.iloc[:, 1]

        # Bollinger Bands
        bbands = ta.bbands(close, length=20)
        if bbands is not None:
            df["bb_upper"] = bbands.iloc[:, 0]
            df["bb_mid"] = bbands.iloc[:, 1]
            df["bb_lower"] = bbands.iloc[:, 2]

        # Simple / exponential moving averages
        df["sma_20"] = ta.sma(close, length=20)
        df["sma_50"] = ta.sma(close, length=50)
        df["ema_9"] = ta.ema(close, length=9)

        # ATR (for position sizing)
        df["atr"] = ta.atr(high, low, close, length=14)

        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, ticker: str, df: pd.DataFrame) -> None:
        table = _safe_table_name(ticker)
        with sqlite3.connect(self._db_path) as conn:
            df_to_save = df.copy()
            df_to_save.index.name = "date"
            df_to_save.to_sql(table, conn, if_exists="replace")
        logger.debug("Persisted %d rows for %s to %s", len(df), ticker, self._db_path)


def _safe_table_name(ticker: str) -> str:
    """Convert a ticker symbol to a safe SQLite table name."""
    return ticker.replace("-", "_").replace(".", "_").upper()
