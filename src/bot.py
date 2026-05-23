"""Main orchestration loop for the AI trading bot.

Usage
-----
    python -m src.bot                     # run once immediately
    python -m src.bot --schedule          # run on the configured schedule

The bot:
1. Loads config from ``config/settings.yaml`` (and ``.env`` for secrets).
2. Connects to Trading 212 (demo or live depending on config).
3. For each configured instrument, fetches market data and runs the model.
4. Applies risk checks, then executes approved trades.
5. Checks existing positions for stop-loss breaches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Bootstrap: load .env before anything else imports os.environ
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", path)
    return cfg


def build_model(model_type: str, cfg: dict):  # noqa: ANN201
    """Instantiate the requested model type."""
    if model_type == "ml":
        from src.models.ml_model import MLModel  # noqa: PLC0415

        return MLModel(
            min_confidence=cfg["model"].get("min_confidence", 0.55),
            model_path=cfg["model"].get("model_path"),
        )
    if model_type == "llm":
        from src.models.llm_model import LLMModel  # noqa: PLC0415

        return LLMModel(
            model_name=cfg["model"].get("llm_model_name", "gpt-4o-mini"),
            min_confidence=cfg["model"].get("min_confidence", 0.6),
        )
    if model_type == "hybrid":
        from src.models.ml_model import MLModel  # noqa: PLC0415
        from src.models.llm_model import LLMModel  # noqa: PLC0415
        from src.models.hybrid_model import HybridModel  # noqa: PLC0415

        ml = MLModel(
            min_confidence=cfg["model"].get("min_confidence", 0.55),
            model_path=cfg["model"].get("model_path"),
        )
        llm = LLMModel(
            model_name=cfg["model"].get("llm_model_name", "gpt-4o-mini"),
        )
        return HybridModel(
            ml_model=ml,
            llm_model=llm,
            require_llm_agreement=cfg["model"].get("require_llm_agreement", True),
        )
    raise ValueError(f"Unknown model type: {model_type!r}. Choose ml/llm/hybrid.")


def run_once(cfg: dict) -> None:
    """Execute one full cycle: fetch → model → risk check → trade."""
    from src.trading212.client import Trading212Client  # noqa: PLC0415
    from src.data.market_data import MarketDataFetcher  # noqa: PLC0415
    from src.strategy.risk_manager import RiskManager  # noqa: PLC0415
    from src.strategy.executor import TradeExecutor  # noqa: PLC0415

    t212_cfg = cfg["trading212"]
    risk_cfg = cfg.get("risk", {})
    model_cfg = cfg.get("model", {})

    api_key = os.environ.get("T212_API_KEY", "")
    if not api_key:
        logger.error(
            "T212_API_KEY not set — add it to your .env file and retry."
        )
        return

    demo_mode: bool = t212_cfg.get("demo", True)
    paper_trading: bool = t212_cfg.get("paper_trading", True)

    client = Trading212Client(api_key=api_key, demo=demo_mode)

    # Account summary
    try:
        info = client.get_account_info()
        cash = client.get_account_cash()
        logger.info(
            "Account: %s | free cash: %.2f %s",
            info.get("id"),
            cash.get("free", 0),
            info.get("currencyCode", ""),
        )
    except Exception:
        logger.exception("Failed to fetch account info")
        return

    available_cash: float = float(cash.get("free", 0))
    portfolio = client.get_portfolio()

    fetcher = MarketDataFetcher(db_path=cfg.get("db_path"))

    risk = RiskManager(
        max_trade_value=risk_cfg.get("max_trade_value", 500),
        max_open_positions=risk_cfg.get("max_open_positions", 5),
        max_daily_loss=risk_cfg.get("max_daily_loss", 100),
        stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.05),
        cooldown_minutes=risk_cfg.get("cooldown_minutes", 60),
    )

    executor = TradeExecutor(
        t212_client=client,
        risk_manager=risk,
        paper_trading=paper_trading,
        log_path=cfg.get("trade_log_path", "trades.csv"),
    )

    model_type = model_cfg.get("type", "ml")
    model = build_model(model_type, cfg)

    instruments: list[dict] = cfg.get("instruments", [])
    if not instruments:
        logger.warning("No instruments configured — nothing to trade")
        return

    # ------------------------------------------------------------------
    # Stop-loss check on existing positions
    # ------------------------------------------------------------------
    current_prices: dict[str, float] = {}
    for position in portfolio:
        ticker_raw = position.get("ticker", "")
        # Use T212 ticker mapped from settings or directly
        yf_ticker = _t212_to_yfinance(ticker_raw, instruments)
        if yf_ticker:
            try:
                bar = fetcher.fetch_latest_bar(yf_ticker)
                current_prices[ticker_raw] = float(bar["close"])
            except Exception:
                logger.debug("Could not fetch price for %s", yf_ticker)

    executor.check_stop_losses(portfolio, current_prices)

    # ------------------------------------------------------------------
    # Main signal loop
    # ------------------------------------------------------------------
    for instrument in instruments:
        t212_ticker = instrument.get("t212_ticker")
        yf_ticker = instrument.get("yf_ticker", t212_ticker)

        if not t212_ticker or not yf_ticker:
            logger.warning("Instrument entry missing ticker: %s", instrument)
            continue

        logger.info("--- Processing %s ---", t212_ticker)
        try:
            df = fetcher.fetch(yf_ticker, period="6mo", interval="1d")
            if df.empty:
                logger.warning("No data for %s — skipping", yf_ticker)
                continue
            bar = df.iloc[-1]
        except Exception:
            logger.exception("Failed to fetch data for %s", yf_ticker)
            continue

        # Get signal
        try:
            if model_type == "hybrid":
                signal, confidence, reasoning = model.predict(yf_ticker, bar, df)
            elif model_type == "llm":
                signal, confidence, reasoning = model.predict(yf_ticker, bar, df)
            else:
                signal, confidence = model.predict(bar)
                reasoning = f"ML signal: {signal} (confidence={confidence:.2f})"
        except Exception:
            logger.exception("Model prediction failed for %s", yf_ticker)
            continue

        logger.info(
            "Signal for %s: %s (conf=%.2f) | %s",
            t212_ticker,
            signal,
            confidence,
            reasoning[:100],
        )

        if signal == "HOLD":
            continue

        price = float(bar["close"])
        quantity = risk.suggested_quantity(price, available_cash)
        if quantity <= 0:
            logger.info("Skipping %s — insufficient cash or zero quantity", t212_ticker)
            continue

        allowed, reason = risk.check(
            signal=signal,
            ticker=t212_ticker,
            price=price,
            quantity=quantity,
            open_positions=portfolio,
        )
        if not allowed:
            logger.info("Risk check blocked trade for %s: %s", t212_ticker, reason)
            continue

        executor.execute(
            signal=signal,
            ticker=t212_ticker,
            quantity=quantity,
            price=price,
            confidence=confidence,
            reasoning=reasoning,
        )

        available_cash -= price * quantity

    logger.info("Cycle complete. Daily P&L: %.2f", risk.get_daily_pnl())


def _t212_to_yfinance(
    t212_ticker: str, instruments: list[dict]
) -> str | None:
    for inst in instruments:
        if inst.get("t212_ticker") == t212_ticker:
            return inst.get("yf_ticker")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on the configured schedule rather than once",
    )
    parser.add_argument(
        "--config",
        default=str(_CONFIG_PATH),
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if not args.schedule:
        run_once(cfg)
        return

    # Scheduled mode
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: PLC0415
    except ImportError:
        logger.error("apscheduler not installed — run: pip install apscheduler")
        sys.exit(1)

    interval_minutes: int = cfg.get("schedule_minutes", 5)
    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, "interval", minutes=interval_minutes, args=[cfg])
    logger.info("Scheduler started — running every %d minutes", interval_minutes)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
