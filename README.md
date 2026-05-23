# improved-guacamole — AI-Powered Trading Bot for Trading 212

An automated trading bot that integrates with the **Trading 212** brokerage API and uses **AI/ML models** to generate BUY / SELL / HOLD signals.

> ⚠️ **This bot defaults to paper-trading mode.** No real orders are placed until you explicitly enable live trading in `config/settings.yaml`. Automated trading carries real financial risk — use at your own discretion.

---

## Features

| Feature | Details |
|---------|---------|
| **Trading 212 integration** | Full REST API client — account info, portfolio, market & limit orders, cancellations |
| **Market data** | Yahoo Finance via `yfinance`; RSI, MACD, Bollinger Bands, SMA/EMA, ATR |
| **ML model** | Random Forest trained on technical indicators → BUY / SELL / HOLD + confidence score |
| **LLM model** | GPT-4o-mini analyses recent price action → reasoned signal |
| **Hybrid model** | ML signal confirmed by LLM before execution |
| **Risk controls** | Max trade value, max open positions, daily loss limit, stop-loss %, per-ticker cooldown |
| **Paper trading** | Log decisions without placing real orders |
| **Scheduled runs** | APScheduler runs the bot on a configurable interval |
| **Audit trail** | Every trade decision logged to `trades.csv` with timestamp, signal, confidence, reasoning |

---

## Project Structure

```
improved-guacamole/
├── src/
│   ├── trading212/
│   │   └── client.py          # Trading 212 REST API wrapper
│   ├── data/
│   │   └── market_data.py     # yfinance fetcher + technical indicators
│   ├── models/
│   │   ├── ml_model.py        # Random Forest classifier
│   │   ├── llm_model.py       # OpenAI GPT signal generator
│   │   └── hybrid_model.py    # Combined ML + LLM model
│   ├── strategy/
│   │   ├── risk_manager.py    # Risk controls & stop-loss tracking
│   │   └── executor.py        # Order execution & trade logging
│   └── bot.py                 # Main orchestration loop
├── config/
│   └── settings.yaml          # All configurable parameters
├── tests/                     # pytest test suite
├── .env.example               # API key template
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/davesmallnhs/improved-guacamole.git
cd improved-guacamole
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and add your Trading 212 API key (and OpenAI key if using LLM/hybrid)
```

Your Trading 212 API key can be found in the app under **Settings → API**.

### 3. Configure the bot

Edit `config/settings.yaml`:

```yaml
trading212:
  demo: true          # true = demo account (safe)
  paper_trading: true # true = log only, no real orders

model:
  type: "ml"          # ml | llm | hybrid

instruments:
  - t212_ticker: "AAPL_US_EQ"
    yf_ticker: "AAPL"
```

### 4. Run the bot

**Single cycle** (fetch data, generate signals, execute approved trades):

```bash
python -m src.bot
```

**Scheduled mode** (runs every N minutes as configured):

```bash
python -m src.bot --schedule
```

---

## Model Types

### `ml` — Random Forest (default, no API cost)

Trains a scikit-learn Random Forest on technical indicators computed from 6 months of daily history.  Labels are auto-generated using forward returns (configurable threshold).

The trained model is saved to `data/model.pkl` and reloaded on subsequent runs.  Retrain by deleting the file.

### `llm` — OpenAI GPT

Requires `OPENAI_API_KEY` in `.env`.  Sends a structured prompt containing the latest price/indicator values to GPT-4o-mini and parses the JSON response.

### `hybrid` — ML + LLM (most conservative)

The ML model generates a candidate signal.  The LLM is then asked to confirm.  Both must agree for a trade to proceed — this substantially reduces false positives at the cost of some missed trades.

---

## Risk Controls (`config/settings.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_trade_value` | 500 | Maximum £/$ notional per single order |
| `max_open_positions` | 5 | Maximum simultaneously held instruments |
| `max_daily_loss` | 150 | Bot pauses for the day if realised losses exceed this |
| `stop_loss_pct` | 0.05 | Automatic sell if position drops 5 % from entry |
| `cooldown_minutes` | 60 | Minimum minutes between trades on the same ticker |

---

## Enabling Live Trading

Change **both** flags in `config/settings.yaml`:

```yaml
trading212:
  demo: false         # connects to live.trading212.com
  paper_trading: false  # places real orders
```

> ⚠️ This will place real orders with real money. Ensure you understand the risks and have tested thoroughly in paper/demo mode first.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Trade Log

Every decision is appended to `trades.csv`:

```
timestamp,signal,ticker,quantity,price,confidence,paper,reasoning
2026-01-15T09:32:11,BUY,AAPL_US_EQ,3.33,150.12,0.6812,True,ML signal: BUY (confidence=0.68)
```

---

## Disclaimer

This software is provided for educational purposes only.  It is **not** financial advice.  Past model performance does not guarantee future returns.  Always test in demo/paper mode before risking real capital.