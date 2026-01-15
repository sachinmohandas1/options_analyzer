# Options Chain Analyzer

A quantitative tool for scanning options chains across index ETFs and identifying high-probability premium selling opportunities.

## Features

- **Multi-Strategy Analysis**: Cash-secured puts, credit spreads (put and call), iron condors
- **Volatility Surface Analysis**: IV skew, term structure, anomaly detection
- **Configurable Criteria**: Probability thresholds, return targets, DTE limits, delta constraints
- **Capital Management**: Position sizing with risk limits and portfolio constraints
- **Rich CLI Interface**: Beautiful terminal output with interactive mode
- **Backtesting**: Historical strategy testing with up to 10 years of data

## Installation

```bash
cd options_analyzer
pip install -r requirements.txt
```

For faster Greeks calculations (optional):
```bash
pip install py_vollib_vectorized numba
```

## Quick Start

```bash
# Run with defaults ($13,000 capital, 70% prob, 1% weekly return, 5 DTE max)
python main.py

# Analyze specific symbols
python main.py -s SPY QQQ IWM

# Custom criteria
python main.py --capital 25000 --prob 0.75 --return 1.5 --max-dte 3

# Interactive mode
python main.py -i

# JSON output for automation
python main.py --json > results.json
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --symbols` | Major ETFs | Symbols to analyze |
| `--capital` | 13000 | Total capital available |
| `--prob` | 0.70 | Minimum probability of profit |
| `--return` | 1.0 | Minimum weekly return % |
| `--max-dte` | 5 | Maximum days to expiration |
| `--min-dte` | 1 | Minimum days to expiration |
| `--max-delta` | 0.30 | Maximum delta for short options |
| `--max-price` | None | Optional: Filter symbols by share price |
| `--top` | 20 | Number of top trades to show |
| `--strategies` | csp put_spread call_spread iron_condor | Strategies to use |
| `--no-vol` | false | Skip volatility analysis |
| `-i, --interactive` | false | Interactive browse mode |
| `-v, --verbose` | false | Verbose output |
| `--json` | false | JSON output |

## Capital-Based Trade Filtering

By default, trades are filtered by **collateral requirements (max loss) vs available capital**, not by share price. This allows:

- **Credit spreads on expensive stocks** like GOOGL, AMZN, TSLA where the max loss fits within your capital
- **CSPs on any stock** where strike × 100 is affordable

If you want to additionally filter by share price upfront, use `--max-price`:

```bash
# Only analyze stocks under $150/share
python main.py --max-price 150
```

## Strategies

### Cash Secured Put (CSP)
Sell OTM puts with cash collateral. Bullish/neutral strategy.
- Max profit: Premium received
- Max loss: Strike - Premium (if stock goes to $0)
- Collateral: Strike x 100

### Put Credit Spread
Sell OTM put, buy further OTM put. Bullish strategy with defined risk.
- Max profit: Net credit
- Max loss: Spread width - Credit
- Collateral: Max loss

### Call Credit Spread
Sell OTM call, buy further OTM call. Bearish strategy with defined risk.
- Max profit: Net credit
- Max loss: Spread width - Credit
- Collateral: Max loss

### Iron Condor
Combine put spread + call spread. Neutral strategy that profits from low volatility.
- Sell OTM put + buy further OTM put (bull put spread)
- Sell OTM call + buy further OTM call (bear call spread)
- Max profit: Total credit from both spreads (when price stays between short strikes)
- Max loss: Wider spread width - Total credit (only one side can be breached)
- Collateral: Max loss (defined risk)

## Volatility Surface Analysis

The analyzer builds IV surfaces and extracts trading signals:

- **IV Rank**: Current IV relative to 52-week range
- **Term Structure**: Contango vs backwardation
- **Skew**: 25-delta put vs call IV differential
- **Anomalies**: Statistical outliers in IV

## Quantum ML Scoring (Experimental)

Enhance trade selection with a Variational Quantum Circuit (VQC) trained on your backtest data. The quantum model learns non-linear feature interactions that simple weighted scoring might miss.

### Quick Start

```bash
# Enable QML scoring (auto-trains on 12 months of data)
python main.py --qml -s SPY QQQ IWM

# Custom training period
python main.py --qml --qml-months 6 -s SPY QQQ IWM

# Force retrain (ignore cached model)
python main.py --qml --qml-retrain -s SPY QQQ
```

### How It Works

1. **First run**: Automatically trains on backtest data for specified symbols (~30-60 seconds)
2. **Subsequent runs**: Loads cached model instantly (cache expires after 7 days)
3. **Scoring**: Blends QML score (70%) with original score (30%) for stability

### QML Options

| Option | Default | Description |
|--------|---------|-------------|
| `--qml` | off | Enable quantum ML scoring |
| `--qml-months` | 12 | Months of backtest data for training |
| `--qml-retrain` | off | Force model retraining (ignore cache) |

### Output

When QML is enabled, the trade table shows a **QML Δ** column indicating how the quantum model adjusted each trade's score:
- **Green (+)**: QML boosted the trade (model predicts higher success)
- **Red (-)**: QML penalized the trade (model predicts lower success)
- **Gray**: Minimal change

### Installation

```bash
pip install pennylane pennylane-lightning torch
```

Falls back to classical logistic regression if dependencies unavailable.

For detailed documentation, see [docs/QUANTUM_SCORER.txt](docs/QUANTUM_SCORER.txt).

## Backtesting

Test your strategies against historical data (up to 10 years). The backtester uses real price data from Yahoo Finance and synthesizes options chains using Black-Scholes modeling.

### Quick Start

```bash
# Basic 1-month backtest
python main.py --backtest --start-date 2024-01-01 --end-date 2024-02-01 -s SPY

# 1-year backtest with multiple symbols
python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01 -s SPY QQQ IWM

# 10-year stress test
python main.py --backtest --start-date 2014-01-01 --end-date 2024-01-01 -s SPY

# Custom exit rules
python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01 \
    --profit-target 0.5 --stop-loss 2.0 --max-positions 5
```

### Backtest Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backtest` | - | Enable backtest mode |
| `--start-date` | Required | Backtest start (YYYY-MM-DD) |
| `--end-date` | Required | Backtest end (YYYY-MM-DD) |
| `--profit-target` | 0.5 | Exit at X% of max profit (0.5 = 50%) |
| `--stop-loss` | 2.0 | Exit at X times premium lost (2.0 = 2x) |
| `--max-positions` | 5 | Maximum concurrent positions |

### How It Works

1. **Data Loading**: Fetches daily OHLCV from Yahoo Finance, calculates 30-day rolling volatility
2. **Chain Synthesis**: Generates realistic options chains using Black-Scholes pricing
3. **Daily Simulation**: Iterates through each trading day, managing entries and exits
4. **Performance Metrics**: Calculates win rate, Sharpe ratio, max drawdown, and more

### Exit Rules

- **Profit Target**: Close when unrealized profit reaches 50% of max profit (captures gains, reduces gamma risk)
- **Stop Loss**: Close when loss reaches 2x premium received (limits downside)
- **Expiration**: Settle at expiration based on final underlying price

For detailed documentation, see [docs/BACKTESTING.txt](docs/BACKTESTING.txt).

## Architecture

```
options_analyzer/
├── core/
│   ├── config.py      # Configuration management
│   └── models.py      # Data models
├── data/
│   └── fetcher.py     # Options chain data fetching
├── analysis/
│   ├── greeks.py      # Greeks calculations
│   ├── volatility_surface.py  # Vol surface analysis
│   ├── position_sizer.py      # Position sizing
│   ├── quantum_scorer.py      # VQC-based trade scoring
│   └── qml_integration.py     # Streamlined QML CLI integration
├── strategies/
│   ├── base.py        # Base strategy class
│   ├── secured_premium.py  # CSP, covered calls
│   └── credit_spreads.py   # Spreads, iron condors
├── backtesting/
│   ├── historical_data.py  # Yahoo Finance + Black-Scholes synthesis
│   ├── backtester.py       # Main simulation engine
│   ├── trade_manager.py    # Trade lifecycle management
│   └── performance.py      # Metrics calculation
├── ui/
│   └── display.py     # Terminal display
├── analyzer.py        # Main orchestrator
├── main.py           # CLI entry point
└── train_quantum_scorer.py  # Standalone QML training script
```

## Customization

### Adding Custom Strategies

```python
from strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "My Custom Strategy"

    @property
    def strategy_type(self) -> str:
        return "custom"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    def find_candidates(self, chain, surface=None):
        # Your logic here
        pass

    def calculate_metrics(self, candidate):
        # Your calculations
        pass
```

### Modifying Criteria at Runtime

```python
from analyzer import OptionsAnalyzer

analyzer = OptionsAnalyzer()
analyzer.update_config(
    trade_criteria={'min_prob_profit': 0.75, 'max_dte': 3}
)
result = analyzer.run_analysis()
```

### Using as a Library

```python
from analyzer import create_analyzer

# Quick setup
analyzer = create_analyzer(
    capital=20000,
    min_prob_profit=0.75,
    min_weekly_return=1.5,
    max_dte=3,
    symbols=['SPY', 'QQQ']
)

result = analyzer.run_analysis()

# Access results
for candidate in result.top_candidates:
    print(f"{candidate.strategy_name} {candidate.underlying_symbol}: "
          f"{candidate.weekly_return:.2%} weekly, {candidate.prob_profit:.0%} prob")
```

## Data Source

Uses Yahoo Finance via `yfinance` library. Data is free but may have delays.
For production use, consider integrating a paid data provider.

## Disclaimer

This tool is for educational and research purposes. Options trading involves
significant risk. Past performance does not guarantee future results. Always
do your own research and consider consulting a financial advisor.
