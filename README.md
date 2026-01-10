# Options Chain Analyzer

A quantitative tool for scanning options chains across index ETFs and identifying high-probability premium selling opportunities.

## Features

- **Multi-Strategy Analysis**: Cash-secured puts, credit spreads (put and call), iron condors
- **Volatility Surface Analysis**: IV skew, term structure, anomaly detection
- **Configurable Criteria**: Probability thresholds, return targets, DTE limits, delta constraints
- **Capital Management**: Position sizing with risk limits and portfolio constraints
- **Rich CLI Interface**: Beautiful terminal output with interactive mode

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
| `--top` | 20 | Number of top trades to show |
| `--strategies` | csp put_spread call_spread | Strategies to use |
| `--no-vol` | false | Skip volatility analysis |
| `-i, --interactive` | false | Interactive browse mode |
| `-v, --verbose` | false | Verbose output |
| `--json` | false | JSON output |

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
Combine put spread + call spread. Neutral strategy.
- Max profit: Total credit from both spreads
- Max loss: Wider spread width - Total credit
- Collateral: Max loss

## Volatility Surface Analysis

The analyzer builds IV surfaces and extracts trading signals:

- **IV Rank**: Current IV relative to 52-week range
- **Term Structure**: Contango vs backwardation
- **Skew**: 25-delta put vs call IV differential
- **Anomalies**: Statistical outliers in IV

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
│   └── position_sizer.py      # Position sizing
├── strategies/
│   ├── base.py        # Base strategy class
│   ├── secured_premium.py  # CSP, covered calls
│   └── credit_spreads.py   # Spreads, iron condors
├── ui/
│   └── display.py     # Terminal display
├── analyzer.py        # Main orchestrator
└── main.py           # CLI entry point
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
