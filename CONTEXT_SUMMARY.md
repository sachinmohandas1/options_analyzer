# Options Analyzer - Project Context Summary

## Project Overview

An options analysis tool for premium selling strategies designed for a trader with:
- **$13,000 initial capital**
- **1% weekly profit target**
- **5 DTE maximum**
- **70% probability of profit minimum**
- **$120 max share price** (so CSP collateral stays under ~$12,000)

## Supported Strategies

1. **Cash-Secured Puts (CSP)** - Sell OTM puts with cash collateral = strike × 100
2. **Put Credit Spreads** - Sell OTM put, buy further OTM put (defined risk)
3. **Call Credit Spreads** - Sell OTM call, buy further OTM call (defined risk)
4. **Covered Calls** - Strategy exists in code but NOT enabled in CLI yet (user was considering adding it)

**NOT supported (by design):** Naked calls, naked puts, iron condors

## Repository

- **GitHub:** https://github.com/sachinmohandas1/options_analyzer
- **Local path:** C:\Users\sachi\options_analyzer

## Key Files and Architecture

```
options_analyzer/
├── main.py                 # CLI entry point
├── analyzer.py             # Main analysis orchestrator
├── core/
│   ├── config.py           # All configuration (TradeCriteria, CapitalConfig, etc.)
│   └── models.py           # Data models (OptionContract, TradeCandidate, etc.)
├── data/
│   ├── fetcher.py          # yfinance data fetching with validation
│   └── discovery.py        # Symbol discovery (S&P 500, Nasdaq 100, ETFs)
├── analysis/
│   ├── greeks.py           # Black-Scholes Greeks calculation
│   ├── volatility_surface.py # IV surface analysis, skew detection
│   └── position_sizer.py   # Position sizing logic
├── strategies/
│   ├── base.py             # BaseStrategy abstract class
│   ├── secured_premium.py  # CSP and Covered Call strategies
│   └── credit_spreads.py   # Put/Call credit spreads (+ unused IronCondor)
├── ui/
│   └── display.py          # Terminal output formatting
└── docs/                   # Documentation files
```

## CLI Usage

```bash
# Default scan (~100 symbols from config)
python main.py

# Full market scan (S&P 500 + Nasdaq 100 + ETFs, ~600 symbols)
python main.py --full-scan

# Specific strategies only
python main.py --strategies csp put_spread

# Custom parameters
python main.py --capital 15000 --min-prob 0.75 --max-dte 7 --max-price 100
```

## Key Technical Details

### Weekly Return Calculation (models.py:170-185)
```python
weekly_return = (return_on_collateral / DTE) × 7
# Capped at 5 × return_on_collateral to avoid extreme extrapolation
```

Where `return_on_collateral`:
- **CSP:** premium_received / (strike × 100)
- **Credit Spread:** premium_received / max_loss

### Data Validation (fetcher.py)

The system has multiple layers to catch stale/invalid data:

1. **Price Fetching Priority** (fetcher.py:43-95, discovery.py:136-193):
   - Intraday 1-minute data (most current)
   - fast_info
   - info dict
   - 5-day history (fallback)

2. **Strike Price Sanity Check** (fetcher.py:97-130):
   - Strikes must be 30%-300% of current price
   - Filters out pre-split stale data

3. **Options Chain Consistency Validation** (fetcher.py:168-253):
   - Must have strikes within 20% of current price
   - ATM premium can't exceed 30% of stock price
   - ITM options must have at least 50% of intrinsic value
   - Catches cases like SPXS where stock price updated but options chain is stale

4. **Data Freshness Check** (fetcher.py:132-166):
   - Warns if last trade > 1 day ago (weekdays) or > 3 days (weekends)

### Greeks Calculation (greeks.py)

- Uses py_vollib Black-Scholes model
- Vectorized mode DISABLED (py_vollib_vectorized API incompatibility)
- Sequential calculation works fine for typical volumes

### Symbol Discovery (discovery.py)

Two modes:
1. **Default:** ~100 symbols from config list
2. **Full Scan:** Fetches S&P 500 + Nasdaq 100 from Wikipedia + core ETFs (~600 symbols)

Wikipedia fetching requires User-Agent header to avoid 403 errors.

## Known Issues and Fixes Applied

1. **Greeks vectorized calculation warning** - Disabled vectorized mode, using sequential
2. **HTTP 403 from Wikipedia** - Added User-Agent header to requests
3. **Stale options data (e.g., SPXS post-split)** - Added comprehensive validation:
   - Strike proximity check
   - ATM premium sanity check
   - ITM intrinsic value check

## Configuration Defaults (core/config.py)

```python
TradeCriteria:
    min_weekly_return_pct: 1.0
    min_prob_profit: 0.70
    max_dte: 5
    min_dte: 1
    max_delta: 0.30
    min_open_interest: 100
    min_volume: 50
    max_bid_ask_spread_pct: 5.0

CapitalConfig:
    total_capital: 13000.0
    max_single_position_pct: 0.25
    max_total_exposure_pct: 0.80

UnderlyingConfig:
    max_share_price: 120.0
```

## Dependencies (requirements.txt)

- yfinance
- pandas
- numpy
- py_vollib
- scipy
- rich (for terminal UI)
- requests
- lxml

## Potential Future Enhancements

1. **Enable Covered Calls** - Strategy code exists, just needs CLI integration
2. **Real-time data** - yfinance is delayed; could integrate paid providers (IBKR, Polygon, etc.)
3. **Backtesting module** - Test strategies on historical data
4. **Portfolio tracking** - Track open positions and P&L
5. **Alerts/notifications** - When opportunities matching criteria appear
6. **Web UI** - Flask/FastAPI dashboard instead of CLI
7. **More strategies** - Strangles, straddles (if risk tolerance changes)
8. **Earnings calendar integration** - Avoid/target earnings plays
9. **Sector rotation** - Weight opportunities by sector performance

## User Preferences

- Prefers premium selling (theta decay strategies)
- Risk-averse: defined risk only, no naked options
- Short-term focus: 5 DTE max
- High probability: 70%+ POP minimum
- Capital constrained: $13k, so needs affordable underlyings
