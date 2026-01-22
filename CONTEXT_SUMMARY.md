# Options Analyzer - Project Context Summary

## Project Overview

An options analysis tool for premium selling strategies designed for a trader with:
- **$13,000 initial capital**
- **1% weekly profit target**
- **5 DTE maximum**
- **70% probability of profit minimum**
- **Collateral-based filtering** (trades filtered by max loss vs available capital, not share price)

## Supported Strategies

1. **Cash-Secured Puts (CSP)** - Sell OTM puts with cash collateral = strike × 100
2. **Put Credit Spreads** - Sell OTM put, buy further OTM put (defined risk)
3. **Call Credit Spreads** - Sell OTM call, buy further OTM call (defined risk)
4. **Iron Condors** - Combine put spread + call spread for neutral outlook (defined risk)
5. **Covered Calls** - Strategy exists in code but NOT enabled in CLI (requires stock ownership tracking)

**NOT supported (by design):** Naked calls, naked puts

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
│   ├── fetcher.py          # yfinance data fetching with retry logic
│   ├── discovery.py        # Symbol discovery (S&P 500, Nasdaq 100, ETFs)
│   ├── synthetic_chain.py  # After-hours synthetic pricing
│   └── news_fetcher.py     # News data fetching
├── analysis/
│   ├── greeks.py           # Black-Scholes + Bjerksund-Stensland (American options)
│   ├── volatility_surface.py # IV surface analysis, skew detection
│   ├── position_sizer.py   # Position sizing logic
│   ├── risk_metrics.py     # CVaR, earnings calendar, liquidity scoring (NEW)
│   ├── sentiment.py        # News sentiment analysis (FinBERT)
│   ├── quantum_scorer.py   # VQC-based trade scoring
│   └── qml_integration.py  # QML CLI integration
├── strategies/
│   ├── base.py             # BaseStrategy + EnhancedScorer (NEW)
│   ├── secured_premium.py  # CSP and Covered Call strategies
│   └── credit_spreads.py   # Put/Call credit spreads, Iron Condors
├── backtesting/            # Historical backtesting
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
python main.py --capital 15000 --prob 0.75 --max-dte 7

# Optional: Add upfront share price filtering
python main.py --max-price 100
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

### Data Fetching Reliability (fetcher.py)

1. **Retry Logic**:
   - Exponential backoff with 3 retries
   - Base delay 1s, max delay 30s
   - Jitter to prevent thundering herd

2. **Live Market Data**:
   - Risk-free rate from 10Y Treasury (^TNX)
   - Dividend yields per symbol
   - 4-hour cache TTL

### Data Validation (fetcher.py)

The system has multiple layers to catch stale/invalid data:

1. **Price Fetching Priority**:
   - Intraday 1-minute data (most current)
   - fast_info
   - info dict
   - 5-day history (fallback)

2. **Strike Price Sanity Check**:
   - Strikes must be 30%-300% of current price
   - Filters out pre-split stale data

3. **Options Chain Consistency Validation**:
   - Must have strikes within 20% of current price
   - ATM premium can't exceed 30% of stock price
   - ITM options must have at least 50% of intrinsic value

4. **Data Freshness Check**:
   - Warns if last trade > 1 day ago (weekdays) or > 3 days (weekends)

### Greeks Calculation (greeks.py) - ENHANCED

- **Black-Scholes** for European-style Greeks
- **Bjerksund-Stensland (2002)** for American options pricing (NEW)
- **Live risk-free rate** from 10Y Treasury (NEW)
- **Dividend yield support** for accurate pricing (NEW)
- **Higher-order Greeks**: Vanna, Charm, Vomma available (NEW)
- Vectorized mode DISABLED (API incompatibility)

### Risk Metrics (risk_metrics.py) - NEW

1. **Earnings Calendar**:
   - Fetches earnings dates from yfinance
   - Flags trades with earnings in window
   - Risk levels: "low", "elevated", "high"

2. **CVaR (Conditional Value at Risk)**:
   - Expected Shortfall at 95% and 99% confidence
   - Uses 252-day historical returns
   - Better tail risk measure than VaR

3. **Enhanced Liquidity Score**:
   - Spread score (40%): 5% spread = 0, 0% = 100
   - OI score (35%): 1000+ OI = 100
   - Volume score (25%): 500+ volume = 100

### Enhanced Scoring System (base.py) - NEW

**Base Score (0-100):**
```
Score = (
    weekly_return_score × 25% +
    prob_profit_score × 25% +
    liquidity_score × 20% +
    iv_rank_score × 15% +
    theta_efficiency × 15%
)
```

**Risk Multipliers:**
- Earnings in window: 0.5×
- Earnings within 3 days: 0.7×
- Stressed market regime: 0.7×
- High CVaR (>5%): 0.7-1.0× (graduated)

### Symbol Discovery (discovery.py)

Two modes:
1. **Default:** ~100 symbols from config list
2. **Full Scan:** Fetches S&P 500 + Nasdaq 100 from Wikipedia + core ETFs (~600 symbols)

Wikipedia fetching requires User-Agent header to avoid 403 errors.

## Known Issues and Fixes Applied

1. **Greeks vectorized calculation warning** - Disabled vectorized mode, using sequential
2. **HTTP 403 from Wikipedia** - Added User-Agent header to requests
3. **Stale options data (e.g., SPXS post-split)** - Added comprehensive validation
4. **API failures** - Added exponential backoff retry logic
5. **European vs American options** - Added Bjerksund-Stensland model
6. **HTTP 404 for ETF calendars** - Suppressed yfinance error logging for expected failures

## Configuration Defaults (core/config.py)

```python
TradeCriteria:
    min_trade_return_pct: 1.0
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
    max_share_price: float('inf')  # No upfront price filter by default
    # Trades filtered by collateral (max loss) vs available capital instead
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

## Recent Enhancements (v2.0)

### Completed
- ✅ Exponential backoff retry logic (3 retries, 1-30s delays)
- ✅ Live risk-free rate from Treasury
- ✅ Dividend yield fetching per symbol
- ✅ Bjerksund-Stensland American options pricing
- ✅ Earnings calendar integration
- ✅ CVaR risk metric (Expected Shortfall)
- ✅ Enhanced liquidity scoring (40/35/25 weighted)
- ✅ IV Rank/Percentile calculation
- ✅ New composite scoring system with risk multipliers

### Future Enhancements
1. **Market Regime Detection** - HMM-based regime classification
2. **Portfolio Correlation** - Sector exposure limits
3. **ORATS Integration** - Professional-grade IV analytics
4. **Backtest Validation** - Empirical weight optimization

## User Preferences

- Prefers premium selling (theta decay strategies)
- Risk-averse: defined risk only, no naked options
- Short-term focus: 5 DTE max
- High probability: 70%+ POP minimum
- Capital constrained: $13k, trades filtered by collateral vs capital (allows credit spreads on expensive stocks)
