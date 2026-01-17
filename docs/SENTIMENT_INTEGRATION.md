# News Sentiment Analysis Integration

## Overview

This document summarizes the news sentiment analysis feature added to the options analyzer, designed to serve as a **risk filter** for options premium selling strategies.

## What Was Implemented

### Phase 1: Foundation (Complete)

| File | Purpose | Lines |
|------|---------|-------|
| `core/models.py` | Added `SentimentSignal` dataclass | +62 |
| `data/news_fetcher.py` | News fetching with yfinance + file caching | ~250 |
| `analysis/sentiment.py` | FinBERT sentiment analysis + aggregation | ~450 |

### Phase 2: CLI Integration (Complete)

| File | Changes |
|------|---------|
| `main.py` | Added `--sentiment` and `--sentiment-lookback` flags |
| `ui/display.py` | Added `display_sentiment_summary()`, made table columns dynamic |

### Phase 3: QML Integration (Complete)

Added sentiment as a 7th feature to the quantum scorer circuit.

| File | Changes |
|------|---------|
| `analysis/quantum_scorer.py` | Added `sentiment_score` to FeatureExtractor, updated n_qubits to 7 |
| `analysis/qml_integration.py` | Added `set_sentiment()` method, n_qubits default to 7 |
| `main.py` | Pass sentiment signals to QML scorer when both flags enabled |

## Architecture

```
data/
  news_fetcher.py       # NewsArticle, NewsCache, YFinanceNewsSource
                        # 1-hour TTL file cache in .news_cache/

analysis/
  sentiment.py          # FinBERTAnalyzer, FallbackAnalyzer, SentimentAnalyzer
                        # Uses ProsusAI/finbert (89% accuracy on financial text)

core/models.py
  SentimentSignal       # Aggregated sentiment with risk filtering logic
```

## Key Design Decisions

### 1. Risk Filter, Not Directional Predictor

Research shows sentiment is most useful for detecting **event risk**, not predicting short-term direction. The system outputs risk levels instead of buy/sell signals:

| Risk Level | Meaning | Action |
|------------|---------|--------|
| `low_risk` | Stable sentiment, normal news volume | Good for premium selling |
| `elevated` | Low confidence or shifting sentiment | Caution advised |
| `high_risk` | News volume spike (z-score > 2) | Avoid or hedge |

### 2. FinBERT Over VADER

- **FinBERT**: 89% accuracy on financial text, ~440MB model
- **VADER**: 44-56% accuracy, but 339x faster
- Decision: Use FinBERT for accuracy, with VADER fallback if dependencies unavailable

### 3. Time-Weighted Aggregation

- Newer articles weighted more heavily (24-hour half-life exponential decay)
- Confidence based on article count AND sentiment agreement
- Disagreement between articles reduces overall confidence

## SentimentSignal Dataclass

```python
@dataclass
class SentimentSignal:
    symbol: str
    sentiment_score: float      # -1 (bearish) to +1 (bullish)
    confidence: float           # 0-1, based on article count and agreement

    # Risk filter signals
    news_volume: int            # Number of articles in window
    news_volume_zscore: float   # >2 = unusual activity (event risk)
    sentiment_momentum: float   # Current - 24h ago (trend shift)

    # Metadata
    article_count: int
    avg_relevance: float
    dominant_sentiment: str     # "positive", "negative", "neutral"
    top_headlines: List[str]

    @property
    def risk_flag(self) -> str:
        """Returns 'low_risk', 'elevated', or 'high_risk'"""
```

## CLI Usage

```bash
# Enable sentiment analysis
python main.py --sentiment -s SPY QQQ IWM

# Custom lookback window (default 48 hours)
python main.py --sentiment --sentiment-lookback 24 -s SPY

# Combined with QML scoring
python main.py --sentiment --qml -s SPY QQQ IWM
```

## Dependencies Added

```bash
pip install transformers torch
```

- `transformers`: For FinBERT model loading
- `torch`: For model inference (already required for QML)

The system gracefully falls back to a keyword-based analyzer if dependencies are unavailable.

## Phase 3: QML Integration (Complete)

Sentiment was added as a 7th feature to the quantum scorer using Option A (add 7th qubit).

### Implementation Details

**FeatureExtractor** (in `analysis/quantum_scorer.py`):

```python
class FeatureExtractor:
    FEATURE_NAMES = [
        'prob_profit',
        'weekly_return',
        'net_delta',
        'theta_ratio',
        'iv_rank',
        'dte_norm',
        'sentiment_score',  # NEW: -1 to +1 scaled to 0-1
    ]

    def set_sentiment(self, sentiment_signals: Dict[str, Any]) -> None:
        """Set sentiment signals for feature extraction."""
        # Maps symbol -> sentiment_score for lookup during extraction
```

**QuantumScorerConfig**:
```python
n_qubits: int = 7  # One per feature (was 6)
```

### How It Works

1. When `--sentiment` and `--qml` are both enabled, sentiment signals are passed to the QML scorer
2. The FeatureExtractor looks up each candidate's symbol to get its sentiment score
3. Sentiment score (-1 to +1) is scaled to (0 to 1) for the quantum circuit
4. If no sentiment available for a symbol, defaults to neutral (0.5 after scaling)

### Usage

```bash
# Enable both sentiment and QML for integrated scoring
python main.py --sentiment --qml -s SPY QQQ IWM
```

### Notes

- Training uses historical backtest data where sentiment defaults to neutral
- Live analysis uses real-time sentiment from news
- The circuit learns non-linear interactions between sentiment and other features

## File Locations

- News cache: `.news_cache/` (1-hour TTL)
- FinBERT model cache: `~/.cache/huggingface/hub/models--ProsusAI--finbert/`

## Testing

```python
# Quick test
from analysis.sentiment import analyze_sentiment

signals = analyze_sentiment(['SPY', 'QQQ', 'AAPL'])
for symbol, signal in signals.items():
    print(f"{symbol}: {signal.display_score}, Risk: {signal.risk_flag}")
```

## Output Example

```
╭─────────────────────────────────────────────────────────────────────────────╮
│ Sentiment Risk Filter                                                       │
╰─────────────────────────────────────────────────────────────────────────────╯
┌────────┬───────────┬────────────┬──────────┬──────────┬──────────┬──────────┐
│ Symbol │ Sentiment │ Confidence │   Risk   │ News Vol │ Dominant │ Headline │
├────────┼───────────┼────────────┼──────────┼──────────┼──────────┼──────────┤
│ QQQ    │   +0.31   │        45% │ Elevated │        9 │ Positive │ Put op...│
│ SPY    │   -0.02   │        43% │   Low    │        5 │ Negative │ Invest...│
└────────┴───────────┴────────────┴──────────┴──────────┴──────────┴──────────┘
```

## Commit Reference

```
8e7114a Add news sentiment analysis with FinBERT risk filter
```
