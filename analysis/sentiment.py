"""
Sentiment analysis module using FinBERT.

Provides financial-domain sentiment analysis optimized for use as a
risk filter in options trading. Uses ProsusAI/finbert, a BERT model
fine-tuned on financial text with 89% accuracy.

Key Features:
- FinBERT for high-accuracy financial sentiment
- Time-weighted aggregation for recent news priority
- News volume anomaly detection for event risk
- Graceful fallback when FinBERT unavailable

Usage:
    from analysis.sentiment import SentimentAnalyzer, analyze_sentiment

    # Quick usage
    signals = analyze_sentiment(['SPY', 'QQQ', 'AAPL'])

    # Full control
    analyzer = SentimentAnalyzer()
    signal = analyzer.analyze_symbol('SPY')
    print(f"SPY sentiment: {signal.display_score}, Risk: {signal.risk_flag}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np

from core.models import SentimentSignal
from data.news_fetcher import NewsArticle, NewsFetcher, get_news_for_symbols

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning(
        "transformers/torch not installed. Sentiment analysis will use fallback. "
        "Install with: pip install transformers torch"
    )


@dataclass
class ArticleSentiment:
    """Sentiment result for a single article."""
    article_id: str
    title: str
    sentiment: str  # "positive", "negative", "neutral"
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    published_at: datetime


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer.

    Uses ProsusAI/finbert, a BERT model fine-tuned on financial text.
    Achieves ~89% accuracy on financial sentiment classification.

    The model outputs probabilities for positive, negative, and neutral,
    which we convert to a -1 to +1 score.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = None):
        """
        Initialize FinBERT model.

        Args:
            device: 'cuda', 'cpu', or None for auto-detect
        """
        if not FINBERT_AVAILABLE:
            raise RuntimeError(
                "FinBERT requires transformers and torch. "
                "Install with: pip install transformers torch"
            )

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        logger.info(f"Loading FinBERT model on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(device)
        self.model.eval()
        logger.info("FinBERT model loaded")

        # Label mapping (FinBERT outputs: positive, negative, neutral)
        self.labels = ['positive', 'negative', 'neutral']

    def analyze(self, text: str) -> Tuple[str, float, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze (headline or summary)

        Returns:
            Tuple of (sentiment_label, score, confidence)
            - sentiment_label: "positive", "negative", or "neutral"
            - score: -1 (bearish) to +1 (bullish)
            - confidence: 0-1 probability of predicted class
        """
        # Truncate long text to model max length
        text = text[:512]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # Get predicted class
        pred_idx = probs.argmax().item()
        sentiment = self.labels[pred_idx]
        confidence = probs[pred_idx].item()

        # Convert to -1 to +1 score
        # positive prob contributes positively, negative prob contributes negatively
        pos_prob = probs[0].item()  # positive
        neg_prob = probs[1].item()  # negative
        score = pos_prob - neg_prob  # Range: -1 to +1

        return sentiment, score, confidence

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float, float]]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze

        Returns:
            List of (sentiment_label, score, confidence) tuples
        """
        if not texts:
            return []

        # Truncate texts
        texts = [t[:512] for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        results = []
        for i in range(len(texts)):
            pred_idx = probs[i].argmax().item()
            sentiment = self.labels[pred_idx]
            confidence = probs[i][pred_idx].item()

            pos_prob = probs[i][0].item()
            neg_prob = probs[i][1].item()
            score = pos_prob - neg_prob

            results.append((sentiment, score, confidence))

        return results


class FallbackAnalyzer:
    """
    Simple rule-based fallback when FinBERT is unavailable.

    Uses basic keyword matching - much less accurate than FinBERT
    but provides some signal when dependencies aren't installed.
    """

    POSITIVE_WORDS = {
        'surge', 'soar', 'jump', 'gain', 'rise', 'rally', 'bull', 'bullish',
        'upgrade', 'beat', 'exceed', 'strong', 'growth', 'profit', 'win',
        'positive', 'optimistic', 'boom', 'breakthrough', 'success'
    }

    NEGATIVE_WORDS = {
        'drop', 'fall', 'plunge', 'crash', 'decline', 'bear', 'bearish',
        'downgrade', 'miss', 'weak', 'loss', 'fail', 'negative', 'pessimistic',
        'recession', 'crisis', 'warning', 'risk', 'concern', 'fear', 'sell'
    }

    def analyze(self, text: str) -> Tuple[str, float, float]:
        """Analyze using keyword matching."""
        text_lower = text.lower()
        words = set(text_lower.split())

        pos_count = len(words & self.POSITIVE_WORDS)
        neg_count = len(words & self.NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            return 'neutral', 0.0, 0.3  # Low confidence for neutral

        score = (pos_count - neg_count) / total
        confidence = min(0.5, total * 0.1)  # Cap at 0.5 for fallback

        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, score, confidence

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float, float]]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]


class SentimentAnalyzer:
    """
    Main sentiment analysis interface.

    Orchestrates news fetching, FinBERT analysis, and aggregation
    into actionable SentimentSignal objects for risk filtering.

    Usage:
        analyzer = SentimentAnalyzer()
        signal = analyzer.analyze_symbol('SPY')

        if signal.risk_flag == "high_risk":
            print("Event risk detected, consider avoiding")
    """

    def __init__(
        self,
        use_finbert: bool = True,
        cache_ttl_minutes: int = 60,
        lookback_hours: int = 48,
        device: str = None
    ):
        """
        Initialize sentiment analyzer.

        Args:
            use_finbert: Use FinBERT if available (recommended)
            cache_ttl_minutes: News cache TTL
            lookback_hours: How far back to consider news
            device: 'cuda', 'cpu', or None for auto
        """
        self.news_fetcher = NewsFetcher(cache_ttl_minutes=cache_ttl_minutes)
        self.lookback = timedelta(hours=lookback_hours)

        # Initialize sentiment model
        if use_finbert and FINBERT_AVAILABLE:
            try:
                self.model = FinBERTAnalyzer(device=device)
                self.model_name = "finbert"
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}, using fallback")
                self.model = FallbackAnalyzer()
                self.model_name = "fallback"
        else:
            self.model = FallbackAnalyzer()
            self.model_name = "fallback"

        # Historical data for z-score calculation
        self._volume_history: Dict[str, List[int]] = {}

    def analyze_symbol(
        self,
        symbol: str,
        max_articles: int = 15
    ) -> SentimentSignal:
        """
        Analyze sentiment for a single symbol.

        Args:
            symbol: Ticker symbol
            max_articles: Max articles to analyze

        Returns:
            SentimentSignal with aggregated sentiment and risk flags
        """
        symbol = symbol.upper()

        # Fetch news
        articles = self.news_fetcher.fetch(symbol, max_articles)

        if not articles:
            # No news - return neutral with low confidence
            return SentimentSignal(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                news_volume=0,
                news_volume_zscore=0.0,
                sentiment_momentum=0.0,
                article_count=0,
                avg_relevance=0.0,
                dominant_sentiment="neutral",
                top_headlines=[]
            )

        # Filter to lookback window
        # Handle timezone-aware datetimes from yfinance
        now = datetime.now()
        cutoff = now - self.lookback

        def is_recent(article):
            pub = article.published_at
            # Make comparison timezone-naive if needed
            if pub.tzinfo is not None:
                pub = pub.replace(tzinfo=None)
            return pub > cutoff

        recent_articles = [a for a in articles if is_recent(a)]

        if not recent_articles:
            recent_articles = articles[:5]  # Use most recent if all old

        # Analyze sentiment for each article
        texts = [f"{a.title}. {a.summary}" for a in recent_articles]
        sentiments = self.model.analyze_batch(texts)

        # Build article sentiments with time weights
        article_sentiments = []

        for article, (sent_label, score, conf) in zip(recent_articles, sentiments):
            # Time decay: newer articles weighted more heavily
            pub = article.published_at
            if pub.tzinfo is not None:
                pub = pub.replace(tzinfo=None)
            age_hours = (now - pub).total_seconds() / 3600
            time_weight = np.exp(-age_hours / 24)  # Half-life of 24 hours

            article_sentiments.append({
                'article': article,
                'sentiment': sent_label,
                'score': score,
                'confidence': conf,
                'time_weight': time_weight,
            })

        # Aggregate sentiment with time weighting
        total_weight = sum(a['time_weight'] * a['confidence'] for a in article_sentiments)
        if total_weight > 0:
            weighted_score = sum(
                a['score'] * a['time_weight'] * a['confidence']
                for a in article_sentiments
            ) / total_weight
        else:
            weighted_score = 0.0

        # Calculate confidence based on article count and agreement
        confidences = [a['confidence'] for a in article_sentiments]
        scores = [a['score'] for a in article_sentiments]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Reduce confidence if sentiments disagree
        if len(scores) > 1:
            score_std = np.std(scores)
            agreement_factor = max(0.5, 1.0 - score_std)
        else:
            agreement_factor = 0.7

        final_confidence = avg_confidence * agreement_factor

        # Determine dominant sentiment
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for a in article_sentiments:
            sentiment_counts[a['sentiment']] += 1
        dominant = max(sentiment_counts, key=sentiment_counts.get)

        # Calculate news volume z-score for event detection
        news_volume = len(recent_articles)
        volume_zscore = self._calculate_volume_zscore(symbol, news_volume)

        # Calculate sentiment momentum (would need historical sentiment for real momentum)
        # For now, use variance in recent scores as proxy
        if len(scores) > 2:
            recent_scores = scores[:len(scores)//2]
            older_scores = scores[len(scores)//2:]
            momentum = np.mean(recent_scores) - np.mean(older_scores)
        else:
            momentum = 0.0

        # Get top headlines with URLs for transparency
        top_headlines = [(a['article'].title, a['article'].url) for a in article_sentiments[:5]]

        return SentimentSignal(
            symbol=symbol,
            sentiment_score=float(weighted_score),
            confidence=float(final_confidence),
            news_volume=news_volume,
            news_volume_zscore=float(volume_zscore),
            sentiment_momentum=float(momentum),
            article_count=len(article_sentiments),
            avg_relevance=np.mean([a['article'].relevance_score for a in article_sentiments]),
            dominant_sentiment=dominant,
            top_headlines=top_headlines,
        )

    def analyze_symbols(
        self,
        symbols: List[str],
        max_articles_per_symbol: int = 15
    ) -> Dict[str, SentimentSignal]:
        """
        Analyze sentiment for multiple symbols.

        Args:
            symbols: List of ticker symbols
            max_articles_per_symbol: Max articles per symbol

        Returns:
            Dict mapping symbol to SentimentSignal
        """
        results = {}
        for symbol in symbols:
            try:
                signal = self.analyze_symbol(symbol, max_articles_per_symbol)
                results[symbol.upper()] = signal
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                # Return neutral signal on error
                results[symbol.upper()] = SentimentSignal(
                    symbol=symbol.upper(),
                    sentiment_score=0.0,
                    confidence=0.0,
                    news_volume=0,
                    news_volume_zscore=0.0,
                    sentiment_momentum=0.0,
                    article_count=0,
                    avg_relevance=0.0,
                    dominant_sentiment="neutral",
                    top_headlines=[]
                )

        return results

    def _calculate_volume_zscore(self, symbol: str, current_volume: int) -> float:
        """
        Calculate z-score of news volume for event detection.

        A z-score > 2 suggests unusual news activity (potential event).
        """
        # Initialize history if needed
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []

        history = self._volume_history[symbol]

        # Add current volume to history
        history.append(current_volume)

        # Keep last 30 observations
        if len(history) > 30:
            history.pop(0)

        # Need at least 5 observations for meaningful z-score
        if len(history) < 5:
            return 0.0

        mean = np.mean(history[:-1])  # Exclude current
        std = np.std(history[:-1])

        if std == 0:
            return 0.0

        return (current_volume - mean) / std

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the sentiment model being used."""
        return {
            'model_name': self.model_name,
            'finbert_available': FINBERT_AVAILABLE,
            'device': getattr(self.model, 'device', 'cpu'),
        }


# Module-level singleton for convenience
_global_analyzer: Optional[SentimentAnalyzer] = None


def get_analyzer(use_finbert: bool = True, device: str = None) -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer."""
    global _global_analyzer

    if _global_analyzer is None:
        _global_analyzer = SentimentAnalyzer(use_finbert=use_finbert, device=device)

    return _global_analyzer


def analyze_sentiment(
    symbols: List[str],
    max_articles_per_symbol: int = 15,
    use_finbert: bool = True
) -> Dict[str, SentimentSignal]:
    """
    Convenience function to analyze sentiment for multiple symbols.

    Args:
        symbols: List of ticker symbols
        max_articles_per_symbol: Max articles per symbol
        use_finbert: Use FinBERT if available

    Returns:
        Dict mapping symbol to SentimentSignal

    Example:
        signals = analyze_sentiment(['SPY', 'QQQ', 'AAPL'])
        for symbol, signal in signals.items():
            print(f"{symbol}: {signal.display_score}, Risk: {signal.risk_flag}")
    """
    analyzer = get_analyzer(use_finbert=use_finbert)
    return analyzer.analyze_symbols(symbols, max_articles_per_symbol)
