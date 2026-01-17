"""
News fetching module for sentiment analysis.

Provides abstracted news sources with caching to minimize API calls.
Primary source is yfinance (already a dependency), with extensibility
for RSS feeds and other APIs.

Usage:
    from data.news_fetcher import get_news_for_symbols

    articles = get_news_for_symbols(['SPY', 'QQQ', 'AAPL'])
    for article in articles:
        print(f"{article.symbol}: {article.title}")
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a single news article."""
    symbol: str  # Ticker this article relates to
    title: str
    summary: str
    source: str  # e.g., "Reuters", "Bloomberg", "24/7 Wall St."
    url: str
    published_at: datetime

    # Metadata
    article_id: str = ""
    relevance_score: float = 1.0  # How relevant to the symbol (0-1)

    def __post_init__(self):
        if not self.article_id:
            # Generate ID from title hash
            self.article_id = hashlib.md5(
                f"{self.symbol}:{self.title}".encode()
            ).hexdigest()[:12]


@dataclass
class NewsCacheEntry:
    """Cache entry for news articles."""
    symbol: str
    articles: List[Dict[str, Any]]
    fetched_at: str  # ISO format
    source: str


class NewsCache:
    """
    File-based cache for news articles.

    Caches news per symbol with configurable TTL to minimize API calls.
    """

    def __init__(self, cache_dir: Path = None, ttl_minutes: int = 60):
        self.cache_dir = cache_dir or Path(".news_cache")
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_news.json"

    def get(self, symbol: str) -> Optional[List[NewsArticle]]:
        """Get cached articles if not expired."""
        path = self._cache_path(symbol)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            fetched_at = datetime.fromisoformat(data['fetched_at'])
            if datetime.now() - fetched_at > self.ttl:
                logger.debug(f"Cache expired for {symbol}")
                return None

            articles = []
            for item in data['articles']:
                articles.append(NewsArticle(
                    symbol=item['symbol'],
                    title=item['title'],
                    summary=item['summary'],
                    source=item['source'],
                    url=item['url'],
                    published_at=datetime.fromisoformat(item['published_at']),
                    article_id=item.get('article_id', ''),
                    relevance_score=item.get('relevance_score', 1.0),
                ))

            logger.debug(f"Cache hit for {symbol}: {len(articles)} articles")
            return articles

        except Exception as e:
            logger.warning(f"Error reading cache for {symbol}: {e}")
            return None

    def set(self, symbol: str, articles: List[NewsArticle], source: str):
        """Cache articles for a symbol."""
        path = self._cache_path(symbol)

        data = {
            'symbol': symbol,
            'fetched_at': datetime.now().isoformat(),
            'source': source,
            'articles': [
                {
                    'symbol': a.symbol,
                    'title': a.title,
                    'summary': a.summary,
                    'source': a.source,
                    'url': a.url,
                    'published_at': a.published_at.isoformat(),
                    'article_id': a.article_id,
                    'relevance_score': a.relevance_score,
                }
                for a in articles
            ]
        }

        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Cached {len(articles)} articles for {symbol}")
        except Exception as e:
            logger.warning(f"Error writing cache for {symbol}: {e}")

    def clear(self, symbol: str = None):
        """Clear cache for a symbol or all symbols."""
        if symbol:
            path = self._cache_path(symbol)
            if path.exists():
                path.unlink()
        else:
            for path in self.cache_dir.glob("*_news.json"):
                path.unlink()


class NewsSource(ABC):
    """Abstract base class for news sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this news source."""
        pass

    @abstractmethod
    def fetch(self, symbol: str, max_articles: int = 10) -> List[NewsArticle]:
        """Fetch news articles for a symbol."""
        pass


class YFinanceNewsSource(NewsSource):
    """
    Fetch news from Yahoo Finance via yfinance.

    This is the primary news source since yfinance is already a dependency.
    Provides 10-20 recent articles per ticker with title, summary, and source.
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, symbol: str, max_articles: int = 10) -> List[NewsArticle]:
        """Fetch news for a symbol from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news_data = ticker.news

            if not news_data:
                logger.debug(f"No news found for {symbol}")
                return []

            articles = []
            for item in news_data[:max_articles]:
                try:
                    content = item.get('content', {})
                    if not content:
                        continue

                    # Parse publication date
                    pub_date_str = content.get('pubDate', '')
                    if pub_date_str:
                        # yfinance returns ISO format with Z suffix
                        pub_date_str = pub_date_str.replace('Z', '+00:00')
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str)
                        except ValueError:
                            pub_date = datetime.now()
                    else:
                        pub_date = datetime.now()

                    # Get provider info
                    provider = content.get('provider', {})
                    source_name = provider.get('displayName', 'Unknown')

                    # Build article
                    article = NewsArticle(
                        symbol=symbol,
                        title=content.get('title', ''),
                        summary=content.get('summary', content.get('description', '')),
                        source=source_name,
                        url=content.get('canonicalUrl', content.get('clickThroughUrl', '')),
                        published_at=pub_date,
                        article_id=item.get('id', ''),
                    )

                    if article.title:  # Only add if we have a title
                        articles.append(article)

                except Exception as e:
                    logger.debug(f"Error parsing article for {symbol}: {e}")
                    continue

            logger.info(f"Fetched {len(articles)} articles for {symbol} from yfinance")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []


class NewsFetcher:
    """
    Main interface for fetching news with caching.

    Orchestrates multiple news sources and manages caching.

    Usage:
        fetcher = NewsFetcher()
        articles = fetcher.fetch_for_symbols(['SPY', 'QQQ'])
    """

    def __init__(
        self,
        cache_ttl_minutes: int = 60,
        cache_dir: Path = None,
        sources: List[NewsSource] = None
    ):
        self.cache = NewsCache(
            cache_dir=cache_dir or Path(".news_cache"),
            ttl_minutes=cache_ttl_minutes
        )
        self.sources = sources or [YFinanceNewsSource()]

    def fetch(self, symbol: str, max_articles: int = 10, use_cache: bool = True) -> List[NewsArticle]:
        """
        Fetch news for a single symbol.

        Args:
            symbol: Ticker symbol
            max_articles: Maximum articles to return
            use_cache: Whether to use cached results

        Returns:
            List of NewsArticle objects
        """
        symbol = symbol.upper()

        # Check cache first
        if use_cache:
            cached = self.cache.get(symbol)
            if cached is not None:
                return cached[:max_articles]

        # Fetch from sources
        articles = []
        for source in self.sources:
            try:
                source_articles = source.fetch(symbol, max_articles)
                articles.extend(source_articles)

                if articles:
                    # Cache successful fetch
                    self.cache.set(symbol, articles, source.name)
                    break  # Got articles, no need to try other sources

            except Exception as e:
                logger.warning(f"Source {source.name} failed for {symbol}: {e}")
                continue

        return articles[:max_articles]

    def fetch_for_symbols(
        self,
        symbols: List[str],
        max_articles_per_symbol: int = 10,
        use_cache: bool = True
    ) -> Dict[str, List[NewsArticle]]:
        """
        Fetch news for multiple symbols.

        Args:
            symbols: List of ticker symbols
            max_articles_per_symbol: Max articles per symbol
            use_cache: Whether to use cached results

        Returns:
            Dict mapping symbol to list of articles
        """
        results = {}
        for symbol in symbols:
            articles = self.fetch(symbol, max_articles_per_symbol, use_cache)
            results[symbol.upper()] = articles

        return results

    def clear_cache(self, symbol: str = None):
        """Clear news cache."""
        self.cache.clear(symbol)


# Convenience function
def get_news_for_symbols(
    symbols: List[str],
    max_articles_per_symbol: int = 10,
    cache_ttl_minutes: int = 60
) -> Dict[str, List[NewsArticle]]:
    """
    Convenience function to fetch news for multiple symbols.

    Args:
        symbols: List of ticker symbols
        max_articles_per_symbol: Max articles per symbol
        cache_ttl_minutes: Cache TTL in minutes

    Returns:
        Dict mapping symbol to list of NewsArticle objects
    """
    fetcher = NewsFetcher(cache_ttl_minutes=cache_ttl_minutes)
    return fetcher.fetch_for_symbols(symbols, max_articles_per_symbol)
