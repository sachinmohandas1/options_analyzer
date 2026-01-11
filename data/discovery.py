"""
Symbol Discovery Module.

Discovers symbols and fetches current prices for options analysis.
By default, no upfront price filtering is applied - trades are filtered
by collateral (max loss) vs available capital at the strategy level.
Optional max_share_price filter can be enabled via --max-price CLI flag.

Supports two modes:
1. Default: Use symbols from config list (~100 symbols)
2. Full scan: Fetch S&P 500, Nasdaq 100, ETFs (~600 symbols)
"""

import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from core.config import AnalyzerConfig

# User-Agent header to avoid 403 errors from Wikipedia
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

logger = logging.getLogger(__name__)


# URLs for index constituent lists (Wikipedia)
INDEX_URLS = {
    'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    'nasdaq100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
    'russell2000': None,  # Not easily available, use other sources
}

# Common high-volume ETFs to always include
CORE_ETFS = [
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE",
    # Commodities
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
    # Bonds
    "TLT", "IEF", "HYG", "LQD", "JNK", "BND",
    # International
    "EEM", "EFA", "FXI", "EWZ", "EWJ", "VWO", "KWEB",
    # High IV ETFs
    "ARKK", "ARKG", "XBI", "BITO",
    # Leveraged
    "SOXL", "TQQQ", "SPXL", "TNA", "FAS",
    # VIX/Inverse
    "UVXY", "VXX", "SQQQ", "SPXS",
]


def fetch_sp500_symbols() -> List[str]:
    """Fetch S&P 500 constituent symbols from Wikipedia."""
    try:
        response = requests.get(INDEX_URLS['sp500'], headers=HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        # Symbol column, replace . with - for Yahoo Finance compatibility
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        logger.info(f"Fetched {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 list: {e}")
        return []


def fetch_nasdaq100_symbols() -> List[str]:
    """Fetch Nasdaq 100 constituent symbols from Wikipedia."""
    try:
        response = requests.get(INDEX_URLS['nasdaq100'], headers=HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        # Nasdaq 100 table structure varies, find the one with Ticker column
        for df in tables:
            if 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                logger.info(f"Fetched {len(symbols)} Nasdaq 100 symbols")
                return symbols
            elif 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
                logger.info(f"Fetched {len(symbols)} Nasdaq 100 symbols")
                return symbols
        logger.warning("Could not find Nasdaq 100 ticker column")
        return []
    except Exception as e:
        logger.warning(f"Could not fetch Nasdaq 100 list: {e}")
        return []


def fetch_all_index_symbols() -> Set[str]:
    """Fetch symbols from all major indices."""
    all_symbols: Set[str] = set()

    # Always include core ETFs
    all_symbols.update(CORE_ETFS)

    # Fetch S&P 500
    sp500 = fetch_sp500_symbols()
    all_symbols.update(sp500)

    # Fetch Nasdaq 100
    nasdaq = fetch_nasdaq100_symbols()
    all_symbols.update(nasdaq)

    logger.info(f"Total unique symbols from indices: {len(all_symbols)}")
    return all_symbols


class SymbolDiscovery:
    """
    Discovers tradeable symbols by:
    1. Taking seed symbols from config OR fetching index constituents
    2. Fetching current prices
    3. Optionally filtering by max_share_price (if set)
    4. Checking for options availability

    By default, no price filtering is applied. Trades are filtered by
    collateral (max loss) vs available capital at the strategy level.

    Modes:
    - default: Use symbols from config (fast, ~100 symbols)
    - full_scan: Fetch S&P 500 + Nasdaq 100 + ETFs (slower, ~600 symbols)
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.max_price = config.underlyings.max_share_price
        self._price_filter_enabled = self.max_price != float('inf')
        self._price_cache: Dict[str, float] = {}
        self._scan_mode = "default"

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol using the most reliable method.

        Priority order:
        1. Recent intraday history (most reliable for current price)
        2. fast_info (faster than info dict)
        3. info dict as fallback
        """
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            price = None

            # Method 1: Get most recent price from intraday history (most reliable)
            # This avoids stale data issues with info dict
            try:
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
            except Exception:
                pass

            # Method 2: Try fast_info (faster than info, often more current)
            if not price:
                try:
                    fast = ticker.fast_info
                    price = fast.get('lastPrice') or fast.get('previousClose')
                except Exception:
                    pass

            # Method 3: Fall back to info dict
            if not price:
                try:
                    info = ticker.info
                    price = info.get('regularMarketPrice') or info.get('previousClose')
                except Exception:
                    pass

            # Method 4: Daily history as last resort
            if not price:
                try:
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                except Exception:
                    pass

            if price and price > 0:
                self._price_cache[symbol] = price
                return price

        except Exception as e:
            logger.debug(f"Could not get price for {symbol}: {e}")

        return None

    def has_options(self, symbol: str) -> bool:
        """Check if symbol has options available."""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.options
            return len(options) > 0
        except Exception:
            return False

    def filter_symbols_by_price(
        self,
        symbols: List[str],
        max_workers: int = 10
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Filter symbols and fetch prices in parallel.

        If max_share_price is set (not infinite), filters by price.
        Otherwise, only checks for options availability.

        Returns:
            Tuple of (valid_symbols, price_dict)
        """
        valid_symbols = []
        prices = {}
        total = len(symbols)
        processed = 0

        if self._price_filter_enabled:
            logger.info(f"Checking {total} symbols (max price ${self.max_price:.0f})...")
        else:
            logger.info(f"Checking {total} symbols for options availability...")

        def check_symbol(symbol: str) -> Tuple[str, Optional[float], bool]:
            price = self.get_price(symbol)
            has_opts = False
            # Only check options if price passes filter (or no filter)
            if price and (not self._price_filter_enabled or price <= self.max_price):
                has_opts = self.has_options(symbol)
            return symbol, price, has_opts

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_symbol, sym): sym for sym in symbols}

            for future in as_completed(futures):
                symbol, price, has_opts = future.result()
                processed += 1

                # Progress indicator every 50 symbols or at completion
                if processed % 50 == 0 or processed == total:
                    logger.info(f"  Progress: {processed}/{total} symbols checked, {len(valid_symbols)} valid so far")

                if price:
                    prices[symbol] = price

                    passes_price_filter = not self._price_filter_enabled or price <= self.max_price

                    if passes_price_filter and has_opts:
                        valid_symbols.append(symbol)
                        logger.debug(f"  {symbol}: ${price:.2f} - VALID")
                    elif not passes_price_filter:
                        logger.debug(f"  {symbol}: ${price:.2f} - EXCEEDS MAX PRICE")
                    elif not has_opts:
                        logger.debug(f"  {symbol}: ${price:.2f} - NO OPTIONS")
                else:
                    logger.debug(f"  {symbol}: NO PRICE DATA")

        if self._price_filter_enabled:
            logger.info(f"Found {len(valid_symbols)} symbols under ${self.max_price:.0f} with options")
        else:
            logger.info(f"Found {len(valid_symbols)} symbols with options available")

        return sorted(valid_symbols), prices

    def discover_symbols(self, full_scan: bool = False) -> Tuple[List[str], Dict[str, float]]:
        """
        Main discovery method.

        Args:
            full_scan: If True, fetch S&P 500 + Nasdaq 100 + ETFs (~600 symbols)
                      If False, use config list (~100 symbols)

        Returns filtered list of symbols and their prices.
        """
        if full_scan:
            self._scan_mode = "full_scan"
            logger.info("Full market scan enabled - fetching index constituents...")
            seed_symbols = list(fetch_all_index_symbols())
            # Also add any custom symbols from config
            seed_symbols.extend(self.config.underlyings.custom_symbols)
            seed_symbols = list(set(seed_symbols))  # Dedupe
        else:
            self._scan_mode = "default"
            seed_symbols = self.config.get_active_symbols()

        logger.info(f"Starting symbol discovery with {len(seed_symbols)} symbols ({self._scan_mode} mode)")

        # Filter by price and options availability
        # Use more workers for full scan
        max_workers = 20 if full_scan else 10
        valid_symbols, prices = self.filter_symbols_by_price(seed_symbols, max_workers=max_workers)

        return valid_symbols, prices

    def get_symbols_by_price_range(
        self,
        min_price: float = 0,
        max_price: Optional[float] = None
    ) -> List[str]:
        """Get cached symbols within a price range."""
        max_price = max_price or self.max_price

        return [
            sym for sym, price in self._price_cache.items()
            if min_price <= price <= max_price
        ]


def discover_and_filter_symbols(config: AnalyzerConfig) -> List[str]:
    """
    Convenience function to discover and filter symbols.
    """
    discovery = SymbolDiscovery(config)
    symbols, _ = discovery.discover_symbols()
    return symbols
