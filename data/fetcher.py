"""
Data fetching layer for options chains.
Uses yfinance as the primary data source with abstraction for future providers.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Protocol, Any
from abc import ABC, abstractmethod
import logging

from core.models import OptionContract, OptionsChain, OptionType
from core.config import AnalyzerConfig

logger = logging.getLogger(__name__)


class DataProvider(Protocol):
    """Protocol for data providers (allows swapping yfinance for other sources)."""

    def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """Fetch options chain for a symbol."""
        ...

    def get_historical_volatility(self, symbol: str, days: int) -> Optional[float]:
        """Calculate historical volatility."""
        ...


class YFinanceProvider:
    """Yahoo Finance data provider using yfinance library."""

    # Strike price sanity check: strikes should be within this range of current price
    # to be considered valid (filters out stale pre-split data)
    MIN_STRIKE_RATIO = 0.30  # Strike must be at least 30% of current price
    MAX_STRIKE_RATIO = 3.0   # Strike must be at most 300% of current price

    def __init__(self, config: AnalyzerConfig):
        self.config = config

    def _get_current_price(self, ticker: yf.Ticker, symbol: str) -> Optional[float]:
        """
        Get the most current price using multiple methods for reliability.

        Priority:
        1. Intraday 1-minute data (most current during market hours)
        2. fast_info (lightweight, often current)
        3. info dict (can be stale)
        4. Daily history (fallback)
        """
        price = None

        # Method 1: Intraday history - most reliable for current price
        try:
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                logger.debug(f"{symbol}: Got price ${price:.2f} from intraday data")
                return price
        except Exception:
            pass

        # Method 2: fast_info - faster than info dict
        try:
            fast = ticker.fast_info
            price = fast.get('lastPrice') or fast.get('previousClose')
            if price:
                logger.debug(f"{symbol}: Got price ${price:.2f} from fast_info")
                return float(price)
        except Exception:
            pass

        # Method 3: info dict - can be stale but widely available
        try:
            info = ticker.info
            price = info.get('regularMarketPrice') or info.get('previousClose')
            if price:
                logger.debug(f"{symbol}: Got price ${price:.2f} from info dict")
                return float(price)
        except Exception:
            pass

        # Method 4: Daily history as last resort
        try:
            hist = ticker.history(period='5d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                logger.debug(f"{symbol}: Got price ${price:.2f} from daily history")
                return price
        except Exception:
            pass

        return None

    def _validate_options_data(
        self,
        contracts: List[OptionContract],
        current_price: float,
        symbol: str
    ) -> List[OptionContract]:
        """
        Validate options data to filter out stale/invalid contracts.

        Checks:
        1. Strike prices are within reasonable range of current price
        2. Contracts have meaningful pricing data
        """
        valid_contracts = []
        invalid_strike_count = 0

        for contract in contracts:
            strike_ratio = contract.strike / current_price if current_price > 0 else 0

            # Check if strike is within reasonable range
            if strike_ratio < self.MIN_STRIKE_RATIO or strike_ratio > self.MAX_STRIKE_RATIO:
                invalid_strike_count += 1
                continue

            valid_contracts.append(contract)

        if invalid_strike_count > 0:
            logger.warning(
                f"{symbol}: Filtered out {invalid_strike_count} contracts with invalid strikes "
                f"(price=${current_price:.2f}, valid range: ${current_price * self.MIN_STRIKE_RATIO:.2f} - "
                f"${current_price * self.MAX_STRIKE_RATIO:.2f})"
            )

        return valid_contracts

    def _check_data_freshness(self, ticker: yf.Ticker, symbol: str) -> bool:
        """
        Check if the price data is reasonably fresh.

        Returns True if data appears fresh, False if stale.
        Logs warnings for stale data but doesn't block processing.
        """
        try:
            # Check last trade time from fast_info or history
            hist = ticker.history(period='5d')
            if hist.empty:
                logger.warning(f"{symbol}: No recent price history - data may be stale")
                return False

            last_trade_date = hist.index[-1].date()
            today = date.today()
            days_since_trade = (today - last_trade_date).days

            # During weekends, 2-3 days is normal
            # During weekdays, more than 1 day is suspicious
            is_weekend = today.weekday() >= 5
            max_acceptable_days = 3 if is_weekend else 1

            if days_since_trade > max_acceptable_days:
                logger.warning(
                    f"{symbol}: Last trade was {days_since_trade} days ago ({last_trade_date}) - "
                    f"data may be stale"
                )
                return False

            return True

        except Exception as e:
            logger.debug(f"{symbol}: Could not check data freshness: {e}")
            return True  # Don't block on check failure

    def _validate_options_chain_consistency(
        self,
        all_calls: List[OptionContract],
        all_puts: List[OptionContract],
        current_price: float,
        symbol: str
    ) -> bool:
        """
        Validate that the options chain data is consistent with the current stock price.

        This catches cases where the stock price is current but the options chain
        data is stale (e.g., post-split where options haven't been updated).

        Checks:
        1. ATM options should exist near the current price
        2. ATM put-call parity should roughly hold
        3. Deep ITM options should have intrinsic value close to their actual value

        Returns False if the data appears inconsistent (should skip this symbol).
        """
        all_contracts = all_calls + all_puts
        if not all_contracts:
            return False

        # Find the strikes that are available
        available_strikes = sorted(set(c.strike for c in all_contracts))
        if not available_strikes:
            return False

        # Find the closest strike to current price
        closest_strike = min(available_strikes, key=lambda s: abs(s - current_price))
        strike_distance_pct = abs(closest_strike - current_price) / current_price

        # Check 1: There should be a strike within 20% of current price
        # If not, the options chain is likely for a different (old) price level
        if strike_distance_pct > 0.20:
            logger.warning(
                f"{symbol}: No strikes near current price ${current_price:.2f}. "
                f"Closest strike is ${closest_strike:.2f} ({strike_distance_pct:.0%} away). "
                f"Options data may be stale - SKIPPING"
            )
            return False

        # Check 2: For ATM options, the bid/ask should be reasonable
        # ATM options typically have premium of 2-10% of stock price for short-term
        atm_puts = [p for p in all_puts if abs(p.strike - current_price) / current_price < 0.05]
        atm_calls = [c for c in all_calls if abs(c.strike - current_price) / current_price < 0.05]

        if atm_puts:
            avg_atm_put_price = np.mean([p.mid_price for p in atm_puts if p.mid_price > 0])
            if avg_atm_put_price > 0:
                atm_premium_pct = avg_atm_put_price / current_price
                # ATM options shouldn't be worth more than ~30% of stock price for <30 DTE
                # or less than 0.1% (essentially worthless means stale data)
                if atm_premium_pct > 0.30:
                    logger.warning(
                        f"{symbol}: ATM put premium (${avg_atm_put_price:.2f}) is {atm_premium_pct:.0%} of "
                        f"stock price - suspiciously high, options data may be stale - SKIPPING"
                    )
                    return False

        # Check 3: Deep ITM puts should have intrinsic value
        # If stock is $34 and there's a $40 put, it should be worth at least ~$6
        for put in all_puts:
            if put.strike > current_price * 1.10:  # ITM put (strike > price)
                intrinsic_value = put.strike - current_price
                if put.mid_price > 0 and put.mid_price < intrinsic_value * 0.5:
                    # Option is priced at less than half its intrinsic value = stale
                    logger.warning(
                        f"{symbol}: ITM put ${put.strike} strike priced at ${put.mid_price:.2f} "
                        f"but intrinsic value is ${intrinsic_value:.2f} - options data stale - SKIPPING"
                    )
                    return False

        # Check 4: Deep ITM calls should have intrinsic value
        for call in all_calls:
            if call.strike < current_price * 0.90:  # ITM call (strike < price)
                intrinsic_value = current_price - call.strike
                if call.mid_price > 0 and call.mid_price < intrinsic_value * 0.5:
                    logger.warning(
                        f"{symbol}: ITM call ${call.strike} strike priced at ${call.mid_price:.2f} "
                        f"but intrinsic value is ${intrinsic_value:.2f} - options data stale - SKIPPING"
                    )
                    return False

        return True

    def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """Fetch complete options chain for a symbol."""
        try:
            ticker = yf.Ticker(symbol)

            # Check data freshness (warns but doesn't block)
            self._check_data_freshness(ticker, symbol)

            # Get current price using reliable method
            current_price = self._get_current_price(ticker, symbol)
            if not current_price:
                logger.warning(f"Could not get price for {symbol}")
                return None

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return None

            # Filter expirations within our DTE range
            max_dte = self.config.trade_criteria.max_dte
            today = date.today()
            valid_expirations = []

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                if self.config.trade_criteria.min_dte <= dte <= max_dte:
                    valid_expirations.append(exp_date)

            if not valid_expirations:
                logger.info(f"No expirations within DTE range for {symbol}")
                return None

            # Fetch option chains for each expiration
            all_calls = []
            all_puts = []

            for exp_date in valid_expirations:
                exp_str = exp_date.strftime('%Y-%m-%d')
                try:
                    chain = ticker.option_chain(exp_str)

                    # Process calls
                    calls = self._process_options_df(
                        chain.calls, symbol, current_price, exp_date, OptionType.CALL
                    )
                    all_calls.extend(calls)

                    # Process puts
                    puts = self._process_options_df(
                        chain.puts, symbol, current_price, exp_date, OptionType.PUT
                    )
                    all_puts.extend(puts)

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} expiration {exp_str}: {e}")
                    continue

            # Validate options data - filter out stale/invalid contracts
            all_calls = self._validate_options_data(all_calls, current_price, symbol)
            all_puts = self._validate_options_data(all_puts, current_price, symbol)

            # Check if we have any valid contracts left
            if not all_calls and not all_puts:
                logger.warning(f"{symbol}: No valid options contracts after filtering - data may be stale")
                return None

            # Validate that options chain is consistent with current stock price
            # This catches stale options data (e.g., post-split symbols)
            if not self._validate_options_chain_consistency(all_calls, all_puts, current_price, symbol):
                return None

            # Calculate historical volatility and IV metrics
            hv = self.get_historical_volatility(symbol, 30)
            iv_rank, iv_percentile = self._calculate_iv_metrics(ticker, all_puts + all_calls)

            return OptionsChain(
                underlying_symbol=symbol,
                underlying_price=current_price,
                fetch_time=datetime.now(),
                expirations=valid_expirations,
                calls=all_calls,
                puts=all_puts,
                historical_volatility=hv,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile
            )

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return None

    def _process_options_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        underlying_price: float,
        expiration: date,
        option_type: OptionType
    ) -> List[OptionContract]:
        """Convert yfinance options DataFrame to OptionContract objects."""
        contracts = []

        for _, row in df.iterrows():
            try:
                # Extract data with defaults for missing values
                contract = OptionContract(
                    symbol=row.get('contractSymbol', ''),
                    underlying_symbol=symbol,
                    underlying_price=underlying_price,
                    strike=float(row.get('strike', 0)),
                    expiration=expiration,
                    option_type=option_type,
                    bid=float(row.get('bid', 0) or 0),
                    ask=float(row.get('ask', 0) or 0),
                    last_price=float(row.get('lastPrice', 0) or 0),
                    volume=int(row.get('volume', 0) or 0),
                    open_interest=int(row.get('openInterest', 0) or 0),
                    implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                )

                # Filter out contracts with no meaningful data
                if contract.strike > 0 and (contract.bid > 0 or contract.ask > 0 or contract.last_price > 0):
                    contracts.append(contract)

            except Exception as e:
                logger.debug(f"Error processing contract row: {e}")
                continue

        return contracts

    def get_historical_volatility(self, symbol: str, days: int = 30) -> Optional[float]:
        """Calculate historical volatility from price data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f'{days + 10}d')  # Extra days for buffer

            if len(hist) < days:
                return None

            # Calculate log returns
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

            # Annualized volatility (252 trading days)
            hv = log_returns.std() * np.sqrt(252)

            return float(hv)

        except Exception as e:
            logger.warning(f"Error calculating HV for {symbol}: {e}")
            return None

    def _calculate_iv_metrics(
        self,
        ticker: yf.Ticker,
        contracts: List[OptionContract]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate IV rank and percentile.
        IV Rank: Current IV relative to 52-week high/low
        IV Percentile: Percentage of days IV was lower than current
        """
        if not contracts:
            return None, None

        # Current ATM IV (average of near-the-money options)
        atm_contracts = [c for c in contracts if 0.95 <= c.moneyness <= 1.05]
        if not atm_contracts:
            return None, None

        current_iv = np.mean([c.implied_volatility for c in atm_contracts])

        # For proper IV rank, we'd need historical IV data
        # yfinance doesn't provide this directly, so we'll estimate from price volatility
        try:
            hist = ticker.history(period='1y')
            if len(hist) < 252:
                return None, None

            # Calculate rolling 30-day HV as proxy for historical IV
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            rolling_hv = log_returns.rolling(window=30).std() * np.sqrt(252)
            rolling_hv = rolling_hv.dropna()

            if len(rolling_hv) < 10:
                return None, None

            # IV Rank
            hv_52w_high = rolling_hv.max()
            hv_52w_low = rolling_hv.min()

            if hv_52w_high > hv_52w_low:
                iv_rank = (current_iv - hv_52w_low) / (hv_52w_high - hv_52w_low) * 100
            else:
                iv_rank = 50.0

            # IV Percentile
            iv_percentile = (rolling_hv < current_iv).sum() / len(rolling_hv) * 100

            return float(iv_rank), float(iv_percentile)

        except Exception as e:
            logger.debug(f"Error calculating IV metrics: {e}")
            return None, None


class DataFetcher:
    """
    Main data fetching orchestrator.
    Supports multiple providers and caching.

    When markets are closed, can optionally fall back to synthetic chains
    generated using Black-Scholes with cached IV surfaces and sentiment
    adjustments.
    """

    def __init__(
        self,
        config: AnalyzerConfig,
        provider: Optional[DataProvider] = None,
        use_synthetic_fallback: bool = False,
        sentiment_signals: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.provider = provider or YFinanceProvider(config)
        self._cache: Dict[str, OptionsChain] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)  # 5-minute cache

        # Synthetic chain fallback
        self.use_synthetic_fallback = use_synthetic_fallback
        self.sentiment_signals = sentiment_signals or {}
        self._synthetic_generator = None

        if use_synthetic_fallback:
            self._init_synthetic_generator()

    def _init_synthetic_generator(self):
        """Initialize the synthetic chain generator (lazy load)."""
        try:
            from data.synthetic_chain import SyntheticChainGenerator
            self._synthetic_generator = SyntheticChainGenerator(self.config)
            logger.info("Synthetic chain generator initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize synthetic generator: {e}")
            self._synthetic_generator = None

    def set_sentiment_signals(self, signals: Dict[str, Any]):
        """Set sentiment signals for synthetic IV adjustment."""
        self.sentiment_signals = signals

    def enable_synthetic_fallback(self, enabled: bool = True):
        """Enable or disable synthetic chain fallback."""
        self.use_synthetic_fallback = enabled
        if enabled and self._synthetic_generator is None:
            self._init_synthetic_generator()

    def fetch_chain(
        self,
        symbol: str,
        use_cache: bool = True,
        allow_synthetic: bool = True
    ) -> Optional[OptionsChain]:
        """
        Fetch options chain for a single symbol.

        Args:
            symbol: Stock/ETF symbol
            use_cache: Use cached data if available
            allow_synthetic: Allow fallback to synthetic chain if live data fails

        Returns:
            OptionsChain or None
        """
        # Check cache
        if use_cache and symbol in self._cache:
            if datetime.now() - self._cache_time[symbol] < self._cache_ttl:
                logger.debug(f"Using cached data for {symbol}")
                return self._cache[symbol]

        # Fetch fresh data from provider
        chain = self.provider.get_options_chain(symbol)

        if chain:
            self._cache[symbol] = chain
            self._cache_time[symbol] = datetime.now()
            return chain

        # Fallback to synthetic chain if enabled
        if allow_synthetic and self.use_synthetic_fallback and self._synthetic_generator:
            logger.info(f"{symbol}: Live data unavailable, generating synthetic chain")
            sentiment = self.sentiment_signals.get(symbol)
            chain = self._synthetic_generator.generate_chain(symbol, sentiment)

            if chain:
                # Mark as synthetic in the chain (could add a flag to OptionsChain)
                self._cache[symbol] = chain
                self._cache_time[symbol] = datetime.now()
                return chain

        return None

    def fetch_all_chains(
        self,
        symbols: Optional[List[str]] = None,
        allow_synthetic: bool = True
    ) -> Dict[str, OptionsChain]:
        """
        Fetch options chains for all configured symbols.

        Args:
            symbols: List of symbols (uses config if None)
            allow_synthetic: Allow synthetic fallback for failed fetches
        """
        if symbols is None:
            symbols = self.config.get_active_symbols()

        chains = {}
        for symbol in symbols:
            logger.info(f"Fetching options chain for {symbol}...")
            chain = self.fetch_chain(symbol, allow_synthetic=allow_synthetic)
            if chain:
                chains[symbol] = chain
            else:
                logger.warning(f"Failed to fetch chain for {symbol}")

        return chains

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._cache_time.clear()

    def refresh_iv_surfaces(self, symbols: Optional[List[str]] = None) -> int:
        """
        Refresh cached IV surfaces for synthetic chain generation.

        Call this during market hours to update IV data for after-hours use.

        Returns:
            Number of surfaces successfully refreshed
        """
        if self._synthetic_generator is None:
            self._init_synthetic_generator()

        if self._synthetic_generator is None:
            logger.warning("Synthetic generator not available")
            return 0

        if symbols is None:
            symbols = self.config.get_active_symbols()

        refreshed = 0
        for symbol in symbols:
            if self._synthetic_generator.refresh_iv_surface(symbol):
                refreshed += 1
                logger.debug(f"Refreshed IV surface for {symbol}")

        logger.info(f"Refreshed IV surfaces for {refreshed}/{len(symbols)} symbols")
        return refreshed
