"""
Historical data provider with yfinance integration and Black-Scholes synthesis.
Provides options chain data for any historical date within the past 10 years.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf

from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega

from core.models import OptionContract, OptionsChain, OptionType


@dataclass
class HistoricalDataConfig:
    """Configuration for historical data provider."""
    start_date: date
    end_date: date
    symbols: List[str]
    use_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".backtest_cache"))
    risk_free_rate: float = 0.05
    volatility_window: int = 30  # Days for HV calculation
    max_dte: int = 5  # Maximum DTE for generated expirations


class HistoricalDataProvider:
    """
    Provides historical options data via yfinance + Black-Scholes synthesis.

    Since Yahoo Finance doesn't provide historical options chain snapshots,
    we synthesize options chains using:
    1. Historical underlying prices from yfinance
    2. Rolling historical volatility
    3. Black-Scholes model for option pricing and Greeks
    """

    def __init__(self, config: HistoricalDataConfig):
        self.config = config
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.volatility_cache: Dict[str, pd.Series] = {}

        if config.use_cache:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_underlying_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical daily OHLCV data from yfinance.
        Returns DataFrame with Date index and OHLC columns.
        """
        if symbol in self.price_cache:
            return self.price_cache[symbol]

        cache_file = self._get_cache_path(symbol, "prices")
        if cache_file and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                self.price_cache[symbol] = df
                return df
            except Exception:
                pass

        try:
            # Fetch with buffer for volatility calculation
            buffer_start = self.config.start_date - timedelta(days=60)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=buffer_start, end=self.config.end_date + timedelta(days=1))

            if df.empty:
                return None

            # Clean up column names (yfinance returns title case)
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            # Ensure we have the close column
            if 'close' not in df.columns:
                return None

            # Normalize index to timezone-naive dates for consistent comparison
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            self.price_cache[symbol] = df

            # Cache to disk
            if cache_file:
                df.to_csv(cache_file)

            return df

        except Exception as e:
            print(f"Error fetching history for {symbol}: {e}")
            return None

    def calculate_historical_volatility(self, symbol: str) -> Optional[pd.Series]:
        """
        Calculate rolling historical volatility for Black-Scholes synthesis.
        Returns Series with Date index and annualized volatility values.
        """
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]

        df = self.load_underlying_history(symbol)
        if df is None or 'close' not in df.columns:
            return None

        # Calculate daily log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))

        # Rolling standard deviation, annualized
        rolling_std = log_returns.rolling(window=self.config.volatility_window).std()
        hv = rolling_std * np.sqrt(252)  # Annualize

        self.volatility_cache[symbol] = hv
        return hv

    def get_underlying_price(self, symbol: str, as_of_date: date) -> Optional[float]:
        """Get the closing price for a symbol on a specific date."""
        df = self.load_underlying_history(symbol)
        if df is None:
            return None

        target = pd.Timestamp(as_of_date)

        # Find exact or most recent trading day
        valid_dates = df.index[df.index <= target]
        if len(valid_dates) == 0:
            return None

        closest_date = valid_dates[-1]
        return float(df.loc[closest_date, 'close'])

    def get_volatility(self, symbol: str, as_of_date: date) -> Optional[float]:
        """Get historical volatility for a symbol on a specific date."""
        hv = self.calculate_historical_volatility(symbol)
        if hv is None:
            return 0.25  # Default

        target = pd.Timestamp(as_of_date)

        # Find closest date
        valid_dates = hv.index[hv.index <= target]
        if len(valid_dates) == 0:
            return 0.25  # Default

        closest_date = valid_dates[-1]
        vol = hv.loc[closest_date]

        # Ensure valid volatility
        if pd.isna(vol) or vol <= 0:
            return 0.25  # Default 25% if no valid HV

        return float(vol)

    def get_options_chain(self, symbol: str, as_of_date: date) -> Optional[OptionsChain]:
        """
        Get options chain for a specific historical date.
        Uses Black-Scholes synthesis since historical options data isn't available.
        """
        price = self.get_underlying_price(symbol, as_of_date)
        if price is None:
            return None

        volatility = self.get_volatility(symbol, as_of_date)
        if volatility is None:
            volatility = 0.25  # Default

        return self.synthesize_options_chain(symbol, as_of_date, price, volatility)

    def synthesize_options_chain(
        self,
        symbol: str,
        as_of_date: date,
        underlying_price: float,
        volatility: float
    ) -> OptionsChain:
        """
        Generate synthetic options chain using Black-Scholes.

        Creates realistic strike prices and weekly expirations,
        then prices all options using the BS model.
        """
        strikes = self._generate_strikes(underlying_price)
        expirations = self._generate_expirations(as_of_date, self.config.max_dte)

        calls = []
        puts = []

        for expiration in expirations:
            dte = (expiration - as_of_date).days
            if dte < 1:
                continue

            for strike in strikes:
                # Generate call
                call = self._synthesize_option(
                    symbol=symbol,
                    option_type='c',
                    spot=underlying_price,
                    strike=strike,
                    expiration=expiration,
                    dte=dte,
                    volatility=volatility,
                    as_of_date=as_of_date
                )
                if call:
                    calls.append(call)

                # Generate put
                put = self._synthesize_option(
                    symbol=symbol,
                    option_type='p',
                    spot=underlying_price,
                    strike=strike,
                    expiration=expiration,
                    dte=dte,
                    volatility=volatility,
                    as_of_date=as_of_date
                )
                if put:
                    puts.append(put)

        return OptionsChain(
            underlying_symbol=symbol,
            underlying_price=underlying_price,
            fetch_time=datetime.combine(as_of_date, datetime.min.time()),
            expirations=expirations,
            calls=calls,
            puts=puts,
            historical_volatility=volatility,
            iv_rank=50.0,  # Synthetic - use 50th percentile as baseline
            iv_percentile=50.0
        )

    def _generate_strikes(self, spot_price: float) -> List[float]:
        """Generate realistic strike prices around spot."""
        # Determine increment based on price level
        if spot_price < 25:
            increment = 0.5
        elif spot_price < 50:
            increment = 1.0
        elif spot_price < 200:
            increment = 2.5
        else:
            increment = 5.0

        # Generate from 70% to 130% of spot
        low = round(spot_price * 0.70 / increment) * increment
        high = round(spot_price * 1.30 / increment) * increment

        strikes = []
        strike = low
        while strike <= high:
            strikes.append(strike)
            strike += increment

        return strikes

    def _generate_expirations(self, as_of_date: date, max_dte: int) -> List[date]:
        """Generate weekly Friday expirations within DTE range."""
        expirations = []

        # Find next Friday
        days_until_friday = (4 - as_of_date.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # Skip same-day expiration

        next_friday = as_of_date + timedelta(days=days_until_friday)

        # Generate Fridays within max_dte
        while (next_friday - as_of_date).days <= max_dte:
            expirations.append(next_friday)
            next_friday += timedelta(days=7)

        # If no expirations found (max_dte too short), add at least one
        if not expirations:
            next_friday = as_of_date + timedelta(days=days_until_friday)
            expirations.append(next_friday)

        return expirations

    def _synthesize_option(
        self,
        symbol: str,
        option_type: str,  # 'c' or 'p'
        spot: float,
        strike: float,
        expiration: date,
        dte: int,
        volatility: float,
        as_of_date: date
    ) -> Optional[OptionContract]:
        """Create synthetic option using Black-Scholes."""
        try:
            t = max(dte / 365.0, 0.001)
            r = self.config.risk_free_rate
            sigma = max(volatility, 0.05)  # Minimum 5% vol

            # Calculate theoretical price
            price = black_scholes(option_type, spot, strike, t, r, sigma)

            if price < 0.01:
                return None  # Skip nearly worthless options

            # Add synthetic bid-ask spread based on moneyness
            moneyness = abs(strike / spot - 1)
            if moneyness < 0.05:  # Near ATM
                spread_pct = 0.01
            elif moneyness < 0.10:
                spread_pct = 0.015
            else:
                spread_pct = 0.02

            bid = max(0.01, price * (1 - spread_pct))
            ask = price * (1 + spread_pct)

            # Calculate Greeks
            opt_delta = delta(option_type, spot, strike, t, r, sigma)
            opt_gamma = gamma(option_type, spot, strike, t, r, sigma)
            opt_theta = theta(option_type, spot, strike, t, r, sigma) * 365  # Annualized
            opt_vega = vega(option_type, spot, strike, t, r, sigma)

            # Create contract symbol (synthetic)
            exp_str = expiration.strftime("%y%m%d")
            opt_char = "C" if option_type == 'c' else "P"
            strike_str = f"{int(strike * 1000):08d}"
            contract_symbol = f"{symbol}{exp_str}{opt_char}{strike_str}"

            contract = OptionContract(
                symbol=contract_symbol,
                underlying_symbol=symbol,
                underlying_price=spot,
                strike=strike,
                expiration=expiration,
                option_type=OptionType.CALL if option_type == 'c' else OptionType.PUT,
                bid=bid,
                ask=ask,
                last_price=price,
                volume=1000,  # Synthetic liquidity
                open_interest=5000,
                implied_volatility=sigma,
                delta=opt_delta,
                gamma=opt_gamma,
                theta=opt_theta,
                vega=opt_vega,
            )

            # Manually set DTE since __post_init__ uses date.today()
            contract.dte = dte

            return contract

        except Exception as e:
            # Skip options that cause calculation errors
            return None

    def _get_cache_path(self, symbol: str, data_type: str) -> Optional[Path]:
        """Get cache file path for a symbol."""
        if not self.config.use_cache:
            return None

        # Create hash of date range for cache invalidation
        range_str = f"{self.config.start_date}_{self.config.end_date}"
        range_hash = hashlib.md5(range_str.encode()).hexdigest()[:8]

        filename = f"{symbol}_{data_type}_{range_hash}.csv"
        return self.config.cache_dir / filename

    def preload_symbols(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Preload price and volatility data for symbols.
        Returns dict of symbol -> success status.
        """
        symbols = symbols or self.config.symbols
        results = {}

        for symbol in symbols:
            df = self.load_underlying_history(symbol)
            hv = self.calculate_historical_volatility(symbol) if df is not None else None
            results[symbol] = df is not None and hv is not None

        return results

    def get_trading_days(self, symbol: str = "SPY") -> List[date]:
        """
        Get list of valid trading days in the date range.
        Uses SPY as reference for market calendar.
        """
        df = self.load_underlying_history(symbol)
        if df is None:
            # Fallback: generate weekdays
            current = self.config.start_date
            days = []
            while current <= self.config.end_date:
                if current.weekday() < 5:  # Monday = 0, Friday = 4
                    days.append(current)
                current += timedelta(days=1)
            return days

        # Filter to date range (index is already timezone-naive from load)
        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date)

        valid_dates = df.index[(df.index >= start_ts) & (df.index <= end_ts)]
        return [d.date() for d in valid_dates]
