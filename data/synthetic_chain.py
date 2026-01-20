"""
Synthetic Options Chain Generator.

Generates theoretical option prices using Black-Scholes when live market data
is unavailable (outside market hours). Uses cached IV surfaces with sentiment-
based adjustments to produce reasonably accurate synthetic chains.

Key Features:
- Extended hours price fetching via yfinance
- IV surface caching with disk persistence
- Sentiment-adjusted IV (fear/greed modifier)
- Configurable strike range and spacing
- Skew modeling for realistic OTM pricing

Usage:
    from data.synthetic_chain import SyntheticChainGenerator

    generator = SyntheticChainGenerator(config)

    # With sentiment adjustment
    chain = generator.generate_chain(
        symbol="SPY",
        sentiment_signal=sentiment_signals.get("SPY")
    )
"""

import json
import logging
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
import yfinance as yf

from py_vollib.black_scholes import black_scholes as bs_price
from py_vollib.black_scholes.greeks.analytical import (
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
)

from core.models import OptionContract, OptionsChain, OptionType, SentimentSignal
from core.config import AnalyzerConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CachedIVSurface:
    """
    Cached implied volatility surface data for a symbol.

    Stores ATM IV by expiration and skew parameters for reconstruction.
    """
    symbol: str
    underlying_price: float
    cached_at: datetime

    # ATM IV by DTE (days to expiration -> IV)
    atm_iv_by_dte: Dict[int, float] = field(default_factory=dict)

    # Skew parameters: how much IV increases per % OTM for puts/calls
    # Positive skew_put means puts get more expensive further OTM
    skew_put_per_pct: float = 0.002  # +0.2% IV per 1% OTM for puts
    skew_call_per_pct: float = -0.001  # -0.1% IV per 1% OTM for calls

    # Term structure slope (IV change per day of DTE)
    term_slope: float = 0.0001  # Slight contango by default

    # Historical volatility for fallback
    historical_vol: Optional[float] = None

    # IV rank at cache time (for context)
    iv_rank: Optional[float] = None

    def get_iv_for_strike_dte(
        self,
        strike: float,
        current_price: float,
        dte: int,
        option_type: OptionType
    ) -> float:
        """
        Get interpolated IV for a specific strike and DTE.

        Applies:
        1. ATM IV lookup/interpolation by DTE
        2. Skew adjustment based on moneyness
        3. Term structure adjustment
        """
        # 1. Get base ATM IV for this DTE
        if dte in self.atm_iv_by_dte:
            base_iv = self.atm_iv_by_dte[dte]
        elif self.atm_iv_by_dte:
            # Interpolate between available DTEs
            dtes = sorted(self.atm_iv_by_dte.keys())
            if dte < dtes[0]:
                base_iv = self.atm_iv_by_dte[dtes[0]]
            elif dte > dtes[-1]:
                # Extrapolate using term slope
                base_iv = self.atm_iv_by_dte[dtes[-1]] + self.term_slope * (dte - dtes[-1])
            else:
                # Linear interpolation
                lower = max(d for d in dtes if d <= dte)
                upper = min(d for d in dtes if d >= dte)
                if lower == upper:
                    base_iv = self.atm_iv_by_dte[lower]
                else:
                    weight = (dte - lower) / (upper - lower)
                    base_iv = (1 - weight) * self.atm_iv_by_dte[lower] + weight * self.atm_iv_by_dte[upper]
        else:
            # Fallback to historical vol or default
            base_iv = self.historical_vol or 0.25

        # 2. Apply skew based on moneyness
        moneyness_pct = (strike - current_price) / current_price * 100  # % from ATM

        if option_type == OptionType.PUT:
            # For puts, negative moneyness (OTM) increases IV
            skew_adjustment = -moneyness_pct * self.skew_put_per_pct
        else:
            # For calls, positive moneyness (OTM) typically decreases IV slightly
            skew_adjustment = moneyness_pct * self.skew_call_per_pct

        final_iv = base_iv + skew_adjustment

        # Clamp to reasonable range
        return max(0.05, min(final_iv, 3.0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            'symbol': self.symbol,
            'underlying_price': self.underlying_price,
            'cached_at': self.cached_at.isoformat(),
            'atm_iv_by_dte': {str(k): v for k, v in self.atm_iv_by_dte.items()},
            'skew_put_per_pct': self.skew_put_per_pct,
            'skew_call_per_pct': self.skew_call_per_pct,
            'term_slope': self.term_slope,
            'historical_vol': self.historical_vol,
            'iv_rank': self.iv_rank,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedIVSurface':
        """Deserialize from JSON storage."""
        return cls(
            symbol=data['symbol'],
            underlying_price=data['underlying_price'],
            cached_at=datetime.fromisoformat(data['cached_at']),
            atm_iv_by_dte={int(k): v for k, v in data['atm_iv_by_dte'].items()},
            skew_put_per_pct=data.get('skew_put_per_pct', 0.002),
            skew_call_per_pct=data.get('skew_call_per_pct', -0.001),
            term_slope=data.get('term_slope', 0.0001),
            historical_vol=data.get('historical_vol'),
            iv_rank=data.get('iv_rank'),
        )


@dataclass
class SyntheticChainConfig:
    """Configuration for synthetic chain generation."""

    # Strike generation
    strike_range_pct: float = 0.15  # Generate strikes ±15% from current price
    strike_spacing_pct: float = 0.01  # 1% spacing between strikes

    # For lower-priced stocks, use dollar-based spacing
    min_strike_spacing: float = 0.5  # Minimum $0.50 spacing

    # DTE range (uses analyzer config by default)
    min_dte: int = 1
    max_dte: int = 7

    # IV surface cache
    cache_dir: Path = field(default_factory=lambda: Path(".iv_cache"))
    cache_ttl_hours: int = 24  # How long cached IV is valid

    # Sentiment adjustment parameters
    sentiment_iv_scale: float = 0.05  # Max ±5% IV adjustment from sentiment
    news_volume_iv_scale: float = 0.03  # Max +3% IV for high news volume

    # Bid-ask spread simulation (as % of theoretical price)
    spread_pct_atm: float = 0.02  # 2% spread for ATM
    spread_pct_otm: float = 0.05  # 5% spread for far OTM

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


class SyntheticChainGenerator:
    """
    Generates synthetic options chains using Black-Scholes pricing.

    Workflow:
    1. Get current/extended hours price for underlying
    2. Load cached IV surface (or build from historical vol)
    3. Apply sentiment-based IV adjustments
    4. Generate strikes across configured range
    5. Price each option using Black-Scholes
    6. Simulate bid/ask spreads
    7. Calculate Greeks

    The resulting chain can be used by the analyzer when markets are closed.
    """

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        synthetic_config: Optional[SyntheticChainConfig] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.synthetic_config = synthetic_config or SyntheticChainConfig()

        # Ensure cache directory exists
        self.synthetic_config.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory IV surface cache
        self._iv_cache: Dict[str, CachedIVSurface] = {}

        # Load persisted cache
        self._load_cache_from_disk()

    def generate_chain(
        self,
        symbol: str,
        sentiment_signal: Optional[SentimentSignal] = None,
        force_refresh_iv: bool = False
    ) -> Optional[OptionsChain]:
        """
        Generate a synthetic options chain for a symbol.

        Args:
            symbol: Stock/ETF symbol
            sentiment_signal: Optional sentiment data for IV adjustment
            force_refresh_iv: Force refresh of cached IV surface

        Returns:
            OptionsChain with synthetic prices, or None if generation fails
        """
        try:
            # 1. Get current price (works in extended hours)
            current_price = self._get_extended_hours_price(symbol)
            if current_price is None:
                logger.error(f"Could not get price for {symbol}")
                return None

            logger.info(f"{symbol}: Current price ${current_price:.2f}")

            # 2. Get or build IV surface
            iv_surface = self._get_iv_surface(symbol, force_refresh_iv)
            if iv_surface is None:
                logger.warning(f"{symbol}: No IV surface available, using defaults")
                iv_surface = self._build_default_iv_surface(symbol, current_price)

            # 3. Apply sentiment adjustments to IV
            iv_adjustment = self._calculate_sentiment_iv_adjustment(sentiment_signal)
            logger.debug(f"{symbol}: Sentiment IV adjustment: {iv_adjustment:+.2%}")

            # 4. Generate expirations
            expirations = self._generate_expirations()
            if not expirations:
                logger.warning(f"{symbol}: No valid expirations in range")
                return None

            # 5. Generate strikes
            strikes = self._generate_strikes(current_price)

            # 6. Generate option contracts
            all_calls = []
            all_puts = []

            for exp_date in expirations:
                dte = (exp_date - date.today()).days

                for strike in strikes:
                    # Generate call
                    call = self._generate_contract(
                        symbol=symbol,
                        underlying_price=current_price,
                        strike=strike,
                        expiration=exp_date,
                        dte=dte,
                        option_type=OptionType.CALL,
                        iv_surface=iv_surface,
                        iv_adjustment=iv_adjustment
                    )
                    if call and call.mid_price > 0.01:
                        all_calls.append(call)

                    # Generate put
                    put = self._generate_contract(
                        symbol=symbol,
                        underlying_price=current_price,
                        strike=strike,
                        expiration=exp_date,
                        dte=dte,
                        option_type=OptionType.PUT,
                        iv_surface=iv_surface,
                        iv_adjustment=iv_adjustment
                    )
                    if put and put.mid_price > 0.01:
                        all_puts.append(put)

            # 7. Build chain object
            chain = OptionsChain(
                underlying_symbol=symbol,
                underlying_price=current_price,
                fetch_time=datetime.now(),
                expirations=expirations,
                calls=all_calls,
                puts=all_puts,
                historical_volatility=iv_surface.historical_vol,
                iv_rank=iv_surface.iv_rank,
                iv_percentile=iv_surface.iv_rank,  # Approximation
            )

            logger.info(
                f"{symbol}: Generated synthetic chain with "
                f"{len(all_calls)} calls, {len(all_puts)} puts, "
                f"{len(expirations)} expirations"
            )

            return chain

        except Exception as e:
            logger.error(f"Error generating synthetic chain for {symbol}: {e}")
            return None

    def _get_extended_hours_price(self, symbol: str) -> Optional[float]:
        """
        Get the most current price, including pre/post market.

        Priority:
        1. Pre/post market quote if available
        2. Last regular session close
        """
        try:
            ticker = yf.Ticker(symbol)

            # Try to get pre/post market data
            # yfinance provides this via fast_info or info dict
            try:
                info = ticker.info

                # Check for pre/post market prices
                pre_market = info.get('preMarketPrice')
                post_market = info.get('postMarketPrice')
                regular_price = info.get('regularMarketPrice') or info.get('previousClose')

                # Use the most recent available
                if post_market and post_market > 0:
                    logger.debug(f"{symbol}: Using post-market price ${post_market:.2f}")
                    return float(post_market)
                elif pre_market and pre_market > 0:
                    logger.debug(f"{symbol}: Using pre-market price ${pre_market:.2f}")
                    return float(pre_market)
                elif regular_price and regular_price > 0:
                    logger.debug(f"{symbol}: Using regular market price ${regular_price:.2f}")
                    return float(regular_price)
            except Exception:
                pass

            # Fallback: recent history
            hist = ticker.history(period='5d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            return None

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def _get_iv_surface(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[CachedIVSurface]:
        """Get cached IV surface, refreshing if stale."""

        # Check memory cache
        if not force_refresh and symbol in self._iv_cache:
            cached = self._iv_cache[symbol]
            age_hours = (datetime.now() - cached.cached_at).total_seconds() / 3600

            if age_hours < self.synthetic_config.cache_ttl_hours:
                logger.debug(f"{symbol}: Using cached IV surface ({age_hours:.1f}h old)")
                return cached

        # Try to build fresh surface from live data
        fresh_surface = self._build_iv_surface_from_market(symbol)

        if fresh_surface:
            self._iv_cache[symbol] = fresh_surface
            self._save_cache_to_disk()
            return fresh_surface

        # Return stale cache if available
        if symbol in self._iv_cache:
            logger.warning(f"{symbol}: Using stale IV cache (could not refresh)")
            return self._iv_cache[symbol]

        return None

    def _build_iv_surface_from_market(self, symbol: str) -> Optional[CachedIVSurface]:
        """Build IV surface from current market data."""
        try:
            ticker = yf.Ticker(symbol)

            # Get current price
            current_price = self._get_extended_hours_price(symbol)
            if not current_price:
                return None

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                return None

            atm_iv_by_dte = {}
            all_put_ivs = []
            all_call_ivs = []
            all_put_moneyness = []
            all_call_moneyness = []

            today = date.today()

            for exp_str in expirations[:5]:  # Limit to first 5 expirations
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    dte = (exp_date - today).days

                    if dte < 1 or dte > 60:
                        continue

                    chain = ticker.option_chain(exp_str)

                    # Find ATM IV (closest strike to current price)
                    all_options = list(chain.calls.itertuples()) + list(chain.puts.itertuples())

                    if not all_options:
                        continue

                    # Get ATM IV
                    atm_options = [
                        opt for opt in all_options
                        if abs(opt.strike - current_price) / current_price < 0.03
                        and hasattr(opt, 'impliedVolatility')
                        and opt.impliedVolatility > 0
                    ]

                    if atm_options:
                        atm_iv = np.mean([opt.impliedVolatility for opt in atm_options])
                        atm_iv_by_dte[dte] = atm_iv

                    # Collect skew data
                    for opt in chain.puts.itertuples():
                        if hasattr(opt, 'impliedVolatility') and opt.impliedVolatility > 0:
                            moneyness = (opt.strike - current_price) / current_price * 100
                            if -30 < moneyness < 0:  # OTM puts
                                all_put_ivs.append(opt.impliedVolatility)
                                all_put_moneyness.append(moneyness)

                    for opt in chain.calls.itertuples():
                        if hasattr(opt, 'impliedVolatility') and opt.impliedVolatility > 0:
                            moneyness = (opt.strike - current_price) / current_price * 100
                            if 0 < moneyness < 30:  # OTM calls
                                all_call_ivs.append(opt.impliedVolatility)
                                all_call_moneyness.append(moneyness)

                except Exception as e:
                    logger.debug(f"{symbol}: Error processing expiration {exp_str}: {e}")
                    continue

            if not atm_iv_by_dte:
                return None

            # Calculate skew from regression
            skew_put = 0.002  # Default
            skew_call = -0.001  # Default

            if len(all_put_ivs) >= 3 and len(all_put_moneyness) >= 3:
                try:
                    # Linear regression: IV = base + skew * moneyness
                    coef = np.polyfit(all_put_moneyness, all_put_ivs, 1)
                    skew_put = -coef[0]  # Negate because OTM puts have negative moneyness
                except Exception:
                    pass

            if len(all_call_ivs) >= 3 and len(all_call_moneyness) >= 3:
                try:
                    coef = np.polyfit(all_call_moneyness, all_call_ivs, 1)
                    skew_call = coef[0]
                except Exception:
                    pass

            # Calculate term structure slope
            term_slope = 0.0001
            if len(atm_iv_by_dte) >= 2:
                dtes = list(atm_iv_by_dte.keys())
                ivs = [atm_iv_by_dte[d] for d in dtes]
                try:
                    coef = np.polyfit(dtes, ivs, 1)
                    term_slope = coef[0]
                except Exception:
                    pass

            # Get historical volatility
            hist_vol = self._calculate_historical_vol(ticker)

            # Calculate IV rank
            iv_rank = self._calculate_iv_rank(ticker, list(atm_iv_by_dte.values())[0] if atm_iv_by_dte else 0.25)

            surface = CachedIVSurface(
                symbol=symbol,
                underlying_price=current_price,
                cached_at=datetime.now(),
                atm_iv_by_dte=atm_iv_by_dte,
                skew_put_per_pct=skew_put,
                skew_call_per_pct=skew_call,
                term_slope=term_slope,
                historical_vol=hist_vol,
                iv_rank=iv_rank,
            )

            logger.info(
                f"{symbol}: Built IV surface - ATM IV: {list(atm_iv_by_dte.values())[0]:.1%}, "
                f"Put skew: {skew_put:.4f}, Call skew: {skew_call:.4f}"
            )

            return surface

        except Exception as e:
            logger.error(f"Error building IV surface for {symbol}: {e}")
            return None

    def _build_default_iv_surface(
        self,
        symbol: str,
        current_price: float
    ) -> CachedIVSurface:
        """Build a default IV surface when no market data available."""

        # Try to get historical volatility
        try:
            ticker = yf.Ticker(symbol)
            hist_vol = self._calculate_historical_vol(ticker)
        except Exception:
            hist_vol = 0.25  # 25% default

        # Use historical vol as base ATM IV
        base_iv = hist_vol or 0.25

        # Build term structure with slight contango
        atm_iv_by_dte = {}
        for dte in range(1, 31):
            atm_iv_by_dte[dte] = base_iv + 0.0001 * dte

        return CachedIVSurface(
            symbol=symbol,
            underlying_price=current_price,
            cached_at=datetime.now(),
            atm_iv_by_dte=atm_iv_by_dte,
            skew_put_per_pct=0.002,
            skew_call_per_pct=-0.001,
            term_slope=0.0001,
            historical_vol=hist_vol,
            iv_rank=50.0,  # Assume middle of range
        )

    def _calculate_historical_vol(self, ticker: yf.Ticker, days: int = 30) -> Optional[float]:
        """Calculate historical volatility from price data."""
        try:
            hist = ticker.history(period='60d')
            if len(hist) < days:
                return None

            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv = log_returns[-days:].std() * np.sqrt(252)
            return float(hv)
        except Exception:
            return None

    def _calculate_iv_rank(self, ticker: yf.Ticker, current_iv: float) -> Optional[float]:
        """Calculate IV rank (percentile over 52 weeks)."""
        try:
            hist = ticker.history(period='1y')
            if len(hist) < 252:
                return None

            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            rolling_hv = log_returns.rolling(window=30).std() * np.sqrt(252)
            rolling_hv = rolling_hv.dropna()

            if len(rolling_hv) < 10:
                return None

            hv_high = rolling_hv.max()
            hv_low = rolling_hv.min()

            if hv_high > hv_low:
                return (current_iv - hv_low) / (hv_high - hv_low) * 100
            return 50.0

        except Exception:
            return None

    def _calculate_sentiment_iv_adjustment(
        self,
        sentiment: Optional[SentimentSignal]
    ) -> float:
        """
        Calculate IV adjustment based on sentiment.

        Logic:
        - Negative sentiment → increase IV (fear premium)
        - Positive sentiment → slight decrease IV (complacency)
        - High news volume → increase IV (uncertainty)
        - Rapid sentiment shift → increase IV (instability)
        """
        if sentiment is None:
            return 0.0

        adjustment = 0.0
        cfg = self.synthetic_config

        # Base sentiment adjustment
        # Negative sentiment (-1) → +5% IV, Positive (+1) → -2% IV
        # Asymmetric because fear spikes IV more than greed reduces it
        if sentiment.sentiment_score < 0:
            adjustment += -sentiment.sentiment_score * cfg.sentiment_iv_scale
        else:
            adjustment -= sentiment.sentiment_score * cfg.sentiment_iv_scale * 0.4

        # News volume spike → increased IV (potential event)
        if sentiment.news_volume_zscore > 1.0:
            vol_adjustment = min(
                sentiment.news_volume_zscore - 1.0,
                2.0
            ) * cfg.news_volume_iv_scale
            adjustment += vol_adjustment

        # Sentiment momentum (rapid shift) → increased IV
        if abs(sentiment.sentiment_momentum) > 0.2:
            momentum_adjustment = abs(sentiment.sentiment_momentum) * 0.02
            adjustment += momentum_adjustment

        # Risk flag escalation
        if sentiment.risk_flag == "high_risk":
            adjustment += 0.03  # +3% for high risk
        elif sentiment.risk_flag == "elevated":
            adjustment += 0.01  # +1% for elevated

        return adjustment

    def _generate_expirations(self) -> List[date]:
        """Generate list of expiration dates within configured range."""
        expirations = []
        today = date.today()

        min_dte = self.synthetic_config.min_dte
        max_dte = self.synthetic_config.max_dte

        # Find Fridays (standard weekly expirations)
        for days_ahead in range(min_dte, max_dte + 1):
            exp_date = today + timedelta(days=days_ahead)

            # Weekly options typically expire on Friday
            # Also include Monday/Wednesday for some products
            if exp_date.weekday() == 4:  # Friday
                expirations.append(exp_date)

        # If no Fridays in range, include the next available weekday
        if not expirations:
            for days_ahead in range(min_dte, max_dte + 1):
                exp_date = today + timedelta(days=days_ahead)
                if exp_date.weekday() < 5:  # Weekday
                    expirations.append(exp_date)
                    break

        return sorted(expirations)

    def _generate_strikes(self, current_price: float) -> List[float]:
        """Generate strike prices around current price."""
        strikes = []
        cfg = self.synthetic_config

        # Calculate strike range
        lower_bound = current_price * (1 - cfg.strike_range_pct)
        upper_bound = current_price * (1 + cfg.strike_range_pct)

        # Calculate spacing
        spacing = max(
            current_price * cfg.strike_spacing_pct,
            cfg.min_strike_spacing
        )

        # Round spacing to nice numbers
        if spacing < 1:
            spacing = 0.5
        elif spacing < 2.5:
            spacing = 1.0
        elif spacing < 5:
            spacing = 2.5
        else:
            spacing = 5.0

        # Generate strikes
        strike = np.floor(lower_bound / spacing) * spacing
        while strike <= upper_bound:
            if strike > 0:
                strikes.append(float(strike))
            strike += spacing

        return sorted(strikes)

    def _generate_contract(
        self,
        symbol: str,
        underlying_price: float,
        strike: float,
        expiration: date,
        dte: int,
        option_type: OptionType,
        iv_surface: CachedIVSurface,
        iv_adjustment: float
    ) -> Optional[OptionContract]:
        """Generate a single synthetic option contract."""
        try:
            # Get IV for this strike/DTE
            base_iv = iv_surface.get_iv_for_strike_dte(
                strike, underlying_price, dte, option_type
            )

            # Apply sentiment adjustment
            iv = base_iv + iv_adjustment
            iv = max(0.05, min(iv, 3.0))  # Clamp to reasonable range

            # Calculate theoretical price
            flag = 'c' if option_type == OptionType.CALL else 'p'
            t = max(dte / 365.0, 0.001)
            r = self.config.risk_free_rate

            theo_price = bs_price(flag, underlying_price, strike, t, r, iv)

            if theo_price <= 0:
                return None

            # Simulate bid/ask spread
            moneyness = abs(strike - underlying_price) / underlying_price
            spread_pct = self.synthetic_config.spread_pct_atm + \
                         moneyness * (self.synthetic_config.spread_pct_otm - self.synthetic_config.spread_pct_atm)

            half_spread = theo_price * spread_pct / 2
            bid = max(0.01, theo_price - half_spread)
            ask = theo_price + half_spread

            # Calculate Greeks
            delta = bs_delta(flag, underlying_price, strike, t, r, iv)
            gamma = bs_gamma(flag, underlying_price, strike, t, r, iv)
            theta = bs_theta(flag, underlying_price, strike, t, r, iv) * 365  # Daily
            vega = bs_vega(flag, underlying_price, strike, t, r, iv)

            # Generate contract symbol
            exp_str = expiration.strftime('%y%m%d')
            opt_type_str = 'C' if option_type == OptionType.CALL else 'P'
            contract_symbol = f"{symbol}{exp_str}{opt_type_str}{int(strike * 1000):08d}"

            contract = OptionContract(
                symbol=contract_symbol,
                underlying_symbol=symbol,
                underlying_price=underlying_price,
                strike=strike,
                expiration=expiration,
                option_type=option_type,
                bid=round(bid, 2),
                ask=round(ask, 2),
                last_price=round(theo_price, 2),
                volume=0,  # Synthetic - no volume
                open_interest=0,  # Synthetic - no OI
                implied_volatility=iv,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
            )

            return contract

        except Exception as e:
            logger.debug(f"Error generating contract: {e}")
            return None

    def _load_cache_from_disk(self):
        """Load IV surface cache from disk."""
        cache_file = self.synthetic_config.cache_dir / "iv_surfaces.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                for symbol, surface_data in data.items():
                    self._iv_cache[symbol] = CachedIVSurface.from_dict(surface_data)

                logger.debug(f"Loaded {len(self._iv_cache)} IV surfaces from cache")

            except Exception as e:
                logger.warning(f"Error loading IV cache: {e}")

    def _save_cache_to_disk(self):
        """Save IV surface cache to disk."""
        cache_file = self.synthetic_config.cache_dir / "iv_surfaces.json"

        try:
            data = {
                symbol: surface.to_dict()
                for symbol, surface in self._iv_cache.items()
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Error saving IV cache: {e}")

    def refresh_iv_surface(self, symbol: str) -> bool:
        """Force refresh IV surface for a symbol."""
        surface = self._build_iv_surface_from_market(symbol)
        if surface:
            self._iv_cache[symbol] = surface
            self._save_cache_to_disk()
            return True
        return False

    def get_cached_symbols(self) -> List[str]:
        """Get list of symbols with cached IV surfaces."""
        return list(self._iv_cache.keys())

    def clear_cache(self):
        """Clear all cached IV surfaces."""
        self._iv_cache.clear()
        cache_file = self.synthetic_config.cache_dir / "iv_surfaces.json"
        if cache_file.exists():
            cache_file.unlink()
