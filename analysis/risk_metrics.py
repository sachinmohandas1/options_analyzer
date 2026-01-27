"""
Risk metrics module for options analysis.

Provides CVaR (Conditional Value at Risk), earnings calendar integration,
and other risk assessment tools.
"""

import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk assessment for a trade candidate."""
    # CVaR / Expected Shortfall
    cvar_95: Optional[float] = None  # Expected loss given we're in worst 5%
    cvar_99: Optional[float] = None  # Expected loss given we're in worst 1%

    # Earnings risk
    days_to_earnings: Optional[int] = None
    earnings_in_trade_window: bool = False
    earnings_risk_flag: str = "low"  # "low", "elevated", "high"

    # Tail risk metrics
    max_historical_move: Optional[float] = None  # Largest % move in lookback
    tail_risk_percentile: Optional[float] = None  # How often moves exceed our breakeven

    # Liquidity risk
    liquidity_score: float = 0.0  # 0-100 composite score
    estimated_slippage_pct: float = 0.0


@dataclass
class EarningsEvent:
    """Earnings event data."""
    symbol: str
    earnings_date: date
    is_confirmed: bool = False
    timing: str = "unknown"  # "BMO" (before market open), "AMC" (after market close), "unknown"


class EarningsCalendar:
    """
    Earnings calendar integration for risk filtering.

    Uses yfinance calendar data with caching to avoid repeated API calls.
    """

    def __init__(self, cache_ttl_hours: int = 24):
        self._cache: Dict[str, EarningsEvent] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._no_earnings_cache: Dict[str, datetime] = {}  # Track symbols with no earnings data

    def get_next_earnings(self, symbol: str, use_cache: bool = True) -> Optional[EarningsEvent]:
        """
        Get next earnings date for a symbol.

        Returns None if no earnings data available or earnings are far out.
        """
        now = datetime.now()

        # Check cache
        if use_cache:
            if symbol in self._cache:
                if now - self._cache_time[symbol] < self._cache_ttl:
                    return self._cache[symbol]

            # Check if we recently found no earnings data
            if symbol in self._no_earnings_cache:
                if now - self._no_earnings_cache[symbol] < self._cache_ttl:
                    return None

        try:
            # Temporarily suppress yfinance ERROR logging for calendar fetch
            # ETFs and some symbols don't have calendar data, which triggers
            # HTTP 404 "No fundamentals data found" - this is expected, not an error
            yf_logger = logging.getLogger('yfinance')
            original_level = yf_logger.level
            yf_logger.setLevel(logging.CRITICAL)

            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
            finally:
                yf_logger.setLevel(original_level)

            # Handle None or empty calendar
            if calendar is None:
                self._no_earnings_cache[symbol] = now
                return None

            # Check for empty - dict or DataFrame
            if isinstance(calendar, dict):
                if not calendar:
                    self._no_earnings_cache[symbol] = now
                    return None
            elif hasattr(calendar, 'empty') and calendar.empty:
                self._no_earnings_cache[symbol] = now
                return None

            # yfinance returns calendar as dict (newer) or DataFrame (older)
            # Format varies - try multiple access patterns
            earnings_date = None

            # New format: dict with 'Earnings Date' as list of dates
            if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                ed = calendar['Earnings Date']
                # Handle list of dates (new yfinance format)
                if isinstance(ed, list) and len(ed) > 0:
                    earnings_date = ed[0]
                else:
                    earnings_date = ed
            # Old DataFrame formats
            elif hasattr(calendar, 'T') and 'Earnings Date' in calendar.T.columns:
                earnings_dates = calendar.T['Earnings Date'].values
                if len(earnings_dates) > 0:
                    earnings_date = earnings_dates[0]
            elif hasattr(calendar, 'index') and 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date'].values[0]

            if earnings_date is None:
                self._no_earnings_cache[symbol] = now
                return None

            # Convert to date object if needed
            if isinstance(earnings_date, date) and not isinstance(earnings_date, datetime):
                pass  # Already a date object
            elif hasattr(earnings_date, 'date'):
                earnings_date = earnings_date.date()
            elif isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date[:10], '%Y-%m-%d').date()
            elif isinstance(earnings_date, np.datetime64):
                earnings_date = datetime.utcfromtimestamp(earnings_date.astype('datetime64[s]').astype('int')).date()

            event = EarningsEvent(
                symbol=symbol,
                earnings_date=earnings_date,
                is_confirmed=True
            )

            self._cache[symbol] = event
            self._cache_time[symbol] = now

            return event

        except Exception as e:
            logger.debug(f"Could not fetch earnings for {symbol}: {e}")
            self._no_earnings_cache[symbol] = now
            return None

    def get_earnings_risk(
        self,
        symbol: str,
        trade_expiration: date
    ) -> Tuple[str, Optional[int]]:
        """
        Assess earnings risk for a trade.

        Returns:
            (risk_level, days_to_earnings)
            risk_level: "low", "elevated", "high"
        """
        event = self.get_next_earnings(symbol)

        if event is None:
            return "low", None

        today = date.today()
        days_to_earnings = (event.earnings_date - today).days
        days_to_expiration = (trade_expiration - today).days

        # Earnings before our trade expires = high risk
        if 0 <= days_to_earnings <= days_to_expiration:
            return "high", days_to_earnings

        # Earnings within a week after expiration = still some risk
        # (IV crush expectations might already be affecting pricing)
        if days_to_expiration < days_to_earnings <= days_to_expiration + 7:
            return "elevated", days_to_earnings

        # Earnings very soon (entering a trade 1-3 days before earnings)
        if 1 <= days_to_earnings <= 3:
            return "high", days_to_earnings

        return "low", days_to_earnings

    def batch_fetch(self, symbols: List[str]) -> Dict[str, Optional[EarningsEvent]]:
        """Fetch earnings for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_next_earnings(symbol)
        return results


class CVaRCalculator:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is superior to VaR for options because it captures tail risk severity,
    which is critical for short premium strategies.
    """

    def __init__(self, lookback_days: int = 252, min_data_points: int = 60):
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self._returns_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._cache_ttl = timedelta(hours=4)

    def get_historical_returns(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """Fetch historical returns for CVaR calculation."""
        now = datetime.now()

        # Check cache
        if use_cache and symbol in self._returns_cache:
            returns, cache_time = self._returns_cache[symbol]
            if now - cache_time < self._cache_ttl:
                return returns

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f'{self.lookback_days + 30}d')

            if len(hist) < self.min_data_points:
                logger.debug(f"{symbol}: Insufficient history for CVaR ({len(hist)} points)")
                return None

            # Calculate daily returns
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values

            self._returns_cache[symbol] = (returns, now)
            return returns

        except Exception as e:
            logger.debug(f"Error fetching returns for {symbol}: {e}")
            return None

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate CVaR (Expected Shortfall) at given confidence level.

        CVaR is the expected loss given that we're in the worst (1-confidence) percentile.
        For 95% confidence, it's the average of the worst 5% of outcomes.

        Returns:
            CVaR as a positive number (representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        # Find the VaR threshold
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        # CVaR is the mean of returns below VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return abs(var)

        cvar = np.mean(tail_returns)

        # Return as positive number for intuitive interpretation
        return abs(cvar)

    def calculate_trade_cvar(
        self,
        symbol: str,
        max_loss: float,
        dte: int,
        confidence: float = 0.95
    ) -> Optional[float]:
        """
        Calculate CVaR-adjusted risk for a specific trade.

        Scales historical daily CVaR to the trade's DTE and applies
        it to the max loss to estimate expected tail loss.
        """
        returns = self.get_historical_returns(symbol)

        if returns is None:
            return None

        daily_cvar = self.calculate_cvar(returns, confidence)

        # Scale to DTE (using sqrt for time scaling, standard for volatility)
        dte_cvar = daily_cvar * np.sqrt(max(1, dte))

        # Express as expected loss amount
        # This represents: if we're in the tail, what's our expected loss?
        return max_loss * (1 + dte_cvar)

    def get_tail_risk_metrics(
        self,
        symbol: str,
        breakeven_move_pct: float
    ) -> Dict[str, float]:
        """
        Calculate tail risk metrics for a trade.

        Args:
            symbol: Underlying symbol
            breakeven_move_pct: Percentage move to breakeven (as decimal, e.g., 0.05 for 5%)

        Returns:
            Dict with:
                - cvar_95: 95% CVaR
                - cvar_99: 99% CVaR
                - max_move: Maximum historical move
                - breach_probability: Historical probability of exceeding breakeven
        """
        returns = self.get_historical_returns(symbol)

        if returns is None:
            return {}

        return {
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'cvar_99': self.calculate_cvar(returns, 0.99),
            'max_move': float(np.abs(returns).max()),
            'breach_probability': float(np.mean(np.abs(returns) > breakeven_move_pct))
        }


class LiquidityScorer:
    """
    Calculate comprehensive liquidity scores for options.

    Weighted composite of:
    - Bid-ask spread (40%)
    - Open interest (35%)
    - Volume (25%)
    """

    @staticmethod
    def calculate_spread_score(bid: float, ask: float, mid: float) -> float:
        """
        Score based on bid-ask spread.

        Tighter spread = higher score.
        5% spread = 0, 0% spread = 100
        """
        if mid <= 0:
            return 0.0

        spread_pct = (ask - bid) / mid
        # 5% or wider = 0, 0% = 100
        score = max(0, 100 - spread_pct * 2000)
        return min(100, score)

    @staticmethod
    def calculate_oi_score(open_interest: int, threshold: int = 1000) -> float:
        """
        Score based on open interest.

        1000+ OI = 100, scales linearly below.
        """
        if open_interest <= 0:
            return 0.0

        score = min(100, (open_interest / threshold) * 100)
        return score

    @staticmethod
    def calculate_volume_score(volume: int, threshold: int = 500) -> float:
        """
        Score based on daily volume.

        500+ volume = 100, scales linearly below.
        """
        if volume <= 0:
            return 0.0

        score = min(100, (volume / threshold) * 100)
        return score

    @classmethod
    def calculate_composite_score(
        cls,
        bid: float,
        ask: float,
        mid: float,
        open_interest: int,
        volume: int,
        weights: Tuple[float, float, float] = (0.40, 0.35, 0.25)
    ) -> float:
        """
        Calculate weighted composite liquidity score.

        Args:
            weights: (spread_weight, oi_weight, volume_weight)

        Returns:
            Score from 0-100
        """
        spread_score = cls.calculate_spread_score(bid, ask, mid)
        oi_score = cls.calculate_oi_score(open_interest)
        vol_score = cls.calculate_volume_score(volume)

        composite = (
            spread_score * weights[0] +
            oi_score * weights[1] +
            vol_score * weights[2]
        )

        return composite

    @classmethod
    def estimate_slippage(
        cls,
        bid: float,
        ask: float,
        volume: int,
        order_size: int = 1
    ) -> float:
        """
        Estimate expected slippage as percentage of mid price.

        Larger orders relative to volume = more slippage.
        Wider spreads = more slippage.
        """
        if bid <= 0 or ask <= 0:
            return 5.0  # Max slippage assumption

        mid = (bid + ask) / 2
        half_spread = (ask - bid) / 2 / mid * 100  # As percentage

        # Volume-based adjustment (more aggressive for large orders)
        if volume > 0:
            volume_impact = (order_size / volume) * 0.5  # 0.5% per 1x volume ratio
        else:
            volume_impact = 1.0  # Assume 1% if no volume

        return min(5.0, half_spread + volume_impact)


# Module-level singleton instances for shared caching
_singleton_earnings_calendar: Optional[EarningsCalendar] = None
_singleton_cvar_calculator: Optional[CVaRCalculator] = None


def _get_singleton_earnings_calendar() -> EarningsCalendar:
    """Get or create singleton EarningsCalendar."""
    global _singleton_earnings_calendar
    if _singleton_earnings_calendar is None:
        _singleton_earnings_calendar = EarningsCalendar()
    return _singleton_earnings_calendar


def _get_singleton_cvar_calculator() -> CVaRCalculator:
    """Get or create singleton CVaRCalculator."""
    global _singleton_cvar_calculator
    if _singleton_cvar_calculator is None:
        _singleton_cvar_calculator = CVaRCalculator()
    return _singleton_cvar_calculator


class RiskAssessor:
    """
    Comprehensive risk assessment for trade candidates.

    Combines CVaR, earnings, and liquidity risk into a unified assessment.
    Uses module-level singletons for EarningsCalendar and CVaRCalculator
    to ensure caching works across all instances.
    """

    def __init__(self):
        # Use singletons to ensure caching works
        self.earnings_calendar = _get_singleton_earnings_calendar()
        self.cvar_calculator = _get_singleton_cvar_calculator()
        self.liquidity_scorer = LiquidityScorer()

    def assess_trade_risk(
        self,
        symbol: str,
        expiration: date,
        max_loss: float,
        dte: int,
        legs: List[Any]  # List of OptionContract
    ) -> RiskMetrics:
        """
        Perform comprehensive risk assessment for a trade.
        """
        metrics = RiskMetrics()

        # 1. Earnings risk
        earnings_risk, days_to_earnings = self.earnings_calendar.get_earnings_risk(
            symbol, expiration
        )
        metrics.earnings_risk_flag = earnings_risk
        metrics.days_to_earnings = days_to_earnings
        metrics.earnings_in_trade_window = earnings_risk == "high"

        # 2. CVaR calculation
        returns = self.cvar_calculator.get_historical_returns(symbol)
        if returns is not None:
            metrics.cvar_95 = self.cvar_calculator.calculate_cvar(returns, 0.95)
            metrics.cvar_99 = self.cvar_calculator.calculate_cvar(returns, 0.99)
            metrics.max_historical_move = float(np.abs(returns).max())

        # 3. Liquidity scoring (average across legs)
        if legs:
            leg_scores = []
            leg_slippages = []

            for leg in legs:
                score = self.liquidity_scorer.calculate_composite_score(
                    leg.bid, leg.ask, leg.mid_price,
                    leg.open_interest, leg.volume
                )
                leg_scores.append(score)

                slippage = self.liquidity_scorer.estimate_slippage(
                    leg.bid, leg.ask, leg.volume
                )
                leg_slippages.append(slippage)

            metrics.liquidity_score = np.mean(leg_scores)
            metrics.estimated_slippage_pct = np.mean(leg_slippages)

        return metrics
