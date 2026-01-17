"""
Data models for the options analyzer.
Using Pydantic for validation and dataclasses for internal structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import numpy as np


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str
    underlying_symbol: str
    underlying_price: float
    strike: float
    expiration: date
    option_type: OptionType
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float

    # Greeks (calculated)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Derived metrics
    dte: int = 0
    moneyness: float = 0.0  # strike / underlying_price
    bid_ask_spread: float = 0.0
    mid_price: float = 0.0

    def __post_init__(self):
        self.mid_price = (self.bid + self.ask) / 2 if self.bid and self.ask else self.last_price
        self.bid_ask_spread = self.ask - self.bid if self.bid and self.ask else 0
        if self.underlying_price > 0:
            self.moneyness = self.strike / self.underlying_price
        if isinstance(self.expiration, date):
            self.dte = (self.expiration - date.today()).days

    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money."""
        if self.option_type == OptionType.CALL:
            return self.underlying_price > self.strike
        return self.underlying_price < self.strike

    @property
    def is_otm(self) -> bool:
        """Check if option is out-of-the-money."""
        return not self.is_itm

    @property
    def prob_otm(self) -> float:
        """Probability of expiring OTM (approximated by 1 - |delta|)."""
        if self.delta is None:
            return 0.5
        return 1.0 - abs(self.delta)

    @property
    def prob_itm(self) -> float:
        """Probability of expiring ITM (approximated by |delta|)."""
        if self.delta is None:
            return 0.5
        return abs(self.delta)

    @property
    def bid_ask_spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid price."""
        if self.mid_price > 0:
            return (self.bid_ask_spread / self.mid_price) * 100
        return float('inf')


@dataclass
class OptionsChain:
    """Represents an options chain for a single underlying."""
    underlying_symbol: str
    underlying_price: float
    fetch_time: datetime
    expirations: List[date]
    calls: List[OptionContract]
    puts: List[OptionContract]

    # Market data
    historical_volatility: Optional[float] = None
    iv_rank: Optional[float] = None  # IV percentile rank
    iv_percentile: Optional[float] = None

    def get_contracts_by_expiration(self, expiration: date) -> Dict[str, List[OptionContract]]:
        """Get calls and puts for a specific expiration."""
        return {
            "calls": [c for c in self.calls if c.expiration == expiration],
            "puts": [p for p in self.puts if p.expiration == expiration]
        }

    def get_contracts_by_dte_range(self, min_dte: int, max_dte: int) -> Dict[str, List[OptionContract]]:
        """Get contracts within a DTE range."""
        return {
            "calls": [c for c in self.calls if min_dte <= c.dte <= max_dte],
            "puts": [p for p in self.puts if min_dte <= p.dte <= max_dte]
        }


@dataclass
class TradeCandidate:
    """Represents a potential trade opportunity."""
    strategy_name: str
    underlying_symbol: str
    underlying_price: float
    legs: List[OptionContract]

    # Trade metrics
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven: float = 0.0
    breakeven_upper: Optional[float] = None  # For spreads/condors

    # Return calculations
    premium_received: float = 0.0
    collateral_required: float = 0.0
    return_on_risk: float = 0.0  # Premium / Max Loss
    return_on_collateral: float = 0.0  # Premium / Collateral
    annualized_return: float = 0.0

    # Probability metrics
    prob_profit: float = 0.0
    prob_max_profit: float = 0.0
    expected_value: float = 0.0

    # Greeks (aggregate)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Volatility metrics
    iv_at_entry: float = 0.0
    iv_rank: Optional[float] = None
    iv_skew_signal: Optional[str] = None  # "bullish", "bearish", "neutral"

    # Quality scores
    liquidity_score: float = 0.0
    overall_score: float = 0.0

    # Metadata
    dte: int = 0
    expiration: Optional[date] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.legs:
            self.expiration = self.legs[0].expiration
            self.dte = self.legs[0].dte

    @property
    def weekly_return(self) -> float:
        """
        Return projected to a weekly basis.

        Note: For very short DTE trades (1-2 days), this extrapolation
        may overstate achievable returns since you can't always roll
        into new positions immediately. Use return_on_collateral for
        actual per-trade return.
        """
        if self.dte > 0 and self.return_on_collateral > 0:
            # Scale return to 7-day basis
            # Cap at 5x the per-trade return to avoid extreme extrapolation
            raw_weekly = (self.return_on_collateral / self.dte) * 7
            return min(raw_weekly, self.return_on_collateral * 5)
        return 0.0

    @property
    def trade_return_pct(self) -> float:
        """Actual return percentage for this specific trade."""
        return self.return_on_collateral * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "Strategy": self.strategy_name,
            "Symbol": self.underlying_symbol,
            "Price": f"${self.underlying_price:.2f}",
            "DTE": self.dte,
            "Premium": f"${self.premium_received:.2f}",
            "Collateral": f"${self.collateral_required:.2f}",
            "Max Loss": f"${self.max_loss:.2f}",
            "Weekly Return": f"{self.weekly_return:.2%}",
            "Prob Profit": f"{self.prob_profit:.1%}",
            "Delta": f"{self.net_delta:.3f}",
            "Theta": f"${self.net_theta:.2f}",
            "IV Rank": f"{self.iv_rank:.0f}%" if self.iv_rank else "N/A",
            "Score": f"{self.overall_score:.1f}",
        }


@dataclass
class VolatilitySurface:
    """Represents a volatility surface for an underlying."""
    underlying_symbol: str
    underlying_price: float
    generated_at: datetime

    # Surface data: dict of {expiration: {strike: iv}}
    surface_data: Dict[date, Dict[float, float]] = field(default_factory=dict)

    # Interpolated surface (for visualization)
    strikes: Optional[np.ndarray] = None
    expirations: Optional[np.ndarray] = None  # In days
    iv_matrix: Optional[np.ndarray] = None

    # Skew analysis
    atm_iv_by_expiration: Dict[date, float] = field(default_factory=dict)
    skew_25d_by_expiration: Dict[date, float] = field(default_factory=dict)  # 25 delta put IV - 25 delta call IV
    butterfly_spread: Dict[date, float] = field(default_factory=dict)  # Curvature measure

    # Term structure
    term_structure: Dict[int, float] = field(default_factory=dict)  # DTE -> ATM IV

    # Anomalies detected
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def get_iv_at_strike_dte(self, strike: float, dte: int) -> Optional[float]:
        """Get interpolated IV for a specific strike and DTE."""
        # Find closest expiration
        for exp, strikes_dict in self.surface_data.items():
            exp_dte = (exp - date.today()).days
            if abs(exp_dte - dte) <= 1:  # Within 1 day
                # Find closest strike
                closest_strike = min(strikes_dict.keys(), key=lambda s: abs(s - strike))
                if abs(closest_strike - strike) / strike < 0.02:  # Within 2%
                    return strikes_dict[closest_strike]
        return None


@dataclass
class SentimentSignal:
    """
    Sentiment analysis result for a symbol or sector.

    Used as a risk filter rather than directional predictor.
    High IV + negative sentiment + low news volume = opportunity (fear without catalyst)
    High IV + news volume spike = avoid (binary event risk)
    """
    symbol: str
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0-1, based on article count and agreement

    # Risk filter signals
    news_volume: int  # Number of articles in window
    news_volume_zscore: float  # Unusual activity detector (>2 = event risk)
    sentiment_momentum: float  # Current - 24h ago (trend shift)

    # Metadata
    article_count: int
    avg_relevance: float  # How relevant articles are to the ticker
    dominant_sentiment: str  # "bullish", "bearish", "neutral"
    last_updated: datetime = field(default_factory=datetime.now)

    # Source articles (for transparency) - list of (title, url) tuples
    top_headlines: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def risk_flag(self) -> str:
        """
        Interpret sentiment as risk signal for options selling.

        Returns:
            "low_risk": Good conditions for premium selling
            "elevated": Caution advised
            "high_risk": Potential binary event, avoid or hedge
        """
        # High news volume = potential event
        if self.news_volume_zscore > 2.0:
            return "high_risk"

        # Rapid sentiment shift = uncertainty
        if abs(self.sentiment_momentum) > 0.3:
            return "elevated"

        # Low confidence = not enough data
        if self.confidence < 0.3:
            return "elevated"

        return "low_risk"

    @property
    def display_score(self) -> str:
        """Human-readable sentiment score."""
        if self.sentiment_score > 0.2:
            return f"+{self.sentiment_score:.2f} (Bullish)"
        elif self.sentiment_score < -0.2:
            return f"{self.sentiment_score:.2f} (Bearish)"
        else:
            return f"{self.sentiment_score:.2f} (Neutral)"


@dataclass
class AnalysisResult:
    """Container for complete analysis results."""
    generated_at: datetime
    config_snapshot: Dict[str, Any]  # Snapshot of config used

    # Results by underlying
    chains_analyzed: Dict[str, OptionsChain] = field(default_factory=dict)
    volatility_surfaces: Dict[str, VolatilitySurface] = field(default_factory=dict)

    # Trade recommendations
    all_candidates: List[TradeCandidate] = field(default_factory=list)
    filtered_candidates: List[TradeCandidate] = field(default_factory=list)
    top_candidates: List[TradeCandidate] = field(default_factory=list)

    # Summary statistics
    total_opportunities_found: int = 0
    opportunities_meeting_criteria: int = 0
    capital_deployable: float = 0.0

    # Errors encountered
    errors: List[Dict[str, str]] = field(default_factory=list)
