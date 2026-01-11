"""
Configuration management for the options analyzer.
All criteria are configurable and can be modified at runtime.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class StrategyType(Enum):
    """Supported option strategies."""
    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"


@dataclass
class TradeCriteria:
    """Configurable criteria for filtering and scoring trades."""
    min_weekly_return_pct: float = 1.0
    min_annualized_return_pct: float = 52.0
    min_prob_profit: float = 0.70
    min_prob_otm: float = 0.70
    max_dte: int = 5
    min_dte: int = 1
    max_delta: float = 0.30
    min_theta: float = 0.0
    max_gamma_risk: float = 0.10
    max_vega_risk: float = 0.50
    min_open_interest: int = 100
    min_volume: int = 50
    max_bid_ask_spread_pct: float = 5.0
    min_premium: float = 0.10
    min_spread_width: float = 1.0
    max_spread_width: float = 10.0
    min_iv_percentile: float = 0.0
    max_iv_percentile: float = 100.0
    prefer_high_iv: bool = True
    min_risk_reward_ratio: float = 0.0


@dataclass
class CapitalConfig:
    """Capital and position sizing configuration."""
    total_capital: float = 13000.0
    max_single_position_pct: float = 0.25
    max_total_exposure_pct: float = 0.80
    reserve_cash_pct: float = 0.20
    max_concurrent_positions: int = 10
    max_positions_per_underlying: int = 2

    @property
    def max_single_position(self) -> float:
        return self.total_capital * self.max_single_position_pct

    @property
    def max_total_exposure(self) -> float:
        return self.total_capital * self.max_total_exposure_pct

    @property
    def reserve_cash(self) -> float:
        return self.total_capital * self.reserve_cash_pct


@dataclass
class VolatilitySurfaceConfig:
    """Configuration for volatility surface analysis."""
    strike_range_pct: float = 0.20
    min_strikes_for_surface: int = 10
    interpolation_method: str = "cubic"
    analyze_skew: bool = True
    skew_percentile_threshold: float = 75.0
    analyze_term_structure: bool = True
    detect_iv_anomalies: bool = True
    anomaly_zscore_threshold: float = 2.0


@dataclass
class UnderlyingConfig:
    """Configuration for which underlyings to analyze."""

    # Maximum share price for upfront symbol filtering.
    # Default is effectively unlimited - trades are filtered by collateral
    # (max loss) vs available capital at the strategy level instead.
    # Use --max-price CLI option to add upfront price filtering if desired.
    max_share_price: float = float('inf')
    
    # Use dynamic discovery to scan broad universe
    use_dynamic_discovery: bool = True
    
    # Minimum average daily volume for liquidity
    min_avg_volume: int = 500000
    
    # Comprehensive symbol list - filtered by max_share_price at runtime
    default_symbols: List[str] = field(default_factory=lambda: [
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
        # Leveraged (high IV)
        "SOXL", "TQQQ", "SPXL", "TNA", "FAS",
        # VIX/Inverse
        "UVXY", "VXX", "SQQQ", "SPXS",
        # Tech stocks
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA", "AMD",
        "NFLX", "CRM", "ORCL", "INTC", "CSCO", "UBER", "SNAP", "SQ", "PYPL",
        "COIN", "HOOD", "PLTR", "NET", "CRWD", "ZS", "DDOG",
        # Finance
        "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "COF",
        # Healthcare
        "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "UNH", "CVS", "MRNA",
        # Consumer
        "WMT", "TGT", "COST", "HD", "LOW", "NKE", "SBUX", "MCD",
        # Energy
        "XOM", "CVX", "OXY", "SLB", "COP",
        # Industrial
        "BA", "CAT", "DE", "UPS", "FDX", "GE",
        # Auto
        "F", "GM", "RIVN", "AAL", "UAL", "DAL",
        # Media
        "DIS", "CMCSA", "ROKU",
        # Telecom
        "T", "VZ", "TMUS",
        # Meme/High IV
        "GME", "AMC", "SOFI", "NIO", "BABA",
        # More high volume
        "ZM", "ABNB", "DASH", "DKNG",
    ])
    
    excluded_symbols: List[str] = field(default_factory=list)
    custom_symbols: List[str] = field(default_factory=list)


@dataclass
class AnalyzerConfig:
    """Main configuration container."""
    trade_criteria: TradeCriteria = field(default_factory=TradeCriteria)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    volatility_surface: VolatilitySurfaceConfig = field(default_factory=VolatilitySurfaceConfig)
    underlyings: UnderlyingConfig = field(default_factory=UnderlyingConfig)
    
    enabled_strategies: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.CASH_SECURED_PUT,
        StrategyType.PUT_CREDIT_SPREAD,
        StrategyType.CALL_CREDIT_SPREAD,
        StrategyType.IRON_CONDOR,
    ])
    
    risk_free_rate: float = 0.05
    top_n_trades: int = 20

    def get_active_symbols(self) -> List[str]:
        """Get list of symbols to analyze."""
        symbols = set(self.underlyings.default_symbols)
        symbols.update(self.underlyings.custom_symbols)
        symbols -= set(self.underlyings.excluded_symbols)
        return sorted(list(symbols))


DEFAULT_CONFIG = AnalyzerConfig()
