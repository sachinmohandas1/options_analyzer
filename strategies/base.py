"""
Base strategy classes and interfaces.
All strategies inherit from BaseStrategy to ensure consistent interface.

Enhanced features:
- New composite scoring system with IV rank, liquidity, and risk factors
- Earnings calendar integration
- CVaR-adjusted risk assessment
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import date
import logging

from core.models import OptionsChain, TradeCandidate, OptionContract, VolatilitySurface
from core.config import AnalyzerConfig, TradeCriteria

logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Liquidity Scoring
# ============================================================================

class LiquidityCalculator:
    """
    Calculate comprehensive liquidity scores for options.

    Weighted composite of:
    - Bid-ask spread (40%) - tighter = better
    - Open interest (35%) - higher = better
    - Volume (25%) - higher = better
    """

    @staticmethod
    def calculate_spread_score(bid: float, ask: float, mid: float) -> float:
        """
        Score based on bid-ask spread.
        5% spread = 0, 0% spread = 100
        """
        if mid <= 0 or bid <= 0:
            return 0.0

        spread_pct = (ask - bid) / mid
        score = max(0, 100 - spread_pct * 2000)
        return min(100, score)

    @staticmethod
    def calculate_oi_score(open_interest: int, threshold: int = 1000) -> float:
        """Score based on open interest. 1000+ OI = 100."""
        if open_interest <= 0:
            return 0.0
        return min(100, (open_interest / threshold) * 100)

    @staticmethod
    def calculate_volume_score(volume: int, threshold: int = 500) -> float:
        """Score based on daily volume. 500+ volume = 100."""
        if volume <= 0:
            return 0.0
        return min(100, (volume / threshold) * 100)

    @classmethod
    def calculate_composite_score(
        cls,
        contracts: List[OptionContract],
        weights: tuple = (0.40, 0.35, 0.25)
    ) -> float:
        """
        Calculate weighted composite liquidity score for a list of contracts.

        Returns score from 0-100.
        """
        if not contracts:
            return 0.0

        scores = []
        for c in contracts:
            spread_score = cls.calculate_spread_score(c.bid, c.ask, c.mid_price)
            oi_score = cls.calculate_oi_score(c.open_interest)
            vol_score = cls.calculate_volume_score(c.volume)

            composite = (
                spread_score * weights[0] +
                oi_score * weights[1] +
                vol_score * weights[2]
            )
            scores.append(composite)

        return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# Shared Risk Metrics Instances (singleton pattern for caching)
# ============================================================================

_shared_earnings_calendar = None
_shared_cvar_calculator = None
_shared_risk_assessor = None


def get_shared_earnings_calendar():
    """Get shared EarningsCalendar instance (singleton for caching)."""
    global _shared_earnings_calendar
    if _shared_earnings_calendar is None:
        try:
            from analysis.risk_metrics import EarningsCalendar
            _shared_earnings_calendar = EarningsCalendar()
        except ImportError:
            pass
    return _shared_earnings_calendar


def get_shared_cvar_calculator():
    """Get shared CVaRCalculator instance (singleton for caching)."""
    global _shared_cvar_calculator
    if _shared_cvar_calculator is None:
        try:
            from analysis.risk_metrics import CVaRCalculator
            _shared_cvar_calculator = CVaRCalculator()
        except ImportError:
            pass
    return _shared_cvar_calculator


def get_shared_risk_assessor():
    """Get shared RiskAssessor instance (singleton for caching)."""
    global _shared_risk_assessor
    if _shared_risk_assessor is None:
        try:
            from analysis.risk_metrics import RiskAssessor
            _shared_risk_assessor = RiskAssessor()
        except ImportError:
            pass
    return _shared_risk_assessor


# ============================================================================
# Enhanced Scoring System
# ============================================================================

@dataclass
class ScoringContext:
    """Context for scoring a trade candidate."""
    iv_rank: Optional[float] = None  # 0-100
    iv_percentile: Optional[float] = None  # 0-100
    days_to_earnings: Optional[int] = None
    earnings_in_window: bool = False
    cvar_95: Optional[float] = None
    market_regime: str = "normal"  # "normal", "stressed", "low_vol"


class EnhancedScorer:
    """
    New composite scoring system for trade candidates.

    Factors:
    1. Weekly return (25%) - higher is better
    2. Probability of profit (25%) - higher is better
    3. Liquidity (20%) - composite score
    4. IV rank (15%) - higher IV = better for selling
    5. Theta efficiency (15%) - theta per dollar at risk

    Risk multipliers (applied after base score):
    - Earnings penalty: 0.5 if earnings in trade window
    - Regime factor: 0.7 if market stressed
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.criteria = config.trade_criteria

        # Scoring weights
        self.weights = {
            'weekly_return': 0.25,
            'prob_profit': 0.25,
            'liquidity': 0.20,
            'iv_rank': 0.15,
            'theta_efficiency': 0.15
        }

    def score_candidate(
        self,
        candidate: TradeCandidate,
        context: Optional[ScoringContext] = None,
        is_credit_strategy: bool = True
    ) -> float:
        """
        Calculate composite score for a trade candidate.

        Args:
            candidate: Trade to score
            context: Additional context (IV rank, earnings, etc.)
            is_credit_strategy: Whether this is a premium selling strategy

        Returns:
            Score from 0-100
        """
        context = context or ScoringContext()

        # 1. Weekly return score (0-100)
        # Target: 1-3% weekly is good, cap at 5%
        weekly_return_pct = candidate.weekly_return * 100
        weekly_score = min(100, (weekly_return_pct / 3.0) * 100)

        # 2. Probability of profit score (0-100)
        # 70% is minimum, 90% is excellent
        prob_score = max(0, (candidate.prob_profit - 0.5) * 200)
        prob_score = min(100, prob_score)

        # 3. Liquidity score (already 0-100)
        liquidity_score = candidate.liquidity_score

        # 4. IV rank score (0-100)
        # Higher IV = better for selling premium
        iv_rank = context.iv_rank or candidate.iv_rank or 50
        if is_credit_strategy:
            iv_score = iv_rank  # Direct mapping for selling
        else:
            iv_score = 100 - iv_rank  # Inverse for buying

        # 5. Theta efficiency score (0-100)
        # Theta per dollar at risk
        theta_efficiency = 0.0
        if candidate.max_loss > 0 and candidate.net_theta > 0:
            # Daily theta as % of max loss
            theta_pct = (candidate.net_theta / candidate.max_loss) * 100
            # 0.5% daily theta is excellent
            theta_efficiency = min(100, theta_pct * 200)

        # Calculate weighted base score
        base_score = (
            weekly_score * self.weights['weekly_return'] +
            prob_score * self.weights['prob_profit'] +
            liquidity_score * self.weights['liquidity'] +
            iv_score * self.weights['iv_rank'] +
            theta_efficiency * self.weights['theta_efficiency']
        )

        # Apply risk multipliers
        multiplier = 1.0

        # Earnings penalty
        if context.earnings_in_window:
            multiplier *= 0.5
            logger.debug(f"{candidate.underlying_symbol}: Earnings penalty applied")
        elif context.days_to_earnings is not None and context.days_to_earnings <= 3:
            multiplier *= 0.7
            logger.debug(f"{candidate.underlying_symbol}: Near-earnings penalty applied")

        # Market regime factor
        if context.market_regime == "stressed":
            multiplier *= 0.7
        elif context.market_regime == "low_vol":
            # Low vol = harder to find good premium
            multiplier *= 0.9

        # CVaR penalty for high tail risk
        if context.cvar_95 is not None and context.cvar_95 > 0.05:
            # High CVaR (>5% daily) is risky
            cvar_penalty = max(0.7, 1 - (context.cvar_95 - 0.05) * 5)
            multiplier *= cvar_penalty

        final_score = base_score * multiplier

        return max(0, min(100, final_score))

    def create_context_from_chain(
        self,
        chain: OptionsChain,
        candidate: TradeCandidate
    ) -> ScoringContext:
        """Create scoring context from options chain data.

        Note: Uses shared singleton instances to avoid redundant API calls.
        Risk data should already be fetched during enrich_candidate().
        """
        context = ScoringContext(
            iv_rank=chain.iv_rank,
            iv_percentile=chain.iv_percentile
        )

        # Check if candidate already has risk data from enrich_candidate()
        if hasattr(candidate, '_risk_metrics') and candidate._risk_metrics is not None:
            rm = candidate._risk_metrics
            context.days_to_earnings = rm.days_to_earnings
            context.earnings_in_window = rm.earnings_in_trade_window
            context.cvar_95 = rm.cvar_95
            return context

        # Fallback: fetch using shared instances (should rarely happen)
        earnings_cal = get_shared_earnings_calendar()
        if earnings_cal:
            try:
                risk_level, days = earnings_cal.get_earnings_risk(
                    chain.underlying_symbol,
                    candidate.expiration or date.today()
                )
                context.days_to_earnings = days
                context.earnings_in_window = (risk_level == "high")
            except Exception:
                pass

        cvar_calc = get_shared_cvar_calculator()
        if cvar_calc:
            try:
                returns = cvar_calc.get_historical_returns(chain.underlying_symbol)
                if returns is not None:
                    context.cvar_95 = cvar_calc.calculate_cvar(returns, 0.95)
            except Exception:
                pass

        return context


# ============================================================================
# Base Strategy Class
# ============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all option strategies.
    Implement this to add new strategy types.
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.criteria = config.trade_criteria

        # Initialize enhanced scorer
        self._scorer = EnhancedScorer(config)
        self._liquidity_calc = LiquidityCalculator()

        # Note: Risk assessors now use module-level singletons for caching
        # See get_shared_risk_assessor() and get_shared_earnings_calendar()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy display name."""
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Strategy type identifier."""
        pass

    @property
    @abstractmethod
    def is_credit_strategy(self) -> bool:
        """Whether this strategy receives premium (credit) at open."""
        pass

    @property
    def requires_stock_ownership(self) -> bool:
        """Whether strategy requires owning the underlying."""
        return False

    def _get_risk_assessor(self):
        """Get shared risk assessor (uses module-level singleton for caching)."""
        return get_shared_risk_assessor()

    def _get_earnings_calendar(self):
        """Get shared earnings calendar (uses module-level singleton for caching)."""
        return get_shared_earnings_calendar()

    @abstractmethod
    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """
        Find all trade candidates for this strategy from the given chain.
        """
        pass

    @abstractmethod
    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """
        Calculate all trade metrics (P&L, Greeks, probabilities).
        """
        pass

    def filter_by_criteria(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """
        Filter candidates by configured criteria.
        Override for strategy-specific filtering.
        """
        filtered = []

        for c in candidates:
            # Weekly return check
            if c.weekly_return < self.criteria.min_weekly_return_pct / 100:
                continue

            # Probability check
            if c.prob_profit < self.criteria.min_prob_profit:
                continue

            # DTE check
            if not (self.criteria.min_dte <= c.dte <= self.criteria.max_dte):
                continue

            # Delta check (for short options, want low delta)
            if abs(c.net_delta) > self.criteria.max_delta:
                continue

            # Premium check
            if c.premium_received < self.criteria.min_premium:
                continue

            filtered.append(c)

        return filtered

    def score_candidate(
        self,
        candidate: TradeCandidate,
        chain: Optional[OptionsChain] = None
    ) -> float:
        """
        Score a candidate from 0-100 for ranking using the enhanced scoring system.
        Higher score = more attractive trade.
        """
        # Create scoring context
        context = ScoringContext()

        if chain:
            context = self._scorer.create_context_from_chain(chain, candidate)
        else:
            # Use whatever data is on the candidate
            context.iv_rank = candidate.iv_rank

        # Calculate and store score
        score = self._scorer.score_candidate(
            candidate,
            context,
            is_credit_strategy=self.is_credit_strategy
        )

        candidate.overall_score = score
        return score

    def enrich_candidate(
        self,
        candidate: TradeCandidate,
        chain: Optional[OptionsChain] = None
    ) -> TradeCandidate:
        """
        Enrich candidate with additional metrics (liquidity, risk assessment).
        """
        # Calculate liquidity score
        candidate.liquidity_score = self._calculate_liquidity_score(candidate.legs)

        # Get IV rank from chain
        if chain and chain.iv_rank is not None:
            candidate.iv_rank = chain.iv_rank

        # Add risk metrics if available
        risk_assessor = self._get_risk_assessor()
        if risk_assessor and candidate.expiration:
            try:
                risk_metrics = risk_assessor.assess_trade_risk(
                    candidate.underlying_symbol,
                    candidate.expiration,
                    candidate.max_loss,
                    candidate.dte,
                    candidate.legs
                )

                # Store risk metrics on candidate (as custom attributes)
                candidate._risk_metrics = risk_metrics
                candidate._earnings_risk = risk_metrics.earnings_risk_flag
                candidate._cvar_95 = risk_metrics.cvar_95

            except Exception as e:
                logger.debug(f"Could not assess risk for {candidate.underlying_symbol}: {e}")

        return candidate

    def _calculate_liquidity_score(self, contracts: List[OptionContract]) -> float:
        """Calculate liquidity score using enhanced calculator."""
        return self._liquidity_calc.calculate_composite_score(contracts)

    def _passes_liquidity_check(self, contract: OptionContract) -> bool:
        """Check if contract meets liquidity requirements."""
        if contract.volume < self.criteria.min_volume:
            return False
        if contract.open_interest < self.criteria.min_open_interest:
            return False
        if contract.bid_ask_spread_pct > self.criteria.max_bid_ask_spread_pct:
            return False
        return True

    def _check_earnings_risk(
        self,
        symbol: str,
        expiration: date
    ) -> tuple:
        """
        Check earnings risk for a trade.

        Returns:
            (risk_level, days_to_earnings)
        """
        earnings_cal = self._get_earnings_calendar()
        if earnings_cal:
            try:
                return earnings_cal.get_earnings_risk(symbol, expiration)
            except Exception as e:
                logger.debug(f"Could not check earnings for {symbol}: {e}")

        return "low", None


# ============================================================================
# Legacy Scoring (for backwards compatibility)
# ============================================================================

def legacy_score_candidate(candidate: TradeCandidate, criteria: TradeCriteria) -> float:
    """
    Legacy scoring method for backwards compatibility.
    Uses the original 6-factor scoring.
    """
    score = 0.0

    # Probability of profit (0-30 points)
    prob_score = (candidate.prob_profit - 0.5) * 60
    score += max(0, min(30, prob_score))

    # Return on risk (0-25 points)
    if candidate.return_on_risk > 0:
        ror_score = min(25, candidate.return_on_risk * 100)
        score += ror_score

    # Weekly return (0-20 points)
    weekly_score = min(20, candidate.weekly_return * 1000)
    score += weekly_score

    # IV rank bonus (0-10 points)
    if candidate.iv_rank:
        iv_score = candidate.iv_rank / 10
        score += min(10, iv_score)

    # Liquidity score (0-10 points)
    score += min(10, candidate.liquidity_score)

    # Theta bonus (0-5 points)
    if candidate.net_theta > 0:
        theta_score = min(5, candidate.net_theta / 10)
        score += theta_score

    return score
