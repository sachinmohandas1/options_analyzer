"""
Base strategy classes and interfaces.
All strategies inherit from BaseStrategy to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.models import OptionsChain, TradeCandidate, OptionContract, VolatilitySurface
from core.config import AnalyzerConfig, TradeCriteria


class BaseStrategy(ABC):
    """
    Abstract base class for all option strategies.
    Implement this to add new strategy types.
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.criteria = config.trade_criteria

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

    def score_candidate(self, candidate: TradeCandidate) -> float:
        """
        Score a candidate from 0-100 for ranking.
        Higher score = more attractive trade.
        Override for custom scoring logic.
        """
        score = 0.0

        # Probability of profit (0-30 points)
        prob_score = (candidate.prob_profit - 0.5) * 60  # 50% = 0, 80% = 18
        score += max(0, min(30, prob_score))

        # Return on risk (0-25 points)
        if candidate.return_on_risk > 0:
            ror_score = min(25, candidate.return_on_risk * 100)
            score += ror_score

        # Weekly return (0-20 points)
        weekly_score = min(20, candidate.weekly_return * 1000)  # 2% weekly = 20 points
        score += weekly_score

        # IV rank bonus (0-10 points) - higher IV = better for selling
        if candidate.iv_rank and self.is_credit_strategy:
            iv_score = candidate.iv_rank / 10
            score += min(10, iv_score)

        # Liquidity score (0-10 points)
        score += min(10, candidate.liquidity_score)

        # Theta bonus (0-5 points) - positive theta for credit strategies
        if self.is_credit_strategy and candidate.net_theta > 0:
            theta_score = min(5, candidate.net_theta / 10)
            score += theta_score

        return score

    def _calculate_liquidity_score(self, contracts: List[OptionContract]) -> float:
        """Calculate liquidity score based on volume, OI, and spread."""
        if not contracts:
            return 0.0

        scores = []
        for c in contracts:
            # Volume score (0-3)
            vol_score = min(3, c.volume / self.criteria.min_volume) if self.criteria.min_volume > 0 else 3

            # OI score (0-3)
            oi_score = min(3, c.open_interest / self.criteria.min_open_interest) if self.criteria.min_open_interest > 0 else 3

            # Spread score (0-4) - tighter is better
            if c.bid_ask_spread_pct < self.criteria.max_bid_ask_spread_pct:
                spread_score = 4 * (1 - c.bid_ask_spread_pct / self.criteria.max_bid_ask_spread_pct)
            else:
                spread_score = 0

            scores.append(vol_score + oi_score + spread_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _passes_liquidity_check(self, contract: OptionContract) -> bool:
        """Check if contract meets liquidity requirements."""
        if contract.volume < self.criteria.min_volume:
            return False
        if contract.open_interest < self.criteria.min_open_interest:
            return False
        if contract.bid_ask_spread_pct > self.criteria.max_bid_ask_spread_pct:
            return False
        return True
