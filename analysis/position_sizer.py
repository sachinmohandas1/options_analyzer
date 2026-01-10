"""
Position Sizing Module.

Manages capital allocation and ensures trades stay within risk limits.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from core.models import TradeCandidate, AnalysisResult
from core.config import AnalyzerConfig, CapitalConfig


@dataclass
class PositionAllocation:
    """Represents capital allocated to a potential position."""
    candidate: TradeCandidate
    contracts: int
    collateral_required: float
    max_loss: float
    max_profit: float
    capital_utilization_pct: float


@dataclass
class Portfolio:
    """Current portfolio state for position sizing calculations."""
    total_capital: float
    deployed_capital: float = 0.0
    reserved_capital: float = 0.0
    available_capital: float = 0.0

    # Position tracking
    positions_by_underlying: Dict[str, int] = field(default_factory=dict)
    total_positions: int = 0

    def __post_init__(self):
        self.available_capital = self.total_capital - self.deployed_capital - self.reserved_capital


class PositionSizer:
    """
    Determines appropriate position sizes based on:
    - Available capital
    - Risk limits per position
    - Portfolio-level constraints
    - Correlation considerations
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.capital_config = config.capital
        self.portfolio = Portfolio(
            total_capital=config.capital.total_capital,
            reserved_capital=config.capital.reserve_cash
        )

    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.portfolio = Portfolio(
            total_capital=self.capital_config.total_capital,
            reserved_capital=self.capital_config.reserve_cash
        )

    def calculate_max_contracts(self, candidate: TradeCandidate) -> int:
        """
        Calculate maximum number of contracts for a given trade.

        Considers:
        - Available capital
        - Max single position size
        - Per-underlying limits
        """
        if candidate.collateral_required <= 0:
            return 0

        # Capital constraints
        available = self.portfolio.available_capital
        max_per_position = self.capital_config.max_single_position

        # Use the smaller of available capital and max position size
        usable_capital = min(available, max_per_position)

        # Calculate contracts that fit
        max_contracts = int(usable_capital / candidate.collateral_required)

        # Check per-underlying limit
        current_positions = self.portfolio.positions_by_underlying.get(
            candidate.underlying_symbol, 0
        )
        remaining_slots = (
            self.capital_config.max_positions_per_underlying - current_positions
        )
        max_contracts = min(max_contracts, remaining_slots)

        # Check total positions limit
        remaining_total = (
            self.capital_config.max_concurrent_positions - self.portfolio.total_positions
        )
        max_contracts = min(max_contracts, remaining_total)

        return max(0, max_contracts)

    def size_position(
        self,
        candidate: TradeCandidate,
        target_contracts: Optional[int] = None
    ) -> Optional[PositionAllocation]:
        """
        Size a position for a trade candidate.

        Returns allocation details or None if trade cannot be sized.
        """
        max_contracts = self.calculate_max_contracts(candidate)

        if max_contracts == 0:
            return None

        # Use target if specified and valid, otherwise use max
        contracts = min(target_contracts or max_contracts, max_contracts)

        if contracts == 0:
            return None

        collateral = candidate.collateral_required * contracts
        max_loss = candidate.max_loss * contracts
        max_profit = candidate.max_profit * contracts

        return PositionAllocation(
            candidate=candidate,
            contracts=contracts,
            collateral_required=collateral,
            max_loss=max_loss,
            max_profit=max_profit,
            capital_utilization_pct=collateral / self.capital_config.total_capital * 100
        )

    def allocate_position(self, allocation: PositionAllocation) -> bool:
        """
        Allocate capital for a position.

        Returns True if allocation successful, False otherwise.
        """
        if allocation.collateral_required > self.portfolio.available_capital:
            return False

        # Update portfolio
        self.portfolio.deployed_capital += allocation.collateral_required
        self.portfolio.available_capital -= allocation.collateral_required

        symbol = allocation.candidate.underlying_symbol
        if symbol not in self.portfolio.positions_by_underlying:
            self.portfolio.positions_by_underlying[symbol] = 0
        self.portfolio.positions_by_underlying[symbol] += allocation.contracts

        self.portfolio.total_positions += allocation.contracts

        return True

    def filter_by_capital(
        self,
        candidates: List[TradeCandidate]
    ) -> List[TradeCandidate]:
        """
        Filter candidates that can be traded with available capital.
        """
        return [
            c for c in candidates
            if c.collateral_required <= self.portfolio.available_capital
            and c.collateral_required <= self.capital_config.max_single_position
        ]

    def optimize_allocation(
        self,
        candidates: List[TradeCandidate],
        strategy: str = "score"
    ) -> List[PositionAllocation]:
        """
        Optimize capital allocation across multiple candidates.

        Strategies:
        - "score": Prioritize by overall score
        - "return": Prioritize by expected return
        - "probability": Prioritize by probability of profit
        - "diversified": Balance across underlyings

        Returns list of allocations that maximize the objective.
        """
        allocations = []

        # Sort candidates based on strategy
        if strategy == "score":
            sorted_candidates = sorted(candidates, key=lambda x: x.overall_score, reverse=True)
        elif strategy == "return":
            sorted_candidates = sorted(candidates, key=lambda x: x.weekly_return, reverse=True)
        elif strategy == "probability":
            sorted_candidates = sorted(candidates, key=lambda x: x.prob_profit, reverse=True)
        elif strategy == "diversified":
            # Group by underlying and interleave
            by_underlying = {}
            for c in candidates:
                if c.underlying_symbol not in by_underlying:
                    by_underlying[c.underlying_symbol] = []
                by_underlying[c.underlying_symbol].append(c)

            # Sort each group by score
            for symbol in by_underlying:
                by_underlying[symbol].sort(key=lambda x: x.overall_score, reverse=True)

            # Interleave (round-robin)
            sorted_candidates = []
            while any(by_underlying.values()):
                for symbol in list(by_underlying.keys()):
                    if by_underlying[symbol]:
                        sorted_candidates.append(by_underlying[symbol].pop(0))
                    if not by_underlying[symbol]:
                        del by_underlying[symbol]
        else:
            sorted_candidates = candidates

        # Greedy allocation
        for candidate in sorted_candidates:
            allocation = self.size_position(candidate, target_contracts=1)
            if allocation:
                if self.allocate_position(allocation):
                    allocations.append(allocation)

        return allocations

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio state summary."""
        return {
            "total_capital": self.capital_config.total_capital,
            "deployed_capital": self.portfolio.deployed_capital,
            "available_capital": self.portfolio.available_capital,
            "reserved_capital": self.portfolio.reserved_capital,
            "utilization_pct": (
                self.portfolio.deployed_capital / self.capital_config.total_capital * 100
                if self.capital_config.total_capital > 0 else 0
            ),
            "total_positions": self.portfolio.total_positions,
            "positions_by_underlying": dict(self.portfolio.positions_by_underlying),
        }
