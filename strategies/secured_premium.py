"""
Secured Premium Strategies.
- Cash Secured Puts (CSP)
- Covered Calls (requires stock ownership)

These are defined-risk premium collection strategies that don't
require naked margin privileges.
"""

from typing import List, Optional
from datetime import date

from core.models import (
    OptionsChain, TradeCandidate, OptionContract,
    VolatilitySurface, OptionType
)
from core.config import AnalyzerConfig
from strategies.base import BaseStrategy


class CashSecuredPutStrategy(BaseStrategy):
    """
    Cash Secured Put (CSP) Strategy.

    Sell OTM puts with cash collateral = strike * 100.
    Profit if stock stays above strike at expiration.
    Max profit = premium received.
    Max loss = strike price - premium (if stock goes to zero).

    This is a bullish to neutral strategy, ideal when you:
    - Are willing to buy the stock at the strike price
    - Believe the stock will stay flat or go up
    - Want to collect premium while waiting
    """

    @property
    def name(self) -> str:
        return "Cash Secured Put"

    @property
    def strategy_type(self) -> str:
        return "cash_secured_put"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """Find all CSP candidates from the options chain."""
        candidates = []

        for put in chain.puts:
            # Skip ITM puts (we want OTM)
            if put.is_itm:
                continue

            # Skip if doesn't meet liquidity requirements
            if not self._passes_liquidity_check(put):
                continue

            # Skip if delta too high (too close to ATM = higher risk)
            if put.delta is not None and abs(put.delta) > self.criteria.max_delta:
                continue

            # Create candidate
            candidate = self._create_candidate(put, chain, surface)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        put: OptionContract,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface]
    ) -> Optional[TradeCandidate]:
        """Create a trade candidate from a put option."""
        # Use bid price for premium (what we'd receive when selling)
        premium = put.bid * 100  # Per contract (100 shares)

        if premium <= 0:
            return None

        # Collateral required = strike * 100 (cash to secure the put)
        collateral = put.strike * 100

        # Max loss = collateral - premium (if stock goes to $0)
        max_loss = collateral - premium

        # Breakeven = strike - premium per share
        breakeven = put.strike - put.bid

        candidate = TradeCandidate(
            strategy_name=self.name,
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            legs=[put],
            max_profit=premium,
            max_loss=max_loss,
            breakeven=breakeven,
            premium_received=premium,
            collateral_required=collateral,
            iv_at_entry=put.implied_volatility,
            iv_rank=chain.iv_rank
        )

        return self.calculate_metrics(candidate)

    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """Calculate all trade metrics for the CSP."""
        put = candidate.legs[0]

        # Return calculations
        if candidate.collateral_required > 0:
            candidate.return_on_collateral = candidate.premium_received / candidate.collateral_required
        if candidate.max_loss > 0:
            candidate.return_on_risk = candidate.premium_received / candidate.max_loss

        # Annualized return (for comparison purposes)
        if candidate.dte > 0:
            candidate.annualized_return = candidate.return_on_collateral * (365 / candidate.dte)

        # Probability metrics
        # Delta approximates probability of ITM, so 1 - |delta| = prob of OTM = prob of profit
        if put.delta is not None:
            candidate.prob_profit = 1 - abs(put.delta)
            candidate.prob_max_profit = candidate.prob_profit  # Max profit = full premium
        else:
            candidate.prob_profit = 0.5  # Default if no delta

        # Aggregate Greeks (single leg, but we're short so flip signs for some)
        # Short put = positive delta (bullish), positive theta (time decay helps)
        candidate.net_delta = abs(put.delta) if put.delta else 0  # Short put = positive delta
        candidate.net_gamma = -(put.gamma or 0)  # Gamma works against us
        candidate.net_theta = abs(put.theta) if put.theta else 0  # Theta helps us (positive)
        candidate.net_vega = -(put.vega or 0)  # Vega works against us if IV rises

        # Expected value calculation
        # EV = (Prob Win * Max Profit) - (Prob Lose * Expected Loss)
        # For simplicity, assume average loss on breach is 50% of max loss
        ev_win = candidate.premium_received * candidate.prob_profit
        avg_loss_on_breach = candidate.max_loss * 0.5  # Conservative estimate
        ev_loss = avg_loss_on_breach * (1 - candidate.prob_profit)
        candidate.expected_value = ev_win - ev_loss

        # Liquidity score
        candidate.liquidity_score = self._calculate_liquidity_score([put])

        # Overall score for ranking
        candidate.overall_score = self.score_candidate(candidate)

        return candidate


class CoveredCallStrategy(BaseStrategy):
    """
    Covered Call Strategy.

    Sell OTM calls against stock you already own.
    Profit if stock stays below strike at expiration.
    Max profit = premium + (strike - purchase price) if called away.
    Risk = stock price decline (offset by premium received).

    Note: This strategy assumes you already own the underlying stock.
    The collateral shown is the value of 100 shares (for reference).
    """

    @property
    def name(self) -> str:
        return "Covered Call"

    @property
    def strategy_type(self) -> str:
        return "covered_call"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    @property
    def requires_stock_ownership(self) -> bool:
        return True

    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """Find all covered call candidates from the options chain."""
        candidates = []

        for call in chain.calls:
            # Skip ITM calls (we want OTM for upside potential)
            if call.is_itm:
                continue

            # Skip if doesn't meet liquidity requirements
            if not self._passes_liquidity_check(call):
                continue

            # Skip if delta too high (too close to ATM)
            if call.delta is not None and abs(call.delta) > self.criteria.max_delta:
                continue

            # Create candidate
            candidate = self._create_candidate(call, chain, surface)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        call: OptionContract,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface]
    ) -> Optional[TradeCandidate]:
        """Create a covered call candidate."""
        # Use bid price for premium
        premium = call.bid * 100

        if premium <= 0:
            return None

        # For covered call, "collateral" is the stock value
        stock_value = chain.underlying_price * 100

        # Max profit if called away = premium + (strike - current price) * 100
        upside_if_called = (call.strike - chain.underlying_price) * 100
        max_profit = premium + max(0, upside_if_called)

        # Max loss = stock goes to zero - premium received
        max_loss = stock_value - premium

        # Breakeven = current stock price - premium per share
        breakeven = chain.underlying_price - call.bid

        candidate = TradeCandidate(
            strategy_name=self.name,
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            legs=[call],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            premium_received=premium,
            collateral_required=stock_value,  # Stock value as reference
            iv_at_entry=call.implied_volatility,
            iv_rank=chain.iv_rank
        )

        return self.calculate_metrics(candidate)

    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """Calculate trade metrics for covered call."""
        call = candidate.legs[0]

        # Return calculations
        # Return on stock value (the premium yield)
        if candidate.collateral_required > 0:
            candidate.return_on_collateral = candidate.premium_received / candidate.collateral_required

        # Return on risk is less meaningful here since risk is stock ownership
        # Use premium / potential downside to breakeven
        if candidate.max_loss > 0:
            candidate.return_on_risk = candidate.premium_received / candidate.max_loss

        # Annualized return
        if candidate.dte > 0:
            candidate.annualized_return = candidate.return_on_collateral * (365 / candidate.dte)

        # Probability of keeping shares (not getting called away)
        if call.delta is not None:
            candidate.prob_profit = 1 - abs(call.delta)  # Prob of expiring OTM
            # Note: With covered call, you profit either way (premium or premium + gains)
            # This prob is for keeping the shares
        else:
            candidate.prob_profit = 0.5

        # Greeks
        # Short call = negative delta (caps upside), positive theta
        candidate.net_delta = -(call.delta or 0)  # Reduces stock delta
        candidate.net_gamma = -(call.gamma or 0)
        candidate.net_theta = abs(call.theta) if call.theta else 0  # Positive (time decay helps)
        candidate.net_vega = -(call.vega or 0)

        # Expected value (simplified - premium is guaranteed, assignment varies)
        candidate.expected_value = candidate.premium_received  # Minimum is premium

        # Liquidity score
        candidate.liquidity_score = self._calculate_liquidity_score([call])

        # Overall score
        candidate.overall_score = self.score_candidate(candidate)

        return candidate
