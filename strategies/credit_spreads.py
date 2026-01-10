"""
Credit Spread Strategies.
- Put Credit Spreads (Bull Put Spreads)
- Call Credit Spreads (Bear Call Spreads)
- Iron Condors

These are defined-risk strategies that combine long and short options.
"""

from typing import List, Optional, Tuple
from datetime import date
import itertools

from core.models import (
    OptionsChain, TradeCandidate, OptionContract,
    VolatilitySurface, OptionType
)
from core.config import AnalyzerConfig
from strategies.base import BaseStrategy


class PutCreditSpreadStrategy(BaseStrategy):
    """
    Put Credit Spread (Bull Put Spread) Strategy.

    Sell OTM put, buy further OTM put.
    - Max profit = net credit received
    - Max loss = spread width - credit received
    - Breakeven = short strike - credit

    Bullish to neutral strategy with defined risk.
    """

    @property
    def name(self) -> str:
        return "Put Credit Spread"

    @property
    def strategy_type(self) -> str:
        return "put_credit_spread"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """Find all put credit spread candidates."""
        candidates = []

        # Get OTM puts sorted by strike (descending - closest to ATM first)
        otm_puts = [p for p in chain.puts if p.is_otm and p.bid > 0]
        otm_puts.sort(key=lambda x: x.strike, reverse=True)

        # Group by expiration
        puts_by_exp = {}
        for put in otm_puts:
            if put.expiration not in puts_by_exp:
                puts_by_exp[put.expiration] = []
            puts_by_exp[put.expiration].append(put)

        # Find valid spreads for each expiration
        for exp, puts in puts_by_exp.items():
            if len(puts) < 2:
                continue

            # Try all combinations of short/long puts
            for i, short_put in enumerate(puts):
                # Skip if short put doesn't meet criteria
                if short_put.delta and abs(short_put.delta) > self.criteria.max_delta:
                    continue

                for long_put in puts[i+1:]:  # Long put must have lower strike
                    spread_width = short_put.strike - long_put.strike

                    # Check spread width constraints
                    if spread_width < self.criteria.min_spread_width:
                        continue
                    if spread_width > self.criteria.max_spread_width:
                        continue

                    # Create candidate
                    candidate = self._create_candidate(
                        short_put, long_put, chain, surface
                    )
                    if candidate:
                        candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        short_put: OptionContract,
        long_put: OptionContract,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface]
    ) -> Optional[TradeCandidate]:
        """Create a put credit spread candidate."""
        # Net credit = short put bid - long put ask
        net_credit = (short_put.bid - long_put.ask) * 100

        if net_credit <= 0:
            return None

        # Spread width
        spread_width = (short_put.strike - long_put.strike) * 100

        # Max loss = spread width - credit
        max_loss = spread_width - net_credit

        # Collateral required = max loss (defined risk spread)
        collateral = max_loss

        # Breakeven = short strike - credit per share
        breakeven = short_put.strike - (net_credit / 100)

        candidate = TradeCandidate(
            strategy_name=self.name,
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            legs=[short_put, long_put],
            max_profit=net_credit,
            max_loss=max_loss,
            breakeven=breakeven,
            premium_received=net_credit,
            collateral_required=collateral,
            iv_at_entry=(short_put.implied_volatility + long_put.implied_volatility) / 2,
            iv_rank=chain.iv_rank
        )

        return self.calculate_metrics(candidate)

    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """Calculate trade metrics for put credit spread."""
        short_put, long_put = candidate.legs[0], candidate.legs[1]

        # Return calculations
        if candidate.collateral_required > 0:
            candidate.return_on_collateral = candidate.premium_received / candidate.collateral_required
        if candidate.max_loss > 0:
            candidate.return_on_risk = candidate.premium_received / candidate.max_loss

        if candidate.dte > 0:
            candidate.annualized_return = candidate.return_on_collateral * (365 / candidate.dte)

        # Probability of profit
        # Approximated using short put delta
        if short_put.delta is not None:
            candidate.prob_profit = 1 - abs(short_put.delta)
        else:
            candidate.prob_profit = 0.5

        # Probability of max profit (both puts expire worthless)
        if long_put.delta is not None:
            candidate.prob_max_profit = 1 - abs(long_put.delta)
        else:
            candidate.prob_max_profit = candidate.prob_profit * 0.9

        # Net Greeks
        candidate.net_delta = (abs(short_put.delta or 0) - abs(long_put.delta or 0))
        candidate.net_gamma = -((short_put.gamma or 0) - (long_put.gamma or 0))
        candidate.net_theta = abs(short_put.theta or 0) - abs(long_put.theta or 0)
        candidate.net_vega = -((short_put.vega or 0) - (long_put.vega or 0))

        # Expected value
        ev_win = candidate.premium_received * candidate.prob_profit
        ev_loss = (candidate.max_loss * 0.5) * (1 - candidate.prob_profit)
        candidate.expected_value = ev_win - ev_loss

        # Liquidity score
        candidate.liquidity_score = self._calculate_liquidity_score([short_put, long_put])

        candidate.overall_score = self.score_candidate(candidate)

        return candidate


class CallCreditSpreadStrategy(BaseStrategy):
    """
    Call Credit Spread (Bear Call Spread) Strategy.

    Sell OTM call, buy further OTM call.
    - Max profit = net credit received
    - Max loss = spread width - credit received
    - Breakeven = short strike + credit

    Bearish to neutral strategy with defined risk.
    """

    @property
    def name(self) -> str:
        return "Call Credit Spread"

    @property
    def strategy_type(self) -> str:
        return "call_credit_spread"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """Find all call credit spread candidates."""
        candidates = []

        # Get OTM calls sorted by strike (ascending - closest to ATM first)
        otm_calls = [c for c in chain.calls if c.is_otm and c.bid > 0]
        otm_calls.sort(key=lambda x: x.strike)

        # Group by expiration
        calls_by_exp = {}
        for call in otm_calls:
            if call.expiration not in calls_by_exp:
                calls_by_exp[call.expiration] = []
            calls_by_exp[call.expiration].append(call)

        # Find valid spreads
        for exp, calls in calls_by_exp.items():
            if len(calls) < 2:
                continue

            for i, short_call in enumerate(calls):
                # Skip if short call doesn't meet delta criteria
                if short_call.delta and abs(short_call.delta) > self.criteria.max_delta:
                    continue

                for long_call in calls[i+1:]:  # Long call must have higher strike
                    spread_width = long_call.strike - short_call.strike

                    if spread_width < self.criteria.min_spread_width:
                        continue
                    if spread_width > self.criteria.max_spread_width:
                        continue

                    candidate = self._create_candidate(
                        short_call, long_call, chain, surface
                    )
                    if candidate:
                        candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        short_call: OptionContract,
        long_call: OptionContract,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface]
    ) -> Optional[TradeCandidate]:
        """Create a call credit spread candidate."""
        # Net credit = short call bid - long call ask
        net_credit = (short_call.bid - long_call.ask) * 100

        if net_credit <= 0:
            return None

        # Spread width
        spread_width = (long_call.strike - short_call.strike) * 100

        # Max loss = spread width - credit
        max_loss = spread_width - net_credit

        # Collateral = max loss
        collateral = max_loss

        # Breakeven = short strike + credit per share
        breakeven = short_call.strike + (net_credit / 100)

        candidate = TradeCandidate(
            strategy_name=self.name,
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            legs=[short_call, long_call],
            max_profit=net_credit,
            max_loss=max_loss,
            breakeven=breakeven,
            premium_received=net_credit,
            collateral_required=collateral,
            iv_at_entry=(short_call.implied_volatility + long_call.implied_volatility) / 2,
            iv_rank=chain.iv_rank
        )

        return self.calculate_metrics(candidate)

    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """Calculate trade metrics for call credit spread."""
        short_call, long_call = candidate.legs[0], candidate.legs[1]

        # Return calculations
        if candidate.collateral_required > 0:
            candidate.return_on_collateral = candidate.premium_received / candidate.collateral_required
        if candidate.max_loss > 0:
            candidate.return_on_risk = candidate.premium_received / candidate.max_loss

        if candidate.dte > 0:
            candidate.annualized_return = candidate.return_on_collateral * (365 / candidate.dte)

        # Probability of profit
        if short_call.delta is not None:
            candidate.prob_profit = 1 - abs(short_call.delta)
        else:
            candidate.prob_profit = 0.5

        # Probability of max profit
        if long_call.delta is not None:
            candidate.prob_max_profit = 1 - abs(long_call.delta)
        else:
            candidate.prob_max_profit = candidate.prob_profit * 0.9

        # Net Greeks
        candidate.net_delta = -((short_call.delta or 0) - (long_call.delta or 0))
        candidate.net_gamma = -((short_call.gamma or 0) - (long_call.gamma or 0))
        candidate.net_theta = abs(short_call.theta or 0) - abs(long_call.theta or 0)
        candidate.net_vega = -((short_call.vega or 0) - (long_call.vega or 0))

        # Expected value
        ev_win = candidate.premium_received * candidate.prob_profit
        ev_loss = (candidate.max_loss * 0.5) * (1 - candidate.prob_profit)
        candidate.expected_value = ev_win - ev_loss

        candidate.liquidity_score = self._calculate_liquidity_score([short_call, long_call])
        candidate.overall_score = self.score_candidate(candidate)

        return candidate


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor Strategy.

    Combines a put credit spread and call credit spread.
    - Sell OTM put, buy further OTM put (bull put spread)
    - Sell OTM call, buy further OTM call (bear call spread)

    Neutral strategy that profits from low volatility.
    Max profit when underlying stays between short strikes.
    """

    @property
    def name(self) -> str:
        return "Iron Condor"

    @property
    def strategy_type(self) -> str:
        return "iron_condor"

    @property
    def is_credit_strategy(self) -> bool:
        return True

    def find_candidates(
        self,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface] = None
    ) -> List[TradeCandidate]:
        """Find all iron condor candidates."""
        candidates = []

        # Use the spread strategies to find legs
        put_spread_strategy = PutCreditSpreadStrategy(self.config)
        call_spread_strategy = CallCreditSpreadStrategy(self.config)

        put_spreads = put_spread_strategy.find_candidates(chain, surface)
        call_spreads = call_spread_strategy.find_candidates(chain, surface)

        # Group by expiration
        put_spreads_by_exp = {}
        for ps in put_spreads:
            exp = ps.expiration
            if exp not in put_spreads_by_exp:
                put_spreads_by_exp[exp] = []
            put_spreads_by_exp[exp].append(ps)

        call_spreads_by_exp = {}
        for cs in call_spreads:
            exp = cs.expiration
            if exp not in call_spreads_by_exp:
                call_spreads_by_exp[exp] = []
            call_spreads_by_exp[exp].append(cs)

        # Combine spreads with same expiration
        for exp in set(put_spreads_by_exp.keys()) & set(call_spreads_by_exp.keys()):
            for put_spread in put_spreads_by_exp[exp]:
                for call_spread in call_spreads_by_exp[exp]:
                    # Ensure short strikes don't overlap
                    short_put_strike = put_spread.legs[0].strike
                    short_call_strike = call_spread.legs[0].strike

                    if short_put_strike >= short_call_strike:
                        continue  # Invalid - short strikes overlap

                    candidate = self._create_candidate(
                        put_spread, call_spread, chain, surface
                    )
                    if candidate:
                        candidates.append(candidate)

        return candidates

    def _create_candidate(
        self,
        put_spread: TradeCandidate,
        call_spread: TradeCandidate,
        chain: OptionsChain,
        surface: Optional[VolatilitySurface]
    ) -> Optional[TradeCandidate]:
        """Create an iron condor candidate from two spreads."""
        # Total credit = sum of both spreads
        total_credit = put_spread.premium_received + call_spread.premium_received

        if total_credit <= 0:
            return None

        # Max loss = max of either spread's max loss
        # (only one side can be breached at expiration)
        max_loss = max(put_spread.max_loss, call_spread.max_loss)

        # Collateral = max loss (one spread's collateral)
        collateral = max_loss

        # All four legs
        all_legs = put_spread.legs + call_spread.legs

        # Breakevens
        breakeven_lower = put_spread.breakeven
        breakeven_upper = call_spread.breakeven

        candidate = TradeCandidate(
            strategy_name=self.name,
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            legs=all_legs,
            max_profit=total_credit,
            max_loss=max_loss,
            breakeven=breakeven_lower,
            breakeven_upper=breakeven_upper,
            premium_received=total_credit,
            collateral_required=collateral,
            iv_at_entry=(put_spread.iv_at_entry + call_spread.iv_at_entry) / 2,
            iv_rank=chain.iv_rank
        )

        return self.calculate_metrics(candidate)

    def calculate_metrics(self, candidate: TradeCandidate) -> TradeCandidate:
        """Calculate trade metrics for iron condor."""
        # Return calculations
        if candidate.collateral_required > 0:
            candidate.return_on_collateral = candidate.premium_received / candidate.collateral_required
        if candidate.max_loss > 0:
            candidate.return_on_risk = candidate.premium_received / candidate.max_loss

        if candidate.dte > 0:
            candidate.annualized_return = candidate.return_on_collateral * (365 / candidate.dte)

        # Probability of profit (between both short strikes)
        # Use short put and short call deltas
        short_put = candidate.legs[0]  # First leg is short put
        short_call = candidate.legs[2]  # Third leg is short call

        if short_put.delta and short_call.delta:
            prob_above_put = 1 - abs(short_put.delta)
            prob_below_call = 1 - abs(short_call.delta)
            # Probability of staying between both = prob_above_put * prob_below_call (approx)
            candidate.prob_profit = prob_above_put * prob_below_call / max(prob_above_put, prob_below_call)
        else:
            candidate.prob_profit = 0.5

        # Probability of max profit (all options expire worthless)
        long_put = candidate.legs[1]
        long_call = candidate.legs[3]

        if long_put.delta and long_call.delta:
            candidate.prob_max_profit = (1 - abs(long_put.delta)) * (1 - abs(long_call.delta)) / 2
        else:
            candidate.prob_max_profit = candidate.prob_profit * 0.7

        # Net Greeks (sum of all legs)
        candidate.net_delta = sum(
            (leg.delta or 0) * (-1 if i % 2 == 0 else 1)
            for i, leg in enumerate(candidate.legs)
        )
        candidate.net_gamma = sum(
            (leg.gamma or 0) * (-1 if i % 2 == 0 else 1)
            for i, leg in enumerate(candidate.legs)
        )
        candidate.net_theta = sum(
            abs(leg.theta or 0) * (1 if i % 2 == 0 else -1)
            for i, leg in enumerate(candidate.legs)
        )
        candidate.net_vega = sum(
            (leg.vega or 0) * (-1 if i % 2 == 0 else 1)
            for i, leg in enumerate(candidate.legs)
        )

        # Expected value
        ev_win = candidate.premium_received * candidate.prob_profit
        ev_loss = (candidate.max_loss * 0.5) * (1 - candidate.prob_profit)
        candidate.expected_value = ev_win - ev_loss

        candidate.liquidity_score = self._calculate_liquidity_score(candidate.legs)
        candidate.overall_score = self.score_candidate(candidate)

        return candidate
