"""
Trade lifecycle management for backtesting.
Handles trade entry, position tracking, and exit logic.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid

from core.models import TradeCandidate, OptionsChain, OptionContract
from core.config import StrategyType


class ExitReason(Enum):
    """Reasons for closing a position."""
    EXPIRATION = "expiration"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    MAX_DAYS = "max_days_held"
    MANUAL = "manual"


@dataclass
class ExitRules:
    """Configurable rules for when to exit positions."""
    profit_target_pct: float = 0.50  # Close at 50% of max profit
    stop_loss_pct: float = 2.0       # Close at 2x credit received (1x actual loss)
    exit_dte: int = 0                # Close at or before this DTE (0 = hold to expiration)
    max_days_held: Optional[int] = None  # Maximum days to hold


@dataclass
class TradeRecord:
    """Complete record of a single trade from entry to exit."""
    id: str
    strategy_type: StrategyType
    strategy_name: str
    symbol: str
    entry_date: date
    entry_price: float  # Underlying price at entry
    premium_received: float
    collateral_required: float
    max_profit: float
    max_loss: float
    entry_candidate: TradeCandidate

    # Leg details for reference
    leg_strikes: List[float] = field(default_factory=list)
    leg_types: List[str] = field(default_factory=list)  # "call" or "put"
    expiration: Optional[date] = None
    entry_dte: int = 0

    # Exit details (filled when closed)
    exit_date: Optional[date] = None
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None  # Underlying price at exit
    close_cost: float = 0.0  # Cost to close position (negative if profit)
    realized_pnl: Optional[float] = None
    days_held: int = 0

    # Running metrics
    high_water_pnl: float = 0.0
    low_water_pnl: float = 0.0

    def __post_init__(self):
        if self.entry_candidate:
            self.expiration = self.entry_candidate.expiration
            self.entry_dte = self.entry_candidate.dte
            for leg in self.entry_candidate.legs:
                self.leg_strikes.append(leg.strike)
                self.leg_types.append(leg.option_type.value)


@dataclass
class OpenPosition:
    """Tracks an open position during simulation."""
    trade_record: TradeRecord
    current_value: float = 0.0  # Current mark-to-market value (cost to close)
    unrealized_pnl: float = 0.0
    current_dte: int = 0
    days_in_trade: int = 0

    def update_dte(self, current_date: date):
        """Update DTE based on current date."""
        if self.trade_record.expiration:
            self.current_dte = (self.trade_record.expiration - current_date).days
        self.days_in_trade = (current_date - self.trade_record.entry_date).days


class TradeManager:
    """
    Manages trade lifecycle: entry, hold, exit.

    Tracks open positions, evaluates exit conditions,
    and maintains trade history for performance analysis.
    """

    def __init__(self, exit_rules: ExitRules):
        self.exit_rules = exit_rules
        self.open_positions: List[OpenPosition] = []
        self.closed_trades: List[TradeRecord] = []
        self.position_count_by_symbol: Dict[str, int] = {}

    def open_trade(
        self,
        candidate: TradeCandidate,
        entry_date: date,
        strategy_type: StrategyType
    ) -> OpenPosition:
        """Record a new trade entry."""
        trade_id = str(uuid.uuid4())[:8]

        trade_record = TradeRecord(
            id=trade_id,
            strategy_type=strategy_type,
            strategy_name=candidate.strategy_name,
            symbol=candidate.underlying_symbol,
            entry_date=entry_date,
            entry_price=candidate.underlying_price,
            premium_received=candidate.premium_received,
            collateral_required=candidate.collateral_required,
            max_profit=candidate.max_profit,
            max_loss=candidate.max_loss,
            entry_candidate=candidate,
        )

        position = OpenPosition(
            trade_record=trade_record,
            current_value=0.0,  # At entry, no cost to close yet
            unrealized_pnl=0.0,
            current_dte=candidate.dte,
            days_in_trade=0,
        )

        self.open_positions.append(position)

        # Track position count by symbol
        symbol = candidate.underlying_symbol
        self.position_count_by_symbol[symbol] = self.position_count_by_symbol.get(symbol, 0) + 1

        return position

    def update_positions(
        self,
        current_date: date,
        current_chains: Dict[str, OptionsChain]
    ) -> None:
        """Mark-to-market all open positions."""
        for position in self.open_positions:
            position.update_dte(current_date)
            symbol = position.trade_record.symbol

            if symbol in current_chains:
                chain = current_chains[symbol]
                position.current_value = self._calculate_position_value(position, chain)
                position.unrealized_pnl = position.trade_record.premium_received - position.current_value

                # Track high/low water marks
                position.trade_record.high_water_pnl = max(
                    position.trade_record.high_water_pnl,
                    position.unrealized_pnl
                )
                position.trade_record.low_water_pnl = min(
                    position.trade_record.low_water_pnl,
                    position.unrealized_pnl
                )

    def _calculate_position_value(
        self,
        position: OpenPosition,
        chain: OptionsChain
    ) -> float:
        """
        Calculate current cost to close position.

        For short premium positions, this is the cost to buy back the options.
        """
        total_cost = 0.0
        entry_candidate = position.trade_record.entry_candidate

        for leg in entry_candidate.legs:
            # Find matching contract in current chain
            current_contract = self._find_matching_contract(leg, chain)

            if current_contract:
                # For short positions, cost to close is the ask price
                # For long positions (spread protection), we'd sell at bid
                if self._is_short_leg(leg, entry_candidate):
                    # Short leg: buy back at ask
                    total_cost += current_contract.ask * 100
                else:
                    # Long leg: sell at bid (reduces cost)
                    total_cost -= current_contract.bid * 100

        return total_cost

    def _find_matching_contract(
        self,
        leg: OptionContract,
        chain: OptionsChain
    ) -> Optional[OptionContract]:
        """Find a contract in the chain matching the leg's strike/expiration/type."""
        contracts = chain.calls if leg.option_type.value == "call" else chain.puts

        for contract in contracts:
            if (contract.strike == leg.strike and
                contract.expiration == leg.expiration):
                return contract

        # Fallback: find closest strike with same expiration
        same_exp = [c for c in contracts if c.expiration == leg.expiration]
        if same_exp:
            return min(same_exp, key=lambda c: abs(c.strike - leg.strike))

        return None

    def _is_short_leg(self, leg: OptionContract, candidate: TradeCandidate) -> bool:
        """Determine if a leg is short based on strategy."""
        strategy = candidate.strategy_name.lower()

        if "put_spread" in strategy or "put credit" in strategy.lower():
            # Short put spread: short the higher strike put
            put_strikes = [l.strike for l in candidate.legs if l.option_type.value == "put"]
            if put_strikes and leg.option_type.value == "put":
                return leg.strike == max(put_strikes)

        elif "call_spread" in strategy or "call credit" in strategy.lower():
            # Short call spread: short the lower strike call
            call_strikes = [l.strike for l in candidate.legs if l.option_type.value == "call"]
            if call_strikes and leg.option_type.value == "call":
                return leg.strike == min(call_strikes)

        elif "csp" in strategy or "cash secured" in strategy.lower():
            # CSP: the put is short
            return leg.option_type.value == "put"

        # Default: first leg is typically short
        return leg == candidate.legs[0]

    def check_exits(self, current_date: date) -> List[TradeRecord]:
        """Evaluate exit rules and close triggered positions."""
        closed_this_cycle = []
        positions_to_close = []

        for position in self.open_positions:
            should_exit, reason = self._should_exit(position, current_date)
            if should_exit:
                positions_to_close.append((position, reason))

        # Close positions
        for position, reason in positions_to_close:
            trade_record = self._close_position(position, current_date, reason)
            closed_this_cycle.append(trade_record)

        return closed_this_cycle

    def _should_exit(
        self,
        position: OpenPosition,
        current_date: date
    ) -> Tuple[bool, Optional[ExitReason]]:
        """Evaluate whether a position should be exited."""
        rules = self.exit_rules
        record = position.trade_record

        # 1. Expiration
        if position.current_dte <= rules.exit_dte:
            return True, ExitReason.EXPIRATION

        # 2. Profit target
        if record.max_profit > 0:
            profit_pct = position.unrealized_pnl / record.max_profit
            if profit_pct >= rules.profit_target_pct:
                return True, ExitReason.PROFIT_TARGET

        # 3. Stop loss
        if record.premium_received > 0:
            loss_ratio = -position.unrealized_pnl / record.premium_received
            if loss_ratio >= rules.stop_loss_pct:
                return True, ExitReason.STOP_LOSS

        # 4. Max days held
        if rules.max_days_held and position.days_in_trade >= rules.max_days_held:
            return True, ExitReason.MAX_DAYS

        return False, None

    def _close_position(
        self,
        position: OpenPosition,
        exit_date: date,
        reason: ExitReason
    ) -> TradeRecord:
        """Close a position and record the exit."""
        record = position.trade_record

        # Calculate realized P&L
        if reason == ExitReason.EXPIRATION and position.current_dte <= 0:
            # At expiration, calculate based on ITM/OTM
            realized_pnl = self._calculate_expiration_pnl(position)
        else:
            # Early exit: P&L is premium - cost to close
            realized_pnl = position.unrealized_pnl

        record.exit_date = exit_date
        record.exit_reason = reason
        record.exit_price = position.trade_record.entry_candidate.underlying_price  # Will be updated by caller
        record.close_cost = position.current_value
        record.realized_pnl = realized_pnl
        record.days_held = position.days_in_trade

        # Remove from open positions
        self.open_positions.remove(position)
        self.closed_trades.append(record)

        # Update position count
        symbol = record.symbol
        self.position_count_by_symbol[symbol] = max(0, self.position_count_by_symbol.get(symbol, 1) - 1)

        return record

    def _calculate_expiration_pnl(self, position: OpenPosition) -> float:
        """Calculate P&L at expiration based on where underlying finished."""
        # Simplified: use max profit if OTM, max loss if ITM
        # In reality, would need final underlying price
        record = position.trade_record

        # If unrealized P&L is positive, likely expired worthless (max profit)
        if position.unrealized_pnl >= 0:
            return record.max_profit
        else:
            # Clamp to max loss
            return max(-record.max_loss, position.unrealized_pnl)

    def get_open_position_count(self) -> int:
        """Get number of currently open positions."""
        return len(self.open_positions)

    def get_position_count_for_symbol(self, symbol: str) -> int:
        """Get number of positions for a specific symbol."""
        return self.position_count_by_symbol.get(symbol, 0)

    def get_total_collateral_used(self) -> float:
        """Get total collateral tied up in open positions."""
        return sum(p.trade_record.collateral_required for p in self.open_positions)

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions."""
        return sum(p.unrealized_pnl for p in self.open_positions)

    def reset(self) -> None:
        """Reset manager state for new backtest."""
        self.open_positions.clear()
        self.closed_trades.clear()
        self.position_count_by_symbol.clear()
