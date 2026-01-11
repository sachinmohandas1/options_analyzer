"""
Main backtest orchestrator with daily simulation loop.
Coordinates data provider, strategies, trade manager, and performance tracking.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
import time

from .historical_data import HistoricalDataProvider, HistoricalDataConfig
from .trade_manager import TradeManager, ExitRules, TradeRecord
from .performance import PerformanceTracker, PerformanceMetrics, BacktestResult

from core.config import AnalyzerConfig, TradeCriteria, StrategyType, CapitalConfig
from core.models import OptionsChain, TradeCandidate

from strategies.secured_premium import CashSecuredPutStrategy
from strategies.credit_spreads import PutCreditSpreadStrategy, CallCreditSpreadStrategy


@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""
    start_date: date
    end_date: date
    symbols: List[str]
    initial_capital: float = 13000.0

    # Strategy selection
    strategy_types: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.CASH_SECURED_PUT,
        StrategyType.PUT_CREDIT_SPREAD,
        StrategyType.CALL_CREDIT_SPREAD,
    ])

    # Trade criteria (reuses existing config)
    trade_criteria: TradeCriteria = field(default_factory=TradeCriteria)

    # Exit rules
    exit_rules: ExitRules = field(default_factory=ExitRules)

    # Position limits
    max_positions: int = 5
    max_positions_per_symbol: int = 1

    # Entry timing
    entry_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # All weekdays

    # Data settings
    use_cache: bool = True
    risk_free_rate: float = 0.05


class OptionBacktester:
    """
    Main backtest engine with daily simulation loop.

    Coordinates:
    - Historical data fetching and synthesis
    - Strategy candidate generation
    - Trade entry and exit management
    - Performance tracking
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Initialize data provider
        data_config = HistoricalDataConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols,
            use_cache=config.use_cache,
            risk_free_rate=config.risk_free_rate,
            max_dte=config.trade_criteria.max_dte,
        )
        self.data_provider = HistoricalDataProvider(data_config)

        # Initialize trade manager
        self.trade_manager = TradeManager(config.exit_rules)

        # Initialize performance tracker
        self.performance = PerformanceTracker(config.initial_capital)

        # Initialize strategies
        self.strategies = self._init_strategies()

        # Tracking
        self.current_capital = config.initial_capital
        self.errors: List[Dict] = []

    def _init_strategies(self) -> Dict[StrategyType, object]:
        """Initialize strategy instances."""
        strategies = {}

        # Create analyzer config for strategies
        analyzer_config = AnalyzerConfig(
            trade_criteria=self.config.trade_criteria,
            capital=CapitalConfig(total_capital=self.config.initial_capital),
            risk_free_rate=self.config.risk_free_rate,
        )

        if StrategyType.CASH_SECURED_PUT in self.config.strategy_types:
            strategies[StrategyType.CASH_SECURED_PUT] = CashSecuredPutStrategy(analyzer_config)

        if StrategyType.PUT_CREDIT_SPREAD in self.config.strategy_types:
            strategies[StrategyType.PUT_CREDIT_SPREAD] = PutCreditSpreadStrategy(analyzer_config)

        if StrategyType.CALL_CREDIT_SPREAD in self.config.strategy_types:
            strategies[StrategyType.CALL_CREDIT_SPREAD] = CallCreditSpreadStrategy(analyzer_config)

        return strategies

    def run(self, verbose: bool = True) -> BacktestResult:
        """
        Run the backtest simulation.

        Iterates through each trading day:
        1. Update open positions (mark-to-market)
        2. Check exit conditions
        3. Generate new candidates if capacity available
        4. Execute entries based on scoring
        5. Track daily P&L and portfolio value
        """
        start_time = time.time()

        if verbose:
            print(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")
            print(f"Symbols: {', '.join(self.config.symbols)}")
            print(f"Initial capital: ${self.config.initial_capital:,.2f}")
            print(f"Strategies: {[s.value for s in self.config.strategy_types]}")
            print()

        # Preload data
        if verbose:
            print("Loading historical data...")
        load_results = self.data_provider.preload_symbols()
        successful = sum(1 for v in load_results.values() if v)
        if verbose:
            print(f"Loaded data for {successful}/{len(self.config.symbols)} symbols")
            print()

        # Get trading days
        trading_days = self.data_provider.get_trading_days()
        trading_days = [d for d in trading_days
                        if self.config.start_date <= d <= self.config.end_date]

        if verbose:
            print(f"Simulating {len(trading_days)} trading days...")
            print()

        # Main simulation loop
        for i, current_date in enumerate(trading_days):
            self._simulate_day(current_date)

            # Progress update
            if verbose and (i + 1) % 50 == 0:
                pnl = self.current_capital - self.config.initial_capital
                open_pos = self.trade_manager.get_open_position_count()
                closed = len(self.trade_manager.closed_trades)
                print(f"Day {i+1}/{len(trading_days)}: {current_date} | "
                      f"Capital: ${self.current_capital:,.2f} | "
                      f"P&L: ${pnl:+,.2f} | "
                      f"Open: {open_pos} | Closed: {closed}")

        # Calculate final metrics
        duration = time.time() - start_time

        if verbose:
            print()
            print(f"Backtest complete in {duration:.1f}s")
            print()

        return self._generate_result(duration)

    def _simulate_day(self, current_date: date) -> None:
        """Process a single trading day."""
        # 1. Fetch current options chains for all symbols with open positions
        current_chains = self._fetch_chains_for_date(current_date)

        # 2. Update open positions with current prices
        self.trade_manager.update_positions(current_date, current_chains)

        # 3. Check exit conditions and close triggered positions
        closed_trades = self.trade_manager.check_exits(current_date)

        # 4. Update capital from closed trades
        for trade in closed_trades:
            if trade.realized_pnl is not None:
                self.current_capital += trade.realized_pnl

        # 5. Check if we should enter new trades today
        if self._should_enter_today(current_date):
            self._generate_and_enter_trades(current_date, current_chains)

        # 6. Record daily portfolio value
        unrealized = self.trade_manager.get_unrealized_pnl()
        portfolio_value = self.current_capital + unrealized
        collateral_used = self.trade_manager.get_total_collateral_used()

        self.performance.record_day(current_date, portfolio_value, collateral_used)

    def _fetch_chains_for_date(self, current_date: date) -> Dict[str, OptionsChain]:
        """Fetch options chains for all relevant symbols."""
        chains = {}

        # Get symbols with open positions
        symbols_needed = set(self.config.symbols)
        for position in self.trade_manager.open_positions:
            symbols_needed.add(position.trade_record.symbol)

        for symbol in symbols_needed:
            try:
                chain = self.data_provider.get_options_chain(symbol, current_date)
                if chain:
                    chains[symbol] = chain
            except Exception as e:
                self.errors.append({
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "error": str(e)
                })

        return chains

    def _should_enter_today(self, current_date: date) -> bool:
        """Check if today is a valid entry day."""
        # Check day of week (0=Monday, 4=Friday)
        if current_date.weekday() not in self.config.entry_days:
            return False

        # Check position capacity
        if self.trade_manager.get_open_position_count() >= self.config.max_positions:
            return False

        # Check available capital
        collateral_used = self.trade_manager.get_total_collateral_used()
        available = self.current_capital - collateral_used
        if available < self.config.initial_capital * 0.1:  # Need at least 10% available
            return False

        return True

    def _generate_and_enter_trades(
        self,
        current_date: date,
        current_chains: Dict[str, OptionsChain]
    ) -> None:
        """Generate candidates and enter best trades."""
        all_candidates = []

        for symbol in self.config.symbols:
            # Check position limit per symbol
            if self.trade_manager.get_position_count_for_symbol(symbol) >= self.config.max_positions_per_symbol:
                continue

            if symbol not in current_chains:
                continue

            chain = current_chains[symbol]

            # Run each strategy
            for strategy_type, strategy in self.strategies.items():
                try:
                    # Generate candidates
                    candidates = strategy.find_candidates(chain, surface=None)

                    # Filter by criteria
                    filtered = strategy.filter_by_criteria(candidates)

                    # Tag with strategy type for later reference
                    for c in filtered:
                        c._strategy_type = strategy_type

                    all_candidates.extend(filtered)

                except Exception as e:
                    self.errors.append({
                        "date": current_date.isoformat(),
                        "symbol": symbol,
                        "strategy": strategy_type.value,
                        "error": str(e)
                    })

        if not all_candidates:
            return

        # Sort by score
        all_candidates.sort(key=lambda c: c.overall_score, reverse=True)

        # Filter by available capital
        collateral_used = self.trade_manager.get_total_collateral_used()
        available_capital = self.current_capital - collateral_used

        # Enter trades up to position limit
        positions_available = self.config.max_positions - self.trade_manager.get_open_position_count()

        for candidate in all_candidates:
            if positions_available <= 0:
                break

            # Check capital
            if candidate.collateral_required > available_capital:
                continue

            # Check symbol limit
            if self.trade_manager.get_position_count_for_symbol(candidate.underlying_symbol) >= self.config.max_positions_per_symbol:
                continue

            # Enter trade
            strategy_type = getattr(candidate, '_strategy_type', StrategyType.CASH_SECURED_PUT)
            self.trade_manager.open_trade(candidate, current_date, strategy_type)

            # Update available capital
            available_capital -= candidate.collateral_required
            positions_available -= 1

    def _generate_result(self, duration: float) -> BacktestResult:
        """Generate final backtest result."""
        trades = self.trade_manager.closed_trades.copy()

        # Calculate overall metrics
        metrics = self.performance.calculate_metrics(trades)

        # Calculate breakdown metrics
        metrics_by_strategy = self.performance.get_metrics_by_strategy(trades)
        metrics_by_symbol = self.performance.get_metrics_by_symbol(trades)
        metrics_by_month = self.performance.get_metrics_by_month(trades)

        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            symbols=self.config.symbols,
            initial_capital=self.config.initial_capital,
            final_capital=self.current_capital,
            metrics=metrics,
            trades=trades,
            equity_curve=self.performance.daily_values.copy(),
            daily_pnl=self.performance.daily_pnl.copy(),
            metrics_by_strategy=metrics_by_strategy,
            metrics_by_symbol=metrics_by_symbol,
            metrics_by_month=metrics_by_month,
            errors=self.errors,
            duration_seconds=duration,
        )


def run_backtest(
    symbols: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 13000.0,
    strategies: Optional[List[StrategyType]] = None,
    profit_target: float = 0.5,
    stop_loss: float = 2.0,
    max_positions: int = 5,
    verbose: bool = True
) -> BacktestResult:
    """
    Convenience function to run a backtest with common parameters.

    Args:
        symbols: List of underlying symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        strategies: List of strategy types (defaults to all premium selling)
        profit_target: Exit at this fraction of max profit (0.5 = 50%)
        stop_loss: Exit at this multiple of premium received (2.0 = 2x)
        max_positions: Maximum concurrent positions
        verbose: Print progress updates

    Returns:
        BacktestResult with complete metrics and trade history
    """
    if strategies is None:
        strategies = [
            StrategyType.CASH_SECURED_PUT,
            StrategyType.PUT_CREDIT_SPREAD,
            StrategyType.CALL_CREDIT_SPREAD,
        ]

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=initial_capital,
        strategy_types=strategies,
        exit_rules=ExitRules(
            profit_target_pct=profit_target,
            stop_loss_pct=stop_loss,
        ),
        max_positions=max_positions,
    )

    backtester = OptionBacktester(config)
    return backtester.run(verbose=verbose)
