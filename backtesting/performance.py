"""
Performance tracking and metrics calculation for backtesting.
Calculates win rate, Sharpe ratio, drawdown, and other key metrics.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .trade_manager import TradeRecord, ExitReason
from core.config import StrategyType


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    total_premium_collected: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss
    expectancy: float = 0.0     # avg pnl per trade

    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    avg_trade_return_pct: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    drawdown_start: Optional[date] = None
    drawdown_end: Optional[date] = None
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0  # annualized return / max drawdown

    # Trade duration
    avg_days_held: float = 0.0
    min_days_held: int = 0
    max_days_held: int = 0

    # Exit analysis
    exits_by_reason: Dict[str, int] = field(default_factory=dict)
    pnl_by_exit_reason: Dict[str, float] = field(default_factory=dict)

    # Capital efficiency
    avg_capital_usage: float = 0.0
    max_capital_usage: float = 0.0
    capital_turnover: float = 0.0  # total premium / avg capital


@dataclass
class BacktestResult:
    """Complete backtest output container."""
    # Configuration
    start_date: date
    end_date: date
    symbols: List[str]
    initial_capital: float
    final_capital: float

    # Overall metrics
    metrics: PerformanceMetrics

    # Trade details
    trades: List[TradeRecord] = field(default_factory=list)

    # Time series data
    equity_curve: List[Tuple[date, float]] = field(default_factory=list)
    daily_pnl: List[Tuple[date, float]] = field(default_factory=list)

    # Breakdown by dimensions
    metrics_by_strategy: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    metrics_by_symbol: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    metrics_by_month: Dict[str, PerformanceMetrics] = field(default_factory=dict)

    # Errors during backtest
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "symbols": self.symbols,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return_pct": self.metrics.total_return_pct,
            "total_trades": self.metrics.total_trades,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "max_drawdown_pct": self.metrics.max_drawdown_pct,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "generated_at": self.generated_at.isoformat(),
        }


class PerformanceTracker:
    """
    Tracks portfolio value and calculates performance metrics.

    Records daily equity values and computes comprehensive
    statistics from trade history.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_values: List[Tuple[date, float]] = []
        self.daily_pnl: List[Tuple[date, float]] = []
        self.capital_usage: List[Tuple[date, float]] = []

    def record_day(
        self,
        d: date,
        portfolio_value: float,
        collateral_used: float = 0.0
    ) -> None:
        """Record end-of-day portfolio value."""
        self.daily_values.append((d, portfolio_value))

        # Calculate daily P&L
        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-2][1]
            daily_change = portfolio_value - prev_value
        else:
            daily_change = portfolio_value - self.initial_capital

        self.daily_pnl.append((d, daily_change))
        self.capital_usage.append((d, collateral_used))
        self.current_capital = portfolio_value

    def calculate_metrics(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Calculate all performance metrics from trade history."""
        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        # Basic trade stats
        metrics.total_trades = len(trades)
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]

        if not pnls:
            return metrics

        metrics.winning_trades = sum(1 for p in pnls if p > 0)
        metrics.losing_trades = sum(1 for p in pnls if p < 0)
        metrics.breakeven_trades = sum(1 for p in pnls if p == 0)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # P&L metrics
        metrics.total_pnl = sum(pnls)
        metrics.total_premium_collected = sum(t.premium_received for t in trades)

        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        metrics.gross_profit = sum(winning_pnls)
        metrics.gross_loss = abs(sum(losing_pnls))
        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        metrics.largest_win = max(winning_pnls) if winning_pnls else 0
        metrics.largest_loss = min(losing_pnls) if losing_pnls else 0
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        metrics.expectancy = np.mean(pnls)

        # Return metrics
        metrics.total_return_pct = (metrics.total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        # Annualize return
        if self.daily_values:
            days = (self.daily_values[-1][0] - self.daily_values[0][0]).days
            if days > 0:
                years = days / 365.0
                metrics.annualized_return_pct = ((1 + metrics.total_return_pct / 100) ** (1 / years) - 1) * 100

        # Trade duration
        days_held = [t.days_held for t in trades if t.days_held is not None]
        if days_held:
            metrics.avg_days_held = np.mean(days_held)
            metrics.min_days_held = min(days_held)
            metrics.max_days_held = max(days_held)

        # Exit reason analysis
        for trade in trades:
            if trade.exit_reason:
                reason = trade.exit_reason.value
                metrics.exits_by_reason[reason] = metrics.exits_by_reason.get(reason, 0) + 1
                if trade.realized_pnl:
                    metrics.pnl_by_exit_reason[reason] = metrics.pnl_by_exit_reason.get(reason, 0) + trade.realized_pnl

        # Drawdown and risk metrics
        self._calculate_drawdown(metrics)
        self._calculate_risk_ratios(metrics)

        # Capital efficiency
        if self.capital_usage:
            usage_values = [u[1] for u in self.capital_usage]
            metrics.avg_capital_usage = np.mean(usage_values)
            metrics.max_capital_usage = max(usage_values)
            if metrics.avg_capital_usage > 0:
                metrics.capital_turnover = metrics.total_premium_collected / metrics.avg_capital_usage

        return metrics

    def _calculate_drawdown(self, metrics: PerformanceMetrics) -> None:
        """Calculate maximum drawdown from equity curve."""
        if not self.daily_values:
            return

        values = [v[1] for v in self.daily_values]
        dates = [v[0] for v in self.daily_values]

        peak = values[0]
        max_dd = 0
        max_dd_pct = 0
        dd_start_idx = 0
        dd_end_idx = 0
        current_dd_start = 0

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                current_dd_start = i
            else:
                dd = peak - value
                dd_pct = (dd / peak) * 100 if peak > 0 else 0

                if dd_pct > max_dd_pct:
                    max_dd = dd
                    max_dd_pct = dd_pct
                    dd_start_idx = current_dd_start
                    dd_end_idx = i

        metrics.max_drawdown_amount = max_dd
        metrics.max_drawdown_pct = max_dd_pct
        if dd_start_idx < len(dates) and dd_end_idx < len(dates):
            metrics.drawdown_start = dates[dd_start_idx]
            metrics.drawdown_end = dates[dd_end_idx]

    def _calculate_risk_ratios(self, metrics: PerformanceMetrics) -> None:
        """Calculate Sharpe, Sortino, and Calmar ratios."""
        if not self.daily_pnl or len(self.daily_pnl) < 2:
            return

        daily_returns = [p[1] / self.initial_capital for p in self.daily_pnl]

        # Sharpe Ratio (annualized, assuming 0 risk-free rate for simplicity)
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            if std_return > 0:
                metrics.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

        # Sortino Ratio (only downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns, ddof=1)
            if downside_std > 0:
                metrics.sortino_ratio = (np.mean(daily_returns) / downside_std) * np.sqrt(252)

        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

    def calculate_metrics_for_subset(
        self,
        trades: List[TradeRecord],
        filter_func=None
    ) -> PerformanceMetrics:
        """Calculate metrics for a subset of trades."""
        if filter_func:
            trades = [t for t in trades if filter_func(t)]
        return self.calculate_metrics(trades)

    def get_metrics_by_strategy(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, PerformanceMetrics]:
        """Get metrics broken down by strategy type."""
        by_strategy = {}
        strategies = set(t.strategy_type for t in trades)

        for strategy in strategies:
            strategy_trades = [t for t in trades if t.strategy_type == strategy]
            by_strategy[strategy.value] = self.calculate_metrics(strategy_trades)

        return by_strategy

    def get_metrics_by_symbol(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, PerformanceMetrics]:
        """Get metrics broken down by underlying symbol."""
        by_symbol = {}
        symbols = set(t.symbol for t in trades)

        for symbol in symbols:
            symbol_trades = [t for t in trades if t.symbol == symbol]
            by_symbol[symbol] = self.calculate_metrics(symbol_trades)

        return by_symbol

    def get_metrics_by_month(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, PerformanceMetrics]:
        """Get metrics broken down by month."""
        by_month = {}

        for trade in trades:
            if trade.exit_date:
                month_key = trade.exit_date.strftime("%Y-%m")
                if month_key not in by_month:
                    by_month[month_key] = []
                by_month[month_key].append(trade)

        return {k: self.calculate_metrics(v) for k, v in by_month.items()}

    def generate_equity_curve(self) -> pd.DataFrame:
        """Generate equity curve as DataFrame."""
        if not self.daily_values:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_values, columns=["date", "portfolio_value"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # Add derived columns
        df["daily_return"] = df["portfolio_value"].pct_change()
        df["cumulative_return"] = (df["portfolio_value"] / self.initial_capital - 1) * 100

        # Add drawdown
        df["peak"] = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - df["peak"]) / df["peak"] * 100

        return df

    def generate_monthly_returns(self) -> pd.DataFrame:
        """Generate monthly returns table."""
        equity_df = self.generate_equity_curve()
        if equity_df.empty:
            return pd.DataFrame()

        # Resample to monthly
        monthly = equity_df["portfolio_value"].resample("M").last()
        monthly_returns = monthly.pct_change() * 100

        # Pivot to year x month format
        df = monthly_returns.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        return pivot

    def reset(self) -> None:
        """Reset tracker for new backtest."""
        self.current_capital = self.initial_capital
        self.daily_values.clear()
        self.daily_pnl.clear()
        self.capital_usage.clear()
