"""
Backtesting module for options analyzer.
Provides historical strategy testing with hybrid data (yfinance + Black-Scholes synthesis).
"""

from .historical_data import HistoricalDataProvider, HistoricalDataConfig
from .trade_manager import TradeManager, TradeRecord, OpenPosition, ExitRules
from .backtester import OptionBacktester, BacktestConfig
from .performance import PerformanceTracker, PerformanceMetrics, BacktestResult

__all__ = [
    "HistoricalDataProvider",
    "HistoricalDataConfig",
    "TradeManager",
    "TradeRecord",
    "OpenPosition",
    "ExitRules",
    "OptionBacktester",
    "BacktestConfig",
    "PerformanceTracker",
    "PerformanceMetrics",
    "BacktestResult",
]
