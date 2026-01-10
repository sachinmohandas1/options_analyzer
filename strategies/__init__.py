from strategies.base import BaseStrategy
from strategies.secured_premium import CashSecuredPutStrategy, CoveredCallStrategy
from strategies.credit_spreads import (
    PutCreditSpreadStrategy,
    CallCreditSpreadStrategy,
)

__all__ = [
    'BaseStrategy',
    'CashSecuredPutStrategy',
    'CoveredCallStrategy',
    'PutCreditSpreadStrategy',
    'CallCreditSpreadStrategy',
]

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'cash_secured_put': CashSecuredPutStrategy,
    'covered_call': CoveredCallStrategy,
    'put_credit_spread': PutCreditSpreadStrategy,
    'call_credit_spread': CallCreditSpreadStrategy,
}


def get_strategy_class(strategy_type: str):
    """Get strategy class by type identifier."""
    return STRATEGY_REGISTRY.get(strategy_type)


def get_all_strategies():
    """Get all available strategy classes."""
    return list(STRATEGY_REGISTRY.values())
