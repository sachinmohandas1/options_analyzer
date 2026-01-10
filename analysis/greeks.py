"""
Greeks calculation module.
Uses py_vollib for Black-Scholes Greeks with vectorized operations where possible.
"""

import numpy as np
from typing import List, Optional
from datetime import date
import logging

from py_vollib.black_scholes import black_scholes as bs_price
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
from py_vollib.black_scholes.greeks.analytical import (
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
    rho as bs_rho
)

# Vectorized mode disabled - py_vollib_vectorized has a different API that requires
# pandas DataFrames and doesn't directly accept numpy arrays. Sequential calculation
# is fast enough for typical use cases (a few hundred contracts per symbol).
VECTORIZED = False

from core.models import OptionContract, OptionsChain, OptionType
from core.config import AnalyzerConfig

logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model.
    Supports both single contract and batch calculations.
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate

    def calculate_greeks(self, contract: OptionContract) -> OptionContract:
        """
        Calculate all Greeks for a single option contract.
        Updates the contract in place and returns it.
        """
        try:
            # Prepare inputs
            flag = 'c' if contract.option_type == OptionType.CALL else 'p'
            S = contract.underlying_price
            K = contract.strike
            t = max(contract.dte / 365.0, 0.001)  # Time in years, minimum to avoid div by zero
            r = self.risk_free_rate
            sigma = contract.implied_volatility

            if sigma <= 0:
                # Try to calculate IV from price if not available
                sigma = self._estimate_iv(contract)
                if sigma <= 0:
                    logger.debug(f"Could not calculate IV for {contract.symbol}")
                    return contract
                contract.implied_volatility = sigma

            # Calculate Greeks
            contract.delta = bs_delta(flag, S, K, t, r, sigma)
            contract.gamma = bs_gamma(flag, S, K, t, r, sigma)
            contract.theta = bs_theta(flag, S, K, t, r, sigma) * 365  # Annualize then daily
            contract.vega = bs_vega(flag, S, K, t, r, sigma)
            contract.rho = bs_rho(flag, S, K, t, r, sigma)

        except Exception as e:
            logger.debug(f"Error calculating Greeks for {contract.symbol}: {e}")

        return contract

    def calculate_greeks_batch(self, contracts: List[OptionContract]) -> List[OptionContract]:
        """
        Calculate Greeks for multiple contracts.
        Uses vectorized operations if available for performance.
        """
        if not contracts:
            return contracts

        if VECTORIZED and len(contracts) > 10:
            return self._calculate_greeks_vectorized(contracts)
        else:
            return [self.calculate_greeks(c) for c in contracts]

    def _calculate_greeks_vectorized(self, contracts: List[OptionContract]) -> List[OptionContract]:
        """Vectorized Greeks calculation using py_vollib_vectorized."""
        try:
            # Prepare arrays - py_vollib_vectorized expects specific formats
            n = len(contracts)
            flags = np.array(['c' if c.option_type == OptionType.CALL else 'p' for c in contracts])
            S = np.array([c.underlying_price for c in contracts], dtype=np.float64)
            K = np.array([c.strike for c in contracts], dtype=np.float64)
            t = np.array([max(c.dte / 365.0, 0.001) for c in contracts], dtype=np.float64)
            r = np.full(n, self.risk_free_rate, dtype=np.float64)
            sigma = np.array([c.implied_volatility if c.implied_volatility > 0 else 0.3 for c in contracts], dtype=np.float64)

            # Calculate all Greeks at once using vectorized functions
            deltas = vec_delta(flags, S, K, t, r, sigma, return_as='numpy')
            gammas = vec_gamma(flags, S, K, t, r, sigma, return_as='numpy')
            thetas = vec_theta(flags, S, K, t, r, sigma, return_as='numpy') * 365
            vegas = vec_vega(flags, S, K, t, r, sigma, return_as='numpy')
            rhos = vec_rho(flags, S, K, t, r, sigma, return_as='numpy')

            # Update contracts
            for i, contract in enumerate(contracts):
                contract.delta = float(deltas[i]) if not np.isnan(deltas[i]) else None
                contract.gamma = float(gammas[i]) if not np.isnan(gammas[i]) else None
                contract.theta = float(thetas[i]) if not np.isnan(thetas[i]) else None
                contract.vega = float(vegas[i]) if not np.isnan(vegas[i]) else None
                contract.rho = float(rhos[i]) if not np.isnan(rhos[i]) else None

        except Exception as e:
            logger.warning(f"Vectorized calculation failed, falling back to sequential: {e}")
            return [self.calculate_greeks(c) for c in contracts]

        return contracts

    def _estimate_iv(self, contract: OptionContract) -> float:
        """
        Estimate implied volatility from option price using Newton-Raphson.
        """
        try:
            flag = 'c' if contract.option_type == OptionType.CALL else 'p'
            price = contract.mid_price if contract.mid_price > 0 else contract.last_price

            if price <= 0:
                return 0.0

            iv = bs_iv(
                price,
                contract.underlying_price,
                contract.strike,
                max(contract.dte / 365.0, 0.001),
                self.risk_free_rate,
                flag
            )
            return iv if iv > 0 else 0.0

        except Exception:
            return 0.0

    def enrich_chain(self, chain: OptionsChain) -> OptionsChain:
        """Calculate Greeks for all contracts in an options chain."""
        chain.calls = self.calculate_greeks_batch(chain.calls)
        chain.puts = self.calculate_greeks_batch(chain.puts)
        return chain


def calculate_theoretical_price(
    option_type: OptionType,
    underlying_price: float,
    strike: float,
    dte: int,
    risk_free_rate: float,
    iv: float
) -> float:
    """Calculate theoretical option price using Black-Scholes."""
    try:
        flag = 'c' if option_type == OptionType.CALL else 'p'
        t = max(dte / 365.0, 0.001)
        return bs_price(flag, underlying_price, strike, t, risk_free_rate, iv)
    except Exception:
        return 0.0
