"""
Greeks calculation module.
Uses py_vollib for Black-Scholes Greeks with vectorized operations where possible.

Enhanced features:
- American options pricing via Bjerksund-Stensland approximation
- Live risk-free rate from Treasury yields
- Dividend yield support for more accurate pricing
- Higher-order Greeks (Vanna, Charm, Vomma) for advanced risk management
"""

import numpy as np
from typing import List, Optional, Dict
from datetime import date
import logging
from scipy.stats import norm

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


# ============================================================================
# Bjerksund-Stensland American Options Pricing
# ============================================================================

def _phi(S: float, T: float, gamma: float, H: float, I: float,
         r: float, b: float, sigma: float) -> float:
    """Helper function for Bjerksund-Stensland model."""
    lambda_val = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2) * T
    d = -(np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
    kappa = 2 * b / sigma**2 + 2 * gamma - 1

    return (np.exp(lambda_val) * S**gamma *
            (norm.cdf(d) - (I / S)**kappa * norm.cdf(d - 2 * np.log(I / S) / (sigma * np.sqrt(T)))))


def bjerksund_stensland_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Bjerksund-Stensland (2002) approximation for American call options.

    This is more accurate than Black-Scholes for American options,
    especially for dividend-paying stocks where early exercise may be optimal.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        American call option price
    """
    if T <= 0:
        return max(0, S - K)

    b = r - q  # Cost of carry

    # If no dividends, American call = European call
    if q <= 0:
        return bs_price('c', S, K, T, r, sigma)

    try:
        # Trigger price for early exercise
        beta = (0.5 - b / sigma**2) + np.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / sigma**2)
        B_inf = beta / (beta - 1) * K
        B_0 = max(K, r / q * K)

        h_T = -(b * T + 2 * sigma * np.sqrt(T)) * B_0 / (B_inf - B_0)
        I = B_0 + (B_inf - B_0) * (1 - np.exp(h_T))

        if S >= I:
            return S - K

        alpha = (I - K) * I**(-beta)

        price = (alpha * S**beta
                 - alpha * _phi(S, T, beta, I, I, r, b, sigma)
                 + _phi(S, T, 1, I, I, r, b, sigma)
                 - _phi(S, T, 1, K, I, r, b, sigma)
                 - K * _phi(S, T, 0, I, I, r, b, sigma)
                 + K * _phi(S, T, 0, K, I, r, b, sigma))

        return max(price, S - K)  # Never less than intrinsic

    except Exception as e:
        logger.debug(f"Bjerksund-Stensland call failed: {e}, falling back to BS")
        return bs_price('c', S, K, T, r, sigma)


def bjerksund_stensland_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Bjerksund-Stensland (2002) approximation for American put options.

    Uses put-call transformation: P(S,K,T,r,q) = C(K,S,T,q,r)

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        American put option price
    """
    if T <= 0:
        return max(0, K - S)

    # Put-call transformation for American options
    return bjerksund_stensland_call(K, S, T, q, r, sigma)


def american_option_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Calculate American option price using Bjerksund-Stensland.

    Args:
        option_type: 'c' for call, 'p' for put
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        Option price
    """
    if option_type.lower() == 'c':
        return bjerksund_stensland_call(S, K, T, r, q, sigma)
    else:
        return bjerksund_stensland_put(S, K, T, r, q, sigma)


# ============================================================================
# Higher-Order Greeks
# ============================================================================

def calculate_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """Calculate d1 and d2 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return None, None

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def calculate_vanna(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Calculate Vanna: d(Delta)/d(Vol) = d(Vega)/d(S)

    Measures how delta changes with volatility. Important for:
    - Delta hedge effectiveness as vol changes
    - Understanding gamma/vega interaction
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0

    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    vanna = vega / S * (1 - d1 / (sigma * np.sqrt(T)))

    return vanna


def calculate_charm(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Calculate Charm: d(Delta)/d(Time) = -d(Theta)/d(S)

    Also called "delta decay". Important for:
    - Weekend risk (2+ days of charm over weekend)
    - Understanding how delta changes as expiration approaches
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, q, sigma)
    if d1 is None or T <= 0:
        return 0.0

    pdf_d1 = norm.pdf(d1)

    if option_type.lower() == 'c':
        charm = -q * np.exp(-q * T) * norm.cdf(d1) + np.exp(-q * T) * pdf_d1 * (
            2 * (r - q) * T - d2 * sigma * np.sqrt(T)
        ) / (2 * T * sigma * np.sqrt(T))
    else:
        charm = q * np.exp(-q * T) * norm.cdf(-d1) + np.exp(-q * T) * pdf_d1 * (
            2 * (r - q) * T - d2 * sigma * np.sqrt(T)
        ) / (2 * T * sigma * np.sqrt(T))

    return charm


def calculate_vomma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Calculate Vomma (Volga): d(Vega)/d(Vol)

    Measures vega convexity. Important for:
    - OTM options are most sensitive to vol-of-vol
    - Understanding volatility regime changes impact
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0

    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    vomma = vega * d1 * d2 / sigma

    return vomma


# ============================================================================
# Main Greeks Calculator
# ============================================================================

class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model.
    Supports both single contract and batch calculations.

    Enhanced features:
    - Live risk-free rate fetching
    - Dividend yield support
    - American options pricing (Bjerksund-Stensland)
    - Higher-order Greeks (Vanna, Charm, Vomma)
    """

    def __init__(self, config: AnalyzerConfig, use_live_rates: bool = True):
        self.config = config
        self._use_live_rates = use_live_rates

        # Market data cache for live rates
        self._market_data = None
        if use_live_rates:
            try:
                from data.fetcher import get_market_data_cache
                self._market_data = get_market_data_cache()
            except ImportError:
                logger.warning("Could not import market data cache, using config rates")

        # Dividend yield cache per symbol
        self._dividend_yields: Dict[str, float] = {}

    @property
    def risk_free_rate(self) -> float:
        """Get risk-free rate (live or from config)."""
        if self._market_data and self._use_live_rates:
            try:
                return self._market_data.get_risk_free_rate()
            except Exception as e:
                logger.debug(f"Could not get live rate: {e}")
        return self.config.risk_free_rate

    def get_dividend_yield(self, symbol: str) -> float:
        """Get dividend yield for a symbol."""
        if symbol in self._dividend_yields:
            return self._dividend_yields[symbol]

        div_yield = 0.0
        if self._market_data and self._use_live_rates:
            try:
                div_yield = self._market_data.get_dividend_yield(symbol)
            except Exception as e:
                logger.debug(f"Could not get dividend yield for {symbol}: {e}")

        self._dividend_yields[symbol] = div_yield
        return div_yield

    def calculate_greeks(
        self,
        contract: OptionContract,
        use_american: bool = True,
        include_higher_order: bool = False
    ) -> OptionContract:
        """
        Calculate all Greeks for a single option contract.
        Updates the contract in place and returns it.

        Args:
            contract: Option contract to calculate Greeks for
            use_american: Use American options pricing (Bjerksund-Stensland)
            include_higher_order: Calculate Vanna, Charm, Vomma
        """
        try:
            # Prepare inputs
            flag = 'c' if contract.option_type == OptionType.CALL else 'p'
            S = contract.underlying_price
            K = contract.strike
            t = max(contract.dte / 365.0, 0.001)  # Time in years, minimum to avoid div by zero
            r = self.risk_free_rate
            q = self.get_dividend_yield(contract.underlying_symbol)
            sigma = contract.implied_volatility

            if sigma <= 0:
                # Try to calculate IV from price if not available
                sigma = self._estimate_iv(contract, r, q)
                if sigma <= 0:
                    logger.debug(f"Could not calculate IV for {contract.symbol}")
                    return contract
                contract.implied_volatility = sigma

            # Calculate first-order Greeks using Black-Scholes
            # (Greeks are similar for American and European for short-dated options)
            contract.delta = bs_delta(flag, S, K, t, r, sigma)
            contract.gamma = bs_gamma(flag, S, K, t, r, sigma)
            contract.theta = bs_theta(flag, S, K, t, r, sigma) * 365  # Annualize then daily
            contract.vega = bs_vega(flag, S, K, t, r, sigma)
            contract.rho = bs_rho(flag, S, K, t, r, sigma)

            # Store higher-order Greeks as additional attributes if requested
            if include_higher_order:
                contract._vanna = calculate_vanna(S, K, t, r, q, sigma)
                contract._charm = calculate_charm(flag, S, K, t, r, q, sigma)
                contract._vomma = calculate_vomma(S, K, t, r, q, sigma)

        except Exception as e:
            logger.debug(f"Error calculating Greeks for {contract.symbol}: {e}")

        return contract

    def calculate_greeks_batch(
        self,
        contracts: List[OptionContract],
        use_american: bool = True
    ) -> List[OptionContract]:
        """
        Calculate Greeks for multiple contracts.
        Uses vectorized operations if available for performance.
        """
        if not contracts:
            return contracts

        if VECTORIZED and len(contracts) > 10:
            return self._calculate_greeks_vectorized(contracts)
        else:
            return [self.calculate_greeks(c, use_american=use_american) for c in contracts]

    def _calculate_greeks_vectorized(self, contracts: List[OptionContract]) -> List[OptionContract]:
        """Vectorized Greeks calculation using py_vollib_vectorized."""
        try:
            # Prepare arrays - py_vollib_vectorized expects specific formats
            n = len(contracts)
            r = self.risk_free_rate

            flags = np.array(['c' if c.option_type == OptionType.CALL else 'p' for c in contracts])
            S = np.array([c.underlying_price for c in contracts], dtype=np.float64)
            K = np.array([c.strike for c in contracts], dtype=np.float64)
            t = np.array([max(c.dte / 365.0, 0.001) for c in contracts], dtype=np.float64)
            r_arr = np.full(n, r, dtype=np.float64)
            sigma = np.array([c.implied_volatility if c.implied_volatility > 0 else 0.3 for c in contracts], dtype=np.float64)

            # Calculate all Greeks at once using vectorized functions
            deltas = vec_delta(flags, S, K, t, r_arr, sigma, return_as='numpy')
            gammas = vec_gamma(flags, S, K, t, r_arr, sigma, return_as='numpy')
            thetas = vec_theta(flags, S, K, t, r_arr, sigma, return_as='numpy') * 365
            vegas = vec_vega(flags, S, K, t, r_arr, sigma, return_as='numpy')
            rhos = vec_rho(flags, S, K, t, r_arr, sigma, return_as='numpy')

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

    def _estimate_iv(
        self,
        contract: OptionContract,
        r: Optional[float] = None,
        q: Optional[float] = None
    ) -> float:
        """
        Estimate implied volatility from option price using Newton-Raphson.
        """
        try:
            flag = 'c' if contract.option_type == OptionType.CALL else 'p'
            price = contract.mid_price if contract.mid_price > 0 else contract.last_price

            if price <= 0:
                return 0.0

            if r is None:
                r = self.risk_free_rate

            iv = bs_iv(
                price,
                contract.underlying_price,
                contract.strike,
                max(contract.dte / 365.0, 0.001),
                r,
                flag
            )
            return iv if iv > 0 else 0.0

        except Exception:
            return 0.0

    def enrich_chain(self, chain: OptionsChain, use_american: bool = True) -> OptionsChain:
        """Calculate Greeks for all contracts in an options chain."""
        chain.calls = self.calculate_greeks_batch(chain.calls, use_american=use_american)
        chain.puts = self.calculate_greeks_batch(chain.puts, use_american=use_american)
        return chain

    def calculate_theoretical_price(
        self,
        contract: OptionContract,
        use_american: bool = True
    ) -> float:
        """Calculate theoretical option price."""
        flag = 'c' if contract.option_type == OptionType.CALL else 'p'
        S = contract.underlying_price
        K = contract.strike
        t = max(contract.dte / 365.0, 0.001)
        r = self.risk_free_rate
        q = self.get_dividend_yield(contract.underlying_symbol)
        sigma = contract.implied_volatility

        if use_american and q > 0:
            return american_option_price(flag, S, K, t, r, q, sigma)
        else:
            return bs_price(flag, S, K, t, r, sigma)


def calculate_theoretical_price(
    option_type: OptionType,
    underlying_price: float,
    strike: float,
    dte: int,
    risk_free_rate: float,
    iv: float,
    dividend_yield: float = 0.0,
    use_american: bool = True
) -> float:
    """Calculate theoretical option price using Black-Scholes or Bjerksund-Stensland."""
    try:
        flag = 'c' if option_type == OptionType.CALL else 'p'
        t = max(dte / 365.0, 0.001)

        if use_american and dividend_yield > 0:
            return american_option_price(flag, underlying_price, strike, t,
                                         risk_free_rate, dividend_yield, iv)
        else:
            return bs_price(flag, underlying_price, strike, t, risk_free_rate, iv)
    except Exception:
        return 0.0
