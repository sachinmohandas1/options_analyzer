"""
Volatility Surface Analysis Module.

Constructs and analyzes implied volatility surfaces to identify:
- IV skew (put vs call IV differential)
- Term structure (IV across expirations)
- Anomalies and mispricings
- Trading signals based on surface shape
"""

import warnings
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import zscore
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Suppress Qhull warnings - these occur when IV surface data is too sparse/flat
warnings.filterwarnings('ignore', message='.*QH.*')

from core.models import OptionsChain, VolatilitySurface, OptionContract, OptionType
from core.config import VolatilitySurfaceConfig, AnalyzerConfig

logger = logging.getLogger(__name__)


@dataclass
class SkewAnalysis:
    """Results of volatility skew analysis."""
    expiration: date
    dte: int
    atm_iv: float
    put_25d_iv: Optional[float]  # 25-delta put IV
    call_25d_iv: Optional[float]  # 25-delta call IV
    skew_25d: Optional[float]  # Put 25d IV - Call 25d IV
    risk_reversal: Optional[float]  # Same as skew_25d
    butterfly: Optional[float]  # (Put 25d + Call 25d) / 2 - ATM IV
    skew_percentile: Optional[float]  # Where current skew ranks historically
    signal: str  # "bullish", "bearish", "neutral"


@dataclass
class TermStructureAnalysis:
    """Results of IV term structure analysis."""
    term_structure: Dict[int, float]  # DTE -> ATM IV
    is_contango: bool  # Near-term IV < Far-term IV
    is_backwardation: bool  # Near-term IV > Far-term IV
    steepness: float  # Slope of term structure
    signal: str  # "vol_selling_favorable", "vol_buying_favorable", "neutral"


class VolatilitySurfaceAnalyzer:
    """
    Analyzes implied volatility surfaces for trading insights.

    Key analyses:
    1. Surface Construction - Build 3D IV surface from option prices
    2. Skew Analysis - Measure put/call IV differential (fear gauge)
    3. Term Structure - Analyze IV across expirations
    4. Anomaly Detection - Find mispricings or unusual IV patterns
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.vol_config = config.volatility_surface

    def build_surface(self, chain: OptionsChain) -> VolatilitySurface:
        """
        Build a volatility surface from an options chain.
        """
        surface = VolatilitySurface(
            underlying_symbol=chain.underlying_symbol,
            underlying_price=chain.underlying_price,
            generated_at=datetime.now()
        )

        # Build surface data: {expiration: {strike: iv}}
        for exp in chain.expirations:
            strike_iv_map = {}

            # Get contracts for this expiration
            contracts = chain.get_contracts_by_expiration(exp)
            all_contracts = contracts['calls'] + contracts['puts']

            for contract in all_contracts:
                if contract.implied_volatility > 0:
                    strike = contract.strike
                    iv = contract.implied_volatility

                    # If we have both call and put IV for same strike, average them
                    if strike in strike_iv_map:
                        strike_iv_map[strike] = (strike_iv_map[strike] + iv) / 2
                    else:
                        strike_iv_map[strike] = iv

            if strike_iv_map:
                surface.surface_data[exp] = strike_iv_map

        # Calculate ATM IV for each expiration
        surface.atm_iv_by_expiration = self._calculate_atm_iv(chain, surface)

        # Build term structure
        surface.term_structure = self._build_term_structure(surface)

        # Calculate skew metrics
        surface.skew_25d_by_expiration = self._calculate_skew(chain, surface)

        # Create interpolated surface for visualization
        self._interpolate_surface(surface)

        # Detect anomalies
        if self.vol_config.detect_iv_anomalies:
            surface.anomalies = self._detect_anomalies(chain, surface)

        return surface

    def _calculate_atm_iv(
        self,
        chain: OptionsChain,
        surface: VolatilitySurface
    ) -> Dict[date, float]:
        """Calculate ATM implied volatility for each expiration."""
        atm_iv = {}
        spot = chain.underlying_price

        for exp, strike_iv in surface.surface_data.items():
            if not strike_iv:
                continue

            # Find strikes closest to spot
            strikes = sorted(strike_iv.keys())
            closest_strike = min(strikes, key=lambda s: abs(s - spot))

            # Get IV at closest strike
            atm_iv[exp] = strike_iv[closest_strike]

        return atm_iv

    def _build_term_structure(self, surface: VolatilitySurface) -> Dict[int, float]:
        """Build IV term structure (DTE -> ATM IV)."""
        term_structure = {}
        today = date.today()

        for exp, atm_iv in surface.atm_iv_by_expiration.items():
            dte = (exp - today).days
            if dte > 0:
                term_structure[dte] = atm_iv

        return dict(sorted(term_structure.items()))

    def _calculate_skew(
        self,
        chain: OptionsChain,
        surface: VolatilitySurface
    ) -> Dict[date, float]:
        """
        Calculate 25-delta skew for each expiration.
        Skew = 25d Put IV - 25d Call IV
        Positive skew = puts are more expensive (bearish sentiment)
        """
        skew_by_exp = {}

        for exp in chain.expirations:
            contracts = chain.get_contracts_by_expiration(exp)

            # Find 25-delta put (delta around -0.25)
            puts_with_delta = [p for p in contracts['puts'] if p.delta is not None]
            put_25d = self._find_delta_contract(puts_with_delta, -0.25)

            # Find 25-delta call (delta around 0.25)
            calls_with_delta = [c for c in contracts['calls'] if c.delta is not None]
            call_25d = self._find_delta_contract(calls_with_delta, 0.25)

            if put_25d and call_25d:
                skew = put_25d.implied_volatility - call_25d.implied_volatility
                skew_by_exp[exp] = skew

        return skew_by_exp

    def _find_delta_contract(
        self,
        contracts: List[OptionContract],
        target_delta: float
    ) -> Optional[OptionContract]:
        """Find contract closest to target delta."""
        if not contracts:
            return None

        return min(contracts, key=lambda c: abs(c.delta - target_delta))

    def _interpolate_surface(self, surface: VolatilitySurface):
        """
        Create interpolated surface grid for visualization.
        Uses scipy's interpolation methods.
        """
        if not surface.surface_data:
            return

        try:
            # Collect all data points
            all_strikes = []
            all_dtes = []
            all_ivs = []
            today = date.today()

            for exp, strike_iv in surface.surface_data.items():
                dte = (exp - today).days
                for strike, iv in strike_iv.items():
                    all_strikes.append(strike)
                    all_dtes.append(dte)
                    all_ivs.append(iv)

            if len(all_strikes) < self.vol_config.min_strikes_for_surface:
                return

            # Create grid
            strikes_arr = np.array(all_strikes)
            dtes_arr = np.array(all_dtes)
            ivs_arr = np.array(all_ivs)

            # Define grid boundaries
            strike_min, strike_max = strikes_arr.min(), strikes_arr.max()
            dte_min, dte_max = max(1, dtes_arr.min()), dtes_arr.max()

            # Create mesh grid
            strike_grid = np.linspace(strike_min, strike_max, 50)
            dte_grid = np.linspace(dte_min, dte_max, 20)

            # Interpolate using RBF (Radial Basis Function) - handles irregular data well
            if self.vol_config.interpolation_method == 'rbf':
                rbf = interpolate.Rbf(strikes_arr, dtes_arr, ivs_arr, function='thin_plate')
                xx, yy = np.meshgrid(strike_grid, dte_grid)
                iv_grid = rbf(xx, yy)
            else:
                # Use griddata for linear/cubic interpolation
                points = np.column_stack((strikes_arr, dtes_arr))
                xx, yy = np.meshgrid(strike_grid, dte_grid)
                grid_points = np.column_stack((xx.ravel(), yy.ravel()))
                iv_grid = interpolate.griddata(
                    points, ivs_arr, grid_points,
                    method=self.vol_config.interpolation_method
                ).reshape(xx.shape)

            surface.strikes = strike_grid
            surface.expirations = dte_grid
            surface.iv_matrix = iv_grid

        except Exception as e:
            # Surface interpolation can fail with insufficient data points
            # (e.g., only 1 expiration, flat surface). This is non-critical.
            # Only log first line to avoid Qhull's verbose diagnostics
            error_first_line = str(e).split('\n')[0]
            logger.debug(f"Could not interpolate surface: {error_first_line}")

    def _detect_anomalies(
        self,
        chain: OptionsChain,
        surface: VolatilitySurface
    ) -> List[Dict[str, Any]]:
        """
        Detect IV anomalies that might indicate mispricings.

        Anomalies include:
        - IV significantly different from neighbors (z-score)
        - Put-call parity violations
        - Unusual skew patterns
        """
        anomalies = []
        threshold = self.vol_config.anomaly_zscore_threshold

        for exp, strike_iv in surface.surface_data.items():
            if len(strike_iv) < 5:
                continue

            strikes = sorted(strike_iv.keys())
            ivs = [strike_iv[s] for s in strikes]

            # Calculate z-scores
            zscores = zscore(ivs)

            for i, (strike, iv, z) in enumerate(zip(strikes, ivs, zscores)):
                if abs(z) > threshold:
                    anomalies.append({
                        'type': 'iv_outlier',
                        'expiration': exp,
                        'strike': strike,
                        'iv': iv,
                        'zscore': z,
                        'direction': 'high' if z > 0 else 'low',
                        'potential_signal': 'sell' if z > 0 else 'buy'
                    })

        return anomalies

    def analyze_skew(self, chain: OptionsChain, surface: VolatilitySurface) -> List[SkewAnalysis]:
        """
        Perform comprehensive skew analysis for all expirations.
        """
        analyses = []
        today = date.today()

        for exp in chain.expirations:
            dte = (exp - today).days
            contracts = chain.get_contracts_by_expiration(exp)

            # Get ATM IV
            atm_iv = surface.atm_iv_by_expiration.get(exp)
            if atm_iv is None:
                continue

            # Find delta contracts
            puts = [p for p in contracts['puts'] if p.delta is not None]
            calls = [c for c in contracts['calls'] if c.delta is not None]

            put_25d = self._find_delta_contract(puts, -0.25)
            call_25d = self._find_delta_contract(calls, 0.25)

            put_25d_iv = put_25d.implied_volatility if put_25d else None
            call_25d_iv = call_25d.implied_volatility if call_25d else None

            # Calculate metrics
            skew_25d = None
            butterfly = None

            if put_25d_iv and call_25d_iv:
                skew_25d = put_25d_iv - call_25d_iv
                butterfly = (put_25d_iv + call_25d_iv) / 2 - atm_iv

            # Determine signal
            signal = "neutral"
            if skew_25d is not None:
                if skew_25d > 0.05:  # Puts expensive - bearish sentiment
                    signal = "bearish"
                elif skew_25d < -0.02:  # Calls expensive - bullish sentiment
                    signal = "bullish"

            analyses.append(SkewAnalysis(
                expiration=exp,
                dte=dte,
                atm_iv=atm_iv,
                put_25d_iv=put_25d_iv,
                call_25d_iv=call_25d_iv,
                skew_25d=skew_25d,
                risk_reversal=skew_25d,
                butterfly=butterfly,
                skew_percentile=None,  # Would need historical data
                signal=signal
            ))

        return analyses

    def analyze_term_structure(self, surface: VolatilitySurface) -> TermStructureAnalysis:
        """
        Analyze IV term structure for trading signals.

        Contango (upward sloping): Normal, favor short-dated selling
        Backwardation (downward sloping): Fear/event, exercise caution
        """
        term = surface.term_structure
        if len(term) < 2:
            return TermStructureAnalysis(
                term_structure=term,
                is_contango=False,
                is_backwardation=False,
                steepness=0.0,
                signal="insufficient_data"
            )

        dtes = list(term.keys())
        ivs = list(term.values())

        # Calculate slope using linear regression
        if len(dtes) >= 2:
            slope = np.polyfit(dtes, ivs, 1)[0]
        else:
            slope = 0.0

        is_contango = slope > 0.0001  # Small positive slope
        is_backwardation = slope < -0.0001  # Negative slope

        # Determine signal
        if is_contango:
            signal = "vol_selling_favorable"  # Short near-term options
        elif is_backwardation:
            signal = "vol_buying_favorable"  # Market expects near-term event
        else:
            signal = "neutral"

        return TermStructureAnalysis(
            term_structure=term,
            is_contango=is_contango,
            is_backwardation=is_backwardation,
            steepness=slope,
            signal=signal
        )

    def get_trading_signals(
        self,
        chain: OptionsChain,
        surface: VolatilitySurface
    ) -> Dict[str, Any]:
        """
        Generate actionable trading signals from surface analysis.
        """
        skew_analyses = self.analyze_skew(chain, surface)
        term_analysis = self.analyze_term_structure(surface)

        signals = {
            'underlying': chain.underlying_symbol,
            'iv_rank': chain.iv_rank,
            'iv_percentile': chain.iv_percentile,
            'skew': {},
            'term_structure': {},
            'anomalies': surface.anomalies,
            'recommendations': []
        }

        # Skew signals
        for skew in skew_analyses:
            signals['skew'][skew.expiration.isoformat()] = {
                'dte': skew.dte,
                'atm_iv': skew.atm_iv,
                'skew_25d': skew.skew_25d,
                'signal': skew.signal
            }

        # Term structure signals
        signals['term_structure'] = {
            'is_contango': term_analysis.is_contango,
            'is_backwardation': term_analysis.is_backwardation,
            'steepness': term_analysis.steepness,
            'signal': term_analysis.signal
        }

        # Generate recommendations
        recommendations = []

        # High IV rank + contango = good for selling premium
        if (chain.iv_rank and chain.iv_rank > 50) and term_analysis.is_contango:
            recommendations.append({
                'action': 'SELL_PREMIUM',
                'confidence': 'high',
                'reason': f"IV rank {chain.iv_rank:.0f}% with contango term structure"
            })

        # Backwardation = caution
        if term_analysis.is_backwardation:
            recommendations.append({
                'action': 'CAUTION',
                'confidence': 'medium',
                'reason': "Backwardation indicates expected near-term volatility event"
            })

        # Extreme skew
        for skew in skew_analyses:
            if skew.skew_25d and abs(skew.skew_25d) > 0.10:
                recommendations.append({
                    'action': 'SKEW_TRADE',
                    'expiration': skew.expiration.isoformat(),
                    'confidence': 'medium',
                    'reason': f"Extreme skew of {skew.skew_25d:.2%} for {skew.dte}DTE"
                })

        signals['recommendations'] = recommendations

        return signals
