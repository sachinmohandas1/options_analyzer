"""
Main Options Analyzer Orchestrator.

Coordinates all analysis components to produce trade recommendations.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Type
from dataclasses import asdict

from core.config import AnalyzerConfig, StrategyType, DEFAULT_CONFIG
from core.models import (
    OptionsChain, TradeCandidate, VolatilitySurface, AnalysisResult
)
from data.fetcher import DataFetcher
from data.discovery import SymbolDiscovery
from analysis.greeks import GreeksCalculator
from analysis.volatility_surface import VolatilitySurfaceAnalyzer
from analysis.position_sizer import PositionSizer
from strategies import (
    BaseStrategy,
    STRATEGY_REGISTRY,
    CashSecuredPutStrategy,
    CoveredCallStrategy,
    PutCreditSpreadStrategy,
    CallCreditSpreadStrategy,
    IronCondorStrategy,
)

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """
    Main orchestrator for options analysis.

    Workflow:
    1. Fetch options chains for configured underlyings
    2. Calculate Greeks for all contracts
    3. Build and analyze volatility surfaces
    4. Generate trade candidates using enabled strategies
    5. Filter candidates by criteria
    6. Score and rank candidates
    7. Apply position sizing constraints
    8. Return final recommendations
    """

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        use_synthetic: bool = False,
        sentiment_signals: Optional[Dict] = None
    ):
        self.config = config or DEFAULT_CONFIG

        # Initialize components
        self.data_fetcher = DataFetcher(
            self.config,
            use_synthetic_fallback=use_synthetic,
            sentiment_signals=sentiment_signals
        )
        self.symbol_discovery = SymbolDiscovery(self.config)
        self.greeks_calculator = GreeksCalculator(self.config)
        self.vol_analyzer = VolatilitySurfaceAnalyzer(self.config)
        self.position_sizer = PositionSizer(self.config)

        # Price cache from discovery
        self._symbol_prices: Dict[str, float] = {}

        # Synthetic mode flag
        self.use_synthetic = use_synthetic

        # Initialize strategies
        self.strategies: List[BaseStrategy] = []
        self._init_strategies()

        # Results storage
        self.last_result: Optional[AnalysisResult] = None

    def enable_synthetic_mode(self, enabled: bool = True, sentiment_signals: Optional[Dict] = None):
        """Enable or disable synthetic chain generation."""
        self.use_synthetic = enabled
        self.data_fetcher.enable_synthetic_fallback(enabled)
        if sentiment_signals:
            self.data_fetcher.set_sentiment_signals(sentiment_signals)

    def refresh_iv_surfaces(self, symbols: Optional[List[str]] = None) -> int:
        """Refresh IV surface cache for synthetic chain generation."""
        return self.data_fetcher.refresh_iv_surfaces(symbols)

    def _init_strategies(self):
        """Initialize enabled strategies."""
        strategy_map = {
            StrategyType.CASH_SECURED_PUT: CashSecuredPutStrategy,
            StrategyType.COVERED_CALL: CoveredCallStrategy,
            StrategyType.PUT_CREDIT_SPREAD: PutCreditSpreadStrategy,
            StrategyType.CALL_CREDIT_SPREAD: CallCreditSpreadStrategy,
            StrategyType.IRON_CONDOR: IronCondorStrategy,
        }

        for strategy_type in self.config.enabled_strategies:
            strategy_class = strategy_map.get(strategy_type)
            if strategy_class:
                self.strategies.append(strategy_class(self.config))
                logger.info(f"Enabled strategy: {strategy_type.value}")

    def add_strategy(self, strategy: BaseStrategy):
        """Add a custom strategy to the analyzer."""
        self.strategies.append(strategy)
        logger.info(f"Added custom strategy: {strategy.name}")

    def _prefetch_risk_data(self, symbols: List[str]) -> None:
        """
        Pre-fetch risk data (earnings, historical returns) for all symbols.

        This populates the caches before candidate processing, avoiding
        redundant API calls when processing multiple candidates per symbol.
        """
        try:
            from analysis.risk_metrics import (
                _get_singleton_earnings_calendar,
                _get_singleton_cvar_calculator
            )

            earnings_cal = _get_singleton_earnings_calendar()
            cvar_calc = _get_singleton_cvar_calculator()

            total = len(symbols)
            for i, symbol in enumerate(symbols):
                if (i + 1) % 50 == 0 or i == 0:
                    logger.info(f"Pre-fetching risk data: {i + 1}/{total} symbols...")

                # Pre-fetch earnings (will be cached)
                try:
                    earnings_cal.get_next_earnings(symbol)
                except Exception:
                    pass

                # Pre-fetch historical returns for CVaR (will be cached)
                try:
                    cvar_calc.get_historical_returns(symbol)
                except Exception:
                    pass

            logger.info(f"Risk data pre-fetch complete for {total} symbols")

        except ImportError:
            logger.debug("Risk metrics module not available for pre-fetching")

    def run_analysis(
        self,
        symbols: Optional[List[str]] = None,
        include_volatility_analysis: bool = True,
        full_scan: bool = False
    ) -> AnalysisResult:
        """
        Run complete analysis pipeline.

        Args:
            symbols: Optional list of symbols to analyze (uses config if None)
            include_volatility_analysis: Whether to perform vol surface analysis
            full_scan: If True, scan S&P 500 + Nasdaq 100 + ETFs (~600 symbols)

        Returns:
            AnalysisResult containing all findings
        """
        result = AnalysisResult(
            generated_at=datetime.now(),
            config_snapshot=self._get_config_snapshot()
        )

        # Reset position sizer for fresh run
        self.position_sizer.reset_portfolio()

        # 1. Discover symbols (optionally filter by price if max_share_price is set)
        if symbols:
            # User provided specific symbols
            filtered_symbols, prices = self.symbol_discovery.filter_symbols_by_price(symbols)
        else:
            # Use dynamic discovery to find symbols with options
            filtered_symbols, prices = self.symbol_discovery.discover_symbols(full_scan=full_scan)

        self._symbol_prices = prices
        logger.info(f"Found {len(filtered_symbols)} symbols with options available")

        if not filtered_symbols:
            logger.warning("No symbols found with options available")
            return result

        # 2. Fetch options chains for filtered symbols
        logger.info("Fetching options chains...")
        chains = self.data_fetcher.fetch_all_chains(filtered_symbols)
        result.chains_analyzed = chains

        if not chains:
            logger.warning("No options chains fetched")
            return result

        # 2. Pre-fetch risk data for all symbols (earnings, historical returns)
        # This populates the caches before candidate processing
        logger.info("Pre-fetching risk data for all symbols...")
        self._prefetch_risk_data(list(chains.keys()))

        # 3. Process each chain
        all_candidates = []

        for symbol, chain in chains.items():
            try:
                # Calculate Greeks
                logger.info(f"Calculating Greeks for {symbol}...")
                chain = self.greeks_calculator.enrich_chain(chain)

                # Volatility surface analysis
                if include_volatility_analysis:
                    logger.info(f"Building volatility surface for {symbol}...")
                    surface = self.vol_analyzer.build_surface(chain)
                    result.volatility_surfaces[symbol] = surface
                else:
                    surface = None

                # Generate candidates from all strategies
                for strategy in self.strategies:
                    logger.debug(f"Running {strategy.name} for {symbol}...")

                    # Skip covered call if we're not tracking owned stocks
                    if strategy.requires_stock_ownership:
                        continue  # TODO: Add owned stocks tracking

                    candidates = strategy.find_candidates(chain, surface)

                    # Filter by criteria
                    filtered = strategy.filter_by_criteria(candidates)

                    # Enrich and score each candidate with the new scoring system
                    for candidate in filtered:
                        strategy.enrich_candidate(candidate, chain)
                        strategy.score_candidate(candidate, chain)

                    all_candidates.extend(filtered)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                result.errors.append({'symbol': symbol, 'error': str(e)})

        # 4. Store all candidates
        result.all_candidates = all_candidates
        result.total_opportunities_found = len(all_candidates)

        # 5. Final filtering by capital
        capital_filtered = self.position_sizer.filter_by_capital(all_candidates)
        result.filtered_candidates = capital_filtered
        result.opportunities_meeting_criteria = len(capital_filtered)

        # 6. Sort by score and get top candidates
        sorted_candidates = sorted(
            capital_filtered,
            key=lambda x: x.overall_score,
            reverse=True
        )
        result.top_candidates = sorted_candidates[:self.config.top_n_trades]

        # 7. Calculate deployable capital
        result.capital_deployable = sum(
            c.collateral_required for c in result.top_candidates
        )

        self.last_result = result
        return result

    def get_volatility_signals(self, symbol: str) -> Optional[Dict]:
        """Get volatility analysis signals for a specific symbol."""
        if not self.last_result:
            return None

        chain = self.last_result.chains_analyzed.get(symbol)
        surface = self.last_result.volatility_surfaces.get(symbol)

        if not chain or not surface:
            return None

        return self.vol_analyzer.get_trading_signals(chain, surface)

    def get_candidates_for_symbol(self, symbol: str) -> List[TradeCandidate]:
        """Get all candidates for a specific symbol."""
        if not self.last_result:
            return []

        return [
            c for c in self.last_result.filtered_candidates
            if c.underlying_symbol == symbol
        ]

    def get_candidates_by_strategy(self, strategy_name: str) -> List[TradeCandidate]:
        """Get all candidates for a specific strategy."""
        if not self.last_result:
            return []

        return [
            c for c in self.last_result.filtered_candidates
            if c.strategy_name == strategy_name
        ]

    def _get_config_snapshot(self) -> Dict:
        """Create a snapshot of current configuration."""
        return {
            'trade_criteria': asdict(self.config.trade_criteria),
            'capital': asdict(self.config.capital),
            'enabled_strategies': [s.value for s in self.config.enabled_strategies],
            'symbols': self.config.get_active_symbols(),
        }

    def update_config(self, **kwargs):
        """
        Update configuration at runtime.

        Supports nested updates like:
            update_config(trade_criteria={'min_prob_profit': 0.75})
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                current = getattr(self.config, key)
                if isinstance(value, dict) and hasattr(current, '__dict__'):
                    # Update nested config
                    for nested_key, nested_value in value.items():
                        if hasattr(current, nested_key):
                            setattr(current, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)

        # Reinitialize components that depend on config
        self.position_sizer = PositionSizer(self.config)


def create_analyzer(
    capital: float = 13000,
    min_prob_profit: float = 0.70,
    min_weekly_return: float = 1.0,
    max_dte: int = 5,
    symbols: Optional[List[str]] = None
) -> OptionsAnalyzer:
    """
    Factory function to create analyzer with common configurations.
    """
    config = AnalyzerConfig()

    # Capital settings
    config.capital.total_capital = capital

    # Trade criteria
    config.trade_criteria.min_prob_profit = min_prob_profit
    config.trade_criteria.min_weekly_return_pct = min_weekly_return
    config.trade_criteria.max_dte = max_dte

    # Symbols
    if symbols:
        config.underlyings.default_symbols = symbols

    return OptionsAnalyzer(config)
