"""
QML Integration - Streamlined quantum scoring for live analysis.

This module provides a simplified interface for using the quantum trade scorer
in live analysis mode. It handles:
- Auto-training on recent backtest data
- Model caching to avoid re-training every run
- Seamless scoring integration with TradeCandidate

Usage:
    from analysis.qml_integration import get_qml_scorer

    scorer = get_qml_scorer(symbols=['SPY', 'QQQ'], training_months=12)
    if scorer:
        for candidate in candidates:
            candidate.overall_score = scorer.score(candidate) * 100
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from core.models import TradeCandidate

logger = logging.getLogger(__name__)

# Check dependencies
try:
    from analysis.quantum_scorer import (
        create_scorer,
        extract_training_data,
        QuantumScorerConfig,
        PENNYLANE_AVAILABLE,
        TORCH_AVAILABLE,
    )
    QML_AVAILABLE = PENNYLANE_AVAILABLE and TORCH_AVAILABLE
except ImportError:
    QML_AVAILABLE = False
    logger.warning("Quantum scorer module not available")


@dataclass
class QMLConfig:
    """Configuration for QML integration."""
    training_months: int = 12
    min_trades_required: int = 30
    cache_dir: Path = Path(".quantum_models")

    # Model parameters
    n_qubits: int = 7  # 7 qubits: 6 trade features + 1 sentiment
    n_layers: int = 3
    epochs: int = 80
    learning_rate: float = 0.01

    # Cache control
    force_retrain: bool = False
    cache_expiry_days: int = 7  # Retrain if cache older than this


class QMLScorer:
    """
    Wrapper for quantum trade scoring with auto-training.

    Automatically trains on recent backtest data if no cached model exists
    or if the cache is stale.
    """

    def __init__(self, config: QMLConfig = None):
        self.config = config or QMLConfig()
        self.scorer = None
        self.is_ready = False
        self.training_info: Dict[str, Any] = {}
        self._sentiment_signals: Dict[str, Any] = {}

    def set_sentiment(self, sentiment_signals: Dict[str, Any]) -> None:
        """
        Set sentiment signals for scoring.

        Args:
            sentiment_signals: Dict mapping symbol to SentimentSignal objects
        """
        self._sentiment_signals = sentiment_signals or {}
        if self.scorer is not None:
            self.scorer.set_sentiment(self._sentiment_signals)

    def initialize(
        self,
        symbols: List[str],
        verbose: bool = True
    ) -> bool:
        """
        Initialize the scorer, training if necessary.

        Args:
            symbols: Symbols to use for training data
            verbose: Print progress messages

        Returns:
            True if scorer is ready to use, False otherwise
        """
        if not QML_AVAILABLE:
            if verbose:
                print("QML dependencies not available. Install with:")
                print("  pip install pennylane pennylane-lightning torch")
            return False

        # Generate cache key based on symbols and config
        cache_key = self._generate_cache_key(symbols)
        cache_path = self.config.cache_dir / f"qml_scorer_{cache_key}.pt"
        meta_path = self.config.cache_dir / f"qml_scorer_{cache_key}_meta.json"

        # Check if valid cache exists
        if self._is_cache_valid(cache_path, meta_path) and not self.config.force_retrain:
            if verbose:
                print(f"Loading cached QML model...")
            return self._load_cached_model(cache_path, meta_path, verbose)

        # Need to train
        if verbose:
            print(f"Training QML scorer on {self.config.training_months} months of data...")

        return self._train_and_cache(symbols, cache_path, meta_path, verbose)

    def _generate_cache_key(self, symbols: List[str]) -> str:
        """Generate unique cache key for this symbol set and config."""
        key_data = {
            'symbols': sorted(symbols),
            'months': self.config.training_months,
            'qubits': self.config.n_qubits,
            'layers': self.config.n_layers,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _is_cache_valid(self, cache_path: Path, meta_path: Path) -> bool:
        """Check if cached model exists and is not expired."""
        if not cache_path.exists() or not meta_path.exists():
            return False

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            trained_date = date.fromisoformat(meta['trained_date'])
            age_days = (date.today() - trained_date).days

            if age_days > self.config.cache_expiry_days:
                logger.info(f"Cache expired ({age_days} days old)")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error reading cache metadata: {e}")
            return False

    def _load_cached_model(
        self,
        cache_path: Path,
        meta_path: Path,
        verbose: bool
    ) -> bool:
        """Load model from cache."""
        try:
            scorer_config = QuantumScorerConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
            )
            self.scorer = create_scorer(scorer_config)
            self.scorer.load(cache_path)

            with open(meta_path) as f:
                self.training_info = json.load(f)

            self.is_ready = True

            if verbose:
                print(f"  Loaded model trained on {self.training_info.get('trained_date', 'unknown')}")
                print(f"  Training accuracy: {self.training_info.get('val_accuracy', 0):.1%}")

            return True

        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")
            return False

    def _train_and_cache(
        self,
        symbols: List[str],
        cache_path: Path,
        meta_path: Path,
        verbose: bool
    ) -> bool:
        """Train model and save to cache."""
        try:
            # Import here to avoid circular imports and speed up initial load
            from backtesting.backtester import run_backtest
            from core.config import StrategyType

            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=self.config.training_months * 30)

            if verbose:
                print(f"  Period: {start_date} to {end_date}")
                print(f"  Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
                print()

            # Run backtest to generate training data
            if verbose:
                print("  Running backtest for training data...")

            result = run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                strategies=[
                    StrategyType.CASH_SECURED_PUT,
                    StrategyType.PUT_CREDIT_SPREAD,
                    StrategyType.CALL_CREDIT_SPREAD,
                ],
                profit_target=0.5,
                stop_loss=2.0,
                max_positions=5,
                verbose=False,  # Suppress backtest output
            )

            if verbose:
                print(f"  Backtest complete: {result.metrics.total_trades} trades, "
                      f"{result.metrics.win_rate:.1%} win rate")

            # Extract training data
            candidates, outcomes = extract_training_data(result.trades)

            if len(candidates) < self.config.min_trades_required:
                if verbose:
                    print(f"  Insufficient trades ({len(candidates)} < {self.config.min_trades_required})")
                    print("  Try a longer training period or more symbols")
                return False

            # Create and train scorer
            scorer_config = QuantumScorerConfig(
                n_qubits=self.config.n_qubits,
                n_layers=self.config.n_layers,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
            )

            self.scorer = create_scorer(scorer_config)

            if verbose:
                print(f"  Training on {len(candidates)} trades...")
                print()

            metrics = self.scorer.train(candidates, outcomes, verbose=verbose)

            # Save to cache
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self.scorer.save(cache_path)

            # Save metadata
            self.training_info = {
                'trained_date': date.today().isoformat(),
                'symbols': symbols,
                'training_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                },
                'n_trades': len(candidates),
                'val_accuracy': metrics.get('final_val_acc', 0),
                'val_loss': metrics.get('final_val_loss', 0),
                'model_type': metrics.get('model_type', 'unknown'),
            }

            with open(meta_path, 'w') as f:
                json.dump(self.training_info, f, indent=2)

            self.is_ready = True

            if verbose:
                print()
                print(f"  Model trained and cached")
                print(f"  Validation accuracy: {metrics.get('final_val_acc', 0):.1%}")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if verbose:
                print(f"  Training failed: {e}")
            return False

    def score(self, candidate: TradeCandidate) -> float:
        """
        Score a single trade candidate.

        Returns score between 0 and 1, or the original overall_score if not ready.
        """
        if not self.is_ready or self.scorer is None:
            # Fallback to normalized original score
            return candidate.overall_score / 100 if candidate.overall_score else 0.5

        try:
            # Ensure sentiment is set before scoring
            if self._sentiment_signals:
                self.scorer.set_sentiment(self._sentiment_signals)
            return self.scorer.score(candidate)
        except Exception as e:
            logger.warning(f"Scoring error: {e}")
            return candidate.overall_score / 100 if candidate.overall_score else 0.5

    def score_and_update(
        self,
        candidates: List[TradeCandidate],
        sentiment_signals: Dict[str, Any] = None
    ) -> List[TradeCandidate]:
        """
        Score candidates and update their overall_score in place.

        Also stores original score and QML delta as attributes for display.

        Args:
            candidates: List of TradeCandidate objects to score
            sentiment_signals: Optional sentiment signals to use for scoring

        Returns:
            The same list with updated overall_score values
        """
        if not self.is_ready:
            return candidates

        # Update sentiment if provided
        if sentiment_signals is not None:
            self.set_sentiment(sentiment_signals)

        for candidate in candidates:
            qml_score = self.score(candidate)
            # Store original score for comparison
            original = candidate.overall_score if candidate.overall_score else 50
            candidate._original_score = original

            # Scale to 0-100 range and blend with original score
            # 70% QML score, 30% original score for stability
            new_score = 0.7 * (qml_score * 100) + 0.3 * original

            # Store QML delta (how much QML changed the score)
            candidate._qml_delta = new_score - original
            candidate._qml_raw = qml_score * 100  # Raw QML score (0-100)

            candidate.overall_score = new_score

        return candidates

    def get_status(self) -> Dict[str, Any]:
        """Get current scorer status and info."""
        return {
            'is_ready': self.is_ready,
            'qml_available': QML_AVAILABLE,
            'training_info': self.training_info,
        }


# Module-level singleton for convenience
_global_scorer: Optional[QMLScorer] = None


def get_qml_scorer(
    symbols: List[str],
    training_months: int = 12,
    force_retrain: bool = False,
    verbose: bool = True,
) -> Optional[QMLScorer]:
    """
    Get or create a QML scorer, auto-training if needed.

    This is the main entry point for using QML in live analysis.

    Args:
        symbols: Symbols to train on (should match analysis symbols)
        training_months: Months of backtest data for training
        force_retrain: Force retraining even if cache exists
        verbose: Print progress messages

    Returns:
        QMLScorer instance if successful, None if QML not available

    Example:
        scorer = get_qml_scorer(['SPY', 'QQQ', 'IWM'])
        if scorer:
            candidates = scorer.score_and_update(candidates)
    """
    global _global_scorer

    if not QML_AVAILABLE:
        if verbose:
            print("[QML] Dependencies not installed. Run: pip install pennylane pennylane-lightning torch")
        return None

    config = QMLConfig(
        training_months=training_months,
        force_retrain=force_retrain,
    )

    scorer = QMLScorer(config)

    if scorer.initialize(symbols, verbose=verbose):
        _global_scorer = scorer
        return scorer

    return None


def score_candidates_with_qml(
    candidates: List[TradeCandidate],
    symbols: List[str],
    training_months: int = 12,
    verbose: bool = False,
    sentiment_signals: Dict[str, Any] = None,
) -> List[TradeCandidate]:
    """
    Convenience function to score candidates with QML.

    Handles initialization and scoring in one call. Uses cached model if available.

    Args:
        candidates: Candidates to score
        symbols: Symbols for training (if needed)
        training_months: Training data period
        verbose: Print progress
        sentiment_signals: Optional sentiment signals for scoring

    Returns:
        Candidates with updated overall_score
    """
    global _global_scorer

    # Try to use existing scorer
    if _global_scorer and _global_scorer.is_ready:
        return _global_scorer.score_and_update(candidates, sentiment_signals)

    # Initialize new scorer
    scorer = get_qml_scorer(symbols, training_months, verbose=verbose)

    if scorer:
        return scorer.score_and_update(candidates, sentiment_signals)

    return candidates
