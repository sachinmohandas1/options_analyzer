"""
Quantum-Enhanced Trade Scoring using PennyLane VQC.

This module implements a Variational Quantum Circuit (VQC) based trade scorer
that learns optimal trade selection from backtest outcomes. It uses hybrid
quantum-classical optimization to find non-linear feature interactions that
simple weighted scoring might miss.

Key Features:
- Angle embedding of trade features into quantum states
- Strongly entangling variational layers for expressiveness
- PyTorch integration for gradient-based optimization
- Compatible with existing TradeCandidate and backtest infrastructure

References:
- PennyLane QML: https://pennylane.ai/qml/quantum-machine-learning
- VQC for Finance: https://www.ijsat.org/papers/2025/4/9033.pdf
- Quantum Kernels: https://pennylane.ai/qml/demos/tutorial_kernels_module
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Check for PennyLane availability
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. Quantum scorer will use classical fallback.")

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Training features disabled.")

from core.models import TradeCandidate
from backtesting.trade_manager import TradeRecord


@dataclass
class QuantumScorerConfig:
    """Configuration for the quantum trade scorer."""

    # Circuit architecture
    n_qubits: int = 7  # One per input feature (including sentiment)
    n_layers: int = 3  # Variational layer depth

    # Device selection
    device_name: str = 'default.qubit'  # Options: 'default.qubit', 'lightning.qubit'

    # Training parameters
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    # Feature scaling
    feature_scale: float = np.pi  # Scale features for rotation gates

    # Model persistence
    model_dir: Path = field(default_factory=lambda: Path(".quantum_models"))

    def __post_init__(self):
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)


class FeatureExtractor:
    """
    Extracts and normalizes features from TradeCandidate objects.

    Features encoded (7 total):
    1. prob_profit - Probability of profit (already 0-1)
    2. weekly_return - Weekly return % (tanh-scaled)
    3. net_delta - Position delta (abs, tanh-scaled)
    4. theta_ratio - Theta relative to premium
    5. iv_rank - IV percentile rank (0-1 normalized)
    6. dte_norm - Days to expiration (normalized)
    7. sentiment_score - News sentiment (-1 to +1, scaled to 0-1)
    """

    FEATURE_NAMES = [
        'prob_profit',
        'weekly_return',
        'net_delta',
        'theta_ratio',
        'iv_rank',
        'dte_norm',
        'sentiment_score',
    ]

    def __init__(self, max_dte: int = 7, feature_scale: float = np.pi):
        self.max_dte = max_dte
        self.feature_scale = feature_scale
        # Sentiment lookup dict: symbol -> sentiment_score (-1 to +1)
        self._sentiment_cache: Dict[str, float] = {}

    def set_sentiment(self, sentiment_signals: Dict[str, Any]) -> None:
        """
        Set sentiment signals for feature extraction.

        Args:
            sentiment_signals: Dict mapping symbol to SentimentSignal objects
        """
        self._sentiment_cache = {}
        for symbol, signal in sentiment_signals.items():
            # Handle both SentimentSignal objects and raw dicts
            if hasattr(signal, 'sentiment_score'):
                self._sentiment_cache[symbol.upper()] = signal.sentiment_score
            elif isinstance(signal, dict):
                self._sentiment_cache[symbol.upper()] = signal.get('sentiment_score', 0.0)

    def extract(self, candidate: TradeCandidate) -> np.ndarray:
        """Extract normalized feature vector from a TradeCandidate."""
        # Get sentiment score for this symbol (default to neutral if not available)
        sentiment = self._sentiment_cache.get(candidate.underlying_symbol.upper(), 0.0)

        features = np.array([
            # 1. Probability of profit (already 0-1)
            candidate.prob_profit,

            # 2. Weekly return (tanh squashes extreme values)
            np.tanh(candidate.weekly_return * 10),

            # 3. Net delta (absolute, scaled)
            np.tanh(abs(candidate.net_delta) * 5),

            # 4. Theta ratio (theta per dollar of premium)
            np.tanh(candidate.net_theta / max(candidate.premium_received * 100, 0.01)),

            # 5. IV rank (normalized to 0-1)
            (candidate.iv_rank or 50) / 100,

            # 6. DTE normalized by max
            min(candidate.dte / self.max_dte, 1.0),

            # 7. Sentiment score: map -1 to +1 -> 0 to 1
            (sentiment + 1) / 2,
        ], dtype=np.float64)

        # Scale for rotation gates
        return features * self.feature_scale

    def extract_batch(self, candidates: List[TradeCandidate]) -> np.ndarray:
        """Extract features for multiple candidates."""
        return np.array([self.extract(c) for c in candidates])


class ClassicalFallbackScorer:
    """
    Classical fallback scorer when PennyLane is not available.
    Uses logistic regression on the same features.
    """

    def __init__(self, config: QuantumScorerConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(feature_scale=1.0)  # No scaling needed
        self.weights = None
        self.bias = 0.0

    def set_sentiment(self, sentiment_signals: Dict[str, Any]) -> None:
        """Set sentiment signals for feature extraction."""
        self.feature_extractor.set_sentiment(sentiment_signals)

    def train(self,
              candidates: List[TradeCandidate],
              outcomes: List[float],
              verbose: bool = True) -> Dict[str, float]:
        """Train using simple logistic regression."""
        X = self.feature_extractor.extract_batch(candidates)
        y = np.array(outcomes)

        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient descent
        lr = self.config.learning_rate
        losses = []

        for epoch in range(self.config.epochs):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = 1 / (1 + np.exp(-z))

            # Binary cross-entropy loss
            eps = 1e-8
            loss = -np.mean(y * np.log(predictions + eps) + (1 - y) * np.log(1 - predictions + eps))
            losses.append(loss)

            # Gradients
            error = predictions - y
            grad_w = X.T @ error / len(y)
            grad_b = np.mean(error)

            # Update
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        return {
            'final_loss': losses[-1],
            'epochs_trained': len(losses),
            'model_type': 'classical_fallback'
        }

    def score(self, candidate: TradeCandidate) -> float:
        """Score a single candidate."""
        if self.weights is None:
            raise ValueError("Model not trained. Call train() first.")

        features = self.feature_extractor.extract(candidate)
        z = features @ self.weights + self.bias
        return float(1 / (1 + np.exp(-z)))

    def score_batch(self, candidates: List[TradeCandidate]) -> np.ndarray:
        """Score multiple candidates."""
        return np.array([self.score(c) for c in candidates])


if PENNYLANE_AVAILABLE and TORCH_AVAILABLE:

    class QuantumTradeScorer:
        """
        VQC-based trade scorer using PennyLane and PyTorch.

        Architecture:
        - Angle embedding: encode features as rotation angles
        - Strongly entangling layers: parameterized rotations + entanglement
        - Measurement: expectation value of Pauli-Z on first qubit

        The quantum circuit can capture non-linear correlations between
        features that linear models miss, potentially improving trade selection.
        """

        def __init__(self, config: QuantumScorerConfig = None):
            self.config = config or QuantumScorerConfig()
            self.feature_extractor = FeatureExtractor(
                feature_scale=self.config.feature_scale
            )

            # Initialize quantum device
            self.dev = qml.device(
                self.config.device_name,
                wires=self.config.n_qubits
            )

            # Build the quantum circuit
            self._build_circuit()

            # Model state
            self.params = None
            self.training_history: List[Dict] = []
            self.is_trained = False

        def set_sentiment(self, sentiment_signals: Dict[str, Any]) -> None:
            """Set sentiment signals for feature extraction."""
            self.feature_extractor.set_sentiment(sentiment_signals)

        def _build_circuit(self):
            """Construct the variational quantum circuit."""
            n_qubits = self.config.n_qubits
            n_layers = self.config.n_layers

            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def circuit(params, features):
                """
                Variational quantum circuit for trade scoring.

                Args:
                    params: Trainable parameters of shape (n_layers, n_qubits, 3)
                    features: Input features of shape (n_qubits,)

                Returns:
                    Expectation value of Pauli-Z on qubit 0
                """
                # Angle embedding: encode features as Y-rotations
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation='Y')

                # Strongly entangling layers
                qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

                # Measure expectation value
                return qml.expval(qml.PauliZ(0))

            self.circuit = circuit

        def _initialize_params(self) -> nn.Parameter:
            """Initialize trainable parameters as nn.Parameter for stable optimization."""
            shape = (self.config.n_layers, self.config.n_qubits, 3)
            params = nn.Parameter(torch.randn(shape, dtype=torch.float64) * 0.1)
            return params

        def train(self,
                  candidates: List[TradeCandidate],
                  outcomes: List[float],
                  verbose: bool = True) -> Dict[str, Any]:
            """
            Train the quantum scorer on historical trade outcomes.

            Args:
                candidates: List of TradeCandidate objects from backtest
                outcomes: List of floats (1.0 for profitable, 0.0 for losing)
                verbose: Print training progress

            Returns:
                Dictionary with training metrics
            """
            # Extract features
            X = self.feature_extractor.extract_batch(candidates)
            X_tensor = torch.tensor(X, dtype=torch.float64)
            y_tensor = torch.tensor(outcomes, dtype=torch.float64)

            # Train/validation split
            n_samples = len(candidates)
            n_val = int(n_samples * self.config.validation_split)
            indices = np.random.permutation(n_samples)

            train_idx = indices[n_val:]
            val_idx = indices[:n_val]

            X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
            X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

            # Initialize parameters as nn.Parameter for stable optimization
            self.params = self._initialize_params()
            optimizer = torch.optim.Adam([self.params], lr=self.config.learning_rate)

            # Training loop
            best_val_loss = float('inf')
            best_params = self.params.data.clone()  # Initialize with starting params
            patience_counter = 0
            history = []

            for epoch in range(self.config.epochs):
                # Training step
                optimizer.zero_grad()

                train_predictions = self._forward_batch(X_train)
                train_loss = self._compute_loss(train_predictions, y_train)

                train_loss.backward()
                optimizer.step()

                # Validation
                with torch.no_grad():
                    val_predictions = self._forward_batch(X_val)
                    val_loss = self._compute_loss(val_predictions, y_val)

                epoch_metrics = {
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'train_acc': self._compute_accuracy(train_predictions, y_train),
                    'val_acc': self._compute_accuracy(val_predictions, y_val),
                }
                history.append(epoch_metrics)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = self.params.data.clone()
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, "
                          f"Val Acc = {epoch_metrics['val_acc']:.2%}")

            # Restore best parameters as nn.Parameter
            self.params = nn.Parameter(best_params.clone().detach())
            self.is_trained = True
            self.training_history = history

            return {
                'final_train_loss': history[-1]['train_loss'],
                'final_val_loss': history[-1]['val_loss'],
                'final_val_acc': history[-1]['val_acc'],
                'epochs_trained': len(history),
                'model_type': 'quantum_vqc',
                'n_qubits': self.config.n_qubits,
                'n_layers': self.config.n_layers,
            }

        def _forward_batch(self, X: torch.Tensor) -> torch.Tensor:
            """Forward pass for a batch of samples."""
            predictions = []
            for features in X:
                raw = self.circuit(self.params, features)
                # Map [-1, 1] to [0, 1]
                pred = (raw + 1) / 2
                predictions.append(pred)
            return torch.stack(predictions)

        def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Binary cross-entropy loss."""
            eps = 1e-8
            loss = -torch.mean(
                targets * torch.log(predictions + eps) +
                (1 - targets) * torch.log(1 - predictions + eps)
            )
            return loss

        def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
            """Compute classification accuracy."""
            pred_classes = (predictions > 0.5).float()
            return float((pred_classes == targets).float().mean())

        def score(self, candidate: TradeCandidate) -> float:
            """
            Score a single trade candidate.

            Args:
                candidate: TradeCandidate to score

            Returns:
                Score between 0 and 1 (higher = more likely profitable)
            """
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")

            features = self.feature_extractor.extract(candidate)
            features_tensor = torch.tensor(features, dtype=torch.float64)

            with torch.no_grad():
                raw = self.circuit(self.params, features_tensor)
                score = float((raw + 1) / 2)

            return score

        def score_batch(self, candidates: List[TradeCandidate]) -> np.ndarray:
            """Score multiple candidates efficiently."""
            return np.array([self.score(c) for c in candidates])

        def save(self, path: Optional[Path] = None) -> Path:
            """Save model parameters to disk."""
            if path is None:
                self.config.model_dir.mkdir(parents=True, exist_ok=True)
                path = self.config.model_dir / "quantum_scorer.pt"

            state = {
                'params': self.params.detach().numpy(),
                'config': {
                    'n_qubits': self.config.n_qubits,
                    'n_layers': self.config.n_layers,
                    'feature_scale': self.config.feature_scale,
                },
                'training_history': self.training_history,
                'is_trained': self.is_trained,
            }

            torch.save(state, path)
            logger.info(f"Model saved to {path}")
            return path

        def load(self, path: Path) -> None:
            """Load model parameters from disk."""
            state = torch.load(path)

            # Verify config compatibility
            saved_config = state['config']
            if (saved_config['n_qubits'] != self.config.n_qubits or
                saved_config['n_layers'] != self.config.n_layers):
                raise ValueError(
                    f"Config mismatch: saved model has {saved_config['n_qubits']} qubits, "
                    f"{saved_config['n_layers']} layers but current config has "
                    f"{self.config.n_qubits} qubits, {self.config.n_layers} layers"
                )

            self.params = nn.Parameter(torch.tensor(state['params'], dtype=torch.float64))
            self.training_history = state.get('training_history', [])
            self.is_trained = state.get('is_trained', True)

            logger.info(f"Model loaded from {path}")

        def get_feature_importance(self) -> Dict[str, float]:
            """
            Estimate feature importance via parameter gradients.

            Returns approximate importance of each input feature based on
            how much the circuit parameters associated with that qubit
            affect the output.
            """
            if not self.is_trained:
                return {name: 0.0 for name in FeatureExtractor.FEATURE_NAMES}

            # Sum absolute parameter values per qubit as rough importance proxy
            param_norms = torch.abs(self.params).sum(dim=(0, 2))  # Sum over layers and rotation types
            param_norms = param_norms / param_norms.sum()  # Normalize

            importance = {}
            for i, name in enumerate(FeatureExtractor.FEATURE_NAMES):
                importance[name] = float(param_norms[i])

            return importance

else:
    # PennyLane or PyTorch not available - define stub
    class QuantumTradeScorer(ClassicalFallbackScorer):
        """Fallback to classical scorer when PennyLane/PyTorch unavailable."""
        pass


def create_scorer(config: Optional[QuantumScorerConfig] = None) -> "QuantumTradeScorer":
    """
    Factory function to create the appropriate scorer.

    Returns QuantumTradeScorer if dependencies available, else ClassicalFallbackScorer.
    """
    config = config or QuantumScorerConfig()

    if PENNYLANE_AVAILABLE and TORCH_AVAILABLE:
        logger.info("Creating quantum trade scorer with PennyLane")
        return QuantumTradeScorer(config)
    else:
        logger.warning("PennyLane/PyTorch not available, using classical fallback")
        return ClassicalFallbackScorer(config)


def extract_training_data(
    closed_trades: List[TradeRecord],
    min_profit_threshold: float = 0.0
) -> Tuple[List[TradeCandidate], List[float]]:
    """
    Extract training data from closed backtest trades.

    Args:
        closed_trades: List of TradeRecord objects from backtest
        min_profit_threshold: Minimum P&L to consider "profitable" (default 0)

    Returns:
        Tuple of (candidates, outcomes) for training
    """
    candidates = []
    outcomes = []

    for trade in closed_trades:
        if trade.entry_candidate is None:
            continue

        candidates.append(trade.entry_candidate)

        # Binary outcome: 1 if profitable, 0 if not
        if trade.realized_pnl is not None:
            outcome = 1.0 if trade.realized_pnl > min_profit_threshold else 0.0
        else:
            outcome = 0.5  # Unknown

        outcomes.append(outcome)

    return candidates, outcomes
