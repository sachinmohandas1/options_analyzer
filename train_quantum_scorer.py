#!/usr/bin/env python3
"""
Training script for the Quantum Trade Scorer.

This script:
1. Runs a backtest to generate training data
2. Extracts trade outcomes from closed positions
3. Trains the quantum VQC scorer on the historical results
4. Validates performance and saves the model

Usage:
    python train_quantum_scorer.py --start-date 2022-01-01 --end-date 2024-01-01 -s SPY QQQ IWM

The trained model can then be loaded in live analysis to score trade candidates.
"""

import argparse
from datetime import date, datetime
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.backtester import run_backtest, BacktestConfig, BacktestResult
from backtesting.trade_manager import ExitRules
from core.config import StrategyType, TradeCriteria

from analysis.quantum_scorer import (
    create_scorer,
    extract_training_data,
    QuantumScorerConfig,
    PENNYLANE_AVAILABLE,
    TORCH_AVAILABLE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Quantum Trade Scorer on backtest results"
    )

    # Backtest parameters
    parser.add_argument(
        '--start-date', type=str, required=True,
        help='Backtest start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str, required=True,
        help='Backtest end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '-s', '--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
        help='Symbols to backtest'
    )
    parser.add_argument(
        '--capital', type=float, default=13000.0,
        help='Initial capital for backtest'
    )

    # Training parameters
    parser.add_argument(
        '--n-qubits', type=int, default=6,
        help='Number of qubits in VQC'
    )
    parser.add_argument(
        '--n-layers', type=int, default=3,
        help='Number of variational layers'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Training epochs'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='Learning rate'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=Path, default=Path('.quantum_models'),
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )

    return parser.parse_args()


def run_backtest_for_training(
    symbols: list,
    start_date: date,
    end_date: date,
    initial_capital: float,
    verbose: bool = True
) -> BacktestResult:
    """Run backtest to generate training data."""

    if verbose:
        print("=" * 60)
        print("PHASE 1: Running Backtest for Training Data")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Capital: ${initial_capital:,.2f}")
        print()

    result = run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategies=[
            StrategyType.CASH_SECURED_PUT,
            StrategyType.PUT_CREDIT_SPREAD,
            StrategyType.CALL_CREDIT_SPREAD,
        ],
        profit_target=0.5,
        stop_loss=2.0,
        max_positions=5,
        verbose=verbose
    )

    if verbose:
        print()
        print(f"Backtest Complete:")
        print(f"  Total Trades: {result.metrics.total_trades}")
        print(f"  Win Rate: {result.metrics.win_rate:.1%}")
        print(f"  Total P&L: ${result.metrics.total_pnl:+,.2f}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print()

    return result


def train_scorer(
    result: BacktestResult,
    config: QuantumScorerConfig,
    verbose: bool = True
) -> dict:
    """Train the quantum scorer on backtest results."""

    if verbose:
        print("=" * 60)
        print("PHASE 2: Training Quantum Scorer")
        print("=" * 60)

    # Extract training data
    candidates, outcomes = extract_training_data(result.trades)

    if len(candidates) < 20:
        raise ValueError(
            f"Insufficient training data: only {len(candidates)} trades. "
            "Need at least 20 trades for meaningful training."
        )

    if verbose:
        n_profitable = sum(1 for o in outcomes if o > 0.5)
        print(f"Training Data:")
        print(f"  Total Samples: {len(candidates)}")
        print(f"  Profitable: {n_profitable} ({n_profitable/len(candidates):.1%})")
        print(f"  Unprofitable: {len(candidates) - n_profitable}")
        print()
        print(f"Model Configuration:")
        print(f"  Qubits: {config.n_qubits}")
        print(f"  Layers: {config.n_layers}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Learning Rate: {config.learning_rate}")
        print()

    # Create and train scorer
    scorer = create_scorer(config)

    if verbose:
        print("Training...")
        print()

    metrics = scorer.train(candidates, outcomes, verbose=verbose)

    if verbose:
        print()
        print("Training Complete:")
        print(f"  Final Validation Loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final Validation Accuracy: {metrics['final_val_acc']:.1%}")
        print(f"  Model Type: {metrics['model_type']}")
        print()

    return scorer, metrics


def evaluate_scorer(
    scorer,
    result: BacktestResult,
    verbose: bool = True
) -> dict:
    """Evaluate scorer on full dataset and analyze feature importance."""

    if verbose:
        print("=" * 60)
        print("PHASE 3: Model Evaluation")
        print("=" * 60)

    candidates, outcomes = extract_training_data(result.trades)

    # Score all trades
    scores = scorer.score_batch(candidates)

    # Analyze correlation between scores and outcomes
    high_score_mask = scores > 0.6
    low_score_mask = scores < 0.4

    high_score_outcomes = [outcomes[i] for i in range(len(outcomes)) if high_score_mask[i]]
    low_score_outcomes = [outcomes[i] for i in range(len(outcomes)) if low_score_mask[i]]

    evaluation = {
        'high_score_count': sum(high_score_mask),
        'high_score_win_rate': sum(high_score_outcomes) / len(high_score_outcomes) if high_score_outcomes else 0,
        'low_score_count': sum(low_score_mask),
        'low_score_win_rate': sum(low_score_outcomes) / len(low_score_outcomes) if low_score_outcomes else 0,
        'baseline_win_rate': sum(outcomes) / len(outcomes),
    }

    if verbose:
        print(f"Score Distribution Analysis:")
        print(f"  High Score (>0.6) Trades: {evaluation['high_score_count']}")
        print(f"    Win Rate: {evaluation['high_score_win_rate']:.1%}")
        print(f"  Low Score (<0.4) Trades: {evaluation['low_score_count']}")
        print(f"    Win Rate: {evaluation['low_score_win_rate']:.1%}")
        print(f"  Baseline Win Rate: {evaluation['baseline_win_rate']:.1%}")
        print()

        # Feature importance
        if hasattr(scorer, 'get_feature_importance'):
            importance = scorer.get_feature_importance()
            print("Feature Importance (approximate):")
            for name, value in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "=" * int(value * 50)
                print(f"  {name:15s}: {value:.3f} {bar}")
            print()

    return evaluation


def save_model_and_report(
    scorer,
    metrics: dict,
    evaluation: dict,
    result: BacktestResult,
    config: QuantumScorerConfig,
    output_dir: Path,
    verbose: bool = True
):
    """Save trained model and generate report."""

    if verbose:
        print("=" * 60)
        print("PHASE 4: Saving Model and Report")
        print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = scorer.save(output_dir / "quantum_scorer.pt")

    # Save training report
    report = {
        'training_date': datetime.now().isoformat(),
        'backtest_period': {
            'start': result.start_date.isoformat(),
            'end': result.end_date.isoformat(),
        },
        'symbols': result.symbols,
        'backtest_metrics': {
            'total_trades': result.metrics.total_trades,
            'win_rate': result.metrics.win_rate,
            'total_pnl': result.metrics.total_pnl,
            'sharpe_ratio': result.metrics.sharpe_ratio,
        },
        'model_config': {
            'n_qubits': config.n_qubits,
            'n_layers': config.n_layers,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
        },
        'training_metrics': metrics,
        'evaluation': evaluation,
        'feature_importance': scorer.get_feature_importance() if hasattr(scorer, 'get_feature_importance') else {},
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    if verbose:
        print(f"Model saved to: {model_path}")
        print(f"Report saved to: {report_path}")
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print()
        print("To use the trained model in live analysis:")
        print()
        print("  from analysis.quantum_scorer import create_scorer, QuantumScorerConfig")
        print("  ")
        print("  scorer = create_scorer()")
        print(f"  scorer.load(Path('{model_path}'))")
        print("  ")
        print("  # Score a trade candidate")
        print("  score = scorer.score(candidate)")
        print()

    return model_path, report_path


def main():
    args = parse_args()
    verbose = not args.quiet

    # Check dependencies
    if verbose:
        print()
        print("Quantum Trade Scorer Training")
        print("=" * 60)
        print()
        if PENNYLANE_AVAILABLE and TORCH_AVAILABLE:
            print("Dependencies: PennyLane + PyTorch available")
            print("Using: Quantum VQC Scorer")
        else:
            print("Dependencies: PennyLane/PyTorch NOT available")
            print("Using: Classical Fallback Scorer")
        print()

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    # Run backtest
    result = run_backtest_for_training(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        verbose=verbose
    )

    # Create config
    config = QuantumScorerConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_dir=args.output_dir,
    )

    # Train scorer
    scorer, metrics = train_scorer(result, config, verbose=verbose)

    # Evaluate
    evaluation = evaluate_scorer(scorer, result, verbose=verbose)

    # Save
    model_path, report_path = save_model_and_report(
        scorer, metrics, evaluation, result, config,
        args.output_dir, verbose=verbose
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
