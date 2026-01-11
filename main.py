#!/usr/bin/env python3
"""
Options Analyzer CLI - Main Entry Point

A tool for scanning options chains and identifying high-probability
premium selling opportunities.
"""

import argparse
import logging
import sys
from datetime import datetime, date
from typing import Optional, List

from analyzer import OptionsAnalyzer, create_analyzer
from core.config import AnalyzerConfig, StrategyType
from ui.display import (
    console,
    display_header,
    display_portfolio_summary,
    display_candidates_table,
    display_trade_detail,
    display_volatility_signals,
    display_analysis_summary,
    display_errors,
    display_menu,
    display_backtest_result,
    display_backtest_header,
)
from analysis.position_sizer import PositionSizer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Options Chain Analyzer - Find high-probability premium selling opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         # Run with defaults
  python main.py -s SPY QQQ IWM          # Analyze specific symbols
  python main.py --capital 25000         # Set capital to $25,000
  python main.py --prob 0.75 --return 1.5 # 75% prob, 1.5% weekly return
  python main.py --max-dte 3             # Only look at 3 DTE or less
  python main.py --no-vol                # Skip volatility analysis
  python main.py -v                      # Verbose output
        """
    )

    parser.add_argument(
        '-s', '--symbols',
        nargs='+',
        help='Symbols to analyze (default: major index ETFs)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=13000,
        help='Total capital available (default: $13,000)'
    )

    parser.add_argument(
        '--prob',
        type=float,
        default=0.70,
        help='Minimum probability of profit (default: 0.70 = 70%%)'
    )

    parser.add_argument(
        '--return',
        type=float,
        default=1.0,
        dest='min_return',
        help='Minimum weekly return %% (default: 1.0 = 1%%)'
    )

    parser.add_argument(
        '--max-dte',
        type=int,
        default=5,
        help='Maximum days to expiration (default: 5)'
    )

    parser.add_argument(
        '--min-dte',
        type=int,
        default=1,
        help='Minimum days to expiration (default: 1)'
    )

    parser.add_argument(
        '--max-delta',
        type=float,
        default=0.30,
        help='Maximum delta for short options (default: 0.30)'
    )

    parser.add_argument(
        '--max-price',
        type=float,
        default=120.0,
        help='Maximum share price for symbols (default: $120)'
    )

    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top trades to display (default: 20)'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['csp', 'put_spread', 'call_spread'],
        default=['csp', 'put_spread', 'call_spread'],
        help='Strategies to use (default: csp put_spread call_spread)'
    )

    parser.add_argument(
        '--no-vol',
        action='store_true',
        help='Skip volatility surface analysis'
    )

    parser.add_argument(
        '--full-scan',
        action='store_true',
        help='Scan S&P 500 + Nasdaq 100 + ETFs (~600 symbols, slower but comprehensive)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - browse results'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    # Backtesting arguments
    backtest_group = parser.add_argument_group('Backtesting')

    backtest_group.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest mode instead of live analysis'
    )

    backtest_group.add_argument(
        '--start-date',
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )

    backtest_group.add_argument(
        '--end-date',
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )

    backtest_group.add_argument(
        '--profit-target',
        type=float,
        default=0.5,
        help='Exit at this fraction of max profit (default: 0.5 = 50%%)'
    )

    backtest_group.add_argument(
        '--stop-loss',
        type=float,
        default=2.0,
        help='Exit at this multiple of premium received as loss (default: 2.0 = 2x)'
    )

    backtest_group.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions in backtest (default: 5)'
    )

    return parser.parse_args()


def map_strategy_arg(arg: str) -> StrategyType:
    """Map CLI strategy argument to StrategyType enum."""
    mapping = {
        'csp': StrategyType.CASH_SECURED_PUT,
        'put_spread': StrategyType.PUT_CREDIT_SPREAD,
        'call_spread': StrategyType.CALL_CREDIT_SPREAD,
    }
    return mapping.get(arg)


def run_interactive(analyzer: OptionsAnalyzer, result):
    """Run interactive mode for browsing results."""
    while True:
        display_candidates_table(result.top_candidates, "Top Trade Candidates")

        choice = display_menu(result.top_candidates)

        if choice == -1:  # Quit
            console.print("[cyan]Goodbye![/cyan]")
            break
        elif choice == -2:  # Refresh
            console.print("[cyan]Refreshing data...[/cyan]")
            result = analyzer.run_analysis()
            continue
        elif choice is not None:
            candidate = result.top_candidates[choice]
            display_trade_detail(candidate)

            # Show volatility signals for this symbol
            signals = analyzer.get_volatility_signals(candidate.underlying_symbol)
            if signals:
                display_volatility_signals(signals)

            console.print("\nPress Enter to continue...")
            console.input()


def run_backtest(args):
    """Run backtest mode."""
    from backtesting import OptionBacktester, BacktestConfig
    from backtesting.trade_manager import ExitRules
    from core.config import TradeCriteria

    # Validate dates
    if not args.start_date or not args.end_date:
        console.print("[red]Error: --start-date and --end-date are required for backtesting[/red]")
        console.print("Example: python main.py --backtest --start-date 2023-01-01 --end-date 2024-01-01 -s SPY QQQ")
        sys.exit(1)

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        console.print("Use format: YYYY-MM-DD")
        sys.exit(1)

    if start_date >= end_date:
        console.print("[red]Error: start-date must be before end-date[/red]")
        sys.exit(1)

    # Get symbols
    symbols = args.symbols if args.symbols else ["SPY", "QQQ", "IWM"]

    # Map strategies
    strategy_types = [
        map_strategy_arg(s) for s in args.strategies
        if map_strategy_arg(s) is not None
    ]

    # Build config
    trade_criteria = TradeCriteria(
        min_prob_profit=args.prob,
        min_weekly_return_pct=args.min_return,
        max_dte=args.max_dte,
        min_dte=args.min_dte,
        max_delta=args.max_delta,
    )

    exit_rules = ExitRules(
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
    )

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=args.capital,
        strategy_types=strategy_types,
        trade_criteria=trade_criteria,
        exit_rules=exit_rules,
        max_positions=args.max_positions,
    )

    # Display header
    if not args.json:
        display_backtest_header(config)

    # Run backtest
    backtester = OptionBacktester(config)
    result = backtester.run(verbose=not args.json)

    # Output results
    if args.json:
        output_backtest_json(result)
    else:
        display_backtest_result(result)

    return result


def output_backtest_json(result):
    """Output backtest results as JSON."""
    import json

    def json_serializer(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializable")

    output = result.to_dict()
    output['trades'] = [
        {
            'id': t.id,
            'symbol': t.symbol,
            'strategy': t.strategy_type.value,
            'entry_date': t.entry_date.isoformat() if t.entry_date else None,
            'exit_date': t.exit_date.isoformat() if t.exit_date else None,
            'exit_reason': t.exit_reason.value if t.exit_reason else None,
            'premium': t.premium_received,
            'pnl': t.realized_pnl,
            'days_held': t.days_held,
        }
        for t in result.trades
    ]

    print(json.dumps(output, indent=2, default=json_serializer))


def output_json(result):
    """Output results as JSON."""
    import json

    def json_serializer(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    output = {
        'generated_at': result.generated_at.isoformat(),
        'summary': {
            'chains_analyzed': len(result.chains_analyzed),
            'total_opportunities': result.total_opportunities_found,
            'matching_criteria': result.opportunities_meeting_criteria,
            'capital_deployable': result.capital_deployable,
        },
        'top_candidates': [
            {
                'strategy': c.strategy_name,
                'symbol': c.underlying_symbol,
                'dte': c.dte,
                'expiration': c.expiration.isoformat() if c.expiration else None,
                'premium': c.premium_received,
                'collateral': c.collateral_required,
                'weekly_return': c.weekly_return,
                'prob_profit': c.prob_profit,
                'score': c.overall_score,
                'strikes': [leg.strike for leg in c.legs],
            }
            for c in result.top_candidates
        ],
        'errors': result.errors,
    }

    print(json.dumps(output, indent=2, default=json_serializer))


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Check for backtest mode
    if args.backtest:
        run_backtest(args)
        return

    # Build configuration
    config = AnalyzerConfig()
    config.capital.total_capital = args.capital
    config.trade_criteria.min_prob_profit = args.prob
    config.trade_criteria.min_weekly_return_pct = args.min_return
    config.trade_criteria.max_dte = args.max_dte
    config.trade_criteria.min_dte = args.min_dte
    config.trade_criteria.max_delta = args.max_delta
    config.underlyings.max_share_price = args.max_price
    config.top_n_trades = args.top

    # Map strategies
    config.enabled_strategies = [
        map_strategy_arg(s) for s in args.strategies
        if map_strategy_arg(s) is not None
    ]

    # Custom symbols
    if args.symbols:
        config.underlyings.default_symbols = args.symbols

    # Create analyzer
    analyzer = OptionsAnalyzer(config)

    # Display header (unless JSON output)
    if not args.json:
        display_header()
        console.print(f"[cyan]Capital:[/cyan] ${config.capital.total_capital:,.2f}")
        console.print(f"[cyan]Min Prob Profit:[/cyan] {config.trade_criteria.min_prob_profit:.0%}")
        console.print(f"[cyan]Min Weekly Return:[/cyan] {config.trade_criteria.min_weekly_return_pct:.1f}%")
        console.print(f"[cyan]Max DTE:[/cyan] {config.trade_criteria.max_dte}")
        console.print(f"[cyan]Max Share Price:[/cyan] ${config.underlyings.max_share_price:.0f}")
        console.print(f"[cyan]Strategies:[/cyan] {', '.join(args.strategies)}")
        if args.full_scan:
            console.print(f"[cyan]Scan Mode:[/cyan] [yellow]Full market scan (S&P 500 + Nasdaq 100 + ETFs)[/yellow]")
        else:
            console.print(f"[cyan]Symbols:[/cyan] {len(config.get_active_symbols())} from config list")
        console.print()

    # Run analysis
    if not args.json:
        status_msg = "[bold green]Full market scan in progress..." if args.full_scan else "[bold green]Analyzing options chains..."
        with console.status(status_msg) as status:
            result = analyzer.run_analysis(
                include_volatility_analysis=not args.no_vol,
                full_scan=args.full_scan
            )
    else:
        result = analyzer.run_analysis(
            include_volatility_analysis=not args.no_vol,
            full_scan=args.full_scan
        )

    # Output results
    if args.json:
        output_json(result)
    elif args.interactive:
        run_interactive(analyzer, result)
    else:
        # Standard output
        display_analysis_summary(result)

        # Show portfolio summary
        portfolio_summary = analyzer.position_sizer.get_portfolio_summary()
        display_portfolio_summary(portfolio_summary)

        # Show top candidates
        display_candidates_table(
            result.top_candidates,
            f"Top {len(result.top_candidates)} Trade Candidates",
            show_all_columns=args.verbose
        )

        # Show volatility insights for top symbols
        if not args.no_vol and result.top_candidates:
            console.print("\n[bold cyan]Volatility Insights:[/bold cyan]")
            seen_symbols = set()
            for candidate in result.top_candidates[:5]:
                if candidate.underlying_symbol not in seen_symbols:
                    signals = analyzer.get_volatility_signals(candidate.underlying_symbol)
                    if signals:
                        display_volatility_signals(signals)
                    seen_symbols.add(candidate.underlying_symbol)

        # Show errors if any
        display_errors(result.errors)

        # Hint for interactive mode
        console.print("\n[dim]Tip: Run with -i for interactive mode to explore trades in detail[/dim]")


if __name__ == '__main__':
    main()
