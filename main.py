#!/usr/bin/env python3
"""
Options Analyzer CLI - Main Entry Point

A tool for scanning options chains and identifying high-probability
premium selling opportunities.
"""

import argparse
import logging
import sys
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


def output_json(result):
    """Output results as JSON."""
    import json
    from datetime import date

    def json_serializer(obj):
        if isinstance(obj, (date,)):
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
