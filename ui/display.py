"""
Display module using Rich for beautiful terminal output.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich import box

from core.models import TradeCandidate, AnalysisResult, VolatilitySurface
from analysis.position_sizer import PositionAllocation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.performance import BacktestResult, PerformanceMetrics
    from backtesting.backtester import BacktestConfig


console = Console()


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def color_by_value(value: float, threshold: float = 0, invert: bool = False) -> str:
    """Return color based on value."""
    if invert:
        return "red" if value > threshold else "green"
    return "green" if value > threshold else "red"


def display_header():
    """Display application header."""
    header = Panel(
        Text("OPTIONS ANALYZER", style="bold cyan", justify="center"),
        subtitle=f"Analysis run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        box=box.DOUBLE
    )
    console.print(header)
    console.print()


def display_portfolio_summary(summary: Dict[str, Any]):
    """Display portfolio/capital summary."""
    table = Table(title="Portfolio Summary", box=box.ROUNDED)

    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Capital", format_currency(summary['total_capital']))
    table.add_row("Deployed Capital", format_currency(summary['deployed_capital']))
    table.add_row("Available Capital", format_currency(summary['available_capital']))
    table.add_row("Reserved Capital", format_currency(summary['reserved_capital']))
    table.add_row("Utilization", f"{summary['utilization_pct']:.1f}%")
    table.add_row("Total Positions", str(summary['total_positions']))

    console.print(table)
    console.print()


def display_candidates_table(
    candidates: List[TradeCandidate],
    title: str = "Trade Candidates",
    show_all_columns: bool = False
):
    """Display trade candidates in a formatted table."""
    if not candidates:
        console.print(f"[yellow]No {title.lower()} found matching criteria.[/yellow]")
        return

    table = Table(title=title, box=box.ROUNDED, show_lines=True)

    # Core columns
    table.add_column("#", style="dim", width=3)
    table.add_column("Strategy", style="cyan", width=18)
    table.add_column("Symbol", style="bold", width=6)
    table.add_column("DTE", justify="right", width=4)
    table.add_column("Strike(s)", width=12)
    table.add_column("Premium", justify="right", width=10)
    table.add_column("Collateral", justify="right", width=10)
    table.add_column("Weekly Ret", justify="right", width=10)
    table.add_column("Prob Profit", justify="right", width=10)
    table.add_column("Score", justify="right", width=6)

    if show_all_columns:
        table.add_column("Delta", justify="right", width=7)
        table.add_column("Theta", justify="right", width=7)
        table.add_column("IV Rank", justify="right", width=8)

    for i, c in enumerate(candidates, 1):
        # Format strikes
        if len(c.legs) == 1:
            strikes = f"{c.legs[0].strike:.0f}"
        elif len(c.legs) == 2:
            strikes = f"{c.legs[0].strike:.0f}/{c.legs[1].strike:.0f}"
        else:
            strikes = "/".join([f"{leg.strike:.0f}" for leg in c.legs])

        # Color weekly return
        weekly_ret = c.weekly_return
        ret_color = "green" if weekly_ret >= 0.01 else "yellow" if weekly_ret >= 0.005 else "red"
        weekly_ret_str = f"[{ret_color}]{format_percent(weekly_ret)}[/{ret_color}]"

        # Color probability
        prob_color = "green" if c.prob_profit >= 0.70 else "yellow" if c.prob_profit >= 0.60 else "red"
        prob_str = f"[{prob_color}]{format_percent(c.prob_profit, 1)}[/{prob_color}]"

        # Color score
        score_color = "green" if c.overall_score >= 60 else "yellow" if c.overall_score >= 40 else "white"
        score_str = f"[{score_color}]{c.overall_score:.0f}[/{score_color}]"

        row = [
            str(i),
            c.strategy_name,
            c.underlying_symbol,
            str(c.dte),
            strikes,
            format_currency(c.premium_received),
            format_currency(c.collateral_required),
            weekly_ret_str,
            prob_str,
            score_str,
        ]

        if show_all_columns:
            row.extend([
                f"{c.net_delta:.3f}",
                f"${c.net_theta:.2f}",
                f"{c.iv_rank:.0f}%" if c.iv_rank else "N/A",
            ])

        table.add_row(*row)

    console.print(table)
    console.print()


def display_trade_detail(candidate: TradeCandidate, allocation: Optional[PositionAllocation] = None):
    """Display detailed view of a single trade."""
    console.print()

    # Header
    title = f"{candidate.strategy_name} - {candidate.underlying_symbol}"
    console.print(Panel(title, style="bold cyan"))

    # Trade structure
    console.print("[bold]Trade Structure:[/bold]")
    for i, leg in enumerate(candidate.legs, 1):
        action = "SELL" if i % 2 == 1 else "BUY"  # Odd = short, Even = long for spreads
        leg_type = "Put" if leg.option_type.value == "put" else "Call"
        console.print(f"  Leg {i}: {action} {leg.strike:.0f} {leg_type} @ ${leg.bid:.2f}/{leg.ask:.2f}")

    console.print()

    # Metrics
    metrics_table = Table(box=box.SIMPLE)
    metrics_table.add_column("Metric", style="cyan", width=20)
    metrics_table.add_column("Value", justify="right", width=15)

    metrics_table.add_row("Expiration", candidate.expiration.strftime('%Y-%m-%d') if candidate.expiration else "N/A")
    metrics_table.add_row("Days to Expiry", str(candidate.dte))
    metrics_table.add_row("Premium Received", format_currency(candidate.premium_received))
    metrics_table.add_row("Collateral Required", format_currency(candidate.collateral_required))
    metrics_table.add_row("Max Profit", format_currency(candidate.max_profit))
    metrics_table.add_row("Max Loss", format_currency(candidate.max_loss))
    metrics_table.add_row("Breakeven", f"${candidate.breakeven:.2f}")
    if candidate.breakeven_upper:
        metrics_table.add_row("Breakeven Upper", f"${candidate.breakeven_upper:.2f}")

    console.print(metrics_table)

    # Returns
    returns_table = Table(title="Returns", box=box.SIMPLE)
    returns_table.add_column("Metric", style="cyan", width=20)
    returns_table.add_column("Value", justify="right", width=15)

    returns_table.add_row("Return on Collateral", format_percent(candidate.return_on_collateral))
    returns_table.add_row("Weekly Return", format_percent(candidate.weekly_return))
    returns_table.add_row("Annualized Return", format_percent(candidate.annualized_return))
    returns_table.add_row("Return on Risk", format_percent(candidate.return_on_risk))

    console.print(returns_table)

    # Probabilities
    prob_table = Table(title="Probabilities", box=box.SIMPLE)
    prob_table.add_column("Metric", style="cyan", width=20)
    prob_table.add_column("Value", justify="right", width=15)

    prob_table.add_row("Prob of Profit", format_percent(candidate.prob_profit))
    prob_table.add_row("Prob of Max Profit", format_percent(candidate.prob_max_profit))
    prob_table.add_row("Expected Value", format_currency(candidate.expected_value))

    console.print(prob_table)

    # Greeks
    greeks_table = Table(title="Greeks", box=box.SIMPLE)
    greeks_table.add_column("Greek", style="cyan", width=12)
    greeks_table.add_column("Value", justify="right", width=12)

    greeks_table.add_row("Delta", f"{candidate.net_delta:.4f}")
    greeks_table.add_row("Gamma", f"{candidate.net_gamma:.4f}")
    greeks_table.add_row("Theta", f"${candidate.net_theta:.2f}/day")
    greeks_table.add_row("Vega", f"${candidate.net_vega:.2f}")

    console.print(greeks_table)

    # Allocation if provided
    if allocation:
        console.print()
        alloc_table = Table(title="Position Sizing", box=box.SIMPLE)
        alloc_table.add_column("Metric", style="cyan", width=20)
        alloc_table.add_column("Value", justify="right", width=15)

        alloc_table.add_row("Contracts", str(allocation.contracts))
        alloc_table.add_row("Total Collateral", format_currency(allocation.collateral_required))
        alloc_table.add_row("Total Max Profit", format_currency(allocation.max_profit))
        alloc_table.add_row("Total Max Loss", format_currency(allocation.max_loss))
        alloc_table.add_row("Capital Utilization", f"{allocation.capital_utilization_pct:.1f}%")

        console.print(alloc_table)


def display_volatility_signals(signals: Dict[str, Any]):
    """Display volatility surface analysis signals."""
    console.print()
    console.print(Panel(f"Volatility Analysis - {signals['underlying']}", style="bold magenta"))

    # IV metrics
    if signals.get('iv_rank') is not None:
        iv_rank = signals['iv_rank']
        rank_color = "green" if iv_rank > 50 else "yellow" if iv_rank > 25 else "red"
        console.print(f"IV Rank: [{rank_color}]{iv_rank:.1f}%[/{rank_color}]")

    if signals.get('iv_percentile') is not None:
        console.print(f"IV Percentile: {signals['iv_percentile']:.1f}%")

    # Term structure
    term = signals.get('term_structure', {})
    if term:
        structure_type = "Contango" if term.get('is_contango') else "Backwardation" if term.get('is_backwardation') else "Flat"
        console.print(f"Term Structure: {structure_type}")
        console.print(f"Signal: {term.get('signal', 'N/A')}")

    # Skew
    skew_data = signals.get('skew', {})
    if skew_data:
        console.print("\n[bold]Skew by Expiration:[/bold]")
        for exp, data in skew_data.items():
            skew_val = data.get('skew_25d')
            if skew_val:
                console.print(f"  {exp} ({data['dte']}DTE): Skew={skew_val:.2%}, Signal={data['signal']}")

    # Recommendations
    recommendations = signals.get('recommendations', [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            action_color = "green" if rec['action'] == 'SELL_PREMIUM' else "yellow" if rec['action'] == 'CAUTION' else "cyan"
            console.print(f"  [{action_color}]{rec['action']}[/{action_color}]: {rec['reason']}")

    # Anomalies
    anomalies = signals.get('anomalies', [])
    if anomalies:
        console.print("\n[bold yellow]Anomalies Detected:[/bold yellow]")
        for anom in anomalies[:5]:  # Show top 5
            console.print(f"  {anom['expiration']}: Strike {anom['strike']}, IV={anom['iv']:.2%}, Z={anom['zscore']:.2f}")


def display_analysis_summary(result: AnalysisResult):
    """Display summary of complete analysis run."""
    console.print()
    summary_table = Table(title="Analysis Summary", box=box.ROUNDED)

    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Chains Analyzed", str(len(result.chains_analyzed)))
    summary_table.add_row("Total Opportunities", str(result.total_opportunities_found))
    summary_table.add_row("Matching Criteria", str(result.opportunities_meeting_criteria))
    summary_table.add_row("Top Recommendations", str(len(result.top_candidates)))
    summary_table.add_row("Deployable Capital", format_currency(result.capital_deployable))

    if result.errors:
        summary_table.add_row("Errors", f"[red]{len(result.errors)}[/red]")

    console.print(summary_table)


def display_errors(errors: List[Dict[str, str]]):
    """Display any errors encountered during analysis."""
    if not errors:
        return

    console.print()
    console.print("[bold red]Errors Encountered:[/bold red]")
    for err in errors:
        console.print(f"  [red]{err.get('symbol', 'Unknown')}: {err.get('error', 'Unknown error')}[/red]")


def display_menu(candidates: List[TradeCandidate]) -> Optional[int]:
    """Display interactive menu for selecting trades."""
    console.print()
    console.print("[bold]Enter trade number for details, 'q' to quit, 'r' to refresh:[/bold]")

    try:
        choice = console.input("> ").strip().lower()

        if choice == 'q':
            return -1
        if choice == 'r':
            return -2

        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return idx
        else:
            console.print("[red]Invalid selection[/red]")
            return None
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return None


def display_backtest_header(config: "BacktestConfig"):
    """Display backtest configuration header."""
    header = Panel(
        Text("BACKTEST MODE", style="bold yellow", justify="center"),
        subtitle=f"Simulation: {config.start_date} to {config.end_date}",
        box=box.DOUBLE
    )
    console.print(header)
    console.print()

    console.print(f"[cyan]Symbols:[/cyan] {', '.join(config.symbols)}")
    console.print(f"[cyan]Initial Capital:[/cyan] ${config.initial_capital:,.2f}")
    console.print(f"[cyan]Strategies:[/cyan] {[s.value for s in config.strategy_types]}")
    console.print(f"[cyan]Profit Target:[/cyan] {config.exit_rules.profit_target_pct:.0%}")
    console.print(f"[cyan]Stop Loss:[/cyan] {config.exit_rules.stop_loss_pct:.1f}x premium")
    console.print(f"[cyan]Max Positions:[/cyan] {config.max_positions}")
    console.print()


def display_backtest_result(result: "BacktestResult"):
    """Display comprehensive backtest results."""
    console.print()

    # Performance summary panel
    metrics = result.metrics
    return_color = "green" if metrics.total_return_pct > 0 else "red"

    summary_panel = Panel(
        f"[bold {return_color}]Total Return: {metrics.total_return_pct:.2f}%[/bold {return_color}]\n"
        f"Final Capital: ${result.final_capital:,.2f}",
        title="Backtest Complete",
        box=box.DOUBLE
    )
    console.print(summary_panel)
    console.print()

    # Key metrics table
    metrics_table = Table(title="Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan", width=25)
    metrics_table.add_column("Value", justify="right", width=15)

    metrics_table.add_row("Total Trades", str(metrics.total_trades))
    metrics_table.add_row("Winning Trades", f"[green]{metrics.winning_trades}[/green]")
    metrics_table.add_row("Losing Trades", f"[red]{metrics.losing_trades}[/red]")
    metrics_table.add_row("Win Rate", f"{metrics.win_rate:.1%}")
    metrics_table.add_row("", "")  # Separator

    metrics_table.add_row("Total P&L", format_currency(metrics.total_pnl))
    metrics_table.add_row("Gross Profit", f"[green]{format_currency(metrics.gross_profit)}[/green]")
    metrics_table.add_row("Gross Loss", f"[red]{format_currency(metrics.gross_loss)}[/red]")
    metrics_table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "Inf")
    metrics_table.add_row("Expectancy", format_currency(metrics.expectancy))
    metrics_table.add_row("", "")

    metrics_table.add_row("Avg Win", format_currency(metrics.avg_win))
    metrics_table.add_row("Avg Loss", format_currency(metrics.avg_loss))
    metrics_table.add_row("Largest Win", f"[green]{format_currency(metrics.largest_win)}[/green]")
    metrics_table.add_row("Largest Loss", f"[red]{format_currency(metrics.largest_loss)}[/red]")
    metrics_table.add_row("", "")

    metrics_table.add_row("Max Drawdown", f"[red]{metrics.max_drawdown_pct:.2f}%[/red]")
    metrics_table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    metrics_table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    metrics_table.add_row("", "")

    metrics_table.add_row("Avg Days Held", f"{metrics.avg_days_held:.1f}")
    metrics_table.add_row("Annualized Return", f"{metrics.annualized_return_pct:.2f}%")

    console.print(metrics_table)
    console.print()

    # Exit reason breakdown
    if metrics.exits_by_reason:
        exit_table = Table(title="Exit Reasons", box=box.ROUNDED)
        exit_table.add_column("Reason", style="cyan", width=20)
        exit_table.add_column("Count", justify="right", width=10)
        exit_table.add_column("P&L", justify="right", width=15)

        for reason, count in metrics.exits_by_reason.items():
            pnl = metrics.pnl_by_exit_reason.get(reason, 0)
            pnl_color = "green" if pnl > 0 else "red"
            exit_table.add_row(
                reason.replace("_", " ").title(),
                str(count),
                f"[{pnl_color}]{format_currency(pnl)}[/{pnl_color}]"
            )

        console.print(exit_table)
        console.print()

    # Strategy breakdown
    if result.metrics_by_strategy:
        strategy_table = Table(title="Performance by Strategy", box=box.ROUNDED)
        strategy_table.add_column("Strategy", style="cyan", width=20)
        strategy_table.add_column("Trades", justify="right", width=8)
        strategy_table.add_column("Win Rate", justify="right", width=10)
        strategy_table.add_column("P&L", justify="right", width=12)
        strategy_table.add_column("Avg Trade", justify="right", width=10)

        for strategy, strat_metrics in result.metrics_by_strategy.items():
            pnl_color = "green" if strat_metrics.total_pnl > 0 else "red"
            strategy_table.add_row(
                strategy.replace("_", " ").title(),
                str(strat_metrics.total_trades),
                f"{strat_metrics.win_rate:.1%}",
                f"[{pnl_color}]{format_currency(strat_metrics.total_pnl)}[/{pnl_color}]",
                format_currency(strat_metrics.expectancy)
            )

        console.print(strategy_table)
        console.print()

    # Symbol breakdown (top 5)
    if result.metrics_by_symbol:
        symbol_table = Table(title="Performance by Symbol (Top 10)", box=box.ROUNDED)
        symbol_table.add_column("Symbol", style="cyan", width=10)
        symbol_table.add_column("Trades", justify="right", width=8)
        symbol_table.add_column("Win Rate", justify="right", width=10)
        symbol_table.add_column("P&L", justify="right", width=12)

        # Sort by P&L
        sorted_symbols = sorted(
            result.metrics_by_symbol.items(),
            key=lambda x: x[1].total_pnl,
            reverse=True
        )[:10]

        for symbol, sym_metrics in sorted_symbols:
            pnl_color = "green" if sym_metrics.total_pnl > 0 else "red"
            symbol_table.add_row(
                symbol,
                str(sym_metrics.total_trades),
                f"{sym_metrics.win_rate:.1%}",
                f"[{pnl_color}]{format_currency(sym_metrics.total_pnl)}[/{pnl_color}]"
            )

        console.print(symbol_table)
        console.print()

    # Recent trades
    if result.trades:
        trades_table = Table(title="Recent Trades (Last 10)", box=box.ROUNDED)
        trades_table.add_column("Date", width=12)
        trades_table.add_column("Symbol", width=8)
        trades_table.add_column("Strategy", width=18)
        trades_table.add_column("Exit", width=14)
        trades_table.add_column("Days", justify="right", width=5)
        trades_table.add_column("P&L", justify="right", width=12)

        for trade in result.trades[-10:]:
            pnl = trade.realized_pnl or 0
            pnl_color = "green" if pnl > 0 else "red"
            exit_reason = trade.exit_reason.value.replace("_", " ").title() if trade.exit_reason else "Open"

            trades_table.add_row(
                trade.entry_date.strftime("%Y-%m-%d") if trade.entry_date else "N/A",
                trade.symbol,
                trade.strategy_name[:18],
                exit_reason,
                str(trade.days_held),
                f"[{pnl_color}]{format_currency(pnl)}[/{pnl_color}]"
            )

        console.print(trades_table)
        console.print()

    # Errors
    if result.errors:
        console.print(f"[yellow]Errors during backtest: {len(result.errors)}[/yellow]")
        for err in result.errors[:5]:
            console.print(f"  [dim]{err.get('date', 'N/A')}: {err.get('symbol', 'N/A')} - {err.get('error', 'Unknown')}[/dim]")

    console.print(f"\n[dim]Backtest completed in {result.duration_seconds:.1f} seconds[/dim]")
