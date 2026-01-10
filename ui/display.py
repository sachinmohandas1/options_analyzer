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
