"""
rsi_dt_report.py
----------------
Human-readable Rich terminal report for run_rsi_dt_optimizer() output.

Usage:
    from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import run_rsi_dt_optimizer
    from tradingagents.quant_ml.optimizers.rsi_dt_report import pretty_print_rsi_dt

    result = run_rsi_dt_optimizer("NVDA", "2026-04-17")
    pretty_print_rsi_dt(result)
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

_CONF_COLOR = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}


def _sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo or 1e-9
    bars = [_SPARK_CHARS[int((v - lo) / rng * (len(_SPARK_CHARS) - 1))] for v in values]
    # trim to width
    if len(bars) > width:
        step = len(bars) / width
        bars = [bars[int(i * step)] for i in range(width)]
    return "".join(bars)


def _prob_bar(prob: float, threshold: float, width: int = 20) -> str:
    filled = int(prob * width)
    thresh_pos = int(threshold * width)
    bar = ["─"] * width
    for i in range(filled):
        bar[i] = "█"
    if 0 <= thresh_pos < width:
        bar[thresh_pos] = "┃"
    return "".join(bar)


def pretty_print_rsi_dt(result: dict[str, Any], *, console: Console | None = None) -> None:
    """Render a human-readable Rich report for an rsi_dt_optimizer result dict."""
    con = console or Console()

    symbol = result.get("symbol", "?")
    curr_date = result.get("curr_date", "?")
    cache_hit = result.get("cache_hit", False)
    last_signal = result.get("last_signal", 0)
    conf = result.get("confidence", "LOW")
    conf_color = _CONF_COLOR.get(conf, "white")

    # ── Header ────────────────────────────────────────────────────────────────
    cache_badge = "[dim](cached)[/dim]" if cache_hit else "[dim](fresh WFO)[/dim]"
    signal_text = Text()
    if last_signal == 1:
        signal_text.append("  ▶  BUY  ◀  ", style="bold white on green")
    else:
        signal_text.append("  ▬  HOLD  ▬  ", style="bold white on dark_orange")

    header = Panel(
        signal_text,
        title=f"[bold cyan]RSI-DT · {symbol}[/bold cyan]  {curr_date}  {cache_badge}",
        subtitle=f"Confidence: [{conf_color}]{conf}[/{conf_color}]",
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    )
    con.print(header)

    # ── Model signals table ───────────────────────────────────────────────────
    prob_a = result.get("last_prob_a", 0.0)
    prob_b = result.get("last_prob_b", 0.0)
    thr_a = result.get("threshold_a", 0.0)
    thr_b = result.get("threshold_b", 0.0)
    rsi = result.get("last_rsi", 0.0)
    atr_pct = result.get("last_atr_pct", 0.0)
    size_mult = result.get("last_size_mult", 1.0)
    sentiment = result.get("last_sentiment")

    def _pass_fail(val: float, thr: float) -> str:
        return "[green]PASS[/green]" if val > thr else "[red]FAIL[/red]"

    sig_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    sig_tbl.add_column("Metric", style="dim", min_width=16)
    sig_tbl.add_column("Value", justify="right", min_width=8)
    sig_tbl.add_column("vs Threshold", min_width=26)
    sig_tbl.add_column("Gate", justify="center", min_width=6)

    sig_tbl.add_row(
        "RSI(14)",
        f"{rsi:.2f}",
        "[yellow]overbought >70[/yellow]" if rsi > 70 else ("[cyan]oversold <30[/cyan]" if rsi < 30 else "neutral"),
        "",
    )
    sig_tbl.add_row(
        "Prob A (sliding)",
        f"{prob_a:.4f}",
        f"{_prob_bar(prob_a, thr_a)} thr={thr_a:.4f}",
        _pass_fail(prob_a, thr_a),
    )
    sig_tbl.add_row(
        "Prob B (fixed)",
        f"{prob_b:.4f}",
        f"{_prob_bar(prob_b, thr_b)} thr={thr_b:.4f}",
        _pass_fail(prob_b, thr_b),
    )

    price_today = result.get("last_atr_norm", 0.0)  # proxy — macro gate shown separately
    macro_gate = result.get("last_signal", 0) == 1 or (
        prob_a > thr_a and prob_b > thr_b
    )
    sig_tbl.add_row(
        "ATR %ile (vol)",
        f"{atr_pct:.1f}%",
        "[red]high-vol → half size[/red]" if atr_pct > 80 else "normal vol",
        f"size×{size_mult}",
    )

    sent_val = f"{sentiment:.3f}" if sentiment is not None else "N/A"
    sent_gate = (
        "[green]PASS[/green]" if sentiment is not None and sentiment > 0.0
        else ("[dim]SKIP[/dim]" if sentiment is None else "[red]FAIL[/red]")
    )
    sig_tbl.add_row("Sentiment (FinBERT)", sent_val, "positive > 0.0", sent_gate)

    con.print(Panel(sig_tbl, title="[bold]Model Signals[/bold]", box=box.ROUNDED, padding=(0, 1)))

    # ── WFO summary table ─────────────────────────────────────────────────────
    wfo_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    wfo_tbl.add_column("Metric", style="dim", min_width=20)
    wfo_tbl.add_column("Value", justify="right", min_width=10)

    wfo_tbl.add_row("OOS Sharpe", f"{result.get('oos_sharpe', 0.0):.3f}")
    wfo_tbl.add_row("OOS Hit Rate", f"{result.get('oos_hit_rate', 0.0):.1%}")
    wfo_tbl.add_row("OOS Trade Count", str(result.get("oos_trade_count", 0)))
    wfo_tbl.add_row("Avg OOS Accuracy", f"{result.get('avg_oos_acc', 0.0):.1%}")
    wfo_tbl.add_row("WFO Folds (n_slides)", str(result.get("n_slides", 0)))
    wfo_tbl.add_row(
        "Confidence",
        f"[{conf_color}]{conf}[/{conf_color}]",
    )

    rsi_p = result.get("rsi_params", {})
    rsi_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    rsi_tbl.add_column("RSI Param", style="dim")
    rsi_tbl.add_column("Value", justify="right")
    rsi_tbl.add_row("Period", str(rsi_p.get("period", 14)))
    rsi_tbl.add_row("Upper (OB)", str(rsi_p.get("upper", 70.0)))
    rsi_tbl.add_row("Lower (OS)", str(rsi_p.get("lower", 30.0)))
    nondefault = result.get("rsi_params_nondefault", False)
    rsi_tbl.add_row("Non-default", "[yellow]yes[/yellow]" if nondefault else "no")

    con.print(Columns([
        Panel(wfo_tbl, title="[bold]WFO Summary[/bold]", box=box.ROUNDED, padding=(0, 1)),
        Panel(rsi_tbl, title="[bold]RSI Params[/bold]", box=box.ROUNDED, padding=(0, 1)),
    ]))

    # ── Slides sparkline ──────────────────────────────────────────────────────
    slides = result.get("slides", [])
    if slides:
        oos_accs = [s.get("oos_acc", 0.0) for s in slides]
        avg_acc = sum(oos_accs) / len(oos_accs)
        spark = _sparkline(oos_accs)
        con.print(Panel(
            f"[dim]OOS acc per fold ({len(slides)} folds, avg={avg_acc:.1%}):[/dim]\n{spark}",
            title="[bold]Fold Accuracy Sparkline[/bold]",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
