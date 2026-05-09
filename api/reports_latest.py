"""Resolve latest portfolio decision markdown for a ticker across reports/ and results/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LatestDecision:
    report_id: str
    date_sort: str
    content: str
    source: str


def _parse_report_header_ticker(first_line: str) -> str | None:
    m = re.match(r"^#\s+Trading Analysis Report:\s+(\S+)\s*$", first_line.strip())
    return m.group(1).upper() if m else None


def _sort_key_form_dir(name: str) -> str:
    """<TICKER>_YYYYMMDD_HHMMSS -> comparable datetime string; unknown shapes sort low."""
    m = re.match(r"^[A-Z0-9.^=\-]+_(\d{8})_(\d{6})$", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return "00000000000000"


def _collect_from_reports_dir(reports_root: Path, ticker_u: str) -> list[LatestDecision]:
    out: list[LatestDecision] = []
    if not reports_root.is_dir():
        return out
    for report_dir in reports_root.iterdir():
        if not report_dir.is_dir():
            continue
        complete = report_dir / "complete_report.md"
        decision = report_dir / "5_portfolio" / "decision.md"
        if not complete.is_file():
            continue
        try:
            with open(complete, encoding="utf-8") as f:
                first = f.readline()
        except OSError:
            continue
        t = _parse_report_header_ticker(first)
        if t != ticker_u:
            continue
        if not decision.is_file():
            continue
        try:
            content = decision.read_text(encoding="utf-8")
        except OSError:
            continue
        sk = _sort_key_form_dir(report_dir.name)
        out.append(
            LatestDecision(
                report_id=report_dir.name,
                date_sort=sk,
                content=content,
                source="reports",
            )
        )
    return out


def _collect_from_results_dir(results_root: Path, ticker_u: str) -> list[LatestDecision]:
    out: list[LatestDecision] = []
    ticker_dir = results_root / ticker_u
    if not ticker_dir.is_dir():
        return out
    for date_dir in ticker_dir.iterdir():
        if not date_dir.is_dir():
            continue
        path = date_dir / "reports" / "final_trade_decision.md"
        if not path.is_file():
            continue
        date_part = date_dir.name
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        norm = date_part.replace("-", "")[:8].ljust(8, "0")
        out.append(
            LatestDecision(
                report_id=f"{ticker_u}_{date_part}",
                date_sort=norm,
                content=content,
                source="results",
            )
        )
    return out


def get_latest_decision_for_ticker(ticker: str, repo_root: Path | None = None) -> LatestDecision | None:
    ticker_u = ticker.upper().strip()
    root = repo_root or Path.cwd()
    candidates = _collect_from_reports_dir(root / "reports", ticker_u) + _collect_from_results_dir(
        root / "results", ticker_u
    )
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.date_sort, reverse=True)
    return candidates[0]
