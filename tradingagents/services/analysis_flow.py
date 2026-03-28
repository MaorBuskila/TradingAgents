"""CLI and API share this path: normalize selections then ``AnalysisRunner.run()``."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from tradingagents.analysis_selections import normalize_analysis_selections


def run_cli_style_analysis(
    selections: Mapping[str, Any],
    *,
    callbacks: Optional[List[Any]] = None,
    on_message: Optional[Callable[..., None]] = None,
    on_tool_call: Optional[Callable[..., None]] = None,
    on_agent_status: Optional[Callable[..., None]] = None,
    on_report_section: Optional[Callable[..., None]] = None,
    on_complete: Optional[Callable[..., None]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Same graph run as ``cli/main.py`` ``run_analysis`` core; used by ``/api/analyze``."""
    from tradingagents.services.analysis_runner import AnalysisRunner

    normalized = normalize_analysis_selections(selections)
    runner = AnalysisRunner(
        selections=normalized,
        callbacks=callbacks or [],
        on_message=on_message,
        on_tool_call=on_tool_call,
        on_agent_status=on_agent_status,
        on_report_section=on_report_section,
        on_complete=on_complete,
    )
    return runner.run()
