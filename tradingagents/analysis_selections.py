"""Analysis selections normalization (no graph / langgraph imports)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

ANALYST_ORDER = ("market", "social", "news", "fundamentals")


def _analyst_key(a: Any) -> str:
    if hasattr(a, "value"):
        return str(a.value).lower().strip()
    return str(a).lower().strip()


def normalize_analysis_selections(selections: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a plain dict in the same shape as CLI ``get_user_selections()`` after coercion.

    - ``analysts``: strings in fixed ``ANALYST_ORDER`` (subset of selected keys)
    - ``ticker``: uppercased symbol
    - ``llm_provider``: lowercased
    - ``research_depth``: int
    """
    out: Dict[str, Any] = dict(selections)
    raw_analysts = out.get("analysts") or []
    keys = {_analyst_key(a) for a in raw_analysts}
    ordered = [k for k in ANALYST_ORDER if k in keys]
    if not ordered:
        raise ValueError("Select at least one analyst (market, social, news, or fundamentals).")
    out["analysts"] = ordered
    out["ticker"] = str(out.get("ticker", "")).strip().upper()
    out["analysis_date"] = str(out.get("analysis_date", "")).strip()
    out["llm_provider"] = str(out.get("llm_provider", "openai")).lower().strip()
    out["backend_url"] = str(out.get("backend_url", "")).strip()
    out["shallow_thinker"] = str(out.get("shallow_thinker", "")).strip()
    out["deep_thinker"] = str(out.get("deep_thinker", "")).strip()
    try:
        out["research_depth"] = int(out.get("research_depth", 1))
    except (TypeError, ValueError):
        out["research_depth"] = 1
    for key in ("google_thinking_level", "openai_reasoning_effort", "anthropic_effort"):
        v = out.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            out[key] = None
        else:
            out[key] = str(v).strip()
    return out
