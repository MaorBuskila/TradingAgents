"""normalize_analysis_selections + ordering match CLI/API contract."""

from tradingagents.analysis_selections import normalize_analysis_selections


class _FakeEnumAnalyst:
    value = "news"


def test_normalize_orders_analysts_and_coerces():
    raw = {
        "ticker": "spy",
        "analysis_date": "2026-01-15",
        "analysts": [_FakeEnumAnalyst(), "market"],
        "research_depth": "3",
        "llm_provider": "Google",
        "backend_url": "https://example.com",
        "shallow_thinker": "m",
        "deep_thinker": "d",
        "google_thinking_level": "high",
        "openai_reasoning_effort": None,
        "anthropic_effort": None,
    }
    out = normalize_analysis_selections(raw)
    assert out["analysts"] == ["market", "news"]
    assert out["ticker"] == "SPY"
    assert out["llm_provider"] == "google"
    assert out["research_depth"] == 3


def test_normalize_gui_style_strings():
    out = normalize_analysis_selections(
        {
            "ticker": "qqq",
            "analysis_date": "2026-01-01",
            "analysts": ["fundamentals", "market"],
            "research_depth": 1,
            "llm_provider": "openai",
            "backend_url": "https://api.openai.com/v1",
            "shallow_thinker": "gpt-4o-mini",
            "deep_thinker": "gpt-4o",
            "google_thinking_level": None,
            "openai_reasoning_effort": "medium",
            "anthropic_effort": None,
        }
    )
    assert out["analysts"] == ["market", "fundamentals"]
