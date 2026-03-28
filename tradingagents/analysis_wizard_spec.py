"""Analysis wizard copy and option lists shared by CLI (questionary) and API/GUI (JSON).

Keeps Step 1–7 wording and numeric enums aligned with ``cli/main.py`` interactive flow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Research depth → max_debate_rounds / max_risk_discuss_rounds (same int as CLI)
RESEARCH_DEPTH_OPTIONS: List[Tuple[str, int]] = [
    ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
    ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
    ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
]

# Display label, analyst key (normalized selection / API)
ANALYST_OPTIONS: List[Tuple[str, str]] = [
    ("Market Analyst", "market"),
    ("Social Media Analyst", "social"),
    ("News Analyst", "news"),
    ("Fundamentals Analyst", "fundamentals"),
]

GOOGLE_THINKING_OPTIONS: List[Tuple[str, str]] = [
    ("Enable Thinking (recommended)", "high"),
    ("Minimal/Disable Thinking", "minimal"),
]

OPENAI_REASONING_OPTIONS: List[Tuple[str, str]] = [
    ("Medium (Default)", "medium"),
    ("High (More thorough)", "high"),
    ("Low (Faster)", "low"),
]

ANTHROPIC_EFFORT_OPTIONS: List[Tuple[str, str]] = [
    ("High (recommended)", "high"),
    ("Medium (balanced)", "medium"),
    ("Low (faster, cheaper)", "low"),
]

# Panel titles + descriptions: same strings as ``cli/main.py`` create_question_box
WIZARD_STEPS: List[Dict[str, str]] = [
    {
        "id": "ticker",
        "title": "Step 1: Ticker Symbol",
        "description": (
            "Enter the exact ticker symbol to analyze, including exchange suffix when needed "
            "(examples: SPY, CNC.TO, 7203.T, 0700.HK)"
        ),
    },
    {
        "id": "analysis_date",
        "title": "Step 2: Analysis Date",
        "description": "Enter the analysis date (YYYY-MM-DD)",
    },
    {
        "id": "analysts",
        "title": "Step 3: Analysts Team",
        "description": "Select your LLM analyst agents for the analysis",
    },
    {
        "id": "research_depth",
        "title": "Step 4: Research Depth",
        "description": "Select your research depth level",
    },
    {
        "id": "llm_provider",
        "title": "Step 5: OpenAI backend",
        "description": "Select which service to talk to",
    },
    {
        "id": "thinking_models",
        "title": "Step 6: Thinking Agents",
        "description": "Select your thinking agents for analysis",
    },
]

# Step 7 varies by provider (CLI branches on provider_lower)
STEP_7_BY_PROVIDER: Dict[str, Dict[str, str]] = {
    "google": {
        "title": "Step 7: Thinking Mode",
        "description": "Configure Gemini thinking mode",
        "prompt": "Select Thinking Mode:",
    },
    "openai": {
        "title": "Step 7: Reasoning Effort",
        "description": "Configure OpenAI reasoning effort level",
        "prompt": "Select Reasoning Effort:",
    },
    "anthropic": {
        "title": "Step 7: Effort Level",
        "description": "Configure Claude effort level",
        "prompt": "Select Effort Level:",
    },
}

# Questionary primary lines (for GUI field labels)
PROMPTS: Dict[str, str] = {
    "ticker": "Enter the exact ticker symbol to analyze",
    "research_depth": "Select Your [Research Depth]:",
    "analysts": "Select Your [Analysts Team]:",
    "llm_provider": "Select your LLM Provider:",
    "quick_thinker": "Select Your [Quick-Thinking LLM Engine]:",
    "deep_thinker": "Select Your [Deep-Thinking LLM Engine]:",
}

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"


def wizard_payload_for_api() -> Dict[str, Any]:
    """JSON-safe spec consumed by the Run Analysis GUI."""
    thinking_options: Dict[str, List[Dict[str, str]]] = {
        "google": [{"label": a, "value": b} for a, b in GOOGLE_THINKING_OPTIONS],
        "openai": [{"label": a, "value": b} for a, b in OPENAI_REASONING_OPTIONS],
        "anthropic": [{"label": a, "value": b} for a, b in ANTHROPIC_EFFORT_OPTIONS],
    }
    return {
        "steps": list(WIZARD_STEPS),
        "step7_by_provider": STEP_7_BY_PROVIDER,
        "prompts": dict(PROMPTS),
        "ticker_examples": TICKER_INPUT_EXAMPLES,
        "research_depth_options": [{"label": lab, "value": val} for lab, val in RESEARCH_DEPTH_OPTIONS],
        "analyst_options": [{"label": lab, "value": val} for lab, val in ANALYST_OPTIONS],
        "thinking_options_by_provider": thinking_options,
    }
