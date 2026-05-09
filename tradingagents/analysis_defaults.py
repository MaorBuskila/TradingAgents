"""Non-interactive analysis defaults: same provider-specific knobs as CLI step 7.

CLI uses questionary; the first/recommended choice per provider is mirrored here for
API / Telegram / GUI bootstrap.
"""

from __future__ import annotations

import os
from typing import Any


def provider_thinking_defaults(llm_provider: str) -> dict[str, str | None]:
    """Match cli/utils.py ask_* defaults (first option in each prompt)."""
    p = (llm_provider or "openai").lower()
    google_thinking_level = None
    openai_reasoning_effort = None
    anthropic_effort = None
    if p == "google":
        google_thinking_level = "high"
    elif p == "openai":
        openai_reasoning_effort = "medium"
    elif p == "anthropic":
        anthropic_effort = "high"
    return {
        "google_thinking_level": google_thinking_level,
        "openai_reasoning_effort": openai_reasoning_effort,
        "anthropic_effort": anthropic_effort,
    }


def infer_llm_defaults_from_env() -> dict[str, Any]:
    """Pick provider/models from env keys (same priority as former api infer_analysis_defaults)."""
    if os.getenv("OPENAI_API_KEY", "").strip():
        base = {
            "llm_provider": "openai",
            "backend_url": "https://api.openai.com/v1",
            "shallow_thinker": "gpt-4o-mini",
            "deep_thinker": "gpt-4o",
        }
    elif os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip():
        base = {
            "llm_provider": "google",
            "shallow_thinker": "gemini-2.5-flash",
            "deep_thinker": "gemini-2.5-pro",
        }
    elif os.getenv("ANTHROPIC_API_KEY", "").strip():
        base = {
            "llm_provider": "anthropic",
            "backend_url": "https://api.anthropic.com/",
            "shallow_thinker": "claude-sonnet-4-6",
            "deep_thinker": "claude-opus-4-6",
        }
    elif os.getenv("XAI_API_KEY", "").strip():
        base = {
            "llm_provider": "xai",
            "backend_url": "https://api.x.ai/v1",
            "shallow_thinker": "grok-4-fast-non-reasoning",
            "deep_thinker": "grok-4-1-fast-reasoning",
        }
    elif os.getenv("OPENROUTER_API_KEY", "").strip():
        base = {
            "llm_provider": "openrouter",
            "backend_url": "https://openrouter.ai/api/v1",
            "shallow_thinker": "openai/gpt-4o-mini",
            "deep_thinker": "openai/gpt-4o",
        }
    else:
        base = {
            "llm_provider": "openai",
            "backend_url": "https://api.openai.com/v1",
            "shallow_thinker": "gpt-4o-mini",
            "deep_thinker": "gpt-4o",
        }

    p = base["llm_provider"]
    base["research_depth"] = 1
    base.update(provider_thinking_defaults(p))
    return base
