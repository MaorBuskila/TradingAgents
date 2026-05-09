"""LangChain-style invoke on chat models from ``create_llm_client`` (same stack as ``TradingAgentsGraph``)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.factory import provider_kwargs_for_provider
from tradingagents.dataflows.config import get_config


def _strip_json_fences(text: str) -> str:
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


def _response_token_usage(response, provider_lc: str) -> tuple[int, int, list]:
    meta = getattr(response, "response_metadata", None) or {}
    meta_keys = list(meta.keys()) if meta else []

    if provider_lc in ("openai", "xai", "openrouter", "ollama"):
        usage = meta.get("token_usage") or {}
        return (
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
            meta_keys,
        )

    if provider_lc == "anthropic":
        usage = meta.get("usage") or {}
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            meta_keys,
        )

    if provider_lc == "google":
        usage = (
            meta.get("usage_metadata")
            or meta.get("usageMetadata")
            or {}
        )
        pt = int(
            usage.get("prompt_token_count")
            or usage.get("promptTokenCount")
            or 0
        )
        ct = int(
            usage.get("candidates_token_count")
            or usage.get("candidatesTokenCount")
            or 0
        )
        if pt == 0 and ct == 0:
            um = getattr(response, "usage_metadata", None)
            if um:
                pt = int(getattr(um, "prompt_token_count", 0) or 0)
                ct = int(getattr(um, "candidates_token_count", 0) or 0)
        return pt, ct, meta_keys

    return 0, 0, meta_keys


_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (1.50, 6.00),
    "gpt-5-nano": (0.30, 1.20),
    "gpt-5.4": (5.00, 20.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.2": (3.00, 12.00),
    "gpt-4.1": (2.00, 8.00),
    "o3": (10.00, 40.00),
    "o4-mini": (1.10, 4.40),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gemini-3-flash": (0.10, 0.40),
    "gemini-3.1-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro": (2.50, 10.00),
    "grok-4": (3.00, 15.00),
    "grok-4-1": (3.00, 15.00),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = None
    for key, val in _PRICING.items():
        if key in (model or ""):
            rates = val
            break
    if rates is None:
        rates = (0.15, 0.60)
    return round(
        prompt_tokens / 1_000_000 * rates[0] + completion_tokens / 1_000_000 * rates[1],
        6,
    )


def invoke_chat_model_human_message(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    *,
    strip_json_fences: bool = True,
) -> tuple[str, dict]:
    """
    ``create_llm_client`` → ``get_llm()`` → ``llm.invoke([HumanMessage(...)])``,
    same chat model stack as analysts; single-turn user message (no tools).
    Returns (text, token_usage_dict).
    """
    config = get_config()
    provider_lc = (provider or config.get("llm_provider", "openai")).lower()
    resolved_model = model or config.get("quick_think_llm")
    base_url = config.get("backend_url")

    llm_kwargs = provider_kwargs_for_provider(config, provider_lc)
    client = create_llm_client(
        provider=provider_lc,
        model=resolved_model,
        base_url=base_url,
        **llm_kwargs,
    )
    llm = client.get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    if strip_json_fences:
        text = _strip_json_fences(text)

    prompt_tokens, completion_tokens, meta_keys = _response_token_usage(response, provider_lc)
    cost = _estimate_cost(resolved_model, prompt_tokens, completion_tokens)
    out = {
        "model": resolved_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
    }
    if meta_keys:
        out["_meta_keys"] = meta_keys
    return text, out
