"""Finnhub API client for analyst consensus and price targets.

Requires FINNHUB_API_KEY env var. Gracefully returns None fields when the key is absent.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 10


def _api_key() -> Optional[str]:
    return (os.getenv("FINNHUB_API_KEY") or "").strip() or None


def _get(path: str, params: dict) -> Optional[dict | list]:
    key = _api_key()
    if not key:
        return None
    try:
        r = requests.get(
            f"{_BASE}{path}",
            params={"token": key, **params},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Finnhub %s failed: %s", path, exc)
        return None


def get_analyst_consensus(ticker: str) -> Optional[dict]:
    """Return buy/hold/sell/strong_buy/strong_sell counts for the most recent period."""
    data = _get("/stock/recommendation", {"symbol": ticker.upper()})
    if not data or not isinstance(data, list):
        return None
    latest = data[0]  # Most recent period is first
    return {
        "strong_buy": latest.get("strongBuy", 0),
        "buy": latest.get("buy", 0),
        "hold": latest.get("hold", 0),
        "sell": latest.get("sell", 0),
        "strong_sell": latest.get("strongSell", 0),
        "period": latest.get("period"),
    }


def get_price_targets(ticker: str) -> Optional[dict]:
    """Return analyst price target high/low/mean/median."""
    data = _get("/stock/price-target", {"symbol": ticker.upper()})
    if not data or not isinstance(data, dict):
        return None
    return {
        "target_high": data.get("targetHigh"),
        "target_low": data.get("targetLow"),
        "target_mean": data.get("targetMean"),
        "target_median": data.get("targetMedian"),
        "last_updated": data.get("lastUpdated"),
        "num_analysts": data.get("numberOfAnalysts"),
    }
