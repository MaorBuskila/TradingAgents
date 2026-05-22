"""Earnings alert: notifies when watchlist tickers report within ALERT_DAYS days."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ALERT_DAYS = 3
API_BASE_DEFAULT = "http://127.0.0.1:8000/api"


async def check_earnings_alerts(bot, chat_id: int, api_base: str = API_BASE_DEFAULT) -> None:
    """Query the Fundamentals Lab calendar endpoint and send Telegram messages for upcoming reports."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{api_base}/fundamentals/calendar")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("Earnings alert: calendar fetch failed: %s", exc)
        return

    rows = data.get("rows", [])
    today = datetime.now(tz=timezone.utc).date()

    alerts: list[dict] = []
    for row in rows:
        days = row.get("days_until")
        if days is None:
            continue
        if 0 <= days <= ALERT_DAYS:
            alerts.append(row)

    if not alerts:
        return

    for row in alerts:
        ticker = row.get("ticker", "")
        name = row.get("short_name") or ticker
        earnings_date = row.get("next_earnings_date", "?")
        days = row.get("days_until", "?")
        time_label = row.get("earnings_time") or ""
        eps_est = row.get("eps_estimate")
        rev_est = row.get("revenue_estimate")

        eps_str = f" | EPS est: ${eps_est:.2f}" if eps_est is not None else ""
        rev_str = f" | Rev est: ${rev_est/1e9:.1f}B" if rev_est is not None else ""
        time_str = f" ({time_label})" if time_label else ""
        days_str = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")

        msg = (
            f"Earnings Alert: {ticker} ({name}) reports {days_str}\n"
            f"Date: {earnings_date}{time_str}{eps_str}{rev_str}"
        )

        try:
            await bot.send_message(chat_id=chat_id, text=msg)
            logger.info("Earnings alert sent for %s (days_until=%s)", ticker, days)
        except Exception as exc:
            logger.warning("Failed to send earnings alert for %s: %s", ticker, exc)
