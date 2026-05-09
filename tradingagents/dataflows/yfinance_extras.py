"""yfinance extras: earnings calendar, screener trending, influencer-mention search."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import yfinance as yf

from .yfinance_news import _extract_article_data
from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earnings calendar
# ---------------------------------------------------------------------------

def get_earnings_calendar(ticker: str) -> dict:
    """
    Return next earnings date and analyst estimates for a single ticker.

    Returns a JSON-serialisable dict with keys:
        ticker, next_earnings_date, all_earnings_dates,
        eps_estimate_low, eps_estimate_high, eps_estimate_avg,
        revenue_estimate_low, revenue_estimate_high, ex_dividend_date, dividend_date
    """
    result = {
        "ticker": ticker,
        "next_earnings_date": None,
        "all_earnings_dates": [],
        "eps_estimate_low": None,
        "eps_estimate_high": None,
        "eps_estimate_avg": None,
        "revenue_estimate_low": None,
        "revenue_estimate_high": None,
        "ex_dividend_date": None,
        "dividend_date": None,
    }
    try:
        cal = yf_retry(lambda: yf.Ticker(ticker).get_calendar())
        if not cal:
            return result

        def _to_iso(val) -> str | None:
            if val is None:
                return None
            if isinstance(val, str):
                return val
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            return str(val)

        today = datetime.now(tz=timezone.utc).date()

        raw_dates = cal.get("Earnings Date", [])
        if not isinstance(raw_dates, list):
            raw_dates = [raw_dates]

        iso_dates = [_to_iso(d) for d in raw_dates if d is not None]
        result["all_earnings_dates"] = iso_dates

        # Pick the nearest future date as next_earnings_date
        future = [d for d in iso_dates if d and d >= str(today)]
        result["next_earnings_date"] = min(future) if future else (iso_dates[0] if iso_dates else None)

        result["eps_estimate_low"] = cal.get("Earnings Low")
        result["eps_estimate_high"] = cal.get("Earnings High")
        result["eps_estimate_avg"] = cal.get("Earnings Average")
        result["revenue_estimate_low"] = cal.get("Revenue Low")
        result["revenue_estimate_high"] = cal.get("Revenue High")
        result["ex_dividend_date"] = _to_iso(cal.get("Ex-Dividend Date"))
        result["dividend_date"] = _to_iso(cal.get("Dividend Date"))

    except Exception as e:
        logger.warning("get_earnings_calendar(%s) failed: %s", ticker, e)

    return result


def get_earnings_calendar_multi(tickers: list[str]) -> list[dict]:
    """
    Batch earnings calendar fetch using a thread pool.
    Returns list sorted by next_earnings_date ascending (None last).
    """
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get_earnings_calendar, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning("Earnings calendar fetch failed for %s: %s", futures[future], e)

    def _sort_key(row: dict):
        d = row.get("next_earnings_date")
        return (d is None, d or "")

    return sorted(results, key=_sort_key)


# ---------------------------------------------------------------------------
# Screener-based trending stocks
# ---------------------------------------------------------------------------

_VALID_SCREENERS = {
    "most_actives",
    "day_gainers",
    "day_losers",
    "most_shorted_stocks",
    "growth_technology_stocks",
}


def get_trending_stocks(screener: str = "most_actives", count: int = 20) -> list[dict]:
    """
    Return top N stocks from a yfinance predefined screener.

    Each dict contains: symbol, display_name, price, change_pct, volume, market_cap.
    Valid screener values: most_actives | day_gainers | day_losers |
                           most_shorted_stocks | growth_technology_stocks
    """
    if screener not in _VALID_SCREENERS:
        screener = "most_actives"
    count = min(max(count, 1), 50)

    try:
        raw = yf_retry(lambda: yf.screen(screener, count=count))
        quotes = raw.get("quotes", []) if raw else []
    except Exception as e:
        logger.warning("get_trending_stocks(%s) failed: %s", screener, e)
        return []

    results = []
    for q in quotes:
        results.append({
            "symbol": q.get("symbol", ""),
            "display_name": q.get("displayName") or q.get("shortName") or q.get("symbol", ""),
            "price": q.get("regularMarketPrice") or 0.0,
            "change_pct": q.get("regularMarketChangePercent") or 0.0,
            "volume": q.get("regularMarketVolume") or 0,
            "market_cap": q.get("marketCap"),
        })
    return results


# ---------------------------------------------------------------------------
# Influencer mention search via yfinance news keyword search
# ---------------------------------------------------------------------------

def get_influencer_mentions(person: str, ticker: str, days: int = 7) -> list[dict]:
    """
    Search Yahoo Finance news for articles mentioning both a person and a ticker.

    Uses yf.Search(query="{person} {ticker}") and filters by publish date within
    the last `days`. Returns a list of article dicts with keys:
        title, publisher, link, pub_date, summary
    """
    query = f"{person} {ticker}"
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

    try:
        search = yf_retry(lambda: yf.Search(query=query, news_count=30, enable_fuzzy_query=True))
        raw_articles = search.news if search and search.news else []
    except Exception as e:
        logger.warning("get_influencer_mentions(%s, %s) failed: %s", person, ticker, e)
        return []

    articles = []
    for raw in raw_articles:
        data = _extract_article_data(raw)
        pub_date = data.get("pub_date")

        # Apply date filter if pub_date is available
        if pub_date:
            pub_aware = pub_date if pub_date.tzinfo else pub_date.replace(tzinfo=timezone.utc)
            if pub_aware < cutoff:
                continue

        articles.append({
            "title": data["title"],
            "publisher": data["publisher"],
            "link": data["link"],
            "pub_date": data["pub_date"].strftime("%Y-%m-%d") if data.get("pub_date") else None,
            "summary": data.get("summary", ""),
        })

    return articles
