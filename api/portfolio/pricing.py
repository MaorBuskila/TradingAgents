from typing import Any, Dict

import yfinance as yf

from .repository import get_all_positions, update_prices


def fetch_price_map(tickers: list[str]) -> dict[str, float]:
    """Fetch closing prices for a list of tickers via yfinance. Returns {ticker: price}."""
    if not tickers:
        return {}
    data = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
    prices: dict[str, float] = {}
    if data.empty or "Close" not in data:
        return prices
    if len(tickers) == 1:
        try:
            prices[tickers[0]] = float(data["Close"].iloc[-1])
        except Exception:
            pass
    else:
        for t in tickers:
            try:
                if t in data["Close"]:
                    prices[t] = float(data["Close"][t].iloc[-1])
            except Exception:
                pass
    return prices


def _tase_yf_ticker(ticker: str) -> str:
    """Return the yfinance ticker for a TASE position (appends .TA if missing)."""
    return ticker if ticker.upper().endswith(".TA") else f"{ticker}.TA"


def run_price_refresh() -> Dict[str, Any]:
    positions = get_all_positions()
    if not positions:
        return {"status": "no positions"}

    # For TASE direct positions (no follow_ticker), fetch with .TA suffix so
    # yfinance returns the TASE price in Agorot rather than the US price in USD.
    us_tickers: list[str] = []
    tase_yf_map: dict[str, str] = {}   # stored_ticker → yf ticker (with .TA)
    follow_tickers: list[str] = []

    for p in positions:
        # Skip positions with a manually-set price — user manages these
        if p.get("manual_price") is not None:
            continue
        is_tase = p.get("exchange") == "Israeli (TASE)"
        follow = p.get("follow_ticker")
        ticker = p["ticker"]
        if is_tase and follow:
            follow_tickers.append(follow)
        elif is_tase:
            yt = _tase_yf_ticker(ticker)
            tase_yf_map[ticker] = yt
        else:
            us_tickers.append(ticker)
        if follow and not is_tase:
            follow_tickers.append(follow)

    us_prices = fetch_price_map(list(set(us_tickers)))
    tase_prices = fetch_price_map(list(set(tase_yf_map.values()))) if tase_yf_map else {}
    follow_prices = fetch_price_map(list(set(follow_tickers))) if follow_tickers else {}

    usd_ils: float | None = None
    needs_fx = any(
        p.get("exchange") == "Israeli (TASE)" and p.get("follow_ticker") for p in positions
    )
    if needs_fx:
        try:
            usd_ils = float(yf.Ticker("ILS=X").fast_info["lastPrice"])
        except Exception:
            usd_ils = None

    final_prices: dict[str, float] = {}

    for p in positions:
        ticker = p["ticker"]
        exchange = p.get("exchange", "US")
        follow = p.get("follow_ticker")
        is_tase = exchange == "Israeli (TASE)"

        if is_tase and follow and follow in follow_prices and usd_ils:
            # follow_ticker path: USD price × USD/ILS rate
            final_prices[ticker] = follow_prices[follow] * usd_ils
        elif is_tase:
            # Direct TASE ticker (e.g. SPY.TA): yfinance returns Agorot → ÷100 = ILS
            yt = tase_yf_map.get(ticker)
            if yt and yt in tase_prices:
                final_prices[ticker] = tase_prices[yt] / 100
        elif ticker in us_prices:
            # US / other exchange: price already in correct currency
            final_prices[ticker] = us_prices[ticker]
        elif follow and follow in follow_prices:
            # Non-TASE mirror: use follow price directly
            final_prices[ticker] = follow_prices[follow]

    update_prices(final_prices)
    return {"status": "success", "updated": len(final_prices)}
