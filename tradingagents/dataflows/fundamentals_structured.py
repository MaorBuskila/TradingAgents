"""yfinance wrappers that return structured JSON-serialisable dicts (not CSV strings).

These are for the Fundamentals Lab API — NOT for the LLM agent tools (which use y_finance.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import yfinance as yf

from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _df_row(df, row_name: str, col_idx: int) -> Optional[float]:
    """Safely extract a value from a DataFrame by row name and column index."""
    if df is None or df.empty:
        return None
    for label in df.index:
        if str(label).strip().lower() == row_name.lower():
            try:
                return _safe_float(df.iloc[df.index.get_loc(label), col_idx])
            except Exception:
                return None
    return None


def get_ratios(ticker: str) -> dict:
    """Return key ratios and company metadata from yfinance.Ticker.info."""
    result: dict[str, Any] = {
        "ticker": ticker.upper(),
        "short_name": None,
        "sector": None,
        "industry": None,
        "current_price": None,
        "market_cap": None,
        "pe_ratio": None,
        "forward_pe": None,
        "peg_ratio": None,
        "price_to_book": None,
        "roe": None,
        "roa": None,
        "profit_margin": None,
        "operating_margin": None,
        "beta": None,
        "52w_high": None,
        "52w_low": None,
        "dividend_yield": None,
    }
    try:
        info = yf_retry(lambda: yf.Ticker(ticker).info) or {}
        result.update({
            "short_name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            "market_cap": _safe_float(info.get("marketCap")),
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "price_to_book": _safe_float(info.get("priceToBook")),
            "roe": _safe_float(info.get("returnOnEquity")),
            "roa": _safe_float(info.get("returnOnAssets")),
            "profit_margin": _safe_float(info.get("profitMargins")),
            "operating_margin": _safe_float(info.get("operatingMargins")),
            "beta": _safe_float(info.get("beta")),
            "52w_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
        })
    except Exception as exc:
        logger.warning("get_ratios(%s) failed: %s", ticker, exc)
    return result


def get_income_trend(ticker: str, quarters: int = 8) -> list[dict]:
    """Return last N quarters of income statement data as a list."""
    rows: list[dict] = []
    try:
        t = yf.Ticker(ticker)
        df = yf_retry(lambda: t.quarterly_income_stmt)
        if df is None or df.empty:
            return rows

        # DataFrame: rows=metrics, cols=period-end dates (newest first)
        cols = list(df.columns[:quarters])
        for col in cols:
            period = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

            def _get(names: list[str]) -> Optional[float]:
                for n in names:
                    for label in df.index:
                        if str(label).strip().lower() == n.lower():
                            return _safe_float(df.at[label, col])
                return None

            rows.append({
                "period": period,
                "revenue": _get(["Total Revenue", "Revenue"]),
                "gross_profit": _get(["Gross Profit"]),
                "net_income": _get(["Net Income", "Net Income Common Stockholders"]),
                "eps": _get(["Basic EPS", "Diluted EPS"]),
                "operating_income": _get(["Operating Income", "EBIT"]),
            })
    except Exception as exc:
        logger.warning("get_income_trend(%s) failed: %s", ticker, exc)
    return rows


def get_balance_sheet_snapshot(ticker: str) -> dict:
    """Return latest quarter balance sheet key metrics."""
    result: dict[str, Any] = {
        "total_assets": None,
        "current_assets": None,
        "total_liabilities": None,
        "stockholders_equity": None,
        "total_debt": None,
        "current_liabilities": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "period": None,
    }
    try:
        t = yf.Ticker(ticker)
        df = yf_retry(lambda: t.quarterly_balance_sheet)
        if df is None or df.empty:
            return result

        col = df.columns[0]
        result["period"] = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

        def _get(names: list[str]) -> Optional[float]:
            for n in names:
                for label in df.index:
                    if str(label).strip().lower() == n.lower():
                        return _safe_float(df.at[label, col])
            return None

        ta = _get(["Total Assets"])
        cl = _get(["Current Liabilities", "Current Liabilities Net Minority Interest"])
        ca = _get(["Current Assets"])
        eq = _get(["Stockholders Equity", "Common Stockholders Equity"])
        td = _get(["Total Debt", "Long Term Debt And Capital Lease Obligation"])
        tl = _get(["Total Liabilities Net Minority Interest", "Total Liabilities"])

        result.update({
            "total_assets": ta,
            "current_assets": ca,
            "total_liabilities": tl,
            "stockholders_equity": eq,
            "total_debt": td,
            "current_liabilities": cl,
            "debt_to_equity": _safe_float(td / eq) if td and eq and eq != 0 else None,
            "current_ratio": _safe_float(ca / cl) if ca and cl and cl != 0 else None,
        })
    except Exception as exc:
        logger.warning("get_balance_sheet_snapshot(%s) failed: %s", ticker, exc)
    return result


def get_cashflow_health(ticker: str) -> dict:
    """Return latest quarter cash flow health metrics."""
    result: dict[str, Any] = {
        "operating_cf": None,
        "free_cf": None,
        "capex": None,
        "period": None,
    }
    try:
        t = yf.Ticker(ticker)
        df = yf_retry(lambda: t.quarterly_cashflow)
        if df is None or df.empty:
            return result

        col = df.columns[0]
        result["period"] = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)

        def _get(names: list[str]) -> Optional[float]:
            for n in names:
                for label in df.index:
                    if str(label).strip().lower() == n.lower():
                        return _safe_float(df.at[label, col])
            return None

        result.update({
            "operating_cf": _get(["Operating Cash Flow"]),
            "free_cf": _get(["Free Cash Flow"]),
            "capex": _get(["Capital Expenditure"]),
        })
    except Exception as exc:
        logger.warning("get_cashflow_health(%s) failed: %s", ticker, exc)
    return result


def get_earnings_history(ticker: str, quarters: int = 8) -> list[dict]:
    """Return last N quarters of EPS estimate vs actual + surprise."""
    rows: list[dict] = []
    try:
        t = yf.Ticker(ticker)
        df = yf_retry(lambda: t.earnings_dates)
        if df is None or df.empty:
            return rows

        today = datetime.now(tz=timezone.utc)
        # Filter to past dates only (no forward estimates)
        past = df[df.index < today].head(quarters)

        for idx, row in past.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            rows.append({
                "date": date_str,
                "eps_estimate": _safe_float(row.get("EPS Estimate")),
                "eps_actual": _safe_float(row.get("Reported EPS")),
                "surprise_pct": _safe_float(row.get("Surprise(%)")),
            })
    except Exception as exc:
        logger.warning("get_earnings_history(%s) failed: %s", ticker, exc)
    return rows
