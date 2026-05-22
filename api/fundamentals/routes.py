"""Fundamentals Lab API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.database import get_db_connection, DB_PATH, list_catalog_rows

def _get_calendar_tickers() -> list[dict]:
    """Return portfolio positions + watchlist favorites (stocks only) as {ticker, name} dicts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Portfolio positions
    cursor.execute("SELECT DISTINCT ticker FROM positions")
    portfolio_tickers = {r["ticker"].upper() for r in cursor.fetchall()}

    # Watchlist favorites (stocks only)
    cursor.execute(
        """SELECT c.ticker, c.name FROM catalog_favorites f
           JOIN catalog_items c ON c.ticker = f.ticker
           WHERE c.asset_type = 'stock'"""
    )
    fav_rows = {r["ticker"].upper(): r["name"] for r in cursor.fetchall()}

    # Also get names for portfolio tickers from catalog
    cursor.execute(
        "SELECT ticker, name FROM catalog_items WHERE ticker IN ({})".format(
            ",".join("?" * len(portfolio_tickers))
        ) if portfolio_tickers else "SELECT ticker, name FROM catalog_items WHERE 0",
        list(portfolio_tickers),
    )
    catalog_names = {r["ticker"].upper(): r["name"] for r in cursor.fetchall()}
    conn.close()

    seen: set[str] = set()
    result: list[dict] = []
    for t in sorted(portfolio_tickers | set(fav_rows.keys())):
        if t in seen:
            continue
        seen.add(t)
        name = fav_rows.get(t) or catalog_names.get(t) or ""
        result.append({"ticker": t, "name": name})
    return result
from tradingagents.dataflows.fundamentals_structured import (
    get_ratios,
    get_income_trend,
    get_balance_sheet_snapshot,
    get_cashflow_health,
    get_earnings_history,
)
from tradingagents.dataflows.yfinance_extras import get_earnings_calendar, get_earnings_calendar_multi
from tradingagents.dataflows.sec_edgar import get_cik_for_ticker, get_recent_filings, get_filing_text
from tradingagents.dataflows.finnhub_fundamentals import get_analyst_consensus, get_price_targets
from tradingagents.dataflows.llm_invoke import invoke_chat_model_human_message
from tradingagents.dataflows.config import get_config

from .models import (
    EarningsCalendarRow,
    EarningsCalendarResponse,
    RatiosData,
    IncomePeriod,
    BalanceSheetSnapshot,
    CashflowHealth,
    EarningsHistoryRow,
    AnalystConsensus,
    TickerFundamentalsResponse,
    SecFiling,
    SecFilingsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fundamentals", tags=["fundamentals"])

_EXECUTOR = ThreadPoolExecutor(max_workers=6)
_CACHE_TTL_HOURS = 24
_ALERT_DAYS = 3


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _cache_fresh(cached_at: Optional[str]) -> bool:
    if not cached_at:
        return False
    try:
        ts = datetime.fromisoformat(cached_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - ts) < timedelta(hours=_CACHE_TTL_HOURS)
    except Exception:
        return False


def _days_until(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        target = datetime.fromisoformat(date_str).date()
        today = datetime.now(tz=timezone.utc).date()
        return (target - today).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GET /api/fundamentals/calendar
# ---------------------------------------------------------------------------

@router.get("/calendar", response_model=EarningsCalendarResponse)
def get_earnings_calendar_route(force: bool = False):
    """Earnings calendar for all watchlist tickers with 24h SQLite cache."""
    conn = get_db_connection()
    cursor = conn.cursor()

    watchlist = _get_calendar_tickers()
    tickers = [r["ticker"] for r in watchlist]

    if not tickers:
        conn.close()
        return EarningsCalendarResponse(rows=[], upcoming_count=0)

    # Check if we have fresh cache for all tickers
    if not force:
        cursor.execute(
            "SELECT ticker, next_earnings_date, earnings_time, eps_estimate, revenue_estimate, history_json, cached_at "
            "FROM earnings_calendar_cache WHERE ticker IN ({})".format(
                ",".join("?" * len(tickers))
            ),
            tickers,
        )
        cached_rows = {r["ticker"]: dict(r) for r in cursor.fetchall()}
        all_fresh = (
            len(cached_rows) == len(tickers)
            and all(_cache_fresh(v["cached_at"]) for v in cached_rows.values())
        )
        if all_fresh:
            conn.close()
            return _build_calendar_response(cached_rows, watchlist)

    conn.close()

    # Fetch fresh from yfinance
    fresh_data = get_earnings_calendar_multi(tickers)
    ticker_name_map = {r["ticker"]: r.get("name") or "" for r in watchlist}

    conn = get_db_connection()
    cursor = conn.cursor()
    now = _now_iso()

    for item in fresh_data:
        t = item["ticker"]
        history = _fetch_earnings_history_for_cache(t)
        history_json = json.dumps(history)
        last = history[0] if history else {}

        cursor.execute(
            """INSERT OR REPLACE INTO earnings_calendar_cache
               (ticker, next_earnings_date, earnings_time, eps_estimate, revenue_estimate, history_json, cached_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                t,
                item.get("next_earnings_date"),
                None,  # yfinance doesn't provide AMC/BMO reliably
                item.get("eps_estimate_avg"),
                item.get("revenue_estimate_low"),
                history_json,
                now,
            ),
        )
    conn.commit()

    # Re-read from cache for consistent response
    cursor.execute(
        "SELECT ticker, next_earnings_date, earnings_time, eps_estimate, revenue_estimate, history_json, cached_at "
        "FROM earnings_calendar_cache WHERE ticker IN ({})".format(
            ",".join("?" * len(tickers))
        ),
        tickers,
    )
    cached_rows = {r["ticker"]: dict(r) for r in cursor.fetchall()}
    conn.close()

    return _build_calendar_response(cached_rows, watchlist)


def _fetch_earnings_history_for_cache(ticker: str) -> list[dict]:
    try:
        return get_earnings_history(ticker, quarters=8)
    except Exception:
        return []


def _build_calendar_response(
    cached_rows: dict[str, dict], watchlist: list[dict]
) -> EarningsCalendarResponse:
    rows: list[EarningsCalendarRow] = []
    upcoming = 0

    for w in watchlist:
        t = w["ticker"]
        c = cached_rows.get(t, {})
        history = json.loads(c.get("history_json") or "[]")
        last = history[0] if history else {}
        days = _days_until(c.get("next_earnings_date"))

        if days is not None and 0 <= days <= _ALERT_DAYS:
            upcoming += 1

        rows.append(EarningsCalendarRow(
            ticker=t,
            short_name=w.get("name"),
            next_earnings_date=c.get("next_earnings_date"),
            days_until=days,
            earnings_time=c.get("earnings_time"),
            eps_estimate=c.get("eps_estimate"),
            revenue_estimate=c.get("revenue_estimate"),
            last_eps_actual=last.get("eps_actual"),
            last_surprise_pct=last.get("surprise_pct"),
        ))

    # Sort: upcoming first, then by date
    def _sort_key(r: EarningsCalendarRow):
        d = r.next_earnings_date
        return (d is None, d or "")

    rows.sort(key=_sort_key)
    return EarningsCalendarResponse(
        rows=rows,
        upcoming_count=upcoming,
        cached_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# GET /api/fundamentals/ticker/{ticker}
# ---------------------------------------------------------------------------

@router.get("/ticker/{ticker}", response_model=TickerFundamentalsResponse)
def get_ticker_fundamentals(ticker: str, force: bool = False):
    """Full fundamentals for a ticker with 24h SQLite cache."""
    t = ticker.upper().strip()
    conn = get_db_connection()
    cursor = conn.cursor()

    if not force:
        cursor.execute(
            "SELECT ratios_json, income_json, balance_json, cashflow_json, analyst_json, cached_at "
            "FROM fundamentals_cache WHERE ticker = ?",
            (t,),
        )
        row = cursor.fetchone()
        if row and _cache_fresh(row["cached_at"]):
            conn.close()
            return _build_fundamentals_response(t, row)

    conn.close()

    # Fetch fresh in parallel using thread executor
    loop = asyncio.new_event_loop()
    try:
        ratios, income, balance, cashflow, history, consensus, targets = loop.run_until_complete(
            _fetch_all_fundamentals(t)
        )
    finally:
        loop.close()

    analyst = _merge_analyst(consensus, targets)

    ratios_json = json.dumps(ratios)
    income_json = json.dumps(income)
    balance_json = json.dumps(balance)
    cashflow_json = json.dumps(cashflow)
    analyst_json = json.dumps({
        "history": history,
        "analyst": analyst,
    })

    now = _now_iso()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO fundamentals_cache
           (ticker, ratios_json, income_json, balance_json, cashflow_json, analyst_json, cached_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (t, ratios_json, income_json, balance_json, cashflow_json, analyst_json, now, now),
    )
    conn.commit()

    cursor.execute(
        "SELECT ratios_json, income_json, balance_json, cashflow_json, analyst_json, cached_at "
        "FROM fundamentals_cache WHERE ticker = ?",
        (t,),
    )
    row = cursor.fetchone()
    conn.close()

    return _build_fundamentals_response(t, row)


async def _fetch_all_fundamentals(ticker: str):
    loop = asyncio.get_event_loop()
    fns = [
        (get_ratios, (ticker,)),
        (get_income_trend, (ticker,)),
        (get_balance_sheet_snapshot, (ticker,)),
        (get_cashflow_health, (ticker,)),
        (get_earnings_history, (ticker,)),
        (get_analyst_consensus, (ticker,)),
        (get_price_targets, (ticker,)),
    ]
    tasks = [loop.run_in_executor(_EXECUTOR, fn, *args) for fn, args in fns]
    return await asyncio.gather(*tasks)


def _merge_analyst(consensus: Optional[dict], targets: Optional[dict]) -> Optional[dict]:
    if not consensus and not targets:
        return None
    result = {**(consensus or {}), **(targets or {})}
    return result


def _build_fundamentals_response(ticker: str, row) -> TickerFundamentalsResponse:
    ratios_raw = json.loads(row["ratios_json"] or "{}")
    income_raw = json.loads(row["income_json"] or "[]")
    balance_raw = json.loads(row["balance_json"] or "{}")
    cashflow_raw = json.loads(row["cashflow_json"] or "{}")
    analyst_extra = json.loads(row["analyst_json"] or "{}")
    history_raw = analyst_extra.get("history") or []
    analyst_raw = analyst_extra.get("analyst")

    ratios = RatiosData(
        ticker=ticker,
        short_name=ratios_raw.get("short_name"),
        sector=ratios_raw.get("sector"),
        industry=ratios_raw.get("industry"),
        current_price=ratios_raw.get("current_price"),
        market_cap=ratios_raw.get("market_cap"),
        pe_ratio=ratios_raw.get("pe_ratio"),
        forward_pe=ratios_raw.get("forward_pe"),
        peg_ratio=ratios_raw.get("peg_ratio"),
        price_to_book=ratios_raw.get("price_to_book"),
        roe=ratios_raw.get("roe"),
        roa=ratios_raw.get("roa"),
        profit_margin=ratios_raw.get("profit_margin"),
        operating_margin=ratios_raw.get("operating_margin"),
        beta=ratios_raw.get("beta"),
        week52_high=ratios_raw.get("52w_high"),
        week52_low=ratios_raw.get("52w_low"),
        dividend_yield=ratios_raw.get("dividend_yield"),
    )

    income = [IncomePeriod(**p) for p in income_raw]

    balance = BalanceSheetSnapshot(
        period=balance_raw.get("period"),
        total_assets=balance_raw.get("total_assets"),
        current_assets=balance_raw.get("current_assets"),
        total_liabilities=balance_raw.get("total_liabilities"),
        stockholders_equity=balance_raw.get("stockholders_equity"),
        total_debt=balance_raw.get("total_debt"),
        current_liabilities=balance_raw.get("current_liabilities"),
        debt_to_equity=balance_raw.get("debt_to_equity"),
        current_ratio=balance_raw.get("current_ratio"),
    )

    cashflow = CashflowHealth(
        period=cashflow_raw.get("period"),
        operating_cf=cashflow_raw.get("operating_cf"),
        free_cf=cashflow_raw.get("free_cf"),
        capex=cashflow_raw.get("capex"),
    )

    earnings_history = [EarningsHistoryRow(**h) for h in history_raw]

    analyst: Optional[AnalystConsensus] = None
    if analyst_raw:
        analyst = AnalystConsensus(
            strong_buy=analyst_raw.get("strong_buy", 0) or 0,
            buy=analyst_raw.get("buy", 0) or 0,
            hold=analyst_raw.get("hold", 0) or 0,
            sell=analyst_raw.get("sell", 0) or 0,
            strong_sell=analyst_raw.get("strong_sell", 0) or 0,
            period=analyst_raw.get("period"),
            target_high=analyst_raw.get("target_high"),
            target_low=analyst_raw.get("target_low"),
            target_mean=analyst_raw.get("target_mean"),
            target_median=analyst_raw.get("target_median"),
            num_analysts=analyst_raw.get("num_analysts"),
        )

    return TickerFundamentalsResponse(
        ticker=ticker,
        ratios=ratios,
        income=income,
        balance=balance,
        cashflow=cashflow,
        earnings_history=earnings_history,
        analyst=analyst,
        cached_at=row["cached_at"],
    )


# ---------------------------------------------------------------------------
# DELETE /api/fundamentals/cache/{ticker}
# ---------------------------------------------------------------------------

@router.delete("/cache/{ticker}")
def invalidate_cache(ticker: str):
    t = ticker.upper().strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fundamentals_cache WHERE ticker = ?", (t,))
    cursor.execute("DELETE FROM earnings_calendar_cache WHERE ticker = ?", (t,))
    conn.commit()
    conn.close()
    return {"ok": True, "ticker": t}


# ---------------------------------------------------------------------------
# GET /api/fundamentals/filings/{ticker}
# ---------------------------------------------------------------------------

@router.get("/filings/{ticker}", response_model=SecFilingsResponse)
def get_filings(ticker: str):
    """Return last 10 SEC filings (8-K, 10-Q, 10-K) with cached LLM summaries."""
    t = ticker.upper().strip()
    cik = get_cik_for_ticker(t)
    if not cik:
        raise HTTPException(status_code=404, detail=f"CIK not found for ticker {t}")

    filings_raw = get_recent_filings(cik, limit=10)

    conn = get_db_connection()
    cursor = conn.cursor()

    results: list[SecFiling] = []
    for f in filings_raw:
        acc = f["accession_number"]
        # Check if we have a cached LLM summary
        cursor.execute(
            "SELECT llm_summary, summarized_at FROM sec_filings_cache WHERE accession_number = ?",
            (acc,),
        )
        cached = cursor.fetchone()
        # Upsert filing metadata
        cursor.execute(
            """INSERT OR IGNORE INTO sec_filings_cache
               (accession_number, ticker, cik, form_type, filing_date, filing_url)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (acc, t, cik, f["form_type"], f["filing_date"], f["filing_url"]),
        )
        results.append(SecFiling(
            accession_number=acc,
            ticker=t,
            cik=cik,
            form_type=f["form_type"],
            filing_date=f["filing_date"],
            filing_url=f["filing_url"],
            llm_summary=cached["llm_summary"] if cached else None,
            summarized_at=cached["summarized_at"] if cached else None,
        ))

    conn.commit()
    conn.close()

    return SecFilingsResponse(ticker=t, cik=cik, filings=results)


# ---------------------------------------------------------------------------
# POST /api/fundamentals/filings/{ticker}/summarize
# ---------------------------------------------------------------------------

_FILING_SUMMARY_PROMPT = """You are a financial analyst. A user has asked you to summarize the following SEC filing.

Ticker: {ticker}
Filing type: {form_type}
Filing date: {filing_date}

Extract and present the following in markdown:

## Key Financial Highlights
- Revenue, EPS, and profit vs prior period (if available)
- Beat or miss vs analyst expectations (if mentioned)

## Management Guidance
- Forward guidance for next quarter/year (if available)

## Notable Events or Risks
- Any material events, restructurings, acquisitions, or risk factors disclosed

## Investment Angle
- One sentence on what this filing means for the stock (bullish / bearish / neutral)

Keep it concise. If data is missing from the filing, say so.

Filing text:
{text}
"""


@router.post("/filings/{ticker}/summarize")
def summarize_filing(ticker: str, body: dict):
    """Fetch filing text from SEC EDGAR and stream an LLM summary as SSE."""
    t = ticker.upper().strip()
    accession_number = body.get("accession_number", "").strip()
    filing_url = body.get("filing_url", "").strip()
    form_type = body.get("form_type", "8-K")
    filing_date = body.get("filing_date", "")

    if not accession_number or not filing_url:
        raise HTTPException(status_code=422, detail="accession_number and filing_url required")

    # Return cached summary if available
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT llm_summary FROM sec_filings_cache WHERE accession_number = ?",
        (accession_number,),
    )
    cached = cursor.fetchone()
    conn.close()

    if cached and cached["llm_summary"]:
        def _cached_stream():
            yield f"data: {json.dumps({'chunk': cached['llm_summary'], 'done': True})}\n\n"
        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    def _generate():
        # Fetch filing text in thread
        text = get_filing_text(filing_url)
        if not text.strip():
            yield f"data: {json.dumps({'error': 'Could not fetch filing text from SEC EDGAR'})}\n\n"
            return

        cfg = get_config()
        provider = cfg.get("llm_provider", "ollama")
        model = cfg.get("deep_think_llm") or cfg.get("quick_think_llm")

        prompt = _FILING_SUMMARY_PROMPT.format(
            ticker=t,
            form_type=form_type,
            filing_date=filing_date,
            text=text[:40_000],
        )
        try:
            summary, _ = invoke_chat_model_human_message(
                prompt,
                provider=provider,
                model=model,
                strip_json_fences=False,
            )
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        # Cache the summary
        try:
            conn2 = get_db_connection()
            cursor2 = conn2.cursor()
            cursor2.execute(
                """UPDATE sec_filings_cache
                   SET llm_summary = ?, summarized_at = ?
                   WHERE accession_number = ?""",
                (summary, _now_iso(), accession_number),
            )
            if cursor2.rowcount == 0:
                cursor2.execute(
                    """INSERT OR REPLACE INTO sec_filings_cache
                       (accession_number, ticker, cik, form_type, filing_date, filing_url, llm_summary, summarized_at)
                       VALUES (?, ?, NULL, ?, ?, ?, ?, ?)""",
                    (accession_number, t, form_type, filing_date, filing_url, summary, _now_iso()),
                )
            conn2.commit()
            conn2.close()
        except Exception as exc:
            logger.warning("Failed to cache filing summary: %s", exc)

        yield f"data: {json.dumps({'chunk': summary, 'done': True})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /api/fundamentals/analyze/{ticker}
# ---------------------------------------------------------------------------

_ANALYZE_PROMPT = """You are a senior equity analyst. Analyze the following fundamental data for {ticker} and provide a structured investment assessment.

## Company Overview
{ratios_summary}

## Earnings History (last 8 quarters)
{earnings_summary}

## Income Trend (last 8 quarters)
{income_summary}

## Balance Sheet
{balance_summary}

## Cash Flow
{cashflow_summary}

## Analyst Consensus
{analyst_summary}

Based on this data, provide:

## Fundamental Assessment
A 2-3 paragraph qualitative analysis of the company's financial health and trajectory.

## Key Strengths
Bullet points of the 3-5 most positive fundamental signals.

## Key Risks
Bullet points of the 3-5 most concerning fundamental signals.

## Valuation View
Is the stock expensive, fairly valued, or cheap based on PE/forward PE/PEG? Compare to sector norms.

## Recommendation
STRONG BUY / BUY / HOLD / SELL / STRONG SELL with a one-sentence rationale.

*This is research analysis only and not financial advice.*
"""


def _fmt_ratios(r: dict) -> str:
    fields = [
        ("Price", r.get("current_price")),
        ("Market Cap", r.get("market_cap")),
        ("PE", r.get("pe_ratio")),
        ("Forward PE", r.get("forward_pe")),
        ("PEG", r.get("peg_ratio")),
        ("P/B", r.get("price_to_book")),
        ("ROE", r.get("roe")),
        ("Profit Margin", r.get("profit_margin")),
        ("Beta", r.get("beta")),
    ]
    return "\n".join(f"- {k}: {v}" for k, v in fields if v is not None)


def _fmt_earnings(history: list) -> str:
    if not history:
        return "No earnings history available."
    lines = []
    for h in history[:4]:
        surprise = f"{h.get('surprise_pct', 0):+.1f}%" if h.get("surprise_pct") is not None else "N/A"
        lines.append(
            f"- {h.get('date')}: EPS est {h.get('eps_estimate')} vs actual {h.get('eps_actual')} ({surprise} surprise)"
        )
    return "\n".join(lines)


def _fmt_income(income: list) -> str:
    if not income:
        return "No income data available."
    lines = []
    for p in income[:4]:
        rev = f"${p.get('revenue', 0)/1e9:.1f}B" if p.get("revenue") else "N/A"
        ni = f"${p.get('net_income', 0)/1e9:.1f}B" if p.get("net_income") else "N/A"
        lines.append(f"- {p.get('period')}: Revenue {rev}, Net Income {ni}, EPS {p.get('eps')}")
    return "\n".join(lines)


def _fmt_balance(b: dict) -> str:
    return "\n".join([
        f"- Total Assets: ${(b.get('total_assets') or 0)/1e9:.1f}B",
        f"- Total Debt: ${(b.get('total_debt') or 0)/1e9:.1f}B",
        f"- Debt/Equity: {b.get('debt_to_equity', 'N/A')}",
        f"- Current Ratio: {b.get('current_ratio', 'N/A')}",
    ])


def _fmt_cashflow(c: dict) -> str:
    ocf = f"${(c.get('operating_cf') or 0)/1e9:.1f}B" if c.get("operating_cf") else "N/A"
    fcf = f"${(c.get('free_cf') or 0)/1e9:.1f}B" if c.get("free_cf") else "N/A"
    capex = f"${abs(c.get('capex') or 0)/1e9:.1f}B" if c.get("capex") else "N/A"
    return f"- Operating CF: {ocf}\n- Free CF: {fcf}\n- CapEx: {capex}"


def _fmt_analyst(a: Optional[dict]) -> str:
    if not a:
        return "No analyst data available (FINNHUB_API_KEY not set)."
    total = sum([a.get("strong_buy", 0), a.get("buy", 0), a.get("hold", 0), a.get("sell", 0), a.get("strong_sell", 0)])
    target = a.get("target_mean")
    return (
        f"- {a.get('strong_buy', 0)} Strong Buy, {a.get('buy', 0)} Buy, {a.get('hold', 0)} Hold, "
        f"{a.get('sell', 0)} Sell, {a.get('strong_sell', 0)} Strong Sell (total: {total} analysts)\n"
        f"- Avg price target: {target}"
    )


@router.post("/analyze/{ticker}")
def analyze_ticker(ticker: str, body: dict = {}):
    """Run full LLM fundamentals analysis for a ticker and stream back as SSE."""
    t = ticker.upper().strip()
    provider = body.get("llm_provider") or get_config().get("llm_provider", "ollama")
    model = body.get("llm_model") or get_config().get("deep_think_llm") or get_config().get("quick_think_llm")

    def _generate():
        # Fetch fundamentals (from cache if available, else fresh)
        try:
            fundamentals = get_ticker_fundamentals(t, force=False)
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Failed to fetch fundamentals: {exc}'})}\n\n"
            return

        # Build analyst summary from cached data
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT analyst_json FROM fundamentals_cache WHERE ticker = ?", (t,))
        row = cursor.fetchone()
        conn.close()
        analyst_extra = json.loads(row["analyst_json"] or "{}") if row else {}
        analyst_raw = analyst_extra.get("analyst")

        prompt = _ANALYZE_PROMPT.format(
            ticker=t,
            ratios_summary=_fmt_ratios(json.loads(
                json.dumps({
                    "current_price": fundamentals.ratios.current_price,
                    "market_cap": fundamentals.ratios.market_cap,
                    "pe_ratio": fundamentals.ratios.pe_ratio,
                    "forward_pe": fundamentals.ratios.forward_pe,
                    "peg_ratio": fundamentals.ratios.peg_ratio,
                    "price_to_book": fundamentals.ratios.price_to_book,
                    "roe": fundamentals.ratios.roe,
                    "profit_margin": fundamentals.ratios.profit_margin,
                    "beta": fundamentals.ratios.beta,
                })
            )),
            earnings_summary=_fmt_earnings([h.model_dump() for h in fundamentals.earnings_history]),
            income_summary=_fmt_income([p.model_dump() for p in fundamentals.income]),
            balance_summary=_fmt_balance(fundamentals.balance.model_dump()),
            cashflow_summary=_fmt_cashflow(fundamentals.cashflow.model_dump()),
            analyst_summary=_fmt_analyst(analyst_raw),
        )

        try:
            summary, _ = invoke_chat_model_human_message(
                prompt,
                provider=provider,
                model=model,
                strip_json_fences=False,
            )
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        yield f"data: {json.dumps({'chunk': summary, 'done': True})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
