from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import uuid
import json
import asyncio
import os
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from .models import (
    AnalysisRequest,
    CatalogItemOut,
    CatalogUserAdd,
    FavoriteBody,
    YouTubeSummarizeRequest,
    YouTubeSummarizeResponse,
    YouTubeSummaryListItem,
    YouTubeSummaryDetail,
    RsiOptimizeRequest,
    RsiOptimizeResponse,
    RsiOptimizerResult,
    RsiPeriodPoint,
    RsiComparisonSummary,
    RsiSignalResponse,
    RsiHistoryPoint,
    MacdOptimizeRequest,
    MacdOptimizeResponse,
    MacdOptimizerResult,
    MacdParamPoint,
    MacdComparisonSummary,
    MacdSignalResponse,
    MacdHistoryPoint,
    WfoAnalyzeRequest,
    WfoAnalysisResult,
    MacdDtOptimizeRequest,
    MacdDtOptimizeResponse,
    MacdDtSlideResult,
    MacdDtPortfolioRequest,
    MacdDtPortfolioResponse,
    MacdDtPosition,
    RsiDtOptimizeRequest,
    RsiDtOptimizeResponse,
    RsiDtSlideResult,
    RsiDtPortfolioRequest,
    RsiDtPortfolioResponse,
    RsiDtPosition,
    RsiDtFeatureLabRequest,
    RsiDtFeatureLabResultRow,
    RsiDtFeatureLabResponse,
    OosTestRequest,
    OosTestResponse,
    OosTestDateRow,
    OosTestSummary,
    SniperOptimizeRequest,
    SniperOptimizeResponse,
    SniperEmaResult,
    SniperDtResult,
)
from . import youtube_summary
from .database import (
    init_db,
    list_catalog_rows,
    sort_catalog_rows,
    set_catalog_favorite,
    add_catalog_item,
    category_counts,
    load_category_labels,
    list_youtube_summaries,
    get_youtube_summary_by_id,
)
import sqlite3
from tradingagents.analysis_selections import normalize_analysis_selections
from tradingagents.services.analysis_flow import run_cli_style_analysis
from tradingagents.services.report_save import save_report_to_disk
from tradingagents.llm_catalog import llm_catalog_for_api
from tradingagents.analysis_defaults import infer_llm_defaults_from_env
from .reports_latest import get_latest_decision_for_ticker
from .portfolio import router as portfolio_router
from .news import router as news_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_analyze_bot_secret(
    x_tradingagents_bot_secret: str | None = Header(default=None, alias="X-TradingAgents-Bot-Secret"),
) -> None:
    expected = (os.getenv("BOT_API_SECRET") or "").strip()
    if not expected:
        return
    got = (x_tradingagents_bot_secret or "").strip()
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing bot secret")


def infer_analysis_defaults() -> dict:
    """LLM defaults from env + CLI-equivalent thinking/research presets (see tradingagents.analysis_defaults)."""
    return infer_llm_defaults_from_env()


def analysis_provider_configured(provider: str) -> bool:
    p = provider.lower()
    if p == "openai":
        return bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if p == "google":
        return bool((os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip())
    if p == "anthropic":
        return bool((os.getenv("ANTHROPIC_API_KEY") or "").strip())
    if p == "xai":
        return bool((os.getenv("XAI_API_KEY") or "").strip())
    if p == "openrouter":
        return bool((os.getenv("OPENROUTER_API_KEY") or "").strip())
    if p == "ollama":
        return True
    return False


def analysis_provider_env_hint(provider: str) -> str:
    p = provider.lower()
    return {
        "openai": "Set OPENAI_API_KEY in .env",
        "google": "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env",
        "anthropic": "Set ANTHROPIC_API_KEY in .env",
        "xai": "Set XAI_API_KEY in .env",
        "openrouter": "Set OPENROUTER_API_KEY in .env",
        "ollama": "Run a local Ollama server",
    }.get(p, "Set the API key for this provider in .env")


app = FastAPI(title="TradingAgents API")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # Only API routes require a key; static UI files and docs stay public so
    # the browser can load the page (a navigation can't send custom headers).
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    # Localhost always allowed (Vite dev proxy, Telegram bot on same machine)
    client_ip = request.client.host if request.client else ""
    if client_ip in ("127.0.0.1", "::1"):
        return await call_next(request)
    expected = (os.getenv("API_KEY") or "").strip()
    if expected:
        provided = (request.headers.get("X-API-Key") or "").strip()
        if provided != expected:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=403)
    return await call_next(request)

# Allow both local dev and the server's public origin
_extra_origins = [o for o in [os.getenv("FRONTEND_ORIGIN", "")] if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"] + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)

app.include_router(portfolio_router, prefix="/api")
app.include_router(news_router, prefix="/api")

from api.tp_tracker import router as tp_tracker_router
app.include_router(tp_tracker_router, prefix="/api")

from api.fundamentals import router as fundamentals_router
app.include_router(fundamentals_router, prefix="/api")

# In-memory job store
jobs = {}
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
def startup_event():
    init_db()


@app.post("/api/youtube/summarize", response_model=YouTubeSummarizeResponse)
def api_youtube_summarize(body: YouTubeSummarizeRequest):
    """Summarize captions (LLM), machine-translate to Hebrew (deep-translator), save to SQLite."""
    try:
        data = youtube_summary.summarize_youtube_url(
            body.url.strip(),
            llm_provider=body.llm_provider.strip(),
            llm_model=body.llm_model,
        )
        return YouTubeSummarizeResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/youtube/summaries", response_model=List[YouTubeSummaryListItem])
def api_youtube_summaries_list(limit: int = 50):
    rows = list_youtube_summaries(limit=min(limit, 200))
    for r in rows:
        if r.get("created_at") is not None:
            r["created_at"] = str(r["created_at"])
    return rows


@app.get("/api/youtube/summaries/{row_id}", response_model=YouTubeSummaryDetail)
def api_youtube_summary_get(row_id: int):
    row = get_youtube_summary_by_id(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="Summary not found")
    if row.get("created_at") is not None:
        row["created_at"] = str(row["created_at"])
    return YouTubeSummaryDetail(**row)


@app.post("/api/youtube/summaries/{row_id}/retranslate-he", response_model=YouTubeSummaryDetail)
def api_youtube_retranslate_he(row_id: int):
    """Re-translate stored English summary to Hebrew only (fixes failed `he` code, etc.)."""
    try:
        row = youtube_summary.retranslate_youtube_summary_hebrew(row_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hebrew translation failed: {e!s}")
    if row.get("created_at") is not None:
        row["created_at"] = str(row["created_at"])
    return YouTubeSummaryDetail(**row)


@app.get("/api/analysis-defaults")
def api_analysis_defaults():
    """LLM fields for Run Analysis, inferred from env; includes full provider/model catalog for the GUI."""
    out = infer_analysis_defaults()
    out["catalog"] = llm_catalog_for_api()
    return out


@app.get("/api/llm-catalog")
def api_llm_catalog():
    """Provider list, default URLs, and shallow/deep model choices (same as CLI)."""
    return llm_catalog_for_api()


@app.post("/api/analyze", dependencies=[Depends(require_analyze_bot_secret)])
async def start_analysis(req: AnalysisRequest):
    if not analysis_provider_configured(req.llm_provider):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No credentials for llm_provider={req.llm_provider!r}. "
                f"{analysis_provider_env_hint(req.llm_provider)} "
                f"or call GET /api/analysis-defaults for a matching preset."
            ),
        )
    try:
        normalized_selections = normalize_analysis_selections(req.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = {
        "status": "running",
        "queue": queue,
        "request": normalized_selections,
    }

    loop = asyncio.get_running_loop()

    def run_job():
        def on_message(msg_type, content):
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "message", "msg_type": msg_type, "content": content}),
                loop,
            )

        def on_tool_call(tool_name, args):
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "tool_call", "tool_name": tool_name, "args": args}),
                loop,
            )

        def on_agent_status(agent_name, status):
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "agent_status", "agent_name": agent_name, "status": status}),
                loop,
            )

        def on_report_section(section_name, content):
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "report_section", "section_name": section_name, "content": content}),
                loop,
            )

        def on_complete(final_state, decision):
            report_id = None
            report_saved = False
            report_save_error = None
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker_prefix = normalized_selections["ticker"].upper().strip()
                report_id = f"{ticker_prefix}_{ts}"
                save_path = Path("reports") / report_id
                save_report_to_disk(final_state, normalized_selections["ticker"], save_path)
                report_saved = True
            except Exception as ex:
                report_save_error = str(ex)
            asyncio.run_coroutine_threadsafe(
                queue.put(
                    {
                        "type": "complete",
                        "decision": decision,
                        "report_id": report_id,
                        "report_saved": report_saved,
                        **({"report_save_error": report_save_error} if report_save_error else {}),
                    }
                ),
                loop,
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["report_id"] = report_id
            jobs[job_id]["report_saved"] = report_saved
            jobs[job_id]["decision"] = decision
            if report_save_error:
                jobs[job_id]["report_save_error"] = report_save_error

        try:
            run_cli_style_analysis(
                normalized_selections,
                on_message=on_message,
                on_tool_call=on_tool_call,
                on_agent_status=on_agent_status,
                on_report_section=on_report_section,
                on_complete=on_complete,
            )
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "error": str(e)}),
                loop,
            )

    loop.run_in_executor(executor, run_job)
    
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    out: Dict[str, Any] = {
        "job_id": job_id,
        "status": j.get("status", "unknown"),
    }
    if j.get("report_id") is not None:
        out["report_id"] = j["report_id"]
    if "report_saved" in j:
        out["report_saved"] = j["report_saved"]
    if j.get("report_save_error"):
        out["report_save_error"] = j["report_save_error"]
    if j.get("decision") is not None:
        out["decision"] = j["decision"]
    if j.get("error"):
        out["error"] = j["error"]
    req = j.get("request") or {}
    if isinstance(req, dict) and req.get("ticker"):
        out["ticker"] = req["ticker"]
    return out


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    queue = job["queue"]
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                    
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                
                if event["type"] in ["complete", "error"]:
                    break
        except asyncio.CancelledError:
            pass
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/catalog/categories")
def catalog_categories_list():
    labels = load_category_labels()
    counts = {row["category"]: row["count"] for row in category_counts()}
    seen: set[str] = set()
    out = []
    for slug, label in sorted(labels.items(), key=lambda x: x[1].lower()):
        seen.add(slug)
        out.append({"slug": slug, "label": label, "count": counts.get(slug, 0)})
    for slug, cnt in counts.items():
        if slug not in seen:
            out.append(
                {
                    "slug": slug,
                    "label": slug.replace("-", " ").title(),
                    "count": cnt,
                }
            )
    out.sort(key=lambda x: x["label"].lower())
    return out


@app.get("/api/catalog/items", response_model=List[CatalogItemOut])
def catalog_items_list(
    category: str | None = None,
    q: str | None = None,
    favorites_only: bool = False,
    sort: str = "ticker",
    order: str = "asc",
):
    rows = list_catalog_rows(
        category=category,
        q=q,
        favorites_only=favorites_only,
    )
    rows = sort_catalog_rows(rows, sort=sort, order=order)
    labels = load_category_labels()
    result = []
    for r in rows:
        slug = r["category"]
        result.append(
            CatalogItemOut(
                id=r["id"],
                ticker=r["ticker"],
                name=r.get("name"),
                category=slug,
                asset_type=r.get("asset_type") or "stock",
                source=r.get("source") or "seed",
                is_favorite=bool(r.get("is_favorite")),
                category_label=labels.get(slug, slug.replace("-", " ").title()),
            )
        )
    return result


@app.post("/api/catalog/favorites/{ticker}")
def catalog_set_favorite(ticker: str, body: FavoriteBody):
    set_catalog_favorite(ticker, body.favorite)
    return {"ticker": ticker.upper(), "favorite": body.favorite}


@app.post("/api/catalog/items", response_model=CatalogItemOut)
def catalog_add_item(body: CatalogUserAdd):
    try:
        row_id = add_catalog_item(
            body.ticker,
            body.name,
            body.category,
            body.asset_type,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Ticker already in catalog")
    labels = load_category_labels()
    slug = body.category.strip()
    return CatalogItemOut(
        id=row_id,
        ticker=body.ticker.upper().strip(),
        name=body.name.strip(),
        category=slug,
        asset_type=body.asset_type.lower().strip(),
        source="user",
        is_favorite=False,
        category_label=labels.get(slug, slug.replace("-", " ").title()),
    )


@app.get("/api/reports")
def list_reports():
    reports = []
    
    # Check results/ directory
    results_dir = Path("results")
    if results_dir.exists():
        for ticker_dir in results_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            for date_dir in ticker_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                reports_dir = date_dir / "reports"
                if reports_dir.exists():
                    reports.append({
                        "id": f"{ticker_dir.name}_{date_dir.name}",
                        "ticker": ticker_dir.name,
                        "date": date_dir.name,
                        "source": "results"
                    })
                    
    # Check reports/ directory
    reports_dir = Path("reports")
    if reports_dir.exists():
        for report_dir in reports_dir.iterdir():
            if not report_dir.is_dir() or not (report_dir / "complete_report.md").exists():
                continue
            
            # Parse ticker and timestamp from dir name (e.g. SPY_20260328_014110)
            parts = report_dir.name.split("_")
            if len(parts) >= 2:
                ticker = parts[0]
                date_str = parts[1]
                # Format date if possible
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                reports.append({
                    "id": report_dir.name,
                    "ticker": ticker,
                    "date": date_str,
                    "source": "reports"
                })
                
    # Sort by date descending
    reports.sort(key=lambda x: x["date"], reverse=True)
    return reports


@app.get("/api/reports/by-ticker/{ticker}/latest-decision")
def report_latest_decision(ticker: str):
    latest = get_latest_decision_for_ticker(ticker, PROJECT_ROOT)
    if not latest:
        raise HTTPException(status_code=404, detail="No saved decision for this ticker")
    return {
        "ticker": ticker.upper().strip(),
        "report_id": latest.report_id,
        "source": latest.source,
        "content": latest.content,
    }


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    # Check reports/ directory first
    reports_dir = Path("reports") / report_id
    complete_report = reports_dir / "complete_report.md"
    
    if complete_report.exists():
        with open(complete_report, "r") as f:
            return {"content": f.read()}
            
    # Check results/ directory
    parts = report_id.split("_")
    if len(parts) == 2:
        ticker, date = parts
        results_report_dir = Path("results") / ticker / date / "reports"
        
        if results_report_dir.exists():
            # Combine all markdown files into one report
            content = f"# Analysis Report: {ticker} on {date}\n\n"
            
            # Order of sections
            sections = [
                ("market_report.md", "Market Analysis"),
                ("sentiment_report.md", "Social Sentiment"),
                ("news_report.md", "News Analysis"),
                ("fundamentals_report.md", "Fundamentals Analysis"),
                ("investment_plan.md", "Research Team Decision"),
                ("trader_investment_plan.md", "Trading Team Plan"),
                ("final_trade_decision.md", "Portfolio Management Decision")
            ]
            
            for file_name, title in sections:
                file_path = results_report_dir / file_name
                if file_path.exists():
                    with open(file_path, "r") as f:
                        content += f"## {title}\n\n{f.read()}\n\n"
                        
            return {"content": content}
            
    raise HTTPException(status_code=404, detail="Report not found")

# ── RSI Optimizer ─────────────────────────────────────────────────────────────

@app.post("/api/rsi-optimize", response_model=RsiOptimizeResponse)
def rsi_optimize(req: RsiOptimizeRequest):
    """
    Run both RSI optimizers (pure algo + LLM-enhanced) in parallel and return
    a side-by-side comparison.

    Both optimizers run Walk-Forward Validation:
      - IS window:  req.is_days trading days of training data
      - OOS window: req.oos_days trading days of unseen validation data

    The LLM optimizer also classifies the market regime and narrows the search
    grid before running — this is what we compare against the full grid search.
    """
    from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import run_algo_optimizer
    from tradingagents.quant_ml.optimizers.rsi_optimizer_llm import run_llm_optimizer

    symbol = req.symbol.upper().strip()

    # Run both optimizers in parallel using the existing thread pool
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        algo_future = pool.submit(
            run_algo_optimizer,
            symbol, req.date, req.is_days, req.oos_days
        )
        llm_future = pool.submit(
            run_llm_optimizer,
            symbol, req.date, req.is_days, req.oos_days, req.llm_provider, req.llm_model
        )

        try:
            algo_raw = algo_future.result(timeout=120)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Algo optimizer failed: {e}")

        try:
            llm_raw = llm_future.result(timeout=120)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM optimizer failed: {e}")

    # ── Build structured results ───────────────────────────────────────────
    def _to_result(raw: dict) -> RsiOptimizerResult:
        return RsiOptimizerResult(
            optimal_period=raw["optimal_period"],
            optimal_upper=raw["optimal_upper"],
            optimal_lower=raw["optimal_lower"],
            is_sharpe=raw["is_sharpe"],
            oos_sharpe=raw["oos_sharpe"],
            confidence=raw["confidence"],
            combos_tested=raw["combos_tested"],
            is_days=raw["is_days"],
            oos_days=raw["oos_days"],
            default_is_sharpe=raw["default_is_sharpe"],
            default_oos_sharpe=raw["default_oos_sharpe"],
            period_sharpes=[RsiPeriodPoint(**p) for p in raw["period_sharpes"]],
            regime=raw.get("regime"),
            llm_reasoning=raw.get("llm_reasoning"),
            llm_available=raw.get("llm_available"),
            llm_grid=raw.get("llm_grid"),
            regime_signals=raw.get("regime_signals"),
            token_usage=raw.get("token_usage"),
            debug_logs=raw.get("debug_logs"),
        )

    algo_result = _to_result(algo_raw)
    llm_result  = _to_result(llm_raw)

    # ── Auto-generate comparison summary ──────────────────────────────────
    period_agreement = (algo_result.optimal_period == llm_result.optimal_period)

    if algo_result.oos_sharpe > llm_result.oos_sharpe + 0.05:
        oos_winner = "algo"
        oos_winner_sharpe = algo_result.oos_sharpe
    elif llm_result.oos_sharpe > algo_result.oos_sharpe + 0.05:
        oos_winner = "llm"
        oos_winner_sharpe = llm_result.oos_sharpe
    else:
        oos_winner = "tie"
        oos_winner_sharpe = max(algo_result.oos_sharpe, llm_result.oos_sharpe)

    algo_beats_default = algo_result.oos_sharpe > algo_result.default_oos_sharpe
    llm_beats_default  = llm_result.oos_sharpe > llm_result.default_oos_sharpe

    # Full grid size: 27 periods × 6 uppers × 6 lowers = 972 (approx)
    full_grid = 972
    combos_reduction_pct = round((1 - llm_result.combos_tested / full_grid) * 100, 1)

    # Build summary sentence
    if period_agreement:
        period_txt = f"Both optimizers agreed on RSI period {algo_result.optimal_period}"
    else:
        period_txt = (
            f"Algo chose period {algo_result.optimal_period}, "
            f"LLM chose period {llm_result.optimal_period}"
        )

    if oos_winner == "tie":
        winner_txt = "with similar OOS performance"
    elif oos_winner == "llm":
        winner_txt = (
            f"LLM achieved higher OOS Sharpe ({llm_result.oos_sharpe:.2f} vs "
            f"{algo_result.oos_sharpe:.2f}) using {combos_reduction_pct:.0f}% fewer combinations"
        )
    else:
        winner_txt = (
            f"Algo achieved higher OOS Sharpe ({algo_result.oos_sharpe:.2f} vs "
            f"{llm_result.oos_sharpe:.2f}) — LLM grid narrowing may have excluded better params"
        )

    summary = f"{period_txt}, {winner_txt}."

    comparison = RsiComparisonSummary(
        period_agreement=period_agreement,
        oos_winner=oos_winner,
        oos_winner_sharpe=oos_winner_sharpe,
        algo_beats_default=algo_beats_default,
        llm_beats_default=llm_beats_default,
        combos_reduction_pct=combos_reduction_pct,
        summary=summary,
    )

    response = RsiOptimizeResponse(
        symbol=symbol,
        date=req.date,
        algo=algo_result,
        llm=llm_result,
        comparison=comparison,
    )

    # ── Auto-save the winning result to rsi_cache.db ──────────────────────
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        from tradingagents.dataflows.rsi_cache import upsert_rsi_params
        if oos_winner == "llm":
            winner, provider = llm_result, "llm"
        else:
            winner, provider = algo_result, "algo"

        upsert_rsi_params(
            ticker             = symbol,
            optimal_period     = winner.optimal_period,
            optimal_upper      = winner.optimal_upper,
            optimal_lower      = winner.optimal_lower,
            oos_sharpe         = winner.oos_sharpe,
            is_sharpe          = winner.is_sharpe,
            confidence         = winner.confidence,
            regime             = llm_result.regime or "",
            training_days      = req.is_days,
            test_days          = req.oos_days,
            optimizer_provider = provider,
            model_used         = req.llm_model or "",
        )
        _log.info("RSI cache saved for %s (provider=%s)", symbol, provider)
    except Exception as _e:
        _log.error("RSI cache save failed for %s: %s", symbol, _e, exc_info=True)

    return response


@app.get("/api/rsi-cache")
def rsi_cache_list():
    """Return all cached RSI optimization results, newest optimized_at first."""
    from tradingagents.dataflows.rsi_cache import list_all_cached_tickers
    rows = list_all_cached_tickers()
    return {"entries": rows}


@app.get("/api/rsi-cache/{ticker}")
def rsi_cache_get(ticker: str):
    """Return cached RSI params for a specific ticker, or 404 if not cached."""
    from tradingagents.dataflows.rsi_cache import get_rsi_params
    params = get_rsi_params(ticker.upper().strip())
    if params is None:
        raise HTTPException(status_code=404, detail=f"No cached params for {ticker.upper()}")
    return params


@app.delete("/api/rsi-cache/{ticker}")
def rsi_cache_delete(ticker: str):
    """Delete the cached RSI params for a specific ticker."""
    from tradingagents.dataflows.rsi_cache import delete_rsi_params
    deleted = delete_rsi_params(ticker.upper().strip())
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No cache entry found for {ticker.upper()}")
    return {"deleted": ticker.upper()}


# ── RSI Signal endpoints ───────────────────────────────────────────────────────

@app.get("/api/rsi-signal/{ticker}", response_model=RsiSignalResponse)
def rsi_signal_get(ticker: str, date: str | None = None):
    """
    Full RSI signal for a single ticker:
      OPTIMIZE → GET PARAMS → CALCULATE RSI → SIGNAL → ACT

    Uses cached optimized params from /api/rsi-optimize if they exist;
    falls back to default RSI-14 / 70 / 30 otherwise.

    Query params:
      date  — as-of date YYYY-MM-DD (default: today)
    """
    from tradingagents.quant_ml.signals.rsi_signal import compute_rsi_signal
    raw = compute_rsi_signal(ticker.upper().strip(), as_of_date=date)
    history = [RsiHistoryPoint(**p) for p in raw.pop("rsi_history", [])]
    return RsiSignalResponse(**raw, rsi_history=history)


@app.get("/api/rsi-signals", response_model=List[RsiSignalResponse])
def rsi_signals_all(date: str | None = None):
    """
    Batch RSI signals for ALL tickers that have cached optimization params.
    One call to get BUY/SELL/HOLD for your entire watch-list.
    """
    from tradingagents.dataflows.rsi_cache import list_all_cached_tickers
    from tradingagents.quant_ml.signals.rsi_signal import compute_rsi_signal
    rows = list_all_cached_tickers()
    results = []
    for row in rows:
        raw = compute_rsi_signal(row["ticker"], as_of_date=date)
        history = [RsiHistoryPoint(**p) for p in raw.pop("rsi_history", [])]
        results.append(RsiSignalResponse(**raw, rsi_history=history))
    return results


# ── MACD Optimizer ────────────────────────────────────────────────────────────

_log = __import__("logging").getLogger(__name__)


@app.post("/api/macd-optimize", response_model=MacdOptimizeResponse)
def macd_optimize(req: MacdOptimizeRequest):
    """
    Run both MACD optimizers (pure algo + LLM-enhanced) in parallel and return
    a side-by-side comparison.

    Grid: fast[6-16] × slow[18-34] × signal[5-13], constraint fast < slow.
    LLM optimizer classifies market regime and narrows the 3D grid first.
    """
    from tradingagents.quant_ml.optimizers.macd_optimizer_algo import run_algo_optimizer
    from tradingagents.quant_ml.optimizers.macd_optimizer_llm import run_llm_optimizer
    import concurrent.futures

    symbol = req.symbol.upper().strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        algo_future = pool.submit(
            run_algo_optimizer,
            symbol, req.date, req.is_days, req.oos_days,
        )
        llm_future = pool.submit(
            run_llm_optimizer,
            symbol, req.date, req.is_days, req.oos_days, req.llm_provider, req.llm_model,
        )
        try:
            algo_raw = algo_future.result(timeout=180)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Algo optimizer failed: {e}")
        try:
            llm_raw = llm_future.result(timeout=180)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM optimizer failed: {e}")

    def _to_result(raw: dict) -> MacdOptimizerResult:
        return MacdOptimizerResult(
            optimal_fast=raw["optimal_fast"],
            optimal_slow=raw["optimal_slow"],
            optimal_signal=raw["optimal_signal"],
            is_sharpe=raw["is_sharpe"],
            oos_sharpe=raw["oos_sharpe"],
            confidence=raw["confidence"],
            combos_tested=raw["combos_tested"],
            is_days=raw["is_days"],
            oos_days=raw["oos_days"],
            default_is_sharpe=raw["default_is_sharpe"],
            default_oos_sharpe=raw["default_oos_sharpe"],
            param_sharpes=[MacdParamPoint(**p) for p in raw["param_sharpes"]],
            regime=raw.get("regime"),
            llm_reasoning=raw.get("llm_reasoning"),
            llm_available=raw.get("llm_available"),
            llm_grid=raw.get("llm_grid"),
            regime_signals=raw.get("regime_signals"),
            token_usage=raw.get("token_usage"),
            debug_logs=raw.get("debug_logs"),
        )

    algo_result = _to_result(algo_raw)
    llm_result  = _to_result(llm_raw)

    # ── Comparison summary ────────────────────────────────────────────────
    params_agreement = (
        algo_result.optimal_fast   == llm_result.optimal_fast and
        algo_result.optimal_slow   == llm_result.optimal_slow and
        algo_result.optimal_signal == llm_result.optimal_signal
    )

    if algo_result.oos_sharpe > llm_result.oos_sharpe + 0.05:
        oos_winner = "algo"
        oos_winner_sharpe = algo_result.oos_sharpe
    elif llm_result.oos_sharpe > algo_result.oos_sharpe + 0.05:
        oos_winner = "llm"
        oos_winner_sharpe = llm_result.oos_sharpe
    else:
        oos_winner = "tie"
        oos_winner_sharpe = max(algo_result.oos_sharpe, llm_result.oos_sharpe)

    algo_beats_default = algo_result.oos_sharpe > algo_result.default_oos_sharpe
    llm_beats_default  = llm_result.oos_sharpe  > llm_result.default_oos_sharpe

    # Combos reduction: how much smaller is LLM's search space vs full grid?
    full_combos = algo_result.combos_tested
    llm_combos  = llm_result.combos_tested
    combos_reduction_pct = round(
        (1 - llm_combos / max(full_combos, 1)) * 100, 1
    ) if full_combos > 0 else 0.0

    # Plain-English summary
    default_label = "MACD(12,26,9)"
    if oos_winner == "tie":
        winner_txt = f"Both optimizers tied (OOS Sharpe ≈ {oos_winner_sharpe:.3f})"
    else:
        winner_txt = f"{'Algo' if oos_winner == 'algo' else 'LLM'} optimizer won OOS (Sharpe {oos_winner_sharpe:.3f})"

    beats_txt = []
    if algo_beats_default: beats_txt.append("Algo")
    if llm_beats_default:  beats_txt.append("LLM")
    default_txt = (
        f"{' and '.join(beats_txt)} beat {default_label}"
        if beats_txt else f"Neither beat {default_label}"
    )

    params_txt = (
        f"Both agreed: MACD({algo_result.optimal_fast},{algo_result.optimal_slow},{algo_result.optimal_signal})."
        if params_agreement else
        f"Algo chose ({algo_result.optimal_fast},{algo_result.optimal_slow},{algo_result.optimal_signal}), "
        f"LLM chose ({llm_result.optimal_fast},{llm_result.optimal_slow},{llm_result.optimal_signal})."
    )

    llm_regime = llm_result.regime or "unknown"
    summary = (
        f"{winner_txt}. {default_txt}. {params_txt} "
        f"LLM classified regime as '{llm_regime.replace('_', ' ')}' and reduced search by {combos_reduction_pct:.0f}%."
    )

    # ── Auto-save winning params to macd_cache.db ─────────────────────────
    try:
        from tradingagents.dataflows.macd_cache import upsert_macd_params
        if oos_winner in ("algo", "tie"):
            winner_result = algo_result
            provider = "algo"
        else:
            winner_result = llm_result
            provider = req.llm_provider or "llm"

        upsert_macd_params(
            ticker=symbol,
            optimal_fast=winner_result.optimal_fast,
            optimal_slow=winner_result.optimal_slow,
            optimal_signal=winner_result.optimal_signal,
            oos_sharpe=winner_result.oos_sharpe,
            is_sharpe=winner_result.is_sharpe,
            confidence=winner_result.confidence,
            regime=winner_result.regime or "",
            training_days=req.is_days,
            test_days=req.oos_days,
            optimizer_provider=provider,
            model_used=req.llm_model or "",
        )
        _log.info("MACD cache saved for %s (provider=%s)", symbol, provider)
    except Exception as _e:
        _log.error("MACD cache save failed for %s: %s", symbol, _e, exc_info=True)

    return MacdOptimizeResponse(
        symbol=symbol,
        date=req.date,
        algo=algo_result,
        llm=llm_result,
        comparison=MacdComparisonSummary(
            params_agreement=params_agreement,
            oos_winner=oos_winner,
            oos_winner_sharpe=oos_winner_sharpe,
            algo_beats_default=algo_beats_default,
            llm_beats_default=llm_beats_default,
            combos_reduction_pct=combos_reduction_pct,
            summary=summary,
        ),
    )


@app.get("/api/macd-cache")
def macd_cache_list():
    """Return all cached MACD optimization results."""
    from tradingagents.dataflows.macd_cache import list_all_cached_tickers
    rows = list_all_cached_tickers()
    return {"entries": rows}


@app.delete("/api/macd-cache/{ticker}")
def macd_cache_delete(ticker: str):
    """Delete the cached MACD params for a specific ticker."""
    from tradingagents.dataflows.macd_cache import delete_macd_params
    deleted = delete_macd_params(ticker.upper().strip())
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No cache entry found for {ticker.upper()}")
    return {"deleted": ticker.upper()}


@app.get("/api/macd-signal/{ticker}", response_model=MacdSignalResponse)
def macd_signal_get(ticker: str, date: str | None = None):
    """
    Full MACD signal for a single ticker:
      MACD crossover → BUY / SELL / HOLD
    Uses cached optimized params if they exist; falls back to MACD(12,26,9).
    """
    from tradingagents.quant_ml.signals.macd_signal import compute_macd_signal
    raw = compute_macd_signal(ticker.upper().strip(), as_of_date=date)
    history = [MacdHistoryPoint(**p) for p in raw.pop("macd_history", [])]
    return MacdSignalResponse(**raw, macd_history=history)


@app.get("/api/macd-signals", response_model=List[MacdSignalResponse])
def macd_signals_all(date: str | None = None):
    """
    Batch MACD signals for ALL tickers that have cached optimization params.
    """
    from tradingagents.dataflows.macd_cache import list_all_cached_tickers
    from tradingagents.quant_ml.signals.macd_signal import compute_macd_signal
    rows = list_all_cached_tickers()
    results = []
    for row in rows:
        raw = compute_macd_signal(row["ticker"], as_of_date=date)
        history = [MacdHistoryPoint(**p) for p in raw.pop("macd_history", [])]
        results.append(MacdSignalResponse(**raw, macd_history=history))
    return results


# ── WFO Analyzer ─────────────────────────────────────────────────────────────

@app.post("/api/macd-wfo-analyze", response_model=WfoAnalysisResult)
def macd_wfo_analyze(req: WfoAnalyzeRequest):
    """
    Standalone Walk-Forward Optimization significance checker.

    Accepts raw IS/OOS Sharpe values + OOS trade count and returns:
      - Walk-Forward Efficiency (OOS/IS Sharpe ratio)
      - Sample size risk classification (HIGH / MEDIUM / LOW)
      - Composite Confidence Score 0–100
      - Narrative risk assessment
    """
    from tradingagents.quant_ml.walk_forward.wfo_analyzer import analyze_wfo
    analysis = analyze_wfo(
        is_days=req.is_days,
        oos_days=req.oos_days,
        is_sharpe=req.is_sharpe,
        oos_sharpe=req.oos_sharpe,
        oos_trade_count=req.oos_trade_count,
        bar_frequency=req.bar_frequency,
    )
    return WfoAnalysisResult(**analysis.to_dict())


# ── DT-Filtered MACD Optimizer ────────────────────────────────────────────────

@app.post("/api/macd-dt-optimize", response_model=MacdDtOptimizeResponse)
def macd_dt_optimize(req: MacdDtOptimizeRequest):
    """
    Run the Decision-Tree–filtered MACD optimizer on a single symbol.

    Trains two models in parallel:
      - Model A (Sliding WFO): IS=100 bars retrained every 30 bars
      - Model B (Fixed 10yr):  crash-regime-aware, shallower tree

    Returns a Long signal only when both models agree (P > 0.60).
    Position size is halved when ATR percentile > 80th (high-vol regime).
    """
    from tradingagents.quant_ml.optimizers.macd_dt_optimizer import run_dt_optimizer

    symbol = req.symbol.upper().strip()
    try:
        raw = run_dt_optimizer(
            symbol=symbol,
            curr_date=req.date,
            is_days=req.is_days,
            oos_days=req.oos_days,
            label_horizon=req.label_horizon,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DT optimizer failed: {e}")

    slides = [MacdDtSlideResult(**{k: v for k, v in s.items()
                                   if k in MacdDtSlideResult.__fields__})
              for s in raw.get("slides", [])]
    return MacdDtOptimizeResponse(
        symbol=raw["symbol"],
        curr_date=raw["curr_date"],
        last_signal=raw["last_signal"],
        last_prob_a=raw.get("last_prob_a", 0.0),
        last_prob_b=raw.get("last_prob_b", raw.get("last_prob_a", 0.0)),
        last_size_mult=raw.get("last_size_mult", 1.0),
        last_atr_pct=raw.get("last_atr_pct", 0.0),
        avg_oos_acc=raw.get("avg_oos_acc", 0.0),
        n_slides=raw.get("n_slides", 0),
        feature_names=raw.get("feature_names", []),
        macd_params=raw.get("macd_params", {}),
        is_days=raw.get("is_days", req.is_days),
        oos_days=raw.get("oos_days", req.oos_days),
        label_horizon=raw.get("label_horizon", req.label_horizon),
        min_prob_threshold=raw.get("min_prob_threshold", 0.55),
        slides=slides,
    )


@app.post("/api/macd-dt-portfolio", response_model=MacdDtPortfolioResponse)
def macd_dt_portfolio(req: MacdDtPortfolioRequest):
    """
    Run DT-filtered MACD optimizer across a universe of symbols and build
    a portfolio obeying the 5%-rule and 20-position cap.

    Liquidity gate: skips symbols with < $5M 30-day avg dollar volume.
    Returns ranked positions, rejected symbols, and capital allocation.
    """
    from tradingagents.quant_ml.walk_forward.macd_dt_portfolio import run_portfolio_optimizer

    symbols = [s.upper().strip() for s in req.symbols if s.strip()]
    if not symbols:
        raise HTTPException(status_code=422, detail="Provide at least one symbol.")
    try:
        raw = run_portfolio_optimizer(
            symbols=symbols,
            curr_date=req.date,
            capital=req.capital,
            min_dollar_vol=req.min_dollar_vol,
            max_positions=req.max_positions,
            is_days=req.is_days,
            oos_days=req.oos_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimizer failed: {e}")

    positions = [MacdDtPosition(**p) for p in raw.get("positions", [])]
    return MacdDtPortfolioResponse(
        positions=positions,
        rejected_liquidity=raw["rejected_liquidity"],
        rejected_no_signal=raw["rejected_no_signal"],
        rejected_errors=raw["rejected_errors"],
        total_deployed_pct=raw["total_deployed_pct"],
        total_deployed_usd=raw["total_deployed_usd"],
        capital=raw["capital"],
        curr_date=raw["curr_date"],
        n_evaluated=raw["n_evaluated"],
        n_selected=raw["n_selected"],
        base_alloc_pct=raw["base_alloc_pct"],
        max_positions=raw["max_positions"],
        min_dollar_vol=raw["min_dollar_vol"],
    )


# ── DT-Filtered RSI Optimizer ─────────────────────────────────────────────────

@app.post("/api/rsi-dt-optimize", response_model=RsiDtOptimizeResponse)
def rsi_dt_optimize(req: RsiDtOptimizeRequest):
    """
    Run the Decision-Tree–filtered RSI Mean-Reversion optimizer on a single symbol.

    Trains two models in parallel:
      - Model A (Sliding WFO): IS=100 bars retrained every 30 bars
      - Model B (Fixed 10yr):  crash-regime-aware, shallower tree

    Features: RSI(14), EMA Ratio, MACD Histogram, Bollinger %B, ATR-14, Volume Ratio.
    Returns a Long signal only when both models agree (P > 0.60).
    Position size is halved when ATR percentile > 80th (high-vol regime).
    """
    from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import run_rsi_dt_optimizer

    symbol = req.symbol.upper().strip()
    try:
        raw = run_rsi_dt_optimizer(
            symbol=symbol,
            curr_date=req.date,
            is_days=req.is_days,
            oos_days=req.oos_days,
            label_horizon=req.label_horizon,
            rsi_period=req.rsi_period,
            rsi_upper=req.rsi_upper,
            rsi_lower=req.rsi_lower,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RSI DT optimizer failed: {e}")

    slides = [RsiDtSlideResult(**{k: v for k, v in s.items()
                                  if k in RsiDtSlideResult.__fields__})
              for s in raw.get("slides", [])]
    return RsiDtOptimizeResponse(
        symbol=raw["symbol"],
        curr_date=raw["curr_date"],
        last_signal=raw["last_signal"],
        last_prob_a=raw.get("last_prob_a", 0.0),
        last_prob_b=raw.get("last_prob_b", raw.get("last_prob_a", 0.0)),
        last_size_mult=raw.get("last_size_mult", 1.0),
        last_atr_pct=raw.get("last_atr_pct", 0.0),
        last_rsi=raw.get("last_rsi", 0.0),
        avg_oos_acc=raw.get("avg_oos_acc", 0.0),
        n_slides=raw.get("n_slides", 0),
        feature_names=raw.get("feature_names", []),
        rsi_params=raw.get("rsi_params", {
            "period": req.rsi_period, "upper": req.rsi_upper, "lower": req.rsi_lower,
        }),
        wfo_params_used=raw.get("wfo_params_used", False),
        is_days=raw.get("is_days", req.is_days),
        oos_days=raw.get("oos_days", req.oos_days),
        label_horizon=raw.get("label_horizon", req.label_horizon),
        min_prob_threshold=raw.get("min_prob_threshold", 0.55),
        slides=slides,
    )


@app.post("/api/rsi-dt-portfolio", response_model=RsiDtPortfolioResponse)
def rsi_dt_portfolio(req: RsiDtPortfolioRequest):
    """
    Run DT-filtered RSI optimizer across a universe of symbols and build
    a portfolio obeying the 5%-rule and 20-position cap.

    Liquidity gate: skips symbols with < $5M 30-day avg dollar volume.
    Returns ranked positions, rejected symbols, and capital allocation.
    """
    from tradingagents.quant_ml.walk_forward.rsi_dt_portfolio import run_rsi_portfolio_optimizer

    symbols = [s.upper().strip() for s in req.symbols if s.strip()]
    if not symbols:
        raise HTTPException(status_code=422, detail="Provide at least one symbol.")
    try:
        raw = run_rsi_portfolio_optimizer(
            symbols=symbols,
            curr_date=req.date,
            capital=req.capital,
            min_dollar_vol=req.min_dollar_vol,
            max_positions=req.max_positions,
            is_days=req.is_days,
            oos_days=req.oos_days,
            rsi_period=req.rsi_period,
            rsi_upper=req.rsi_upper,
            rsi_lower=req.rsi_lower,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RSI portfolio optimizer failed: {e}")

    positions = [RsiDtPosition(**p) for p in raw.get("positions", [])]
    return RsiDtPortfolioResponse(
        positions=positions,
        rejected_liquidity=raw["rejected_liquidity"],
        rejected_no_signal=raw["rejected_no_signal"],
        rejected_errors=raw["rejected_errors"],
        total_deployed_pct=raw["total_deployed_pct"],
        total_deployed_usd=raw["total_deployed_usd"],
        capital=raw["capital"],
        curr_date=raw["curr_date"],
        n_evaluated=raw["n_evaluated"],
        n_selected=raw["n_selected"],
        base_alloc_pct=raw["base_alloc_pct"],
        max_positions=raw["max_positions"],
        min_dollar_vol=raw["min_dollar_vol"],
    )


# ── RSI DT Feature Lab ───────────────────────────────────────────────────────

@app.post("/api/rsi-dt-feature-lab", response_model=RsiDtFeatureLabResponse)
def rsi_dt_feature_lab(req: RsiDtFeatureLabRequest):
    """
    Run selected RSI DT rule sets for a symbol and return side-by-side OOS
    comparison. Data and sentiment are fetched once; rule sets run in parallel.
    Results are persisted to data/rsi_dt_experiments.db.
    """
    from tradingagents.quant_ml.experiments.experiment_runner import run_feature_lab
    import uuid, datetime as _dt

    symbol = req.symbol.upper().strip()
    run_id = str(uuid.uuid4())
    try:
        results = run_feature_lab(
            symbol=symbol,
            curr_date=req.date,
            rule_set_names=req.rule_sets,
            custom_features=req.custom_features,
            custom_vetos=req.custom_vetos,
            is_days=req.is_days,
            oos_days=req.oos_days,
            label_horizon=req.label_horizon,
            tp_mult=req.tp_mult,
            sl_mult=req.sl_mult,
            rsi_period=req.rsi_period,
            run_id=run_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature lab failed: {e}")

    winner = max(results, key=lambda r: r["oos_sharpe"])["rule_set_name"] if results else None
    ran_at = _dt.datetime.utcnow().isoformat()
    return RsiDtFeatureLabResponse(
        run_id=run_id,
        symbol=symbol,
        date=req.date,
        results=[RsiDtFeatureLabResultRow(**{k: v for k, v in r.items() if k in RsiDtFeatureLabResultRow.model_fields}) for r in results],
        winner=winner,
        ran_at=ran_at,
    )


@app.get("/api/rsi-dt-feature-lab/{symbol}", response_model=List[RsiDtFeatureLabResultRow])
def rsi_dt_feature_lab_history(symbol: str, limit: int = 50):
    """Retrieve past Feature Lab experiment runs for a symbol."""
    from tradingagents.quant_ml.experiments.experiment_db import get_experiment_runs
    rows = get_experiment_runs(symbol.upper().strip(), limit=min(limit, 200))
    return [RsiDtFeatureLabResultRow(**r) for r in rows]


# ── OOS Integration Tests ─────────────────────────────────────────────────────

def _build_oos_dates(oos_start: str, oos_end: str, n: int) -> list[str]:
    """Return n evenly-spaced business dates between oos_start and oos_end."""
    import pandas as pd
    import numpy as np
    bdays = pd.bdate_range(start=oos_start, end=oos_end)
    if len(bdays) == 0:
        return []
    if len(bdays) <= n:
        return [d.strftime("%Y-%m-%d") for d in bdays]
    indices = np.linspace(0, len(bdays) - 1, n, dtype=int)
    return [bdays[i].strftime("%Y-%m-%d") for i in indices]


def _build_oos_summary(rows: list, notional: float) -> OosTestSummary:
    import numpy as np
    valid = [r for r in rows if r.error is None and r.pnl_dollar is not None]
    wins   = [r.pnl_dollar for r in valid if r.pnl_dollar > 0]
    losses = [r.pnl_dollar for r in valid if r.pnl_dollar < 0]
    total  = sum(r.pnl_dollar for r in valid)
    pf     = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    avg_oos = float(np.mean([r.oos_sharpe for r in valid if r.oos_sharpe is not None])) if valid else 0.0
    avg_is  = float(np.mean([r.is_sharpe  for r in valid if r.is_sharpe  is not None])) if valid else 0.0
    return OosTestSummary(
        total_pnl_dollar=round(total, 2),
        win_rate_pct=round(len(wins) / len(valid) * 100, 1) if valid else 0.0,
        n_wins=len(wins),
        n_losses=len(losses),
        avg_win_dollar=round(float(np.mean(wins)), 2) if wins else 0.0,
        avg_loss_dollar=round(float(np.mean(losses)), 2) if losses else 0.0,
        profit_factor=round(pf, 3) if pf != float("inf") else 9999.0,
        avg_oos_sharpe=round(avg_oos, 4),
        avg_is_sharpe=round(avg_is, 4),
        high_conf=sum(1 for r in valid if r.confidence == "HIGH"),
        med_conf=sum(1 for r in valid if r.confidence == "MEDIUM"),
        low_conf=sum(1 for r in valid if r.confidence == "LOW"),
    )


@app.post("/api/macd-oos-test", response_model=OosTestResponse)
def macd_oos_test(req: OosTestRequest):
    """
    Walk-forward OOS sweep for the MACD optimizer.
    Runs the algo optimizer across n_dates evenly-spaced historical dates,
    computes simulated dollar P&L for each OOS window, and returns a summary.
    """
    import pandas as pd
    import numpy as np
    from tradingagents.quant_ml.optimizers.macd_optimizer_algo import (
        run_algo_optimizer,
        _load_price_data,
        _calc_macd,
        _generate_positions,
    )
    from tradingagents.dataflows.config import set_config
    set_config({})

    oos_start = req.oos_start or (
        pd.to_datetime(req.oos_end) - pd.DateOffset(months=18)
    ).strftime("%Y-%m-%d")

    dates = _build_oos_dates(oos_start, req.oos_end, req.n_dates)
    if not dates:
        raise HTTPException(status_code=400, detail="No business days found in the given date range.")

    rows: list[OosTestDateRow] = []

    for curr_date in dates:
        try:
            res = run_algo_optimizer(
                req.symbol.upper(), curr_date,
                is_days=req.is_days, oos_days=req.oos_days,
            )
            data = _load_price_data(req.symbol.upper(), freq="D")
            data = data[data["Date"] <= pd.to_datetime(curr_date)].copy()
            closes = data["Close"].reset_index(drop=True)
            total = len(closes)
            oos_start_idx = total - req.oos_days

            _, _, hist = _calc_macd(closes, res["optimal_fast"], res["optimal_slow"], res["optimal_signal"])
            positions  = _generate_positions(hist)
            oos_closes = closes.iloc[oos_start_idx:].reset_index(drop=True)
            oos_pos    = positions.iloc[oos_start_idx:].reset_index(drop=True)

            pos_shifted   = oos_pos.shift(1).fillna(0)
            price_returns = oos_closes.pct_change().fillna(0)
            strat_returns = pos_shifted * price_returns
            trades        = oos_pos.diff().fillna(0) != 0
            strat_returns = strat_returns - trades.astype(float) * 0.001

            equity     = (1 + strat_returns).cumprod() * req.notional
            pnl_pct    = float((equity.iloc[-1] / req.notional - 1) * 100) if len(equity) else 0.0
            pnl_dollar = float(equity.iloc[-1] - req.notional) if len(equity) else 0.0
            rolling_max = equity.cummax()
            dd = (equity - rolling_max) / rolling_max
            max_dd = float(abs(dd.min()) * 100) if len(dd) else 0.0
            winning_days = int((strat_returns > 0).sum())
            active_days  = int((pos_shifted != 0).sum())
            win_rate     = (winning_days / active_days * 100) if active_days > 0 else 0.0

            rows.append(OosTestDateRow(
                date=curr_date,
                fast=res["optimal_fast"],
                slow=res["optimal_slow"],
                signal_period=res["optimal_signal"],
                is_sharpe=res["is_sharpe"],
                oos_sharpe=res["oos_sharpe"],
                confidence=res["confidence"],
                pnl_pct=round(pnl_pct, 2),
                pnl_dollar=round(pnl_dollar, 2),
                win_rate=round(win_rate, 1),
                max_dd_pct=round(max_dd, 2),
                n_trades=int(trades.sum()),
            ))
        except Exception as e:
            rows.append(OosTestDateRow(date=curr_date, error=str(e)))

    summary = _build_oos_summary(rows, req.notional)
    return OosTestResponse(
        symbol=req.symbol.upper(),
        oos_start=oos_start,
        oos_end=req.oos_end,
        is_days=req.is_days,
        oos_days=req.oos_days,
        notional=req.notional,
        n_requested=len(dates),
        n_succeeded=sum(1 for r in rows if r.error is None),
        rows=rows,
        summary=summary,
        algo_type="macd",
    )


@app.post("/api/rsi-oos-test", response_model=OosTestResponse)
def rsi_oos_test(req: OosTestRequest):
    """
    Walk-forward OOS sweep for the RSI optimizer.
    Runs the algo optimizer across n_dates evenly-spaced historical dates,
    computes simulated dollar P&L for each OOS window, and returns a summary.
    """
    import pandas as pd
    import numpy as np
    from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import (
        run_algo_optimizer,
        _load_price_data,
        _calc_rsi,
        _generate_positions,
    )
    from tradingagents.dataflows.config import set_config
    set_config({})

    oos_start = req.oos_start or (
        pd.to_datetime(req.oos_end) - pd.DateOffset(months=18)
    ).strftime("%Y-%m-%d")

    dates = _build_oos_dates(oos_start, req.oos_end, req.n_dates)
    if not dates:
        raise HTTPException(status_code=400, detail="No business days found in the given date range.")

    rows: list[OosTestDateRow] = []

    for curr_date in dates:
        try:
            res = run_algo_optimizer(
                req.symbol.upper(), curr_date,
                is_days=req.is_days, oos_days=req.oos_days,
            )
            data = _load_price_data(req.symbol.upper())
            data = data[data["Date"] <= pd.to_datetime(curr_date)].copy()
            closes = data["Close"].reset_index(drop=True)
            total = len(closes)
            oos_start_idx = total - req.oos_days

            rsi_full  = _calc_rsi(closes, res["optimal_period"])
            positions = _generate_positions(rsi_full, res["optimal_upper"], res["optimal_lower"])
            oos_closes = closes.iloc[oos_start_idx:].reset_index(drop=True)
            oos_pos    = positions.iloc[oos_start_idx:].reset_index(drop=True)

            pos_shifted   = oos_pos.shift(1).fillna(0)
            price_returns = oos_closes.pct_change().fillna(0)
            strat_returns = pos_shifted * price_returns

            equity     = (1 + strat_returns).cumprod() * req.notional
            pnl_pct    = float((equity.iloc[-1] / req.notional - 1) * 100) if len(equity) else 0.0
            pnl_dollar = float(equity.iloc[-1] - req.notional) if len(equity) else 0.0
            rolling_max = equity.cummax()
            dd = (equity - rolling_max) / rolling_max
            max_dd = float(abs(dd.min()) * 100) if len(dd) else 0.0
            winning_days = int((strat_returns > 0).sum())
            active_days  = int((pos_shifted != 0).sum())
            win_rate     = (winning_days / active_days * 100) if active_days > 0 else 0.0

            rows.append(OosTestDateRow(
                date=curr_date,
                period=res["optimal_period"],
                upper=float(res["optimal_upper"]),
                lower=float(res["optimal_lower"]),
                is_sharpe=res["is_sharpe"],
                oos_sharpe=res["oos_sharpe"],
                confidence=res["confidence"],
                pnl_pct=round(pnl_pct, 2),
                pnl_dollar=round(pnl_dollar, 2),
                win_rate=round(win_rate, 1),
                max_dd_pct=round(max_dd, 2),
                n_trades=0,
            ))
        except Exception as e:
            rows.append(OosTestDateRow(date=curr_date, error=str(e)))

    summary = _build_oos_summary(rows, req.notional)
    return OosTestResponse(
        symbol=req.symbol.upper(),
        oos_start=oos_start,
        oos_end=req.oos_end,
        is_days=req.is_days,
        oos_days=req.oos_days,
        notional=req.notional,
        n_requested=len(dates),
        n_succeeded=sum(1 for r in rows if r.error is None),
        rows=rows,
        summary=summary,
        algo_type="rsi",
    )


@app.get("/api/sniper-signal/{ticker}")
def sniper_signal_get(ticker: str, date: str | None = None):
    """Return latest Sniper action, grades, SL/TP, vol regime for a ticker."""
    from tradingagents.quant_ml.signals.sniper_signal import compute_sniper_signal
    result = compute_sniper_signal(ticker.upper().strip(), as_of_date=date)
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/api/sniper-optimize", response_model=SniperOptimizeResponse)
def sniper_optimize(req: SniperOptimizeRequest):
    """
    Run the full Sniper optimization pipeline for a symbol:
      1. RSI algo WFO   → rsi_cache.db
      2. MACD algo WFO  → macd_cache.db
      3. EMA algo WFO   → ema_cache.db
      4. Sniper DT WFO  → sniper_cache.db (DT thresholds)

    The DT optimizer reads EMA params from ema_cache, so step 3 must complete first.
    """
    from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import (
        run_algo_optimizer as rsi_algo,
    )
    from tradingagents.quant_ml.optimizers.macd_optimizer_algo import (
        run_algo_optimizer as macd_algo,
    )
    from tradingagents.quant_ml.optimizers.ema_optimizer_algo import (
        run_algo_optimizer as ema_algo,
    )
    from tradingagents.quant_ml.optimizers.sniper_dt_optimizer import (
        run_sniper_dt_optimizer,
    )
    from tradingagents.dataflows.sniper_cache import upsert_sniper_params

    sym = req.symbol.upper().strip()

    rsi_result  = rsi_algo(sym,  req.date, req.is_days, req.oos_days)
    macd_result = macd_algo(sym, req.date, req.is_days, req.oos_days)
    ema_result  = ema_algo(sym,  req.date, req.is_days, req.oos_days)

    dt_result = run_sniper_dt_optimizer(
        sym, req.date,
        is_days=req.dt_is_days,
        oos_days=req.dt_oos_days,
    )

    upsert_sniper_params(
        ticker=sym,
        threshold_a=float(dt_result["threshold_a"]),
        threshold_b=float(dt_result["threshold_b"]),
        dt_oos_sharpe=float(dt_result["oos_sharpe"]),
        dt_oos_hit_rate=float(dt_result["oos_hit_rate"]),
        training_days=int(req.dt_is_days),
        test_days=int(req.dt_oos_days),
    )

    ema = SniperEmaResult(
        optimal_fast=int(ema_result["optimal_fast"]),
        optimal_slow=int(ema_result["optimal_slow"]),
        optimal_trend=int(ema_result["optimal_trend"]),
        is_sharpe=float(ema_result["is_sharpe"]),
        oos_sharpe=float(ema_result["oos_sharpe"]),
        confidence=str(ema_result["confidence"]),
        combos_tested=int(ema_result["combos_tested"]),
        default_oos_sharpe=float(ema_result["default_oos_sharpe"]),
    )
    dt = SniperDtResult(
        last_signal=int(dt_result["last_signal"]),
        last_prob_a=float(dt_result["last_prob_a"]),
        last_prob_b=float(dt_result["last_prob_b"]),
        threshold_a=float(dt_result["threshold_a"]),
        threshold_b=float(dt_result["threshold_b"]),
        oos_sharpe=float(dt_result["oos_sharpe"]),
        oos_hit_rate=float(dt_result["oos_hit_rate"]),
        bull_score_today=float(dt_result["bull_score_today"]),
        grade_veto_ok=bool(dt_result["grade_veto_ok"]),
        confidence=str(dt_result["confidence"]),
    )
    summary = (
        f"EMA({ema.optimal_fast}/{ema.optimal_slow}/{ema.optimal_trend}) "
        f"OOS {ema.oos_sharpe:.2f} [{ema.confidence}] | "
        f"DT OOS {dt.oos_sharpe:.2f} [{dt.confidence}]"
    )
    return SniperOptimizeResponse(
        symbol=sym,
        date=req.date,
        ema=ema,
        dt=dt,
        rsi=rsi_result,
        macd=macd_result,
        summary=summary,
    )


# Serve the built frontend (production). No-op in local dev if dist/ is absent.
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve index.html for any non-API route so client-side routing works."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(_FRONTEND_DIST / "index.html")
