from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import yfinance as yf
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
    PortfolioPosition,
    PortfolioCreateUpdate,
    AnalysisRequest,
    CatalogItemOut,
    CatalogUserAdd,
    FavoriteBody,
    YouTubeSummarizeRequest,
    YouTubeSummarizeResponse,
    YouTubeSummaryListItem,
    YouTubeSummaryDetail,
)
from . import youtube_summary
from .database import (
    init_db,
    get_all_positions,
    add_position,
    update_position,
    delete_position,
    update_prices,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        data = youtube_summary.summarize_youtube_url(body.url.strip())
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
                report_id = f"FORM_{ts}"
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

@app.get("/api/portfolio/positions", response_model=List[PortfolioPosition])
def read_positions():
    rows = get_all_positions()
    positions = []
    for row in rows:
        pos = PortfolioPosition(**row)
        if pos.current_price is not None:
            pos.market_value = pos.quantity * pos.current_price
            total_cost = pos.quantity * pos.cost_basis
            pos.unrealized_pnl = pos.market_value - total_cost
            if total_cost > 0:
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / total_cost) * 100
        positions.append(pos)
    return positions

@app.post("/api/portfolio/positions", response_model=PortfolioPosition)
def create_position(pos: PortfolioCreateUpdate):
    pos_id = add_position(pos.ticker, pos.quantity, pos.cost_basis)
    return PortfolioPosition(id=pos_id, **pos.dict())

@app.put("/api/portfolio/positions/{pos_id}", response_model=PortfolioPosition)
def edit_position(pos_id: int, pos: PortfolioCreateUpdate):
    update_position(pos_id, pos.ticker, pos.quantity, pos.cost_basis)
    return PortfolioPosition(id=pos_id, **pos.dict())

@app.delete("/api/portfolio/positions/{pos_id}")
def remove_position(pos_id: int):
    delete_position(pos_id)
    return {"status": "success"}

@app.post("/api/portfolio/refresh-prices")
def refresh_prices():
    positions = get_all_positions()
    tickers = list(set(p["ticker"] for p in positions))
    
    if not tickers:
        return {"status": "no positions"}
        
    try:
        data = yf.download(tickers, period="1d")
        prices = {}
        
        if len(tickers) == 1:
            if not data.empty and 'Close' in data:
                prices[tickers[0]] = float(data['Close'].iloc[-1])
        else:
            if not data.empty and 'Close' in data:
                for ticker in tickers:
                    if ticker in data['Close']:
                        prices[ticker] = float(data['Close'][ticker].iloc[-1])
                        
        update_prices(prices)
        return {"status": "success", "updated": len(prices)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
