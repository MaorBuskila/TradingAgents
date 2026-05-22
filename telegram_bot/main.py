"""Telegram bot: manage catalog favorites, run analysis via API, scheduled decision reminders."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import date

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
API_BASE = os.getenv("TRADINGAGENTS_API_BASE", "http://127.0.0.1:8000/api").rstrip("/")
ALLOWED_RAW = os.getenv("ALLOWED_CHAT_IDS", "").strip()
BOT_SECRET = os.getenv("BOT_API_SECRET", "").strip()
REMINDER_ENABLED = os.getenv("TELEGRAM_REMINDER_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
REMINDER_HOUR = int(os.getenv("TELEGRAM_REMINDER_HOUR", "9"))
REMINDER_MINUTE = int(os.getenv("TELEGRAM_REMINDER_MINUTE", "0"))
REMINDER_CHAT_RAW = os.getenv("TELEGRAM_REMINDER_CHAT_ID", "").strip()

TG_MSG_MAX = 3800


def _parse_allowed_chats() -> set[int] | None:
    if not ALLOWED_RAW:
        return None
    out: set[int] = set()
    for part in ALLOWED_RAW.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


ALLOWED_CHATS = _parse_allowed_chats()


def _bot_headers() -> dict[str, str]:
    h: dict[str, str] = {}
    if BOT_SECRET:
        h["X-TradingAgents-Bot-Secret"] = BOT_SECRET
    return h


def _truncate(text: str, max_len: int = TG_MSG_MAX) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=API_BASE,
        headers=_bot_headers(),
        timeout=httpx.Timeout(120.0, connect=10.0),
    )


async def _ensure_allowed(update: Update) -> bool:
    if ALLOWED_CHATS is None:
        return True
    cid = update.effective_chat.id if update.effective_chat else None
    if cid is None or cid not in ALLOWED_CHATS:
        if update.message:
            await update.message.reply_text("Not authorized.")
        return False
    return True


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    await update.message.reply_text(
        "TradingAgents bot\n\n"
        "/favorites — list favorite tickers\n"
        "/fav TICKER — add to favorites (adds to catalog if needed)\n"
        "/unfav TICKER — remove favorite\n"
        "/addticker TICKER [category] — add custom catalog row (default category: first in list)\n"
        "/analyze TICKER — run full analysis (may take several minutes)\n"
        "/summary TICKER — latest saved portfolio decision\n"
        "/digest — latest decision for each favorite\n"
        "/help — this message"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def _default_category(client: httpx.AsyncClient) -> str:
    r = await client.get("/catalog/categories")
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return "green-energy"
    return rows[0]["slug"]


async def _ensure_catalog_ticker(client: httpx.AsyncClient, ticker: str) -> None:
    t = ticker.upper().strip()
    r = await client.get("/catalog/items", params={"q": t})
    r.raise_for_status()
    items = r.json()
    for row in items:
        if row.get("ticker", "").upper() == t:
            return
    cat = await _default_category(client)
    ar = await client.post(
        "/catalog/items",
        json={"ticker": t, "name": "", "category": cat, "asset_type": "stock"},
    )
    if ar.status_code == 409:
        return
    ar.raise_for_status()


async def cmd_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    async with _client() as client:
        r = await client.get("/catalog/items", params={"favorites_only": True, "sort": "ticker"})
        r.raise_for_status()
        rows = r.json()
    if not rows:
        await update.message.reply_text("No favorites yet. Use /fav TICKER")
        return
    lines = [f"• {row['ticker']}" + (f" — {row.get('name')}" if row.get("name") else "") for row in rows]
    await update.message.reply_text("Favorites:\n" + "\n".join(lines))


async def cmd_fav(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /fav TICKER")
        return
    ticker = context.args[0].upper().strip()
    async with _client() as client:
        await _ensure_catalog_ticker(client, ticker)
        r = await client.post(f"/catalog/favorites/{ticker}", json={"favorite": True})
        r.raise_for_status()
    await update.message.reply_text(f"Favorited {ticker}.")


async def cmd_unfav(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /unfav TICKER")
        return
    ticker = context.args[0].upper().strip()
    async with _client() as client:
        r = await client.post(f"/catalog/favorites/{ticker}", json={"favorite": False})
        r.raise_for_status()
    await update.message.reply_text(f"Removed favorite {ticker}.")


async def cmd_addticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /addticker TICKER [category_slug]")
        return
    ticker = context.args[0].upper().strip()
    category = context.args[1].strip() if len(context.args) > 1 else None
    async with _client() as client:
        if not category:
            category = await _default_category(client)
        r = await client.post(
            "/catalog/items",
            json={"ticker": ticker, "name": "", "category": category, "asset_type": "stock"},
        )
        if r.status_code == 409:
            await update.message.reply_text(f"{ticker} is already in the catalog.")
            return
        r.raise_for_status()
    await update.message.reply_text(f"Added {ticker} to catalog (category: {category}).")


def _build_analyze_payload(defaults: dict, ticker: str) -> dict:
    """Same shape as cli/main.py get_user_selections() for AnalysisRunner (via API)."""
    p = (defaults.get("llm_provider") or "openai").lower()
    rd = defaults.get("research_depth", 1)
    try:
        research_depth = int(rd)
    except (TypeError, ValueError):
        research_depth = 1
    return {
        "ticker": ticker.upper().strip(),
        "analysis_date": date.today().isoformat(),
        "analysts": ["market", "social", "news", "fundamentals"],
        "research_depth": research_depth,
        "llm_provider": p,
        "backend_url": defaults["backend_url"],
        "shallow_thinker": defaults["shallow_thinker"],
        "deep_thinker": defaults["deep_thinker"],
        "google_thinking_level": defaults.get("google_thinking_level"),
        "openai_reasoning_effort": defaults.get("openai_reasoning_effort"),
        "anthropic_effort": defaults.get("anthropic_effort"),
    }


async def _poll_job(client: httpx.AsyncClient, job_id: str, timeout_s: float = 1800.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        jr = await client.get(f"/jobs/{job_id}")
        if jr.status_code == 404:
            return {"status": "unknown", "error": "Job not found (API may have restarted)"}
        jr.raise_for_status()
        data = jr.json()
        st = data.get("status")
        if st == "completed":
            return data
        if st == "failed":
            return data
        await asyncio.sleep(3)
    return {"status": "timeout", "error": "Analysis still running; check the web UI or try /summary later."}


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /analyze TICKER")
        return
    ticker = context.args[0].upper().strip()
    await update.message.reply_text(f"Starting analysis for {ticker}… (this can take a while)")
    async with _client() as client:
        dr = await client.get("/analysis-defaults")
        dr.raise_for_status()
        defaults = dr.json()
        payload = _build_analyze_payload(defaults, ticker)
        ar = await client.post("/analyze", json=payload)
        if ar.status_code == 400:
            await update.message.reply_text(f"API error: {ar.text}")
            return
        ar.raise_for_status()
        job_id = ar.json()["job_id"]
        result = await _poll_job(client, job_id)
    st = result.get("status")
    if st == "completed":
        rid = result.get("report_id") or "—"
        dec = result.get("decision") or ""
        msg = f"Done. Report: {rid}\n\n{_truncate(str(dec))}"
        await update.message.reply_text(msg)
        if err := result.get("report_save_error"):
            await update.message.reply_text(f"Report save note: {err}")
    elif st == "failed":
        await update.message.reply_text(f"Analysis failed: {result.get('error', 'unknown')}")
    else:
        await update.message.reply_text(result.get("error", str(result)))


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /summary TICKER")
        return
    ticker = context.args[0].upper().strip()
    async with _client() as client:
        r = await client.get(f"/reports/by-ticker/{ticker}/latest-decision")
        if r.status_code == 404:
            await update.message.reply_text(f"No saved decision for {ticker} yet.")
            return
        r.raise_for_status()
        data = r.json()
    header = f"{ticker} (report {data.get('report_id')}, {data.get('source')})\n\n"
    await update.message.reply_text(_truncate(header + data.get("content", "")))


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_allowed(update):
        return
    async with _client() as client:
        fr = await client.get("/catalog/items", params={"favorites_only": True, "sort": "ticker"})
        fr.raise_for_status()
        favs = fr.json()
        if not favs:
            await update.message.reply_text("No favorites. Use /fav TICKER")
            return
        parts: list[str] = []
        for row in favs:
            t = row["ticker"]
            r = await client.get(f"/reports/by-ticker/{t}/latest-decision")
            if r.status_code == 404:
                parts.append(f"{t}: no saved report yet.")
                continue
            r.raise_for_status()
            d = r.json()
            excerpt = _truncate(d.get("content", ""), 800)
            parts.append(f"{t} ({d.get('report_id')}):\n{excerpt}")
    text = "Favorites digest:\n\n" + "\n\n---\n\n".join(parts)
    await update.message.reply_text(_truncate(text))


def _reminder_chat_id() -> int | None:
    if REMINDER_CHAT_RAW:
        return int(REMINDER_CHAT_RAW.strip())
    if ALLOWED_CHATS and len(ALLOWED_CHATS) == 1:
        return next(iter(ALLOWED_CHATS))
    return None


async def _send_digest_to_chat(app: Application, chat_id: int) -> None:
    async with _client() as client:
        fr = await client.get("/catalog/items", params={"favorites_only": True, "sort": "ticker"})
        fr.raise_for_status()
        favs = fr.json()
        if not favs:
            return
        parts: list[str] = []
        for row in favs:
            t = row["ticker"]
            r = await client.get(f"/reports/by-ticker/{t}/latest-decision")
            if r.status_code == 404:
                parts.append(f"{t}: no saved report yet.")
                continue
            r.raise_for_status()
            d = r.json()
            excerpt = _truncate(d.get("content", ""), 800)
            parts.append(f"{t} ({d.get('report_id')}):\n{excerpt}")
    text = "Daily favorites reminder:\n\n" + "\n\n---\n\n".join(parts)
    try:
        await app.bot.send_message(chat_id=chat_id, text=_truncate(text))
    except Exception as e:
        logger.exception("Reminder send failed: %s", e)


async def post_init(app: Application) -> None:
    if not REMINDER_ENABLED:
        return
    cid = _reminder_chat_id()
    if cid is None:
        logger.warning(
            "TELEGRAM_REMINDER_ENABLED but no TELEGRAM_REMINDER_CHAT_ID "
            "and not exactly one ALLOWED_CHAT_IDS; skipping scheduler"
        )
        return
    loop = asyncio.get_running_loop()
    sched: AsyncIOScheduler = AsyncIOScheduler(event_loop=loop)

    async def job() -> None:
        await _send_digest_to_chat(app, cid)

    async def earnings_job() -> None:
        from telegram_bot.earnings_watcher import check_earnings_alerts
        await check_earnings_alerts(app.bot, cid, api_base=API_BASE)

    sched.add_job(
        job,
        CronTrigger(hour=REMINDER_HOUR, minute=REMINDER_MINUTE),
        id="favorites_digest",
        replace_existing=True,
    )
    sched.add_job(
        earnings_job,
        CronTrigger(hour=8, minute=30),
        id="earnings_alerts",
        replace_existing=True,
    )
    sched.start()
    app.bot_data["scheduler"] = sched
    logger.info("Reminder scheduler started for chat_id=%s at %02d:%02d", cid, REMINDER_HOUR, REMINDER_MINUTE)
    logger.info("Earnings alert scheduler started for chat_id=%s at 08:30 daily", cid)


async def post_shutdown(app: Application) -> None:
    sched = app.bot_data.get("scheduler")
    if sched:
        sched.shutdown(wait=False)


def run() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Set TELEGRAM_BOT_TOKEN in the environment.")
        sys.exit(1)
    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=15.0,
        read_timeout=30.0,
        write_timeout=30.0,
    )
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("favorites", cmd_favorites))
    app.add_handler(CommandHandler("fav", cmd_fav))
    app.add_handler(CommandHandler("unfav", cmd_unfav))
    app.add_handler(CommandHandler("addticker", cmd_addticker))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("digest", cmd_digest))
    logger.info("Polling Telegram; backend API base %s", API_BASE)
    app.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=3)


if __name__ == "__main__":
    run()
