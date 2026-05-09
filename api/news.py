"""News Lab API router — news, social sentiment, earnings calendar, trending."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.yfinance_news import get_news_yfinance_structured
from tradingagents.dataflows.reddit_sentiment import get_reddit_sentiment_structured

from .models import (
    EarningsCalendarOut,
    NewsArticleOut,
    RedditTrendingPost,
    RedditTrendingTickerOut,
    TrendingStockOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _days_ago(n: int) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# GET /api/news/trending  — market movers via yfinance screener
# ---------------------------------------------------------------------------

@router.get("/trending", response_model=List[TrendingStockOut])
async def get_trending(
    screener: str = Query(default="most_actives"),
    count: int = Query(default=20, ge=1, le=50),
):
    """Return top stocks from a yfinance predefined screener."""
    valid = {"most_actives", "day_gainers", "day_losers", "most_shorted_stocks", "growth_technology_stocks"}
    if screener not in valid:
        screener = "most_actives"
    try:
        results = await asyncio.to_thread(
            route_to_vendor, "get_trending_stocks", screener=screener, count=count
        )
        return [TrendingStockOut(**r) for r in results]
    except Exception as e:
        logger.exception("Error fetching trending stocks")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/reddit-trending  — most-mentioned tickers on Reddit
# ---------------------------------------------------------------------------

@router.get("/reddit-trending", response_model=List[RedditTrendingTickerOut])
async def get_reddit_trending(
    limit: int = Query(default=50, ge=1, le=100),
    top_n: int = Query(default=30, ge=1, le=100),
):
    """Return the most-mentioned tickers on r/wallstreetbets, r/stocks, r/options."""
    try:
        results = await asyncio.to_thread(
            route_to_vendor, "get_reddit_trending_tickers", limit=limit
        )
        trimmed = results[:top_n]
        out = []
        for r in trimmed:
            posts = [RedditTrendingPost(**p) for p in r.get("posts", [])]
            out.append(RedditTrendingTickerOut(
                symbol=r["symbol"],
                mentions=r["mentions"],
                posts=posts,
            ))
        return out
    except Exception as e:
        logger.exception("Error fetching Reddit trending tickers")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/earnings  — earnings calendar for a list of tickers
# ---------------------------------------------------------------------------

@router.get("/earnings", response_model=List[EarningsCalendarOut])
async def get_earnings(
    tickers: str = Query(..., description="Comma-separated ticker symbols, e.g. AAPL,NVDA,TSLA"),
):
    """Return next earnings dates and EPS estimates for the requested tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="At least one ticker is required")
    try:
        results = await asyncio.to_thread(
            route_to_vendor, "get_earnings_calendar_multi", tickers=ticker_list
        )
        return [EarningsCalendarOut(**r) for r in results]
    except Exception as e:
        logger.exception("Error fetching earnings calendar")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/ticker-news  — Yahoo Finance news for a ticker
# ---------------------------------------------------------------------------

@router.get("/ticker-news")
async def get_ticker_news(
    ticker: str = Query(...),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """Return formatted news text for a single ticker."""
    if not start_date:
        start_date = _days_ago(7)
    if not end_date:
        end_date = _today()
    try:
        text = await asyncio.to_thread(
            route_to_vendor, "get_news", ticker.upper(), start_date, end_date
        )
        return {"ticker": ticker.upper(), "text": text}
    except Exception as e:
        logger.exception("Error fetching ticker news for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/ticker-articles  — structured article list for ticker news
# ---------------------------------------------------------------------------

@router.get("/ticker-articles", response_model=List[NewsArticleOut])
async def get_ticker_articles(
    ticker: str = Query(...),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """Return structured article list for a single ticker (for card UI rendering)."""
    if not start_date:
        start_date = _days_ago(7)
    if not end_date:
        end_date = _today()
    try:
        articles = await asyncio.to_thread(
            get_news_yfinance_structured, ticker.upper(), start_date, end_date
        )
        return [NewsArticleOut(**a) for a in articles]
    except Exception as e:
        logger.exception("Error fetching structured ticker articles for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/reddit  — Reddit sentiment for a specific ticker
# ---------------------------------------------------------------------------

@router.get("/reddit")
async def get_reddit_sentiment(
    ticker: str = Query(...),
    days: int = Query(default=3, ge=1, le=30),
):
    """Return Reddit sentiment text for a specific ticker."""
    try:
        text = await asyncio.to_thread(
            route_to_vendor, "get_reddit_sentiment", ticker.upper(), days
        )
        return {"ticker": ticker.upper(), "text": text}
    except Exception as e:
        logger.exception("Error fetching Reddit sentiment for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/reddit-posts  — structured Reddit posts for a ticker
# ---------------------------------------------------------------------------

@router.get("/reddit-posts")
async def get_reddit_posts(
    ticker: str = Query(...),
    days: int = Query(default=3, ge=1, le=30),
):
    """Return structured Reddit posts for a ticker (for card UI rendering)."""
    try:
        posts = await asyncio.to_thread(
            get_reddit_sentiment_structured, ticker.upper(), days
        )
        return posts
    except Exception as e:
        logger.exception("Error fetching structured Reddit posts for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/fear-greed  — CNN Fear & Greed Index
# ---------------------------------------------------------------------------

_FG_LINE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) \| Score: (\d+)/100 \| (.+)")


@router.get("/fear-greed")
async def get_fear_greed(
    days: int = Query(default=14, ge=1, le=90),
):
    """Return CNN Fear & Greed Index as text + parsed current score/label + history series."""
    try:
        text = await asyncio.to_thread(route_to_vendor, "get_fear_greed", days)
    except Exception as e:
        logger.exception("Error fetching Fear & Greed index")
        raise HTTPException(status_code=500, detail=str(e))

    # Parse the formatted text into a structured series for the frontend chart
    history = []
    current_score: Optional[int] = None
    current_label: Optional[str] = None

    for line in text.splitlines():
        m = _FG_LINE_RE.match(line.strip())
        if m:
            date, score_str, label = m.group(1), m.group(2), m.group(3)
            score = int(score_str)
            history.append({"date": date, "score": score, "label": label})
            if current_score is None:
                current_score = score
                current_label = label

    return {
        "text": text,
        "current_score": current_score,
        "current_label": current_label,
        "history": history,
    }


# ---------------------------------------------------------------------------
# GET /api/news/influencer-mentions  — news articles mentioning person + ticker
# ---------------------------------------------------------------------------

@router.get("/influencer-mentions", response_model=List[NewsArticleOut])
async def get_influencer_mentions(
    person: str = Query(..., description="Influencer name, e.g. Trump, Musk, Buffett"),
    ticker: str = Query(...),
    days: int = Query(default=7, ge=1, le=90),
):
    """Return Yahoo Finance news articles mentioning both a person and a ticker."""
    try:
        articles = await asyncio.to_thread(
            route_to_vendor,
            "get_influencer_mentions",
            person=person,
            ticker=ticker.upper(),
            days=days,
        )
        return [NewsArticleOut(**a) for a in articles]
    except Exception as e:
        logger.exception("Error fetching influencer mentions for %s + %s", person, ticker)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/news/global  — macro/global market news
# ---------------------------------------------------------------------------

@router.get("/global")
async def get_global_news(
    days: int = Query(default=7, ge=1, le=30),
):
    """Return global macro market news."""
    try:
        text = await asyncio.to_thread(
            route_to_vendor, "get_global_news", _today(), look_back_days=days
        )
        return {"text": text}
    except Exception as e:
        logger.exception("Error fetching global news")
        raise HTTPException(status_code=500, detail=str(e))
