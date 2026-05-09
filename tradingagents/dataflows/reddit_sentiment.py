# -*- coding: utf-8 -*-
"""Reddit-based retail sentiment fetching via public JSON API (no auth required)."""

import logging
import re
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "TradingAgentsBot/0.1"}
_SUBREDDITS = ["wallstreetbets", "stocks", "options"]
_TIMEOUT = 10
_COMMENT_PREVIEW = 200   # max chars per comment shown to LLM
_TOP_POSTS_WITH_COMMENTS = 3   # fetch comments for this many top posts only


def _get_company_name(ticker: str) -> str:
    """Look up the short company name for a ticker via yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ""
    except Exception:
        return ""


def _search_subreddit(subreddit: str, query: str) -> list:
    """Fetch up to 25 posts from a subreddit matching query. Returns raw post dicts."""
    params = {
        "q": query,
        "restrict_sr": 1,
        "sort": "relevance",
        "limit": 25,
        "t": "week",
    }
    try:
        r = requests.get(
            f"https://www.reddit.com/r/{subreddit}/search.json",
            params=params,
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.warning("Reddit API request failed for r/%s: %s", subreddit, e)
        return []

    if r.status_code == 429:
        logger.warning("Reddit API rate limit hit (429) for r/%s", subreddit)
        return []
    if not r.ok:
        logger.warning("Reddit API returned HTTP %s for r/%s", r.status_code, subreddit)
        return []

    r.encoding = "utf-8"
    try:
        return r.json().get("data", {}).get("children", [])
    except ValueError:
        logger.warning("Reddit API returned invalid JSON for r/%s", subreddit)
        return []


def _fetch_top_comments(subreddit: str, post_id: str, limit: int = 20) -> list[str]:
    """
    Fetch top-level comments for a post, sorted by score.
    Returns a list of comment body strings (truncated to _COMMENT_PREVIEW chars).
    """
    try:
        r = requests.get(
            f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json",
            params={"sort": "top", "limit": limit, "depth": 1},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.warning("Reddit comment fetch failed for %s/%s: %s", subreddit, post_id, e)
        return []

    if r.status_code == 429:
        logger.warning("Reddit API rate limit hit (429) fetching comments for %s", post_id)
        return []
    if not r.ok:
        logger.warning("Reddit comment API returned HTTP %s for %s", r.status_code, post_id)
        return []

    r.encoding = "utf-8"
    try:
        data = r.json()
    except ValueError:
        return []

    # Response is [post_listing, comment_listing]
    if len(data) < 2:
        return []

    _BOT_AUTHORS = {"automoderator", "visualmod"}

    comments = []
    for item in data[1].get("data", {}).get("children", []):
        cdata = item.get("data", {})
        author = cdata.get("author", "").lower()
        if author in _BOT_AUTHORS:
            continue
        body = cdata.get("body", "")
        if not body or body == "[deleted]" or body == "[removed]":
            continue
        body = " ".join(body.split())
        comments.append(body[:_COMMENT_PREVIEW])

    return comments


def get_reddit_sentiment(ticker: str, days: int = 3) -> str:
    """
    Fetch recent Reddit posts mentioning a ticker from investing subreddits.

    Searches r/wallstreetbets, r/stocks, and r/options via Reddit's public
    JSON API (no authentication required). Runs separate queries for the ticker
    symbol and company name so posts using either form are captured. Only keeps
    posts whose title contains the ticker or company name. Fetches top comments
    for the three highest-scoring posts to capture actual retail discussion.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA")
        days: Number of days to look back (default 3)

    Returns:
        Formatted string of matching posts with top comments, or a "no posts"
        message when the ticker has no retail Reddit coverage.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

    company_name = _get_company_name(ticker)
    name_keyword = ""
    if company_name:
        first_word = company_name.split()[0]
        if len(first_word) > 3 and first_word.upper() != ticker.upper():
            name_keyword = first_word

    search_terms = [ticker]
    if name_keyword:
        search_terms.append(name_keyword)

    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    for subreddit in _SUBREDDITS:
        for term in search_terms:
            for item in _search_subreddit(subreddit, term):
                post = item.get("data", {})
                post_id = post.get("id", "")

                if post_id in seen_ids:
                    continue

                title_lower = post.get("title", "").lower()
                if ticker.lower() not in title_lower and (
                    not name_keyword or name_keyword.lower() not in title_lower
                ):
                    continue

                created_utc = post.get("created_utc", 0)
                post_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if post_time < cutoff:
                    continue

                seen_ids.add(post_id)
                all_posts.append({
                    "id": post_id,
                    "subreddit": subreddit,
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.0),
                    "flair": post.get("link_flair_text") or "",
                })

    if not all_posts:
        return (
            f"No Reddit posts found mentioning {ticker} in the last {days} days "
            f"across r/wallstreetbets, r/stocks, r/options."
        )

    all_posts.sort(key=lambda p: p["score"], reverse=True)

    for post in all_posts[:_TOP_POSTS_WITH_COMMENTS]:
        post["comments"] = _fetch_top_comments(post["subreddit"], post["id"])

    label = f"{ticker}" + (f" / {name_keyword}" if name_keyword else "")
    lines = [f"Reddit Sentiment for {label} (last {days} days, sorted by score):\n"]

    for post in all_posts:
        ratio_pct = f"{post['upvote_ratio']:.0%}"
        flair_str = f" | Flair: {post['flair']}" if post["flair"] else ""
        lines.append(
            f"[r/{post['subreddit']}] {post['title']}\n"
            f"  Score: {post['score']} | Comments: {post['num_comments']} | "
            f"Upvotes: {ratio_pct}{flair_str}"
        )
        for comment in post.get("comments", []):
            lines.append(f"  > {comment}")
        lines.append("")

    return "\n".join(lines)


def get_reddit_sentiment_structured(ticker: str, days: int = 3) -> list[dict]:
    """
    Same logic as get_reddit_sentiment but returns a list of post dicts instead
    of a formatted string. Each dict has:
        subreddit, title, score, num_comments, upvote_ratio, flair, url, comments
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    company_name = _get_company_name(ticker)
    name_keyword = ""
    if company_name:
        first_word = company_name.split()[0]
        if len(first_word) > 3 and first_word.upper() != ticker.upper():
            name_keyword = first_word

    search_terms = [ticker]
    if name_keyword:
        search_terms.append(name_keyword)

    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    for subreddit in _SUBREDDITS:
        for term in search_terms:
            for item in _search_subreddit(subreddit, term):
                post = item.get("data", {})
                post_id = post.get("id", "")
                if post_id in seen_ids:
                    continue
                title_lower = post.get("title", "").lower()
                if ticker.lower() not in title_lower and (
                    not name_keyword or name_keyword.lower() not in title_lower
                ):
                    continue
                created_utc = post.get("created_utc", 0)
                post_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                if post_time < cutoff:
                    continue
                seen_ids.add(post_id)
                all_posts.append({
                    "id": post_id,
                    "subreddit": subreddit,
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.0),
                    "flair": post.get("link_flair_text") or "",
                    "url": f"https://reddit.com/r/{subreddit}/comments/{post_id}/",
                    "created_utc": created_utc,
                })

    all_posts.sort(key=lambda p: p["score"], reverse=True)

    for post in all_posts[:_TOP_POSTS_WITH_COMMENTS]:
        post["comments"] = _fetch_top_comments(post["subreddit"], post["id"])

    for post in all_posts:
        if "comments" not in post:
            post["comments"] = []
        post.pop("id", None)  # internal only

    return all_posts


# ---------------------------------------------------------------------------
# Trending ticker scanner
# ---------------------------------------------------------------------------

# Common uppercase words/letters that look like tickers but aren't.
# Single letters are never treated as tickers (enforced in regex min-length).
_TICKER_BLOCKLIST = {
    # Two-letter noise
    "AI", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "EV", "FX", "GO",
    "HQ", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON",
    "OR", "PM", "RH", "SO", "TO", "UP", "US", "WE",
    # Three-letter noise / common abbreviations
    "ALL", "AND", "ARE", "ATH", "ATM", "CEO", "CFO", "CPI", "CTO",
    "DID", "DTE", "ETF", "EPS", "FED", "FOR", "GDP", "GET", "GOT",
    "HAS", "HIM", "HIS", "HOW", "IMO", "IPO", "IRS", "ITS", "ITM",
    "LOL", "LOW", "NEW", "NOT", "NOW", "OLD", "ONE", "OTM", "OUR",
    "OUT", "OTC", "PDT", "PMI", "PUT", "ROI", "SAY", "SEC", "SEE",
    "SET", "SHE", "THE", "TOO", "TOP", "TWO", "USA", "USE", "VIX",
    "WAS", "WAY", "WHO", "WHY", "WIN", "YTD",
    # Four-letter noise
    "AFAIK", "ALSO", "BEAR", "BOND", "BULL", "CALL", "CASH", "DEBT",
    "DAYS", "EDIT", "FOMO", "FUND", "GAIN", "GAME", "HIGH", "HODL",
    "HOLD", "IIRC", "IDEA", "LOSS", "LONG", "MOON", "NEWS", "NEXT",
    "PLAN", "PUTS", "RATE", "RISK", "SELL", "THAT", "THEM", "THEN",
    "THEY", "THIS", "TIME", "TLDR", "WEEK", "WITH", "YEAR", "YOLO",
    "YOUR",
    # Five-letter noise
    "AFAIK", "AGAIN", "CHART", "COULD", "EVERY", "FIRST", "GOING",
    "RATES", "STOCK", "THEIR", "THERE", "TODAY", "TRADE", "WHERE",
    "WHICH", "WOULD",
    # Options/trading slang
    "CALLS", "LEAPS", "THETA", "DELTA", "GAMMA", "VANNA", "CHARM",
    "HEDGE", "SHORT", "COVER", "BREAK", "RALLY", "CRASH", "PRINT",
    "FLOOR", "SCALP",
    # Common Reddit slang
    "BASED", "MOONS", "BEARS", "BULLS", "BRRR", "GANG", "RETARD",
    "APES", "BAGS",
}

# Require ≥2 chars for bare uppercase matches to eliminate single-letter noise.
# $TICKER explicit notation still allows 1-char (e.g. $F for Ford).
_TICKER_RE = re.compile(r'\$([A-Z]{1,5})|(?<![A-Z])([A-Z]{2,5})(?![a-z])')


def _extract_tickers_from_text(text: str) -> list[str]:
    """Extract probable ticker symbols from a string."""
    found = []
    for m in _TICKER_RE.finditer(text):
        symbol = m.group(1) or m.group(2)
        if symbol and symbol not in _TICKER_BLOCKLIST and len(symbol) >= 1:
            found.append(symbol)
    return found


def _fetch_hot_posts(subreddit: str, limit: int = 100) -> list[dict]:
    """Fetch hot posts from a subreddit. Returns raw post data dicts."""
    try:
        r = requests.get(
            f"https://www.reddit.com/r/{subreddit}/hot.json",
            params={"limit": limit},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.warning("Reddit hot fetch failed for r/%s: %s", subreddit, e)
        return []

    if r.status_code == 429:
        logger.warning("Reddit rate limit (429) for r/%s hot", subreddit)
        return []
    if not r.ok:
        return []

    r.encoding = "utf-8"
    try:
        children = r.json().get("data", {}).get("children", [])
        return [c.get("data", {}) for c in children]
    except ValueError:
        return []


def get_reddit_trending_tickers(
    subreddits: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Scan hot posts of investing subreddits and rank tickers by mention count.

    Searches r/wallstreetbets, r/stocks, r/options (or custom subreddits).
    Extracts ticker symbols via $TICKER notation and uppercase word patterns,
    filtered against a blocklist of common non-ticker words.

    Args:
        subreddits: List of subreddit names to scan (default: wallstreetbets, stocks, options)
        limit: Number of hot posts to fetch per subreddit (max 100)

    Returns:
        List of dicts sorted by mention count desc:
            {symbol, mentions, posts: [{title, score, url}]}
    """
    subs = subreddits or _SUBREDDITS
    limit = min(max(limit, 1), 100)

    # symbol -> {mentions: int, posts: list[{title, score, url}]}
    ticker_map: dict[str, dict] = {}

    for subreddit in subs:
        posts = _fetch_hot_posts(subreddit, limit=limit)
        for post in posts:
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            score = post.get("score", 0)
            post_id = post.get("id", "")
            url = f"https://reddit.com/r/{subreddit}/comments/{post_id}/"

            combined = f"{title} {selftext}"
            tickers = _extract_tickers_from_text(combined)

            for symbol in set(tickers):
                if symbol not in ticker_map:
                    ticker_map[symbol] = {"mentions": 0, "posts": []}
                # Count per-post occurrences (not per-character) to avoid inflation
                ticker_map[symbol]["mentions"] += combined.count(f"${symbol}") + combined.count(f" {symbol} ") + combined.count(f" {symbol}\n")
                # Keep up to 3 representative posts per ticker
                if len(ticker_map[symbol]["posts"]) < 3:
                    ticker_map[symbol]["posts"].append({
                        "title": title,
                        "score": score,
                        "url": url,
                    })

    results = [
        {"symbol": sym, "mentions": data["mentions"], "posts": data["posts"]}
        for sym, data in ticker_map.items()
    ]
    results.sort(key=lambda x: x["mentions"], reverse=True)
    return results
