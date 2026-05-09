"""Fetch YouTube captions and summarize via the same LLM stack as MACD/RSI labs (create_llm_client)."""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

from tradingagents.dataflows.llm_invoke import invoke_chat_model_human_message

load_dotenv()

MAX_TRANSCRIPT_CHARS = 55_000


def extract_youtube_video_id(url: str) -> Optional[str]:
    if not url or not url.strip():
        return None
    u = url.strip()
    patterns = [
        r"[?&]v=([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, u)
        if m:
            return m.group(1)
    m = re.search(r"^([a-zA-Z0-9_-]{11})$", u)
    return m.group(1) if m else None


def fetch_video_title(url: str) -> Optional[str]:
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=8,
        )
        if r.ok:
            return r.json().get("title")
    except Exception:
        pass
    return None


# youtube-transcript-api 1.x: use instance .fetch(); legacy 0.x had .get_transcript classmethod.
_LANG_TRY = (
    "en",
    "en-US",
    "en-GB",
    "en-IN",
    "he",
    "iw",  # legacy Hebrew code on some videos
    "ar",
    "es",
    "fr",
    "de",
    "ru",
    "pt",
    "ja",
    "ko",
    "zh",
    "zh-Hans",
    "zh-Hant",
    "hi",
)


def fetch_transcript_text(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise RuntimeError("Install dependency: youtube-transcript-api") from e

    api = YouTubeTranscriptApi()
    fetched = None
    last_err: Optional[Exception] = None

    try:
        fetched = api.fetch(video_id, languages=_LANG_TRY)
    except Exception as e:
        last_err = e
        # Fallback: first available track (e.g. only Hebrew auto-generated)
        try:
            for tr in api.list(video_id):
                fetched = tr.fetch()
                break
        except Exception as e2:
            raise ValueError(
                f"Could not load transcript: {last_err!s} (list failed: {e2!s})"
            ) from e2

    if fetched is None:
        raise ValueError(
            f"Could not load transcript: {last_err!s}"
        ) from last_err

    text = " ".join(s.text for s in fetched.snippets)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_summary_prompt(transcript: str, title: Optional[str], video_id: str) -> str:
    clipped = transcript[:MAX_TRANSCRIPT_CHARS]
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        clipped += "\n\n[Transcript truncated for length.]"
    return f"""Summarize this YouTube video from its transcript.

Video ID: {video_id}
Title: {title or "(unknown)"}

Use markdown with ## headings. Structure:

## TL;DR
2–4 sentences on what the video is about and the main claim or takeaway.

## Key points
5–12 bullets with the most important facts, arguments, and conclusions from the transcript.

## Topics & themes
Short bullets or a comma-separated list of main themes (e.g. macro, sector, company story).

## Actionable market angle (only if relevant)
If the video discusses **stocks, ETFs, sectors, trading, or investing**, add a section that helps the reader act *in the sense of "what the speaker is steering attention toward"*, not personal financial advice.

For each **named ticker, ETF, or clear theme** (e.g. "green energy", "semiconductors"), include a compact row or bullet with:
- **Instrument / theme** — ticker or label
- **Speaker's implied stance** — one of: Bullish / Bearish / Neutral / Mixed / Unclear (as expressed in the video), or **Watch / Avoid / Focus here** if they emphasize where to look or what to avoid
- **Action framing** — one short phrase: e.g. "review position", "consider entry only on pullback", "ETF for broad exposure", "stock-specific risk", "wait for catalyst", "education only — no trade"
- If the speaker gives **no** investable angle, say: "No specific buy/sell; focus on [X]."

If the video is **not** about markets (e.g. pure tech, health, politics with no tickers), write: **No market action section — not an investing video.** and skip invented tickers.

## Disclaimer
End with one line in italics: *This is an AI summary of video content and not financial advice.*

If the transcript is incomplete or noisy, say so briefly under a ## Notes heading.

Transcript:
{clipped}
"""


def summarize_transcript(
    transcript: str,
    title: Optional[str],
    video_id: str,
    llm_provider: str,
    llm_model: Optional[str],
) -> Tuple[str, str, str]:
    """Returns (summary, model_name, provider). LangChain: ``HumanMessage`` + chat model ``invoke``."""
    prompt = _build_summary_prompt(transcript, title, video_id)
    provider_lc = llm_provider.strip().lower()
    summary_raw, usage = invoke_chat_model_human_message(
        prompt,
        provider=provider_lc,
        model=llm_model,
        strip_json_fences=False,
    )
    summary = summary_raw.strip()
    model_used = str(usage.get("model") or llm_model or "")
    return summary, model_used, provider_lc


def translate_summary_to_hebrew(text_en: str) -> str:
    """Translate English markdown summary to Hebrew using Google Translate (deep-translator), not an LLM."""
    if not text_en.strip():
        return ""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise RuntimeError("Install dependency: deep-translator") from e

    # Google Translate expects legacy code "iw" for Hebrew, not ISO "he" (deep-translator rejects "he").
    translator = GoogleTranslator(source="en", target="iw")
    max_chunk = 4500
    out: list[str] = []
    start = 0
    n = len(text_en)
    while start < n:
        end = min(start + max_chunk, n)
        if end < n:
            br = text_en.rfind("\n", start, end)
            if br > start + 200:
                end = br + 1
        chunk = text_en[start:end].strip()
        if chunk:
            out.append(translator.translate(chunk))
        start = end
    return "\n\n".join(out) if out else translator.translate(text_en.strip())


def retranslate_youtube_summary_hebrew(row_id: int) -> dict:
    """Re-run Hebrew machine translation from stored English only (no LLM, no transcript fetch)."""
    from .database import get_youtube_summary_by_id, update_youtube_summary_he

    row = get_youtube_summary_by_id(row_id)
    if not row:
        raise ValueError("Summary not found")
    en = (row.get("summary_en") or "").strip()
    if not en:
        raise ValueError("No English summary to translate")
    summary_he = translate_summary_to_hebrew(en)
    if not update_youtube_summary_he(row_id, summary_he):
        raise RuntimeError("Failed to update Hebrew summary")
    updated = get_youtube_summary_by_id(row_id)
    assert updated is not None
    return dict(updated)


def summarize_youtube_url(
    url: str,
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> dict:
    from .database import insert_youtube_summary

    vid = extract_youtube_video_id(url)
    if not vid:
        raise ValueError("Not a valid YouTube URL or video ID.")

    page_url = url.strip() if "http" in url else f"https://www.youtube.com/watch?v={vid}"
    title = fetch_video_title(page_url)
    transcript = fetch_transcript_text(vid)
    if not transcript:
        raise ValueError("Transcript is empty.")

    summary_en, model_used, provider = summarize_transcript(
        transcript, title, vid, llm_provider=llm_provider, llm_model=llm_model
    )

    skip_tr = os.getenv("YOUTUBE_SKIP_TRANSLATE", "").lower() in ("1", "true", "yes")
    if skip_tr:
        summary_he = "[Translation skipped (YOUTUBE_SKIP_TRANSLATE)]"
    else:
        try:
            summary_he = translate_summary_to_hebrew(summary_en)
        except Exception as e:
            summary_he = f"[Hebrew machine translation failed: {e}]"

    row_id = insert_youtube_summary(
        video_id=vid,
        url=page_url,
        title=title,
        transcript_chars=len(transcript),
        summary_en=summary_en,
        summary_he=summary_he,
        provider=provider,
        model=model_used,
    )

    return {
        "id": row_id,
        "video_id": vid,
        "url": page_url,
        "title": title,
        "transcript_chars": len(transcript),
        "summary_en": summary_en,
        "summary_he": summary_he,
        "model": model_used,
        "provider": provider,
    }
