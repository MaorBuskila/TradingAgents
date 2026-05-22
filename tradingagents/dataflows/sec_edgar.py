"""SEC EDGAR API client: CIK lookup, filing metadata, filing text extraction."""

from __future__ import annotations

import functools
import logging
import re
from html.parser import HTMLParser
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "TradingAgents/1.0 maorb@appdome.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 15

TRACKED_FORMS = {"8-K", "10-Q", "10-K"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


@functools.lru_cache(maxsize=1)
def _load_ticker_cik_map() -> dict[str, str]:
    """Load full ticker→padded-CIK map from SEC EDGAR (cached after first call)."""
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in r.json().values()
    }


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Return 10-digit zero-padded CIK for a US ticker, or None if not found."""
    try:
        mapping = _load_ticker_cik_map()
        return mapping.get(ticker.upper().strip())
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
        return None


def get_recent_filings(
    cik: str,
    form_types: Optional[list[str]] = None,
    limit: int = 10,
) -> list[dict]:
    """Return recent SEC filings for a CIK, filtered by form_types (default 8-K/10-Q/10-K)."""
    allowed = set(form_types or TRACKED_FORMS)
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("EDGAR submissions fetch failed for CIK %s: %s", cik, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    acc_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    forms = recent.get("form", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    cik_int = int(cik)
    results: list[dict] = []

    for i, form in enumerate(forms):
        if form not in allowed:
            continue
        acc = acc_numbers[i]
        acc_clean = acc.replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_doc}"
            if primary_doc
            else f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"
        )
        results.append({
            "accession_number": acc,
            "cik": cik,
            "form_type": form,
            "filing_date": filing_dates[i] if i < len(filing_dates) else None,
            "filing_url": filing_url,
            "description": descriptions[i] if i < len(descriptions) else "",
        })
        if len(results) >= limit:
            break

    return results


def get_filing_text(filing_url: str, max_chars: int = 50_000) -> str:
    """Fetch a SEC filing document and return plain text (HTML stripped)."""
    try:
        r = requests.get(filing_url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        raw = r.text

        if "html" in content_type or raw.lstrip().startswith("<"):
            parser = _TextExtractor()
            parser.feed(raw)
            text = parser.get_text()
            # Collapse excessive whitespace
            text = re.sub(r" {3,}", "  ", text)
            text = re.sub(r"\n{4,}", "\n\n\n", text)
        else:
            text = raw

        return text[:max_chars]
    except Exception as exc:
        logger.warning("Filing text fetch failed for %s: %s", filing_url, exc)
        return ""
