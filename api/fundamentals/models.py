"""Pydantic models for Fundamentals Lab API responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EarningsCalendarRow(BaseModel):
    ticker: str
    short_name: Optional[str] = None
    next_earnings_date: Optional[str] = None
    days_until: Optional[int] = None
    earnings_time: Optional[str] = None
    eps_estimate: Optional[float] = None
    eps_estimate_low: Optional[float] = None
    eps_estimate_high: Optional[float] = None
    revenue_estimate: Optional[float] = None
    last_eps_actual: Optional[float] = None
    last_surprise_pct: Optional[float] = None


class EarningsCalendarResponse(BaseModel):
    rows: list[EarningsCalendarRow]
    upcoming_count: int
    cached_at: Optional[str] = None


class RatiosData(BaseModel):
    ticker: str
    short_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    beta: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    dividend_yield: Optional[float] = None


class IncomePeriod(BaseModel):
    period: str
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    operating_income: Optional[float] = None


class BalanceSheetSnapshot(BaseModel):
    period: Optional[str] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    stockholders_equity: Optional[float] = None
    total_debt: Optional[float] = None
    current_liabilities: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None


class CashflowHealth(BaseModel):
    period: Optional[str] = None
    operating_cf: Optional[float] = None
    free_cf: Optional[float] = None
    capex: Optional[float] = None


class EarningsHistoryRow(BaseModel):
    date: str
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    surprise_pct: Optional[float] = None


class AnalystConsensus(BaseModel):
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    period: Optional[str] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    num_analysts: Optional[int] = None


class TickerFundamentalsResponse(BaseModel):
    ticker: str
    ratios: RatiosData
    income: list[IncomePeriod]
    balance: BalanceSheetSnapshot
    cashflow: CashflowHealth
    earnings_history: list[EarningsHistoryRow]
    analyst: Optional[AnalystConsensus] = None
    cached_at: Optional[str] = None


class SecFiling(BaseModel):
    accession_number: str
    ticker: str
    cik: Optional[str] = None
    form_type: str
    filing_date: Optional[str] = None
    filing_url: str
    llm_summary: Optional[str] = None
    summarized_at: Optional[str] = None


class SecFilingsResponse(BaseModel):
    ticker: str
    cik: Optional[str] = None
    filings: list[SecFiling]
