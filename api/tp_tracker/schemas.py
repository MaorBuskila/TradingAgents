from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class TrackerCreate(BaseModel):
    ticker: str
    side: str  # 'long' | 'short'
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rsi_at_entry: Optional[float] = None
    htf_bias: Optional[float] = None
    vol_regime: Optional[str] = None
    bull_score: Optional[float] = None
    as_of_date: Optional[str] = None


class TrackerOut(BaseModel):
    id: int
    ticker: str
    side: str
    entry: float
    sl: float
    trail_sl: float
    tp1: float
    tp2: float
    tp3: float
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    trail_active: bool
    closed: bool
    exit_reason: Optional[str] = None
    rsi_at_entry: Optional[float] = None
    htf_bias: Optional[float] = None
    vol_regime: Optional[str] = None
    bull_score: Optional[float] = None
    as_of_date: Optional[str] = None
    created_at: str
    updated_at: str
    # enriched at read time by routes layer
    current_price: Optional[float] = None
    trail_status: Optional[str] = None


class TrackerSignalRequest(BaseModel):
    ticker: str
    date: Optional[str] = None
