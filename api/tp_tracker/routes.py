from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from . import bridge, repository
from .schemas import TrackerCreate, TrackerOut, TrackerSignalRequest

router = APIRouter(prefix="/tp-tracker", tags=["tp-tracker"])


def _row_to_out(row: dict, current_price: Optional[float] = None) -> TrackerOut:
    return TrackerOut(
        id           = row["id"],
        ticker       = row["ticker"],
        side         = row["side"],
        entry        = row["entry"],
        sl           = row["sl"],
        trail_sl     = row["trail_sl"],
        tp1          = row["tp1"],
        tp2          = row["tp2"],
        tp3          = row["tp3"],
        tp1_hit      = bool(row["tp1_hit"]),
        tp2_hit      = bool(row["tp2_hit"]),
        tp3_hit      = bool(row["tp3_hit"]),
        trail_active = bool(row["trail_active"]),
        closed       = bool(row["closed"]),
        exit_reason  = row.get("exit_reason"),
        rsi_at_entry = row.get("rsi_at_entry"),
        htf_bias     = row.get("htf_bias"),
        vol_regime   = row.get("vol_regime"),
        bull_score   = row.get("bull_score"),
        as_of_date   = row.get("as_of_date"),
        created_at   = row["created_at"],
        updated_at   = row["updated_at"],
        current_price = current_price,
        trail_status  = bridge.describe_trail_status(row),
    )


@router.post("", response_model=TrackerOut, status_code=201)
def create_tracker(body: TrackerCreate):
    """Create a tracker from manually supplied entry/SL/TP values."""
    row = body.dict()
    row["trail_sl"] = row["sl"]
    row.update({"tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0, "trail_active": 0, "closed": 0})
    tracker_id = repository.create_tracker(row)
    return _row_to_out(repository.get_tracker_by_id(tracker_id))


@router.get("", response_model=List[TrackerOut])
def list_trackers(include_closed: bool = False):
    rows = repository.list_all_trackers(include_closed=include_closed)
    return [_row_to_out(r) for r in rows]


@router.post("/from-signal", response_model=TrackerOut, status_code=201)
def create_from_signal(body: TrackerSignalRequest):
    """Call compute_sniper_signal() and create a tracker from the result.

    Returns 422 if the signal is HOLD (no entry direction).
    """
    from tradingagents.quant_ml.signals.sniper_signal import compute_sniper_signal
    signal = compute_sniper_signal(body.ticker.upper().strip(), as_of_date=body.date)
    if signal.get("error"):
        raise HTTPException(status_code=422, detail=signal["error"])
    try:
        db_row = bridge.signal_to_tracker_row(signal)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    tracker_id = repository.create_tracker(db_row)
    return _row_to_out(repository.get_tracker_by_id(tracker_id))


@router.get("/{ticker}", response_model=TrackerOut)
def get_tracker(ticker: str):
    row = repository.get_tracker(ticker.upper())
    if not row:
        raise HTTPException(status_code=404, detail=f"No active tracker for {ticker.upper()}")
    return _row_to_out(row)


@router.post("/{ticker}/refresh", response_model=TrackerOut)
def refresh_tracker(ticker: str):
    """Fetch current price, run ProgressiveTrail.tick(), persist updated state."""
    row = repository.get_tracker(ticker.upper())
    if not row:
        raise HTTPException(status_code=404, detail=f"No active tracker for {ticker.upper()}")
    try:
        price = bridge.fetch_current_price(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Price fetch failed: {exc}")
    updates = bridge.tick_tracker(row, price)
    if updates:
        repository.update_tracker(row["id"], updates)
    refreshed = repository.get_tracker_by_id(row["id"])
    return _row_to_out(refreshed, current_price=price)


@router.delete("/{ticker}", status_code=204)
def close_tracker(ticker: str):
    """Manually close/dismiss a tracker."""
    row = repository.get_tracker(ticker.upper())
    if not row:
        raise HTTPException(status_code=404, detail="Tracker not found")
    repository.close_tracker(row["id"], exit_reason="manual")
