"""bridge.py — convert sniper signal → DB row, tick ProgressiveTrail, describe status."""

from __future__ import annotations

from typing import Any, Dict

from tradingagents.quant_ml.risk.trailing import ProgressiveTrail


def signal_to_tracker_row(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Convert compute_sniper_signal() output → position_targets row dict.

    Raises ValueError if action is HOLD (no trade direction to track).
    """
    action = (signal.get("action") or "").upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(
            f"Signal action={action!r} has no entry direction — only BUY/SELL can be tracked."
        )
    side = "long" if action == "BUY" else "short"
    entry = signal["latest_close"]
    sl = signal["stop_loss"]
    return {
        "ticker":       signal["ticker"],
        "side":         side,
        "entry":        entry,
        "sl":           sl,
        "trail_sl":     sl,  # initialised to original SL; ratcheted on each tick
        "tp1":          signal["tp1"],
        "tp2":          signal["tp2"],
        "tp3":          signal["tp3"],
        "tp1_hit":      0,
        "tp2_hit":      0,
        "tp3_hit":      0,
        "trail_active": 0,
        "closed":       0,
        "rsi_at_entry": signal.get("last_rsi"),
        "htf_bias":     signal.get("htf_bias"),
        "vol_regime":   signal.get("vol_regime"),
        "bull_score":   signal.get("bull_score"),
        "as_of_date":   signal.get("as_of_date"),
    }


def fetch_current_price(ticker: str) -> float:
    """Fetch latest price via yfinance fast_info, with download fallback."""
    import yfinance as yf
    try:
        price = yf.Ticker(ticker).fast_info["lastPrice"]
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass
    df = yf.download(ticker, period="2d", progress=False, multi_level_index=False)
    return float(df["Close"].iloc[-1])


def tick_tracker(row: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """Restore ProgressiveTrail from stored state, call on_bar(), return changed fields.

    Uses current_price as both high and low (daily refresh; conservative trigger).
    Returns an empty dict if nothing changed.
    """
    if row.get("closed"):
        return {}

    trail = ProgressiveTrail(
        side    = row["side"],
        entry   = row["entry"],
        sl      = row["trail_sl"],   # use the current ratcheted SL, not original
        tp1     = row["tp1"],
        tp2     = row["tp2"],
        tp3     = row["tp3"],
        tp1_hit = bool(row["tp1_hit"]),
        tp2_hit = bool(row["tp2_hit"]),
        tp3_hit = bool(row["tp3_hit"]),
    )

    trail.on_bar(high=current_price, low=current_price)

    # Detect changes
    new_sl     = trail.sl
    new_tp1hit = trail.tp1_hit
    new_tp2hit = trail.tp2_hit
    new_tp3hit = trail.tp3_hit
    sl_changed = abs(new_sl - row["trail_sl"]) > 1e-9
    hit_changed = (
        int(new_tp1hit) != int(row["tp1_hit"])
        or int(new_tp2hit) != int(row["tp2_hit"])
        or int(new_tp3hit) != int(row["tp3_hit"])
    )
    if not sl_changed and not hit_changed and not trail.closed:
        return {}

    updates: Dict[str, Any] = {
        "trail_sl":     new_sl,
        "tp1_hit":      int(new_tp1hit),
        "tp2_hit":      int(new_tp2hit),
        "tp3_hit":      int(new_tp3hit),
        "trail_active": int(new_tp1hit),
    }
    if trail.closed:
        updates["closed"]      = 1
        updates["exit_reason"] = trail.exit_reason
    return updates


def describe_trail_status(row: Dict[str, Any]) -> str:
    """Return a human-readable status string for the sidebar, e.g. 'TP1 ✓ — Trail to Entry'."""
    if row.get("closed"):
        reason = row.get("exit_reason") or "Closed"
        if reason == "SL_HIT":
            return "SL Hit — Closed"
        if reason == "TP3":
            return "TP1 ✓  TP2 ✓  TP3 ✓ — Closed"
        return f"Closed ({reason})"

    tp1 = bool(row.get("tp1_hit"))
    tp2 = bool(row.get("tp2_hit"))
    tp3 = bool(row.get("tp3_hit"))

    if tp2:
        return "TP1 ✓  TP2 ✓ — Trail to TP1"
    if tp1:
        return "TP1 ✓ — Trail to Entry"
    return "Open"
