# TP Tracker — Feature Documentation

## Overview

The TP Tracker is a WillyAlgoTrader-style take-profit monitoring system built into the TradingAgents framework. It persists trade targets (entry, stop-loss, TP1/TP2/TP3) derived from the Sniper signal, tracks which price levels have been hit as the market moves, and displays a live sidebar card showing checkmarks, trailing stop status, RSI, and trend bias — mirroring the visual language of professional algo-chart overlays.

---

## Concepts

### Take Profit Ladder (R-Multiple System)

When a trade is entered, three profit targets are calculated using a fixed Risk:Reward ratio ladder:

| Level | R:R | Formula (long) |
|-------|-----|----------------|
| TP1   | 1:1 | `entry + 1 × risk` |
| TP2   | 2:1 | `entry + 2 × risk` |
| TP3   | 3:1 | `entry + 3 × risk` |

Where `risk = |entry − stop_loss|`.

**Example (IREN):**
- Entry: $48.82 · SL: $39.93 → risk = $8.89
- TP1: $57.71 · TP2: $66.59 · TP3: $75.48

### Stop Loss — Structure + ATR Hybrid

The initial stop loss is not a fixed percentage. It is computed by `structure_sl.py` using two competing methods, picking the tighter result:

1. **Swing-based**: `swing_low − atr_pad × ATR` (last 10 bars)
2. **ATR-based**: `entry − atr_mult × ATR` (default mult = 1.5)

A minimum floor of `entry − 0.5 × ATR` is enforced so the stop can never be unreasonably tight.

For shorts, the logic mirrors above the entry price.

ATR is computed as Wilder's 14-period EMA of True Range.

### Progressive Trailing Stop

Once a TP level is hit, the stop-loss ratchets up automatically — this is the "Trail" mechanic:

| Event | Stop Moves To | Status String |
|-------|--------------|---------------|
| Trade open | Original SL | `Open` |
| TP1 hit | Entry price (break-even) | `TP1 ✓ — Trail to Entry` |
| TP2 hit | TP1 price | `TP1 ✓  TP2 ✓ — Trail to TP1` |
| TP3 hit | Position closes | `TP1 ✓  TP2 ✓  TP3 ✓ — Closed` |
| SL hit | — | `SL Hit — Closed` |

This guarantees that once TP1 is reached, the trade can never result in a loss (the stop is at break-even). Once TP2 is reached, a minimum 1R profit is locked in.

The trail logic is evaluated per-bar: `ProgressiveTrail.on_bar(high, low)`. SL is checked against the **start-of-bar** value before TP levels are checked — preventing the same bar from simultaneously triggering a new TP and then stopping out at the freshly-ratcheted level.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│  TPTracker.tsx — list panel + chart + sidebar card      │
│  SniperLab.tsx — "Track This Signal" button             │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│                FastAPI  (api/main.py)                   │
│  /api/tp-tracker/*          ← tp_tracker router         │
│  /api/sniper-signal/{ticker}                            │
└────────┬──────────────────────────┬─────────────────────┘
         │                          │
┌────────▼──────────┐   ┌──────────▼──────────────────────┐
│  api/tp_tracker/  │   │  tradingagents/quant_ml/         │
│  routes.py        │   │  signals/sniper_signal.py        │
│  bridge.py        │   │  risk/structure_sl.py            │
│  repository.py    │   │  risk/tp_ladder.py               │
│  schemas.py       │   │  risk/trailing.py                │
└────────┬──────────┘   └─────────────────────────────────┘
         │
┌────────▼──────────┐
│  SQLite            │
│  data/portfolio.db │
│  position_targets  │
└───────────────────┘
```

---

## Backend

### `tradingagents/quant_ml/risk/tp_ladder.py`

```python
compute_tps(side, entry, sl, ratios=(1.0, 2.0, 3.0)) -> tuple[float, float, float]
```

Takes the trade side (`"long"` or `"short"`), entry price, stop-loss, and R-multiple ratios. Returns `(tp1, tp2, tp3)`.

---

### `tradingagents/quant_ml/risk/structure_sl.py`

```python
compute_stop(side, entry, df, entry_idx, lookback=10, atr_mult=1.5, atr_pad=0.2, atr_floor=0.5, atr_len=14) -> float
```

Computes the initial stop-loss using the swing-structure + ATR hybrid described above.

---

### `tradingagents/quant_ml/risk/trailing.py` — `ProgressiveTrail`

```python
@dataclass
class ProgressiveTrail:
    side: str        # "long" | "short"
    entry: float
    sl: float        # current trailing stop (mutated on each TP hit)
    tp1: float
    tp2: float
    tp3: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    closed: bool = False
    exit_reason: str | None = None  # "TP3" | "SL_HIT"

    def on_bar(self, high: float, low: float) -> str | None:
        ...  # returns "TP1" | "TP2" | "TP3" | "SL_HIT" | None
```

The state machine is restored from the DB on every refresh call by constructing `ProgressiveTrail` with the stored `tp1_hit`, `tp2_hit`, `tp3_hit`, and `trail_sl` fields.

---

### `tradingagents/quant_ml/signals/sniper_signal.py` — `compute_sniper_signal()`

Returns a signal dict with:

| Field | Description |
|-------|-------------|
| `action` | `"BUY"` / `"SELL"` / `"HOLD"` |
| `latest_close` | Price at signal time (used as entry) |
| `stop_loss` | ATR+structure SL |
| `tp1`, `tp2`, `tp3` | R-multiple targets |
| `last_rsi` | RSI-14 value at signal time |
| `htf_bias` | Higher-timeframe EMA bias value |
| `vol_regime` | `"HIGH"` / `"NORMAL"` / `"LOW"` |
| `bull_score` | Sniper bull score (0–10) |

---

### `api/tp_tracker/bridge.py`

Three pure functions that sit between the signal layer and the DB:

**`signal_to_tracker_row(signal)`**
Converts `compute_sniper_signal()` output to a `position_targets` insert dict. Raises `ValueError` if action is `HOLD`.

**`tick_tracker(row, current_price)`**
Restores `ProgressiveTrail` from the stored DB row, calls `on_bar(high=price, low=price)`, and returns a dict of changed fields only. Empty dict = no state change. Using the same price for both high and low is conservative — it only triggers a TP event if the price has definitively crossed the level.

**`describe_trail_status(row)`**
Produces the human-readable status string shown in the sidebar:
- `"Open"` → no TPs hit yet
- `"TP1 ✓ — Trail to Entry"` → TP1 hit, trail at entry
- `"TP1 ✓  TP2 ✓ — Trail to TP1"` → TP2 hit, trail at TP1
- `"TP1 ✓  TP2 ✓  TP3 ✓ — Closed"` → full run
- `"SL Hit — Closed"` → stopped out

---

### `api/tp_tracker/repository.py`

Raw SQLite CRUD over the `position_targets` table:

| Function | Description |
|----------|-------------|
| `create_tracker(row)` | INSERT, returns new id |
| `get_tracker_by_id(id)` | SELECT by primary key |
| `get_tracker(ticker)` | SELECT latest non-closed row for ticker |
| `list_all_trackers(include_closed)` | SELECT all, optionally including closed |
| `update_tracker(id, fields)` | Dynamic UPDATE, always stamps `updated_at` |
| `close_tracker(id, exit_reason)` | SET closed=1, exit_reason=? |

---

### API Endpoints

All routes are mounted at `/api/tp-tracker`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/tp-tracker` | Create tracker from manual entry/SL/TP values |
| `POST` | `/api/tp-tracker/from-signal` | Generate signal for ticker, create tracker automatically |
| `GET`  | `/api/tp-tracker` | List all active trackers (`?include_closed=true` for all) |
| `GET`  | `/api/tp-tracker/{ticker}` | Get latest active tracker for a ticker |
| `POST` | `/api/tp-tracker/{ticker}/refresh` | Fetch current price, run trail tick, persist updates |
| `DELETE` | `/api/tp-tracker/{ticker}` | Manually close a tracker (`exit_reason = "manual"`) |
| `GET`  | `/api/sniper-signal/{ticker}` | Get latest Sniper signal (entry/SL/TP/RSI/vol) |

---

### Database Schema — `position_targets`

```sql
CREATE TABLE position_targets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    side         TEXT    NOT NULL,           -- 'long' | 'short'
    entry        REAL    NOT NULL,           -- price at signal time
    sl           REAL    NOT NULL,           -- original stop loss
    trail_sl     REAL    NOT NULL,           -- current trailing SL (ratchets up)
    tp1          REAL    NOT NULL,
    tp2          REAL    NOT NULL,
    tp3          REAL    NOT NULL,
    tp1_hit      INTEGER NOT NULL DEFAULT 0, -- 0 | 1
    tp2_hit      INTEGER NOT NULL DEFAULT 0,
    tp3_hit      INTEGER NOT NULL DEFAULT 0,
    trail_active INTEGER NOT NULL DEFAULT 0, -- becomes 1 after TP1 hit
    closed       INTEGER NOT NULL DEFAULT 0,
    exit_reason  TEXT,                       -- 'TP3' | 'SL_HIT' | 'manual' | NULL
    rsi_at_entry REAL,                       -- RSI-14 at signal time
    htf_bias     REAL,                       -- EMA-based trend bias
    vol_regime   TEXT,                       -- 'HIGH' | 'NORMAL' | 'LOW'
    bull_score   REAL,
    as_of_date   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`trail_sl` is the single source of truth for the current effective stop — it is initialised to the original SL and ratcheted on each TP hit. This means the `refresh` endpoint only needs to run a single `on_bar()` call rather than replaying all historical bars.

---

## Frontend

### `TPTracker.tsx` — `/tp-tracker`

Two-panel layout:

**Left panel (280 px) — Tracker list**
- Each row shows: ticker, BUY/SELL badge, three TP dots (green = hit, gray = pending), trail badge, status string
- "Track Signal" button opens an inline form: ticker + optional date → `POST /api/tp-tracker/from-signal`
- "Refresh" button calls refresh for all active trackers sequentially

**Right panel — Detail card**

Split into a chart area (left) and a sidebar (right):

*Chart:* recharts `ComposedChart` with 90 days of close price (sourced from `/api/rsi-signal/{ticker}` `rsi_history`) and `ReferenceLine` for each level:
- Entry — indigo dashed
- SL / Trail SL — red solid
- TP1/TP2/TP3 — green dashed (unhit) → green solid (hit), label shows ✓

*Sidebar metrics:*
```
Status:   TP1 ✓ — Trail to Entry   ← colour-coded banner
Entry:    $48.82
SL:       $39.93  (original)
Trail SL: $48.82  (break-even)
──── Take Profits ────
TP1:  $57.71  ✓
TP2:  $66.59  ○
TP3:  $75.48  ○
──── Indicators ────
RSI:       64.0
HTF Bias:  +0.012
Vol:       NORMAL
Bull Sc:   7.20
```

### SniperLab Integration

After running an optimisation, the "Latest Signal" card shows a **Track This Signal** button (hidden for HOLD signals). Clicking it calls `POST /api/tp-tracker/from-signal` and shows an inline link to `/tp-tracker`.

---

## Data Flow — End to End

```
User clicks "Track This Signal" in SniperLab (or "Track Signal" in TP Tracker)
  ↓
POST /api/tp-tracker/from-signal  { ticker: "IREN", date: "2026-05-11" }
  ↓
compute_sniper_signal("IREN")
  → loads 15yr price history from cache / yfinance
  → builds EMA + RSI + MACD feature frame
  → computes bull_score / bear_score / cross signals
  → compute_stop("long", entry, df, ...)   → sl = $39.93
  → compute_tps("long", entry, sl)         → tp1=$57.71, tp2=$66.59, tp3=$75.48
  → returns { action:"BUY", latest_close:48.82, stop_loss:39.93,
               tp1:57.71, tp2:66.59, tp3:75.48, last_rsi:64.0, ... }
  ↓
signal_to_tracker_row(signal)  → position_targets row dict
  ↓
repository.create_tracker(row) → INSERT → id=42
  ↓
TrackerOut response  { id:42, trail_status:"Open", ... }

────────────────── Later, user clicks Refresh ──────────────────

POST /api/tp-tracker/IREN/refresh
  ↓
bridge.fetch_current_price("IREN")  → $58.50  (yfinance fast_info)
  ↓
bridge.tick_tracker(row, 58.50)
  → ProgressiveTrail(side='long', entry=48.82, sl=39.93, tp1=57.71, ...)
  → on_bar(high=58.50, low=58.50)
  → price > tp1 ($57.71) → tp1_hit=True, sl ratchets to entry ($48.82)
  → returns { trail_sl:48.82, tp1_hit:1, trail_active:1 }
  ↓
repository.update_tracker(42, { trail_sl:48.82, tp1_hit:1, trail_active:1 })
  ↓
TrackerOut response  { trail_status:"TP1 ✓ — Trail to Entry", trail_sl:48.82, ... }
  ↓
UI updates: TP1 dot turns green, Trail badge appears, chart SL line moves to $48.82
```

---

## Files Changed / Created

| File | Change |
|------|--------|
| `api/database.py` | Added `position_targets` table + migration guards |
| `tradingagents/quant_ml/signals/sniper_signal.py` | Added `last_rsi`, `htf_bias` to output dict |
| `api/main.py` | Registered `tp_tracker_router`; added `GET /api/sniper-signal/{ticker}` |
| `api/tp_tracker/__init__.py` | New — exports router |
| `api/tp_tracker/schemas.py` | New — `TrackerCreate`, `TrackerOut`, `TrackerSignalRequest` |
| `api/tp_tracker/repository.py` | New — SQLite CRUD for `position_targets` |
| `api/tp_tracker/bridge.py` | New — signal→row, tick_tracker, describe_trail_status |
| `api/tp_tracker/routes.py` | New — FastAPI APIRouter with 6 endpoints |
| `frontend/src/pages/TPTracker.tsx` | New — full TP Tracker UI page |
| `frontend/src/App.tsx` | Added SniperLab + TPTracker routes and nav entries |
| `frontend/src/components/Sidebar.tsx` | Added TP Tracker nav entry |
| `frontend/src/pages/SniperLab.tsx` | Added "Track This Signal" button + state/handler |

---

## Testing

Start the API server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Create a tracker from signal:
```bash
curl -s -X POST localhost:8000/api/tp-tracker/from-signal \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"IREN"}' | python -m json.tool
```

Refresh and check trail update:
```bash
curl -s -X POST localhost:8000/api/tp-tracker/IREN/refresh | python -m json.tool
```

List all active trackers:
```bash
curl -s localhost:8000/api/tp-tracker | python -m json.tool
```

Open the UI at `http://localhost:5173/tp-tracker`.
