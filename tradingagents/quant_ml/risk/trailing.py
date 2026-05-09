"""
trailing.py
-----------
Progressive trailing stop state machine.

After TP1 hit → SL moves to entry (breakeven)
After TP2 hit → SL moves to TP1
After TP3 hit → SL moves to TP2   (then position closes entirely)

Each bar is evaluated with the SL value as of the START of the bar, so that
a single bar cannot both "hit TP1" and "stop out at the new TP1-level SL" —
this mirrors the Pine `canCheckTPSL = bar_index > entryBar` guard.

Usage:
    trail = ProgressiveTrail(side="long", entry=100, sl=95, tp1=105, tp2=110, tp3=115)
    for bar in bars_after_entry:
        event = trail.on_bar(bar.high, bar.low)
        # event ∈ {None, "TP1", "TP2", "TP3", "SL_HIT"}
        if trail.closed:
            break
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressiveTrail:
    side: str            # "long" or "short"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float

    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    closed: bool = False
    exit_reason: Optional[str] = None  # "TP3" | "SL_HIT"

    def on_bar(self, high: float, low: float) -> Optional[str]:
        """Process one bar. Returns the first event of: None, TP1, TP2, TP3, SL_HIT.

        Evaluates SL against the START-of-bar value, then checks TPs, then
        ratchets the SL. This ordering avoids same-bar conflicts between a new
        TP hit and the subsequent SL-bump.
        """
        if self.closed:
            return None

        if self.side == "long":
            if low <= self.sl:
                self.closed = True
                self.exit_reason = "SL_HIT"
                return "SL_HIT"

            event: Optional[str] = None
            if not self.tp3_hit and high >= self.tp3:
                self.tp1_hit = self.tp2_hit = self.tp3_hit = True
                self.sl = self.tp2
                self.closed = True
                self.exit_reason = "TP3"
                return "TP3"
            if not self.tp2_hit and high >= self.tp2:
                self.tp1_hit = self.tp2_hit = True
                self.sl = self.tp1
                event = "TP2"
            elif not self.tp1_hit and high >= self.tp1:
                self.tp1_hit = True
                self.sl = self.entry
                event = "TP1"
            return event

        # short
        if high >= self.sl:
            self.closed = True
            self.exit_reason = "SL_HIT"
            return "SL_HIT"

        event = None
        if not self.tp3_hit and low <= self.tp3:
            self.tp1_hit = self.tp2_hit = self.tp3_hit = True
            self.sl = self.tp2
            self.closed = True
            self.exit_reason = "TP3"
            return "TP3"
        if not self.tp2_hit and low <= self.tp2:
            self.tp1_hit = self.tp2_hit = True
            self.sl = self.tp1
            event = "TP2"
        elif not self.tp1_hit and low <= self.tp1:
            self.tp1_hit = True
            self.sl = self.entry
            event = "TP1"
        return event

    def r_multiple(self) -> float:
        """Final R-multiple outcome: +3 / +2 / +1 / -1."""
        if self.tp3_hit:
            return 3.0
        if self.tp2_hit:
            return 2.0
        if self.tp1_hit:
            return 1.0
        if self.exit_reason == "SL_HIT":
            return -1.0
        return 0.0  # still open
