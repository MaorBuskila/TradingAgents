"""
direction_lock.py
-----------------
After a long fires, only a short may fire next (and vice versa), until the
current trade closes via SL/TP3/opposing signal.

Matches the Pine script's "direction lock" — prevents same-side re-entries
that would pyramid during a sustained move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DirectionLock:
    last_side: Optional[str] = None  # "long" | "short" | None
    open: bool = False

    def can_enter(self, proposed_side: str) -> bool:
        """Return True iff `proposed_side` is allowed given current state."""
        if proposed_side not in ("long", "short"):
            raise ValueError("proposed_side must be 'long' or 'short'")
        if not self.open and self.last_side is None:
            return True  # first trade — anything goes
        if self.open:
            return False  # already in a position, no new entries
        # between trades: require alternation
        return proposed_side != self.last_side

    def on_enter(self, side: str) -> None:
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        if self.open:
            raise RuntimeError("Cannot enter: trade already open")
        self.last_side = side
        self.open = True

    def on_exit(self) -> None:
        self.open = False
