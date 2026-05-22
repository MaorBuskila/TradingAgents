from __future__ import annotations

from dataclasses import dataclass, field

BASE_FEATURES = ["rsi_14", "ema_ratio_50_200", "macd_hist", "atr_norm_14"]
EXTRA_VOLUME  = ["volume_ratio"]
EXTRA_TREND   = ["adx_14"]


@dataclass(frozen=True)
class RuleSet:
    name: str
    feature_cols: tuple[str, ...]
    veto_rules: tuple[str, ...]


BASE = RuleSet(
    name="BASE",
    feature_cols=tuple(BASE_FEATURES),
    veto_rules=("ema200", "sentiment"),
)

VOLUME = RuleSet(
    name="VOLUME",
    feature_cols=tuple(BASE_FEATURES + EXTRA_VOLUME),
    veto_rules=("ema200", "sentiment", "volume_burst"),
)

TREND = RuleSet(
    name="TREND",
    feature_cols=tuple(BASE_FEATURES + EXTRA_TREND),
    veto_rules=("ema200", "sentiment", "adx_gt_20"),
)

FULL = RuleSet(
    name="FULL",
    feature_cols=tuple(BASE_FEATURES + EXTRA_VOLUME + EXTRA_TREND),
    veto_rules=("ema200", "sentiment", "volume_burst", "adx_gt_20"),
)

PREDEFINED: dict[str, RuleSet] = {
    "BASE": BASE,
    "VOLUME": VOLUME,
    "TREND": TREND,
    "FULL": FULL,
}


def make_custom(feature_cols: list[str], veto_rules: list[str]) -> RuleSet:
    return RuleSet(
        name="CUSTOM",
        feature_cols=tuple(feature_cols),
        veto_rules=tuple(veto_rules),
    )
