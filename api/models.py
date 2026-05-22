from pydantic import BaseModel, Field
from typing import List, Optional


class AnalysisRequest(BaseModel):
    ticker: str
    analysis_date: str
    analysts: List[str]
    research_depth: int
    llm_provider: str
    backend_url: str
    shallow_thinker: str
    deep_thinker: str
    google_thinking_level: Optional[str] = None
    openai_reasoning_effort: Optional[str] = None
    anthropic_effort: Optional[str] = None


class CatalogItemOut(BaseModel):
    id: int
    ticker: str
    name: Optional[str] = None
    category: str
    asset_type: str
    source: str
    is_favorite: bool
    category_label: Optional[str] = None


class CatalogUserAdd(BaseModel):
    ticker: str
    name: str = ""
    category: str
    asset_type: str = "stock"


class FavoriteBody(BaseModel):
    favorite: bool = True


class YouTubeSummarizeRequest(BaseModel):
    url: str = Field(..., description="YouTube watch URL, short URL, or 11-char video ID")
    llm_provider: str = "ollama"
    llm_model: Optional[str] = None


class YouTubeSummarizeResponse(BaseModel):
    id: int
    video_id: str
    url: str
    title: Optional[str] = None
    transcript_chars: int
    summary_en: str
    summary_he: str
    model: str
    provider: str = "openai"


class YouTubeSummaryListItem(BaseModel):
    id: int
    video_id: str
    url: str
    title: Optional[str] = None
    transcript_chars: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str


class YouTubeSummaryDetail(YouTubeSummarizeResponse):
    created_at: str


# ── RSI Optimizer ─────────────────────────────────────────────────────────────

class RsiOptimizeRequest(BaseModel):
    symbol: str                          # e.g. "AAPL"
    date: str                            # e.g. "2025-01-01" — optimize up to this date
    is_days: int = 180                   # in-sample window (training)
    oos_days: int = 90                   # out-of-sample window (validation)
    llm_provider: str = "google"         # "google" | "openai" | "anthropic" | "heuristic"
    llm_model: Optional[str] = None     # e.g. "gemini-2.5-flash" — defaults per provider if None


class RsiPeriodPoint(BaseModel):
    """One bar in the Sharpe-by-period chart."""
    period: int
    is_sharpe: float
    oos_sharpe: float


class RsiOptimizerResult(BaseModel):
    """Results from one optimizer (algo or llm)."""
    optimal_period: int
    optimal_upper: float
    optimal_lower: float
    is_sharpe: float
    oos_sharpe: float
    confidence: str              # "HIGH" | "MEDIUM" | "LOW"
    combos_tested: int
    is_days: int
    oos_days: int
    default_is_sharpe: float
    default_oos_sharpe: float
    period_sharpes: List[RsiPeriodPoint]
    # LLM-only fields (None for algo optimizer)
    regime: Optional[str] = None
    llm_reasoning: Optional[str] = None
    llm_available: Optional[bool] = None
    llm_grid: Optional[dict] = None
    regime_signals: Optional[dict] = None
    token_usage: Optional[dict] = None   # {model, prompt_tokens, completion_tokens, total_tokens, cost_usd}
    debug_logs: Optional[List[str]] = None  # step-by-step log of what happened


class RsiComparisonSummary(BaseModel):
    """Auto-generated comparison between the two optimizers."""
    period_agreement: bool       # did both pick the same period?
    oos_winner: str              # "algo" | "llm" | "tie"
    oos_winner_sharpe: float
    algo_beats_default: bool     # does algo OOS beat RSI-14/70/30 OOS?
    llm_beats_default: bool      # does llm OOS beat RSI-14/70/30 OOS?
    combos_reduction_pct: float  # how much did LLM reduce the search space?
    summary: str                 # 1-2 sentence plain English takeaway


class RsiOptimizeResponse(BaseModel):
    symbol: str
    date: str
    algo: RsiOptimizerResult
    llm: RsiOptimizerResult
    comparison: RsiComparisonSummary


# ── RSI Signal (OPTIMIZE → GET PARAMS → CALC RSI → SIGNAL → ACT) ─────────────

class RsiHistoryPoint(BaseModel):
    date: str
    rsi: Optional[float] = None
    close: float


class RsiSignalResponse(BaseModel):
    ticker: str
    as_of_date: str
    action: Optional[str] = None          # "BUY" | "SELL" | "HOLD"
    current_rsi: Optional[float] = None
    rsi_zone: Optional[str] = None        # "OVERSOLD" | "OVERBOUGHT" | "NEUTRAL"
    period: Optional[int] = None
    overbought: Optional[float] = None
    oversold: Optional[float] = None
    using_cached: bool = False
    confidence: Optional[str] = None
    regime: Optional[str] = None
    optimizer_provider: Optional[str] = None
    optimized_at: Optional[str] = None
    latest_close: Optional[float] = None
    price_change_pct: Optional[float] = None
    rsi_history: List[RsiHistoryPoint] = []
    error: Optional[str] = None


# ── MACD Optimizer ────────────────────────────────────────────────────────────

class MacdOptimizeRequest(BaseModel):
    symbol: str                          # e.g. "AAPL"
    date: str                            # optimize up to this date
    is_days: int = 180
    oos_days: int = 90
    llm_provider: str = "google"
    llm_model: Optional[str] = None


class MacdParamPoint(BaseModel):
    """One entry in the top-20 param combos chart."""
    fast: int
    slow: int
    signal: int
    is_sharpe: float
    oos_sharpe: float


class MacdOptimizerResult(BaseModel):
    optimal_fast: int
    optimal_slow: int
    optimal_signal: int
    is_sharpe: float
    oos_sharpe: float
    confidence: str              # "HIGH" | "MEDIUM" | "LOW"
    combos_tested: int
    is_days: int
    oos_days: int
    default_is_sharpe: float
    default_oos_sharpe: float
    param_sharpes: List[MacdParamPoint]
    # LLM-only
    regime: Optional[str] = None
    llm_reasoning: Optional[str] = None
    llm_available: Optional[bool] = None
    llm_grid: Optional[dict] = None
    regime_signals: Optional[dict] = None
    token_usage: Optional[dict] = None
    debug_logs: Optional[List[str]] = None


class MacdComparisonSummary(BaseModel):
    params_agreement: bool        # did both pick same fast+slow+signal?
    oos_winner: str               # "algo" | "llm" | "tie"
    oos_winner_sharpe: float
    algo_beats_default: bool
    llm_beats_default: bool
    combos_reduction_pct: float
    summary: str


class MacdOptimizeResponse(BaseModel):
    symbol: str
    date: str
    algo: MacdOptimizerResult
    llm: MacdOptimizerResult
    comparison: MacdComparisonSummary


# ── MACD Signal ───────────────────────────────────────────────────────────────

class MacdHistoryPoint(BaseModel):
    date: str
    macd: Optional[float] = None
    signal: Optional[float] = None
    histogram: Optional[float] = None
    close: float


class MacdSignalResponse(BaseModel):
    ticker: str
    as_of_date: str
    action: Optional[str] = None          # "BUY" | "SELL" | "HOLD"
    macd_line: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    histogram_direction: Optional[str] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    using_cached: bool = False
    confidence: Optional[str] = None
    regime: Optional[str] = None
    optimizer_provider: Optional[str] = None
    optimized_at: Optional[str] = None
    latest_close: Optional[float] = None
    price_change_pct: Optional[float] = None
    macd_history: List[MacdHistoryPoint] = []
    error: Optional[str] = None


# ── WFO Analyzer ─────────────────────────────────────────────────────────────

class WfoAnalyzeRequest(BaseModel):
    is_days: int = Field(..., description="In-sample window in calendar days")
    oos_days: int = Field(..., description="Out-of-sample window in calendar days")
    is_sharpe: float = Field(..., description="Annualized Sharpe ratio from IS optimization")
    oos_sharpe: float = Field(..., description="Annualized Sharpe ratio from OOS backtest")
    oos_trade_count: int = Field(..., description="Number of round-trip trades in OOS window")
    bar_frequency: str = Field("daily", description='"weekly" or "daily"')


class WfoAnalysisResult(BaseModel):
    is_days: int
    oos_days: int
    is_sharpe: float
    oos_sharpe: float
    oos_trade_count: int
    bar_frequency: str
    wfe: float
    wfe_label: str
    sample_risk: str
    sample_risk_label: str
    confidence_score: int
    confidence_label: str
    assessment: str


# ── DT-Filtered MACD Optimizer ────────────────────────────────────────────────

class MacdDtOptimizeRequest(BaseModel):
    symbol:        str   = Field(..., description="Ticker symbol, e.g. AAPL")
    date:          str   = Field(..., description="Current date YYYY-MM-DD")
    is_days:       int   = Field(100,  description="Sliding IS window in bars")
    oos_days:      int   = Field(30,   description="OOS / re-train cycle in bars")
    label_horizon: int   = Field(10,   description="Forward bars for profitability label")


class MacdDtSlideResult(BaseModel):
    slide_idx:   int
    is_start:    int
    is_end:      int
    oos_start:   int
    oos_end:     int
    train_acc:   float
    oos_acc:     float


class MacdDtOptimizeResponse(BaseModel):
    symbol:             str
    curr_date:          str
    last_signal:        int           # 1 = Long, 0 = No position
    last_prob_a:        float         # Model A (sliding) P(profitable)
    last_prob_b:        float         # Model B (fixed/crash-aware) P(profitable)
    last_size_mult:     float         # 1.0 = full 5%, 0.5 = half in high-vol
    last_atr_pct:       float         # ATR percentile rank (vol regime proxy)
    avg_oos_acc:        float         # mean OOS accuracy across all slides
    n_slides:           int
    feature_names:      List[str]
    macd_params:        dict
    is_days:            int
    oos_days:           int
    label_horizon:      int
    min_prob_threshold: float
    slides:             List[MacdDtSlideResult] = []


# ── DT Portfolio Optimizer ────────────────────────────────────────────────────

class MacdDtPortfolioRequest(BaseModel):
    symbols:        List[str] = Field(..., description="List of ticker symbols to evaluate")
    date:           str       = Field(..., description="Current date YYYY-MM-DD")
    capital:        float     = Field(100_000.0, description="Total portfolio capital USD")
    min_dollar_vol: float     = Field(5_000_000.0, description="Min 30-day avg dollar volume")
    max_positions:  int       = Field(20, description="Max simultaneous long positions")
    is_days:        int       = Field(100, description="Sliding IS window in bars")
    oos_days:       int       = Field(30,  description="OOS / re-train cycle in bars")


class MacdDtPosition(BaseModel):
    symbol:        str
    signal:        int
    prob_a:        float
    prob_b:        float
    atr_pct:       float
    size_mult:     float
    alloc_pct:     float
    alloc_dollars: float
    avg_oos_acc:   float


class MacdDtPortfolioResponse(BaseModel):
    positions:              List[MacdDtPosition]
    rejected_liquidity:     List[str]
    rejected_no_signal:     List[str]
    rejected_errors:        List[str]
    total_deployed_pct:     float
    total_deployed_usd:     float
    capital:                float
    curr_date:              str
    n_evaluated:            int
    n_selected:             int
    base_alloc_pct:         float
    max_positions:          int
    min_dollar_vol:         float


# ── DT-Filtered RSI Optimizer ─────────────────────────────────────────────────

class RsiDtOptimizeRequest(BaseModel):
    symbol:        str   = Field(...,  description="Ticker symbol, e.g. AAPL")
    date:          str   = Field(...,  description="Current date YYYY-MM-DD")
    is_days:       int   = Field(100,  description="Sliding IS window in bars")
    oos_days:      int   = Field(30,   description="OOS / re-train cycle in bars")
    label_horizon: int   = Field(10,   description="Forward bars for profitability label")
    # WFO-optimized RSI params (override defaults when use_wfo_params=True)
    rsi_period:    int   = Field(14,   description="RSI period (WFO-optimized or default 14)")
    rsi_upper:     float = Field(70.0, description="RSI overbought threshold (WFO or default 70)")
    rsi_lower:     float = Field(30.0, description="RSI oversold threshold (WFO or default 30)")


class RsiDtSlideResult(BaseModel):
    slide_idx:   int
    is_start:    int
    is_end:      int
    oos_start:   int
    oos_end:     int
    train_acc:   float
    oos_acc:     float


class RsiDtOptimizeResponse(BaseModel):
    symbol:             str
    curr_date:          str
    last_signal:        int           # 1 = Long, 0 = No position
    last_prob_a:        float         # Model A (sliding) P(profitable)
    last_prob_b:        float         # Model B (fixed/crash-aware) P(profitable)
    last_size_mult:     float         # 1.0 = full 5%, 0.5 = half in high-vol
    last_atr_pct:       float         # ATR percentile rank
    last_rsi:           float         # Current RSI value
    avg_oos_acc:        float         # mean OOS accuracy across all slides
    n_slides:           int
    feature_names:      List[str]
    rsi_params:         dict          # period, upper, lower (actual values used)
    wfo_params_used:    bool = False  # True when WFO-optimized params were used
    is_days:            int
    oos_days:           int
    label_horizon:      int
    min_prob_threshold: float
    slides:             List[RsiDtSlideResult] = []


# ── RSI DT Portfolio Optimizer ────────────────────────────────────────────────

class RsiDtPortfolioRequest(BaseModel):
    symbols:        List[str] = Field(...,         description="List of ticker symbols to evaluate")
    date:           str       = Field(...,         description="Current date YYYY-MM-DD")
    capital:        float     = Field(100_000.0,   description="Total portfolio capital USD")
    min_dollar_vol: float     = Field(5_000_000.0, description="Min 30-day avg dollar volume")
    max_positions:  int       = Field(20,          description="Max simultaneous long positions")
    is_days:        int       = Field(100,         description="Sliding IS window in bars")
    oos_days:       int       = Field(30,          description="OOS / re-train cycle in bars")
    # WFO-optimized RSI params (shared across all symbols in the universe)
    rsi_period:     int       = Field(14,          description="RSI period (WFO-optimized or default 14)")
    rsi_upper:      float     = Field(70.0,        description="RSI overbought threshold")
    rsi_lower:      float     = Field(30.0,        description="RSI oversold threshold")


class RsiDtPosition(BaseModel):
    symbol:        str
    signal:        int
    prob_a:        float
    prob_b:        float
    atr_pct:       float
    last_rsi:      float
    size_mult:     float
    alloc_pct:     float
    alloc_dollars: float
    avg_oos_acc:   float


class RsiDtPortfolioResponse(BaseModel):
    positions:              List[RsiDtPosition]
    rejected_liquidity:     List[str]
    rejected_no_signal:     List[str]
    rejected_errors:        List[str]
    total_deployed_pct:     float
    total_deployed_usd:     float
    capital:                float
    curr_date:              str
    n_evaluated:            int
    n_selected:             int
    base_alloc_pct:         float
    max_positions:          int
    min_dollar_vol:         float


# ── OOS Integration Test ──────────────────────────────────────────────────────

class OosTestRequest(BaseModel):
    symbol: str
    oos_end: str                      # last analysis date (YYYY-MM-DD)
    n_dates: int = 6                  # evenly-spaced dates to sweep
    oos_start: Optional[str] = None   # default: 18 months before oos_end
    is_days: int = 180
    oos_days: int = 90
    notional: float = 10000.0


class OosTestDateRow(BaseModel):
    date: str
    # MACD-specific (None for RSI rows)
    fast: Optional[int] = None
    slow: Optional[int] = None
    signal_period: Optional[int] = None
    # RSI-specific (None for MACD rows)
    period: Optional[int] = None
    upper: Optional[float] = None
    lower: Optional[float] = None
    # common metrics
    is_sharpe: Optional[float] = None
    oos_sharpe: Optional[float] = None
    confidence: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_dollar: Optional[float] = None
    win_rate: Optional[float] = None
    max_dd_pct: Optional[float] = None
    n_trades: Optional[int] = None
    error: Optional[str] = None


class OosTestSummary(BaseModel):
    total_pnl_dollar: float
    win_rate_pct: float
    n_wins: int
    n_losses: int
    avg_win_dollar: float
    avg_loss_dollar: float
    profit_factor: float           # sum(wins) / abs(sum(losses)); inf if no losses
    avg_oos_sharpe: float
    avg_is_sharpe: float
    high_conf: int
    med_conf: int
    low_conf: int


class OosTestResponse(BaseModel):
    symbol: str
    oos_start: str
    oos_end: str
    is_days: int
    oos_days: int
    notional: float
    n_requested: int
    n_succeeded: int
    rows: List[OosTestDateRow]
    summary: OosTestSummary
    algo_type: str                 # "macd" | "rsi"


# ---------------------------------------------------------------------------
# News Lab models
# ---------------------------------------------------------------------------

class TrendingStockOut(BaseModel):
    symbol: str
    display_name: str
    price: float
    change_pct: float
    volume: int
    market_cap: Optional[float] = None


class EarningsCalendarOut(BaseModel):
    ticker: str
    next_earnings_date: Optional[str] = None
    all_earnings_dates: List[str] = []
    eps_estimate_avg: Optional[float] = None
    eps_estimate_low: Optional[float] = None
    eps_estimate_high: Optional[float] = None
    revenue_estimate_low: Optional[int] = None
    revenue_estimate_high: Optional[int] = None
    ex_dividend_date: Optional[str] = None
    dividend_date: Optional[str] = None


class NewsArticleOut(BaseModel):
    title: str
    publisher: str
    link: str
    pub_date: Optional[str] = None
    summary: str = ""


class RedditTrendingPost(BaseModel):
    title: str
    score: int
    url: str


class RedditTrendingTickerOut(BaseModel):
    symbol: str
    mentions: int
    posts: List[RedditTrendingPost] = []


# ── RSI DT Feature Lab ────────────────────────────────────────────────────────

class RsiDtFeatureLabRequest(BaseModel):
    symbol: str
    date: str
    rule_sets: List[str] = ["BASE", "VOLUME", "TREND", "FULL"]
    custom_features: Optional[List[str]] = None
    custom_vetos: Optional[List[str]] = None
    is_days: int = 1000
    oos_days: int = 20
    label_horizon: int = 10
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    rsi_period: int = 14


class RsiDtFeatureLabResultRow(BaseModel):
    run_id: str
    symbol: str
    date: str
    rule_set_name: str
    features_used: List[str]
    veto_rules: List[str]
    oos_sharpe: float
    hit_rate: float
    trade_count: int
    confidence: str
    n_slides: int
    avg_oos_acc: float
    ran_at: str


class RsiDtFeatureLabResponse(BaseModel):
    run_id: str
    symbol: str
    date: str
    results: List[RsiDtFeatureLabResultRow]
    winner: Optional[str] = None
    ran_at: str


# ── Sniper Lab ────────────────────────────────────────────────────────────────

class SniperOptimizeRequest(BaseModel):
    symbol: str
    date: str
    is_days: int = 180
    oos_days: int = 90
    dt_is_days: int = 1000
    dt_oos_days: int = 20


class SniperEmaResult(BaseModel):
    optimal_fast: int
    optimal_slow: int
    optimal_trend: int
    is_sharpe: float
    oos_sharpe: float
    confidence: str
    combos_tested: int
    default_oos_sharpe: float


class SniperDtResult(BaseModel):
    last_signal: int
    last_prob_a: float
    last_prob_b: float
    threshold_a: float
    threshold_b: float
    oos_sharpe: float
    oos_hit_rate: float
    bull_score_today: float
    grade_veto_ok: bool
    confidence: str


class SniperOptimizeResponse(BaseModel):
    symbol: str
    date: str
    ema: SniperEmaResult
    dt: SniperDtResult
    rsi: Optional[dict] = None
    macd: Optional[dict] = None
    summary: str = ""

