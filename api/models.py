from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

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

class PortfolioPosition(BaseModel):
    id: Optional[int] = None
    ticker: str
    quantity: float
    cost_basis: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None

class PortfolioCreateUpdate(BaseModel):
    ticker: str
    quantity: float
    cost_basis: float


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
