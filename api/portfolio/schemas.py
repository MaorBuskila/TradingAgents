from pydantic import BaseModel
from typing import Optional


class PortfolioPosition(BaseModel):
    id: Optional[int] = None
    ticker: str
    quantity: float
    cost_basis: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    exchange: Optional[str] = "US"
    target_weight: Optional[float] = None
    notes: Optional[str] = None
    has_lots: Optional[bool] = False
    follow_ticker: Optional[str] = None
    manual_price: Optional[float] = None
    category: Optional[str] = None


class PortfolioCreateUpdate(BaseModel):
    ticker: str
    quantity: float
    cost_basis: float
    exchange: Optional[str] = "US"
    target_weight: Optional[float] = None
    notes: Optional[str] = None
    follow_ticker: Optional[str] = None
    category: Optional[str] = None


class PortfolioMetaPatch(BaseModel):
    exchange: Optional[str] = None
    target_weight: Optional[float] = None
    notes: Optional[str] = None
    follow_ticker: Optional[str] = None
    manual_price: Optional[float] = None
    category: Optional[str] = None


class PositionLot(BaseModel):
    id: Optional[int] = None
    position_id: int
    purchased_at: str
    price_per_share: float
    quantity: float
    notes: Optional[str] = None
    created_at: Optional[str] = None


class PositionLotCreate(BaseModel):
    purchased_at: str
    price_per_share: float
    quantity: float
    notes: Optional[str] = None
