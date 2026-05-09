from typing import List

from fastapi import APIRouter, HTTPException

from .schemas import (
    PortfolioPosition,
    PortfolioCreateUpdate,
    PortfolioMetaPatch,
    PositionLot,
    PositionLotCreate,
)
from . import repository
from .pricing import run_price_refresh
from api.database import set_catalog_favorite

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/positions", response_model=List[PortfolioPosition])
def read_positions():
    rows = repository.get_all_positions()
    positions = []
    for row in rows:
        lots = repository.get_lots_for_position(row["id"])
        pos = PortfolioPosition(**row, has_lots=len(lots) > 0)
        # manual_price takes precedence over auto-fetched current_price
        effective_price = pos.manual_price if pos.manual_price is not None else pos.current_price
        if effective_price is not None:
            pos.current_price = effective_price
            pos.market_value = pos.quantity * effective_price
            total_cost = pos.quantity * pos.cost_basis
            pos.unrealized_pnl = pos.market_value - total_cost
            if total_cost > 0:
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / total_cost) * 100
        positions.append(pos)
    return positions


@router.post("/positions", response_model=PortfolioPosition)
def create_position(pos: PortfolioCreateUpdate):
    pos_id = repository.add_position(pos.ticker, pos.quantity, pos.cost_basis)
    repository.patch_position_meta(pos_id, pos.exchange, pos.target_weight, pos.notes, pos.follow_ticker, category=pos.category)
    # Auto-favorite this ticker in the watchlist catalog
    try:
        set_catalog_favorite(pos.ticker.upper().strip(), True)
    except Exception:
        pass
    return PortfolioPosition(id=pos_id, **pos.dict())


@router.put("/positions/{pos_id}", response_model=PortfolioPosition)
def edit_position(pos_id: int, pos: PortfolioCreateUpdate):
    repository.update_position(pos_id, pos.ticker, pos.quantity, pos.cost_basis)
    repository.patch_position_meta(pos_id, pos.exchange, pos.target_weight, pos.notes, pos.follow_ticker, category=pos.category)
    return PortfolioPosition(id=pos_id, **pos.dict())


@router.patch("/positions/{pos_id}", response_model=PortfolioPosition)
def patch_position(pos_id: int, body: PortfolioMetaPatch):
    repository.patch_position_meta(pos_id, body.exchange, body.target_weight, body.notes, body.follow_ticker, body.manual_price, body.category)
    rows = repository.get_all_positions()
    row = next((r for r in rows if r["id"] == pos_id), None)
    if not row:
        raise HTTPException(status_code=404, detail="Position not found")
    return PortfolioPosition(**row)


@router.delete("/positions/{pos_id}")
def remove_position(pos_id: int):
    repository.delete_position(pos_id)
    return {"status": "success"}


@router.get("/positions/{pos_id}/lots", response_model=List[PositionLot])
def list_lots(pos_id: int):
    return repository.get_lots_for_position(pos_id)


@router.post("/positions/{pos_id}/lots", response_model=PositionLot)
def create_lot(pos_id: int, body: PositionLotCreate):
    lot_id = repository.add_lot(pos_id, body.purchased_at, body.price_per_share, body.quantity, body.notes)
    return PositionLot(id=lot_id, position_id=pos_id, **body.dict())


@router.put("/positions/{pos_id}/lots/{lot_id}", response_model=PositionLot)
def edit_lot(pos_id: int, lot_id: int, body: PositionLotCreate):
    repository.update_lot(lot_id, body.purchased_at, body.price_per_share, body.quantity, body.notes)
    return PositionLot(id=lot_id, position_id=pos_id, **body.dict())


@router.delete("/positions/{pos_id}/lots/{lot_id}")
def remove_lot(pos_id: int, lot_id: int):
    repository.delete_lot(lot_id)
    return {"status": "success"}


@router.post("/refresh-prices")
def refresh_prices():
    return run_price_refresh()


@router.get("/fx-rate")
def get_fx_rate():
    """Return the current USD→ILS exchange rate via yfinance (ILS=X)."""
    import yfinance as yf
    try:
        rate = float(yf.Ticker("ILS=X").fast_info["lastPrice"])
        return {"usd_to_ils": rate}
    except Exception:
        return {"usd_to_ils": 3.7}  # fallback
