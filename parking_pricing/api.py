"""REST API that serves live prices and ingests new sensor readings.

    uvicorn parking_pricing.api:app --reload
    # or: parking-pricing api

On startup the engine is warmed up by replaying the bundled dataset (disable
with ``PARKING_WARM_START=0``), so ``GET /prices`` immediately has data. New
readings POSTed to ``/readings`` are priced by the same engine in real time.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG, PricingConfig
from .data import DEFAULT_DATASET, load_dataset, lot_registry
from .engine import PricingEngine, Quote, Reading
from .replay import run_replay

HISTORY_LIMIT = 5000


class ReadingIn(BaseModel):
    lot_id: str
    timestamp: datetime
    occupancy: int = Field(ge=0)
    capacity: int = Field(gt=0)
    queue_length: int = Field(0, ge=0)
    traffic: str = Field("average", description="low | average | high")
    vehicle_type: str = Field("car", description="cycle | bike | car | truck")
    is_special_day: int = Field(0, ge=0, le=1)


class QuoteOut(BaseModel):
    timestamp: datetime
    lot_id: str
    occupancy_rate: float
    demand: float
    price_baseline: float
    price_demand: float
    price_competitive: float
    competitor_price: float | None
    action: str
    reroute_to: str | None

    @classmethod
    def from_quote(cls, q: Quote) -> QuoteOut:
        return cls(**q.to_dict())


class IngestResult(BaseModel):
    quotes: list[QuoteOut]
    rejected: list[dict]


class LotOut(BaseModel):
    lot_id: str
    capacity: int
    latitude: float
    longitude: float
    valid_coords: bool
    neighbors: list[str]


class AppState:
    def __init__(self, engine: PricingEngine, history: pd.DataFrame):
        self.engine = engine
        self.history = history


def _build_state(cfg: PricingConfig, warm_start: bool, dataset: str) -> AppState:
    df = load_dataset(dataset)
    engine = PricingEngine(lot_registry(df), cfg)
    if warm_start:
        history = run_replay(df, cfg, engine=engine)
    else:
        history = pd.DataFrame()
    return AppState(engine, history)


def create_app(
    cfg: PricingConfig = DEFAULT_CONFIG,
    *,
    warm_start: bool | None = None,
    dataset: str | None = None,
) -> FastAPI:
    if warm_start is None:
        warm_start = os.environ.get("PARKING_WARM_START", "1") != "0"
    dataset = dataset or str(DEFAULT_DATASET)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pricing = _build_state(cfg, warm_start, dataset)
        yield

    app = FastAPI(
        title="Urban Parking Dynamic Pricing",
        version="2.0.0",
        description="Real-time prices from three models: baseline, demand-based, competitive.",
        lifespan=lifespan,
    )

    def state() -> AppState:
        return app.state.pricing

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "lots_priced": len(state().engine.latest())}

    @app.get("/config")
    def config() -> dict:
        return cfg.to_dict()

    @app.get("/lots", response_model=list[LotOut])
    def lots() -> list[LotOut]:
        eng = state().engine
        return [
            LotOut(
                lot_id=lot,
                capacity=int(row.capacity),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                valid_coords=bool(row.valid_coords),
                neighbors=[n for n, _ in eng.neighbors.get(lot, [])],
            )
            for lot, row in eng.lots.iterrows()
        ]

    @app.get("/prices", response_model=list[QuoteOut])
    def latest_prices() -> list[QuoteOut]:
        latest = state().engine.latest()
        return [QuoteOut.from_quote(latest[k]) for k in sorted(latest)]

    @app.get("/prices/{lot_id}", response_model=QuoteOut)
    def latest_price(lot_id: str) -> QuoteOut:
        q = state().engine.latest().get(lot_id)
        if q is None:
            raise HTTPException(404, f"no price for lot {lot_id!r}")
        return QuoteOut.from_quote(q)

    @app.get("/history/{lot_id}")
    def history(
        lot_id: str,
        day: str | None = Query(None, description="YYYY-MM-DD"),
        limit: int = Query(500, ge=1, le=HISTORY_LIMIT),
    ) -> list[dict]:
        h = state().history
        if h.empty:
            return []
        h = h[h["lot_id"] == lot_id]
        if day:
            try:
                d = pd.Timestamp(day).date()
            except ValueError as exc:
                raise HTTPException(422, "day must be YYYY-MM-DD") from exc
            h = h[h["timestamp"].dt.date == d]
        cols = ["timestamp", "occupancy_rate", "price_baseline",
                "price_demand", "price_competitive", "action", "reroute_to"]
        out = h[cols].tail(limit).astype(object)
        return out.where(out.notna(), None).to_dict(orient="records")

    @app.post("/readings", response_model=IngestResult)
    def ingest(readings: list[ReadingIn]) -> IngestResult:
        quotes, rejected = state().engine.process(
            Reading(**r.model_dump()) for r in readings
        )
        return IngestResult(
            quotes=[QuoteOut.from_quote(q) for q in quotes],
            rejected=[
                {"lot_id": r.reading.lot_id, "timestamp": r.reading.timestamp, "reason": r.reason}
                for r in rejected
            ],
        )

    return app


app = create_app()
