"""Replay a historical dataset through the engine as if it were a live stream."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from .config import DEFAULT_CONFIG, PricingConfig
from .data import lot_registry
from .engine import PricingEngine, Quote, Reading

QUOTE_COLUMNS = [
    "timestamp", "lot_id", "occupancy_rate", "demand",
    "price_baseline", "price_demand", "price_competitive",
    "competitor_price", "action", "reroute_to",
]


def iter_readings(df: pd.DataFrame) -> Iterator[Reading]:
    cols = ["lot_id", "timestamp", "occupancy", "capacity", "queue_length",
            "traffic", "vehicle_type", "is_special_day"]
    for row in df[cols].itertuples(index=False):
        yield Reading(
            lot_id=row.lot_id,
            timestamp=row.timestamp.to_pydatetime(),
            occupancy=int(row.occupancy),
            capacity=int(row.capacity),
            queue_length=int(row.queue_length),
            traffic=row.traffic,
            vehicle_type=row.vehicle_type,
            is_special_day=int(row.is_special_day),
        )


def quotes_to_frame(quotes: list[Quote]) -> pd.DataFrame:
    frame = pd.DataFrame([q.to_dict() for q in quotes], columns=QUOTE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def run_replay(
    df: pd.DataFrame,
    cfg: PricingConfig = DEFAULT_CONFIG,
    *,
    engine: PricingEngine | None = None,
    on_quotes: Callable[[list[Quote]], None] | None = None,
) -> pd.DataFrame:
    """Stream ``df`` through a pricing engine one timestamp at a time.

    Returns the readings joined with every model's price.
    """
    engine = engine or PricingEngine(lot_registry(df), cfg)
    all_quotes: list[Quote] = []
    for _, chunk in df.groupby("timestamp", sort=True):
        quotes, _ = engine.process(iter_readings(chunk))
        all_quotes.extend(quotes)
        if on_quotes:
            on_quotes(quotes)

    prices = quotes_to_frame(all_quotes)
    return df.merge(prices, on=["timestamp", "lot_id"], how="left", validate="one_to_one")
