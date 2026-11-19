"""Real-time pricing pipeline built on Pathway.

Dataflow::

    readings (replay_csv / csv / kafka ...)
        |-- groupby(lot, day) -> sorted history -> UDF: Models 1 & 2  -> prices
        |                                                                |
        |       static neighbour pairs --join--> neighbours' prices -----+
        |                                                                v
        |                                          UDF: Model 3 -> live_prices
        `-- tumbling 1-day window per lot -> daily occupancy KPIs -> daily_summary

Every new reading updates its lot's row in ``live_prices`` (Pathway emits the
change as a retraction + insertion), so a sink receives a continuous stream of
price updates. The maths lives in :mod:`parking_pricing.models`, shared with
the in-process engine, so the two backends agree.

Pathway has no Windows wheels: on Windows use Docker, WSL or Colab.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import models
from .config import DEFAULT_CONFIG, PricingConfig
from .data import READING_COLUMNS, lot_registry, to_stream_csv
from .geo import build_neighbor_map, neighbor_pairs_frame

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def require_pathway():
    try:
        import pathway as pw
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Pathway is not installed. `pip install pathway` on Linux/macOS, or run "
            "`docker build -t parking-pricing . && docker run parking-pricing stream`."
        ) from exc
    if not hasattr(pw, "Schema"):  # the Windows placeholder package
        raise RuntimeError(
            "The installed `pathway` package is the non-functional Windows placeholder. "
            "Run the streaming pipeline in Docker, WSL or Colab instead."
        )
    return pw


@dataclass
class PipelineTables:
    readings: Any
    live_prices: Any
    daily_summary: Any


def _reading_schema(pw):
    class ReadingSchema(pw.Schema):
        timestamp: str
        lot_id: str
        capacity: int
        occupancy: int
        queue_length: int
        traffic: str
        vehicle_type: str
        is_special_day: int
        latitude: float
        longitude: float

    return ReadingSchema


def read_stream(path: str | Path, *, mode: str = "replay", input_rate: float = 1000.0):
    """Open the readings CSV as a Pathway table.

    ``mode="replay"`` emits rows at ``input_rate`` rows/second to simulate live
    sensors; ``mode="static"`` reads the whole file at once (tests / backfills).
    """
    pw = require_pathway()
    schema = _reading_schema(pw)
    if mode == "replay":
        return pw.demo.replay_csv(str(path), schema=schema, input_rate=input_rate)
    if mode == "static":
        return pw.io.csv.read(str(path), schema=schema, mode="static")
    raise ValueError(f"unknown mode {mode!r}")


def build_pipeline(
    readings, neighbor_pairs: pd.DataFrame, cfg: PricingConfig = DEFAULT_CONFIG
) -> PipelineTables:
    pw = require_pathway()
    cfg.validate()

    # ---------------------------------------------------------------- UDFs
    @pw.udf
    def per_lot_models(history: tuple) -> tuple[float, float, float, float]:
        # (occupancy_rate, demand, price_baseline, price_demand) of the latest reading
        return models.price_lot_day(history, cfg)[-1]

    @pw.udf
    def competitive(
        own_price: float, own_occ: float, competitors: Any
    ) -> tuple[float, float, str, str]:
        comps = [
            models.Competitor(lot_id=c[0], price=c[1], occupancy_rate=c[2], distance_km=c[3])
            for c in (competitors or ())
        ]
        d = models.competitive_price(own_price, own_occ, comps, cfg)
        comp_price = math.nan if d.competitor_price is None else d.competitor_price
        return (d.price, comp_price, d.action, d.reroute_to or "")

    # ------------------------------------------------ Models 1 & 2 (per lot)
    readings = readings.with_columns(day=pw.this.timestamp.str.slice(0, 10))
    history = readings.groupby(pw.this.lot_id, pw.this.day).reduce(
        pw.this.lot_id,
        pw.this.day,
        timestamp=pw.reducers.max(pw.this.timestamp),
        history=pw.reducers.sorted_tuple(
            pw.make_tuple(
                pw.this.timestamp, pw.this.occupancy, pw.this.capacity,
                pw.this.queue_length, pw.this.traffic, pw.this.vehicle_type,
                pw.this.is_special_day,
            )
        ),
    )
    prices = history.select(
        pw.this.lot_id, pw.this.day, pw.this.timestamp,
        m=per_lot_models(pw.this.history),
    ).select(
        pw.this.lot_id,
        pw.this.day,
        pw.this.timestamp,
        occupancy_rate=pw.this.m[0],
        demand=pw.this.m[1],
        price_baseline=pw.this.m[2],
        price_demand=pw.this.m[3],
    )

    # ------------------------------------------------ Model 3 (competition)
    pairs = pw.debug.table_from_pandas(
        neighbor_pairs[["lot_id", "neighbor_id", "distance_km"]].astype(
            {"lot_id": str, "neighbor_id": str, "distance_km": float}
        )
    )
    lot_neighbors = prices.join(pairs, pw.left.lot_id == pw.right.lot_id).select(
        pw.left.lot_id, pw.left.day, pw.right.neighbor_id, pw.right.distance_km,
    )
    neighbor_prices = lot_neighbors.join(
        prices,
        pw.left.neighbor_id == pw.right.lot_id,
        pw.left.day == pw.right.day,
    ).select(
        pw.left.lot_id,
        pw.left.day,
        competitor=pw.make_tuple(
            pw.right.lot_id, pw.right.price_demand,
            pw.right.occupancy_rate, pw.left.distance_km,
        ),
    )
    competition = neighbor_prices.groupby(pw.this.lot_id, pw.this.day).reduce(
        pw.this.lot_id,
        pw.this.day,
        competitors=pw.reducers.sorted_tuple(pw.this.competitor),
    )
    live_prices = (
        prices.join_left(
            competition,
            pw.left.lot_id == pw.right.lot_id,
            pw.left.day == pw.right.day,
        )
        .select(
            *pw.left,
            c=competitive(pw.left.price_demand, pw.left.occupancy_rate, pw.right.competitors),
        )
        .select(
            pw.this.timestamp,
            pw.this.lot_id,
            pw.this.occupancy_rate,
            pw.this.demand,
            pw.this.price_baseline,
            pw.this.price_demand,
            price_competitive=pw.this.c[0],
            competitor_price=pw.this.c[1],
            action=pw.this.c[2],
            reroute_to=pw.this.c[3],
        )
    )

    # ------------------------------- daily KPIs (tumbling window, per lot)
    timed = readings.with_columns(
        t=pw.this.timestamp.dt.strptime(TIME_FMT),
        occ_rate=pw.apply_with_type(
            lambda o, c: models.occupancy_rate(o, c), float, pw.this.occupancy, pw.this.capacity
        ),
    )
    daily_summary = timed.windowby(
        pw.this.t,
        window=pw.temporal.tumbling(dt.timedelta(days=1)),
        instance=pw.this.lot_id,
        behavior=pw.temporal.exactly_once_behavior(),
    ).reduce(
        lot_id=pw.this._pw_instance,
        day_start=pw.this._pw_window_start,
        readings=pw.reducers.count(),
        occupancy_mean=pw.reducers.avg(pw.this.occ_rate),
        occupancy_peak=pw.reducers.max(pw.this.occ_rate),
        queue_peak=pw.reducers.max(pw.this.queue_length),
    )

    return PipelineTables(readings=readings, live_prices=live_prices, daily_summary=daily_summary)


def prepare_inputs(
    df: pd.DataFrame, out_dir: str | Path, cfg: PricingConfig = DEFAULT_CONFIG
) -> tuple[Path, pd.DataFrame]:
    """Write the stream CSV and compute the static neighbour pairs."""
    out_dir = Path(out_dir)
    stream_csv = to_stream_csv(df[READING_COLUMNS], out_dir / "parking_stream.csv")
    neighbors = build_neighbor_map(lot_registry(df), cfg.neighbor_radius_km)
    return stream_csv, neighbor_pairs_frame(neighbors)


def run_stream(
    df: pd.DataFrame,
    out_dir: str | Path = "outputs",
    cfg: PricingConfig = DEFAULT_CONFIG,
    *,
    input_rate: float = 1000.0,
    mode: str = "replay",
) -> Path:
    """Run the pipeline to completion, writing CSV change-streams to ``out_dir``."""
    pw = require_pathway()
    out_dir = Path(out_dir)
    stream_csv, pairs = prepare_inputs(df, out_dir, cfg)
    tables = build_pipeline(read_stream(stream_csv, mode=mode, input_rate=input_rate), pairs, cfg)
    pw.io.csv.write(tables.live_prices, str(out_dir / "live_prices_stream.csv"))
    pw.io.csv.write(tables.daily_summary, str(out_dir / "daily_summary_stream.csv"))
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    return out_dir


def live_price_plot(live_prices, lot_id: str | None = None):
    """Live Bokeh chart of a Pathway price table, for notebooks (``pn.Column(...)``)."""
    pw = require_pathway()
    import bokeh.plotting

    table = live_prices.with_columns(t=pw.this.timestamp.dt.strptime(TIME_FMT))
    if lot_id is not None:
        table = table.filter(pw.this.lot_id == lot_id)

    def plotter(source):
        fig = bokeh.plotting.figure(
            height=380, width=860, x_axis_type="datetime",
            title=f"Live price - {lot_id or 'all lots'}",
        )
        fig.line("t", "price_demand", source=source, line_width=2, legend_label="Model 2")
        fig.scatter("t", "price_competitive", source=source, size=4, color="orange",
                    legend_label="Model 3")
        fig.legend.location = "top_left"
        return fig

    return table.plot(plotter, sorting_col="t")
