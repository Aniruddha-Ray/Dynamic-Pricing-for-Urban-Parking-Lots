"""Stateful, event-driven pricing engine.

The engine consumes parking readings one timestamp at a time (exactly as they
would arrive from sensors) and emits a :class:`Quote` with all three model
prices for every reading. It keeps only O(#lots) state, so it can sit behind a
REST endpoint, a message queue consumer or a Pathway ``subscribe`` callback.

Semantics
---------
* Models 1 and 2 are per lot and reset at the start of each calendar day.
* Model 3 compares a lot with its neighbours' *Model 2* prices as of the same
  timestamp (latest reading ``<= t`` on the same day). Using Model 2 rather
  than Model 3 avoids feedback loops between neighbours, and processing all
  readings of one timestamp together makes the result independent of the order
  in which lots report.
* Readings older than the lot's last processed reading are rejected.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from itertools import groupby
from typing import Any

import pandas as pd

from . import models
from .config import DEFAULT_CONFIG, PricingConfig
from .geo import build_neighbor_map

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reading:
    lot_id: str
    timestamp: datetime
    occupancy: int
    capacity: int
    queue_length: int = 0
    traffic: str = "average"
    vehicle_type: str = "car"
    is_special_day: int = 0

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Reading:
        ts = m["timestamp"]
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        return cls(
            lot_id=str(m["lot_id"]),
            timestamp=ts,
            occupancy=int(m["occupancy"]),
            capacity=int(m["capacity"]),
            queue_length=int(m.get("queue_length", 0)),
            traffic=str(m.get("traffic", "average")),
            vehicle_type=str(m.get("vehicle_type", "car")),
            is_special_day=int(m.get("is_special_day", 0)),
        )


@dataclass(frozen=True)
class Quote:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _LotState:
    day: date
    timestamp: datetime
    occupancy_rate: float
    price_baseline: float
    price_demand: float


@dataclass
class RejectedReading:
    reading: Reading
    reason: str


@dataclass
class PricingEngine:
    lots: pd.DataFrame
    cfg: PricingConfig = DEFAULT_CONFIG
    neighbors: Mapping[str, list[tuple[str, float]]] = field(init=False)

    def __post_init__(self) -> None:
        self.cfg.validate()
        self.neighbors = build_neighbor_map(self.lots, self.cfg.neighbor_radius_km)
        self._state: dict[str, _LotState] = {}
        self._latest: dict[str, Quote] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ public API
    def process(
        self, readings: Iterable[Reading]
    ) -> tuple[list[Quote], list[RejectedReading]]:
        """Price a batch of readings. Returns ``(quotes, rejected)``."""
        ordered = sorted(readings, key=lambda r: (r.timestamp, r.lot_id))
        quotes: list[Quote] = []
        rejected: list[RejectedReading] = []
        with self._lock:
            for _, group in groupby(ordered, key=lambda r: r.timestamp):
                q, rej = self._process_instant(list(group))
                quotes.extend(q)
                rejected.extend(rej)
        return quotes, rejected

    def latest(self) -> dict[str, Quote]:
        with self._lock:
            return dict(self._latest)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()
            self._latest.clear()

    # ------------------------------------------------------------- internals
    def _process_instant(
        self, readings: list[Reading]
    ) -> tuple[list[Quote], list[RejectedReading]]:
        """All readings here share one timestamp."""
        accepted: list[tuple[Reading, float, float]] = []
        rejected: list[RejectedReading] = []
        seen: set[str] = set()

        # pass 1: per-lot models (1 and 2) update the lot state
        for r in readings:
            if r.lot_id in seen:
                rejected.append(RejectedReading(r, "duplicate reading for this timestamp"))
                continue
            if r.capacity <= 0 or r.occupancy < 0 or r.queue_length < 0:
                rejected.append(RejectedReading(r, "invalid capacity/occupancy/queue"))
                continue
            prev = self._state.get(r.lot_id)
            if prev is not None and r.timestamp <= prev.timestamp:
                rejected.append(
                    RejectedReading(r, f"stale: lot already priced at {prev.timestamp}")
                )
                continue
            seen.add(r.lot_id)

            same_day = prev is not None and prev.day == r.timestamp.date()
            f = models.encode_features(
                occupancy=r.occupancy,
                capacity=r.capacity,
                queue_length=r.queue_length,
                traffic=r.traffic,
                vehicle_type=r.vehicle_type,
                is_special_day=r.is_special_day,
                prev_occupancy_rate=prev.occupancy_rate if same_day else None,
                cfg=self.cfg,
            )
            p1 = models.baseline_price(
                prev.price_baseline if same_day else None, f.occupancy_rate, self.cfg
            )
            p2, nd = models.demand_price(
                prev.price_demand if same_day else None, f, self.cfg
            )
            self._state[r.lot_id] = _LotState(
                day=r.timestamp.date(),
                timestamp=r.timestamp,
                occupancy_rate=f.occupancy_rate,
                price_baseline=p1,
                price_demand=p2,
            )
            accepted.append((r, nd, p1))

        # pass 2: competitive model, now that every lot at this instant is updated
        quotes = []
        for r, nd, p1 in accepted:
            st = self._state[r.lot_id]
            decision = models.competitive_price(
                st.price_demand, st.occupancy_rate, self._competitors(r), self.cfg
            )
            q = Quote(
                timestamp=r.timestamp,
                lot_id=r.lot_id,
                occupancy_rate=st.occupancy_rate,
                demand=nd,
                price_baseline=p1,
                price_demand=st.price_demand,
                price_competitive=decision.price,
                competitor_price=decision.competitor_price,
                action=decision.action,
                reroute_to=decision.reroute_to,
            )
            self._latest[r.lot_id] = q
            quotes.append(q)

        for rej in rejected:
            log.warning("rejected reading %s @ %s: %s",
                        rej.reading.lot_id, rej.reading.timestamp, rej.reason)
        return quotes, rejected

    def _competitors(self, r: Reading) -> list[models.Competitor]:
        out = []
        for nid, dist in self.neighbors.get(r.lot_id, []):
            st = self._state.get(nid)
            if st is None or st.day != r.timestamp.date() or st.timestamp > r.timestamp:
                continue
            out.append(models.Competitor(nid, st.price_demand, st.occupancy_rate, dist))
        return out
