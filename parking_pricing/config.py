"""All tunable parameters of the pricing engine live here.

Every model reads from a single frozen :class:`PricingConfig`, so an experiment
is fully described by one object (and can be loaded from / dumped to JSON).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PricingConfig:
    # ------------------------------------------------------------------ bounds
    base_price: float = 10.0
    #: final price is always clipped to [min_multiplier, max_multiplier] * base_price
    min_multiplier: float = 0.5
    max_multiplier: float = 2.0

    # ------------------------------------------------- Model 1: baseline linear
    #: price moves by ``alpha * (occupancy_rate - target_occupancy)`` per reading,
    #: starting from ``base_price`` at the beginning of each day.
    baseline_alpha: float = 1.0
    target_occupancy: float = 0.6

    # ------------------------------------------------- Model 2: demand based
    # demand = w_occ*occ + w_queue*queue - w_traffic*traffic
    #          + w_special*special + w_vehicle*vehicle + w_trend*trend
    # Every input feature is normalised to [0, 1] (trend to [-1, 1]).
    w_occupancy: float = 0.55
    w_queue: float = 0.20
    w_traffic: float = 0.10
    w_special: float = 0.10
    w_vehicle: float = 0.05
    w_trend: float = 0.10
    #: how strongly normalised demand moves the price around ``base_price``
    demand_lambda: float = 1.5
    #: EMA factor applied within a day (1.0 = no smoothing)
    smoothing: float = 0.5

    # Reference ("neutral") conditions at which Model 2 charges exactly base_price.
    neutral_occupancy: float = 0.6
    neutral_queue: float = 0.3
    neutral_traffic: float = 0.5
    neutral_vehicle: float = 0.7

    # ---------------------------------------------- Model 3: competitive
    neighbor_radius_km: float = 1.0
    #: fraction of the gap to the (distance-weighted) competitor price we close
    competition_kappa: float = 0.3
    #: occupancy rate at or above which a lot is considered full
    full_threshold: float = 0.9
    #: drivers are rerouted to the neighbour with space that minimises
    #: ``price + reroute_cost_per_km * distance`` (walking/driving penalty)
    reroute_cost_per_km: float = 2.0

    # ------------------------------------------------------ feature encoding
    #: queue length at which the queue feature saturates at 1.0
    queue_saturation: int = 10
    #: occupancy change between consecutive readings that maps to trend = +/-1
    trend_saturation: float = 0.25
    traffic_levels: Mapping[str, float] = field(
        default_factory=lambda: {"low": 0.0, "average": 0.5, "high": 1.0}
    )
    vehicle_weights: Mapping[str, float] = field(
        default_factory=lambda: {"cycle": 0.2, "bike": 0.4, "car": 0.7, "truck": 1.0}
    )

    # ------------------------------------------------------------- helpers
    @property
    def min_price(self) -> float:
        return self.base_price * self.min_multiplier

    @property
    def max_price(self) -> float:
        return self.base_price * self.max_multiplier

    @property
    def demand_range(self) -> tuple[float, float]:
        """Theoretical (min, max) of the raw demand score."""
        lo = -self.w_traffic - self.w_trend
        hi = (
            self.w_occupancy
            + self.w_queue
            + self.w_special
            + self.w_vehicle
            + self.w_trend
        )
        return lo, hi

    def validate(self) -> PricingConfig:
        if self.base_price <= 0:
            raise ValueError("base_price must be positive")
        if not 0 < self.min_multiplier <= 1 <= self.max_multiplier:
            raise ValueError("need 0 < min_multiplier <= 1 <= max_multiplier")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0 <= self.competition_kappa <= 1:
            raise ValueError("competition_kappa must be in [0, 1]")
        if self.reroute_cost_per_km < 0:
            raise ValueError("reroute_cost_per_km must be non-negative")
        if not 0 < self.full_threshold <= 1:
            raise ValueError("full_threshold must be in (0, 1]")
        weights = [
            self.w_occupancy, self.w_queue, self.w_traffic,
            self.w_special, self.w_vehicle, self.w_trend,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("demand weights must be non-negative")
        lo, hi = self.demand_range
        if hi <= lo:
            raise ValueError("demand weights must not all be zero")
        return self

    # ---------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["traffic_levels"] = dict(self.traffic_levels)
        d["vehicle_weights"] = dict(self.vehicle_weights)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PricingConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> PricingConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **overrides: Any) -> PricingConfig:
        return replace(self, **overrides).validate()


DEFAULT_CONFIG = PricingConfig().validate()
