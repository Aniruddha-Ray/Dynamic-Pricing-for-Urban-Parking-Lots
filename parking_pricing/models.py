"""The three pricing models, written as small pure functions.

These functions are the single source of truth for the pricing maths. They are
used by the in-process :class:`~parking_pricing.engine.PricingEngine`, by the
REST API and (as UDFs) by the Pathway streaming pipeline, so every backend
produces identical prices.

Model 1 - baseline linear
    Starting from ``base_price`` each day, the price drifts up when the lot is
    busier than ``target_occupancy`` and down when it is emptier::

        p_t = clip(p_{t-1} + alpha * (occupancy_rate_t - target))

Model 2 - demand based
    A weighted, normalised demand score built from occupancy, queue length,
    nearby traffic, special days, vehicle type and the short-term occupancy
    trend is mapped linearly around ``base_price`` and EMA-smoothed::

        demand  = w_occ*occ + w_q*queue - w_tr*traffic + w_sp*special
                  + w_veh*vehicle + w_trend*trend
        d_norm  = (demand - demand_min) / (demand_max - demand_min)      in [0, 1]
        raw     = base * (1 + lambda * (d_norm - d_neutral))
        p_t     = clip(s * raw + (1 - s) * p_{t-1})

Model 3 - competitive
    Takes the Model 2 price and nudges it towards the distance-weighted price
    of neighbouring lots (within ``neighbor_radius_km``). A full lot never cuts
    its price; instead it suggests rerouting drivers to the nearby lot with space
    that is cheapest once a per-km travel penalty is added.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import PricingConfig

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def clip_price(price: float, cfg: PricingConfig) -> float:
    return clip(price, cfg.min_price, cfg.max_price)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """Normalised model inputs for one reading."""

    occupancy_rate: float  # [0, 1]
    queue: float           # [0, 1]
    traffic: float         # [0, 1]
    special: float         # {0, 1}
    vehicle: float         # [0, 1]
    trend: float           # [-1, 1]


def occupancy_rate(occupancy: float, capacity: float) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return clip(occupancy / capacity, 0.0, 1.0)


def encode_features(
    *,
    occupancy: float,
    capacity: float,
    queue_length: float,
    traffic: str,
    vehicle_type: str,
    is_special_day: int | bool,
    prev_occupancy_rate: float | None,
    cfg: PricingConfig,
) -> Features:
    occ = occupancy_rate(occupancy, capacity)
    if prev_occupancy_rate is None:
        trend = 0.0
    else:
        trend = clip((occ - prev_occupancy_rate) / cfg.trend_saturation, -1.0, 1.0)
    return Features(
        occupancy_rate=occ,
        queue=clip(queue_length / cfg.queue_saturation, 0.0, 1.0),
        # unknown labels fall back to the neutral level rather than failing
        traffic=float(cfg.traffic_levels.get(str(traffic).lower(), cfg.neutral_traffic)),
        special=1.0 if is_special_day else 0.0,
        vehicle=float(cfg.vehicle_weights.get(str(vehicle_type).lower(), cfg.neutral_vehicle)),
        trend=trend,
    )


# --------------------------------------------------------------------------
# Model 1: baseline linear
# --------------------------------------------------------------------------


def baseline_price(
    prev_price: float | None, occ_rate: float, cfg: PricingConfig
) -> float:
    start = cfg.base_price if prev_price is None else prev_price
    return clip_price(start + cfg.baseline_alpha * (occ_rate - cfg.target_occupancy), cfg)


# --------------------------------------------------------------------------
# Model 2: demand based
# --------------------------------------------------------------------------


def demand_score(f: Features, cfg: PricingConfig) -> float:
    return (
        cfg.w_occupancy * f.occupancy_rate
        + cfg.w_queue * f.queue
        - cfg.w_traffic * f.traffic
        + cfg.w_special * f.special
        + cfg.w_vehicle * f.vehicle
        + cfg.w_trend * f.trend
    )


def normalized_demand(f: Features, cfg: PricingConfig) -> float:
    lo, hi = cfg.demand_range
    return clip((demand_score(f, cfg) - lo) / (hi - lo), 0.0, 1.0)


def neutral_demand(cfg: PricingConfig) -> float:
    """Normalised demand under the configured 'typical' conditions."""
    neutral = Features(
        occupancy_rate=cfg.neutral_occupancy,
        queue=cfg.neutral_queue,
        traffic=cfg.neutral_traffic,
        special=0.0,
        vehicle=cfg.neutral_vehicle,
        trend=0.0,
    )
    return normalized_demand(neutral, cfg)


def demand_price(
    prev_price: float | None, f: Features, cfg: PricingConfig
) -> tuple[float, float]:
    """Return ``(price, normalised_demand)`` for Model 2."""
    nd = normalized_demand(f, cfg)
    raw = clip_price(
        cfg.base_price * (1.0 + cfg.demand_lambda * (nd - neutral_demand(cfg))), cfg
    )
    if prev_price is None:
        return raw, nd
    smoothed = cfg.smoothing * raw + (1.0 - cfg.smoothing) * prev_price
    return clip_price(smoothed, cfg), nd


# --------------------------------------------------------------------------
# Model 3: competitive
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Competitor:
    lot_id: str
    price: float
    occupancy_rate: float
    distance_km: float


@dataclass(frozen=True)
class CompetitiveDecision:
    price: float
    competitor_price: float | None
    action: str  # no_competitors | raise | lower | hold | reroute
    reroute_to: str | None = None


#: softens 1/distance weighting so co-located lots don't get infinite weight
_DISTANCE_SOFTENING_KM = 0.05


def competitor_reference_price(competitors: Sequence[Competitor]) -> float:
    weights = [1.0 / (c.distance_km + _DISTANCE_SOFTENING_KM) for c in competitors]
    return sum(w * c.price for w, c in zip(weights, competitors, strict=True)) / sum(weights)


def competitive_price(
    own_price: float,
    own_occupancy_rate: float,
    competitors: Iterable[Competitor],
    cfg: PricingConfig,
) -> CompetitiveDecision:
    competitors = list(competitors)
    if not competitors:
        return CompetitiveDecision(own_price, None, "no_competitors")

    comp = competitor_reference_price(competitors)
    is_full = own_occupancy_rate >= cfg.full_threshold

    reroute_to = None
    if is_full:
        with_space = [c for c in competitors if c.occupancy_rate < cfg.full_threshold]
        if with_space:
            best = min(
                with_space,
                key=lambda c: (c.price + cfg.reroute_cost_per_km * c.distance_km, c.lot_id),
            )
            reroute_to = best.lot_id

    if is_full and comp <= own_price:
        # demand exceeds supply: never undercut, send overflow to a neighbour
        price = own_price
    else:
        price = clip_price(own_price + cfg.competition_kappa * (comp - own_price), cfg)

    if reroute_to is not None:
        action = "reroute"
    elif price > own_price + 1e-9:
        action = "raise"
    elif price < own_price - 1e-9:
        action = "lower"
    else:
        action = "hold"
    return CompetitiveDecision(price, comp, action, reroute_to)


# --------------------------------------------------------------------------
# whole-day replay for one lot (used by the Pathway UDF)
# --------------------------------------------------------------------------

#: (timestamp, occupancy, capacity, queue_length, traffic, vehicle_type, is_special_day)
DayReading = tuple[str, int, int, int, str, str, int]


def price_lot_day(
    readings: Sequence[DayReading], cfg: PricingConfig
) -> list[tuple[float, float, float, float]]:
    """Run Models 1 and 2 over one lot's readings for one day.

    Readings are sorted by timestamp first, so the result is independent of
    arrival order. Returns ``(occupancy_rate, demand, price_baseline,
    price_demand)`` for every reading.
    """
    out = []
    prev_occ = prev_p1 = prev_p2 = None
    for _ts, occ, cap, queue, traffic, vehicle, special in sorted(readings):
        f = encode_features(
            occupancy=occ, capacity=cap, queue_length=queue, traffic=traffic,
            vehicle_type=vehicle, is_special_day=special,
            prev_occupancy_rate=prev_occ, cfg=cfg,
        )
        p1 = baseline_price(prev_p1, f.occupancy_rate, cfg)
        p2, nd = demand_price(prev_p2, f, cfg)
        out.append((f.occupancy_rate, nd, p1, p2))
        prev_occ, prev_p1, prev_p2 = f.occupancy_rate, p1, p2
    return out
