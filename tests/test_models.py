import math

import pytest

from parking_pricing import models
from parking_pricing.config import PricingConfig
from parking_pricing.models import Competitor, Features


def feats(cfg, occ=0.6, queue=3, traffic="average", vehicle="car", special=0, prev=None):
    return models.encode_features(
        occupancy=occ * 100, capacity=100, queue_length=queue, traffic=traffic,
        vehicle_type=vehicle, is_special_day=special, prev_occupancy_rate=prev, cfg=cfg,
    )


# ----------------------------------------------------------------- features
def test_occupancy_rate_is_clipped():
    assert models.occupancy_rate(120, 100) == 1.0
    assert models.occupancy_rate(50, 100) == 0.5
    with pytest.raises(ValueError):
        models.occupancy_rate(1, 0)


def test_feature_encoding(cfg):
    f = feats(cfg, occ=0.5, queue=20, traffic="HIGH", vehicle="truck", special=1, prev=0.25)
    assert f.queue == 1.0                     # saturates
    assert f.traffic == 1.0                   # case-insensitive
    assert f.vehicle == 1.0
    assert f.special == 1.0
    assert f.trend == pytest.approx(1.0)      # +0.25 occupancy == trend saturation


def test_unknown_labels_fall_back_to_neutral(cfg):
    f = feats(cfg, traffic="gridlock", vehicle="spaceship")
    assert f.traffic == cfg.neutral_traffic
    assert f.vehicle == cfg.neutral_vehicle


# ------------------------------------------------------------------ Model 1
def test_baseline_moves_with_occupancy_vs_target(cfg):
    assert models.baseline_price(None, cfg.target_occupancy, cfg) == cfg.base_price
    assert models.baseline_price(None, 1.0, cfg) > cfg.base_price
    assert models.baseline_price(None, 0.0, cfg) < cfg.base_price


def test_baseline_is_bounded(cfg):
    p = None
    for _ in range(200):
        p = models.baseline_price(p, 1.0, cfg)
    assert p == cfg.max_price
    for _ in range(400):
        p = models.baseline_price(p, 0.0, cfg)
    assert p == cfg.min_price


# ------------------------------------------------------------------ Model 2
def test_neutral_conditions_price_at_base(cfg):
    f = Features(cfg.neutral_occupancy, cfg.neutral_queue, cfg.neutral_traffic,
                 0.0, cfg.neutral_vehicle, 0.0)
    price, _ = models.demand_price(None, f, cfg)
    assert price == pytest.approx(cfg.base_price)


@pytest.mark.parametrize(
    "low, high",
    [
        (dict(occ=0.2), dict(occ=0.9)),
        (dict(queue=0), dict(queue=8)),
        (dict(traffic="high"), dict(traffic="low")),   # congestion lowers price
        (dict(special=0), dict(special=1)),
        (dict(vehicle="cycle"), dict(vehicle="truck")),
        (dict(prev=0.9), dict(prev=0.3)),             # rising occupancy -> higher
    ],
)
def test_demand_price_is_monotone_in_each_driver(cfg, low, high):
    p_low, _ = models.demand_price(None, feats(cfg, **low), cfg)
    p_high, _ = models.demand_price(None, feats(cfg, **high), cfg)
    assert p_high > p_low


def test_demand_is_normalised_and_bounded(cfg):
    extreme_hi = Features(1, 1, 0, 1, 1, 1)
    extreme_lo = Features(0, 0, 1, 0, 0, -1)
    assert models.normalized_demand(extreme_hi, cfg) == pytest.approx(1.0)
    assert models.normalized_demand(extreme_lo, cfg) == pytest.approx(0.0)
    for f in (extreme_hi, extreme_lo):
        p, _ = models.demand_price(None, f, cfg)
        assert cfg.min_price <= p <= cfg.max_price


def test_smoothing_dampens_jumps(cfg):
    f = feats(cfg, occ=1.0, queue=10)
    raw, _ = models.demand_price(None, f, cfg)
    smoothed, _ = models.demand_price(cfg.base_price, f, cfg)
    assert cfg.base_price < smoothed < raw
    no_smooth = cfg.with_overrides(smoothing=1.0)
    assert models.demand_price(cfg.base_price, f, no_smooth)[0] == pytest.approx(raw)


# ------------------------------------------------------------------ Model 3
def comp(lot, price, occ=0.5, dist=0.3):
    return Competitor(lot, price, occ, dist)


def test_no_competitors_keeps_price(cfg):
    d = models.competitive_price(10.0, 0.5, [], cfg)
    assert (d.price, d.action, d.competitor_price) == (10.0, "no_competitors", None)


def test_pricier_neighbours_allow_raise_but_stay_cheaper(cfg):
    d = models.competitive_price(10.0, 0.5, [comp("B", 14.0)], cfg)
    assert d.action == "raise"
    assert 10.0 < d.price < 14.0
    assert d.price == pytest.approx(10.0 + cfg.competition_kappa * 4.0)


def test_cheaper_neighbours_force_lower_when_not_full(cfg):
    d = models.competitive_price(12.0, 0.5, [comp("B", 9.0)], cfg)
    assert d.action == "lower" and 9.0 < d.price < 12.0


def test_full_lot_never_undercuts_and_reroutes(cfg):
    comps = [comp("B", 9.0, occ=0.4, dist=0.5), comp("C", 8.0, occ=0.95), comp("D", 9.0, occ=0.2, dist=0.2)]
    d = models.competitive_price(12.0, 0.95, comps, cfg)
    assert d.price == 12.0
    assert d.action == "reroute"
    # C is cheapest but full; B and D tie on price, D is nearer
    assert d.reroute_to == "D"


def test_reroute_trades_price_against_distance(cfg):
    far_cheap = comp("FAR", 8.0, occ=0.2, dist=0.9)    # 8.0 + 2.0*0.9 = 9.8
    near = comp("NEAR", 9.0, occ=0.2, dist=0.01)       # 9.0 + 0.02   = 9.02
    assert models.competitive_price(12.0, 0.95, [far_cheap, near], cfg).reroute_to == "NEAR"
    no_penalty = cfg.with_overrides(reroute_cost_per_km=0.0)
    assert models.competitive_price(12.0, 0.95, [far_cheap, near], no_penalty).reroute_to == "FAR"


def test_full_lot_with_all_neighbours_full_does_not_reroute(cfg):
    d = models.competitive_price(12.0, 0.95, [comp("B", 9.0, occ=0.99)], cfg)
    assert d.reroute_to is None and d.action == "hold"


def test_full_lot_may_raise_towards_pricier_neighbours(cfg):
    d = models.competitive_price(10.0, 0.95, [comp("B", 15.0, occ=0.99)], cfg)
    assert d.action == "raise" and d.price > 10.0


def test_closer_competitors_weigh_more():
    near_cheap = [comp("B", 8.0, dist=0.05), comp("C", 16.0, dist=0.9)]
    assert models.competitor_reference_price(near_cheap) < 10.0


# ------------------------------------------------------------- day replay
def test_price_lot_day_is_order_independent(cfg):
    day = [
        ("2016-10-04 08:00:00", 30, 100, 1, "low", "car", 0),
        ("2016-10-04 08:30:00", 60, 100, 4, "average", "bike", 0),
        ("2016-10-04 09:00:00", 95, 100, 9, "high", "truck", 0),
    ]
    assert models.price_lot_day(day, cfg) == models.price_lot_day(day[::-1], cfg)
    assert all(not math.isnan(v) for row in models.price_lot_day(day, cfg) for v in row)


# ------------------------------------------------------------------ config
@pytest.mark.parametrize(
    "override",
    [dict(base_price=0), dict(min_multiplier=1.5), dict(smoothing=0),
     dict(competition_kappa=2), dict(w_queue=-1), dict(reroute_cost_per_km=-1)],
)
def test_invalid_config_rejected(cfg, override):
    with pytest.raises(ValueError):
        cfg.with_overrides(**override)


def test_config_round_trip(cfg, tmp_path):
    path = tmp_path / "cfg.json"
    import json
    path.write_text(json.dumps(cfg.with_overrides(base_price=12.5).to_dict()))
    assert PricingConfig.from_json(path).base_price == 12.5
    with pytest.raises(ValueError, match="unknown"):
        PricingConfig.from_dict({"nope": 1})
