import pytest
from conftest import reading

from parking_pricing import models
from parking_pricing.engine import PricingEngine


def test_neighbours_respect_radius(lots, cfg):
    eng = PricingEngine(lots, cfg)
    assert [n for n, _ in eng.neighbors["A"]] == ["B"]
    assert eng.neighbors["C"] == []


def test_same_timestamp_is_order_independent(lots, cfg):
    batch = [reading("A", "2016-10-04 08:00", 95), reading("B", "2016-10-04 08:00", 30)]
    q1, _ = PricingEngine(lots, cfg).process(batch)
    q2, _ = PricingEngine(lots, cfg).process(batch[::-1])
    assert sorted(q1, key=lambda q: q.lot_id) == sorted(q2, key=lambda q: q.lot_id)
    a = next(q for q in q1 if q.lot_id == "A")
    assert a.competitor_price is not None  # sees B even though B "arrived" second
    assert a.reroute_to == "B"


def test_engine_matches_whole_day_replay(lots, cfg):
    day = [
        ("2016-10-04 08:00:00", 30, 100, 1, "low", "car", 0),
        ("2016-10-04 08:30:00", 70, 100, 5, "average", "bike", 0),
        ("2016-10-04 09:00:00", 97, 100, 9, "high", "truck", 1),
    ]
    eng = PricingEngine(lots, cfg)
    quotes = []
    for ts, occ, cap, q, tr, veh, sp in day:
        out, _ = eng.process([reading("C", ts, occ, capacity=cap, queue_length=q,
                                      traffic=tr, vehicle_type=veh, is_special_day=sp)])
        quotes += out
    expected = models.price_lot_day(day, cfg)
    for quote, (occ, nd, p1, p2) in zip(quotes, expected, strict=True):
        assert (quote.occupancy_rate, quote.demand, quote.price_baseline, quote.price_demand) == (
            pytest.approx(occ), pytest.approx(nd), pytest.approx(p1), pytest.approx(p2))


def test_state_resets_each_day(lots, cfg):
    eng = PricingEngine(lots, cfg)
    for h in range(8, 17):
        eng.process([reading("C", f"2016-10-04 {h:02d}:00", 100)])
    assert eng.latest()["C"].price_baseline > cfg.base_price + 1
    q, _ = eng.process([reading("C", "2016-10-05 08:00", 60)])
    assert q[0].price_baseline == pytest.approx(cfg.base_price)


def test_neighbours_from_previous_day_are_ignored(lots, cfg):
    eng = PricingEngine(lots, cfg)
    eng.process([reading("B", "2016-10-04 16:00", 50)])
    q, _ = eng.process([reading("A", "2016-10-05 08:00", 50)])
    assert q[0].action == "no_competitors"


def test_stale_duplicate_and_invalid_readings_rejected(lots, cfg):
    eng = PricingEngine(lots, cfg)
    eng.process([reading("A", "2016-10-04 09:00", 50)])
    quotes, rejected = eng.process([
        reading("A", "2016-10-04 08:00", 50),                  # older than state
        reading("B", "2016-10-04 09:00", 50),
        reading("B", "2016-10-04 09:00", 60),                  # duplicate
        reading("C", "2016-10-04 09:00", 50, capacity=0),      # invalid
    ])
    assert [q.lot_id for q in quotes] == ["B"]
    reasons = sorted(r.reason.split(":")[0] for r in rejected)
    assert reasons == ["duplicate reading for this timestamp",
                       "invalid capacity/occupancy/queue", "stale"]


def test_unknown_lot_is_priced_without_competitors(lots, cfg):
    q, _ = PricingEngine(lots, cfg).process([reading("NEW", "2016-10-04 09:00", 50)])
    assert q[0].action == "no_competitors"
    assert q[0].price_competitive == q[0].price_demand
