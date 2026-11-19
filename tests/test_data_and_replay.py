import pandas as pd
import pytest

from parking_pricing import metrics
from parking_pricing.data import clean, load_raw, lot_registry
from parking_pricing.replay import run_replay


def test_cleaning_real_dataset():
    df, report = clean(load_raw())
    assert report.lots == 14 and report.days == 73
    assert report.duplicate_rows == 86
    assert report.invalid_coordinate_lots == ["BHMMBMMBX01"]
    assert not df.duplicated(["timestamp", "lot_id"]).any()
    assert df["timestamp"].is_monotonic_increasing
    assert set(df["traffic"]) == {"low", "average", "high"}


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="missing columns"):
        clean(pd.DataFrame({"SystemCodeNumber": ["x"]}))


def test_bad_rows_are_dropped():
    raw = load_raw().head(5).copy()
    raw.loc[0, "LastUpdatedDate"] = "not-a-date"
    raw.loc[1, "Capacity"] = 0
    df, report = clean(raw)
    assert report.dropped_rows == 2 and len(df) == 3


def test_full_replay(dataset, cfg):
    priced = run_replay(dataset, cfg)
    assert len(priced) == len(dataset)
    for col in metrics.MODEL_COLUMNS.values():
        assert priced[col].notna().all()
        assert priced[col].between(cfg.min_price, cfg.max_price).all()

    # lot with bogus coordinates never competes
    bad = priced[priced["lot_id"] == "BHMMBMMBX01"]
    assert (bad["action"] == "no_competitors").all()

    summary = metrics.model_summary(priced, cfg)
    # the demand model should track occupancy far better than the baseline
    assert summary.loc["Model 2 - demand", "corr_with_occupancy"] > 0.85
    assert (
        summary.loc["Model 2 - demand", "corr_with_occupancy"]
        > summary.loc["Model 1 - baseline", "corr_with_occupancy"]
    )
    # reroute targets are real neighbours with spare capacity at that instant
    reg = lot_registry(dataset)
    assert set(priced["reroute_to"].dropna()) <= set(reg.index[reg["valid_coords"]])
