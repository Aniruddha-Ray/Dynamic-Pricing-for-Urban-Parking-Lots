"""Parity test: the Pathway pipeline must agree with the in-process engine.

Skipped automatically where Pathway is unavailable (e.g. Windows); run it with
``docker run --rm --entrypoint pytest parking-pricing``.
"""

import pandas as pd
import pytest

from parking_pricing.pathway_pipeline import (
    build_pipeline,
    prepare_inputs,
    read_stream,
    require_pathway,
)
from parking_pricing.replay import run_replay

try:
    pw = require_pathway()
except RuntimeError:
    pytest.skip("Pathway not available on this platform", allow_module_level=True)


@pytest.fixture(scope="module")
def three_days(dataset):
    days = sorted(dataset["timestamp"].dt.date.unique())[:3]
    return dataset[dataset["timestamp"].dt.date.isin(days)].reset_index(drop=True)


@pytest.fixture(scope="module")
def pathway_result(three_days, tmp_path_factory):
    from parking_pricing.config import DEFAULT_CONFIG

    stream_csv, pairs = prepare_inputs(three_days, tmp_path_factory.mktemp("pw"))
    tables = build_pipeline(read_stream(stream_csv, mode="static"), pairs, DEFAULT_CONFIG)
    live, daily = pw.debug.table_to_pandas(tables.live_prices), pw.debug.table_to_pandas(
        tables.daily_summary
    )
    return live, daily


def _engine_end_of_day(df):
    priced = run_replay(df)
    priced["day"] = priced["timestamp"].dt.strftime("%Y-%m-%d")
    return priced.sort_values("timestamp").groupby(["lot_id", "day"]).tail(1)


def test_one_row_per_lot_day(pathway_result, three_days):
    live, _ = pathway_result
    expected = three_days.groupby(["lot_id", three_days["timestamp"].dt.date]).ngroups
    assert len(live) == expected


def test_models_1_and_2_match_engine(pathway_result, three_days):
    live, _ = pathway_result
    live = live.assign(day=live["timestamp"].str.slice(0, 10))
    eng = _engine_end_of_day(three_days)
    merged = eng.merge(live, on=["lot_id", "day"], suffixes=("_eng", "_pw"))
    assert len(merged) == len(live)
    for col in ("occupancy_rate", "demand", "price_baseline", "price_demand"):
        pd.testing.assert_series_equal(
            merged[f"{col}_eng"], merged[f"{col}_pw"], check_names=False, rtol=1e-9
        )


def test_model_3_matches_engine_when_lots_report_together(pathway_result, three_days):
    """At end of day the pipeline sees every neighbour's final price; the engine
    sees the same thing whenever all lots' last readings share a timestamp."""
    live, _ = pathway_result
    live = live.assign(day=live["timestamp"].str.slice(0, 10))
    eng = _engine_end_of_day(three_days)
    last_ts = eng.groupby("day")["timestamp"].nunique()
    aligned_days = last_ts[last_ts == 1].index
    assert len(aligned_days) > 0
    merged = eng[eng["day"].isin(aligned_days)].merge(
        live, on=["lot_id", "day"], suffixes=("_eng", "_pw")
    )
    pd.testing.assert_series_equal(
        merged["price_competitive_eng"], merged["price_competitive_pw"],
        check_names=False, rtol=1e-9,
    )
    assert (merged["action_eng"] == merged["action_pw"]).all()
    assert (merged["reroute_to_eng"].fillna("") == merged["reroute_to_pw"]).all()


def test_daily_window_kpis(pathway_result, three_days):
    _, daily = pathway_result
    assert len(daily) == three_days.groupby(
        ["lot_id", three_days["timestamp"].dt.date]
    ).ngroups
    assert daily["occupancy_peak"].between(0, 1).all()
    assert (daily["occupancy_mean"] <= daily["occupancy_peak"] + 1e-12).all()
