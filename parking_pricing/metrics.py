"""Evaluation metrics for comparing pricing models.

There is no "true price" in the data, so models are judged on behaviour:

* **responsiveness** - correlation between price and occupancy (higher is better)
* **stability** - mean absolute change between consecutive readings of a lot
* **saturation** - share of readings pinned at the min/max price bound
* **revenue index** - sum(price * occupancy) relative to a flat base price.
  This is a *static* proxy: it assumes demand does not react to price, so it
  measures how prices line up with busy periods, not a forecast of revenue.
"""

from __future__ import annotations

import pandas as pd

from .config import DEFAULT_CONFIG, PricingConfig

MODEL_COLUMNS = {
    "Model 1 - baseline": "price_baseline",
    "Model 2 - demand": "price_demand",
    "Model 3 - competitive": "price_competitive",
}


def model_summary(priced: pd.DataFrame, cfg: PricingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    base_revenue = (cfg.base_price * priced["occupancy"]).sum()
    rows = []
    for name, col in MODEL_COLUMNS.items():
        price = priced[col]
        step = priced.groupby(["lot_id", priced["timestamp"].dt.date])[col].diff().abs()
        at_bound = (price <= cfg.min_price + 1e-9) | (price >= cfg.max_price - 1e-9)
        rows.append(
            {
                "model": name,
                "mean_price": price.mean(),
                "std_price": price.std(),
                "min_price": price.min(),
                "max_price": price.max(),
                "corr_with_occupancy": price.corr(priced["occupancy_rate"]),
                "mean_abs_step": step.mean(),
                "pct_at_bounds": 100 * at_bound.mean(),
                "revenue_index": (price * priced["occupancy"]).sum() / base_revenue,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def lot_summary(priced: pd.DataFrame) -> pd.DataFrame:
    g = priced.groupby("lot_id")
    out = pd.DataFrame(
        {
            "mean_occupancy": g["occupancy_rate"].mean(),
            "pct_full": 100 * g["occupancy_rate"].apply(lambda s: (s >= 0.9).mean()),
            "m1_mean": g["price_baseline"].mean(),
            "m2_mean": g["price_demand"].mean(),
            "m3_mean": g["price_competitive"].mean(),
            "reroutes": g["reroute_to"].count(),
        }
    )
    return out.sort_values("mean_occupancy", ascending=False)


def action_breakdown(priced: pd.DataFrame) -> pd.Series:
    return priced["action"].value_counts(normalize=True).mul(100).round(1)
