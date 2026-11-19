"""Loading, cleaning and validating the raw parking dataset."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import haversine_km

log = logging.getLogger(__name__)

def _default_dataset() -> Path:
    """``$PARKING_DATASET``, else ``./dataset.csv``, else the repository checkout's copy."""
    env = os.environ.get("PARKING_DATASET")
    if env:
        return Path(env)
    for candidate in (Path.cwd() / "dataset.csv",
                      Path(__file__).resolve().parent.parent / "dataset.csv"):
        if candidate.exists():
            return candidate
    return Path("dataset.csv")


DEFAULT_DATASET = _default_dataset()

RAW_COLUMNS = {
    "SystemCodeNumber": "lot_id",
    "Capacity": "capacity",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Occupancy": "occupancy",
    "VehicleType": "vehicle_type",
    "TrafficConditionNearby": "traffic",
    "QueueLength": "queue_length",
    "IsSpecialDay": "is_special_day",
}

#: canonical column order of a cleaned readings frame
READING_COLUMNS = [
    "timestamp", "lot_id", "capacity", "occupancy", "queue_length",
    "traffic", "vehicle_type", "is_special_day", "latitude", "longitude",
]

#: lots further than this from the median lot location are treated as having
#: bad coordinates (the raw data contains one lot geocoded to the centre of India)
MAX_KM_FROM_CENTER = 50.0


@dataclass
class DataQualityReport:
    rows: int
    lots: int
    days: int
    start: pd.Timestamp
    end: pd.Timestamp
    over_capacity_rows: int
    duplicate_rows: int
    dropped_rows: int
    unknown_traffic: int
    unknown_vehicle: int
    invalid_coordinate_lots: list[str]

    def __str__(self) -> str:
        lines = [
            f"rows={self.rows:,}  lots={self.lots}  days={self.days}  "
            f"span={self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d}",
            f"occupancy > capacity: {self.over_capacity_rows} rows (rate clipped to 1.0)",
            f"duplicate (lot, timestamp) readings collapsed: {self.duplicate_rows}",
            f"dropped (unparseable/invalid): {self.dropped_rows} rows",
            f"unknown traffic labels: {self.unknown_traffic}, "
            f"unknown vehicle types: {self.unknown_vehicle}",
            f"lots with invalid coordinates (excluded from competition): "
            f"{self.invalid_coordinate_lots or 'none'}",
        ]
        return "\n".join(lines)


def load_raw(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Normalise the raw CSV into a tidy, time-ordered readings frame."""
    missing = set(RAW_COLUMNS) | {"LastUpdatedDate", "LastUpdatedTime"}
    missing -= set(raw.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    df = raw.rename(columns=RAW_COLUMNS).copy()
    df["timestamp"] = pd.to_datetime(
        df["LastUpdatedDate"].astype(str) + " " + df["LastUpdatedTime"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    df["lot_id"] = df["lot_id"].astype(str).str.strip()
    df["traffic"] = df["traffic"].astype(str).str.strip().str.lower()
    df["vehicle_type"] = df["vehicle_type"].astype(str).str.strip().str.lower()
    for col in ("capacity", "occupancy", "queue_length", "is_special_day"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["timestamp", "capacity", "occupancy", "queue_length"])
    df = df[(df["capacity"] > 0) & (df["occupancy"] >= 0) & (df["queue_length"] >= 0)]
    dropped = n0 - len(df)

    df = df.astype(
        {"capacity": int, "occupancy": int, "queue_length": int}
    )
    df["is_special_day"] = df["is_special_day"].fillna(0).astype(int).clip(0, 1)

    # A few sensors logged several events in the same minute. Occupancy is
    # identical across them, so keep the last record (file order) per minute.
    dup_mask = df.duplicated(["timestamp", "lot_id"], keep="last")
    duplicates = int(dup_mask.sum())
    df = df[~dup_mask]

    df = (
        df[READING_COLUMNS]
        .sort_values(["timestamp", "lot_id"], kind="stable")
        .reset_index(drop=True)
    )

    lots = lot_registry(df)
    report = DataQualityReport(
        rows=len(df),
        lots=df["lot_id"].nunique(),
        days=df["timestamp"].dt.normalize().nunique(),
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        over_capacity_rows=int((df["occupancy"] > df["capacity"]).sum()),
        duplicate_rows=duplicates,
        dropped_rows=int(dropped),
        unknown_traffic=int((~df["traffic"].isin(["low", "average", "high"])).sum()),
        unknown_vehicle=int(
            (~df["vehicle_type"].isin(["car", "bike", "truck", "cycle"])).sum()
        ),
        invalid_coordinate_lots=lots.index[~lots["valid_coords"]].tolist(),
    )
    return df, report


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    df, report = clean(load_raw(path))
    log.info("loaded dataset\n%s", report)
    return df


def lot_registry(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lot: capacity, location and whether the location is usable."""
    lots = df.groupby("lot_id").agg(
        capacity=("capacity", "max"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    )
    lat0, lon0 = lots["latitude"].median(), lots["longitude"].median()
    dist = haversine_km(lots["latitude"].to_numpy(), lots["longitude"].to_numpy(), lat0, lon0)
    lots["valid_coords"] = np.asarray(dist) <= MAX_KM_FROM_CENTER
    return lots


def to_stream_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write readings in the flat format consumed by the Pathway pipeline."""
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[READING_COLUMNS].to_csv(path, index=False)
    return path
