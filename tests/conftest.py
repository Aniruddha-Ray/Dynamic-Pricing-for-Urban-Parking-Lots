from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from parking_pricing.config import DEFAULT_CONFIG
from parking_pricing.data import clean, load_raw
from parking_pricing.engine import Reading


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    df, _ = clean(load_raw())
    return df


@pytest.fixture
def lots() -> pd.DataFrame:
    """Three lots: A and B ~0.3 km apart, C far away (no neighbours)."""
    return pd.DataFrame(
        {
            "capacity": [100, 100, 100],
            "latitude": [26.1400, 26.1427, 26.5000],
            "longitude": [91.7300, 91.7300, 91.7300],
            "valid_coords": [True, True, True],
        },
        index=pd.Index(["A", "B", "C"], name="lot_id"),
    )


def reading(lot: str, ts: str, occ: int, **kw) -> Reading:
    return Reading(
        lot_id=lot,
        timestamp=datetime.fromisoformat(ts),
        occupancy=occ,
        capacity=kw.pop("capacity", 100),
        **kw,
    )
