"""Geographic helpers: distances and competitor neighbourhoods."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_neighbor_map(
    lots: pd.DataFrame, radius_km: float
) -> Mapping[str, list[tuple[str, float]]]:
    """Map each lot to ``[(neighbor_id, distance_km), ...]`` sorted by distance.

    ``lots`` is the frame returned by :func:`parking_pricing.data.lot_registry`.
    Lots without valid coordinates get no neighbours and are never anyone's
    neighbour.
    """
    neighbors: dict[str, list[tuple[str, float]]] = {lot: [] for lot in lots.index}
    valid = lots[lots["valid_coords"]]
    ids = valid.index.to_list()
    lat = valid["latitude"].to_numpy()
    lon = valid["longitude"].to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    for i, lot in enumerate(ids):
        pairs = [
            (ids[j], float(dist[i, j]))
            for j in range(len(ids))
            if j != i and dist[i, j] <= radius_km
        ]
        neighbors[lot] = sorted(pairs, key=lambda p: (p[1], p[0]))
    return neighbors


def neighbor_pairs_frame(neighbors: Mapping[str, list[tuple[str, float]]]) -> pd.DataFrame:
    rows = [
        {"lot_id": lot, "neighbor_id": n, "distance_km": d}
        for lot, ns in neighbors.items()
        for n, d in ns
    ]
    return pd.DataFrame(rows, columns=["lot_id", "neighbor_id", "distance_km"])
