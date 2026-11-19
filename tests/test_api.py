import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by the test client
from fastapi.testclient import TestClient  # noqa: E402

from parking_pricing.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(warm_start=True)) as c:
        yield c


def test_health_and_lots(client):
    assert client.get("/health").json()["lots_priced"] == 14
    lots = {lot["lot_id"]: lot for lot in client.get("/lots").json()}
    assert len(lots) == 14
    assert lots["BHMMBMMBX01"]["valid_coords"] is False
    assert lots["BHMMBMMBX01"]["neighbors"] == []


def test_latest_prices_and_history(client):
    prices = client.get("/prices").json()
    assert len(prices) == 14
    one = client.get("/prices/Shopping").json()
    assert one["lot_id"] == "Shopping"
    assert client.get("/prices/nope").status_code == 404

    hist = client.get("/history/Shopping", params={"day": "2016-10-04"}).json()
    assert 15 <= len(hist) <= 18
    assert client.get("/history/Shopping", params={"day": "bad"}).status_code == 422


def test_ingest_live_readings(client):
    body = [
        {"lot_id": "BHMBCCTHL01", "timestamp": "2016-12-20T08:00:00",
         "occupancy": 380, "capacity": 387, "queue_length": 8, "traffic": "high"},
        {"lot_id": "BHMBCCMKT01", "timestamp": "2016-12-20T08:00:00",
         "occupancy": 100, "capacity": 577, "queue_length": 1, "traffic": "low"},
    ]
    res = client.post("/readings", json=body).json()
    assert res["rejected"] == []
    thl = next(q for q in res["quotes"] if q["lot_id"] == "BHMBCCTHL01")
    assert thl["action"] == "reroute" and thl["reroute_to"] == "BHMBCCMKT01"
    assert client.get("/prices/BHMBCCTHL01").json()["timestamp"].startswith("2016-12-20T08:00")

    # replaying the same reading is stale
    again = client.post("/readings", json=body[:1]).json()
    assert again["quotes"] == [] and "stale" in again["rejected"][0]["reason"]


def test_validation_errors(client):
    bad = [{"lot_id": "X", "timestamp": "2016-12-20T08:00:00", "occupancy": 5, "capacity": 0}]
    assert client.post("/readings", json=bad).status_code == 422
