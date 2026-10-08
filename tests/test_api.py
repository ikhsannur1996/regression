import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EXAMPLE

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())
BATCH = json.loads((ROOT / "sample_batch_request.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_model_info(client):
    body = client.get("/model-info").json()
    assert len(body["features"]) == 16
    assert 0.0 < body["trend"]["annual_growth"] < 0.2
    assert body["interval"]["q_low"] < 0 < body["interval"]["q_high"]


def test_valuation_interval_and_ltv(client):
    body = client.post("/valuation", json=SAMPLE).json()
    assert body["lower_90"] < body["estimated_value"] < body["upper_90"]
    assert body["max_loan"] == pytest.approx(body["max_ltv_policy"] * body["estimated_value"], rel=1e-3)
    assert body["conservative_max_loan"] < body["max_loan"]
    assert body["loan_check"]["loan_amount"] == SAMPLE["loan_amount"]


def test_example_payload_is_valid(client):
    assert client.post("/valuation", json=EXAMPLE).status_code == 200


def test_later_valuation_date_increases_value(client):
    early = client.post("/valuation", json={**EXAMPLE, "valuation_date": "2025-01-01"}).json()["estimated_value"]
    later = client.post("/valuation", json={**EXAMPLE, "valuation_date": "2026-01-01"}).json()["estimated_value"]
    assert later > early   # tren harga positif


def test_far_future_gives_warning(client):
    body = client.post("/valuation", json={**EXAMPLE, "valuation_date": "2027-06-01"}).json()
    assert any("refit" in w for w in body["warnings"])


def test_batch(client):
    body = client.post("/valuation/batch", json=BATCH).json()
    assert len(body["results"]) == len(BATCH["instances"])
    assert body["total_estimated_value"] == pytest.approx(sum(r["estimated_value"] for r in body["results"]), abs=0.5)


def test_nulls_accepted_with_warning(client):
    body = client.post("/valuation", json={**EXAMPLE, "building_age": None, "road_width_m": None}).json()
    assert any("missing" in w for w in body["warnings"])


@pytest.mark.parametrize("patch", [
    {"city": "medan"},                         # kota di luar cakupan model
    {"land_area": -10},
    {"building_area": 1000, "land_area": 100, "floors": 2},   # luas bangunan tidak masuk akal
    {"valuation_date": "2019-01-01"},
    {"unknown_field": 1},
])
def test_invalid_input_rejected(client, patch):
    assert client.post("/valuation", json={**EXAMPLE, **patch}).status_code == 422
