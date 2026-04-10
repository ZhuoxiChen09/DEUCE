import pytest
from fastapi.testclient import TestClient

from deuce.api.app import create_app
from deuce.providers.mock import MockLiveScoreProvider


@pytest.fixture
def client():
    app = create_app(db_url="sqlite:///:memory:", live_provider=MockLiveScoreProvider())
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_simulation_reconcile(client: TestClient):
    r = client.post(
        "/simulation/reconcile",
        json={
            "rows": [
                {
                    "date": "20260323",
                    "tournament": "T",
                    "draw": "WTA",
                    "is_live": True,
                    "is_completed": False,
                    "is_scheduled": False,
                    "serve": "1",
                    "source": "test",
                    "players": [
                        {"name": "A", "score_tokens": ["0", "15"]},
                        {"name": "B", "score_tokens": ["0", "0"]},
                    ],
                }
            ]
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["states"]) == 1
    assert body["states"][0].get("status") == "live"


def test_ingest_and_list_matches():
    app = create_app(db_url="sqlite:///:memory:", live_provider=MockLiveScoreProvider())
    with TestClient(app) as client:
        r = client.post("/ingest", json={"dates": ["20260323"]})
        assert r.status_code == 200
        assert r.json()["processed"] >= 1
        r2 = client.get("/matches", params={"date": "20260323"})
        assert r2.status_code == 200
        assert len(r2.json()) >= 1
