from fastapi.testclient import TestClient

from deuce.api.app import create_app
from deuce.ingestion.adapters import LocalCsvMatchAdapter, build_match_key


def _client(**kwargs) -> TestClient:
    app = create_app(db_url="sqlite:///:memory:", **kwargs)
    return TestClient(app)


def test_health_reports_model_and_scrapers_off():
    with _client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["live_scores_enabled"] is False
    assert body["model_loaded"] is True


def test_ingest_does_not_scrape_by_default():
    with _client() as client:
        response = client.post("/ingest", json={"dates": ["20260323"]})
    assert response.status_code == 200
    body = response.json()
    assert body["provider_id"] == "disabled"
    assert body["fetched"] == 0
    assert body["processed"] == 0
    assert any("disabled" in error for error in body["errors"])


def test_live_demo_uses_calibrated_model():
    with _client() as client:
        response = client.get("/predictions/live/demo")
        again = client.get("/predictions/live/demo", params={"surface": "Clay"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["player_a"] == "Carlos Alcaraz"
    assert body["player_b"] == "Jannik Sinner"
    assert body["surface"] == "Hard"
    assert body["model_used"] == "core"
    assert body["confidence"] == "model"
    assert body["calibrated_prob_a"] == body["model_prob_a"]
    assert 0.0 < body["model_prob_a"] < 1.0
    assert abs(body["model_prob_a"] + body["model_prob_b"] - 1.0) < 1e-6
    assert again.status_code == 200
    assert again.json()["available"] is True
    assert again.json()["surface"] == "Clay"


def test_matchup_input_is_validated():
    with _client() as client:
        same = client.get("/predictions/live/demo", params={"player_a": "Ada Player", "player_b": "ada player"})
        bad_surface = client.get("/predictions/live/demo", params={"surface": "Ice"})
        missing = client.get("/predictions/historical/does-not-exist")
    assert same.status_code == 422
    assert bad_surface.status_code == 422
    assert missing.status_code == 404


def test_import_endpoint_then_historical_prediction(tmp_path):
    path = tmp_path / "matches.csv"
    path.write_text(
        "player_a,player_b,winner,date,tournament,surface,round,rank_a,rank_b,points_a,points_b,best_of,tourney_level\n"
        "Ada Player,Bea Player,A,20240101,Test Open,Hard,R32,1,100,10000,500,3,250\n",
        encoding="utf-8",
    )
    adapter = LocalCsvMatchAdapter(path)
    with _client(import_adapter=adapter) as client:
        imported = client.post("/imports")
        assert imported.status_code == 200
        body = imported.json()
        assert body["status"] == "loaded"
        assert body["rows_loaded"] == 1
        key = build_match_key("local_csv", "20240101", "Test Open", "R32", "Ada Player", "Bea Player")
        predicted = client.get(f"/predictions/historical/{key}")
    assert predicted.status_code == 200
    payload = predicted.json()
    assert payload["available"] is True
    assert payload["model_used"] == "core"
    assert payload["confidence"] == "model"
    assert payload["model_prob_a"] > 0.5
    assert payload["player_a"] == "Ada Player"


def test_live_scores_flag_selects_provider_without_fetching(monkeypatch):
    from deuce.services.prediction_service import LivePredictionService

    monkeypatch.setenv("DEUCE_LIVE_SCORES_ENABLED", "1")
    monkeypatch.setenv("DEUCE_LIVE_SCORE_PROVIDER", "espn")
    app = create_app(
        db_url="sqlite:///:memory:",
        prediction_service=LivePredictionService(model_core=object()),
    )
    assert app.state.live_scores_enabled is True
    assert app.state.live_provider.provider_id == "live_scores:espn"
    app.state.engine.dispose()
