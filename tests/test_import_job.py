import logging
from pathlib import Path

import pytest
from sqlalchemy import func, select

from deuce.ingestion.adapters import LocalCsvMatchAdapter
from deuce.ingestion.job import run_import
from deuce.storage.database import init_db, make_engine, make_session_factory
from deuce.storage.schemas import HistoricalMatchRow, ImportRunRow

HEADER = (
    "player_a,player_b,winner,date,tournament,surface,round,"
    "rank_a,rank_b,points_a,points_b,best_of,tourney_level"
)
GOOD = "Ada Player,Bea Player,A,20240101,Test Open,Hard,R32,10,50,5000,1500,3,250"
BAD_NULL = ",Bea Player,A,20240102,Test Open,Hard,R32,10,50,5000,1500,3,250"


def _factory():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


def _write(path: Path, body: str, header: str = HEADER) -> LocalCsvMatchAdapter:
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")
    return LocalCsvMatchAdapter(path)


def test_block_mode_writes_nothing_and_logs(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="deuce.ingestion")
    adapter = _write(tmp_path / "mixed.csv", f"{GOOD}\n{BAD_NULL}")
    factory = _factory()
    outcome = run_import(session_factory=factory, adapter=adapter, mode="block")
    assert outcome.status == "blocked"
    assert outcome.rows_loaded == 0
    assert outcome.rows_rejected == 2
    assert "import blocked" in caplog.text
    assert "dq check=nulls" in caplog.text
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(HistoricalMatchRow)) == 0
        run = session.scalar(select(ImportRunRow))
        assert run is not None
        assert run.status == "blocked"


def test_flag_mode_loads_valid_rows_and_logs(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="deuce.ingestion")
    adapter = _write(tmp_path / "mixed.csv", f"{GOOD}\n{BAD_NULL}")
    factory = _factory()
    outcome = run_import(session_factory=factory, adapter=adapter, mode="flag")
    assert outcome.status == "flagged"
    assert outcome.rows_loaded == 1
    assert outcome.rows_rejected == 1
    assert "import flagged" in caplog.text
    with factory() as session:
        row = session.scalar(select(HistoricalMatchRow))
        assert row is not None
        assert row.player_a == "Ada Player"
        assert row.match_date == "20240101"
        assert row.rank_a == 10


def test_schema_failure_blocks_even_in_flag_mode(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="deuce.ingestion")
    adapter = _write(tmp_path / "schema.csv", GOOD, header="player_a,player_b,date,tournament,surface")
    outcome = run_import(session_factory=_factory(), adapter=adapter, mode="flag")
    assert outcome.status == "blocked"
    assert outcome.rows_loaded == 0
    assert outcome.issues[0].check == "schema"
    assert "dq check=schema" in caplog.text


def test_warning_flags_without_rejecting_the_row(tmp_path):
    adapter = _write(
        tmp_path / "warn.csv",
        "Ada Player,Bea Player,A,20240101,Test Open,Hard,R32,10,50,5000,1500,3,ZZZ",
    )
    outcome = run_import(session_factory=_factory(), adapter=adapter, mode="block")
    assert outcome.status == "flagged"
    assert outcome.rows_loaded == 1
    assert any(issue.severity == "warning" for issue in outcome.issues)


def test_reimport_upserts_one_match(tmp_path):
    adapter = _write(tmp_path / "one.csv", GOOD)
    factory = _factory()
    first = run_import(session_factory=factory, adapter=adapter, mode="block")
    second = run_import(session_factory=factory, adapter=adapter, mode="block")
    assert first.status == "loaded"
    assert second.status == "loaded"
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(HistoricalMatchRow)) == 1
        assert session.scalar(select(func.count()).select_from(ImportRunRow)) == 2


def test_missing_source_is_logged(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="deuce.ingestion")
    adapter = LocalCsvMatchAdapter(tmp_path / "absent.csv")
    outcome = run_import(session_factory=_factory(), adapter=adapter, mode="block")
    assert outcome.status == "error"
    assert outcome.rows_loaded == 0
    assert "import error" in caplog.text


def test_alembic_upgrade_creates_product_tables(tmp_path, monkeypatch):
    pytest.importorskip("alembic")
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from deuce.storage.database import make_engine

    db_path = tmp_path / "deuce.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DEUCE_DB_URL", url)
    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    names = set(inspect(make_engine(url)).get_table_names())
    assert {"matches", "events", "predictions", "import_runs", "historical_matches"} <= names


def test_scheduler_is_off_unless_enabled(monkeypatch):
    from deuce.api.app import create_app

    monkeypatch.delenv("DEUCE_IMPORT_ENABLED", raising=False)
    app = create_app(db_url="sqlite:///:memory:", prediction_service=_quiet_service())
    assert app.state.scheduler is None

    monkeypatch.setenv("DEUCE_IMPORT_ENABLED", "1")
    monkeypatch.setenv("DEUCE_IMPORT_ON_START", "0")
    monkeypatch.setenv("DEUCE_IMPORT_INTERVAL_MINUTES", "15")
    scheduled = create_app(
        db_url="sqlite:///:memory:",
        prediction_service=_quiet_service(),
        start_scheduler=True,
    )
    job = scheduled.state.scheduler.get_job("deuce-import")
    assert job is not None
    assert scheduled.state.scheduler.running is False


def _quiet_service():
    from deuce.services.prediction_service import LivePredictionService

    return LivePredictionService(model_core=object())
