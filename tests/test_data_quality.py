from datetime import date

from deuce.ingestion.adapters import LocalCsvMatchAdapter, build_match_key
from deuce.ingestion.quality import evaluate_records, schema_issues

HEADER = (
    "player_a,player_b,winner,date,tournament,surface,round,"
    "rank_a,rank_b,points_a,points_b,best_of,tourney_level"
)
BOUNDS = dict(date_min=date(1990, 1, 1), date_max=date(2035, 12, 31))
GOOD = "Ada Player,Bea Player,A,20240101,Test Open,Hard,R32,10,50,5000,1500,3,250"


def _write(path, body: str, header: str = HEADER) -> None:
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")


def _checks(report):
    return {issue.check for issue in report.errors}


def test_schema_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "missing.csv"
    _write(path, GOOD, header="player_a,player_b,date,tournament,surface")
    loaded = LocalCsvMatchAdapter(path).load()
    issues = schema_issues(loaded.fieldnames)
    assert issues and issues[0].check == "schema"
    assert "winner" in issues[0].message


def test_null_duplicate_date_value_and_referential_checks(tmp_path):
    path = tmp_path / "bad.csv"
    _write(
        path,
        "\n".join(
            [
                GOOD,
                ",Bea Player,A,20240102,Test Open,Hard,R32,10,50,5000,1500,3,250",
                "Ada Player,Bea Player,A,18000101,Test Open,Hard,R32,10,50,5000,1500,3,250",
                "Ada Player,Bea Player,A,not-a-date,Test Open,Hard,R32,10,50,5000,1500,3,250",
                "Ada Player,Bea Player,A,20240103,Test Open,Ice,R32,10,50,5000,1500,9,250",
                "Ada Player,Ada Player,A,20240104,Test Open,Hard,R32,10,50,5000,1500,3,250",
                "Cara Player,Dee Player,A,2024-01-05,Test Open,Hard,R32,0,50,-5,1500,3,ZZZ",
                GOOD,
            ]
        ),
    )
    report = evaluate_records(LocalCsvMatchAdapter(path).load().records, **BOUNDS)
    assert "nulls" in _checks(report)
    assert "duplicates" in _checks(report)
    assert "date_range" in _checks(report)
    assert "value_range" in _checks(report)
    assert "referential" in _checks(report)
    assert len(report.valid) == 1
    assert any(issue.severity == "warning" and issue.field == "tourney_level" for issue in report.issues)


def test_iso_and_compact_dates_share_a_match_key():
    left = build_match_key("local_csv", "20240101", "Test Open", "R32", "Ada Player", "Bea Player")
    right = build_match_key("local_csv", "2024-01-01", "Test Open", "R32", "Ada Player", "Bea Player")
    assert left == right


def test_committed_demo_csv_passes_hard_checks():
    from deuce.config.settings import import_source_path

    loaded = LocalCsvMatchAdapter(import_source_path()).load()
    assert schema_issues(loaded.fieldnames) == []
    report = evaluate_records(loaded.records, **BOUNDS)
    assert report.ok
    assert report.rows_seen == 1232
    assert len(report.valid) == 1232
