import sqlite3
from contextlib import closing

from src.data_preprocessing import build_database, named_queries, run_named_query, run_quality_checks


def test_rebuilt_database_passes_quality_gate_and_serves_every_query(tmp_path):
    # A rebuild (e.g. notebook 01, or a fresh clone where churn.db is not committed)
    # must restore the scores table, or the retention queries break.
    db = build_database(db_path=tmp_path / "churn.db")
    with closing(sqlite3.connect(db)) as con:
        assert (run_quality_checks(con)["violations"] == 0).all()
        for name in named_queries():
            assert len(run_named_query(name, con)) > 0, name


def test_retention_targets_only_lists_active_customers_by_expected_loss(tmp_path):
    db = build_database(db_path=tmp_path / "churn.db")
    with closing(sqlite3.connect(db)) as con:
        targets = run_named_query("retention_targets", con)
        churned = {r[0] for r in con.execute("SELECT customer_id FROM customers WHERE churn = 1")}
    assert not set(targets["customer_id"]) & churned
    assert targets["expected_loss"].is_monotonic_decreasing
