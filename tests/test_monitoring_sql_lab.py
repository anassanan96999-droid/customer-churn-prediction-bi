import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import load_features
from src.monitoring import categorical_psi, drift_report, numeric_psi, overall_status
from src.sql_lab import QueryError, run_query, schema, templates


# --------------------------------------------------------------------------- #
# Drift monitoring
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return load_features()


def test_same_population_is_stable(features):
    sample = features.sample(1500, random_state=0)
    report = drift_report(features, sample)
    assert overall_status(report) == "Stable"
    assert report["psi"].max() < 0.10


def test_shifted_population_is_flagged(features):
    new = features[features["contract"] == "Month-to-month"].copy()
    new["monthly_charges"] = new["monthly_charges"] * 1.4
    report = drift_report(features, new).set_index("feature")
    assert report.loc["contract", "status"] == "Shifted"
    assert report.loc["monthly_charges", "status"] in {"Watch", "Shifted"}
    assert overall_status(report.reset_index()) == "Shifted"


def test_psi_basics():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert numeric_psi(a, a) == pytest.approx(0, abs=1e-9)
    assert numeric_psi(a, a + 1.5) > 0.25
    assert categorical_psi(pd.Series(list("aabb")), pd.Series(list("aabb"))) == pytest.approx(0)


# --------------------------------------------------------------------------- #
# Read-only SQL
# --------------------------------------------------------------------------- #
def test_select_works_and_is_capped():
    df, truncated = run_query("SELECT customer_id FROM customers", max_rows=100)
    assert len(df) == 100 and truncated
    df, truncated = run_query("SELECT COUNT(*) AS n FROM customers;")
    assert df.iloc[0, 0] == 7043 and not truncated


@pytest.mark.parametrize("sql", [
    "DELETE FROM customers",
    "DROP TABLE customers",
    "UPDATE customers SET churn = 0",
    "INSERT INTO customers (customer_id) VALUES ('x')",
    "CREATE TABLE hack (x)",
    "ATTACH DATABASE 'other.db' AS other",
    "PRAGMA writable_schema = 1",
    "SELECT 1; DROP TABLE customers",
])
def test_anything_but_reading_is_rejected(sql):
    with pytest.raises(QueryError):
        run_query(sql)
    df, _ = run_query("SELECT COUNT(*) FROM customers")       # database untouched
    assert df.iloc[0, 0] == 7043


def test_schema_and_templates_are_usable_and_fast():
    import time
    tables = schema()
    assert {"customers", "customer_features", "customer_scores"} <= set(tables)
    assert "raw_customers" not in tables
    for name, sql in templates().items():
        t0 = time.perf_counter()
        df, _ = run_query(sql)
        # regression: without the customer_id indexes the score joins took > 5 s
        assert time.perf_counter() - t0 < 1.0, f"{name} is too slow"
        assert len(df) > 0, name
