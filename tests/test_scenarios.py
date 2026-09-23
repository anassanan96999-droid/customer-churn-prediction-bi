import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import load_features
from src.predict import load_bundle
from src.scenarios import (INTERVENTIONS, apply_interventions, next_best_actions,
                           recompute_derived, simulate)

DERIVED = ["service_count", "streaming_count", "has_protection_bundle", "family_account",
           "is_month_to_month", "is_electronic_check", "is_auto_payment", "is_new_customer",
           "avg_monthly_spend", "spend_trend_ratio", "charge_per_service", "revenue_at_risk_annual"]


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return load_features()


@pytest.fixture(scope="module")
def pipeline():
    return load_bundle()["pipeline"]


def test_python_mirror_matches_the_sql_view(features):
    # Scenario Lab edits profiles in Python; it must derive features exactly like SQL does.
    mirror = recompute_derived(features)
    for col in DERIVED:
        assert np.allclose(mirror[col].astype(float), features[col].astype(float), atol=0.011), col


def test_interventions_only_touch_eligible_customers(features):
    changed, eligible, cost = apply_interventions(features, ["annual_contract"], {"annual_contract": 1})
    m2m = features["contract"] == "Month-to-month"
    assert (eligible == m2m).all()
    assert (changed.loc[m2m, "contract"] == "One year").all()
    assert (changed.loc[~m2m, "contract"] == features.loc[~m2m, "contract"]).all()
    assert np.allclose(cost[m2m], features.loc[m2m, "monthly_charges"])     # one free month
    assert (cost[~m2m] == 0).all()


def test_bundle_updates_derived_features(features):
    changed, eligible, _ = apply_interventions(features, ["protection_bundle"])
    assert (changed.loc[eligible, "has_protection_bundle"] == 1).all()
    assert (changed.loc[eligible, "service_count"] >= features.loc[eligible, "service_count"]).all()


def test_discount_cost_is_revenue_given_away(features):
    sample = features.head(50)
    _, eligible, cost = apply_interventions(sample, ["price_discount"], {"discount": 0.2, "horizon": 12})
    assert np.allclose(cost[eligible], sample.loc[eligible, "monthly_charges"] * 0.2 * 12)


def test_annual_contract_lowers_modelled_risk(features, pipeline):
    m2m = features[(features["contract"] == "Month-to-month") & (features["churn"] == 0)].head(300)
    summary, detail = simulate(m2m, pipeline, ["annual_contract"], acceptance=0.5)
    assert summary["eligible"] == len(m2m)
    assert summary["avg_risk_after"] < summary["avg_risk_before"]
    assert summary["churners_avoided"] == pytest.approx(0.5 * detail["risk_drop"].sum())
    assert summary["net_value"] == pytest.approx(summary["margin_retained"] - summary["offer_cost"])


def test_zero_acceptance_changes_nothing(features, pipeline):
    summary, _ = simulate(features.head(100), pipeline, ["auto_pay"], acceptance=0.0)
    assert summary["churners_avoided"] == 0 and summary["offer_cost"] == 0


def test_next_best_actions_ranked_and_eligible(features, pipeline):
    row = features[(features["contract"] == "Month-to-month")
                   & (features["payment_method"] == "Electronic check")].iloc[0]
    actions = next_best_actions(row, pipeline)
    assert list(actions["risk_drop"]) == sorted(actions["risk_drop"], reverse=True)
    assert "Upgrade to a 1-year contract" in set(actions["action"])
    assert set(actions["key"]) <= set(INTERVENTIONS)
    assert (actions["p_after"].between(0, 1)).all()
    assert config.ID_COL in features.columns
