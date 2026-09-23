import numpy as np
import pytest

from src.config import Economics
from src.economics import (campaign_outcome, expected_net_gain, expected_value_target,
                           optimal_threshold, threshold_curve)

ECON = Economics(offer_cost=20, save_rate=0.5, horizon_months=10, gross_margin=0.5)
# customer value = monthly * 10 * 0.5 = 5 x monthly


def test_campaign_outcome_counts_money_correctly():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    monthly = np.array([100.0, 100.0, 100.0, 100.0])
    out = campaign_outcome(y, p, monthly, threshold=0.5, econ=ECON)
    # contacts customers 0 and 1; only customer 0 churns -> saved 0.5 * 500 = 250
    assert out["contacted"] == 2
    assert out["churners_caught"] == 1
    assert out["retained_value"] == pytest.approx(250.0)
    assert out["campaign_cost"] == pytest.approx(40.0)
    assert out["net_value"] == pytest.approx(210.0)
    assert out["recall"] == pytest.approx(0.5)


def test_contacting_nobody_is_worth_zero():
    y = np.array([1, 0])
    out = campaign_outcome(y, [0.3, 0.2], [50, 50], target=[False, False], econ=ECON)
    assert out["net_value"] == 0


def test_optimal_threshold_beats_or_matches_default():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)  # perfectly calibrated scores
    monthly = rng.uniform(20, 110, size=2000)
    t = optimal_threshold(y, p, monthly, ECON)
    curve = threshold_curve(y, p, monthly, ECON)
    assert curve["net_value"].max() >= curve.loc[np.isclose(curve["threshold"], 0.5), "net_value"].iloc[0]
    # with cheap offers and valuable customers the optimum sits below 0.5
    assert t < 0.5


def test_expected_value_rule_prefers_valuable_customers():
    p = np.array([0.2, 0.2])
    monthly = np.array([20.0, 100.0])      # values 100 and 500
    # expected saving: 0.2*0.5*100 = 10 (< 20)  vs  0.2*0.5*500 = 50 (> 20)
    assert expected_value_target(p, monthly, ECON).tolist() == [False, True]
    assert expected_net_gain(p, monthly, ECON).tolist() == pytest.approx([-10.0, 30.0])
