import pandas as pd

from src.explain import DEFAULT_ACTION, describe, recommend_actions
from src.llm import generate_brief, template_brief

REFERENCE = {c: list(range(0, 101)) for c in
             ("monthly_charges", "total_charges", "charge_per_service")}

ROW = pd.Series({
    "tenure": 2, "monthly_charges": 95.3, "contract": "Month-to-month",
    "payment_method": "Electronic check", "internet_service": "Fiber optic",
    "tech_support": "No", "online_security": "No", "spend_trend_ratio": 1.2,
})


def test_describe_reads_like_a_business_sentence():
    assert describe("tenure", ROW, REFERENCE) == "Short tenure (only 2 months)"
    assert describe("contract", ROW, REFERENCE) == "Month-to-month contract"
    assert describe("payment_method", ROW, REFERENCE) == "Pays by electronic check"
    assert describe("tech_support", ROW, REFERENCE) == "No tech support add-on"
    assert describe("spend_trend_ratio", ROW, REFERENCE).startswith("Current bill 20% above")
    assert describe("monthly_charges", ROW, REFERENCE).startswith("High monthly bill of $95.30")


def test_actions_follow_the_risk_factors_in_order():
    actions = recommend_actions(["contract", "tech_support", "tenure"], ROW)
    assert actions == ["Offer a discounted 12-month contract",
                       "Free 3-month trial of Tech Support + Online Security"]
    assert recommend_actions(["gender"], ROW) == [DEFAULT_ACTION]


CTX = {
    "customer_id": "0000-TEST", "churn_probability": "82%", "risk_level": "HIGH",
    "expected_margin_loss_next_12_months": "$280", "tenure_months": 2,
    "contract": "Month-to-month", "monthly_charges": "$95.30",
    "internet_service": "Fiber optic", "payment_method": "Electronic check",
    "risk_factors": ["Month-to-month contract", "Short tenure (only 2 months)"],
    "protective_factors": [], "approved_actions": ["Offer a discounted 12-month contract"],
}


def test_template_brief_uses_only_given_facts():
    brief = template_brief(CTX)
    assert brief["source"] == "template"
    assert "82%" in brief["summary"] and "month-to-month contract" in brief["summary"]
    assert "12-month plan" in brief["customer_message"]
    assert "churn" not in brief["customer_message"].lower()


def test_generate_brief_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert generate_brief(CTX)["source"] == "template"
