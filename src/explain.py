"""Why is this customer at risk? SHAP values turned into business language.

Pipeline: fitted model -> SHAP value per *encoded* column -> summed back to the
source feature (all `contract_*` columns become "contract") -> the features
pushing this customer towards churn, phrased the way an account manager would
say them, plus a playbook action for each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.feature_engineering import output_feature_groups, percentile_of


class ChurnExplainer:
    """SHAP explainer for a fitted preprocessing + model pipeline.

    Values are reported per source feature, in the model's raw output space
    (log-odds for logistic regression and boosting; probability for forests).
    Positive = pushes towards churn.
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, max_background: int = 200):
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.features = list(config.MODEL_FEATURES)

        groups = output_feature_groups(self.preprocess)
        self._aggregate = np.zeros((len(groups), len(self.features)))
        for i, group in enumerate(groups):
            self._aggregate[i, self.features.index(group)] = 1.0

        if isinstance(self.model, LogisticRegression):
            sample = background[self.features].sample(
                min(max_background, len(background)), random_state=config.RANDOM_STATE)
            self._explainer = shap.LinearExplainer(self.model, self.preprocess.transform(sample))
            self._is_tree = False
        else:
            self._explainer = shap.TreeExplainer(self.model)
            self._is_tree = True

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (SHAP values per source feature, base value per row)."""
        encoded = self.preprocess.transform(X[self.features])
        if self._is_tree:
            values = self._explainer.shap_values(encoded, check_additivity=False)
        else:
            values = self._explainer.shap_values(encoded)
        base = np.asarray(self._explainer.expected_value, dtype=float)

        values = np.asarray(values)
        if values.ndim == 3:                     # (rows, cols, classes) -> churn class
            values = values[:, :, 1]
        if base.ndim and base.size > 1:
            base = base[1]
        grouped = values @ self._aggregate
        return (pd.DataFrame(grouped, columns=self.features, index=X.index),
                np.full(len(X), float(base)))


# --------------------------------------------------------------------------- #
# Plain-language phrasing
# --------------------------------------------------------------------------- #
_ADDONS = {
    "online_security": "online security add-on",
    "online_backup": "online backup add-on",
    "device_protection": "device protection add-on",
    "tech_support": "tech support add-on",
}


def describe(feature: str, row: pd.Series, reference: dict[str, list[float]]) -> str:
    """One factual phrase for a feature's value on this customer."""
    v = row[feature]

    def pct() -> float:
        return percentile_of(float(v), reference[feature])

    if feature == "tenure":
        v = int(v)
        if v == 0:
            return "Brand-new customer (not billed yet)"
        if v <= 12:
            return f"Short tenure (only {v} month{'s' if v != 1 else ''})"
        return f"{v} months as a customer"
    if feature == "monthly_charges":
        p = pct()
        return (f"High monthly bill of ${v:,.2f}, above {p:.0f}% of customers"
                if p >= 60 else f"Monthly bill of ${v:,.2f}")
    if feature == "total_charges":
        return f"Low lifetime spend so far (${v:,.0f})" if pct() < 40 else f"Lifetime spend of ${v:,.0f}"
    if feature == "contract":
        return f"{v} contract"
    if feature == "payment_method":
        return f"Pays by {v[0].lower() + v[1:]}"
    if feature == "internet_service":
        return {"Fiber optic": "Fiber-optic internet plan", "DSL": "DSL internet plan"}.get(
            v, "No internet service")
    if feature in _ADDONS:
        return f"No {_ADDONS[feature]}" if v == "No" else f"Has the {_ADDONS[feature]}"
    if feature == "streaming_tv":
        return "Streams TV" if v == "Yes" else "No streaming TV"
    if feature == "streaming_movies":
        return "Streams movies" if v == "Yes" else "No streaming movies"
    if feature == "paperless_billing":
        return "Paperless billing" if v == "Yes" else "Paper billing"
    if feature == "senior_citizen":
        return "Senior citizen" if int(v) == 1 else "Not a senior citizen"
    if feature == "partner":
        return "No partner on the account" if v == "No" else "Partner on the account"
    if feature == "dependents":
        return "No dependents on the account" if v == "No" else "Dependents on the account"
    if feature == "family_account":
        return "Family account" if int(v) == 1 else "Single-person account"
    if feature == "phone_service":
        return "Has phone service" if v == "Yes" else "No phone line"
    if feature == "multiple_lines":
        return "Multiple phone lines" if v == "Yes" else "Single phone line"
    if feature == "gender":
        return f"Gender: {v}"
    if feature == "avg_monthly_spend":
        return f"Lifetime average bill of ${v:,.2f}/month"
    if feature == "spend_trend_ratio":
        if v >= 1.05:
            return f"Current bill {(v - 1) * 100:.0f}% above their lifetime average"
        if v <= 0.95:
            return f"Current bill {(1 - v) * 100:.0f}% below their lifetime average"
        return "Current bill in line with their lifetime average"
    if feature == "service_count":
        v = int(v)
        return f"Only {v} service{'s' if v != 1 else ''} subscribed" if v <= 2 else f"{v} services subscribed"
    if feature == "charge_per_service":
        p = pct()
        return (f"Pays ${v:,.2f} per service, above {p:.0f}% of customers"
                if p >= 60 else f"Pays ${v:,.2f} per service")
    if feature == "has_protection_bundle":
        return "Has security + tech-support bundle" if int(v) == 1 else "No security + tech-support bundle"
    return f"{config.FEATURE_LABELS.get(feature, feature)}: {v}"


# --------------------------------------------------------------------------- #
# Retention playbook: which action answers which risk factor
# --------------------------------------------------------------------------- #
_PLAYBOOK = [
    (("contract",), lambda r: r["contract"] == "Month-to-month",
     "Offer a discounted 12-month contract"),
    (("tenure", "total_charges"), lambda r: r["tenure"] <= 12,
     "Early-life onboarding call: check the setup, walk through the bill"),
    (("tech_support", "online_security", "has_protection_bundle"),
     lambda r: r["tech_support"] == "No" or r["online_security"] == "No",
     "Free 3-month trial of Tech Support + Online Security"),
    (("payment_method",), lambda r: r["payment_method"] == "Electronic check",
     "Move to automatic payment with a small bill credit"),
    (("monthly_charges", "charge_per_service", "spend_trend_ratio", "avg_monthly_spend"),
     lambda r: True, "Plan review: right-size the bundle or apply a loyalty discount"),
    (("internet_service",), lambda r: r["internet_service"] == "Fiber optic",
     "Proactive fiber service check (speed test, outage history)"),
]
DEFAULT_ACTION = "Personal check-in call from the customer-success team"


def recommend_actions(risk_features: list[str], row: pd.Series, k: int = 2) -> list[str]:
    actions: list[str] = []
    for feature in risk_features:
        for triggers, applies, action in _PLAYBOOK:
            if feature in triggers and applies(row) and action not in actions:
                actions.append(action)
    return actions[:k] or [DEFAULT_ACTION]


def explain_customer(shap_row: pd.Series, row: pd.Series, reference: dict[str, list[float]],
                     k: int = 4) -> dict:
    """Top risk factors, top protective factors and recommended actions."""
    ranked = shap_row.sort_values(ascending=False)
    risk = [f for f, s in ranked.items() if s > 0.01][:k]
    protective = [f for f, s in ranked[::-1].items() if s < -0.01][:2]
    return {
        "risk_factors": [{"feature": f, "text": describe(f, row, reference),
                          "impact": float(shap_row[f])} for f in risk],
        "protective_factors": [{"feature": f, "text": describe(f, row, reference),
                                "impact": float(shap_row[f])} for f in protective],
        "actions": recommend_actions(risk, row),
    }
