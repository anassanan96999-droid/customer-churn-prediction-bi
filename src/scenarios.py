"""Scenario Lab: model-based what-if analysis of retention interventions.

Each intervention changes the *profile* of the customers it applies to (e.g.
month-to-month -> one-year contract) and the champion model re-scores them.
The drop in churn probability, multiplied by customer value and by the share
of customers who accept the offer, is the expected benefit; the offer cost is
paid for every customer who accepts.

Caveat, shown in the UI too: this is what the *model* expects, based on how
similar customers behave today. It is correlational, not a proven causal
effect - an A/B test is the way to confirm it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.config import ECONOMICS, Economics

ADDON_COLUMNS = ["online_security", "online_backup", "device_protection", "tech_support",
                 "streaming_tv", "streaming_movies"]


# --------------------------------------------------------------------------- #
# Derived features - a Python mirror of the customer_features view in
# sql/01_create_schema.sql (tests/test_scenarios.py checks they agree).
# --------------------------------------------------------------------------- #
def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def yes(col: str) -> pd.Series:
        return (out[col] == "Yes").astype(int)

    services = ((out["phone_service"] == "Yes").astype(int) + yes("multiple_lines")
                + (out["internet_service"] != "No").astype(int)
                + sum(yes(c) for c in ADDON_COLUMNS))
    tenure = out["tenure"].astype(float)
    monthly = out["monthly_charges"].astype(float)
    lifetime_avg = out["total_charges"].astype(float) / tenure.where(tenure > 0)

    out["service_count"] = services
    out["streaming_count"] = yes("streaming_tv") + yes("streaming_movies")
    out["has_protection_bundle"] = ((out["online_security"] == "Yes")
                                    & (out["tech_support"] == "Yes")).astype(int)
    out["family_account"] = ((out["partner"] == "Yes") | (out["dependents"] == "Yes")).astype(int)
    out["is_month_to_month"] = (out["contract"] == "Month-to-month").astype(int)
    out["is_electronic_check"] = (out["payment_method"] == "Electronic check").astype(int)
    out["is_auto_payment"] = out["payment_method"].str.contains("automatic").astype(int)
    out["is_new_customer"] = (tenure <= 6).astype(int)
    out["avg_monthly_spend"] = lifetime_avg.round(2).fillna(monthly)
    out["spend_trend_ratio"] = (monthly / lifetime_avg.where(lifetime_avg != 0)).round(3).fillna(1.0)
    out["charge_per_service"] = (monthly / services.where(services > 0)).round(2)
    out["revenue_at_risk_annual"] = (monthly * 12).round(2)
    return out


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intervention:
    key: str
    label: str
    description: str
    cost_label: str
    default_cost: float
    eligible: Callable[[pd.DataFrame], pd.Series]
    apply: Callable[[pd.DataFrame, dict], pd.DataFrame]
    # cost per accepting customer, given the rows and the scenario parameters
    cost: Callable[[pd.DataFrame, dict], pd.Series]


def _set(**values):
    def apply(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
        return rows.assign(**values)
    return apply


def _discount(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    return rows.assign(monthly_charges=(rows["monthly_charges"]
                                       * (1 - params.get("discount", 0.10))).round(2))


INTERVENTIONS: dict[str, Intervention] = {iv.key: iv for iv in [
    Intervention(
        "annual_contract", "Upgrade to a 1-year contract",
        "Month-to-month customers move to a one-year contract.",
        "Free months offered", 1.0,
        eligible=lambda d: d["contract"] == "Month-to-month",
        apply=_set(contract="One year"),
        cost=lambda d, p: d["monthly_charges"] * p.get("annual_contract", 1.0)),
    Intervention(
        "two_year_contract", "Upgrade to a 2-year contract",
        "Customers not yet on a two-year contract move to one.",
        "Free months offered", 2.0,
        eligible=lambda d: d["contract"] != "Two year",
        apply=_set(contract="Two year"),
        cost=lambda d, p: d["monthly_charges"] * p.get("two_year_contract", 2.0)),
    Intervention(
        "protection_bundle", "Add Online Security + Tech Support",
        "Internet customers without the bundle get both add-ons at no extra charge.",
        "Cost of add-ons given away ($ per customer)", 60.0,
        eligible=lambda d: (d["internet_service"] != "No") & ~((d["online_security"] == "Yes")
                                                                & (d["tech_support"] == "Yes")),
        apply=_set(online_security="Yes", tech_support="Yes"),
        cost=lambda d, p: pd.Series(p.get("protection_bundle", 60.0), index=d.index)),
    Intervention(
        "auto_pay", "Switch to automatic payment",
        "Check payers move to automatic card payment, with a one-off bill credit.",
        "Bill credit ($ per customer)", 5.0,
        eligible=lambda d: ~d["payment_method"].str.contains("automatic"),
        apply=_set(payment_method="Credit card (automatic)"),
        cost=lambda d, p: pd.Series(p.get("auto_pay", 5.0), index=d.index)),
    Intervention(
        "price_discount", "Loyalty discount on the monthly bill",
        "Everyone targeted gets a percentage off their bill; the lost revenue is the cost.",
        "Discount (% of the bill)", 0.10,
        eligible=lambda d: d["monthly_charges"] > 0,
        apply=_discount,
        cost=lambda d, p: d["monthly_charges"] * p.get("discount", 0.10) * p.get("horizon", 12)),
]}


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _predict(pipeline, features: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(features[config.MODEL_FEATURES])[:, 1]


def apply_interventions(features: pd.DataFrame, keys: list[str],
                        params: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (changed profiles, eligible-for-anything mask, offer cost per customer)."""
    params = params or {}
    changed = features.copy()
    eligible = pd.Series(False, index=features.index)
    cost = pd.Series(0.0, index=features.index)
    for key in keys:
        iv = INTERVENTIONS[key]
        mask = iv.eligible(changed)
        if mask.any():
            cost[mask] += iv.cost(changed.loc[mask], params)
            changed.loc[mask] = iv.apply(changed.loc[mask], params)
            eligible |= mask
    return recompute_derived(changed), eligible, cost


def simulate(features: pd.DataFrame, pipeline, keys: list[str], acceptance: float,
             params: dict | None = None, econ: Economics = ECONOMICS) -> tuple[dict, pd.DataFrame]:
    """Expected effect of offering `keys` to every customer in `features`."""
    params = {"horizon": econ.horizon_months, **(params or {})}
    base = _predict(pipeline, features)
    changed, eligible, cost = apply_interventions(features, keys, params)
    after = np.where(eligible, _predict(pipeline, changed), base)

    value = econ.customer_value(features["monthly_charges"])
    drop = base - after
    detail = pd.DataFrame({
        config.ID_COL: features[config.ID_COL].to_numpy(),
        "eligible": eligible.to_numpy(),
        "p_before": base, "p_after": after, "risk_drop": drop,
        "value": value, "offer_cost": cost.to_numpy(),
        "expected_benefit": acceptance * drop * value,
        "expected_cost": acceptance * cost.to_numpy(),
    })
    detail["expected_net"] = detail["expected_benefit"] - detail["expected_cost"]

    n_eligible = int(eligible.sum())
    benefit, spend = detail["expected_benefit"].sum(), detail["expected_cost"].sum()
    summary = {
        "customers": len(features),
        "eligible": n_eligible,
        "accepting": acceptance * n_eligible,
        "churners_before": float(base.sum()),
        "churners_after": float((base - acceptance * drop).sum()),
        "churners_avoided": float(acceptance * drop.sum()),
        "margin_retained": float(benefit),
        "offer_cost": float(spend),
        "net_value": float(benefit - spend),
        "roi": float((benefit - spend) / spend) if spend else float("nan"),
        "avg_risk_before": float(base[eligible.to_numpy()].mean()) if n_eligible else float("nan"),
        "avg_risk_after": float(after[eligible.to_numpy()].mean()) if n_eligible else float("nan"),
    }
    return summary, detail


def next_best_actions(row: pd.Series, pipeline, params: dict | None = None,
                      econ: Economics = ECONOMICS) -> pd.DataFrame:
    """Every intervention this customer is eligible for, ranked by risk reduction."""
    one = row.to_frame().T.infer_objects()
    rows = []
    for key, iv in INTERVENTIONS.items():
        summary, detail = simulate(one, pipeline, [key], acceptance=1.0, params=params, econ=econ)
        if not summary["eligible"]:
            continue
        d = detail.iloc[0]
        rows.append({"action": iv.label, "key": key, "p_before": d["p_before"],
                     "p_after": d["p_after"], "risk_drop": d["risk_drop"],
                     "margin_saved": d["expected_benefit"], "offer_cost": d["offer_cost"],
                     "net_if_accepted": d["expected_net"]})
    if not rows:
        return pd.DataFrame(columns=["action", "key", "p_before", "p_after", "risk_drop",
                                     "margin_saved", "offer_cost", "net_if_accepted"])
    return pd.DataFrame(rows).sort_values("risk_drop", ascending=False, ignore_index=True)
