"""Data-drift checks for files scored after training.

A model is only as good as the resemblance between today's customers and the
ones it learned from. The Population Stability Index (PSI) measures how far a
feature's distribution in a new file has moved from the training data:

    PSI < 0.10   stable
    0.10 - 0.25  moderate shift - watch it
    > 0.25       significant shift - scores for this file are less reliable
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: pd.Series, new: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant feature: compare as categories
        return categorical_psi(reference.astype(str), new.astype(str))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference.astype(float), edges)[0] / len(reference)
    act = np.histogram(new.astype(float), edges)[0] / max(len(new), 1)
    return _psi(ref, act)


def categorical_psi(reference: pd.Series, new: pd.Series) -> float:
    levels = sorted(set(reference.unique()) | set(new.unique()))
    ref = reference.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    act = new.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return _psi(ref, act)


def status(psi: float) -> str:
    return "Stable" if psi < 0.10 else ("Watch" if psi < 0.25 else "Shifted")


def drift_report(reference: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """One row per model feature: PSI and a stable / watch / shifted verdict."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        psi = numeric_psi(reference[col], new[col])
        rows.append({"feature": col, "type": "numeric", "psi": psi,
                     "reference": f"median {reference[col].median():,.2f}",
                     "new_file": f"median {new[col].median():,.2f}"})
    for col in config.CATEGORICAL_FEATURES:
        psi = categorical_psi(reference[col].astype(str), new[col].astype(str))
        top_ref = reference[col].value_counts(normalize=True)
        top_new = new[col].value_counts(normalize=True)
        rows.append({"feature": col, "type": "categorical", "psi": psi,
                     "reference": f"{top_ref.index[0]} {top_ref.iloc[0]:.0%}",
                     "new_file": f"{top_new.index[0]} {top_new.iloc[0]:.0%}"})
    report = pd.DataFrame(rows)
    report["status"] = report["psi"].map(status)
    report["label"] = report["feature"].map(config.FEATURE_LABELS)
    return report.sort_values("psi", ascending=False, ignore_index=True)


def overall_status(report: pd.DataFrame) -> str:
    if (report["status"] == "Shifted").any():
        return "Shifted"
    return "Watch" if (report["status"] == "Watch").any() else "Stable"
