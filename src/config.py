"""Central configuration for the Customer Churn Prediction & BI System.

Everything that a business user might want to change - file locations, the
economics of a retention campaign, model hyper-parameters - lives here so that
the rest of the code never hard-codes a magic number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_CSV = RAW_DIR / "Telco-Customer-Churn.csv"
DB_PATH = DATA_DIR / "churn.db"
FEATURES_CSV = PROCESSED_DIR / "customers_features.csv"

MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METRICS_JSON = REPORTS_DIR / "model_comparison.json"
THRESHOLD_JSON = REPORTS_DIR / "threshold_economics.json"
SCORED_CSV = PROCESSED_DIR / "customers_scored.csv"
GLOBAL_IMPORTANCE_CSV = REPORTS_DIR / "global_importance.csv"
SEGMENTS_CSV = PROCESSED_DIR / "customer_segments.csv"
LLM_EXAMPLES_JSON = REPORTS_DIR / "llm_examples.json"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SQL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
TARGET = "churn"
ID_COL = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 25

# Metric used to pick the champion model. ROC-AUC is threshold independent and
# is not fooled by the 73/27 class imbalance the way accuracy is.
SELECTION_METRIC = "roc_auc"

# Columns straight from the source system (after SQL cleaning).
BASE_NUMERIC = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
BASE_CATEGORICAL = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]

# Business features derived in sql/01_create_schema.sql. Pure recodings of an
# existing column (is_month_to_month, is_electronic_check, tenure_bucket, ...)
# are kept for BI but left out of the model: one-hot encoding already carries
# that information, and duplicates would split SHAP credit between two columns.
ENGINEERED_NUMERIC = [
    "avg_monthly_spend", "spend_trend_ratio", "service_count",
    "charge_per_service", "has_protection_bundle", "family_account",
]

NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = BASE_CATEGORICAL
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Scored columns published to the SQL layer as the `customer_scores` table.
SCORE_TABLE_COLUMNS = [
    "customer_id", "churn_probability", "risk_level", "customer_value", "expected_loss",
    "expected_net_gain", "contact_recommended", "top_reasons", "recommended_action", "segment",
]

# Columns a raw upload must contain (the original IBM Telco headers).
RAW_REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# --------------------------------------------------------------------------- #
# Campaign economics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Economics:
    """The money model behind the retention campaign.

    The classifier outputs a probability; turning that probability into a
    *decision* ("do we call this customer?") requires a price list. These are
    the assumptions the optimal threshold is derived from - all of them are
    exposed as sliders in the dashboard so a manager can test their own numbers.
    """

    # One-off cost of extending a retention offer (discount, credit, call time).
    # Paid for every customer we contact, whether or not they were going to leave.
    offer_cost: float = 20.0

    # Share of genuinely at-risk customers who accept the offer and stay.
    # Retention campaigns are not magic; 30% is a deliberately sober default.
    save_rate: float = 0.30

    # How far ahead we count the revenue of a customer we manage to keep.
    horizon_months: int = 12

    # Contribution margin on telco revenue - the part of the bill that is
    # actually profit rather than network and support cost.
    gross_margin: float = 0.30

    def customer_value(self, monthly_charges):
        """Margin value of keeping a customer over the planning horizon."""
        return np.asarray(monthly_charges, dtype=float) * self.horizon_months * self.gross_margin

    def as_dict(self) -> dict:
        return asdict(self)


ECONOMICS = Economics()

# --------------------------------------------------------------------------- #
# Risk banding (applied to churn probability). Bands are for communication;
# who gets contacted is decided by the money-optimal threshold.
# --------------------------------------------------------------------------- #
RISK_BANDS = [(0.60, "HIGH"), (0.30, "MEDIUM"), (0.00, "LOW")]


def risk_level(prob: float) -> str:
    for cutoff, label in RISK_BANDS:
        if prob >= cutoff:
            return label
    return "LOW"


# --------------------------------------------------------------------------- #
# LLM (retention summaries). Optional: without an API key the app falls back to
# a deterministic template, so nothing breaks on a fresh clone.
# --------------------------------------------------------------------------- #
LLM_MODEL = "claude-opus-5"

# --------------------------------------------------------------------------- #
# Human-readable names for model features (used in charts and explanations)
# --------------------------------------------------------------------------- #
FEATURE_LABELS = {
    "tenure": "Tenure",
    "monthly_charges": "Monthly charges",
    "total_charges": "Total charges to date",
    "senior_citizen": "Senior citizen",
    "gender": "Gender",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone service",
    "multiple_lines": "Multiple lines",
    "internet_service": "Internet service",
    "online_security": "Online security",
    "online_backup": "Online backup",
    "device_protection": "Device protection",
    "tech_support": "Tech support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
    "contract": "Contract type",
    "paperless_billing": "Paperless billing",
    "payment_method": "Payment method",
    "avg_monthly_spend": "Avg. monthly spend (lifetime)",
    "spend_trend_ratio": "Current bill vs. lifetime avg.",
    "service_count": "Number of services",
    "charge_per_service": "Price per service",
    "has_protection_bundle": "Security + support bundle",
    "family_account": "Family account",
}
