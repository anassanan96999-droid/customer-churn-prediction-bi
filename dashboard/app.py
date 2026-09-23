"""Customer Churn Prediction & BI System - Streamlit dashboard.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Churn Prediction & BI", page_icon=":material/insights:",
                   layout="wide")

# Streamlit Cloud: pick up the Claude key from secrets without ever showing it.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file locally - the app falls back to templates
    pass

views = HERE / "views"
nav = st.navigation({
    "Business": [
        st.Page(views / "overview.py", title="Executive overview", icon=":material/dashboard:",
                url_path="overview", default=True),
        st.Page(views / "targets.py", title="Retention targets", icon=":material/target:",
                url_path="targets"),
        st.Page(views / "economics.py", title="Campaign economics", icon=":material/payments:",
                url_path="economics"),
    ],
    "Analysis": [
        st.Page(views / "customers.py", title="Customer & churn analysis", icon=":material/groups:",
                url_path="analysis"),
        st.Page(views / "drivers.py", title="Churn drivers & segments", icon=":material/psychology:",
                url_path="drivers"),
        st.Page(views / "models.py", title="Model performance", icon=":material/model_training:",
                url_path="models"),
    ],
    "Tools": [
        st.Page(views / "predict.py", title="Score new customers", icon=":material/upload_file:",
                url_path="predict"),
    ],
})

with st.sidebar:
    st.caption("IBM Telco dataset - 7,043 customers. Scores are out-of-fold: every customer "
               "is scored by a model that never saw them.")

nav.run()
