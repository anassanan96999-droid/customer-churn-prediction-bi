"""Shared data loading, economics state and chart styling for the dashboard."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_preprocessing import named_queries, run_named_query  # noqa: E402
from src.economics import (expected_loss, expected_net_gain, optimal_threshold,  # noqa: E402
                           threshold_curve)
from src.predict import load_bundle, shap_frame  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette - validated categorical order, blue<->red diverging, status for risk
# --------------------------------------------------------------------------- #
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BAR, BAR_MUTED = SERIES[0], "#b7d3f6"
STAYED, CHURNED = SERIES[0], SERIES[1]
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"
RISK_COLORS = {"HIGH": "#d03b3b", "MEDIUM": "#ec835a", "LOW": "#0ca30c"}
RISK_ORDER = ["HIGH", "MEDIUM", "LOW"]


def style(fig: go.Figure, height: int = 340, title: str | None = None,
          legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=title,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color=INK_2, size=12),
        title_font=dict(size=15, color=INK), title_x=0, title_xanchor="left",
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        showlegend=legend, bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="")
    return fig


def hbar(labels, values, text=None, highlight_max=True, value_format="{:.1f}%",
         hover=None) -> go.Figure:
    values = list(values)
    colors = [BAR if (not highlight_max or v == max(values)) else BAR_MUTED for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=list(labels), orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=text if text is not None else [value_format.format(v) for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate=hover or "%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, max(values) * 1.22])   # room for the value labels
    return fig


def chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False})


def show_sql(name: str) -> None:
    with st.expander("SQL behind this chart"):
        st.code(named_queries()[name], language="sql")


def money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def md(text: str) -> str:
    """Escape dollar signs: Streamlit markdown renders text between two '$' as LaTeX."""
    return text.replace("$", r"\$")


# --------------------------------------------------------------------------- #
# Cached artifacts
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    return pd.read_csv(config.SCORED_CSV)


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    return json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_threshold_report() -> dict:
    return json.loads(config.THRESHOLD_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / name)


@st.cache_data(show_spinner=False)
def load_llm_examples() -> list:
    return json.loads(config.LLM_EXAMPLES_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def query(name: str) -> pd.DataFrame:
    return run_named_query(name)


@st.cache_resource(show_spinner=False)
def bundle() -> dict:
    return load_bundle()


@st.cache_resource(show_spinner=False)
def explainer():
    from src.explain import ChurnExplainer
    b = bundle()
    return ChurnExplainer(b["pipeline"], b["background"])


@st.cache_data(show_spinner=False)
def shap_values() -> pd.DataFrame:
    return shap_frame(load_scored())


# --------------------------------------------------------------------------- #
# Campaign economics shared across pages (set on the Economics page)
# --------------------------------------------------------------------------- #
def economics() -> config.Economics:
    return st.session_state.get("economics", config.ECONOMICS)


@st.cache_data(show_spinner=False)
def _optimal_threshold(offer_cost, save_rate, horizon, margin) -> float:
    s = load_scored()
    econ = config.Economics(offer_cost, save_rate, horizon, margin)
    return optimal_threshold(s["churn"], s["churn_probability"], s["monthly_charges"], econ)


def threshold() -> float:
    """Money-optimal threshold for the current assumptions.

    With the default assumptions this is the threshold the pipeline chose on
    training out-of-fold scores; when a user changes the assumptions it is
    re-optimised on the out-of-fold scores of all customers.
    """
    econ = economics()
    if econ == config.ECONOMICS:
        return float(bundle()["threshold"])
    return _optimal_threshold(econ.offer_cost, econ.save_rate, econ.horizon_months,
                              econ.gross_margin)


def with_economics(scored: pd.DataFrame) -> pd.DataFrame:
    """Recompute money columns and the contact flag under the current assumptions."""
    econ, t = economics(), threshold()
    out = scored.copy()
    out["customer_value"] = econ.customer_value(out["monthly_charges"])
    out["expected_loss"] = expected_loss(out["churn_probability"], out["monthly_charges"], econ)
    out["expected_net_gain"] = expected_net_gain(out["churn_probability"], out["monthly_charges"], econ)
    out["contact_recommended"] = out["churn_probability"] >= t
    return out


def active_customers() -> pd.DataFrame:
    s = with_economics(load_scored())
    return s[s["churn"] == 0].sort_values("expected_loss", ascending=False)


def curve_for(econ: config.Economics) -> pd.DataFrame:
    s = load_scored()
    return threshold_curve(s["churn"], s["churn_probability"], s["monthly_charges"], econ)


def economics_caption() -> str:
    e, t = economics(), threshold()
    return md(f"Assumptions: offer ${e.offer_cost:,.0f} per contact, {e.save_rate:.0%} of "
              f"contacted churners saved, value = {e.horizon_months} months x "
              f"{e.gross_margin:.0%} margin. Contact threshold {t:.2f}. Change them on the "
              "Campaign economics page.")


def economics_changed() -> bool:
    return economics() != config.ECONOMICS


def replace_economics(**kwargs) -> None:
    st.session_state["economics"] = replace(economics(), **kwargs)


def risk_badge(level: str) -> str:
    icon = {"HIGH": ":material/error:", "MEDIUM": ":material/warning:", "LOW": ":material/check_circle:"}
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    return f":{color[level]}-badge[{icon[level]} {level}]"
