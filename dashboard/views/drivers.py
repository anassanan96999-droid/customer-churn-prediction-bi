import plotly.graph_objects as go
import streamlit as st

from common import (BAR, INK_3, SERIES, chart, hbar, load_csv, load_report, load_scored,
                    money, style)
from src.config import FEATURE_LABELS
from theme import hero, section

report = load_report()
hero("Churn drivers & segments",
     f"Why customers leave, from SHAP on the champion model ({report['champion']}) - computed "
     "out-of-fold for all 7,043 customers and summed back from encoded columns to the business "
     "feature they came from - cross-checked with permutation importance.",
     eyebrow="Explainable AI",
     chips=["SHAP", "Permutation importance", "Dependence plots", "k-means segments"])

left, right = st.columns(2)
with left:
    imp = load_csv("global_importance.csv").head(12)
    fig = hbar(imp["label"], imp["mean_abs_shap"], highlight_max=False, value_format="{:.2f}",
               hover="%{y}: %{x:.3f}<extra></extra>")
    fig.update_xaxes(title_text="mean |SHAP| (log-odds)")
    chart(style(fig, 440, "SHAP: average impact on the prediction", legend=False))
with right:
    perm = load_csv("permutation_importance.csv").head(12)
    fig = go.Figure(go.Bar(
        x=perm["importance_mean"], y=[FEATURE_LABELS.get(f, f) for f in perm["feature"]],
        orientation="h", marker_color=BAR,
        error_x=dict(type="data", array=perm["importance_std"], color=INK_3, thickness=1),
        hovertemplate="%{y}: ROC-AUC drops %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Drop in test ROC-AUC when shuffled")
    chart(style(fig, 440, "Permutation importance (test set)", legend=False))
st.caption("Two independent methods - SHAP (how much each feature moves individual predictions) "
           "and permutation importance (how much accuracy is lost without it) - agree on the "
           "top three: contract, tenure and internet service, well ahead of payment method and "
           "the rest. Correlated features (tenure and total charges) share credit under "
           "permutation, which is why total charges ranks low there.")

st.divider()
section("How the main drivers behave", "each point is one customer")
scored = load_scored().sample(3000, random_state=42)
left, right = st.columns(2)
for col, feature, title, fmt in ((left, "tenure", "Tenure effect by contract type", "{:.0f} months"),
                                 (right, "monthly_charges", "Monthly-charge effect by contract type",
                                  "${:.0f}")):
    fig = go.Figure()
    for i, contract in enumerate(["Month-to-month", "One year", "Two year"]):
        d = scored[scored["contract"] == contract]
        fig.add_trace(go.Scattergl(
            x=d[feature], y=d[f"shap__{feature}"], mode="markers", name=contract,
            marker=dict(color=SERIES[i], size=5, opacity=0.5),
            hovertemplate=f"{contract}<br>{feature} %{{x}}<br>SHAP %{{y:+.2f}}<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title_text=FEATURE_LABELS[feature])
    fig.update_yaxes(title_text="SHAP (pushes to churn ->)")
    with col:
        chart(style(fig, 380, title))
st.caption("Short tenure adds a lot of risk in the first months; the effect falls steeply through "
           "year one, crosses zero at around 12-18 months and keeps protecting (more slowly) after "
           "that. Each point is one customer; above zero means the feature raises their risk.")

st.divider()
section("Behavioural segments", "k-means on tenure, monthly charges and number of services")
seg = load_csv("segment_profile.csv")
left, right = st.columns([2, 3])
with left:
    fig = hbar(seg["segment"], seg["churn_rate"] * 100, hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 300, "Churn rate by segment", legend=False))
with right:
    show = seg.assign(
        share=seg["share_of_customers"], churn=seg["churn_rate"],
        expected_loss=seg["expected_loss_active"].map(money))[
        ["segment", "customers", "share", "churn", "avg_tenure", "avg_monthly_charges",
         "avg_services", "active_customers", "expected_loss"]]
    st.dataframe(show, hide_index=True, column_config={
        "segment": "Segment", "customers": "Customers",
        "share": st.column_config.NumberColumn("Share", format="percent"),
        "churn": st.column_config.NumberColumn("Churn rate", format="percent"),
        "avg_tenure": st.column_config.NumberColumn("Avg tenure (mo)", format="%.0f"),
        "avg_monthly_charges": st.column_config.NumberColumn("Avg bill", format="dollar"),
        "avg_services": st.column_config.NumberColumn("Avg services", format="%.1f"),
        "active_customers": "Active", "expected_loss": "Expected loss (active)"})
    st.caption("Segments give the campaign team a vocabulary: the model scores individuals, "
               "segments decide which offer template and channel to use.")
