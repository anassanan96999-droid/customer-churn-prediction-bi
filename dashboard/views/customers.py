import plotly.graph_objects as go
import streamlit as st

from common import (BAR, CHURNED, STAYED, chart, hbar, load_scored, query, show_sql, style)

st.title("Customer & churn analysis")
st.caption("Every segment chart on this page is a named query in sql/03_business_queries.sql, "
           "run live against the SQLite analytical layer.")

kpi = query("kpi_overview").iloc[0]
overall = kpi["churn_rate_pct"]

DIMENSIONS = {
    "Contract": "churn_by_contract",
    "Tenure": "churn_by_tenure_bucket",
    "Payment method": "churn_by_payment_method",
    "Internet service": "churn_by_internet_service",
    "Security / support add-ons": "churn_by_protection",
    "Age group": "churn_by_senior",
    "Number of services": "churn_by_service_count",
}
dim = st.segmented_control("Break churn down by", list(DIMENSIONS), default="Contract",
                           required=True)
name = DIMENSIONS[dim]
df = query(name)
left, right = st.columns([3, 2])
with left:
    fig = hbar(df["segment"].astype(str), df["churn_rate_pct"],
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.add_vline(x=overall, line_dash="dot", line_color="#8a8984",
                  annotation_text=f"all customers {overall:.1f}%", annotation_position="top")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 360, f"Churn rate by {dim.lower()}", legend=False))
with right:
    st.dataframe(df, hide_index=True)
    show_sql(name)

st.divider()
left, right = st.columns(2)
with left:
    cx = query("churn_contract_x_internet")
    pivot = cx.pivot(index="contract", columns="internet_service", values="churn_rate_pct")
    pivot = pivot.loc[["Month-to-month", "One year", "Two year"], ["Fiber optic", "DSL", "No"]]
    counts = cx.pivot(index="contract", columns="internet_service", values="customers").loc[
        pivot.index, pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, zmin=0, zmax=60,
        colorscale=[[0, "#f0efec"], [1, "#256abf"]], colorbar=dict(ticksuffix="%", len=0.8),
        text=[[f"{v:.0f}%<br>n={n:,}" for v, n in zip(r1, r2)]
              for r1, r2 in zip(pivot.values, counts.values)],
        texttemplate="%{text}", hovertemplate="%{y} + %{x}: %{z:.1f}% churn<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, title_text="Internet service")
    chart(style(fig, 340, "Contract x internet service: churn rate", legend=False))
    show_sql("churn_contract_x_internet")
with right:
    tm = query("churn_by_tenure_month")
    tm = tm[tm["tenure"] > 0]
    fig = go.Figure(go.Scatter(x=tm["tenure"], y=tm["cumulative_share_of_churners_pct"],
                               mode="lines", line=dict(color=BAR, width=2.5),
                               hovertemplate="month %{x}: %{y:.1f}% of churners<extra></extra>"))
    for m in (1, 6, 12):
        v = tm.loc[tm["tenure"] == m, "cumulative_share_of_churners_pct"].iloc[0]
        fig.add_annotation(x=m, y=v, text=f"{v:.0f}% by month {m}", showarrow=True,
                           arrowhead=0, ax=50, ay=10, font=dict(size=11))
    fig.update_yaxes(ticksuffix="%", range=[0, 102])
    fig.update_xaxes(title_text="Tenure (months)")
    chart(style(fig, 340, "Churn is front-loaded: cumulative share of churners", legend=False))
    show_sql("churn_by_tenure_month")

st.divider()
st.subheader("Spending behaviour and demographics")
scored = load_scored()
left, right = st.columns(2)
with left:
    fig = go.Figure()
    for label, value, color in (("Stayed", 0, STAYED), ("Churned", 1, CHURNED)):
        fig.add_trace(go.Histogram(
            x=scored.loc[scored["churn"] == value, "monthly_charges"], name=label,
            histnorm="percent", xbins=dict(start=15, end=125, size=10), marker_color=color,
            hovertemplate=f"{label}: %{{y:.1f}}% of group<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.08)
    fig.update_xaxes(title_text="Monthly charges ($)", tickprefix="$")
    fig.update_yaxes(title_text="% of group", ticksuffix="%")
    chart(style(fig, 340, "Monthly charges: churners pay more"))
with right:
    rows = []
    for col, yes_label, no_label in (("senior_citizen", "Senior", "Non-senior"),
                                     ("partner", "Has partner", "No partner"),
                                     ("dependents", "Has dependents", "No dependents"),
                                     ("gender", "Female", "Male")):
        grp = scored.groupby(col)["churn"].mean() * 100
        for key, rate in grp.items():
            label = {1: yes_label, 0: no_label, "Yes": yes_label, "No": no_label,
                     "Female": "Female", "Male": "Male"}[key]
            rows.append((label, rate))
    fig = hbar([r[0] for r in rows], [r[1] for r in rows], highlight_max=False,
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 340, "Churn rate by demographic group", legend=False))
    st.caption("Gender makes no difference; seniors and single-person households churn more.")
