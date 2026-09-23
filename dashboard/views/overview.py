import plotly.graph_objects as go
import streamlit as st

from common import (RISK_COLORS, RISK_ORDER, active_customers, chart, economics,
                    economics_caption, hbar, load_csv, md, money, query, show_sql, style,
                    threshold)

st.title("Executive overview")
st.caption(economics_caption())

active = active_customers()          # sorted by expected loss, highest first
flagged = active[active["contact_recommended"]]
kpi = query("kpi_overview").iloc[0]
econ, t = economics(), threshold()

revenue_at_risk = float((active["churn_probability"] * active["monthly_charges"] * 12).sum())
campaign_value = float(flagged["expected_net_gain"].sum())
loss_share = flagged["expected_loss"].sum() / active["expected_loss"].sum()

cols = st.columns(5)
cols[0].metric("Customers", f"{int(kpi['customers']):,}", border=True)
cols[1].metric("Churn rate", f"{kpi['churn_rate_pct']:.1f}%", border=True,
               help=f"Customers who left last month: {int(kpi['churned']):,}. They took "
                    f"{kpi['revenue_churn_pct']:.1f}% of monthly revenue with them.")
cols[2].metric("To contact", f"{len(flagged):,}", border=True,
               help=f"Active customers with churn probability >= {t:.2f}, the threshold that "
                    "maximises campaign value under the current assumptions.")
cols[3].metric("Revenue at risk / yr", money(revenue_at_risk), border=True,
               help="Sum over active customers of P(churn) x monthly bill x 12.")
cols[4].metric("Campaign value", money(campaign_value), border=True,
               help="Expected net value of contacting them: sum of P(churn) x save rate x "
                    "customer value - offer cost.")

st.info(md(
    f"**{len(flagged):,} of {len(active):,} active customers** ({len(flagged) / len(active):.0%}) "
    f"carry **{loss_share:.0%} of the expected margin loss**. Contacting them costs about "
    f"{money(len(flagged) * econ.offer_cost)} and is expected to return "
    f"{money(campaign_value)} net over {econ.horizon_months} months."),
    icon=":material/lightbulb:")

left, mid, right = st.columns(3)
with left:
    by_risk = (active.groupby("risk_level")
               .agg(customers=("customer_id", "size"), expected_loss=("expected_loss", "sum"))
               .reindex(RISK_ORDER).fillna(0))
    fig = go.Figure(go.Bar(
        x=[f"{r}<br>n={int(n):,}" for r, n in zip(by_risk.index, by_risk["customers"])],
        y=by_risk["expected_loss"],
        marker_color=[RISK_COLORS[r] for r in by_risk.index],
        text=[money(v) for v in by_risk["expected_loss"]],
        textposition="outside", cliponaxis=False, textfont=dict(size=12),
        hovertemplate="%{x}: %{y:$,.0f} expected loss<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f",
                     range=[0, by_risk["expected_loss"].max() * 1.25])
    fig.update_xaxes(tickangle=0)
    chart(style(fig, 360, "Expected margin loss by risk level", legend=False))
    st.caption("Active customers. HIGH >= 60%, MEDIUM 30-60%, LOW < 30% churn probability.")

with mid:
    df = query("churn_by_contract")
    fig = hbar(df["segment"], df["churn_rate_pct"],
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 360, "Churn rate by contract type", legend=False))
    show_sql("churn_by_contract")

with right:
    imp = load_csv("global_importance.csv").head(8)
    fig = hbar(imp["label"], imp["mean_abs_shap"], highlight_max=False, value_format="{:.2f}",
               hover="%{y}: %{x:.3f}<extra></extra>")
    fig.update_xaxes(title_text="mean |SHAP| (log-odds)")
    chart(style(fig, 360, "Top churn drivers", legend=False))

st.markdown("##### Call these customers first")
top = flagged.head(10)[["customer_id", "churn_probability", "expected_loss", "contract",
                        "tenure", "monthly_charges", "reason_1", "reason_2", "recommended_action"]]
st.dataframe(
    top, hide_index=True,
    column_config={
        "customer_id": "Customer",
        "churn_probability": st.column_config.ProgressColumn(
            "P(churn)", format="percent", min_value=0, max_value=1),
        "expected_loss": st.column_config.NumberColumn("Expected loss", format="dollar"),
        "contract": "Contract",
        "tenure": st.column_config.NumberColumn("Tenure (mo)"),
        "monthly_charges": st.column_config.NumberColumn("Monthly bill", format="dollar"),
        "reason_1": "Main reason",
        "reason_2": "Second reason",
        "recommended_action": "Recommended action",
    })
st.page_link("views/targets.py", label="Full retention list with explanations",
             icon=":material/arrow_forward:")
