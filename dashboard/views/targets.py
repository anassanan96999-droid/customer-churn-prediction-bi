import json

import plotly.graph_objects as go
import streamlit as st

from common import (INK_3, RISK_DOWN, RISK_ORDER, RISK_UP, active_customers, bundle, chart,
                    economics, economics_caption, md, money, risk_badge, shap_values, style)
from src.config import FEATURE_LABELS
from src.explain import explain_customer
from src.llm import build_context, generate_brief, llm_available

st.title("Retention targets")
st.caption("Active customers ranked by **expected loss** = P(churn) x customer value, so the "
           "team calls the customers where the money is first. " + economics_caption())

active = active_customers()

f1, f2, f3, f4 = st.columns([1.2, 1.4, 1.6, 1])
risk = f1.pills("Risk level", RISK_ORDER, selection_mode="multi", default=["HIGH", "MEDIUM"])
contracts = f2.multiselect("Contract", sorted(active["contract"].unique()))
segments = f3.multiselect("Segment", sorted(active["segment"].unique()))
only_contact = f4.toggle("Recommended for contact only", value=True)

view = active
if risk:
    view = view[view["risk_level"].isin(risk)]
if contracts:
    view = view[view["contract"].isin(contracts)]
if segments:
    view = view[view["segment"].isin(segments)]
if only_contact:
    view = view[view["contact_recommended"]]

econ = economics()
m = st.columns(4)
m[0].metric("Customers in list", f"{len(view):,}", border=True)
m[1].metric("Expected margin loss", money(view["expected_loss"].sum()), border=True)
m[2].metric("Campaign cost", money(len(view) * econ.offer_cost), border=True)
m[3].metric("Expected net value of contacting", money(view["expected_net_gain"].sum()), border=True)

table = view[["customer_id", "churn_probability", "risk_level", "expected_loss",
              "expected_net_gain", "contract", "tenure", "monthly_charges", "reason_1",
              "reason_2", "reason_3", "recommended_action", "segment"]].reset_index()
event = st.dataframe(
    table.drop(columns="index"), hide_index=True, height=360, on_select="rerun",
    selection_mode="single-row", key="targets_table",
    column_config={
        "customer_id": "Customer",
        "churn_probability": st.column_config.ProgressColumn("P(churn)", format="percent",
                                                             min_value=0, max_value=1),
        "risk_level": "Risk",
        "expected_loss": st.column_config.NumberColumn("Expected loss", format="dollar"),
        "expected_net_gain": st.column_config.NumberColumn("Net gain if contacted", format="dollar"),
        "contract": "Contract", "tenure": "Tenure (mo)",
        "monthly_charges": st.column_config.NumberColumn("Monthly bill", format="dollar"),
        "reason_1": "Reason 1", "reason_2": "Reason 2", "reason_3": "Reason 3",
        "recommended_action": "Recommended action", "segment": "Segment",
    })
st.download_button("Download this list (CSV)", table.drop(columns="index").to_csv(index=False),
                   file_name="retention_targets.csv", mime="text/csv", icon=":material/download:")

if table.empty:
    st.stop()

# --------------------------------------------------------------------------- #
# Customer detail
# --------------------------------------------------------------------------- #
st.divider()
selected_rows = event.selection.rows if event and event.selection else []
default_id = table.loc[selected_rows[0], "customer_id"] if selected_rows else table["customer_id"].iloc[0]
ids = table["customer_id"].tolist()
customer_id = st.selectbox("Customer detail (or click a row above)", ids, index=ids.index(default_id))
idx = table.loc[table["customer_id"] == customer_id, "index"].iloc[0]
row = active.loc[idx]
shap_row = shap_values().loc[idx]
explanation = explain_customer(shap_row, row, bundle()["reference"])

st.subheader(f"{customer_id}  {risk_badge(row['risk_level'])}")
left, right = st.columns([2, 3])
with left:
    a, b = st.columns(2)
    a.metric("Churn probability", f"{row['churn_probability']:.1%}")
    b.metric("Expected loss", money(row["expected_loss"]))
    st.markdown(md(
        f"**{row['contract']}** contract, **{int(row['tenure'])} month"
        f"{'s' if int(row['tenure']) != 1 else ''}** tenure, "
        f"**${row['monthly_charges']:,.2f}/month**, {row['internet_service']} internet, "
        f"pays by {row['payment_method'].lower()}. Segment: *{row['segment']}*."))
    st.markdown("**Why they are at risk**")
    st.markdown(md("\n".join(f"{i}. {f['text']}"
                             for i, f in enumerate(explanation["risk_factors"], 1))))
    if explanation["protective_factors"]:
        st.markdown("**What keeps them**")
        st.markdown(md("\n".join(f"- {f['text']}" for f in explanation["protective_factors"])))
    st.markdown("**Recommended action**")
    for a_ in explanation["actions"]:
        st.markdown(f"- {a_}")

with right:
    top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(10)[::-1]
    labels = [f"{FEATURE_LABELS[f]} = {row[f]:.2f}" if isinstance(row[f], float)
              else f"{FEATURE_LABELS[f]} = {row[f]}" for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=[RISK_UP if v > 0 else RISK_DOWN for v in top.values],
        hovertemplate="%{y}<br>impact %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title_text="SHAP impact (red = pushes towards churn)")
    chart(style(fig, 400, "What drives this prediction", legend=False))

# --------------------------------------------------------------------------- #
# LLM brief
# --------------------------------------------------------------------------- #
st.markdown("#### Retention brief")


@st.cache_data(show_spinner=False)
def _brief(ctx_json: str, use_llm: bool) -> dict:
    return generate_brief(json.loads(ctx_json), use_llm=use_llm)


ctx = build_context(row, explanation)
ctx_json = json.dumps(ctx, sort_keys=True)
if llm_available():
    if st.button("Write brief with Claude", icon=":material/auto_awesome:", type="primary"):
        st.session_state[f"brief_{customer_id}"] = True
    use_llm = st.session_state.get(f"brief_{customer_id}", False)
else:
    use_llm = False
    st.caption("No ANTHROPIC_API_KEY configured - showing the deterministic template. "
               "Add the key to generate the brief with Claude.")

with st.spinner("Writing brief..."):
    brief = _brief(ctx_json, use_llm)
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**For the account manager**")
    st.markdown(md(brief["summary"]))
with c2.container(border=True):
    st.markdown("**Message to the customer**")
    st.markdown(md(brief["customer_message"]))
st.caption(f"Source: {brief['source']}. The brief uses only the facts below - the model's "
           "probability, SHAP reasons and approved playbook actions.")
with st.expander("Facts sent to the LLM"):
    st.json(ctx)
