import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK_3, SERIES, bundle, chart, economics, load_scored, money, money_short,
                    style, with_economics)
from src import config
from src.scenarios import INTERVENTIONS, simulate
from theme import callout, hero, section

hero("Scenario Lab",
     "Test a retention strategy before you spend on it. Pick who to target and what to offer; "
     "the model re-scores every customer with the new profile and prices the result.",
     eyebrow="Model-based what-if simulator",
     chips=["Re-scores with the live model", "Churners avoided", "Margin retained", "ROI"])

scored = with_economics(load_scored())
active = scored[scored["churn"] == 0]
econ = economics()

AUDIENCES = {
    "Contact list (above threshold)": lambda d: d[d["contact_recommended"]],
    "HIGH risk only": lambda d: d[d["risk_level"] == "HIGH"],
    "All active customers": lambda d: d,
}

with st.container(key="glass_controls"):
    c1, c2, c3 = st.columns([1.3, 1.6, 1])
    audience_name = c1.selectbox("Who gets the offer", list(AUDIENCES))
    segments = c1.multiselect("Limit to segments", sorted(active["segment"].unique()))
    keys = c2.multiselect(
        "What to offer (combine freely)", list(INTERVENTIONS),
        default=["annual_contract"], format_func=lambda k: INTERVENTIONS[k].label)
    acceptance = c3.slider("Acceptance rate", 0.05, 1.0, 0.30, 0.05, format="%.2f",
                           help="Share of eligible customers who take up the offer.")

    params: dict = {}
    if keys:
        pcols = st.columns(len(keys))
        for col, key in zip(pcols, keys):
            iv = INTERVENTIONS[key]
            if key == "price_discount":
                params["discount"] = col.slider(iv.cost_label, 0.05, 0.40, 0.10, 0.05,
                                                format="%.2f", key=f"p_{key}")
            elif key in ("annual_contract", "two_year_contract"):
                params[key] = col.slider(f"{iv.label}: {iv.cost_label.lower()}", 0.0, 3.0,
                                         iv.default_cost, 0.5, key=f"p_{key}")
            else:
                params[key] = col.slider(f"{iv.label}: {iv.cost_label.lower()}", 0.0, 150.0,
                                         iv.default_cost, 5.0, key=f"p_{key}")

audience = AUDIENCES[audience_name](active)
if segments:
    audience = audience[audience["segment"].isin(segments)]
if not keys or audience.empty:
    st.info("Choose at least one offer and a non-empty audience to run a scenario.")
    st.stop()


@st.cache_data(show_spinner="Re-scoring customers...")
def run(ids: tuple, keys: tuple, params: tuple, acceptance: float, econ_values: tuple):
    rows = load_scored().set_index(config.ID_COL).loc[list(ids)].reset_index()
    return simulate(rows, bundle()["pipeline"], list(keys), acceptance, dict(params),
                    config.Economics(*econ_values))


econ_values = (econ.offer_cost, econ.save_rate, econ.horizon_months, econ.gross_margin)
summary, detail = run(tuple(audience[config.ID_COL]), tuple(keys), tuple(sorted(params.items())),
                      acceptance, econ_values)

m = st.columns(3)
m[0].metric("Customers offered", f"{summary['eligible']:,}", border=True,
            help=f"Eligible for at least one offer, out of {summary['customers']:,} in the audience.")
m[1].metric("Expected take-up", f"{summary['accepting']:,.0f}", border=True)
m[2].metric("Churners avoided", f"{summary['churners_avoided']:,.1f}", border=True,
            delta=f"{-summary['churners_avoided'] / max(summary['churners_before'], 1e-9):.1%} churn",
            delta_color="inverse")
m = st.columns(3)
m[0].metric("Margin retained", money_short(summary["margin_retained"]), border=True,
            help=money(summary["margin_retained"]))
m[1].metric("Offer cost", money_short(summary["offer_cost"]), border=True,
            help=money(summary["offer_cost"]))
m[2].metric("Net value", money_short(summary["net_value"]), border=True,
            help=money(summary["net_value"]),
            delta=f"{summary['roi']:+.0%} ROI" if summary["offer_cost"] else None)

if summary["eligible"]:
    callout(f"Among the <b>{summary['eligible']:,}</b> eligible customers the model's average churn "
            f"risk moves from <b>{summary['avg_risk_before']:.0%}</b> to "
            f"<b>{summary['avg_risk_after']:.0%}</b> if they accept. At a {acceptance:.0%} take-up "
            f"this avoids about <b>{summary['churners_avoided']:,.0f}</b> departures and is worth "
            f"<b>{money(summary['net_value'])}</b> net over {econ.horizon_months} months.", icon="🧪")

left, right = st.columns(2)
with left:
    fig = go.Figure(go.Waterfall(
        x=["Expected churners today", "Avoided by the offer", "Expected churners after"],
        measure=["absolute", "relative", "total"],
        y=[summary["churners_before"], -summary["churners_avoided"], 0],
        text=[f"{summary['churners_before']:,.0f}", f"-{summary['churners_avoided']:,.1f}",
              f"{summary['churners_after']:,.0f}"],
        textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color=INK_3, width=1)),
        decreasing=dict(marker=dict(color=SERIES[2])),
        totals=dict(marker=dict(color=SERIES[0])), increasing=dict(marker=dict(color=SERIES[1]))))
    fig.update_yaxes(range=[0, summary["churners_before"] * 1.18])
    chart(style(fig, 360, "Expected churners in the audience", legend=False))
with right:
    bins = [i / 10 for i in range(11)]
    elig = detail[detail["eligible"]]
    before = pd.cut(elig["p_before"], bins, include_lowest=True).value_counts(sort=False)
    after = pd.cut(elig["p_after"], bins, include_lowest=True).value_counts(sort=False)
    labels = [f"{int(b * 100)}-{int((b + .1) * 100)}%" for b in bins[:-1]]
    fig = go.Figure([
        go.Bar(x=labels, y=before.values, name="Today", marker_color=SERIES[0]),
        go.Bar(x=labels, y=after.values, name="If they accept", marker_color=SERIES[2]),
    ])
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(title_text="Churn probability")
    fig.update_yaxes(title_text="Customers")
    chart(style(fig, 360, "Risk distribution of eligible customers"))

section("Strategy comparison", f"every offer on its own, same audience, {acceptance:.0%} take-up")
compare = []
for key, iv in INTERVENTIONS.items():
    s, _ = run(tuple(audience[config.ID_COL]), (key,), tuple(sorted(params.items())),
               acceptance, econ_values)
    compare.append({"Offer": iv.label, "Eligible": s["eligible"],
                    "Churners avoided": s["churners_avoided"], "Margin retained": s["margin_retained"],
                    "Cost": s["offer_cost"], "Net value": s["net_value"],
                    "ROI": s["roi"] if s["offer_cost"] else None})
compare = pd.DataFrame(compare).sort_values("Net value", ascending=False, ignore_index=True)
left, right = st.columns([3, 2])
with left:
    fig = go.Figure(go.Bar(
        x=compare["Net value"], y=compare["Offer"], orientation="h",
        marker_color=[SERIES[2] if v >= 0 else SERIES[7] for v in compare["Net value"]],
        text=[money(v) for v in compare["Net value"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:$,.0f} net<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    lo, hi = min(compare["Net value"].min(), 0), max(compare["Net value"].max(), 0)
    pad = 0.35 * max(hi - lo, 1)          # room for the outside labels on both ends
    fig.update_xaxes(tickprefix="$", tickformat=",.0f",
                     range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else pad * 0.2)])
    chart(style(fig, 320, "Net value by offer", legend=False))
with right:
    st.dataframe(compare, hide_index=True, column_config={
        "Churners avoided": st.column_config.NumberColumn(format="%.1f"),
        "Margin retained": st.column_config.NumberColumn(format="dollar"),
        "Cost": st.column_config.NumberColumn(format="dollar"),
        "Net value": st.column_config.NumberColumn(format="dollar"),
        "ROI": st.column_config.NumberColumn(format="percent")})

section("Customers who respond most", "largest drop in modelled churn risk")
top = (detail[detail["eligible"]].sort_values("risk_drop", ascending=False).head(15)
       .merge(scored[[config.ID_COL, "contract", "tenure", "monthly_charges", "reason_1"]],
              on=config.ID_COL))
st.dataframe(top[[config.ID_COL, "p_before", "p_after", "risk_drop", "value", "contract",
                  "tenure", "monthly_charges", "reason_1"]], hide_index=True, column_config={
    "customer_id": "Customer",
    "p_before": st.column_config.ProgressColumn("Risk today", format="percent", min_value=0, max_value=1),
    "p_after": st.column_config.ProgressColumn("Risk if accepted", format="percent", min_value=0, max_value=1),
    "risk_drop": st.column_config.NumberColumn("Risk drop", format="percent"),
    "value": st.column_config.NumberColumn("Customer value", format="dollar"),
    "monthly_charges": st.column_config.NumberColumn("Monthly bill", format="dollar"),
    "tenure": "Tenure (mo)", "contract": "Contract", "reason_1": "Main reason"})

st.caption("How to read this: the model estimates how churn risk changes when a customer's "
           "profile changes, based on how similar customers behave today. That is correlation, "
           "not a proven causal effect - run the winning offer as an A/B test with a control group "
           "before rolling it out. Offer costs apply to every customer who accepts.")
