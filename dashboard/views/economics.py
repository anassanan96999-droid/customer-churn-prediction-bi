import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (BAR, INK_3, SERIES, chart, curve_for, economics, load_report,
                    load_threshold_report, money, style, threshold)
from src import config
from src.economics import campaign_outcome, expected_value_target

st.title("Campaign economics")
st.markdown(
    "A churn model outputs a probability; **who to call is a money decision**. Each contact "
    "costs the offer, whether or not the customer would have left. Each churner we reach is "
    "saved with some probability and is worth their margin over the planning horizon. "
    "Change the assumptions below: the contact threshold used across the dashboard is "
    "re-optimised for them.")

econ = economics()
with st.container(border=True):
    c = st.columns(4)
    offer = c[0].slider("Offer cost per contact ($)", 5, 100, int(econ.offer_cost), 5)
    save = c[1].slider("Save rate (share of contacted churners kept)", 0.05, 0.80,
                       float(econ.save_rate), 0.05, format="%.2f")
    horizon = c[2].slider("Value horizon (months)", 3, 36, int(econ.horizon_months), 3)
    margin = c[3].slider("Gross margin", 0.10, 0.80, float(econ.gross_margin), 0.05, format="%.2f")
    new = config.Economics(float(offer), float(save), int(horizon), float(margin))
    if new != econ:
        st.session_state["economics"] = new
        st.rerun()
    if new != config.ECONOMICS and st.button("Reset to defaults", icon=":material/restart_alt:"):
        st.session_state.pop("economics", None)
        st.rerun()

econ, t = economics(), threshold()
curve = curve_for(econ)
at = lambda th: curve.loc[np.isclose(curve["threshold"], th)].iloc[0]  # noqa: E731
best, default = at(t), at(0.5)
uplift = best["net_value"] - default["net_value"]

st.subheader("Backtest on last month's churn (all 7,043 customers, out-of-fold scores)")
m = st.columns(4)
m[0].metric("Money-optimal threshold", f"{t:.2f}", border=True,
            help="The threshold chosen by the pipeline on training data when the assumptions "
                 "are the defaults; re-optimised on all out-of-fold scores otherwise.")
m[1].metric("Net value at this threshold", money(best["net_value"]), border=True,
            delta=f"{money(uplift)} vs. 0.50 threshold")
m[2].metric("Customers contacted", f"{int(best['contacted']):,}", border=True,
            delta=f"{int(best['contacted'] - default['contacted']):+,} vs. 0.50", delta_color="off")
m[3].metric("Churners reached (recall)", f"{best['recall']:.0%}", border=True,
            delta=f"{(best['recall'] - default['recall']) * 100:+.0f} pts vs. 0.50", delta_color="off")

fig = go.Figure()
for col, name, color, width in (("retained_value", "Value retained", SERIES[2], 1.6),
                                ("campaign_cost", "Campaign cost", SERIES[1], 1.6),
                                ("net_value", "Net value", BAR, 3)):
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve[col], name=name, mode="lines",
                             line=dict(color=color, width=width),
                             hovertemplate=f"{name}: %{{y:$,.0f}}<extra></extra>"))
for th, label in ((0.5, "0.50 default"), (t, f"{t:.2f} optimal")):
    fig.add_vline(x=th, line_dash="dot", line_color=INK_3, line_width=1)
    fig.add_annotation(x=th, y=at(th)["net_value"], text=f"{label}<br>{money(at(th)['net_value'])}",
                       showarrow=True, arrowhead=0, ax=40, ay=-40, font=dict(size=11))
fig.update_layout(hovermode="x unified")
fig.update_xaxes(title_text="Contact every customer with P(churn) >= threshold")
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
chart(style(fig, 420, "Campaign value by decision threshold"))

# --------------------------------------------------------------------------- #
st.subheader("Targeting rules compared")
s = pd.read_csv(config.SCORED_CSV)
y, p, mc = s["churn"], s["churn_probability"], s["monthly_charges"]
rules = [("Contact nobody", np.zeros(len(s), bool)), ("Contact everyone", np.ones(len(s), bool)),
         ("Default threshold (0.50)", p >= 0.5), (f"Money-optimal threshold ({t:.2f})", p >= t),
         ("Expected-value rule (per customer)", expected_value_target(p, mc, econ))]
rows = pd.DataFrame([{"Rule": name, **campaign_outcome(y, p, mc, target=mask, econ=econ)}
                     for name, mask in rules])
for col in ("retained_value", "campaign_cost", "net_value"):
    rows[col] = rows[col].map(money)          # whole dollars read better for campaign totals
st.dataframe(
    rows[["Rule", "contacted", "churners_caught", "precision", "recall", "retained_value",
          "campaign_cost", "net_value"]],
    hide_index=True,
    column_config={
        "contacted": st.column_config.NumberColumn("Contacted", format="localized"),
        "churners_caught": st.column_config.NumberColumn("Churners reached", format="localized"),
        "precision": st.column_config.NumberColumn("Precision", format="percent"),
        "recall": st.column_config.NumberColumn("Recall", format="percent"),
        "retained_value": "Value retained", "campaign_cost": "Cost", "net_value": "Net value",
    })
st.caption("The expected-value rule contacts a customer when P(churn) x save rate x value > "
           "offer cost, so a high-value customer qualifies at a lower probability than a "
           "low-value one. It relies on calibrated probabilities - see Model performance.")

with st.expander("Held-out test result from the pipeline (default assumptions)"):
    rep = load_threshold_report()
    test = pd.DataFrame(rep["curve_test"]).iloc[0]
    n_test = load_report()["dataset"]["test"]
    st.markdown(f"Threshold **{rep['chosen_threshold']:.2f}** was chosen on {rep['chosen_on']} "
                f"and then applied, unchanged, to the untouched test set "
                f"({int(test['churners_total']):,} churners among {n_test:,} customers):")
    held_out = pd.DataFrame(rep["policy_comparison_test"])[
        ["policy", "contacted", "churners_caught", "precision", "recall", "net_value"]]
    held_out["net_value"] = held_out["net_value"].map(money)
    st.dataframe(held_out, hide_index=True,
                 column_config={"precision": st.column_config.NumberColumn(format="percent"),
                                "recall": st.column_config.NumberColumn(format="percent"),
                                "net_value": "Net value"})
