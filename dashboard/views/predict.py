import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK_3, RISK_DOWN, RISK_UP, bundle, chart, explainer, gauge, md, money,
                    money_short, risk_badge, style)
from src import config
from src.data_preprocessing import load_raw_csv, prepare_features
from src.explain import explain_customer
from src.monitoring import drift_report, overall_status
from src.predict import OUTPUT_COLUMNS, score_features, shap_frame
from theme import callout, hero, section, status_chip

b = bundle()
trained = pd.Timestamp(b["trained_at"]).strftime("%Y-%m-%d %H:%M UTC")
hero("Score new customers",
     f"Upload a customer file or build a single profile. {b['model_name']} (trained {trained}) "
     "scores it through the same SQL cleaning and feature logic as the training data, explains "
     "every score and checks whether the new data still looks like what the model learned from.",
     eyebrow="Real-time scoring",
     chips=[f"Contact threshold <b>{b['threshold']:.2f}</b>", "SHAP reasons per row",
            "Data-drift check (PSI)", "What-if simulator"])


@st.cache_data(show_spinner=False)
def reference_features() -> pd.DataFrame:
    return pd.read_csv(config.FEATURES_CSV)

tab_upload, tab_single = st.tabs([":material/upload_file: Upload a file",
                                  ":material/person: Single customer (what-if)"])

with tab_upload:
    sample = load_raw_csv().drop(columns="Churn").sample(25, random_state=7)
    st.download_button("Download a sample file (25 customers, IBM Telco format)",
                       sample.to_csv(index=False), file_name="sample_customers.csv",
                       mime="text/csv", icon=":material/download:")
    upload = st.file_uploader("CSV with the original IBM Telco columns (Churn column optional)",
                              type="csv")
    if upload is not None:
        raw = pd.read_csv(upload, dtype=str, keep_default_na=False)
        try:
            features = prepare_features(raw)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        with st.spinner(f"Scoring {len(features):,} customers..."):
            scored = score_features(features, b, explainer=explainer())
        drift = drift_report(reference_features(), features)
        health = overall_status(drift)
        m = st.columns(5)
        m[0].metric("Customers scored", f"{len(scored):,}", border=True)
        m[1].metric("To contact", f"{int(scored['contact_recommended'].sum()):,}", border=True)
        m[2].metric("High risk", f"{int((scored['risk_level'] == 'HIGH').sum()):,}", border=True)
        m[3].metric("Expected loss", money_short(scored["expected_loss"].sum()), border=True)
        m[4].metric("Data health", health, border=True,
                    help="Population Stability Index of every model feature vs. the training data. "
                         "Stable < 0.10 <= Watch < 0.25 <= Shifted.")

        shifted = drift[drift["status"] != "Stable"]
        if health == "Stable":
            callout("<b>Data health: stable.</b> Every feature in this file is distributed like the "
                    "training data, so the scores are as reliable as the test-set metrics.", icon="✅")
        else:
            names = ", ".join(shifted["label"].head(5))
            callout(f"<b>Data health: {health.lower()}.</b> {len(shifted)} feature(s) differ from "
                    f"the training data ({names}). Scores for this file are less certain; if this "
                    "is the new normal, retrain with <code>python -m src.pipeline</code>.", icon="⚠️")
        with st.expander(":material/monitoring: Drift report (PSI per feature)"):
            st.markdown(" ".join(f"{status_chip(s)} {n}" for s, n in
                                 drift["status"].value_counts().items()), unsafe_allow_html=True)
            st.dataframe(drift[["label", "type", "psi", "status", "reference", "new_file"]],
                         hide_index=True, column_config={
                             "label": "Feature", "psi": st.column_config.ProgressColumn(
                                 "PSI", format="%.3f", min_value=0, max_value=0.5),
                             "reference": "Training data", "new_file": "This file"})

        section("Scores")
        st.dataframe(scored[OUTPUT_COLUMNS], hide_index=True, column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "P(churn)", format="percent", min_value=0, max_value=1),
            "expected_loss": st.column_config.NumberColumn("Expected loss", format="dollar"),
            "expected_net_gain": st.column_config.NumberColumn("Net gain if contacted",
                                                               format="dollar"),
            "monthly_charges": st.column_config.NumberColumn("Monthly bill", format="dollar")})
        st.download_button("Download scores (CSV)", scored[OUTPUT_COLUMNS].to_csv(index=False),
                           file_name="churn_scores.csv", mime="text/csv",
                           icon=":material/download:", type="primary")

with tab_single:
    with st.form("what_if"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 4)
        monthly = c1.number_input("Monthly charges ($)", 18.0, 120.0, 89.5, 0.5)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = c1.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)",
                                                  "Credit card (automatic)"])
        internet = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = c2.toggle("Online security")
        support = c2.toggle("Tech support")
        backup = c2.toggle("Online backup")
        protection = c2.toggle("Device protection")
        tv = c2.toggle("Streaming TV", value=True)
        movies = c2.toggle("Streaming movies")
        phone = c3.toggle("Phone service", value=True)
        lines = c3.toggle("Multiple lines")
        paperless = c3.toggle("Paperless billing", value=True)
        senior = c3.toggle("Senior citizen")
        partner = c3.toggle("Partner")
        dependents = c3.toggle("Dependents")
        gender = c3.radio("Gender", ["Female", "Male"], horizontal=True)
        submitted = st.form_submit_button("Score this customer", type="primary")

    yn = lambda v: "Yes" if v else "No"  # noqa: E731
    net = internet != "No"
    raw = pd.DataFrame([{
        "customerID": "WHAT-IF", "gender": gender, "SeniorCitizen": str(int(senior)),
        "Partner": yn(partner), "Dependents": yn(dependents), "tenure": str(tenure),
        "PhoneService": yn(phone),
        "MultipleLines": yn(lines and phone) if phone else "No phone service",
        "InternetService": internet,
        **{k: (yn(v) if net else "No internet service") for k, v in (
            ("OnlineSecurity", security), ("OnlineBackup", backup),
            ("DeviceProtection", protection), ("TechSupport", support),
            ("StreamingTV", tv), ("StreamingMovies", movies))},
        "Contract": contract, "PaperlessBilling": yn(paperless), "PaymentMethod": payment,
        "MonthlyCharges": f"{monthly:.2f}", "TotalCharges": f"{monthly * tenure:.2f}",
    }])
    scored = score_features(prepare_features(raw), b, explainer=explainer())
    row = scored.iloc[0]
    shap_row = shap_frame(scored).iloc[0]
    exp = explain_customer(shap_row, row, b["reference"])

    g, left, right = st.columns([1.2, 1.6, 2.2])
    with g:
        st.plotly_chart(gauge(row["churn_probability"], b["threshold"]), width="stretch",
                        theme=None, config={"displayModeBar": False})
    with left:
        st.markdown(f"#### {risk_badge(row['risk_level'])} risk")
        st.markdown(md(f"Expected margin loss **{money(row['expected_loss'])}** - "
                       + ("**contact recommended**" if row["contact_recommended"]
                          else "below the contact threshold")))
        st.markdown("**Risk factors**")
        st.markdown(md("\n".join(f"- {f['text']}" for f in exp["risk_factors"])))
        if exp["protective_factors"]:
            st.markdown("**Protective factors**")
            st.markdown(md("\n".join(f"- {f['text']}" for f in exp["protective_factors"])))
        st.markdown("**Recommended action:** " + "; ".join(exp["actions"]))
    with right:
        top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig = go.Figure(go.Bar(
            x=top.values, y=[config.FEATURE_LABELS[f] for f in top.index], orientation="h",
            marker_color=[RISK_UP if v > 0 else RISK_DOWN for v in top.values],
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color=INK_3, line_width=1)
        fig.update_xaxes(title_text="SHAP impact (red = raises risk)")
        chart(style(fig, 380, "What drives this prediction", legend=False))
    st.caption("Try switching the contract to 'Two year' or adding Tech support to see how the "
               "probability and the reasons move.")
