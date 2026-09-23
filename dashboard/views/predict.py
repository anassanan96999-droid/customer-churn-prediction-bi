import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK_3, RISK_DOWN, RISK_UP, bundle, chart, explainer, md, money,
                    risk_badge, style)
from src import config
from src.data_preprocessing import load_raw_csv, prepare_features
from src.explain import explain_customer
from src.predict import OUTPUT_COLUMNS, score_features, shap_frame

st.title("Score new customers")
b = bundle()
trained = pd.Timestamp(b["trained_at"]).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Model: {b['model_name']} trained on all 7,043 customers ({trained}). "
           f"Contact threshold {b['threshold']:.2f}. Uploaded files go through the same SQL "
           "cleaning and feature logic as the training data.")

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
        m = st.columns(4)
        m[0].metric("Customers scored", f"{len(scored):,}", border=True)
        m[1].metric("Recommended for contact", f"{int(scored['contact_recommended'].sum()):,}",
                    border=True)
        m[2].metric("High risk", f"{int((scored['risk_level'] == 'HIGH').sum()):,}", border=True)
        m[3].metric("Expected margin loss", money(scored["expected_loss"].sum()), border=True)
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

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"### {row['churn_probability']:.0%} churn probability  "
                    f"{risk_badge(row['risk_level'])}")
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
        fig.update_xaxes(title_text="SHAP impact (red = pushes towards churn)")
        chart(style(fig, 380, "What drives this prediction", legend=False))
    st.caption("Try switching the contract to 'Two year' or adding Tech support to see how the "
               "probability and the reasons move.")
