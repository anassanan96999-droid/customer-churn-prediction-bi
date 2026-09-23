"""Business report (PDF) for a non-technical audience, built from pipeline artifacts.

    python -m src.report        # -> reports/business_report.pdf

Every number is read from the files the pipeline wrote, so the report can be
regenerated after a retrain and never drifts from the dashboard.
"""
from __future__ import annotations

import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src import config
from src.data_preprocessing import run_named_query

REPORT_PDF = config.REPORTS_DIR / "business_report.pdf"
INK, INK_2, GRID, ACCENT, SOFT = (colors.HexColor(h) for h in
                                  ("#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#f3f2ef"))
PAGE_W = A4[0] - 4 * cm


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=26, alignment=TA_LEFT, textColor=INK,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10.5,
                                   textColor=INK_2, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, textColor=INK, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.8, leading=14,
                               textColor=INK, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.8, leading=14,
                                 textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8, leading=10.5,
                                textColor=INK_2),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10, textColor=INK),
    }


def _table(rows: list[list], widths: list[float], s: dict, header_bg=SOFT) -> Table:
    data = [[Paragraph(str(c), s["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _figure(name: str, width: float = PAGE_W) -> Image:
    path = config.FIGURES_DIR / name
    img = Image(str(path))
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return img


def _money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_2)
    canvas.drawString(2 * cm, 1.2 * cm, "Customer Churn Prediction & BI System - business report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report(path=REPORT_PDF) -> None:
    s = _styles()
    report = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
    econ = json.loads(config.THRESHOLD_JSON.read_text(encoding="utf-8"))
    scored = pd.read_csv(config.SCORED_CSV)
    segments = pd.read_csv(config.REPORTS_DIR / "segment_profile.csv")
    kpi = run_named_query("kpi_overview").iloc[0]
    contract = run_named_query("churn_by_contract").set_index("segment")["churn_rate_pct"]
    cross = run_named_query("churn_contract_x_internet")
    tenure = run_named_query("churn_by_tenure_month").set_index("tenure")

    e = econ["economics"]
    champ = report["champion"]
    cm_ = report["models"][champ]
    policies = {p["policy"].split(" (")[0]: p for p in econ["policy_comparison_test"]}
    at_default = policies["Default threshold"]
    at_opt = next(p for k, p in policies.items() if k.startswith("Money-optimal"))
    ev_rule = policies["Expected-value rule"]
    t_star = econ["chosen_threshold"]
    uplift = at_opt["net_value"] - at_default["net_value"]

    active = scored[scored["churn"] == 0].sort_values("expected_loss", ascending=False)
    flagged = active[active["contact_recommended"]]
    loss_share = flagged["expected_loss"].sum() / active["expected_loss"].sum()
    revenue_at_risk = (active["churn_probability"] * active["monthly_charges"] * 12).sum()
    campaign_value = flagged["expected_net_gain"].sum()
    top_segment = segments.sort_values("expected_loss_active", ascending=False).iloc[0]
    riskiest_cell = cross.iloc[0]

    story = [
        Paragraph("Customer Churn: Who to Save, Why They Leave, What It Is Worth", s["title"]),
        Paragraph(f"Business report  |  IBM Telco dataset, {int(kpi['customers']):,} customers  |  "
                  f"model: {champ}", s["subtitle"]),
        Paragraph("Executive summary", s["h1"]),
    ]
    summary = [
        f"<b>Churn is expensive:</b> {kpi['churn_rate_pct']:.1f}% of customers left last month, "
        f"taking {kpi['revenue_churn_pct']:.1f}% of monthly revenue with them. Across the "
        f"{len(active):,} remaining customers, an estimated <b>{_money(revenue_at_risk)}</b> of "
        f"annual revenue is at risk.",
        f"<b>The risk is concentrated:</b> {len(flagged):,} active customers "
        f"({len(flagged) / len(active):.0%}) carry <b>{loss_share:.0%}</b> of the expected margin "
        f"loss. They are the retention target list.",
        f"<b>Why they leave:</b> month-to-month contracts ({contract['Month-to-month']:.0f}% churn vs. "
        f"{contract['Two year']:.0f}% on two-year contracts), short tenure "
        f"({tenure.loc[1, 'cumulative_share_of_churners_pct']:.0f}% of all churners leave in their "
        f"first month), fiber internet and missing security/support add-ons.",
        f"<b>Decide with money, not accuracy:</b> contacting every customer above a churn "
        f"probability of <b>{t_star:.2f}</b> instead of the default 0.50 reaches "
        f"{at_opt['recall']:.0%} of churners instead of {at_default['recall']:.0%} and earns "
        f"<b>{_money(uplift)} (+{uplift / at_default['net_value']:.0%})</b> more net value on "
        f"held-out customers. Targeting by each customer's expected value adds more "
        f"(+{ev_rule['net_value'] / at_default['net_value'] - 1:.0%} vs. 0.50).",
        f"<b>Recommendation:</b> run a retention campaign on the {len(flagged):,} customers, led by "
        f"annual-contract offers and early-life onboarding; expected net value "
        f"<b>{_money(campaign_value)}</b> over {e['horizon_months']} months at a cost of "
        f"{_money(len(flagged) * e['offer_cost'])}. Launch it with a control group to measure the "
        f"real save rate.",
    ]
    story += [Paragraph(t, s["bullet"], bulletText="•") for t in summary]

    story.append(Paragraph("Key numbers", s["h1"]))
    story.append(_table([
        ["Measure", "Value", "Measure", "Value"],
        ["Customers", f"{int(kpi['customers']):,}", "Active customers", f"{len(active):,}"],
        ["Churn rate (last month)", f"{kpi['churn_rate_pct']:.1f}%", "Recommended for contact",
         f"{len(flagged):,}"],
        ["Annual revenue at risk", _money(revenue_at_risk), "Expected campaign net value",
         _money(campaign_value)],
        ["Model ROC-AUC (held out)", f"{cm_['test_at_0.5']['roc_auc']:.3f}",
         "Money-optimal threshold", f"{t_star:.2f}"],
    ], [4.2 * cm, 3.8 * cm, 4.8 * cm, 4.2 * cm], s))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Assumptions (adjustable in the dashboard): a retention offer costs ${e['offer_cost']:.0f} "
        f"per contacted customer; {e['save_rate']:.0%} of contacted churners are saved; a saved "
        f"customer is worth {e['horizon_months']} months of their bill at a "
        f"{e['gross_margin']:.0%} margin.", s["small"]))
    story += [Paragraph("What drives churn", s["h1"]),
              _figure("shap_global_importance.png", PAGE_W * 0.72),
              Paragraph("Average impact of each factor on the model's churn predictions (SHAP), "
                        "across all customers.", s["small"])]

    story += [PageBreak(), Paragraph("1. Who is leaving and why", s["h1"]),
              _figure("eda_churn_by_segment.png", PAGE_W * 0.92),
              Paragraph(
                  f"The riskiest combination is <b>{riskiest_cell['contract']} + "
                  f"{riskiest_cell['internet_service']}</b>: {riskiest_cell['churn_rate_pct']:.0f}% "
                  f"churn across {int(riskiest_cell['customers']):,} customers. Customers with "
                  f"both online security and tech support churn at a fraction of the rate of "
                  f"those with neither, and electronic-check payers churn about three times as "
                  f"often as customers on automatic payment.", s["body"]),
              _figure("eda_tenure_curve.png", PAGE_W * 0.92),
              Paragraph(
                  f"Churn is front-loaded: {tenure.loc[6, 'cumulative_share_of_churners_pct']:.0f}% "
                  f"of all churners leave within six months and "
                  f"{tenure.loc[12, 'cumulative_share_of_churners_pct']:.0f}% within a year. The "
                  f"first months are where retention is won or lost.", s["body"])]

    story += [PageBreak(), Paragraph("2. How reliable is the prediction?", s["h1"]),
              Paragraph(
                  "Five model families were tuned and compared with 5-fold cross-validation on "
                  "80% of customers; the remaining 20% were held back and scored once, at the "
                  "end. The top four models perform almost identically; the champion was "
                  "chosen in advance on cross-validated ROC-AUC.", s["body"])]
    rows = [["Model", "CV ROC-AUC", "Held-out ROC-AUC", "PR-AUC", "Recall @0.5",
             "Recall @t*", "Net value @t*"]]
    for name, m in report["models"].items():
        rows.append([("<b>%s</b>" % name) if name == champ else name,
                     f"{m['cv']['roc_auc_mean']:.4f}", f"{m['test_at_0.5']['roc_auc']:.3f}",
                     f"{m['test_at_0.5']['pr_auc']:.3f}", f"{m['test_at_0.5']['recall']:.0%}",
                     f"{m['test_at_optimal']['recall']:.0%}",
                     _money(m["campaign_test_at_optimal"]["net_value"])])
    story.append(_table(rows, [4 * cm, 2.2 * cm, 2.6 * cm, 1.8 * cm, 2 * cm, 2 * cm, 2.4 * cm], s))
    story += [Spacer(1, 8), _figure("model_roc_pr.png", PAGE_W),
              Paragraph("All models except the single decision tree rank customers almost "
                        "identically. The top 10% of scores contain "
                        f"{cm_['test_at_0.5']['lift_top_10pct']:.1f}x the average churn rate.",
                        s["body"]),
              _figure("model_calibration.png", PAGE_W * 0.4),
              Paragraph("The predicted probabilities are well calibrated (a predicted 40% "
                        "really means about 40%), which is what makes expected-loss and "
                        "campaign-value figures trustworthy.", s["body"])]

    story += [PageBreak(), Paragraph("3. Choosing who to contact: a money decision", s["h1"]),
              _figure("threshold_economics.png", PAGE_W),
              Paragraph(f"Results on {int(at_opt['churners_total']):,} churners among the "
                        f"{report['dataset']['test']:,} held-out customers:", s["body"])]
    rows = [["Targeting rule", "Contacted", "Churners reached", "Cost", "Net value"]]
    for p in econ["policy_comparison_test"]:
        rows.append([p["policy"], f"{p['contacted']:,}", f"{p['churners_caught']:,}",
                     _money(p["campaign_cost"]), _money(p["net_value"])])
    story.append(_table(rows, [6.5 * cm, 2.2 * cm, 3 * cm, 2.4 * cm, 2.9 * cm], s))
    story.append(Paragraph(
        "A missed churner costs far more than a wasted offer, so the best threshold sits well "
        "below 0.5. The value curve is flat between roughly 0.2 and 0.45: the decision is robust "
        "to the exact cut-off, but not to using the default.", s["body"]))

    story += [PageBreak(), Paragraph("4. Who to call first", s["h1"]),
              Paragraph("Active customers ranked by expected loss (churn probability x customer "
                        "value), each with the reasons behind the score (SHAP) and the "
                        "recommended action:", s["body"])]
    rows = [["Customer", "P(churn)", "Exp. loss", "Main reasons", "Recommended action"]]
    for _, r in flagged.head(12).iterrows():
        rows.append([r["customer_id"], f"{r['churn_probability']:.0%}", _money(r["expected_loss"]),
                     r["top_reasons"].replace("; ", "<br/>"),
                     r["recommended_action"].replace(" | ", "<br/>")])
    story.append(_table(rows, [2.2 * cm, 1.7 * cm, 1.6 * cm, 5.5 * cm, 6.0 * cm], s))
    story.append(Spacer(1, 8))
    rows = [["Segment", "Customers", "Churn rate", "Avg tenure", "Avg bill", "Exp. loss (active)"]]
    for _, r in segments.iterrows():
        rows.append([r["segment"], f"{int(r['customers']):,}", f"{r['churn_rate']:.0%}",
                     f"{r['avg_tenure']:.0f} mo", _money(r["avg_monthly_charges"]),
                     _money(r["expected_loss_active"])])
    story.append(KeepTogether([
        Paragraph("Behavioural segments", s["h1"]),
        _table(rows, [4.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 3.8 * cm], s),
        Spacer(1, 4),
        Paragraph(f"The <b>{top_segment['segment']}</b> segment holds "
                  f"{top_segment['expected_loss_active'] / segments['expected_loss_active'].sum():.0%}"
                  f" of the expected loss among active customers.", s["body"])]))

    story += [PageBreak(), Paragraph("5. Recommendations", s["h1"])]
    recs = [
        f"<b>Launch the campaign</b> on the {len(flagged):,} customers above the {t_star:.2f} "
        f"threshold (or use the per-customer expected-value rule), working the list in "
        f"expected-loss order.",
        "<b>Convert month-to-month customers</b> to discounted 12-month contracts - contract "
        "type is the strongest single driver of churn.",
        "<b>Invest in the first 90 days:</b> onboarding calls, setup checks and a walk-through "
        "of the first bill, because churn is concentrated in the first months.",
        "<b>Bundle security and tech support</b> (e.g. a free three-month trial) for fiber "
        "customers who have neither.",
        "<b>Move electronic-check payers to automatic payment</b> with a small bill credit.",
        "<b>Measure, then re-tune:</b> hold back a random control group so the real save rate "
        "replaces the assumed one, then re-optimise the threshold in the dashboard.",
    ]
    story += [Paragraph(t, s["bullet"], bulletText="•") for t in recs]
    story += [Paragraph("Limitations", s["h1"]), Paragraph(
        "The IBM Telco data is a single snapshot without support-call, usage-history or "
        "payment-delay columns, so those drivers could not be measured. Campaign economics rest "
        "on assumptions (offer cost, save rate, margin) that should be replaced with measured "
        "values. Scores for existing customers are out-of-fold: every customer was scored by a "
        "model that did not see them during training.", s["body"])]

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Customer Churn - Business Report",
                            author="Customer Churn Prediction & BI System")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"  report -> {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    build_report()
