"""Turn a customer's SHAP reasons into plain English with Claude.

The ML model decides *who* is at risk and *why* (SHAP). The LLM only rewrites
those facts for two audiences:

* an account-manager brief ("why is this customer at risk, what do I do?")
* a short retention message to the customer, built on an approved playbook action

Guardrails: the prompt carries only structured facts and playbook actions, and
the model is told not to invent offers. Without an API key (or on any API
error) a deterministic template produces the same two fields, so the dashboard
never breaks on a fresh clone.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from pydantic import BaseModel, Field

from src import config
from src.explain import DEFAULT_ACTION


class RetentionBrief(BaseModel):
    summary: str = Field(description="2-3 sentences for the account manager: why this "
                                     "customer is at risk and what to do first.")
    customer_message: str = Field(description="Friendly message to the customer, at most 80 "
                                              "words, built around the recommended action.")


SYSTEM_PROMPT = """You write retention briefs for the customer-success team of a telecom company.

You receive facts about one customer from a churn model: the churn probability, the \
main factors pushing the customer towards leaving (from SHAP), factors keeping them, \
and the retention actions the company has approved for this customer.

Write two things:
1. summary - for the account manager. Plain English, 2-3 sentences. Say how at-risk \
the customer is, the main reasons in business terms, and which approved action to lead with.
2. customer_message - to the customer. Warm, specific, at most 80 words. Offer only the \
approved actions. Never mention churn, risk scores, models or data analysis, and never \
promise a price, discount size or date that is not in the facts.

Use only the facts provided."""


def build_context(row: pd.Series, explanation: dict) -> dict:
    """The facts the LLM is allowed to use for this customer."""
    return {
        "customer_id": row[config.ID_COL],
        "churn_probability": f"{row['churn_probability']:.0%}",
        "risk_level": row["risk_level"],
        "expected_margin_loss_next_12_months": f"${row['expected_loss']:,.0f}",
        "tenure_months": int(row["tenure"]),
        "contract": row["contract"],
        "monthly_charges": f"${row['monthly_charges']:,.2f}",
        "internet_service": row["internet_service"],
        "payment_method": row["payment_method"],
        "risk_factors": [f["text"] for f in explanation["risk_factors"]],
        "protective_factors": [f["text"] for f in explanation["protective_factors"]],
        "approved_actions": explanation["actions"],
    }


# --------------------------------------------------------------------------- #
# Template fallback
# --------------------------------------------------------------------------- #
_CUSTOMER_OFFERS = {
    "Offer a discounted 12-month contract":
        "we can lock in a lower monthly price for you on a 12-month plan",
    "Early-life onboarding call: check the setup, walk through the bill":
        "we'd like to set up a quick call to make sure everything works as it should "
        "and walk you through your bill",
    "Free 3-month trial of Tech Support + Online Security":
        "we'd like to give you three months of Tech Support and Online Security, free of charge",
    "Move to automatic payment with a small bill credit":
        "if you switch to automatic payments, we'll add a small credit to your next bill",
    "Plan review: right-size the bundle or apply a loyalty discount":
        "we'd like to review your plan with you so you only pay for what you actually use",
    "Proactive fiber service check (speed test, outage history)":
        "we'd like to run a quick check on your fiber connection to make sure you get the "
        "speed you pay for",
    DEFAULT_ACTION: "we'd love to hear how things are going and what we could do better",
}


def _join(items: list[str]) -> str:
    items = [i[0].lower() + i[1:] for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def template_brief(ctx: dict) -> dict:
    reasons = ctx["risk_factors"][:3] or ["no single dominant factor"]
    action = ctx["approved_actions"][0] if ctx["approved_actions"] else DEFAULT_ACTION
    summary = (f"{ctx['customer_id']} is {ctx['risk_level']} risk with a churn probability of "
               f"{ctx['churn_probability']}, putting about {ctx['expected_margin_loss_next_12_months']} "
               f"of margin at stake over the next year. The main drivers are {_join(reasons)}. "
               f"Lead with: {action[0].lower() + action[1:]}.")
    offer = _CUSTOMER_OFFERS.get(action, _CUSTOMER_OFFERS[DEFAULT_ACTION])
    months = ctx["tenure_months"]
    since = f" for the past {months} months" if months > 1 else ""
    message = (f"Hi, thank you for being with us{since}. {offer[0].upper() + offer[1:]}. "
               f"Just reply to this message or give us a call and we'll sort it out. "
               f"- The Customer Care Team")
    return {"summary": summary, "customer_message": message, "source": "template"}


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def claude_brief(ctx: dict, model: str = config.LLM_MODEL) -> dict:
    import anthropic

    client = anthropic.Anthropic(timeout=60.0)
    response = client.beta.messages.parse(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(ctx, indent=2)}],
        output_format=RetentionBrief,
        # Short, well-specified rewriting task: low effort keeps it fast and cheap.
        output_config={"effort": "low"},
        # If a safety classifier declines, retry server-side on the recommended model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"no usable output (stop_reason={response.stop_reason})")
    brief = response.parsed_output
    return {"summary": brief.summary, "customer_message": brief.customer_message,
            "source": response.model}


def generate_brief(ctx: dict, use_llm: bool = True) -> dict:
    """Claude when available, template otherwise. Never raises."""
    if not (use_llm and llm_available()):
        return template_brief(ctx)
    import anthropic

    try:
        return claude_brief(ctx)
    except anthropic.AuthenticationError:
        note = "invalid API key"
    except anthropic.RateLimitError:
        note = "rate limited"
    except anthropic.APIStatusError as e:
        note = f"API error {e.status_code}"
    except anthropic.APIConnectionError:
        note = "network error"
    except (anthropic.AnthropicError, RuntimeError) as e:
        note = str(e)
    brief = template_brief(ctx)
    brief["source"] = f"template (Claude unavailable: {note})"
    return brief
