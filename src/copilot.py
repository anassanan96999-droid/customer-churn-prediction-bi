"""AI Copilot: ask the churn data questions in plain English.

Claude answers by calling two tools:
* run_sql          - read-only SQL against the analytical database (src/sql_lab.py
                     enforces read-only access, a time limit and a row cap);
* explain_customer - probability, SHAP reasons and actions for one customer.

The loop is written by hand (rather than with the SDK's beta tool runner) so it
can cap the number of steps, record every query for display, and be tested
offline with a fake client (tests/test_copilot.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from src import config
from src.sql_lab import QueryError, run_query, schema_text

SYSTEM_TEMPLATE = """You are the analytics copilot inside a customer-churn dashboard for a telecom company. Business users ask questions in plain English; answer them from the data.

Database (SQLite, read-only):
{schema}

Meaning of key columns:
- customers.churn = 1 means the customer left last month (historical label); active customers have churn = 0.
- customer_scores holds model output for every customer (out-of-fold scores): churn_probability (0-1); risk_level HIGH (>= 0.60), MEDIUM (0.30-0.60) or LOW; expected_loss = churn_probability x 12-month margin value, in USD; contact_recommended = 1 when the customer is above the money-optimal contact threshold of {threshold:.2f}; top_reasons lists SHAP reasons; segment is a k-means behavioural segment.
- customer_features adds engineered columns such as tenure_bucket, service_count, avg_monthly_spend and charge_per_service.
- Money columns are USD per month unless the name says annual.

How to work:
- Use run_sql for every number you state; never estimate or invent figures. Prefer aggregates to long row lists.
- Use explain_customer when asked why a specific customer is at risk or what to do about them.
- If the data cannot answer a question (for example support calls or usage history, which are not in this dataset), say so plainly.
- Answer for a business audience: lead with the answer in one or two sentences, then at most four short bullet points. Format money as $1,234 and rates as percentages."""

TOOLS = [
    {
        "name": "run_sql",
        "description": "Run one read-only SQLite SELECT (or WITH ... SELECT) statement against the "
                       "churn analytics database and receive up to 50 result rows as CSV.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SQLite SELECT statement."},
                "purpose": {"type": "string",
                            "description": "One short sentence saying what the query answers."},
            },
            "required": ["query", "purpose"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_customer",
        "description": "Get churn probability, risk level, expected loss, the top SHAP reasons "
                       "and recommended retention actions for one customer ID.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string",
                                           "description": "Customer ID, e.g. 1393-IMKZG."}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
]

SUGGESTED_QUESTIONS = [
    "Which contract type loses the most revenue to churn?",
    "How many HIGH-risk active customers pay by electronic check?",
    "Why is customer 1393-IMKZG at risk, and what should we do?",
    "Compare churn for senior and non-senior fiber customers.",
    "Which segment holds the most expected loss among active customers?",
]


@dataclass
class Step:
    tool: str
    input: dict
    result: pd.DataFrame | dict | None = None
    error: str | None = None


@dataclass
class Answer:
    text: str
    steps: list[Step] = field(default_factory=list)
    messages: list = field(default_factory=list)   # full API history, append-only
    stop_reason: str | None = None


@lru_cache(maxsize=1)
def _scored() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from src.predict import load_bundle, shap_frame
    scored = pd.read_csv(config.SCORED_CSV)
    return scored, shap_frame(scored), load_bundle()["reference"]


def explain_customer_tool(customer_id: str) -> dict:
    from src.explain import explain_customer
    scored, shap_values, reference = _scored()
    hits = scored.index[scored[config.ID_COL] == customer_id.strip()]
    if not len(hits):
        raise KeyError(f"No customer with ID {customer_id!r}.")
    i = hits[0]
    row = scored.loc[i]
    exp = explain_customer(shap_values.loc[i], row, reference)
    return {
        "customer_id": row[config.ID_COL],
        "already_churned": bool(row["churn"]),
        "churn_probability": round(float(row["churn_probability"]), 3),
        "risk_level": row["risk_level"],
        "expected_loss_usd": round(float(row["expected_loss"]), 2),
        "contract": row["contract"], "tenure_months": int(row["tenure"]),
        "monthly_charges": float(row["monthly_charges"]),
        "internet_service": row["internet_service"], "payment_method": row["payment_method"],
        "risk_factors": [f["text"] for f in exp["risk_factors"]],
        "protective_factors": [f["text"] for f in exp["protective_factors"]],
        "recommended_actions": exp["actions"],
    }


class Copilot:
    def __init__(self, client=None, model: str = config.LLM_MODEL, max_steps: int = 8,
                 threshold: float = 0.32):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(timeout=120.0)
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.system = SYSTEM_TEMPLATE.format(schema=schema_text(), threshold=threshold)

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool, Step]:
        step = Step(tool=name, input=dict(args))
        try:
            if name == "run_sql":
                df, truncated = run_query(args.get("query", ""), max_rows=200)
                step.result = df
                note = " (showing the first 50)" if len(df) > 50 else ""
                note += " (result capped at 200 rows)" if truncated else ""
                return f"{len(df)} rows{note}\n{df.head(50).to_csv(index=False)}", False, step
            if name == "explain_customer":
                step.result = explain_customer_tool(args.get("customer_id", ""))
                return json.dumps(step.result), False, step
            raise KeyError(f"Unknown tool {name!r}.")
        except (QueryError, KeyError) as e:
            step.error = str(e).strip("'\"")
            return f"Error: {step.error}", True, step

    def ask(self, question: str, history: list | None = None) -> Answer:
        messages = list(history or []) + [{"role": "user", "content": question}]
        steps: list[Step] = []
        for _ in range(self.max_steps):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                tools=TOOLS,
                messages=messages,
                output_config={"effort": "medium"},
                # If a safety classifier declines, retry server-side on the recommended model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    content, is_error, step = self._run_tool(block.name, block.input)
                    steps.append(step)
                    result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "refusal":
                text = text or "I can't help with that request."
            elif response.stop_reason == "max_tokens":
                text += "\n\n_(The answer was cut off - try a narrower question.)_"
            return Answer(text, steps, messages, response.stop_reason)

        return Answer(f"I stopped after {self.max_steps} tool calls without a final answer. "
                      "Try a narrower question.", steps, messages, "max_steps")
