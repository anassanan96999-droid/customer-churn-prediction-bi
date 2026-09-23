"""The Copilot's tool loop, tested offline with a scripted fake Claude client."""
from types import SimpleNamespace as NS

import pandas as pd

from src.copilot import Copilot, explain_customer_tool


class FakeClient:
    """Returns pre-scripted responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def test_tool_call_then_answer():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            text("Let me check."),
            tool_use("t1", "run_sql", {"query": "SELECT contract, COUNT(*) AS n FROM customers "
                                                "GROUP BY contract ORDER BY n DESC",
                                       "purpose": "customers per contract"})]),
        NS(stop_reason="end_turn", content=[text("Month-to-month is the largest group.")]),
    ])
    answer = Copilot(client=client).ask("Which contract is most common?")

    assert answer.text == "Month-to-month is the largest group."
    assert answer.stop_reason == "end_turn"
    assert len(answer.steps) == 1 and isinstance(answer.steps[0].result, pd.DataFrame)
    assert answer.steps[0].result.iloc[0]["contract"] == "Month-to-month"
    # second request carries the assistant turn and a matching tool_result
    second = client.requests[1]["messages"]
    assert second[-1]["role"] == "user"
    result = second[-1]["content"][0]
    assert result["tool_use_id"] == "t1" and "Month-to-month" in result["content"]
    assert "is_error" not in result
    assert client.requests[0]["tools"][0]["name"] == "run_sql"


def test_bad_sql_is_returned_as_tool_error_not_raised():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            tool_use("t1", "run_sql", {"query": "DROP TABLE customers", "purpose": "x"})]),
        NS(stop_reason="end_turn", content=[text("I can only read data.")]),
    ])
    answer = Copilot(client=client).ask("Delete everything")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "read-only" in result["content"]
    assert answer.steps[0].error


def test_explain_customer_tool_and_unknown_id():
    info = explain_customer_tool("1393-IMKZG")
    assert info["risk_level"] == "HIGH" and info["risk_factors"]
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "explain_customer", {"customer_id": "NOPE"})]),
        NS(stop_reason="end_turn", content=[text("Unknown customer.")]),
    ])
    Copilot(client=client).ask("Explain NOPE")
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_step_limit_and_refusal():
    loop = [NS(stop_reason="tool_use", content=[
        tool_use(f"t{i}", "run_sql", {"query": "SELECT 1", "purpose": "x"})]) for i in range(3)]
    answer = Copilot(client=FakeClient(loop), max_steps=3).ask("loop forever")
    assert answer.stop_reason == "max_steps" and len(answer.steps) == 3

    refused = Copilot(client=FakeClient([NS(stop_reason="refusal", content=[])])).ask("?")
    assert refused.stop_reason == "refusal" and refused.text


def test_history_is_append_only():
    client = FakeClient([NS(stop_reason="end_turn", content=[text("A1")]),
                         NS(stop_reason="end_turn", content=[text("A2")])])
    bot = Copilot(client=client)
    first = bot.ask("Q1")
    second = bot.ask("Q2", history=first.messages)
    assert second.messages[:len(first.messages)] == first.messages
    assert [m["role"] for m in second.messages] == ["user", "assistant", "user", "assistant"]
