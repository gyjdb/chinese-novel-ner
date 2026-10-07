import json
from types import SimpleNamespace

import pytest

from novel_ner import gpt_api
from novel_ner.gpt_api import Budget, BudgetStop, build_prompt, cost, extract_json, run_pass


def test_prompt_puts_the_shared_parts_first(tmp_path):
    for name, body in (("instructions.md", "说明"), ("guide.md", "规则"), ("chapters.md", "原文")):
        (tmp_path / name).write_text(body, encoding="utf-8")
    prompt = build_prompt(tmp_path)
    assert prompt.startswith(gpt_api.REQUEST)
    assert prompt.index("说明") < prompt.index("规则") < prompt.index("原文")


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的：{"a": {"b": 2}} 以上') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("没有 JSON")


def test_cost_counts_cached_input_at_the_cached_price():
    usage = {"input_tokens": 1_000_000, "cached_tokens": 500_000, "output_tokens": 100_000}
    assert cost(usage, {"input": 2.0, "cached_input": 0.2, "output": 10.0}) == pytest.approx(1.0 + 0.1 + 1.0)


def test_budget_reserves_before_requests(tmp_path):
    b = Budget(cap=3.0, ledger=tmp_path / "ledger.json")
    first, second = b.reserve(), b.reserve()  # a typical request is 1.5 before anything is measured
    assert first == second == 1.5 and b.reserve() is None
    b.release(first)
    b.record({"estimated_usd": 0.4})
    assert b.spent == 0.4 and json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["total_estimated_usd"] == 0.4


class FakeAPI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        assert kwargs["store"] is False and kwargs["reasoning"] == {"effort": "xhigh"}
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500,
                                input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=300))
        return SimpleNamespace(id="resp_1", model="gpt-test", status="completed", usage=usage, output_text=self.replies.pop(0))


def task(tmp_path):
    for name in ("instructions.md", "guide.md", "chapters.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    return tmp_path


def test_run_pass_retries_invalid_json_once_and_keeps_both_replies(tmp_path, monkeypatch):
    imported = []
    monkeypatch.setattr(gpt_api, "import_answer", lambda f, run, model: imported.append((f.name, run)))
    b = Budget(cap=10, ledger=tmp_path / "ledger.json")
    result = run_pass(FakeAPI(["不是 JSON", '{"chapters": []}']), task(tmp_path), "A", "gpt-test", "xhigh", b, "r1")
    assert result["status"] == "ok" and result["attempts"] == 2 and imported == [("pass_A.json", "A")]
    assert (tmp_path / "api_A_attempt1.json").exists() and (tmp_path / "api_A_attempt2.json").exists()
    assert [e.get("invalid_json", False) for e in b.entries] == [True, False]
    assert run_pass(FakeAPI([]), tmp_path, "A", "gpt-test", "xhigh", b, "r1")["status"] == "exists"  # never overwritten


def test_run_pass_stops_at_the_budget(tmp_path):
    b = Budget(cap=1.0, ledger=tmp_path / "ledger.json")
    with pytest.raises(BudgetStop):
        run_pass(FakeAPI(['{"chapters": []}']), task(tmp_path), "A", "gpt-test", "xhigh", b, "r1")
