import json

import novel_ner.claude_run as cr


def package(tmp_path, chapters=("ch0001",), name="test_b99"):
    task = tmp_path / name
    task.mkdir()
    for file in ("instructions.md", "guide.md", "chapters.md"):
        (task / file).write_text(f"{file} 内容", encoding="utf-8")
    (task / "task.json").write_text(json.dumps({"name": name, "chapters": list(chapters)}), encoding="utf-8")
    return task


def reply(text, error=False):
    return {"result": text, "is_error": error, "subtype": "success", "stop_reason": "end_turn", "num_turns": 1,
            "usage": {"input_tokens": 3, "output_tokens": 5, "output_tokens_details": {"thinking_tokens": 2}},
            "modelUsage": {"claude-test-1": {}}, "total_cost_usd": 0.01}


def test_every_attempt_is_logged_failures_retry_once_and_nothing_is_called_a_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(cr, "claude_version", lambda: "test")
    task = package(tmp_path)
    log = cr.CallLog("run1", {"label": "m_r1"})
    seen = []

    def failing_then_ok(prompt, model, effort, timeout):
        seen.append(prompt)
        if len(seen) == 1:
            raise RuntimeError("claude exited 1 with no output")
        return reply('```json\n{"chapters": []}\n```')

    result = cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run1", log, call_fn=failing_then_ok)
    assert result == {"task": "test_b99", "label": "m_r1", "status": "ok", "attempts": 2}
    assert "instructions.md 内容" in seen[0] and "chapters.md 内容" in seen[0]
    calls = json.loads((tmp_path / "runs" / "run1" / "claude_calls.json").read_text(encoding="utf-8"))["calls"]
    assert [c.get("failed", "")[:12] for c in calls] == ["RuntimeError", ""] and calls[1]["answer_sha256"]
    assert calls[1]["models_used"] == ["claude-test-1"] and calls[1]["num_turns"] == 1
    assert json.loads((task / "pass_m_r1.json").read_text(encoding="utf-8")) == {"chapters": []}
    assert not list(task.glob("draft_*"))
    # a second run finds the answer and calls nothing
    assert cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run1", log, call_fn=None)["status"] == "exists"


def test_two_bad_replies_fail_the_pass_and_both_are_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    task = package(tmp_path)
    log = cr.CallLog("run2", {"label": "m_r1"})
    replies = iter([reply("no json here"), reply("", error=True)])
    result = cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run2", log, call_fn=lambda *a: next(replies))
    assert result["status"] == "failed_twice" and not (task / "pass_m_r1.json").exists()
    calls = json.loads((tmp_path / "runs" / "run2" / "claude_calls.json").read_text(encoding="utf-8"))["calls"]
    assert [c["attempt"] for c in calls] == [1, 2] and calls[0]["invalid_json"] and calls[1]["is_error"]


def test_malformed_replies_are_logged_and_retried_and_open_logs_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    task = package(tmp_path)
    log = cr.CallLog("run3", {"label": "m_r1"})
    replies = iter([reply("```json"), reply("[1, 2]")])
    result = cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run3", log, call_fn=lambda *a: next(replies))
    assert result["status"] == "failed_twice"
    calls = json.loads((tmp_path / "runs" / "run3" / "claude_calls.json").read_text(encoding="utf-8"))["calls"]
    assert [c.get("invalid_json") for c in calls] == [True, True]
    import pytest
    with pytest.raises(SystemExit):  # the first log never finished: a second writer is refused
        cr.CallLog("run3", {"label": "m_r1"})
    log.finish([result])
    cr.CallLog("run3", {"label": "m_r1"})


def test_run_many_passes_the_stub_through(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(cr, "claude_version", lambda: "test")
    task = package(tmp_path)
    results = cr.run_many([task], "m_r1", "claude-test", "xhigh", "run4", 1, call_fn=lambda *a: reply('{"chapters": []}'))
    assert results[0]["status"] == "ok"
    data = json.loads((tmp_path / "runs" / "run4" / "claude_calls.json").read_text(encoding="utf-8"))
    assert data["invocations"][0]["finished_at"] and len(data["calls"]) == 1


def test_run_many_refuses_a_package_holding_a_sealed_test2_chapter_before_anything_is_logged(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(cr, "claude_version", lambda: "test")
    ok = package(tmp_path, ("ch0001", "ch0002"), "teach_b01")
    sealed = package(tmp_path, ("ch0005", "ch1262"), "test2_b01")
    calls = []
    stub = lambda *a: calls.append(a) or reply('{"chapters": []}')
    with pytest.raises(SystemExit, match="sealed test2"):
        cr.run_many([ok, sealed], "m_r1", "claude-test", "xhigh", "run6", 1, call_fn=stub, sealed={"ch1262", "ch1345"})
    assert calls == [] and not (tmp_path / "runs" / "run6").exists() and not (sealed / "pass_m_r1.json").exists()
    assert cr.run_many([ok], "m_r1", "claude-test", "xhigh", "run6", 1, call_fn=stub, sealed={"ch1262"})[0]["status"] == "ok"
    assert cr.task_chapters(ok) == {"ch0001", "ch0002"}
    # an adjudication package names its chapters through its places; a package without task.json cannot be checked
    adj = tmp_path / "adjudicate_x"
    adj.mkdir()
    (adj / "task.json").write_text(json.dumps({"name": "adjudicate_x", "kind": "adjudicate",
                                               "places": [{"id": "q1", "chapter_id": "ch1262"}, {"id": "q2", "chapter_id": "ch0003"}]}), encoding="utf-8")
    assert cr.task_chapters(adj) == {"ch1262", "ch0003"}
    with pytest.raises(SystemExit, match="sealed test2"):
        cr.refuse_sealed([adj], {"ch1262"})
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match="task.json"):
        cr.refuse_sealed([bare], {"ch1262"})
    # the default reads the repository's seal: the sealed chapters are after 990, so an early chapter passes;
    # a test2 chapter is refused while the seal holds, and passes once the seal was opened for the final evaluation
    cr.refuse_sealed([ok])
    from novel_ner.tagger import read_seal
    late = package(tmp_path, ("ch1262",), "late_x")
    if read_seal()[0]["status"] == "sealed":
        with pytest.raises(SystemExit, match="sealed test2"):
            cr.refuse_sealed([late])
    else:
        cr.refuse_sealed([late])


def test_a_rerun_keeps_the_earlier_raw_reply_and_logs_error_text(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    task = package(tmp_path)
    log = cr.CallLog("run5", {"label": "m_r1"})
    limit = reply("You've hit your monthly spend limit", error=True)
    cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run5", log, call_fn=lambda *a: limit)
    log.finish([])
    log2 = cr.CallLog("run5", {"label": "m_r1"})
    cr.run_pass(task, "m_r1", "claude-test", "xhigh", "run5", log2, call_fn=lambda *a: reply('{"chapters": []}'))
    assert len(list(task.glob("claude_m_r1_attempt1*.json"))) == 2
    calls = json.loads((tmp_path / "runs" / "run5" / "claude_calls.json").read_text(encoding="utf-8"))["calls"]
    assert calls[0]["error_text"].startswith("You've hit") and "error_text" not in calls[-1]
