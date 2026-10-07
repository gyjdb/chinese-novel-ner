import json

import novel_ner.claude_run as cr
import novel_ner.codex_run as xr


def package(tmp_path, chapters=("ch0057",)):
    task = tmp_path / "calib_x"
    task.mkdir()
    for name in ("instructions.md", "guide.md", "chapters.md"):
        (task / name).write_text(f"{name} 内容", encoding="utf-8")
    (task / "task.json").write_text(json.dumps({"name": "calib_x", "chapters": list(chapters)}), encoding="utf-8")
    return task


def events(*items, usage=True):
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"}), json.dumps({"type": "turn.started"})]
    for i, (kind, text) in enumerate(items):
        lines.append(json.dumps({"type": "item.completed", "item": {"id": f"item_{i}", "type": kind, "text": text}}))
    if usage:
        lines.append(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4,
                                                                     "reasoning_output_tokens": 2}}))
    return lines


def reply(last, *items, code=0):
    return {"events": events(*items), "last_message": last, "returncode": code, "stderr_tail": ""}


def test_command_turns_off_web_search_user_config_and_tools():
    cmd = xr.command("codex.exe", "gpt-6-astra", "xhigh", "C:/tmp/empty", "C:/tmp/out/last.txt")
    assert cmd[:4] == ["codex.exe", "exec", "-m", "gpt-6-astra"] and cmd[-1] == "-"
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json"):
        assert flag in cmd
    assert 'web_search="disabled"' in cmd and 'model_reasoning_effort="xhigh"' in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only" and cmd[cmd.index("-C") + 1] == "C:/tmp/empty"
    disabled = {cmd[i + 1] for i, x in enumerate(cmd) if x == "--disable"}
    assert {"memories", "shell_tool", "unified_exec", "multi_agent", "code_mode_host"} <= disabled


def test_summarize_events_separates_tool_items_and_errors():
    ev = xr.summarize_events(events(("reasoning", ""), ("command_execution", "dir"), ("agent_message", "{}")))
    assert ev["thread_id"] == "t1" and ev["tool_items"] == {"command_execution": 1}
    assert ev["items"] == {"reasoning": 1, "command_execution": 1, "agent_message": 1} and ev["usage"]["input_tokens"] == 10
    bad = xr.summarize_events(['{"type": "turn.failed", "error": {"message": "usage limit"}}', "not json"])
    assert len(bad["errors"]) == 2
    notice = {"type": "item.completed", "item": {"type": "error", "message": xr.EXPECTED_NOTICE + ". Code mode will fail closed"}}
    other = {"type": "item.completed", "item": {"type": "error", "message": "stream disconnected"}}
    ev = xr.summarize_events([json.dumps(notice), json.dumps(other)])
    assert ev["tool_items"] == {} and len(ev["notices"]) == 1 and ev["errors"] == ["stream disconnected"]


def test_tool_use_is_rejected_and_retried_once_and_every_attempt_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    task = package(tmp_path)
    log = cr.CallLog("run1", {"label": "astra"}, name="codex_calls.json")
    prompts = []
    replies = iter([reply('{"chapters": []}', ("command_execution", "type answers.json"), ("agent_message", '{"chapters": []}')),
                    reply('```json\n{"chapters": []}\n```', ("reasoning", ""), ("agent_message", "..."))])

    def fake(prompt, model, effort, timeout):
        prompts.append(prompt)
        return next(replies)

    result = xr.run_pass(task, "astra", "gpt-6-astra", "xhigh", "run1", log, call_fn=fake)
    assert result == {"task": "calib_x", "label": "astra", "status": "ok", "attempts": 2}
    assert "guide.md 内容" in prompts[0]
    calls = json.loads((tmp_path / "runs" / "run1" / "codex_calls.json").read_text(encoding="utf-8"))["calls"]
    assert calls[0]["rejected"] == "used a tool" and "answer_sha256" not in calls[0]
    assert calls[1]["answer_sha256"] and calls[1]["tool_items"] == {}
    assert json.loads((task / "pass_astra.json").read_text(encoding="utf-8")) == {"chapters": []}
    assert (task / "codex_astra_attempt1.jsonl").exists() and not list(task.glob("draft_*"))
    assert xr.run_pass(task, "astra", "gpt-6-astra", "xhigh", "run1", log, call_fn=None)["status"] == "exists"


def test_run_many_refuses_a_sealed_test2_chapter_before_looking_for_the_cli(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(xr, "codex_exe", lambda: (_ for _ in ()).throw(AssertionError("the CLI must not be looked up")))
    task = package(tmp_path, ("ch0057", "ch1345"))
    with pytest.raises(SystemExit, match="sealed test2"):
        xr.run_many([task], "astra", "gpt-6-astra", "xhigh", "run3", sealed={"ch1345"})
    assert not (tmp_path / "runs" / "run3").exists() and not (task / "pass_astra.json").exists()
    (tmp_path / "other").mkdir()
    from novel_ner.tagger import read_seal
    if read_seal()[0]["status"] == "sealed":  # the repository's own seal, read by default (open after stage 3)
        with pytest.raises(SystemExit, match="sealed test2"):
            xr.run_many([package(tmp_path / "other", ("ch1262",))], "astra", "gpt-6-astra", "xhigh", "run3")
    # adjudication makes the answers and is exempt from the seal (configs/test2_seal.json answer_making_exempt):
    # the same sealed package passes the guard and the next step, the CLI lookup, is reached
    with pytest.raises(AssertionError, match="must not be looked up"):
        xr.run_many([task], "astra", "gpt-6-astra", "xhigh", "run3", kind="adjudicate", sealed={"ch1345"})
    assert not (tmp_path / "runs" / "run3").exists()


def test_errors_and_unreadable_replies_fail_the_pass_after_two_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    task = package(tmp_path)
    log = cr.CallLog("run2", {"label": "astra"}, name="codex_calls.json")
    replies = iter([reply("", code=1), reply("no json here", ("agent_message", "no json here"))])
    result = xr.run_pass(task, "astra", "gpt-6-astra", "xhigh", "run2", log, call_fn=lambda *a: next(replies))
    assert result["status"] == "failed_twice" and not (task / "pass_astra.json").exists()
    calls = json.loads((tmp_path / "runs" / "run2" / "codex_calls.json").read_text(encoding="utf-8"))["calls"]
    assert calls[0]["returncode"] == 1 and calls[1]["invalid_json"]
