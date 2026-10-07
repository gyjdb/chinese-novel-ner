"""Run GPT-6-Astra (or another Codex model) as the reviewer of the test answers, through the local Codex CLI
(the reviewer is GPT-6-Astra via Codex; every chapter of this book may go to external model
services, nothing is made public). Astra is not a compared system and gave no training labels; it is the same
GPT-6 family as the drafter (gpt-6.1-sol), which the report states.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.codex_run run data/llm_tasks/calib_api --label astra --run-id <id>
    .venv\\Scripts\\python -m novel_ner.codex_run score-calib data/llm_tasks/calib_api --label astra --gold <dev answers> --run-id <id>

Isolation: every call runs `codex exec` in a fresh empty temporary directory, read-only sandbox, without the
user's config.toml (--ignore-user-config: no live web search, no profile) or exec rules, without saving the
session, with web search off and memories, shell, apps, plugins, browser, computer use, image tools,
multi-agent, code mode and goals disabled. Codex keeps a few tools it cannot switch off (a code runner, agent
messaging); every call's event stream is checked, and an answer from a call that used any tool is not kept
(the call counts as a failed attempt). The model sees only the prompt GPT got: the same one-line request and
three files (gpt_api.build_prompt). Sampling: Codex defaults at the given reasoning effort; nothing else can
be set.

Fixed rules (set before the runs): one call per package; an error, a reply without valid JSON, or any tool use
is retried once. The JSON is stored as pass_<label>.json untouched; the raw event stream of every attempt is
kept next to the package (codex_<label>_attempt<n>.jsonl, holds model text, not committed). Every call is
appended to runs/<run-id>/codex_calls.json (model, settings, timing, token counts, item types seen, the sha256
of the stored answer; no novel text). Nothing is written under the name draft_*.
test2 seal: run_many refuses, before anything is called or logged, an annotation package (kind "annotate") whose
task.json names a chapter of the sealed second test (claude_run.refuse_sealed). Adjudication packages (kind
"adjudicate") are exempt: adjudicating pending places with the user's words is part of making the answers, which
configs/test2_seal.json lists as allowed (the adjudicator is not a compared system and sees no system's output).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from novel_ner.claude_run import CallLog, refuse_sealed
from novel_ner.gpt_api import PROMPT_VERSION, build_prompt, extract_json
from novel_ner.llm_task import convert
from novel_ner.score import _context, _touches, draft_spans, gold_spans, strict
from novel_ner.snapshot import REPO_ROOT, load_snapshot
from novel_ner.zeroshot import git_state

DISABLED = ["memories", "shell_tool", "unified_exec", "apps", "plugins", "browser_use", "browser_use_external",
            "computer_use", "image_generation", "multi_agent", "multi_agent_v2", "code_mode_host", "code_mode",
            "goals", "view_image", "sleep_tool", "request_permissions_tool", "in_app_browser", "skill_search",
            "tool_suggest", "hooks", "remote_plugin"]
MESSAGE_ITEMS = {"agent_message", "reasoning"}  # anything else in the event stream is a tool call or an error
# Codex posts this as an "error" item when code mode is switched off (its code runner then fails closed):
# expected, recorded as a notice. Any other error item fails the attempt.
EXPECTED_NOTICE = "Code Mode is unavailable because code-mode host is disabled"
ISOLATION = ("codex exec in an empty temp dir, --sandbox read-only --ignore-user-config --ignore-rules --ephemeral "
             "--skip-git-repo-check, web_search disabled, features off: " + ", ".join(DISABLED)
             + "; calls whose events show any item other than agent_message/reasoning are rejected")


def codex_exe():
    found = os.environ.get("NOVEL_NER_CODEX") or shutil.which("codex")
    if not found:  # the Codex desktop app ships the CLI under %LOCALAPPDATA%\OpenAI\Codex\bin\<build>\
        builds = glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenAI", "Codex", "bin", "*", "codex.exe"))
        found = max(builds, key=os.path.getmtime) if builds else None
    if not found:
        raise SystemExit("no Codex CLI found (set NOVEL_NER_CODEX)")
    return found


def command(exe, model, effort, workdir, last_file):
    cmd = [exe, "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-c", 'web_search="disabled"',
           "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
           "-C", str(workdir), "--json", "-o", str(last_file)]
    for feature in DISABLED:
        cmd += ["--disable", feature]
    return cmd + ["-"]  # the prompt comes on stdin


def codex_version(exe):
    return subprocess.run([exe, "--version"], capture_output=True, text=True, encoding="utf-8").stdout.strip()


def summarize_events(lines):
    """Item types, usage and errors from a `codex exec --json` event stream (one JSON object per line)."""
    items, usage, errors, notices, thread = Counter(), None, [], [], None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            errors.append("unparsed event line")
            continue
        kind = e.get("type")
        if kind == "thread.started":
            thread = e.get("thread_id")
        elif kind == "item.completed":
            item = e.get("item") or {}
            items[item.get("type")] += 1
            if item.get("type") == "error":
                message = str(item.get("message") or "")[:200]
                (notices if message.startswith(EXPECTED_NOTICE) else errors).append(message)
        elif kind == "turn.completed":
            usage = e.get("usage")
        elif kind in ("turn.failed", "error"):
            errors.append(str(e.get("error") or e.get("message") or kind)[:200])
    tools = {k: v for k, v in items.items() if k not in MESSAGE_ITEMS | {"error"}}
    return {"thread_id": thread, "items": dict(items), "tool_items": tools, "usage": usage, "errors": errors,
            "notices": notices}


def call(prompt, model, effort, timeout, exe=None):
    """One isolated call; returns the event lines, the last message and the exit code."""
    exe = exe or codex_exe()
    with tempfile.TemporaryDirectory(prefix="novel-ner-codex-") as empty, \
            tempfile.TemporaryDirectory(prefix="novel-ner-codex-out-") as out:
        last = Path(out) / "last.txt"
        done = subprocess.run(command(exe, model, effort, empty, last), input=prompt, capture_output=True, text=True,
                              encoding="utf-8", cwd=empty, timeout=timeout)
        message = last.read_text(encoding="utf-8") if last.exists() else ""
    return {"events": done.stdout.splitlines(), "last_message": message, "returncode": done.returncode,
            "stderr_tail": done.stderr.strip()[-300:]}


ADJUDICATION_REQUEST = "请按 instructions.md 逐处裁决 questions.md 里列出的地方，只输出 JSON。"


def build_adjudication_prompt(task_dir):
    """The adjudication package (review.py package): the request line and three files, like build_prompt."""
    parts = [ADJUDICATION_REQUEST, ""]
    for name in ("instructions.md", "guide.md", "questions.md"):
        parts += [f"===== {name} =====", (Path(task_dir) / name).read_text(encoding="utf-8").strip(), ""]
    return "\n".join(parts)


BUILDERS = {"annotate": build_prompt, "adjudicate": build_adjudication_prompt}


def run_pass(task_dir, label, model, effort, run_id, log, timeout=3600, call_fn=None, kind="annotate"):
    call_fn = call_fn or call
    task_dir = Path(task_dir)
    pass_file = task_dir / f"pass_{label}.json"
    if pass_file.exists():
        return {"task": task_dir.name, "label": label, "status": "exists"}
    prompt = BUILDERS[kind](task_dir)
    for old in task_dir.glob(f"codex_{label}_attempt[0-9].jsonl"):  # an earlier invocation's attempts keep their own names
        old.rename(old.with_name(f"{old.stem}.{uuid.uuid4().hex[:8]}.jsonl"))
    for attempt in (1, 2):
        started = time.time()
        rec = {"run_id": run_id, "task": task_dir.name, "label": label, "attempt": attempt, "model": model,
               "effort": effort, "kind": kind, "prompt_version": PROMPT_VERSION if kind == "annotate" else "adjudicate_v1", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        try:
            out = call_fn(prompt, model, effort, timeout)
        except Exception as e:  # a failed call is a recorded attempt
            log.add({**rec, "failed": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - started, 1)})
            continue
        ev = summarize_events(out["events"])
        rec.update({"seconds": round(time.time() - started, 1), "returncode": out["returncode"], "thread_id": ev["thread_id"],
                    "items": ev["items"], "tool_items": ev["tool_items"], "usage": ev["usage"], "event_errors": ev["errors"],
                    "notices": ev["notices"]})
        (task_dir / f"codex_{label}_attempt{attempt}.jsonl").write_text("\n".join(out["events"]) + "\n", encoding="utf-8")
        if out["returncode"] != 0 or ev["errors"]:
            rec["error_text"] = (ev["errors"] or [out["stderr_tail"]])[0][:200]  # the CLI's message, not novel text
            log.add(rec)
            continue
        if ev["tool_items"]:
            rec["rejected"] = "used a tool"
            log.add(rec)
            continue
        try:
            answer = extract_json(out["last_message"])
        except Exception:
            answer = None
        if not isinstance(answer, dict):
            rec["invalid_json"] = True
            log.add(rec)
            continue
        pass_file.write_text(json.dumps(answer, ensure_ascii=False, indent=1), encoding="utf-8")
        rec["answer_sha256"] = hashlib.sha256(pass_file.read_bytes()).hexdigest()
        log.add(rec)
        return {"task": task_dir.name, "label": label, "status": "ok", "attempts": attempt}
    return {"task": task_dir.name, "label": label, "status": "failed_twice"}


ROLE = "reviewer of the test answers (not a compared system)"


def run_many(tasks, label, model, effort, run_id, workers=1, call_fn=None, kind="annotate", role=ROLE, sealed=None):
    if kind != "adjudicate":  # adjudication makes the answers (the seal exempts it); an annotation pass is a system run
        refuse_sealed(tasks, sealed)
    exe = codex_exe()
    header = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": git_state(), "codex_cli": codex_version(exe),
              "model_requested": model, "label": label, "effort": effort,
              "prompt_version": PROMPT_VERSION if kind == "annotate" else "adjudicate_v1",
              "isolation": ISOLATION, "role": role,
              "workers": workers, "kind": kind, "tasks": [Path(t).name for t in tasks]}
    log = CallLog(run_id, header, name="codex_calls.json")

    def one(task):  # each package is its own fresh session; the log is shared and locked
        try:
            result = run_pass(task, label, model, effort, run_id, log, call_fn=call_fn, kind=kind)
        except Exception as e:
            result = {"task": Path(task).name, "label": label, "status": f"error: {type(e).__name__}: {e}"}
        print(result, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:  # the user chose 3 at a time for the test (2026-10-02)
        results = list(pool.map(one, tasks))
    log.finish(results)
    return results


def score_calibration(task_dir, label, gold_file, run_id):
    """The reviewer's pass on a calibration package against the answers, as score.calibrate does for GPT."""
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    chapters = task["chapters"]
    snapshot = load_snapshot(task["snapshot_id"], verify=False)
    export = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    gold = gold_spans(export, chapters)
    answer = json.loads((task_dir / f"pass_{label}.json").read_text(encoding="utf-8"))
    review = {"task": task["name"], "run": label, "source_file": f"pass_{label}.json",
              "snapshot_id": task["snapshot_id"], "instructions_version": answer.get("instructions_version"),
              "guide_version": answer.get("guide_version"), "chapters": convert(answer, snapshot, chapters)}
    (task_dir / f"review_{label}.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    spans = draft_spans(review, chapters)
    vs = strict(gold, spans)
    metrics = {"run_id": run_id, "task": task["name"], "chapters": chapters, "label": label,
               "gold_file": Path(gold_file).name,
               "gold_provenance": {c: export["chapters"][c].get("provenance") for c in chapters},
               "gold_spans": sum(len(g["spans"]) for g in gold.values()),
               "gold_unscored_regions": sum(len(g["unscored"]) for g in gold.values()),
               "review": {"spans": sum(len(s["spans"]) for s in spans.values()),
                          "pending": sum(len(s["pending"]) for s in spans.values()),
                          "unlocated": sum(s["errors"] for s in spans.values()), "vs_gold": vs},
               "gate": {"rule": "same as GPT's calibration: P >= 0.98 and R >= 0.88",
                        "passed": vs["precision"] >= 0.98 and vs["recall"] >= 0.88}}
    lines = [f"# 复核模型校准报告 {run_id}（{label}）", "", "含小说原文，不入库。【】是范围。", ""]
    for cid in chapters:
        text = snapshot.chapters[cid].text
        g = {s for s in gold[cid]["spans"] if not _touches(s, gold[cid]["unscored"])}
        p = {s for s in spans[cid]["spans"] if not _touches(s, gold[cid]["unscored"])}
        lines.append(f"## {cid}")
        for s, e, t in sorted(g - p):
            lines.append(f"- 漏了（答案 {t}）：{_context(text, s, e)}")
        for s, e, t in sorted(p - g):
            lines.append(f"- 多了或标错（{label} 标 {t}）：{_context(text, s, e)}")
        for s, e in spans[cid]["pending"]:
            lines.append(f"- {label} 标了待审：{_context(text, s, e)}")
        for err in review["chapters"][cid]["errors"]:
            lines.append(f"- 定位失败：{err}")
        lines.append("")
    out = REPO_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"run_id": run_id, "task": task, "label": label,
                                                 "gold_file": Path(gold_file).name, "code": git_state(),
                                                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (task_dir / f"report_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("tasks", nargs="+")
    r.add_argument("--model", default="gpt-6-astra")
    r.add_argument("--label", required=True, help="names the pass files, e.g. astra")
    r.add_argument("--effort", default="xhigh")
    r.add_argument("--run-id", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--kind", choices=sorted(BUILDERS), default="annotate")
    r.add_argument("--role", default=ROLE, help="what the model is in this run, for the call log (dev answers: "
                                                "\"adjudicator of the dev answers' pending places (not a compared system)\")")
    s = sub.add_parser("score-calib")
    s.add_argument("task")
    s.add_argument("--label", required=True)
    s.add_argument("--gold", required=True)
    s.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if args.label in ("A", "B"):
        raise SystemExit("labels A and B are GPT's passes")
    if args.action == "run":
        run_many(args.tasks, args.label, args.model, args.effort, args.run_id, args.workers, kind=args.kind, role=args.role)
    else:
        m = score_calibration(args.task, args.label, args.gold, args.run_id)
        print(json.dumps({k: m["review"][k] for k in ("spans", "pending", "unlocated")}), json.dumps(m["gate"]))
        print({k: m["review"]["vs_gold"][k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")})


if __name__ == "__main__":
    main()
