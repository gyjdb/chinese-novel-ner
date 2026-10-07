"""Run Claude models (Opus, Sonnet) as systems under test on the test task packages, through the local
Claude Code CLI (test chapter text may go to Anthropic this way; no API key).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.claude_run run data/llm_tasks/test_b03 --model claude-sonnet-5-5 --label sonnet_r1 --run-id <id>
    .venv\\Scripts\\python -m novel_ner.claude_run run-test --model claude-opus-5-5 --label opus_r1 --run-id <id> [--workers 3]

Isolation: every call runs `claude -p` in a fresh empty temporary directory with all tools disabled
(--tools ""), every customization off (--safe-mode: no CLAUDE.md, memory, skills, hooks, plugins), no MCP
servers and no saved session, under a one-line neutral system prompt. The model sees only the prompt: the
same one-line request and three files GPT got (gpt_api.build_prompt: instructions.md, guide.md, chapters.md),
so it cannot read the answers or anything else on this machine. The effort is passed as --effort and also as
CLAUDE_CODE_EFFORT_LEVEL and CLAUDE_EFFORT, so nothing inherited from a parent Claude Code session can differ.
Sampling: Claude Code's defaults; temperature, top-p and seed cannot be set, so repeated runs differ.

Fixed rules (set before the formal runs): one call per 5-chapter package; a reply that is an error or holds
no valid JSON is retried once; the JSON is stored as pass_<label>.json untouched and scored by
evaluate.llm_predictions (positions from paragraph id, text and occurrence). Nothing is written under
the name draft_*: model outputs are never drafts for the answers (project annotation protocol).
Each raw reply is kept next to the package (claude_<label>_attempt<n>.json, not committed). Every call,
failed or not, is appended to runs/<run-id>/claude_calls.json as soon as it returns (model, settings,
timing, token counts, the CLI's list-price estimate, the sha256 of the stored answer; no novel text).
test2 seal (the material-scope rules): before anything is called or logged, run_many reads every package's task.json
and refuses the whole invocation if a package holds a chapter of the sealed second test
(tagger.sealed_test2_chapters, configs/test2_seal.json). The GPT drafting passes (gpt_api) are not guarded: they
make test2's answers.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from novel_ner.annotate import test_chapters
from novel_ner.gpt_api import PROMPT_VERSION, build_prompt, extract_json
from novel_ner.llm_task import TASK_ROOT
from novel_ner.snapshot import REPO_ROOT
from novel_ner.zeroshot import git_state

SYSTEM_PROMPT = "Follow the user's instructions."
SAMPLING = "Claude Code CLI defaults; temperature, top-p and seed cannot be set; adaptive thinking at the given effort"
ISOLATION = "claude -p in an empty temp dir, --tools '' --safe-mode --strict-mcp-config --no-session-persistence --disable-slash-commands"


def command(model, effort):
    exe = shutil.which("claude")
    if exe is None:
        raise SystemExit("the claude CLI is not on PATH")
    return [exe, "-p", "--model", model, "--effort", effort, "--tools", "", "--safe-mode", "--strict-mcp-config",
            "--no-session-persistence", "--disable-slash-commands", "--output-format", "json",
            "--system-prompt", SYSTEM_PROMPT]


def claude_version():
    done = subprocess.run([shutil.which("claude") or "claude", "--version"], capture_output=True, text=True, encoding="utf-8")
    return done.stdout.strip()


def call(prompt, model, effort, timeout):
    """One isolated call; returns the CLI's JSON result."""
    env = {**os.environ, "CLAUDE_EFFORT": effort, "CLAUDE_CODE_EFFORT_LEVEL": effort}
    with tempfile.TemporaryDirectory(prefix="novel-ner-claude-") as empty:
        done = subprocess.run(command(model, effort), input=prompt, capture_output=True, text=True, encoding="utf-8",
                              cwd=empty, timeout=timeout, env=env)
    if not done.stdout.strip():
        raise RuntimeError(f"claude exited {done.returncode} with no output: {done.stderr.strip()[:300]}")
    out = json.loads(done.stdout)
    if not isinstance(out, dict):
        raise RuntimeError(f"claude printed {type(out).__name__}, not a JSON object")
    return out


def record_of(out, task, label, attempt, effort, seconds, run_id):
    usage = out.get("usage") or {}
    models = out.get("modelUsage") or {}
    return {
        "run_id": run_id, "task": task, "label": label, "attempt": attempt, "effort": effort,
        "models_used": sorted(models), "system_prompt": SYSTEM_PROMPT, "prompt_version": PROMPT_VERSION,
        "is_error": out.get("is_error"), "subtype": out.get("subtype"), "stop_reason": out.get("stop_reason"),
        "num_turns": out.get("num_turns"), "seconds": round(seconds, 1), "input_tokens": usage.get("input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"), "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        "list_price_estimate_usd": out.get("total_cost_usd"), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class CallLog:
    """runs/<run-id>/<name> (claude_calls.json by default), rewritten after every call so nothing is lost if a
    pass fails later."""

    def __init__(self, run_id, header, name="claude_calls.json"):
        self.path = REPO_ROOT / "runs" / run_id / name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.data = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                     else {"run_id": run_id, "invocations": [], "calls": []})
        open_ones = [i for i in self.data.get("invocations", []) if "finished_at" not in i]
        if open_ones:  # another process may be writing this log; it would overwrite this one's records
            raise SystemExit(f"{self.path} has an invocation that never finished (started {open_ones[-1].get('started_at')}); "
                             "use another run id, or mark it finished after checking no run is going on")
        self.data.setdefault("invocations", []).append(header)
        self._write()

    def add(self, rec):
        with self.lock:
            self.data["calls"].append(rec)
            self._write()

    def finish(self, results):
        with self.lock:
            self.data["invocations"][-1]["results"] = results
            self.data["invocations"][-1]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._write()

    def _write(self):
        calls = self.data["calls"]
        self.data["list_price_estimate_usd_total"] = round(sum(c.get("list_price_estimate_usd") or 0 for c in calls), 4)
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)  # a reader never sees a half-written file


def run_pass(task_dir, label, model, effort, run_id, log, timeout=3600, call_fn=None):
    """One independent pass over one task package; every attempt is logged as it happens."""
    call_fn = call_fn or call  # looked up when called, so tests can replace call
    task_dir = Path(task_dir)
    pass_file = task_dir / f"pass_{label}.json"
    if pass_file.exists():
        return {"task": task_dir.name, "label": label, "status": "exists"}
    prompt = build_prompt(task_dir)
    for old in task_dir.glob(f"claude_{label}_attempt[0-9].json"):  # an earlier invocation's replies keep their own names
        old.rename(old.with_name(f"{old.stem}.{uuid.uuid4().hex[:8]}.json"))
    for attempt in (1, 2):
        started = time.time()
        try:
            out = call_fn(prompt, model, effort, timeout)
        except Exception as e:  # a failed call is a recorded attempt
            log.add({"run_id": run_id, "task": task_dir.name, "label": label, "attempt": attempt, "effort": effort,
                     "failed": f"{type(e).__name__}: {e}"[:300], "seconds": round(time.time() - started, 1),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            continue
        rec = record_of(out, task_dir.name, label, attempt, effort, time.time() - started, run_id)
        if out.get("is_error"):  # the CLI's own message (e.g. a usage limit), never novel text
            rec["error_text"] = str(out.get("result") or "")[:200]
        (task_dir / f"claude_{label}_attempt{attempt}.json").write_text(json.dumps(
            {"record": rec, "result": out.get("result")}, ensure_ascii=False, indent=1), encoding="utf-8")
        answer = None
        if not out.get("is_error"):
            try:
                answer = extract_json(out.get("result"))
            except Exception:  # anything unreadable is a failed attempt, logged and retried
                rec["invalid_json"] = True
            if answer is not None and not isinstance(answer, dict):
                answer, rec["invalid_json"] = None, True
        if answer is None:
            log.add(rec)
            continue
        text = json.dumps(answer, ensure_ascii=False, indent=1)
        pass_file.write_text(text, encoding="utf-8")
        rec["answer_sha256"] = hashlib.sha256(pass_file.read_bytes()).hexdigest()
        log.add(rec)
        return {"task": task_dir.name, "label": label, "status": "ok", "attempts": attempt}
    return {"task": task_dir.name, "label": label, "status": "failed_twice"}


def task_chapters(task_dir):
    """The chapter ids a package touches, from its task.json: "chapters" (annotation packages) and the places of an
    adjudication package. A package without task.json cannot be checked against the seal and is refused."""
    path = Path(task_dir) / "task.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: the package's chapters cannot be checked against the test2 seal")
    task = json.loads(path.read_text(encoding="utf-8"))
    ids = set(task.get("chapters") or []) | {p.get("chapter_id") for p in task.get("places") or []}
    return ids - {None}


def refuse_sealed(tasks, sealed=None):
    """Stop if any package holds a chapter of the sealed test2. sealed: the sealed ids (tests), default the seal
    file's (tagger.sealed_test2_chapters; tagger imports torch, so it is imported only here, when a run starts)."""
    if sealed is None:
        from novel_ner.tagger import sealed_test2_chapters

        sealed = sealed_test2_chapters()
    sealed = set(sealed)
    for task in tasks:
        hit = sorted(task_chapters(task) & sealed)
        if hit:
            raise SystemExit(f"{Path(task).name} holds {len(hit)} chapters of the sealed test2 (configs/test2_seal.json): "
                             "refused until the seal is opened in stage 3")


def run_many(tasks, label, model, effort, run_id, workers, call_fn=None, sealed=None):
    refuse_sealed(tasks, sealed)
    header = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": git_state(), "claude_cli": claude_version(),
              "model_requested": model, "label": label, "effort": effort, "system_prompt": SYSTEM_PROMPT,
              "sampling": SAMPLING, "prompt_version": PROMPT_VERSION, "isolation": ISOLATION,
              "tasks": [Path(t).name for t in tasks]}
    log = CallLog(run_id, header)

    def one(task):
        try:
            return run_pass(task, label, model, effort, run_id, log, call_fn=call_fn)
        except Exception as e:  # keep the other passes going; the error is reported
            return {"task": Path(task).name, "label": label, "status": f"error: {type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, tasks))
    log.finish(results)
    for r in results:
        print(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("run", "run-test"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("task")
        else:
            p.add_argument("--config", default="configs/test_781_990.json")
        p.add_argument("--model", required=True)
        p.add_argument("--label", required=True, help="names the pass files, e.g. opus_r1")
        p.add_argument("--effort", default="xhigh")
        p.add_argument("--run-id", required=True)
        p.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.label in ("A", "B"):
        raise SystemExit("labels A and B are GPT's passes")
    if args.action == "run":
        tasks = [Path(args.task)]
    else:
        cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
        n_batches = len(test_chapters(Path(args.config).with_suffix(".order.json").as_posix(), cfg["n"])) // cfg["batch_size"]
        tasks = [TASK_ROOT / f"test_b{b:02d}" for b in range(1, n_batches + 1)]
    run_many(tasks, args.label, args.model, args.effort, args.run_id, args.workers)


if __name__ == "__main__":
    main()
