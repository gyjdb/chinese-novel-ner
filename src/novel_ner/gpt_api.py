"""Run GPT annotation passes through the OpenAI API (the user's own key and account).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.gpt_api models --match 6.1 sol
    .venv\\Scripts\\python -m novel_ner.gpt_api run data/llm_tasks/calib_api --model <id> --run-id <id>
    .venv\\Scripts\\python -m novel_ner.gpt_api run-test --model <id> --run-id <id>

Each pass is its own request carrying what a person pastes into a fresh chat: the one-line request,
then instructions.md, guide.md and chapters.md (the parts every request shares come first, so the API
can cache them). Nothing is kept on OpenAI's side (store=False). The raw reply stays next to the
package under data/ (not committed); runs/<run-id>/api_calls.json and runs/api_ledger.json keep model,
settings, token counts and estimated cost per request (no novel text). A reply that is not valid JSON
is retried once and the retry is recorded. Before every request the budget guard checks that the
estimated spend so far (all runs, from the ledger) plus a typical request stays under the cap.
The key is read from OPENAI_API_KEY (process, or the Windows user environment) and never printed.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from novel_ner.annotate import test_chapters
from novel_ner.llm_task import TASK_ROOT, import_answer
from novel_ner.snapshot import REPO_ROOT

REQUEST = "请按 instructions.md 标注 chapters.md 里的全部章节，只输出 JSON。"
PROMPT_VERSION = "api_prompt_v1"
# What one request carries: the one-line request, then these files in this order (the parts every request shares first,
# so the API can cache them), under this cache key; "import" says whether the reply is an annotation answer that
# llm_task.import_answer turns into a draft. The default is the drafting pass; a module that sends another kind of package
# through the same mechanics (novel_ner.qa, the exclusion check) passes its own spec.
ANNOTATION_PROMPT = {"version": PROMPT_VERSION, "request": REQUEST, "files": ("instructions.md", "guide.md", "chapters.md"),
                     "cache_key": "novel-ner-annotation", "import": True}
# USD per million tokens. Third-party quote for GPT-6 Sol (2026-10-01); check OpenAI's price page.
PRICES = {"input": 2.0, "cached_input": 0.2, "output": 10.0}
LEDGER = REPO_ROOT / "runs" / "api_ledger.json"
FIRST_GUESS_USD = 1.5  # a typical request before any has been measured (generous on purpose)


def api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key and sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "OPENAI_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set; set it as a user environment variable (the key is never printed here)")
    return key


def client():
    from openai import OpenAI

    return OpenAI(api_key=api_key(), timeout=1800, max_retries=2)


def build_prompt(task_dir, prompt=ANNOTATION_PROMPT):
    parts = [prompt["request"], ""]
    for name in prompt["files"]:
        parts += [f"===== {name} =====", (Path(task_dir) / name).read_text(encoding="utf-8").strip(), ""]
    return "\n".join(parts)


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in text else ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def usage_of(resp):
    u = resp.usage
    cached = (u.input_tokens_details.cached_tokens if u.input_tokens_details else 0) or 0
    reasoning = (u.output_tokens_details.reasoning_tokens if u.output_tokens_details else 0) or 0
    return {"input_tokens": u.input_tokens, "cached_tokens": cached, "output_tokens": u.output_tokens,
            "reasoning_tokens": reasoning}


def cost(usage, prices=PRICES):
    fresh = usage["input_tokens"] - usage["cached_tokens"]
    return (fresh * prices["input"] + usage["cached_tokens"] * prices["cached_input"]
            + usage["output_tokens"] * prices["output"]) / 1e6


class Budget:
    """Estimated spend across all runs (the ledger); a request reserves a typical cost before it starts."""

    def __init__(self, cap, ledger=LEDGER):
        self.cap, self.ledger, self.lock, self.reserved = cap, Path(ledger), threading.Lock(), 0.0
        self.entries = json.loads(self.ledger.read_text(encoding="utf-8"))["requests"] if self.ledger.exists() else []

    @property
    def spent(self):
        return sum(e["estimated_usd"] for e in self.entries)

    def typical(self):
        costs = [e["estimated_usd"] for e in self.entries]
        return max(costs) if costs else FIRST_GUESS_USD

    def reserve(self):
        """Reserve a typical request's cost; None if that would pass the cap."""
        with self.lock:
            amount = self.typical()
            if self.spent + self.reserved + amount > self.cap:
                return None
            self.reserved += amount
            return amount

    def release(self, amount):
        with self.lock:
            self.reserved -= amount

    def record(self, entry):
        with self.lock:
            self.entries.append(entry)
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            self.ledger.write_text(json.dumps({"prices_usd_per_million": PRICES, "total_estimated_usd": round(self.spent, 4),
                                               "requests": self.entries}, ensure_ascii=False, indent=1), encoding="utf-8")


class BudgetStop(Exception):
    pass


def run_pass(api, task_dir, run, model, effort, budget, run_id, prompt=ANNOTATION_PROMPT):
    """One independent pass: request, keep the raw reply, write pass_<run>.json, import it (when the prompt spec says
    the reply is an annotation answer; see ANNOTATION_PROMPT)."""
    task_dir = Path(task_dir)
    pass_file = task_dir / f"pass_{run}.json"
    if pass_file.exists():
        return {"task": task_dir.name, "run": run, "status": "exists"}
    prompt_text = build_prompt(task_dir, prompt)
    for attempt in (1, 2):
        held = budget.reserve()
        if held is None:
            raise BudgetStop(f"estimated spend {budget.spent:.2f} USD + a typical request would pass the cap {budget.cap} USD")
        try:
            started = time.time()
            resp = api.responses.create(model=model, input=prompt_text, reasoning={"effort": effort}, store=False,
                                        prompt_cache_key=prompt["cache_key"])
        finally:
            budget.release(held)
        usage = usage_of(resp)
        entry = {"run_id": run_id, "task": task_dir.name, "run": run, "attempt": attempt, "response_id": resp.id,
                 "model": resp.model, "effort": effort, "prompt_version": prompt["version"], "status": resp.status,
                 "seconds": round(time.time() - started, 1), **usage, "estimated_usd": round(cost(usage), 4),
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (task_dir / f"api_{run}_attempt{attempt}.json").write_text(json.dumps(
            {"record": entry, "output_text": resp.output_text}, ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            answer = extract_json(resp.output_text)
        except (ValueError, json.JSONDecodeError):
            entry["invalid_json"] = True
            budget.record(entry)
            continue
        budget.record(entry)
        pass_file.write_text(json.dumps(answer, ensure_ascii=False, indent=1), encoding="utf-8")
        if prompt["import"]:
            import_answer(pass_file, run, f"{resp.model}（API，reasoning effort {effort}，{prompt['version']}）")
        return {"task": task_dir.name, "run": run, "status": "ok", "attempts": attempt}
    return {"task": task_dir.name, "run": run, "status": "invalid_json_twice"}


def run_many(jobs, model, effort, cap, run_id, workers, prompt=ANNOTATION_PROMPT):
    api = client()
    budget = Budget(cap)
    results, stopped = [], []

    def one(job):
        try:
            return run_pass(api, job[0], job[1], model, effort, budget, run_id, prompt)
        except BudgetStop as e:
            stopped.append(str(e))
            return {"task": Path(job[0]).name, "run": job[1], "status": "budget_stop"}
        except Exception as e:  # keep the other passes going; the error is reported
            return {"task": Path(job[0]).name, "run": job[1], "status": f"error: {type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    out = REPO_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    mine = [e for e in budget.entries if e["run_id"] == run_id]
    (out / "api_calls.json").write_text(json.dumps({
        "run_id": run_id, "model": model, "effort": effort, "prompt_version": prompt["version"],
        "prompt_files": list(prompt["files"]),
        "prices_usd_per_million": PRICES, "budget_cap_usd": cap, "results": results, "requests": mine,
        "estimated_usd_this_run": round(sum(e["estimated_usd"] for e in mine), 4),
        "estimated_usd_all_runs": round(budget.spent, 4)}, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:
        print(r)
    print(f"estimated spend: this run {sum(e['estimated_usd'] for e in mine):.2f} USD, all runs {budget.spent:.2f} USD "
          f"(cap {cap}); {stopped[0] if stopped else ''}")
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    m = sub.add_parser("models")
    m.add_argument("--match", nargs="*", default=[])
    for name in ("run", "run-test"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("task", nargs="+", help="one or more task packages, run in one process under one budget")
        else:
            p.add_argument("--config", default="configs/test_781_990.json")
        p.add_argument("--model", required=True)
        p.add_argument("--effort", default="xhigh")
        p.add_argument("--runs", nargs="+", default=["A", "B"])
        p.add_argument("--budget", type=float, default=30.0, help="cap on the estimated spend of all runs, USD")
        p.add_argument("--run-id", required=True)
        p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.action == "models":
        ids = sorted(m.id for m in client().models.list())
        print("\n".join(i for i in ids if all(s.lower() in i.lower() for s in args.match)))
        return
    if args.action == "run":
        tasks = [Path(t) for t in args.task]
    else:
        cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
        n_batches = len(test_chapters(Path(args.config).with_suffix(".order.json").as_posix(), cfg["n"])) // cfg["batch_size"]
        tasks = [TASK_ROOT / f"test_b{b:02d}" for b in range(1, n_batches + 1)]
    jobs = [(t, run) for t in tasks for run in args.runs]
    run_many(jobs, args.model, args.effort, args.budget, args.run_id, args.workers)


if __name__ == "__main__":
    main()
