"""Strict entity scoring (annotation guide, appendix 1) and the GPT calibration report.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.score calibrate --task data/llm_tasks/calib \\
        --gold data/gold/dev/<file>.json --run-id 20261001-calib

A prediction counts only if chapter, start, end and type all match. Pending marks in the gold
standard and the chapter's recorded unscored ranges (gold.py revise) are unscored regions: any gold
or predicted span touching one is left out. A draft's
own pending marks are abstentions, reported but not scored; mentions it could not locate count
as wrong predictions. runs/<run-id>/ gets numbers only; the side-by-side report with the
novel's text goes next to the task package under data/.
"""
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.snapshot import REPO_ROOT, load_snapshot


def gold_spans(export, chapter_ids):
    out = {}
    for cid in chapter_ids:
        anns = export["chapters"][cid]["annotations"]
        out[cid] = {
            "spans": {(a["start"], a["end"], a["type"]) for a in anns if not a.get("pending")},
            "unscored": [(a["start"], a["end"]) for a in anns if a.get("pending")]
                        + [(u["start"], u["end"]) for u in export["chapters"][cid].get("unscored", [])],
        }
    return out


def draft_spans(draft, chapter_ids):
    out = {}
    for cid in chapter_ids:
        c = draft["chapters"].get(cid, {"spans": [], "errors": []})
        out[cid] = {
            "spans": {(s["start"], s["end"], s["type"]) for s in c["spans"] if not s["pending"]},
            "pending": [(s["start"], s["end"]) for s in c["spans"] if s["pending"]],
            "errors": len(c["errors"]),
        }
    return out


def _touches(span, regions):
    return any(span[0] < e and s < span[1] for s, e in regions)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def strict(reference, prediction):
    """Micro P/R/F1 of prediction against reference, overall and per type, plus error kinds."""
    tp = fp = fn = invalid = boundary_ok_type_wrong = overlap_only = abstained_on_gold = 0
    per_type = {t: Counter() for t in ENTITY_TYPES}
    for cid, ref in reference.items():
        pred = prediction[cid]
        unscored = ref.get("unscored", [])
        gold = {s for s in ref["spans"] if not _touches(s, unscored)}
        guess = {s for s in pred["spans"] if not _touches(s, unscored)}
        hit = gold & guess
        tp += len(hit)
        fp += len(guess - hit)
        fn += len(gold - hit)
        invalid += pred.get("errors", 0)
        for s in hit:
            per_type[s[2]]["tp"] += 1
        for s in guess - hit:
            per_type[s[2]]["fp"] += 1
        for s in gold - hit:
            per_type[s[2]]["fn"] += 1
        gold_bounds = {(s, e): t for s, e, t in gold - hit}
        for s, e, t in guess - hit:
            if (s, e) in gold_bounds:
                boundary_ok_type_wrong += 1
            elif any(s < ge and gs < e for gs, ge in gold_bounds):
                overlap_only += 1
        abstained_on_gold += sum(1 for s in gold - hit if _touches(s, pred.get("pending", [])))
    fp_all = fp + invalid  # mentions that could not be located are wrong predictions
    return {
        "tp": tp, "fp": fp_all, "fn": fn, **_prf(tp, fp_all, fn),
        "invalid_predictions": invalid,
        "fp_same_span_wrong_type": boundary_ok_type_wrong,
        "fp_overlapping_gold": overlap_only,
        "fn_where_draft_was_pending": abstained_on_gold,
        "per_type": {t: {**_prf(c["tp"], c["fp"], c["fn"]), "support": c["tp"] + c["fn"]} for t, c in per_type.items()},
    }


def disagreements(a, b):
    """Spans the two drafts do not share (by chapter), and pending marks in either: what a person must decide."""
    out = {}
    for cid in a:
        only_a = a[cid]["spans"] - b[cid]["spans"]
        only_b = b[cid]["spans"] - a[cid]["spans"]
        pending = sorted(set(a[cid]["pending"]) | set(b[cid]["pending"]))
        out[cid] = {"only_a": sorted(only_a), "only_b": sorted(only_b), "pending": pending,
                    "to_review": len(only_a) + len(only_b) + len(pending)}
    return out


def _context(text, start, end, width=14):
    left = text[max(0, start - width):start].split("\n")[-1]
    right = text[end:end + width].split("\n")[0]
    return f"{left}【{text[start:end]}】{right}"


def calibrate(task_dir, gold_file, run_id):
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    chapters = task["chapters"]
    export = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    snapshot = load_snapshot(task["snapshot_id"], verify=False)
    gold = gold_spans(export, chapters)
    drafts = {}
    for run in ("A", "B"):
        f = task_dir / f"draft_{run}.json"
        if f.exists():
            drafts[run] = json.loads(f.read_text(encoding="utf-8"))
    if not drafts:
        raise SystemExit("no draft_A.json / draft_B.json yet: import GPT's answers first")
    spans = {run: draft_spans(d, chapters) for run, d in drafts.items()}
    metrics = {"run_id": run_id, "task": task["name"], "chapters": chapters,
               "gold_file": Path(gold_file).name, "gold_provenance": {c: export["chapters"][c].get("provenance") for c in chapters},
               "gold_spans": sum(len(g["spans"]) for g in gold.values()),
               "gold_unscored_regions": sum(len(g["unscored"]) for g in gold.values()),
               "drafts": {run: {"model": d["model"], "instructions_version": d["instructions_version"],
                                "spans": sum(len(s["spans"]) for s in spans[run].values()),
                                "pending": sum(len(s["pending"]) for s in spans[run].values()),
                                "vs_gold": strict(gold, spans[run])} for run, d in drafts.items()}}
    lines = [f"# 校准报告 {run_id}", "", "含小说原文，不入库。【】是范围。", ""]
    if len(drafts) == 2:
        metrics["a_vs_b"] = strict({c: {"spans": spans["A"][c]["spans"], "unscored": []} for c in chapters},
                                   {c: {**spans["B"][c], "pending": []} for c in chapters})
        dis = disagreements(spans["A"], spans["B"])
        metrics["to_review_per_chapter"] = {c: d["to_review"] for c, d in dis.items()}
        lines += ["## 两遍之间不一致（正式标注时要你裁决的）", ""]
        for cid, d in dis.items():
            text = snapshot.chapters[cid].text
            lines.append(f"### {cid}：{d['to_review']} 处")
            for s, e, t in d["only_a"]:
                lines.append(f"- 只有 A 标 {t}：{_context(text, s, e)}")
            for s, e, t in d["only_b"]:
                lines.append(f"- 只有 B 标 {t}：{_context(text, s, e)}")
            for s, e in d["pending"]:
                lines.append(f"- 待审：{_context(text, s, e)}")
            lines.append("")
    for run in drafts:
        lines += [f"## 第 {run} 遍和你的答案对不上的", ""]
        for cid in chapters:
            text = snapshot.chapters[cid].text
            g = {s for s in gold[cid]["spans"] if not _touches(s, gold[cid]["unscored"])}
            p = {s for s in spans[run][cid]["spans"] if not _touches(s, gold[cid]["unscored"])}
            lines.append(f"### {cid}")
            for s, e, t in sorted(g - p):
                lines.append(f"- 漏了（你标 {t}）：{_context(text, s, e)}")
            for s, e, t in sorted(p - g):
                lines.append(f"- 多了或标错（GPT 标 {t}）：{_context(text, s, e)}")
            for err in drafts[run]["chapters"][cid]["errors"]:
                lines.append(f"- 定位失败：{err}")
            lines.append("")
    out = REPO_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    (out / "config.json").write_text(json.dumps({"run_id": run_id, "task": task, "gold_file": Path(gold_file).name,
                                                 "code_commit": commit, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    report = task_dir / f"report_{run_id}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics, report


def agreement(task_dirs, run_id):
    """Pass A against pass B for tasks without gold (the test batches): a quality signal per batch."""
    rows, all_a, all_b = [], {}, {}
    for task_dir in map(Path, task_dirs):
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        files = [task_dir / f"draft_{run}.json" for run in ("A", "B")]
        if not all(f.exists() for f in files):
            rows.append({"task": task["name"], "status": "missing drafts"})
            continue
        a, b = (draft_spans(json.loads(f.read_text(encoding="utf-8")), task["chapters"]) for f in files)
        m = strict({c: {"spans": a[c]["spans"], "unscored": []} for c in task["chapters"]},
                   {c: {**b[c], "pending": []} for c in task["chapters"]})
        dis = disagreements(a, b)
        rows.append({"task": task["name"], "chapters": task["chapters"], "f1_a_vs_b": m["f1"], "shared": m["tp"],
                     "only_a": sum(len(d["only_a"]) for d in dis.values()), "only_b": sum(len(d["only_b"]) for d in dis.values()),
                     "pending_marks": sum(len(d["pending"]) for d in dis.values()),
                     "unlocated": {"A": sum(x["errors"] for x in a.values()), "B": sum(x["errors"] for x in b.values())},
                     "to_review_per_chapter": {c: d["to_review"] for c, d in dis.items()}})
        all_a.update(a)
        all_b.update(b)
    overall = strict({c: {"spans": all_a[c]["spans"], "unscored": []} for c in all_a},
                     {c: {**all_b[c], "pending": []} for c in all_b}) if all_a else None
    out = REPO_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    result = {"run_id": run_id, "tasks": rows,
              "overall_f1_a_vs_b": overall["f1"] if overall else None,
              "to_review_total": sum(sum(r.get("to_review_per_chapter", {}).values()) for r in rows)}
    (out / "agreement.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--task", required=True)
    c.add_argument("--gold", required=True)
    c.add_argument("--run-id", required=True)
    g = sub.add_parser("agreement")
    g.add_argument("tasks", nargs="+")
    g.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if args.action == "agreement":
        result = agreement(args.tasks, args.run_id)
        for r in result["tasks"]:
            print(r["task"], {k: v for k, v in r.items() if k not in ("task", "chapters")})
        print(f"overall A vs B F1 {result['overall_f1_a_vs_b']}; to review in all: {result['to_review_total']}")
        return
    metrics, report = calibrate(args.task, args.gold, args.run_id)
    for run, d in metrics["drafts"].items():
        v = d["vs_gold"]
        print(f"pass {run} vs your answers: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(tp {v['tp']}, fp {v['fp']}, fn {v['fn']}, pending {d['pending']}, invalid {v['invalid_predictions']})")
    if "a_vs_b" in metrics:
        print(f"A vs B: F1={metrics['a_vs_b']['f1']:.3f}; to review per chapter: {metrics['to_review_per_chapter']}")
    print(f"wrote runs/{metrics['run_id']}/metrics.json and {report}")


if __name__ == "__main__":
    main()
