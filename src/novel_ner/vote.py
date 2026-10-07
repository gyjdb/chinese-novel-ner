"""Type vote over the model's own predictions on the training chapters (arm C2; inference only).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.tagger vote --model-run runs/<training run> --predictions runs/<predict run> --run-id <id> [--material-run runs/<predict run>]

Material (the material-scope rules, "全书投票"): the model's predictions on the training chapters, chapters 1-780 of the
model's snapshot minus the dev chapters of the training config's dev_set (the weak build's split "train", 749 chapters),
never dev2, the formal test, the sealed test2, a dev chapter of configs/annotate_dev.json or any chapter after 780
(check_material refuses each by name, with tagger.dev2_chapters, tagger.exam_chapters, tagger.sealed_test2_chapters and
tagger.dev_chapters). Without --material-run the subcommand predicts them itself with the training run's best checkpoint
as an ordinary predict run, runs/<run-id>-material/ (predictions not committed), with the target run's chunking; with
it, that run must be a predict run of the same training run on exactly the material chapter set, with the target run's
chunking (the vote never mixes inference variants). A target run made with a C1 chunking variant needs "c1" in the
vote's run id too, like the predict run itself.

Rule (vote_types): a predicted span of the target run whose text occurs as a predicted span in the material (same
text, any type) takes the majority type over the material's counts plus the target chapter's own predicted spans of
that text (the span itself included); a tie keeps the model's type; a text never seen in the material keeps its type.
Scores are the model's, unchanged. Output: runs/<run-id>/predictions.jsonl (the revised rows, same shape; not
committed), config.json and metrics.json (counts only). The run id must contain "c2", and "test"/"preann" by predict's
rule since the output holds predictions for the target chapters; a target chapter in the sealed test2 or among the
material chapters is refused.
"""
import json
import time
from collections import Counter, defaultdict

import torch

from novel_ner import tagger
from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import sha256, write_jsonl
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.tagger import (DEFAULT_CHUNKING, MAX_TRAIN_CHAPTER, dev2_chapters, dev_chapters, exam_chapters, plain_run_id,
                              predict_run, refuse_sealed, sealed_test2_chapters)
from novel_ner.weak import ids_sha256
from novel_ner.zeroshot import environment, git_state

MATERIAL_RULE = ("the model's predictions on the training chapters: chapters 1-780 of the model's snapshot minus the "
                 "dev chapters of the training config's dev_set (and of configs/annotate_dev.json by name); never dev2, "
                 "the formal test, test2 or chapters after 780 (the material-scope rules); the material and the target are "
                 "predicted with the same chunking")
VOTE_RULE = ("a predicted span whose text occurs as a predicted span in the material (same text, any type) takes the "
             "majority type over the material counts plus the target chapter's own predicted spans of that text (itself "
             "included); a tie keeps the model's type; a text unseen in the material keeps its type; scores unchanged")


def type_votes(rows):
    """{text: Counter(type)} over prediction rows."""
    out = defaultdict(Counter)
    for r in rows:
        out[r["text"]][r["type"]] += 1
    return out


def vote_types(target_rows, material_rows):
    """(the target rows with VOTE_RULE applied, counts): spans_seen, spans_with_material, spans_unseen,
    spans_changed, ties and changes_by_type {"from>to": n}."""
    material = type_votes(material_rows)
    own = defaultdict(list)
    for r in target_rows:
        own[r["chapter_id"]].append(r)
    own = {cid: type_votes(rows) for cid, rows in own.items()}
    out, stats, changes = [], Counter(), Counter()
    for keys in ("spans_seen", "spans_with_material", "spans_unseen", "spans_changed", "ties"):
        stats[keys] += 0
    for r in target_rows:
        stats["spans_seen"] += 1
        counts = material.get(r["text"])
        if counts is None:
            stats["spans_unseen"] += 1
            out.append(dict(r))
            continue
        stats["spans_with_material"] += 1
        total = counts + own[r["chapter_id"]][r["text"]]
        top = max(total.values())
        winners = [t for t, n in total.items() if n == top]
        new = r["type"]
        if len(winners) > 1:
            stats["ties"] += 1
        else:
            new = winners[0]
        if new != r["type"]:
            stats["spans_changed"] += 1
            changes[f"{r['type']}>{new}"] += 1
        out.append({**r, "type": new})
    return out, {**stats, "changes_by_type": dict(sorted(changes.items()))}


def check_material(chapter_ids, snapshot, dev_ids):
    """Refuse a material chapter that is in the sealed test2, dev2, the formal test, a dev chapter, after chapter 780,
    not in the snapshot, or listed twice."""
    for name, forbidden in (("sealed test2 (" + tagger.SEAL_FILE.name + ")", sealed_test2_chapters()),
                            ("dev2 decision (" + tagger.DEV2_FILE.name + ")", dev2_chapters()),
                            ("formal test (" + tagger.TEST_FILE.name + ")", exam_chapters()), ("dev", dev_ids),
                            ("dev set (" + tagger.DEV_FILE.name + ")", dev_chapters())):
        bad = [c for c in chapter_ids if c in forbidden]
        if bad:
            raise SystemExit(f"{len(bad)} material chapters are {name} chapters: the vote may only use the training chapters")
    missing = [c for c in chapter_ids if c not in snapshot.chapters]
    if missing or len(set(chapter_ids)) != len(chapter_ids):
        raise SystemExit(f"material chapters missing from snapshot {snapshot.id} or listed twice: {missing[:5]}")
    late = [c for c in chapter_ids if snapshot.chapters[c].number > MAX_TRAIN_CHAPTER]
    if late:
        raise SystemExit(f"{len(late)} material chapters are after chapter {MAX_TRAIN_CHAPTER}: the vote may only use the training chapters")


def read_prediction_run(ref, what):
    """A finished predict run's record, metrics and rows, checked: config.json, metrics.json and predictions.jsonl
    exist, the record is a predict run's (chapters, model_run) and the rows are the file metrics.json recorded."""
    rid = plain_run_id(ref)
    d = REPO_ROOT / "runs" / rid
    for name in ("config.json", "metrics.json", "predictions.jsonl"):
        if not (d / name).is_file():
            raise SystemExit(f"{what} {rid}: no {name} (not a finished predict run)")
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    if "chapters" not in config or "model_run" not in config:
        raise SystemExit(f"{what} {rid}: config.json is not a predict run's record (no chapters or model_run)")
    if metrics.get("predictions_sha256") != sha256(d / "predictions.jsonl"):
        raise SystemExit(f"{what} {rid}: predictions.jsonl differs from the file metrics.json recorded")
    return {"run_id": rid, "config": config, "metrics": metrics, "rows": read_jsonl(d / "predictions.jsonl"),
            "config_sha256": sha256(d / "config.json"), "predictions_sha256": metrics["predictions_sha256"]}


def material_chapters(cfg, snapshot):
    """(the material chapter ids sorted, the dev chapter ids): chapters 1-780 of the snapshot minus the training
    config's dev set (annotate.chapter_order of cfg["dev_set"])."""
    dev_ids = frozenset(chapter_order(json.loads((REPO_ROOT / cfg["dev_set"]).read_text(encoding="utf-8"))))
    ids = sorted(c for c, ch in snapshot.chapters.items() if ch.number <= MAX_TRAIN_CHAPTER and c not in dev_ids)
    return ids, dev_ids


def vote_run(model_run, predictions_run, run_id, material_run=None, snapshot=None):
    """The C2 vote as a run: revise the types of predict run `predictions_run` (made by training run `model_run`)
    with the material of VOTE_RULE; writes runs/<run_id>/ (predictions.jsonl not committed, config.json and
    metrics.json counts only). snapshot: the model's snapshot when already loaded (tests)."""
    began = time.perf_counter()
    run_id = plain_run_id(run_id)
    if "c2" not in run_id:
        raise SystemExit("the vote is the inference variant C2: the run id must contain 'c2'")
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    model_dir = REPO_ROOT / "runs" / plain_run_id(model_run)
    trained = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = trained["config"]
    target = read_prediction_run(predictions_run, "predictions run")
    if target["config"]["model_run"] != trained["run_id"]:
        raise SystemExit(f"predictions run {target['run_id']} was made by {target['config']['model_run']!r}, not by "
                         f"{trained['run_id']!r}: the vote material and the target must come from one model")
    target_ids = list(target["config"]["chapters"])
    refuse_sealed(target_ids)
    exam = target["metrics"]["chapters_after_780"] - target["metrics"]["chapters_after_990"]
    if exam and "test" not in run_id:
        raise SystemExit(f"{exam} target chapters are in the test range: the run id must contain 'test'")
    if target["metrics"]["chapters_after_990"] and "preann" not in run_id:
        raise SystemExit(f"{target['metrics']['chapters_after_990']} target chapters are after 990: the run id must contain 'preann'")
    chunking = target["config"].get("chunking", DEFAULT_CHUNKING)
    if chunking != DEFAULT_CHUNKING and "c1" not in run_id:
        raise SystemExit(f"the target run uses chunking {chunking!r} (the inference variant C1), so the vote's output and its "
                         "material do too: the run id must contain 'c1'")
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    ids, dev_ids = material_chapters(cfg, snapshot)
    check_material(ids, snapshot, dev_ids)
    overlap = sorted(set(target_ids) & set(ids))
    if overlap:
        raise SystemExit(f"{len(overlap)} target chapters are material (training) chapters: their own predictions would "
                         "count twice; the vote is for chapters outside the training chapters")
    if material_run:
        material = read_prediction_run(material_run, "material run")
        if material["config"]["model_run"] != trained["run_id"]:
            raise SystemExit(f"material run {material['run_id']} was made by {material['config']['model_run']!r}, not by {trained['run_id']!r}")
        check_material(list(material["config"]["chapters"]), snapshot, dev_ids)
        if set(material["config"]["chapters"]) != set(ids):
            raise SystemExit(f"material run {material['run_id']} covers {len(material['config']['chapters'])} chapters, not exactly "
                             f"the {len(ids)} training chapters of {trained['run_id']}")
        its_chunking = material["config"].get("chunking", DEFAULT_CHUNKING)
        if its_chunking != chunking:
            raise SystemExit(f"material run {material['run_id']} uses chunking {its_chunking!r}, the target run {chunking!r}: "
                             "the vote must not mix inference variants")
        predicted_here = False
    else:
        predict_run(model_run, ids, f"{run_id}-material", snapshot=snapshot, chunking=chunking)
        material = read_prediction_run(f"{run_id}-material", "material run")
        predicted_here = True
    rows, counts = vote_types(target["rows"], material["rows"])
    by_text = type_votes(material["rows"])
    record = {"run_id": run_id, "model_run": trained["run_id"], "model_run_config_sha256": sha256(model_dir / "config.json"),
              "setting": trained.get("setting"), "seed": trained.get("seed"),
              "predictions_run": {"run_id": target["run_id"], "config_sha256": target["config_sha256"],
                                  "predictions_sha256": target["predictions_sha256"], "chunking": chunking,
                                  "chapters": target_ids},
              "material_run": {"run_id": material["run_id"], "predicted_here": predicted_here,
                               "config_sha256": material["config_sha256"], "predictions_sha256": material["predictions_sha256"],
                               "chunking": material["config"].get("chunking", DEFAULT_CHUNKING), "chapters": len(ids),
                               "chapter_ids_sha256": ids_sha256(ids), "chapter_ids_sha256_rule": "weak.ids_sha256 of the sorted ids",
                               "equals_model_training_chapters": (trained.get("weak_labels") or {}).get("train_ids_sha256") == ids_sha256(ids),
                               "dev_chapters_excluded": len(dev_ids), "dev_set": cfg["dev_set"]},
              "material_rule": MATERIAL_RULE, "vote_rule": VOTE_RULE,
              "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
              "code": git_state(), "environment": {**environment(), "torch_cuda": torch.version.cuda},
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    write_jsonl(out_dir / "predictions.jsonl", rows)
    metrics = {"run_id": run_id, "chapters": len(target_ids), "material_chapters": len(ids), "material_spans": len(material["rows"]),
               "material_distinct_texts": len(by_text), "material_texts_with_more_than_one_type": sum(len(c) > 1 for c in by_text.values()),
               **counts, "outputs": len(rows),
               "outputs_by_type_before": {t: sum(r["type"] == t for r in target["rows"]) for t in ENTITY_TYPES},
               "outputs_by_type": {t: sum(r["type"] == t for r in rows) for t in ENTITY_TYPES},
               "predictions_sha256": sha256(out_dir / "predictions.jsonl"), "seconds": round(time.perf_counter() - began, 2),
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{counts['spans_changed']} of {counts['spans_seen']} spans changed type ({counts['spans_with_material']} had material, "
          f"{counts['ties']} ties); wrote {out_dir}")
    return metrics
