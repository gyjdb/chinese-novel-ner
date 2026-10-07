import hashlib
import inspect
import json

import pytest

from novel_ner import llm_task, tagger, teacher
from novel_ner.evaluate import sha256
from novel_ner.gold import FORMAT
from novel_ner.snapshot import Chapter, Snapshot
from novel_ner.weak import ids_sha256

# text: "林小雨看着林小雨的影子。\n她去了灰塔城。"  林小雨 [0,3) and [5,8), 影子 [9,11), 灰塔城 [16,19)
PARAS = [("p0001", "林小雨看着林小雨的影子。"), ("p0002", "她去了灰塔城。")]
DEV = ["ch0006", "ch0003"]  # the dev config's order


def chapter(n):
    return Chapter(f"ch{n:04d}", n, PARAS, f"第{n}章", f"h{n}")


def snap():
    chapters = [chapter(n) for n in range(1, 13)] + [chapter(781), chapter(782)]  # 781 is the test chapter, 782 only late
    return Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in chapters}, [], [])


def config():
    return {"snapshot": "snap-test", "dev_config": "configs/dev.json", "max_train_chapter": 780, "n": 10, "seed": 7,
            "batch_size": 5, "strata": 4, "dev_pilot": {"n": 2, "seed": 7},
            "packs": {"prefix": "teach_b", "dev_prefix": "teach_dev_b", "pilot_training_packs": 1},
            "chapters_file": "configs/teach.chapters.json", "frozen": {"instructions_version": "v1", "guide_version": "1.1"},
            "exclude": {"test": "configs/test.json", "dev2": "configs/dev2.json", "test2": "configs/test2.json"},
            "teacher": {"label": "opus_t1", "model": "claude-test", "effort": "xhigh", "workers": 1},
            "weak_build": "wb", "weak_setting": "S3", "labels_build_id": "tb", "gold_dev": "data/gold/dev/answers.json"}


def weak_row(c, split):
    return {"chapter_id": c.chapter_id, "number": c.number, "split": split,
            "spans": [{"start": 0, "end": 3, "type": "person", "text": "林小雨", "sources": [["chapter_list", "person"]]},
                      {"start": 5, "end": 8, "type": "person", "text": "林小雨", "sources": [["chapter_list", "person"]]}],
            "uncertain": [{"start": 16, "end": 19, "text": "灰塔城", "types": ["location"], "reason": "other_chapters_only",
                           "sources": [["other_chapters", "location"]]}]}


def mentions(full=True):
    """The teacher's answer for one chapter: two positives, one pending, one that cannot be located, one overlap."""
    m = [{"paragraph_id": "p0001", "text": "林小雨", "occurrence": 1, "type": "person"},
         {"paragraph_id": "p0001", "text": "林小雨", "occurrence": 2, "type": "person"},
         {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": "location"},
         {"paragraph_id": "p0001", "text": "影子", "occurrence": 1, "type": None, "pending": True, "stuck": ["qualification"],
          "candidates": "item / 不标", "doubt": "?"}]
    if full:
        m += [{"paragraph_id": "p0002", "text": "白塔城", "occurrence": 1, "type": "location"},  # not in the text
              {"paragraph_id": "p0001", "text": "小雨", "occurrence": 1, "type": "item"}]  # overlaps the first 林小雨
    return m


def answer(chapter_ids, no_list=()):
    return {"instructions_version": "v1", "guide_version": "1.1",
            "chapters": [{"chapter_id": c, "mentions": "none" if c in no_list else mentions()} for c in chapter_ids]}


def gold_export(chapters):
    return {"format": FORMAT, "set": "dev", "snapshot_id": "snap-test", "guide_version": "1.1", "precedents": [], "chapters": {
        c.chapter_id: {"content_hash": c.content_hash, "text_sha256": hashlib.sha256(c.text.encode("utf-8")).hexdigest(),
                       "status": "done", "provenance": "manual", "checks": {"a": True}, "annotations": [
                           {"start": 0, "end": 3, "text": "林小雨", "type": "person", "pending": False},
                           {"start": 5, "end": 8, "text": "林小雨", "type": "person", "pending": False},
                           {"start": 9, "end": 11, "text": "影子", "type": None, "pending": True, "stuck": ["qualification"],
                            "candidates": "item / 不标", "doubt": "?"},
                           {"start": 16, "end": 19, "text": "灰塔城", "type": "location", "pending": False}]}
        for c in chapters}}


def ext_snap():
    """40 training chapters for the extension tests (the base sample takes 10, the pool is 28), plus 781 and 782."""
    chapters = [chapter(n) for n in range(1, 41)] + [chapter(781), chapter(782)]
    return Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in chapters}, [], [])


def ext_weak_row(c, split):
    """weak_row plus number % 6 non-person spans (signal_a; ties on purpose). Offsets past the text are never
    checked by extend and are copied as they are by the labels build."""
    row = weak_row(c, split)
    row["spans"] = row["spans"] + [{"start": 20 + 2 * i, "end": 21 + 2 * i, "type": "item", "text": "x", "sources": [["dictionary", "item"]]}
                                   for i in range(c.number % 6)]
    return row


def ext_config():
    """An extension config in the shape of configs/teacher_train300.json: 10 new chapters (A 5, B 5) in packs
    teach_b03 and teach_b04; the weak file and the three model runs are named in the rules as in the real config."""
    return {"base_config": "configs/teach.json", "base_chapters_file": "configs/teach.chapters.json", "base_labels_build": "tb",
            "snapshot": "snap-test", "dev_config": "configs/dev.json", "max_train_chapter": 780,
            "exclude": {"test": "configs/test.json", "dev2": "configs/dev2.json", "test2": "configs/test2.json"},
            "pool_rule": "chapters 1..780 of the snapshot minus the dev chapters minus the base chapters (28 chapters)",
            "n_new": 10,
            "selection": {
                "signal_a": {"name": "non_person_weak_spans", "rule": "non-person positive spans in data/weak_labels/wb/labels_S3.jsonl"},
                "signal_b": {"name": "student_uncertainty", "rule": "the three models (runs/m1, runs/m2, runs/m3) predict every pool "
                             "chapter (run ids <model run>-predict-pool28); u as the real rule", "low_score": 0.9, "min_union_spans": 3},
                "rule": "A = top n/2 by signal_a, B = top n/2 by signal_b among the rest; ties lower number first",
                "packs": {"prefix": "teach_b", "first_index": 3, "batch_size": 5}},
            "teacher": {"label": "opus_t1", "model": "claude-test", "effort": "xhigh", "workers": 1},
            "frozen": {"instructions_version": "v1", "guide_version": "1.1"}, "labels_build_id": "tb300"}


def expected_signals(n):
    """The fake students' (fake_predict) union, uncertain count and u for chapter n, and signal_a of ext_weak_row."""
    union = 2 + n % 5
    uncertain = n % 5 + (1 if n % 7 == 0 else 0)
    return {"signal_a": n % 6, "union": union, "uncertain": uncertain, "u": uncertain / union if union >= 3 else 0.0}


def fake_predict(root, s, calls):
    """A predict_pool stand-in writing runs/<run id>/ as tagger.predict_run does: three seeds m1..m3; every seed gives
    both person spans (score 0.99; m2 scores the second one 0.5 when number % 7 == 0), m1 alone adds number % 5 item
    spans (not predicted by all three)."""
    def run(model_run, chapter_ids, run_id):
        rows = []
        for cid in chapter_ids:
            n = s.chapters[cid].number
            rows.append({"chapter_id": cid, "start": 0, "end": 3, "text": "林小雨", "type": "person", "score": 0.99})
            rows.append({"chapter_id": cid, "start": 5, "end": 8, "text": "林小雨", "type": "person",
                         "score": 0.5 if model_run == "m2" and n % 7 == 0 else 0.99})
            if model_run == "m1":
                rows += [{"chapter_id": cid, "start": 10 + i, "end": 11 + i, "text": "x", "type": "item", "score": 0.95} for i in range(n % 5)]
        out = root / "runs" / run_id
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps({"run_id": run_id, "model_run": model_run, "chapters": list(chapter_ids),
                                                     "chunking": "paragraph500", "snapshot": {"id": s.id, "source_commit": "c"}}), encoding="utf-8")
        (out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps({"run_id": run_id, "outputs": len(rows),
                                                      "predictions_sha256": sha256(out / "predictions.jsonl")}), encoding="utf-8")
        calls.append(run_id)
    return run


def make_repo(tmp_path, monkeypatch, s, weak_fn):
    """A repository layout under tmp_path: configs, the weak build, the instructions; no data/gold directory."""
    monkeypatch.setattr(teacher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(teacher, "TASK_ROOT", tmp_path / "data" / "llm_tasks")
    monkeypatch.setattr(llm_task, "TASK_ROOT", tmp_path / "data" / "llm_tasks")
    monkeypatch.setattr(llm_task, "load_snapshot", lambda sid, **kw: s)
    monkeypatch.setattr(teacher, "load_snapshot", lambda sid, **kw: s)
    monkeypatch.setattr(teacher, "chapter_order", lambda cfg: list(DEV))
    monkeypatch.setattr(teacher, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(teacher, "environment", lambda: {"python": "test"})
    (tmp_path / "docs").mkdir()
    for key, store in (("v1", llm_task.INSTRUCTIONS), ("1.1", llm_task.GUIDES)):
        f = tmp_path / "docs" / f"{key}.md"
        f.write_text(f"{key} 内容", encoding="utf-8")
        monkeypatch.setitem(store, key, f)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "dev.json").write_text("{}", encoding="utf-8")
    (configs / "test.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
    (configs / "test.order.json").write_text(json.dumps({"order": [{"chapter_id": "ch0781"}, {"chapter_id": "ch0790"}]}), encoding="utf-8")
    (configs / "dev2.json").write_text(json.dumps({"chapters": ["ch0795"]}), encoding="utf-8")
    (configs / "test2.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
    (configs / "test2.order.json").write_text(json.dumps({"order": [{"chapter_id": "ch0999"}]}), encoding="utf-8")
    labels = tmp_path / "data" / "weak_labels" / "wb" / "labels_S3.jsonl"
    labels.parent.mkdir(parents=True)
    rows = [weak_fn(c, "dev" if c.chapter_id in DEV else "train") for c in s.chapters.values() if c.number <= 780]
    labels.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "runs" / "wb").mkdir(parents=True)
    (tmp_path / "runs" / "wb" / "config.json").write_text(json.dumps(
        {"snapshot": {"id": "snap-test"}, "labels": {"S3": {"sha256": sha256(labels)}},
         "split": {"train_ids_sha256": ids_sha256(r["chapter_id"] for r in rows if r["split"] == "train")}}), encoding="utf-8")
    return tmp_path, s


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return make_repo(tmp_path, monkeypatch, snap(), weak_row)


@pytest.fixture
def ext_repo(tmp_path, monkeypatch):
    """The extension layout: 40 training chapters, the base config written to configs/teach.json and sampled (10
    chapters in teach_b01, teach_b02), predict_pool replaced by fake_predict (the run ids it made are in `calls`)."""
    root, s = make_repo(tmp_path, monkeypatch, ext_snap(), ext_weak_row)
    (root / "configs" / "teach.json").write_text(json.dumps(config()), encoding="utf-8")
    teacher.sample(config(), "configs/teach.json", snapshot=s)
    calls = []
    monkeypatch.setattr(teacher, "predict_pool", fake_predict(root, s, calls))
    return root, s, calls


def ext_cfg():
    return teacher.resolve_config(ext_config(), "configs/teach300.json")


def ext_pool(root, s):
    """The pool the rule gives: chapters 1..40 minus the dev chapters minus the base sample, by number."""
    base = json.loads((root / "configs" / "teach.chapters.json").read_text(encoding="utf-8"))
    base_ids = {c for name in base["training_packs"] for c in base["packs"][name]}
    return [c.chapter_id for c in sorted(s.chapters.values(), key=lambda c: c.number)
            if c.number <= 780 and c.chapter_id not in DEV and c.chapter_id not in base_ids]


def expected_lists(pool, s):
    num = lambda c: s.chapters[c].number
    a = sorted(pool, key=lambda c: (-expected_signals(num(c))["signal_a"], num(c)))[:5]
    b = sorted([c for c in pool if c not in a], key=lambda c: (-expected_signals(num(c))["u"], num(c)))[:5]
    return a, b


def write_pass(root, name, chapter_ids, label="opus_t1", no_list=(), run_id="opus-run-1"):
    task = root / "data" / "llm_tasks" / name
    (task / f"pass_{label}.json").write_text(json.dumps(answer(chapter_ids, no_list), ensure_ascii=False), encoding="utf-8")
    (task / f"claude_{label}_attempt1.json").write_text(json.dumps({"record": {"run_id": run_id, "attempt": 1}, "result": "..."}), encoding="utf-8")


def test_sample_draws_training_chapters_by_stratum_and_a_dev_pilot(repo):
    root, s = repo
    cfg = config()
    data = teacher.sample(cfg, "configs/teach.json", snapshot=s)
    out = json.loads((root / "configs" / "teach.chapters.json").read_text(encoding="utf-8"))
    assert out["order"] == data["order"] and out["code_commit"] == "abc" and out["snapshot"] == "snap-test"
    numbers = [row["number"] for row in data["order"]]
    assert sorted(numbers) == [1, 2, 4, 5, 7, 8, 9, 10, 11, 12]  # the pool: 1-12 minus the dev chapters 3 and 6
    assert data["pool"] == {"chapters": 10, "numbers": [1, 12], "max_train_chapter": 780, "dev_chapters_excluded": 2, "rule": data["pool"]["rule"]}
    assert [v["chapters"] for v in data["strata"].values()] == [2, 3, 2, 3]  # equal-count ranges, sorted by number
    assert all(v["sampled"] == v["chapters"] for v in data["strata"].values())
    assert data["training_packs"] == ["teach_b01", "teach_b02"] and data["dev_pack"] == "teach_dev_b01"
    assert [len(data["packs"][p]) for p in ("teach_b01", "teach_b02")] == [5, 5]
    assert data["packs"]["teach_b01"] == [r["chapter_id"] for r in data["order"][:5]]
    assert sorted(data["packs"]["teach_dev_b01"]) == sorted(DEV) and data["dev_pilot"]["chapters"] == ["ch0003", "ch0006"]
    assert data["pilot"] == ["teach_b01", "teach_dev_b01"]
    assert teacher.sample(cfg, "configs/teach.json", snapshot=s)["order"] == data["order"]  # unchanged: no error
    with pytest.raises(SystemExit, match="different sample"):
        teacher.sample({**cfg, "seed": 8}, "configs/teach.json", snapshot=s)
    assert teacher.sample({**cfg, "seed": 8}, "configs/teach.json", snapshot=s, force=True)["order"] != data["order"]
    with pytest.raises(SystemExit, match="multiple of batch_size"):
        teacher.sample({**cfg, "n": 7}, "configs/teach.json", snapshot=s, force=True)
    # every prefix of the stratified order is spread over the strata: the first pack holds chapters of at least 3 strata
    teacher.sample(cfg, "configs/teach.json", snapshot=s, force=True)
    assert len({r["stratum"] for r in data["order"][:5]}) >= 3


def test_build_packs_checks_every_chapter_and_never_overwrites(repo):
    root, s = repo
    cfg = config()
    teacher.sample(cfg, "configs/teach.json", snapshot=s)
    built, kept = teacher.build_packs(cfg, "pilot", snapshot=s)
    assert built == ["teach_b01", "teach_dev_b01"] and kept == []
    task = json.loads((root / "data" / "llm_tasks" / "teach_b01" / "task.json").read_text(encoding="utf-8"))
    assert task["chapters"] == teacher.read_chapters(cfg)["packs"]["teach_b01"] and task["snapshot_id"] == "snap-test"
    assert (root / "data" / "llm_tasks" / "teach_b01" / "chapters.md").exists()
    built, kept = teacher.build_packs(cfg, "all", snapshot=s)
    assert built == ["teach_b02"] and kept == ["teach_b01", "teach_dev_b01"]
    assert teacher.select_packs(teacher.read_chapters(cfg), "rest") == ["teach_b02"]
    assert teacher.select_packs(teacher.read_chapters(cfg), "all", with_dev=False) == ["teach_b01", "teach_b02"]
    # a late chapter, a dev chapter in a training pack, a test / dev2 / test2 chapter, a training chapter in the dev pack
    data = teacher.read_chapters(cfg)
    chapters_file = root / "configs" / "teach.chapters.json"
    for pack, bad, word in (("teach_b02", "ch0782", "after chapter 780"), ("teach_b02", "ch0003", "dev chapter"),
                            ("teach_b02", "ch0781", "test chapter"), ("teach_b02", "ch0795", "dev2 chapter"),
                            ("teach_b02", "ch0999", "test2 chapter"), ("teach_b02", "ch0790", "not in snapshot"),
                            ("teach_dev_b01", "ch0001", "not a dev chapter")):
        wrong = json.loads(json.dumps(data))
        wrong["packs"][pack] = [bad] + wrong["packs"][pack][1:]
        chapters_file.write_text(json.dumps(wrong), encoding="utf-8")
        with pytest.raises(SystemExit, match=word):
            teacher.build_packs(cfg, "all", snapshot=s)
    chapters_file.write_text(json.dumps(data), encoding="utf-8")


def test_check_counts_mentions_pending_overlaps_weak_agreement_and_dev_pilot_scores(repo):
    root, s = repo
    cfg = config()
    teacher.sample(cfg, "configs/teach.json", snapshot=s)
    teacher.build_packs(cfg, "all", snapshot=s)
    data = teacher.read_chapters(cfg)
    b01, dev_pack = data["packs"]["teach_b01"], data["packs"]["teach_dev_b01"]
    write_pass(root, "teach_b01", b01, no_list=b01[4:])
    write_pass(root, "teach_dev_b01", dev_pack, run_id="opus-run-2")
    (root / "data" / "gold" / "dev").mkdir(parents=True)
    (root / "data" / "gold" / "dev" / "answers.json").write_text(json.dumps(gold_export([s.chapters[c] for c in DEV]), ensure_ascii=False), encoding="utf-8")
    m = teacher.check(cfg, "configs/teach.json", "all", "chk1", snapshot=s)
    p = m["packs"]["teach_b01"]
    assert p["chapters"] == 5 and p["json_valid"] and p["mentions"] == 24 and p["chapters_without_a_mentions_list"] == 1
    assert (p["positives"], p["pending"], p["pending_located"], p["invalid"]) == (16, 4, 4, 4)  # per answered chapter: 4, 1, 1, 1
    assert p["pending_share_of_mentions"] == round(4 / 24, 4)
    assert p["problems"] == {"could_not_be_located": 4, "chapter_without_a_mentions_list": 1}
    assert p["overlap_regions"] == 4 and p["overlap_members"] == {"positive": 8, "pending": 0}
    assert p["positive_mentions_by_type"] == {"person": 8, "organization": 0, "location": 4, "dungeon": 0, "item": 4, "trait": 0}
    assert p["positives_by_type"] == {"person": 4, "organization": 0, "location": 4, "dungeon": 0, "item": 0, "trait": 0}
    assert p["spans_after_conversion"] == 8 and p["pending_regions"] == 4 and p["chapters_missing"] == []
    assert m["packs_without_pass"] == [{"pack": "teach_b02", "state": "missing"}]
    assert m["totals"]["packs"] == 2 and m["totals"]["chapters"] == 7 and m["totals"]["positives"] == 24
    a = m["agreement_with_weak"]["training_packs"]
    assert (a["chapters"], a["teacher_spans"], a["weak_spans"], a["teacher_spans_in_weak_uncertain"]) == (4, 8, 8, 4)
    assert (a["teacher_vs_weak_positives"]["tp"], a["teacher_vs_weak_positives"]["fp"], a["teacher_vs_weak_positives"]["fn"]) == (4, 4, 4)
    masked = a["teacher_vs_weak_positives_outside_weak_uncertain"]
    assert (masked["tp"], masked["fp"], masked["fn"]) == (4, 0, 4)
    assert m["agreement_with_weak"]["dev_pack"]["chapters"] == 2
    # scored on the raw located positives, as evaluate scores a compared system: per chapter 3 right, the overlapping
    # 小雨 item wrong, the unlocated mention wrong, the pending 影子 unscored
    d = m["dev_pilot"]
    assert d["chapters"] == dev_pack and d["gold_spans"] == 6 and d["gold_unscored_regions"] == 2
    assert (d["strict"]["tp"], d["strict"]["fp"], d["strict"]["fn"], d["strict"]["invalid_predictions"]) == (6, 4, 0, 2)
    assert (d["strict"]["precision"], d["strict"]["recall"]) == (0.6, 1.0)
    assert d["strict"]["fp_overlapping_gold"] == 0  # 小雨 overlaps a matched answer span, which strict does not count here
    assert d["non_person_micro"]["f1"] == 0.6667 and d["per_type"]["location"]["recall"] == 1.0
    assert m["dev_pilot_skipped"] is None
    for name in ("config.json", "metrics.json"):
        text = (root / "runs" / "chk1" / name).read_text(encoding="utf-8")
        assert not set(text) & set("林小雨看着的影子她去了灰塔城")  # counts only, no novel text
    rec = json.loads((root / "runs" / "chk1" / "config.json").read_text(encoding="utf-8"))
    assert rec["claude_calls_run_ids"] == ["opus-run-1", "opus-run-2"] and rec["pass_files"]["teach_b01"]["sha256"]
    assert rec["weak_source"]["sha256_matches_record"] and rec["packs"] == ["teach_b01", "teach_dev_b01"]
    with pytest.raises(SystemExit, match="already has results"):
        teacher.check(cfg, "configs/teach.json", "all", "chk1", snapshot=s)
    # without the dev pack's pass the dev pilot is skipped and said so; the agreement still comes out
    (root / "data" / "llm_tasks" / "teach_dev_b01" / "pass_opus_t1.json").unlink()
    m2 = teacher.check(cfg, "configs/teach.json", "pilot", "chk2", snapshot=s)
    assert m2["dev_pilot"] is None and "no pass" in m2["dev_pilot_skipped"] and m2["agreement_with_weak"]["dev_pack"] is None


def test_convert_chapter_follows_the_written_rules():
    ch = chapter(1)
    positives = [(0, 3, "person"), (5, 8, "person"), (5, 8, "person"), (16, 19, "location"), (1, 3, "item")]
    row, counts = teacher.convert_chapter(ch, positives, [(9, 11), (9, 11), (17, 18)], "teacher_x")
    assert [(x["start"], x["end"], x["type"]) for x in row["spans"]] == [(5, 8, "person")]
    assert [(x["start"], x["end"], x["reason"]) for x in row["uncertain"]] == [(0, 3, "teacher_overlap"), (9, 11, "teacher_pending"), (16, 19, "teacher_overlap")]
    assert all(x["types"] is None and x["sources"] == ["teacher_x"] for x in row["uncertain"])
    assert all(x["sources"] == ["teacher_x"] and ch.text[x["start"]:x["end"]] == x["text"] for x in row["spans"] + row["uncertain"])
    assert counts["duplicate_positives_merged"] == 1 and counts["duplicate_pending_merged"] == 1
    assert counts["overlap_regions"] == 2 and counts["overlap_members_positive"] == 3 and counts["overlap_members_pending"] == 1
    assert counts["pending_regions"] == 1 and counts["spans"] == 1 and counts["spans_person"] == 1
    assert row["split"] == "train" and row["number"] == 1
    # two pending places that overlap only each other merge into one pending region
    row, counts = teacher.convert_chapter(ch, [], [(0, 3), (1, 5)], "teacher_x")
    assert [(x["start"], x["end"], x["reason"]) for x in row["uncertain"]] == [(0, 5, "teacher_pending")]
    assert counts["pending_places_merged_into_one_region"] == 1 and "overlap_regions" not in counts


def test_labels_build_runs_without_any_gold_directory_and_tagger_accepts_it(repo):
    root, s = repo
    cfg = config()
    teacher.sample(cfg, "configs/teach.json", snapshot=s)
    teacher.build_packs(cfg, "all", snapshot=s)
    data = teacher.read_chapters(cfg)
    b01 = data["packs"]["teach_b01"]
    write_pass(root, "teach_b01", b01, no_list=b01[4:])
    write_pass(root, "teach_dev_b01", data["packs"]["teach_dev_b01"])
    assert not (root / "data" / "gold").exists()
    m = teacher.build_labels(cfg, "configs/teach.json", "all", "tb", snapshot=s)
    assert not (root / "data" / "gold").exists()
    assert m["chapters"] == 4 and m["packs"] == 1 and m["packs_without_pass"] == [{"pack": "teach_b02", "state": "missing"}]
    t = m["settings"]["T"]
    assert t["spans"] == 8 and t["spans_by_type"]["person"] == 4 and t["spans_by_type"]["location"] == 4
    assert t["uncertain_by_reason"] == {"teacher_overlap": 4, "teacher_pending": 4} and t["invalid_mentions_dropped"] == 4
    assert t["chapters_dropped"] == {"pass_missing": 5, "no_mentions_list": 1} and t["overlap_members"] == {"positive": 8, "pending": 0}
    assert m["settings"]["W"] == {"chapters": 4, "spans": 8, "spans_by_type": {"person": 8, "organization": 0, "location": 0, "dungeon": 0, "item": 0, "trait": 0},
                                  "uncertain": 4, "uncertain_by_reason": {"other_chapters_only": 4}}
    assert m["agreement_T_vs_W"]["teacher_vs_weak_positives"]["f1"] == 0.5
    build = root / "data" / "weak_labels" / "tb"
    rows_t = [json.loads(l) for l in (build / "labels_T.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_w = [json.loads(l) for l in (build / "labels_W.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = sorted(r["chapter_id"] for r in rows_t)
    assert ids == sorted(b01[:4]) and [r["chapter_id"] for r in rows_w] == [r["chapter_id"] for r in rows_t]
    assert all(r["split"] == "train" for r in rows_t + rows_w) and rows_w[0]["spans"] == weak_row(s.chapters[ids[0]], "train")["spans"]
    rec = json.loads((root / "runs" / "tb" / "config.json").read_text(encoding="utf-8"))
    assert rec["snapshot"]["id"] == "snap-test" and rec["labels"]["T"]["sha256"] == sha256(build / "labels_T.jsonl")
    assert rec["labels"]["W"]["sha256"] == sha256(build / "labels_W.jsonl")
    assert rec["split"]["train_ids_sha256"] == ids_sha256(ids) and rec["split"]["max_train_chapter"] == 780
    assert rec["split"]["dev_chapters"] == ["ch0003", "ch0006"] and rec["split"]["train_chapters"] == 4
    assert rec["teacher"]["packs"] == ["teach_b01"] and rec["teacher"]["claude_calls_run_ids"] == ["opus-run-1"]
    assert rec["teacher"]["pass_files"]["teach_b01"]["sha256"] == sha256(root / "data" / "llm_tasks" / "teach_b01" / "pass_opus_t1.json")
    assert rec["conversion_rules"] == teacher.CONVERSION_RULES and rec["weak_source"]["setting"] == "S3"
    for name in ("config.json", "metrics.json"):
        assert not set((root / "runs" / "tb" / name).read_text(encoding="utf-8")) & set("林小雨看着的影子她去了灰塔城")
    # tagger.training_rows takes the rows (the dev2 list is passed, since the repository's file is not this layout)
    kept = tagger.training_rows(rows_t, s, set(DEV), dev2_ids=frozenset())
    assert len(kept) == 4
    with pytest.raises(SystemExit, match="never overwritten"):
        teacher.build_labels(cfg, "configs/teach.json", "all", "tb", snapshot=s)
    # the weak file must match its record
    labels = root / "data" / "weak_labels" / "wb" / "labels_S3.jsonl"
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the file"):
        teacher.build_labels(cfg, "configs/teach.json", "all", "tb2", snapshot=s)


def test_labels_build_refuses_dev_chapters_the_dev_pack_and_late_chapters(repo):
    root, s = repo
    cfg = config()
    teacher.sample(cfg, "configs/teach.json", snapshot=s)
    teacher.build_packs(cfg, "all", snapshot=s)
    data = teacher.read_chapters(cfg)
    chapters_file = root / "configs" / "teach.chapters.json"
    for bad in ("ch0003", "ch0781"):  # a dev chapter, a chapter after 780, each in a package of its own
        name = f"teach_b0{'3' if bad == 'ch0003' else '4'}"
        folder = root / "data" / "llm_tasks" / name
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps({"name": name, "snapshot_id": "snap-test", "chapters": [bad],
                                                      "instructions_version": "v1", "guide_version": "1.1"}), encoding="utf-8")
        write_pass(root, name, [bad])
        wrong = json.loads(json.dumps(data))
        wrong["packs"][name] = [bad]
        wrong["training_packs"].append(name)
        chapters_file.write_text(json.dumps(wrong), encoding="utf-8")
        with pytest.raises(SystemExit, match="refused"):
            teacher.build_labels(cfg, "configs/teach.json", "all", f"tb-{bad}", snapshot=s)
        assert not (root / "data" / "weak_labels" / f"tb-{bad}").exists()
    chapters_file.write_text(json.dumps(data), encoding="utf-8")
    # the dev pack never enters a build, whatever the selection; nothing to build without a training pass
    write_pass(root, "teach_dev_b01", data["packs"]["teach_dev_b01"])
    with pytest.raises(SystemExit, match="no chapter with a teacher pass"):
        teacher.build_labels(cfg, "configs/teach.json", "pilot", "tb3", snapshot=s)
    # the config's labels_build_id ("tb") belongs to the full build: a pilot build may not take it
    with pytest.raises(SystemExit, match="reserved for --packs all"):
        teacher.build_labels(cfg, "configs/teach.json", "pilot", "tb", snapshot=s)
    assert not (root / "data" / "weak_labels" / "tb").exists() and not (root / "runs" / "tb").exists()
    # a pack that was never built is named, for check and labels alike
    unbuilt = json.loads(json.dumps(data))
    unbuilt["packs"]["teach_b09"] = ["ch0001"]
    unbuilt["training_packs"].append("teach_b09")
    chapters_file.write_text(json.dumps(unbuilt), encoding="utf-8")
    with pytest.raises(SystemExit, match="teach_b09: no package .*build-packs first"):
        teacher.build_labels(cfg, "configs/teach.json", "all", "tb5", snapshot=s)
    with pytest.raises(SystemExit, match="teach_b09: no package .*build-packs first"):
        teacher.check(cfg, "configs/teach.json", "all", "chk-unbuilt", snapshot=s)
    assert not (root / "runs" / "chk-unbuilt").exists()
    chapters_file.write_text(json.dumps(data), encoding="utf-8")
    # a package whose chapters differ from the chapters file is refused
    write_pass(root, "teach_b01", data["packs"]["teach_b01"])
    task = root / "data" / "llm_tasks" / "teach_b01" / "task.json"
    t = json.loads(task.read_text(encoding="utf-8"))
    task.write_text(json.dumps({**t, "chapters": t["chapters"][::-1]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="differ from the chapters file"):
        teacher.build_labels(cfg, "configs/teach.json", "pilot", "tb4", snapshot=s)


def test_only_the_dev_pilot_scoring_mentions_an_answer_file():
    """The labels builder and everything it calls reference no gold path (the material-scope rules)."""
    allowed = {"dev_pilot_scores", "check"}  # check passes the config's answer path to dev_pilot_scores for the dev pack
    mentioning = {name for name, f in inspect.getmembers(teacher, inspect.isfunction)
                  if f.__module__ == teacher.__name__ and "gold" in inspect.getsource(f).lower()}
    assert mentioning == allowed
    for name in ("build_labels", "convert_chapter", "pass_places", "weak_rows", "read_task", "read_pass", "sample", "build_packs"):
        assert "gold" not in inspect.getsource(getattr(teacher, name)).lower()
    header = inspect.getsource(teacher).split("\ndef ", 1)[0].split('"""', 2)[2]  # module imports and constants, after the docstring
    assert "gold" not in header.lower()
    assert "gold" not in inspect.getsource(teacher.main).lower()


def test_run_packs_needs_built_packages_and_hands_the_teacher_settings_to_claude_run(repo, monkeypatch):
    root, s = repo
    cfg = config()
    teacher.sample(cfg, "configs/teach.json", snapshot=s)
    with pytest.raises(SystemExit, match="build-packs first"):
        teacher.run_packs(cfg, "pilot", "run-x")
    teacher.build_packs(cfg, "pilot", snapshot=s)
    seen = {}
    import novel_ner.claude_run as cr
    monkeypatch.setattr(cr, "run_many", lambda tasks, label, model, effort, run_id, workers: seen.update(
        tasks=[t.name for t in tasks], label=label, model=model, effort=effort, run_id=run_id, workers=workers) or [])
    teacher.run_packs(cfg, "pilot", "run-x")
    assert seen == {"tasks": ["teach_b01", "teach_dev_b01"], "label": "opus_t1", "model": "claude-test", "effort": "xhigh",
                    "run_id": "run-x", "workers": 1}


def test_uncertainty_and_rank_lists_on_a_hand_made_pool():
    rows = {"m1": [{"chapter_id": "c1", "start": 0, "end": 2, "type": "person", "score": 0.99},   # all three, high
                   {"chapter_id": "c1", "start": 4, "end": 6, "type": "item", "score": 0.99},     # all three, m3 low
                   {"chapter_id": "c1", "start": 8, "end": 9, "type": "item", "score": 0.99},     # m1 alone
                   {"chapter_id": "c2", "start": 0, "end": 2, "type": "person", "score": 0.99},   # all three, high
                   {"chapter_id": "c2", "start": 4, "end": 6, "type": "person", "score": 0.99}],  # m1 and m2 only
            "m2": [{"chapter_id": "c1", "start": 0, "end": 2, "type": "person", "score": 0.95},
                   {"chapter_id": "c1", "start": 4, "end": 6, "type": "item", "score": 0.95},
                   {"chapter_id": "c2", "start": 0, "end": 2, "type": "person", "score": 0.95},
                   {"chapter_id": "c2", "start": 4, "end": 6, "type": "person", "score": 0.95}],
            "m3": [{"chapter_id": "c1", "start": 0, "end": 2, "type": "person", "score": 0.91},
                   {"chapter_id": "c1", "start": 4, "end": 6, "type": "item", "score": 0.5},
                   {"chapter_id": "c1", "start": 4, "end": 6, "type": "person", "score": 0.3},   # same place, other type: its own span
                   {"chapter_id": "c2", "start": 0, "end": 2, "type": "person", "score": 0.91}]}
    u = teacher.uncertainty(rows, ["c1", "c2", "c3"], 0.9, 3)
    assert {k: v for k, v in u["c1"].items() if k != "_exact"} == {"union_spans": 4, "uncertain_spans": 3, "signal_b": 0.75}
    assert {k: v for k, v in u["c2"].items() if k != "_exact"} == {"union_spans": 2, "uncertain_spans": 1, "signal_b": 0.0}  # under 3: u = 0
    assert u["c3"] == {"union_spans": 0, "uncertain_spans": 0, "signal_b": 0.0, "_exact": 0.0}
    assert teacher.uncertainty(rows, ["c2"], 0.9, 2)["c2"]["signal_b"] == 0.5
    with pytest.raises(SystemExit, match="same span twice"):
        teacher.uncertainty({"m1": rows["m1"] + rows["m1"][:1]}, ["c1"], 0.9, 1)
    # ranking: ties by the lower chapter number; B is drawn from the pool minus A, so a chapter high on both goes to A
    pool = ["c1", "c2", "c3", "c4", "c5", "c6"]
    number = {c: i + 1 for i, c in enumerate(pool)}
    sa = {"c1": 3, "c2": 5, "c3": 5, "c4": 1, "c5": 0, "c6": 3}
    sb = {"c1": 0.9, "c2": 0.9, "c3": 0.1, "c4": 0.4, "c5": 0.4, "c6": 0.0}
    a, b = teacher.rank_lists(pool, sa, sb, 2, number.get)
    assert a == ["c2", "c3"] and b == ["c1", "c4"]
    a, b = teacher.rank_lists(pool, sa, sb, 3, number.get)
    assert a == ["c2", "c3", "c1"] and b == ["c4", "c5", "c6"]
    with pytest.raises(SystemExit, match="cannot take two lists"):
        teacher.rank_lists(pool, sa, sb, 4, number.get)
    assert teacher.pool_size_from_rule("x minus y (649 chapters)") == 649
    assert teacher.model_runs_from_rule("the three (runs/a-1, runs/b_2.e60, runs/c; the arm) predict") == ["a-1", "b_2.e60", "c"]
    with pytest.raises(SystemExit, match="exactly 3 distinct"):
        teacher.model_runs_from_rule("runs/a, runs/a, runs/b")
    assert teacher.predict_suffix_from_rule("run ids <model run>-predict-pool649)", 649) == "-predict-pool649"
    with pytest.raises(SystemExit, match="pool of 649"):
        teacher.predict_suffix_from_rule("<model run>-predict-pool649", 648)


def test_resolve_config_inherits_the_weak_source_and_names_the_chapters_file(ext_repo):
    cfg = ext_cfg()
    assert (cfg["weak_build"], cfg["weak_setting"], cfg["chapters_file"]) == ("wb", "S3", "configs/teach300.chapters.json")
    assert cfg["inherited"] == {"weak_build": "wb", "weak_setting": "S3", "chapters_file": "configs/teach300.chapters.json"}
    plain = teacher.resolve_config(config(), "configs/teach.json")
    assert plain == config()  # a sampled config names everything itself
    with pytest.raises(SystemExit, match="drawn by `extend`"):
        teacher.sample(cfg, "configs/teach300.json")
    with pytest.raises(SystemExit, match="needs an extension config"):
        teacher.extend(plain, "configs/teach.json")


def test_extend_ranks_the_pool_by_the_two_signals_and_writes_the_extension_file(ext_repo):
    root, s, calls = ext_repo
    cfg = ext_cfg()
    num = lambda c: s.chapters[c].number
    data = teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    out = root / "configs" / "teach300.chapters.json"
    assert json.loads(out.read_text(encoding="utf-8")) == data
    pool = ext_pool(root, s)
    assert data["pool"]["chapters"] == 28 and [r["chapter_id"] for r in data["pool_signals"]] == pool
    assert data["pool"] == {"chapters": 28, "numbers": [num(pool[0]), num(pool[-1])], "max_train_chapter": 780,
                            "dev_chapters_excluded": 2, "base_chapters_excluded": 10, "rule": cfg["pool_rule"]}
    for r in data["pool_signals"]:
        e = expected_signals(r["number"])
        assert (r["signal_a"], r["union_spans"], r["uncertain_spans"], r["signal_b"]) == (e["signal_a"], e["union"], e["uncertain"], round(e["u"], 4))
    assert calls == ["m1-predict-pool28", "m2-predict-pool28", "m3-predict-pool28"]
    a, b = expected_lists(pool, s)
    assert data["lists"] == {"A": a, "B": b} and not set(a) & set(b)
    assert data["new_packs"] == ["teach_b03", "teach_b04"] and data["base_packs"] == ["teach_b01", "teach_b02"]
    assert data["training_packs"] == ["teach_b01", "teach_b02", "teach_b03", "teach_b04"] and data["dev_pack"] is None
    assert data["packs"]["teach_b03"] == a and data["packs"]["teach_b04"] == b  # A by signal_a descending, then B by signal_b descending
    base = json.loads((root / "configs" / "teach.chapters.json").read_text(encoding="utf-8"))
    assert all(data["packs"][n] == base["packs"][n] for n in ("teach_b01", "teach_b02")) and "teach_dev_b01" not in data["packs"]
    assert [r["chapter_id"] for r in data["new"]] == a + b and [r["list"] for r in data["new"]] == ["A"] * 5 + ["B"] * 5
    assert [r["pack"] for r in data["new"]] == ["teach_b03"] * 5 + ["teach_b04"] * 5
    sa = [r["signal_a"] for r in data["new"][:5]]
    sb = [r["signal_b"] for r in data["new"][5:]]
    assert sa == sorted(sa, reverse=True) and sb == sorted(sb, reverse=True)
    assert data["cut"]["signal_a_100th"] == sa[-1] and data["cut"]["signal_b_100th"] == sb[-1]
    assert data["overlap"]["u_0"]["list_b"] == sum(1 for c in b if expected_signals(num(c))["union"] < 3)
    assert data["overlap"]["u_0"]["pool"] == sum(1 for c in pool if expected_signals(num(c))["union"] < 3)
    assert data["overlap"]["list_b_with_signal_a_above_cut_a"] == 0
    assert set(data["strata"]) == set(base["strata"]) and sum(v["new"] for v in data["strata"].values()) == 10
    assert all(v["new"] == v["new_a"] + v["new_b"] and v["base_sampled"] == base["strata"][k]["sampled"] for k, v in data["strata"].items())
    assert sum(v["pool"] for v in data["strata"].values()) == 28
    assert all(r["stratum"] in base["strata"] for r in data["new"])
    sig = data["signals"]
    assert sig["signal_a"]["source"]["sha256"] == sha256(root / "data" / "weak_labels" / "wb" / "labels_S3.jsonl")
    assert sig["signal_a"]["source"]["sha256_matches_record"] and sig["signal_a"]["pool_spread"]["max"] == 5
    assert sig["signal_b"]["model_runs"] == ["m1", "m2", "m3"] and sig["signal_b"]["predict_run_suffix"] == "-predict-pool28"
    for m in ("m1", "m2", "m3"):
        p = sig["signal_b"]["predict_runs"][m]
        assert p["run_id"] == f"{m}-predict-pool28" and p["reused"] is False
        assert p["predictions_sha256"] == sha256(root / "runs" / p["run_id"] / "predictions.jsonl")
    assert data["base"]["sha256"] == sha256(root / "configs" / "teach.chapters.json") and data["base"]["labels_build"] == "tb"
    assert data["code_commit"] == "abc" and data["snapshot"] == "snap-test"
    assert not set(out.read_text(encoding="utf-8")) & set("林小雨看着的影子她去了灰塔城")  # ids and numbers only
    # a second extend reuses the predict runs and finds the file unchanged
    again = teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    assert again == data and calls == ["m1-predict-pool28", "m2-predict-pool28", "m3-predict-pool28"]
    # a changed rule gives another extension: refused without --force, replaced with it; the predict runs are reused
    other = json.loads(json.dumps(cfg))
    other["selection"]["signal_b"]["min_union_spans"] = 7  # every u becomes 0: list B is the lowest-numbered rest
    with pytest.raises(SystemExit, match="different extension"):
        teacher.extend(other, "configs/teach300.json", snapshot=s)
    forced = teacher.extend(other, "configs/teach300.json", snapshot=s, force=True)
    assert forced["lists"]["A"] == a and forced["lists"]["B"] == sorted([c for c in pool if c not in a], key=num)[:5]
    assert all(p["reused"] for p in forced["signals"]["signal_b"]["predict_runs"].values()) and len(calls) == 3
    # a predict run whose predictions no longer match its recorded sha256, or hold other chapters, is refused
    pred = root / "runs" / "m2-predict-pool28" / "predictions.jsonl"
    pred.write_text(pred.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the sha256"):
        teacher.extend(cfg, "configs/teach300.json", snapshot=s, force=True)
    import shutil
    shutil.rmtree(root / "runs" / "m2-predict-pool28")
    rec = root / "runs" / "m1-predict-pool28" / "config.json"
    r = json.loads(rec.read_text(encoding="utf-8"))
    rec.write_text(json.dumps({**r, "chapters": r["chapters"][1:]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the pool"):
        teacher.extend(cfg, "configs/teach300.json", snapshot=s, force=True)
    rec.write_text(json.dumps({**r, "snapshot": {"id": "snap-other", "source_commit": "c"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="predicted on snapshot snap-other, not snap-test"):
        teacher.extend(cfg, "configs/teach300.json", snapshot=s, force=True)
    rec.write_text(json.dumps(r), encoding="utf-8")
    # the pool size the rule states and the weak file it names are checked
    bad = json.loads(json.dumps(cfg))
    bad["pool_rule"] = "whatever (27 chapters)"
    with pytest.raises(SystemExit, match="pool has 28 chapters, the config's pool_rule says 27"):
        teacher.extend(bad, "configs/teach300.json", snapshot=s, force=True)
    bad = json.loads(json.dumps(cfg))
    bad["selection"]["signal_a"]["rule"] = "some other file"
    with pytest.raises(SystemExit, match="does not name the weak labels file"):
        teacher.extend(bad, "configs/teach300.json", snapshot=s, force=True)


def test_extend_refuses_a_selected_chapter_that_is_dev_base_test_dev2_test2_or_late(ext_repo):
    root, s, calls = ext_repo
    cfg = ext_cfg()
    out = root / "configs" / "teach300.chapters.json"
    pool = ext_pool(root, s)
    a, b = expected_lists(pool, s)
    configs = root / "configs"
    # an exclusion config that lists a chapter the rule selects: test (order), dev2 (list), test2 (order)
    for key, write in (("test", lambda c: (configs / "test.order.json").write_text(json.dumps({"order": [{"chapter_id": c}, {"chapter_id": "ch0790"}]}), encoding="utf-8")),
                       ("dev2", lambda c: (configs / "dev2.json").write_text(json.dumps({"chapters": [c]}), encoding="utf-8")),
                       ("test2", lambda c: (configs / "test2.order.json").write_text(json.dumps({"order": [{"chapter_id": c}]}), encoding="utf-8"))):
        write(a[0] if key != "dev2" else b[0])
        with pytest.raises(SystemExit, match=f"is a {key} chapter: refused"):
            teacher.extend(cfg, "configs/teach300.json", snapshot=s)
        assert not out.exists()
        write("ch0781" if key != "dev2" else "ch0795")
    (configs / "test2.order.json").write_text(json.dumps({"order": [{"chapter_id": "ch0999"}]}), encoding="utf-8")
    # a ranking that returns a dev chapter, a base chapter or a chapter after 780 (the pool excludes them, so the
    # check after the ranking is exercised directly)
    base = json.loads((configs / "teach.chapters.json").read_text(encoding="utf-8"))["packs"]["teach_b01"][0]
    real = teacher.rank_lists
    for bad, word in (("ch0003", "dev chapter: refused"), (base, "base chapters: refused"), ("ch0781", "after chapter 780: refused"),
                      ("ch0790", "not in snapshot")):
        teacher.rank_lists = lambda pool, sa, sb, n, number, bad=bad: ([bad] + real(pool, sa, sb, n, number)[0][1:], real(pool, sa, sb, n, number)[1])
        try:
            with pytest.raises(SystemExit, match=word):
                teacher.extend(cfg, "configs/teach300.json", snapshot=s)
        finally:
            teacher.rank_lists = real
        assert not out.exists()
    teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    assert out.exists()


def test_build_packs_run_and_select_for_the_extension(ext_repo, monkeypatch):
    root, s, calls = ext_repo
    cfg = ext_cfg()
    data = teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    for which in ("pilot", "rest", "all"):
        with pytest.raises(SystemExit, match="new or all300"):
            teacher.select_packs(data, which)
    with pytest.raises(SystemExit, match="pilot, rest or all"):
        teacher.select_packs(teacher.read_chapters(config()), "new")
    assert teacher.select_packs(data, "new") == ["teach_b03", "teach_b04"]
    assert teacher.select_packs(data, "all300") == ["teach_b01", "teach_b02", "teach_b03", "teach_b04"]
    built, kept = teacher.build_packs(cfg, "new", snapshot=s)
    assert built == ["teach_b03", "teach_b04"] and kept == []
    task = json.loads((root / "data" / "llm_tasks" / "teach_b03" / "task.json").read_text(encoding="utf-8"))
    assert task["chapters"] == data["lists"]["A"] and task["snapshot_id"] == "snap-test"
    assert (task["instructions_version"], task["guide_version"]) == ("v1", "1.1")
    teacher.build_packs(config(), "all", snapshot=s)  # the base packs, as stage 1 built them
    built, kept = teacher.build_packs(cfg, "all300", snapshot=s)
    assert built == [] and kept == ["teach_b01", "teach_b02", "teach_b03", "teach_b04"]
    # a chapter in two training packs is refused
    wrong = json.loads(json.dumps(data))
    wrong["packs"]["teach_b04"] = [data["packs"]["teach_b01"][0]] + wrong["packs"]["teach_b04"][1:]
    (root / "configs" / "teach300.chapters.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(SystemExit, match="two training packs"):
        teacher.build_packs(cfg, "new", snapshot=s)
    (root / "configs" / "teach300.chapters.json").write_text(json.dumps(data), encoding="utf-8")
    seen = {}
    import novel_ner.claude_run as cr
    monkeypatch.setattr(cr, "run_many", lambda tasks, label, model, effort, run_id, workers: seen.update(
        tasks=[t.name for t in tasks], label=label, model=model, effort=effort, run_id=run_id, workers=workers) or [])
    teacher.run_packs(cfg, "new", "run-300")
    assert seen == {"tasks": ["teach_b03", "teach_b04"], "label": "opus_t1", "model": "claude-test", "effort": "xhigh",
                    "run_id": "run-300", "workers": 1}
    # the base packs are never re-run: a selection holding one is refused before run_many is reached
    seen.clear()
    with pytest.raises(SystemExit, match="never re-run"):
        teacher.run_packs(cfg, "all300", "run-301")
    assert seen == {}


def base_build(root, s):
    """Stage 1 as the extension finds it: base packs built, passes written, the base labels build tb made."""
    teacher.build_packs(config(), "all", snapshot=s)
    data = teacher.read_chapters(config())
    for name in ("teach_b01", "teach_b02"):
        write_pass(root, name, data["packs"][name], run_id="opus-run-base")
    return teacher.build_labels(config(), "configs/teach.json", "all", "tb", snapshot=s)


def test_labels_build_for_the_extension_reuses_the_base_passes_and_refuses_a_changed_one(ext_repo):
    root, s, calls = ext_repo
    base_build(root, s)
    cfg = ext_cfg()
    data = teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    teacher.build_packs(cfg, "new", snapshot=s)
    for name in ("teach_b03", "teach_b04"):
        write_pass(root, name, data["packs"][name], run_id="opus-run-new")
    with pytest.raises(SystemExit, match="reserved for --packs all300"):
        teacher.build_labels(cfg, "configs/teach300.json", "new", "tb300", snapshot=s)
    assert not (root / "data" / "gold").exists()
    m = teacher.build_labels(cfg, "configs/teach300.json", "all300", "tb300", snapshot=s)
    assert not (root / "data" / "gold").exists()
    ids = sorted(c for n in data["training_packs"] for c in data["packs"][n])
    assert m["chapters"] == 20 and m["packs"] == 4 and m["packs_without_pass"] == []
    assert m["settings"]["T"]["spans"] == 40 and m["settings"]["W"]["chapters"] == 20
    rec = json.loads((root / "runs" / "tb300" / "config.json").read_text(encoding="utf-8"))
    base_rec = json.loads((root / "runs" / "tb" / "config.json").read_text(encoding="utf-8"))
    assert sorted(rec) == sorted(base_rec)  # the same record keys as the base build
    assert rec["kind"] == teacher.TEACHER_BUILD_KIND and rec["config"]["inherited"]["weak_build"] == "wb"
    assert rec["teacher"]["packs"] == ["teach_b01", "teach_b02", "teach_b03", "teach_b04"]
    assert rec["teacher"]["reused_passes"]["packs"] == ["teach_b01", "teach_b02"] and rec["teacher"]["reused_passes"]["sha256_match"]
    assert rec["teacher"]["reused_passes"]["build"] == "tb" and rec["teacher"]["claude_calls_run_ids"] == ["opus-run-base", "opus-run-new"]
    for name in ("teach_b01", "teach_b02"):
        assert rec["teacher"]["pass_files"][name]["sha256"] == base_rec["teacher"]["pass_files"][name]["sha256"]
    assert rec["split"]["train_ids_sha256"] == ids_sha256(ids) and rec["split"]["train_chapters"] == 20
    assert rec["chapters_file"] == {"file": "configs/teach300.chapters.json", "sha256": sha256(root / "configs" / "teach300.chapters.json")}
    build = root / "data" / "weak_labels" / "tb300"
    rows_t = [json.loads(l) for l in (build / "labels_T.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_w = [json.loads(l) for l in (build / "labels_W.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["chapter_id"] for r in rows_t) == ids and [r["chapter_id"] for r in rows_w] == [r["chapter_id"] for r in rows_t]
    assert rec["labels"]["T"]["sha256"] == sha256(build / "labels_T.jsonl") and rec["labels"]["W"]["sha256"] == sha256(build / "labels_W.jsonl")
    # evaluate and posthoc recompute the training chapters from the record through build_train_ids
    assert teacher.build_train_ids(rec, s.chapters, root) == sorted(ids, key=lambda c: s.chapters[c].number)
    assert len(tagger.training_rows(rows_t, s, set(DEV), dev2_ids=frozenset())) == 20
    for name in ("config.json", "metrics.json"):
        assert not set((root / "runs" / "tb300" / name).read_text(encoding="utf-8")) & set("林小雨看着的影子她去了灰塔城")
    # a base pass that changed since the base build, or went missing, stops the build and the check
    old = root / "data" / "llm_tasks" / "teach_b01" / "pass_opus_t1.json"
    original = old.read_text(encoding="utf-8")
    old.write_text(original.replace("灰塔城", "白塔城"), encoding="utf-8")
    with pytest.raises(SystemExit, match="teach_b01: pass_opus_t1.json is missing or differs from the file base labels build tb recorded"):
        teacher.build_labels(cfg, "configs/teach300.json", "all300", "tb300b", snapshot=s)
    with pytest.raises(SystemExit, match="differs from the file base labels build tb recorded"):
        teacher.check(cfg, "configs/teach300.json", "all300", "chk300-bad", snapshot=s)
    assert not (root / "data" / "weak_labels" / "tb300b").exists() and not (root / "runs" / "chk300-bad").exists()
    old.unlink()
    with pytest.raises(SystemExit, match="missing or differs"):
        teacher.build_labels(cfg, "configs/teach300.json", "all300", "tb300b", snapshot=s)
    old.write_text(original, encoding="utf-8")
    # the base chapters file must be the one the base build hashed
    base_file = root / "configs" / "teach.chapters.json"
    base_text = base_file.read_text(encoding="utf-8")
    base_file.write_text(base_text + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the one base labels build tb hashed"):
        teacher.build_labels(cfg, "configs/teach300.json", "all300", "tb300b", snapshot=s)
    base_file.write_text(base_text, encoding="utf-8")
    assert not (root / "data" / "weak_labels" / "tb300b").exists()


def test_check_on_the_extension_reports_new_and_base_packs_apart(ext_repo):
    root, s, calls = ext_repo
    base_build(root, s)
    cfg = ext_cfg()
    data = teacher.extend(cfg, "configs/teach300.json", snapshot=s)
    teacher.build_packs(cfg, "new", snapshot=s)
    b03, b04 = data["packs"]["teach_b03"], data["packs"]["teach_b04"]
    write_pass(root, "teach_b03", b03, no_list=b03[4:], run_id="opus-run-new")
    write_pass(root, "teach_b04", b04, run_id="opus-run-new")
    m = teacher.check(cfg, "configs/teach300.json", "new", "chk-new", snapshot=s)
    assert sorted(m["packs"]) == ["teach_b03", "teach_b04"] and m["totals"]["packs"] == 2 and m["totals"]["chapters"] == 10
    assert m["dev_pilot"] is None and "no pass" in m["dev_pilot_skipped"] and m["agreement_with_weak"]["dev_pack"] is None
    g = m["by_group"]
    assert g["new_packs"]["packs"] == ["teach_b03", "teach_b04"] and g["base_packs"] == {"packs": [], "totals": None, "agreement_with_weak": None}
    t = g["new_packs"]["totals"]
    assert t["packs"] == 2 and t["chapters"] == 10 and t["chapters_without_a_mentions_list"] == 1 and t["positives"] == 36
    assert t["positives_by_type"] == {"person": 9, "organization": 0, "location": 9, "dungeon": 0, "item": 0, "trait": 0}
    assert t["positive_mentions_by_type"]["item"] == 9 and (t["pending"], t["invalid"], t["overlap_regions"]) == (9, 9, 9)
    assert t["pending_share_of_mentions"] == round(9 / 54, 4)
    a = g["new_packs"]["agreement_with_weak"]
    assert a["chapters"] == 9 and a["teacher_spans"] == 18 and "per_type" in a["teacher_vs_weak_positives"]
    assert m["agreement_with_weak"]["training_packs"]["chapters"] == 9  # the whole selection: the same 9 here
    rec = json.loads((root / "runs" / "chk-new" / "config.json").read_text(encoding="utf-8"))
    assert rec["base_passes"]["packs"] == [] and rec["base_passes"]["build"] == "tb" and rec["packs"] == ["teach_b03", "teach_b04"]
    m2 = teacher.check(cfg, "configs/teach300.json", "all300", "chk-all300", snapshot=s)
    assert m2["totals"]["packs"] == 4 and m2["totals"]["chapters"] == 20 and m2["packs_without_pass"] == []
    g2 = m2["by_group"]
    assert g2["new_packs"]["totals"] == t and g2["base_packs"]["packs"] == ["teach_b01", "teach_b02"]
    assert g2["base_packs"]["totals"]["chapters"] == 10 and g2["base_packs"]["agreement_with_weak"]["chapters"] == 10
    assert m2["agreement_with_weak"]["training_packs"]["chapters"] == 19
    rec2 = json.loads((root / "runs" / "chk-all300" / "config.json").read_text(encoding="utf-8"))
    assert rec2["base_passes"]["packs"] == ["teach_b01", "teach_b02"] and rec2["base_passes"]["sha256_match"]
    assert rec2["claude_calls_run_ids"] == ["opus-run-base", "opus-run-new"]
    for name in ("config.json", "metrics.json"):
        assert not set((root / "runs" / "chk-all300" / name).read_text(encoding="utf-8")) & set("林小雨看着的影子她去了灰塔城")
