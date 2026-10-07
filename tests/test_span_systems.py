"""Trained span systems in evaluate.py: the checks before scoring, pooled seeds, limited pairs, new-name recall.

Synthetic data only: a tiny snapshot in memory and run records written under tmp_path."""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from novel_ner import evaluate
from novel_ner.evaluate import (as_prediction, bootstrap, check_models, model_summary, new_name_view, resolve_pairs,
                                span_system, type_counts)
from novel_ner.novelty import novelty, weak_names
from novel_ner.snapshot import Chapter, Snapshot
from novel_ner.weak import ids_sha256

TYPES = ("person", "organization", "location", "dungeon", "item", "trait")
TEST_PARAGRAPHS = {781: ["唐医生到了汀城。", "白塔很高。"],  # 唐 0-1, 汀城 5-7, 白塔 9-11
                   782: ["灰塔和白塔。", "汀城下雨，唐走了。"]}  # 灰塔 0-2, 白塔 3-5, 汀城 7-9, 下雨 9-11, 唐 12-13
TRAIN = ["ch0001", "ch0002", "ch0004", "ch0005", "ch0006"]  # chapters 1-6 minus the dev chapter 3
TEST_IDS = ["ch0781", "ch0782"]
GOLD = {"ch0781": [(0, 1, "person"), (5, 7, "location"), (9, 11, "location")],
        "ch0782": [(0, 2, "location"), (3, 5, "location"), (7, 9, "location"), (12, 13, "person")]}
ROWS = {1: [("ch0781", 9, 11, "location"), ("ch0781", 0, 1, "person"), ("ch0782", 0, 2, "organization"),
            ("ch0782", 3, 5, "location"), ("ch0782", 9, 11, "item")],
        2: [("ch0781", 9, 11, "location"), ("ch0782", 0, 2, "location"), ("ch0782", 7, 9, "location")]}


def make_snapshot():
    chapters = {}
    for n in range(1, 7):
        paragraphs = [("p0001", "唐医生到了汀城。")] if n == 3 else [("p0001", f"第{n}章，唐和汀城。")]
        chapters[f"ch{n:04d}"] = Chapter(f"ch{n:04d}", n, paragraphs, "", "h")
    for n, texts in TEST_PARAGRAPHS.items():
        chapters[f"ch{n:04d}"] = Chapter(f"ch{n:04d}", n, [(f"p{i:04d}", t) for i, t in enumerate(texts, 1)], "", "h")
    mentions = [{"chapter_id": "ch0002", "name": "唐", "type": "person"},
                {"chapter_id": "ch0781", "name": "白塔", "type": "location"}]  # after the cutoff: never "seen"
    dictionary = [{"name": "汀城", "type": "location", "source": "registry"}]
    return Snapshot("snap", None, {"source_commit": "abc", "extracted_chapters": 780}, chapters, mentions, dictionary)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def row(snap, cid, start, end, kind):
    return {"chapter_id": cid, "start": start, "end": end, "text": snap.chapters[cid].text[start:end], "type": kind, "score": 0.9}


def world(root, monkeypatch, seeds=(1,), change=None):
    """Runs on disk: weak build wb, training runs m<seed>, predict runs p<seed>-test; change(w) edits the records
    before they are written (and hashed), so each check can be made to fail on its own."""
    monkeypatch.setattr(evaluate, "REPO_ROOT", root)
    monkeypatch.setattr(evaluate, "chapter_order", lambda cfg: list(cfg["ids"]))
    snap = make_snapshot()
    w = {"dev": {"ids": ["ch0003"]},
         "order": {"order": [{"chapter_id": "ch0781", "unit_id": "u1"}, {"chapter_id": "ch0782", "unit_id": "u2"}]},
         "build": {"snapshot": {"id": "snap"}, "code": {"commit": "b", "dirty": False},
                   "config": {"dev_config": "configs/dev.json", "test_order": "configs/order.json", "max_train_chapter": 780,
                              "dictionary_cutoff": None},
                   "split": {"train_ids_sha256": ids_sha256(TRAIN), "train_numbers": [1, 6]},
                   "labels": {"S3": {"sha256": "labels-S3"}}},
         "runs": {}, "spec": {},
         "test_ids": list(TEST_IDS), "dev_ids": ["ch0003"], "order_ids": list(TEST_IDS)}
    for seed in seeds:
        w["runs"][seed] = {
            "model": {"run_id": f"m{seed}", "config_file": "configs/train.json", "config": {"lr": 3e-5, "weak_build": "wb"},
                      "setting": "S3", "seed": seed, "snapshot": {"id": "snap"},
                      "weak_labels": {"build": "wb", "sha256": "labels-S3", "train_chapters": 5,
                                      "train_ids_sha256": ids_sha256(TRAIN), "train_chapter_numbers": [1, 6]},
                      "gold_dev": {"file": "data/gold/dev.json", "sha256": "d", "chapters": ["ch0003"]},
                      "backbone": {"id": "hfl/x", "revision": "r"}, "code": {"commit": "m", "dirty": False}},
            "model_metrics": {"dev": {"at_best": {"micro": {"precision": 1.0, "recall": 0.5, "f1": 0.6667}}}},
            "predict": {"run_id": f"p{seed}-test", "model_run": f"m{seed}", "setting": "S3", "seed": seed,
                        "chapters": list(TEST_IDS), "snapshot": {"id": "snap"}, "code": {"commit": "p", "dirty": False}},
            "rows": [row(snap, *r) for r in ROWS[1 if seed % 2 else 2]], "outputs": None}
        w["spec"][seed] = {"name": f"ws_s{seed}", "predictions_run": f"runs/p{seed}-test", "input_condition": "(d) chapter text only"}
    if change:
        change(w)
    write(root / "configs" / "dev.json", w["dev"])
    write(root / "configs" / "order.json", w["order"])
    write(root / "runs" / "wb" / "config.json", w["build"])
    for seed, r in w["runs"].items():
        write(root / "runs" / f"m{seed}" / "config.json", r["model"])
        write(root / "runs" / f"m{seed}" / "metrics.json", r["model_metrics"])
        out = root / "runs" / f"p{seed}-test"
        out.mkdir(parents=True, exist_ok=True)
        evaluate.write_jsonl(out / "predictions.jsonl", r["rows"])
        write(out / "config.json", {**r["predict"], "model_run_config_sha256": evaluate.sha256(root / "runs" / f"m{seed}" / "config.json")})
        write(out / "metrics.json", {"run_id": r["predict"]["run_id"],
                                     "outputs": len(r["rows"]) if r["outputs"] is None else r["outputs"],
                                     "predictions_sha256": evaluate.sha256(out / "predictions.jsonl"),
                                     "predict_stats": {"chunks": 2, "seconds": 0.1}})
    return snap, w


def load(snap, w, seed=1):
    return span_system(w["spec"][seed], snap, w["test_ids"], w["dev_ids"], w["order_ids"])


def test_span_system_accepts_a_clean_run_and_records_where_it_came_from(tmp_path, monkeypatch):
    snap, w = world(tmp_path, monkeypatch)
    pred, rows, rec, pool = load(snap, w)
    assert pred["ch0781"]["spans"] == {(9, 11, "location"), (0, 1, "person")}
    assert pred["ch0782"]["spans"] == {(0, 2, "organization"), (3, 5, "location"), (9, 11, "item")}
    assert len(rows) == 5 and rec["outputs"]["outputs"] == 5 and rec["outputs_crossing_a_paragraph_break"] == 0
    assert rec["predictions_sha256"] == evaluate.sha256(tmp_path / "runs" / "p1-test" / "predictions.jsonl")
    assert (rec["setting"], rec["seed"], rec["model_run"], rec["predictions_run"]) == ("S3", 1, "runs/m1", "runs/p1-test")
    assert rec["input_condition"] == "(d) chapter text only"
    assert rec["weak_build"]["build"] == "runs/wb" and rec["weak_build"]["train_chapters"] == 5
    assert rec["weak_build"]["train_chapter_numbers"] == [1, 6] and rec["dev_at_best"]["f1"] == 0.6667
    assert (pool["setting"], pool["seed"], pool["weak_build"]) == ("S3", 1, "runs/wb")


def R(w):
    return w["runs"][1]


def _set(path, value):
    def change(w):
        target = w
        for k in path[:-1]:
            target = target[k] if not callable(k) else k(target)
        target[path[-1]] = value
    return change


def _both_sha(w):
    w["build"]["split"]["train_ids_sha256"] = "0" * 64
    R(w)["model"]["weak_labels"]["train_ids_sha256"] = "0" * 64


def _row(**kw):
    def change(w):
        R(w)["rows"][0].update(kw)
    return change


def _extra_row(w):
    R(w)["rows"].append({"chapter_id": "ch0001", "start": 0, "end": 1, "text": "第", "type": "person", "score": 0.5})


def _past_end(w):
    R(w)["rows"].append({"chapter_id": "ch0781", "start": 13, "end": 17, "text": "。", "type": "person", "score": 0.5})


def _tamper(relative, how):
    def act(root):
        path = root / relative
        if how == "delete":
            path.unlink()
        else:
            path.write_text(path.read_text(encoding="utf-8") + how, encoding="utf-8")
    return act


def _metrics_run_id(value):
    def act(root):
        path = root / "runs" / "p1-test" / "metrics.json"
        write(path, {**json.loads(path.read_text(encoding="utf-8")), "run_id": value})
    return act


REFUSALS = [
    ("predict records name another run", _set([R, "predict", "run_id"], "p7-test"), None, "both must be 'p1-test'"),
    ("predict metrics name another run", None, _metrics_run_id("p2-test"), "name runs 'p1-test' and 'p2-test'"),
    ("predict metrics name no run", None, _metrics_run_id(None), "name runs 'p1-test' and None"),
    ("predict run on another snapshot", _set([R, "predict", "snapshot", "id"], "other"), None, "is from snapshot other"),
    ("predictions edited after hashing", None, _tamper("runs/p1-test/predictions.jsonl", "\n{}\n"), "is not the file its metrics.json hashed"),
    ("a chapter not scored here", lambda w: R(w)["predict"]["chapters"].append("ch0783"), None, "covers 1 chapters not scored here and misses 0"),
    ("a scored chapter missing", _set([R, "predict", "chapters"], ["ch0781"]), None, "misses 1 scored ones"),
    ("a chapter listed twice", _set([R, "predict", "chapters"], ["ch0781", "ch0781", "ch0782"]), None, "names a chapter twice"),
    ("row count differs from the record", _set([R, "outputs"], 99), None, "its metrics.json counted 99"),
    ("row text not at its offsets", _row(text="黑塔"), None, "prediction row 1 "),
    ("row type not one of the six", _row(type="weapon"), None, "prediction row 1 "),
    ("row in a chapter not scored", _extra_row, None, "prediction row 6 "),
    ("row running past the chapter end", _past_end, None, "prediction row 6 "),
    ("predict run from uncommitted code", _set([R, "predict", "code", "dirty"], True), None, "predict run was not made from committed code"),
    ("training run from uncommitted code", _set([R, "model", "code", "dirty"], True), None, "runs/m1 was not made from committed code"),
    ("training record changed after predicting", None, _tamper("runs/m1/config.json", " "), "is not the file the predict run hashed"),
    ("training record missing", None, _tamper("runs/m1/config.json", "delete"), "has no config.json"),
    ("training run not finished", None, _tamper("runs/m1/metrics.json", "delete"), "training run not finished"),
    ("training run on another snapshot", _set([R, "model", "snapshot", "id"], "other"), None, "the training run is from snapshot other"),
    ("seed differs between the runs", _set([R, "predict", "seed"], 2), None, "setting and seed are not its training run's"),
    ("training chapters past 780", _set([R, "model", "weak_labels", "train_chapter_numbers"], [1, 781]), None, "within 1-780"),
    ("weak build record missing", _set([R, "model", "weak_labels", "build"], "nobuild"), None, "runs/nobuild/config.json does not exist"),
    ("weak build on another snapshot", _set(["build", "snapshot", "id"], "other"), None, "weak build wb is from snapshot other"),
    ("weak build from uncommitted code", _set(["build", "code", "dirty"], True), None, "weak build wb was not made from committed code"),
    ("weak build without a code record", lambda w: w["build"].pop("code"), None, "weak build wb was not made from committed code"),
    ("training chapters not the build's", _set(["build", "split", "train_ids_sha256"], "0" * 64), None, "are not weak build wb's"),
    ("label file not the build's", _set([R, "model", "weak_labels", "sha256"], "other"), None, "label file is not the one"),
    ("recorded hash not what the build record gives", _both_sha, None, "do not give the sha256 it recorded"),
    ("build dev set changed since", _set(["dev", "ids"], ["ch0003", "ch0004"]), None, "do not give the sha256 it recorded"),
    ("build record cannot be recomputed", _set(["build", "config", "max_train_chapter"], 781), None, "cannot be recomputed"),
    ("training chapter count differs", _set([R, "model", "weak_labels", "train_chapters"], 4), None, "counts 4 training chapters"),
    ("training on a development chapter", _set(["dev_ids"], ["ch0003", "ch0002"]), None, "development chapters: 1"),
    ("training on a test-order chapter", _set(["order_ids"], [*TEST_IDS, "ch0004"]), None, "test order: 1"),
    ("checkpoint chosen on a non-dev chapter", _set([R, "model", "gold_dev", "chapters"], ["ch0001"]), None, "not all development chapters"),
    ("unknown key in the spec", lambda w: w["spec"][1].update(predictions="x"), None, "unknown keys ['predictions']"),
    ("no input condition", lambda w: w["spec"][1].pop("input_condition"), None, "give its input_condition"),
    ("nested run path", _set(["spec", 1, "predictions_run"], "runs/a/b"), None, "must be runs/<id> or <id>"),
]


@pytest.mark.parametrize("change,tamper,message", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_span_system_refuses_each_failed_check(tmp_path, monkeypatch, change, tamper, message):
    snap, w = world(tmp_path, monkeypatch, change=change)
    if tamper:
        tamper(tmp_path)
    with pytest.raises(SystemExit) as err:
        load(snap, w)
    assert message in str(err.value)
    assert "白塔" not in str(err.value) and "黑塔" not in str(err.value)  # never quotes a prediction's text


def test_span_system_refuses_files_copied_between_predict_runs(tmp_path, monkeypatch):
    import shutil
    snap, w = world(tmp_path, monkeypatch, seeds=(1, 2))
    runs = tmp_path / "runs"
    # a whole predict run copied under another folder name
    shutil.copytree(runs / "p1-test", runs / "p9-test")
    w["spec"][9] = {**w["spec"][1], "name": "ws_s9", "predictions_run": "runs/p9-test"}
    with pytest.raises(SystemExit, match="name runs 'p1-test' and 'p1-test'; both must be 'p9-test'"):
        load(snap, w, 9)
    # seed 2's predictions and metrics spliced into seed 1's folder: its hash chain holds, the run ids do not
    for f in ("predictions.jsonl", "metrics.json"):
        shutil.copyfile(runs / "p2-test" / f, runs / "p1-test" / f)
    with pytest.raises(SystemExit) as err:
        load(snap, w, 1)
    assert "name runs 'p1-test' and 'p2-test'" in str(err.value) and "白塔" not in str(err.value)
    assert load(snap, w, 2)[2]["seed"] == 2  # the run the files came from still loads


def lineage_world(root, monkeypatch, seeds=(1,), change=None):
    """world() over two snapshots, as stage 3 scores test2: the training snapshot "train" holds the chapters up to 780
    with the dictionary and the mentions; the scoring snapshot "score" is a later export (every chapter of "train" with
    the same text, plus the test chapters) whose dictionary gained a row from a later chapter (白塔). The weak build and
    the training runs are from "train"; the predict runs are on "score" and record the lineage tagger.predict_run writes.
    Returns (train, score, w)."""
    def relabel(w):
        w["build"]["snapshot"]["id"] = "train"
        for r in w["runs"].values():
            r["model"]["snapshot"]["id"] = "train"
            r["predict"]["snapshot"]["id"] = "score"
            r["predict"]["snapshot_lineage"] = {"model_snapshot": "train", "shared_chapters": 6, "shared_text_identical": True}
        if change:
            change(w)
    full, w = world(root, monkeypatch, seeds, relabel)
    train = Snapshot("train", None, dict(full.meta), {c: ch for c, ch in full.chapters.items() if ch.number <= 780},
                     [m for m in full.mentions if int(m["chapter_id"][2:]) <= 780], list(full.dictionary))
    score = Snapshot("score", None, {**full.meta, "extracted_chapters": 782}, dict(full.chapters), list(full.mentions),
                     full.dictionary + [{"name": "白塔", "type": "location", "source": "registry"}])
    return train, score, w


def test_span_system_accepts_a_predict_run_on_a_later_snapshot_with_a_lineage_from_the_training_snapshot(tmp_path, monkeypatch):
    train, score, w = lineage_world(tmp_path, monkeypatch)
    pred, rows, rec, pool = span_system(w["spec"][1], score, w["test_ids"], w["dev_ids"], w["order_ids"], train)
    assert pred["ch0781"]["spans"] == {(9, 11, "location"), (0, 1, "person")} and len(rows) == 5
    assert rec["training_snapshot"] == "train" and rec["snapshot_lineage"]["model_snapshot"] == "train"
    assert rec["weak_build"]["train_chapters"] == 5 and rec["weak_build"]["train_chapter_numbers"] == [1, 6]
    assert "lineage from the training snapshot train" in rec["checks"][0] and pool["weak_build"] == "runs/wb"
    # without the training snapshot the training run is simply from another snapshot than the scored one
    with pytest.raises(SystemExit, match="the training run is from snapshot train, not score"):
        span_system(w["spec"][1], score, w["test_ids"], w["dev_ids"], w["order_ids"])
    # a run without a lineage keeps today's record: no training_snapshot, the plain "snapshot" check
    plain_snap, plain_w = world(tmp_path / "plain", monkeypatch)
    plain_rec = span_system(plain_w["spec"][1], plain_snap, plain_w["test_ids"], plain_w["dev_ids"], plain_w["order_ids"], plain_snap)[2]
    assert "training_snapshot" not in plain_rec and "snapshot_lineage" not in plain_rec and plain_rec["checks"][0] == "snapshot"


LINEAGE_REFUSALS = [
    ("no lineage recorded", lambda w: R(w)["predict"].pop("snapshot_lineage"), "does not record a lineage from the training snapshot train"),
    ("lineage from another snapshot", _set([R, "predict", "snapshot_lineage", "model_snapshot"], "other"), "does not record a lineage"),
    ("shared text not identical", _set([R, "predict", "snapshot_lineage", "shared_text_identical"], False), "does not record a lineage"),
    ("predict run on the model's own snapshot", _set([R, "predict", "snapshot", "id"], "train"), "is from snapshot train, not score"),
    ("training run on the scoring snapshot", _set([R, "model", "snapshot", "id"], "score"), "the training run is from snapshot score, not train"),
    ("weak build on the scoring snapshot", _set(["build", "snapshot", "id"], "score"), "weak build wb is from snapshot score, not train"),
]


@pytest.mark.parametrize("change,message", [r[1:] for r in LINEAGE_REFUSALS], ids=[r[0] for r in LINEAGE_REFUSALS])
def test_span_system_refuses_a_lineage_that_does_not_hold(tmp_path, monkeypatch, change, message):
    train, score, w = lineage_world(tmp_path, monkeypatch, change=change)
    with pytest.raises(SystemExit) as err:
        span_system(w["spec"][1], score, w["test_ids"], w["dev_ids"], w["order_ids"], train)
    assert message in str(err.value) and "白塔" not in str(err.value)


def gold():
    return {cid: {"spans": set(spans), "unscored": []} for cid, spans in GOLD.items()}


def test_seeds_pool_through_models_like_two_runs_of_opus(tmp_path, monkeypatch):
    snap, w = world(tmp_path, monkeypatch, seeds=(1, 2, 3))
    loaded = {f"ws_s{s}": load(snap, w, s) for s in (1, 2, 3)}
    pool = {k: v[3] for k, v in loaded.items()}
    names = {*loaded, "dictionary"}
    about = check_models({"ws": ["ws_s1", "ws_s2", "ws_s3"]}, names, pool)
    assert about == {"ws": {"setting": "S3", "weak_build": "runs/wb", "seeds": [1, 2, 3]}}
    counts = {k: type_counts(gold(), v[0]) for k, v in loaded.items()}
    counts["dictionary"] = type_counts(gold(), as_prediction([row(snap, "ch0781", 5, 7, "location")], TEST_IDS))
    out = model_summary({"ws": ["ws_s1", "ws_s2", "ws_s3"], "dict": ["dictionary"]}, counts, {"ch0781": "u1", "ch0782": "u2"},
                        {"n": 50, "seed": 1}, None, None, about)
    ws = out["models"]["ws"]
    assert ws["seeds"] == [1, 2, 3] and ws["setting"] == "S3" and ws["n_runs"] == 3
    f1 = {1: 2 * 3 / (2 * 3 + 2 + 4), 2: 2 * 3 / (2 * 3 + 0 + 4)}  # seeds 1 and 3 give the same rows
    assert ws["mean"]["micro_f1"] == round((2 * f1[1] + f1[2]) / 3, 4)
    assert ws["range"]["micro_f1"] == [round(f1[1], 4), round(f1[2], 4)]
    assert ws["pooled_counts_micro_f1"] == round(2 * 9 / (2 * 9 + 4 + 12), 4)  # tp 3+3+3, fp 2+0+2, fn 4+4+4
    assert list(out["pooled_paired"]) == ["ws - dict"]
    assert "seeds" not in out["models"]["dict"]  # models of other systems are reported as before
    with pytest.raises(SystemExit, match="mixes trained span systems"):
        check_models({"bad": ["ws_s1", "dictionary"]}, names, pool)
    with pytest.raises(SystemExit, match="not systems in this config"):
        check_models({"bad": ["ws_s1", "ws_s9"]}, names, pool)
    with pytest.raises(SystemExit, match="each once"):
        check_models({"bad": ["ws_s1", "ws_s1"]}, names, pool)
    with pytest.raises(SystemExit, match="same seed"):
        check_models({"bad": ["ws_s1", "ws_s2"]}, names, {**pool, "ws_s2": {**pool["ws_s2"], "seed": 1}})


def test_seeds_that_differ_in_more_than_the_seed_do_not_pool(tmp_path, monkeypatch):
    def change(w):
        w["runs"][2]["model"]["config"]["lr"] = 1e-4
    snap, w = world(tmp_path, monkeypatch, seeds=(1, 2), change=change)
    pool = {f"ws_s{s}": load(snap, w, s)[3] for s in (1, 2)}
    with pytest.raises(SystemExit, match="differ in more than the seed"):
        check_models({"ws": ["ws_s1", "ws_s2"]}, set(pool), pool)


def synthetic_counts():
    counts = {}
    for k, s in enumerate(("x", "y", "z")):
        counts[s] = {f"c{i}": {t: [0, 0, 0] for t in TYPES} for i in range(12)}
        for i in range(12):
            counts[s][f"c{i}"]["person"] = [6 + (i + k) % 3, k + i % 2, 3 - k]
    return counts


def test_pairs_limit_the_differences_and_keep_the_same_draws():
    counts, strata = synthetic_counts(), {f"c{i}": "u1" if i < 8 else "u2" for i in range(12)}
    every = bootstrap(counts, strata, 200, 7)
    some = bootstrap(counts, strata, 200, 7, pairs=[("x", "z")])
    assert list(every["paired_difference"]) == ["x - y", "x - z", "y - z"]
    assert list(some["paired_difference"]) == ["x - z"]
    assert some["paired_difference"]["x - z"] == every["paired_difference"]["x - z"]
    assert some["systems"] == every["systems"]  # every system keeps its interval
    reverse = bootstrap(counts, strata, 200, 7, pairs=[("z", "x")])
    assert list(reverse["paired_difference"]) == ["z - x"]
    with pytest.raises(ValueError):
        bootstrap(counts, strata, 20, 7, pairs=[("x", "q")])
    models = {"m": ["x", "y"], "n": ["z"], "o": ["y"]}
    assert list(model_summary(models, counts, strata, {"n": 50, "seed": 1})["pooled_paired"]) == ["m - n", "m - o", "n - o"]
    assert list(model_summary(models, counts, strata, {"n": 50, "seed": 1}, pairs=[("n", "m")])["pooled_paired"]) == ["n - m"]
    assert model_summary(models, counts, strata, {"n": 50, "seed": 1}, pairs=[])["pooled_paired"] == {}


def test_resolve_pairs_sorts_systems_from_models_and_refuses_unclear_pairs():
    systems, models = ["x", "y", "z"], {"m": ["x", "y"], "n": ["z"]}
    assert resolve_pairs(None, systems, models) == (None, None)
    assert resolve_pairs([["x", "y"], ["n", "m"]], systems, models) == ([("x", "y")], [("n", "m")])
    assert resolve_pairs([["n", "m"]], systems, models) == ([], [("n", "m")])
    for bad, message in [([["x", "m"]], "joins a system and a model"), ([["x", "q"]], "neither a system nor a model"),
                         ([["x", "y"], ["y", "x"]], "listed twice"), ([["x", "x"]], "two different names"),
                         ([["x"]], "two different names")]:
        with pytest.raises(SystemExit, match=message):
            resolve_pairs(bad, systems, models)
    with pytest.raises(SystemExit, match="both a system and a model"):
        resolve_pairs([["x", "y"]], systems, {"x": ["x"]})


def test_new_name_view_equals_counts_by_hand():
    snap = make_snapshot()
    texts = {cid: snap.chapters[cid].text for cid in TEST_IDS}
    g = gold()
    g["ch0782"]["unscored"] = [(4, 5)]  # touches the gold 白塔 at 3-5: left out of every count
    seen = {"汀城", "唐"}
    systems = {
        "a": as_prediction([row(snap, *r) for r in ROWS[1]], TEST_IDS),
        "b": as_prediction([row(snap, *r) for r in ROWS[2]], TEST_IDS),
        "dict": as_prediction([row(snap, "ch0781", 5, 7, "location"), row(snap, "ch0782", 7, 9, "location")], TEST_IDS),
    }
    systems["a"]["ch0781"]["errors"] = 4  # invalid outputs have no text and are in neither number
    out, counts = new_name_view(g, systems, texts, {s: seen for s in systems}, {"ch0781": "u1", "ch0782": "u2"},
                                {"n": 200, "seed": 3}, {"ab": ["a", "b"], "d": ["dict"]}, [("a", "dict")], [("ab", "d")])
    # new gold: ch0781 白塔 9-11, ch0782 灰塔 0-2 (白塔 3-5 is unscored)
    # a: 白塔 hit; 灰塔 as organization (unseen string, wrong) -> fp and 灰塔 fn; 白塔 3-5 unscored; 下雨 item fp; 唐 seen
    assert {k: out["systems"]["a"][k] for k in ("tp", "fp", "fn", "new_gold_spans")} == {"tp": 1, "fp": 2, "fn": 1, "new_gold_spans": 2}
    assert out["systems"]["a"]["recall"] == 0.5 and out["systems"]["a"]["precision_among_predictions_with_unseen_text"] == round(1 / 3, 4)
    # b: 白塔 hit, 灰塔 hit; 汀城 is seen: left out
    assert (out["systems"]["b"]["tp"], out["systems"]["b"]["fp"], out["systems"]["b"]["fn"]) == (2, 0, 0)
    assert out["systems"]["b"]["recall"] == 1.0
    # the dictionary only predicts seen names: recall 0 by definition, precision undefined
    assert out["systems"]["dict"]["recall"] == 0.0 and out["systems"]["dict"]["precision_among_predictions_with_unseen_text"] is None
    assert out["systems"]["dict"]["precision_among_predictions_with_unseen_text_ci95"] is None
    lo, hi = out["systems"]["a"]["recall_ci95"]
    assert 0 <= lo <= 0.5 <= hi <= 1
    assert list(out["paired_difference_recall"]) == ["a - dict"]
    pooled = out["models"]["ab"]
    assert pooled["recall_by_run"] == {"a": 0.5, "b": 1.0} and pooled["mean_recall"] == 0.75 and pooled["range_recall"] == [0.5, 1.0]
    assert (pooled["pooled"]["tp"], pooled["pooled"]["fp"], pooled["pooled"]["fn"], pooled["pooled"]["recall"]) == (3, 2, 1, 0.75)
    # tp + fn is 4 (two runs), but there are 2 new gold spans: the pooled entry counts them once, like a system's
    assert pooled["pooled"]["new_gold_spans"] == 2 == out["systems"]["a"]["new_gold_spans"] == out["systems"]["b"]["new_gold_spans"]
    assert out["models"]["d"]["pooled"]["new_gold_spans"] == 2
    assert list(out["pooled_paired_difference_recall"]) == ["ab - d"]
    assert novelty(g, texts, seen)["new_spans"] == out["systems"]["a"]["new_gold_spans"]
    assert sum(c[t][0] for c in counts["b"].values() for t in TYPES) == 2
    # a model whose runs count new gold spans from different seen sets has no single gold count (main refuses it earlier)
    with pytest.raises(AssertionError, match="runs disagree on the new gold spans"):
        new_name_view(g, systems, texts, {"a": seen, "b": {"汀城"}, "dict": seen}, {"ch0781": "u1", "ch0782": "u2"},
                      {"n": 20, "seed": 3}, {"ab": ["a", "b"]}, [], [])


def test_weak_names_can_truncate_the_dictionary_for_a_learning_curve_tier():
    snap = make_snapshot()
    snap.dictionary = [{"name": "汀城", "canonical": "汀城", "type": "location"},
                       {"name": "唐医生", "canonical": "唐医生", "type": "person", "known": 5},
                       {"name": "白塔", "canonical": "白塔", "type": "location"}]
    dictionary, mentions, info = weak_names(snap, 780)
    assert dictionary == {"汀城", "唐医生", "白塔"} and mentions == {"唐"} and "dictionary_cutoff" not in info
    late = [c for c, ch in snap.chapters.items() if ch.number > 780]
    dictionary, mentions, info = weak_names(snap, 3, dictionary_cutoff=3, dev_ids=[], test_ids=late)
    # 唐医生 occurs in chapter 3 but is revealed at 5; 白塔 occurs only after 780
    assert dictionary == {"汀城"} and mentions == {"唐"}
    assert (info["dictionary_rows_kept"], info["dictionary_rows_dropped"]) == (1, 2)
    assert weak_names(snap, 6, dictionary_cutoff=6, dev_ids=[], test_ids=late)[0] == {"汀城", "唐医生"}
    # as in the weak build, a row without a reveal chapter is dated by its first Opus listing: 汀城 first listed at 5
    snap.mentions = snap.mentions + [{"chapter_id": "ch0005", "name": "汀城", "type": "location", "same_as": None}]
    assert weak_names(snap, 3, dictionary_cutoff=3, dev_ids=[], test_ids=late)[0] == set()
    assert weak_names(snap, 6, dictionary_cutoff=6, dev_ids=[], test_ids=late)[0] == {"汀城", "唐医生"}
    # a development chapter's text does not count as a training text (the build leaves it out)
    assert weak_names(snap, 6, dictionary_cutoff=6, dev_ids=["ch0003"], test_ids=late)[0] == {"汀城"}


class FakeTokenizer:
    unk_token_id = 0

    def __call__(self, text):
        return {"input_ids": [1] * len(text)}


class FakeGLiNER:
    """Finds 白塔 in chapter 781 only, like ChineseGLiNER.predict (chunk offsets)."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace(config=SimpleNamespace(max_len=384, max_width=12))

    def configure(self, *args):
        pass

    def token_count(self, text, labels):
        return len(text)

    def predict(self, texts, labels, threshold, batch_size):
        out = []
        for t in texts:
            at = t.find("白塔") if t.startswith("唐医生") and "白塔" in t else -1
            out.append([{"start": at, "end": at + 2, "text": "白塔", "label": "location", "score": 0.9}] if at >= 0 else [])
        return out


def e2e_config(tmp_path, monkeypatch, lineage=False, snapshots=None):
    """Two seeds on disk, test and dev answers, GPT drafts, a fake GLiNER, main()'s other inputs patched; returns the
    config main() would read (written by run_main). lineage: the two snapshots of lineage_world instead of one (the
    test answers name the scoring snapshot "score", the dev answers the training snapshot "train", the config names
    training_snapshot and dictionary.from_training_snapshot); snapshots (a dict) is filled with the snapshots by id."""
    snapshots = {} if snapshots is None else snapshots
    if lineage:
        train, snap, w = lineage_world(tmp_path, monkeypatch, seeds=(1, 2))
        snapshots.update({"train": train, "score": snap})
    else:
        snap, w = world(tmp_path, monkeypatch, seeds=(1, 2))
        snapshots[snap.id] = snap
    ann = lambda s, e, t, **kw: {"start": s, "end": e, "type": t, "pending": False, **kw}
    test_export = {"guide_version": "1.1", "snapshot_id": snap.id,
                   "llm_drafts": {"t1": {"A": {"model": "gpt", "instructions_version": 1},
                                         "B": {"model": "gpt", "instructions_version": 1}}},
                   "chapters": {cid: {"status": "done", "provenance": "model_double_adjudicated", "unscored": [],
                                      "annotations": [ann(*sp, cross_chapter=(cid, sp[0]) == ("ch0781", 9)) for sp in spans]}
                                for cid, spans in GOLD.items()}}
    write(tmp_path / "data" / "gold" / "test.json", test_export)
    write(tmp_path / "data" / "gold" / "dev.json", {"snapshot_id": "train" if lineage else snap.id,
                                                    "chapters": {"ch0003": {"status": "done", "annotations": [
                                                        ann(0, 1, "person"), ann(5, 7, "location")]}}})
    draft = {"chapters": {cid: {"spans": [{"start": s, "end": e, "type": t, "pending": False} for s, e, t in spans]}
                          for cid, spans in GOLD.items()}}
    write(tmp_path / "data" / "llm_tasks" / "t1" / "task.json", {"chapters": TEST_IDS})
    write(tmp_path / "data" / "llm_tasks" / "t1" / "draft_A.json", draft)
    draft["chapters"]["ch0782"]["spans"] = draft["chapters"]["ch0782"]["spans"][1:]  # pass B misses 灰塔: decided by the user
    write(tmp_path / "data" / "llm_tasks" / "t1" / "draft_B.json", draft)
    write(tmp_path / "runs" / "calib" / "metrics.json", {"chapters": ["ch0003"], "gold_file": "g",
                                                          "drafts": {"A": {"vs_gold": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}})
    cfg = {"snapshot": snap.id, **({"training_snapshot": "train"} if lineage else {}),
           "gold_test": "data/gold/test.json", "gold_dev": "data/gold/dev.json",
           "test_order": "configs/order.json", "calibration_metrics": "runs/calib/metrics.json", "dev_set": "configs/dev.json",
           "gliner": {"variant": {"splitter": "char", "space_mode": "per_word"}, "labels": {t: t for t in TYPES},
                      "chunk_max_chars": 200, "batch_size": 4, "save_threshold": 0.1, "thresholds": [0.5]},
           "dictionary": {"min_len": 2, **({"from_training_snapshot": True} if lineage else {})}, "bootstrap": {"n": 100, "seed": 5},
           "span_systems": [{"name": f"ws_s{s}", "predictions_run": f"runs/p{s}-test", "input_condition": "(d) chapter text only"}
                            for s in (1, 2)],
           "models": {"ws": ["ws_s1", "ws_s2"], "gl": ["gliner_zero_shot"]},
           "pairs": [["ws_s1", "dictionary"], ["ws", "gl"]],
           "new_names": {"cutoff_chapter": 780}}
    fake = types.ModuleType("novel_ner.gliner_zh")
    fake.BACKBONE_ID, fake.BACKBONE_REVISION, fake.MODEL_ID, fake.MODEL_REVISION = "b", "br", "m", "mr"
    fake.ChineseGLiNER = FakeGLiNER
    monkeypatch.setitem(sys.modules, "novel_ner.gliner_zh", fake)
    monkeypatch.setattr(evaluate, "load_snapshot", lambda sid: snapshots[sid])
    # like gold.check, the answers must name the snapshot they are checked against (their offsets are not re-checked here)
    monkeypatch.setattr(evaluate, "check_gold", lambda export, snapshot: (
        [f"snapshot {export.get('snapshot_id')} != {snapshot.id}"] if export.get("snapshot_id") != snapshot.id else [], [], []))
    monkeypatch.setattr(evaluate, "git_state", lambda: {"commit": "e", "dirty": False})
    monkeypatch.setattr(evaluate, "environment", lambda: {"python": "test"})
    return cfg


def run_main(tmp_path, monkeypatch, cfg, run_id="e2e"):
    write(tmp_path / "configs" / "eval.json", cfg)
    monkeypatch.setattr(sys, "argv", ["evaluate", str(tmp_path / "configs" / "eval.json"), "--run-id", run_id])
    evaluate.main()
    return tmp_path / "runs" / run_id


def test_main_scores_span_systems_with_pooled_seeds_pairs_and_new_names(tmp_path, monkeypatch):
    cfg = e2e_config(tmp_path, monkeypatch)
    out = run_main(tmp_path, monkeypatch, cfg)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert list(m["systems"]) == ["gliner_zero_shot", "dictionary", "ws_s1", "ws_s2"]
    assert m["systems"]["ws_s1"]["setting"] == "S3" and m["systems"]["ws_s2"]["seed"] == 2
    assert m["scores"]["ws_s1"]["micro"] == {"precision": 0.6, "recall": 0.4286, "f1": 0.5}  # tp 3, fp 2, fn 4
    assert "by_gold_origin_post_hoc" in m["scores"]["ws_s1"] and "cross_chapter_group" in m["scores"]["ws_s1"]
    assert m["per_chapter_counts"]["ws_s2"] == {"ch0781": [1, 0, 2], "ch0782": [2, 0, 2]}
    assert list(m["intervals"]["paired_difference"]) == ["ws_s1 - dictionary"]
    assert list(m["intervals_by_whole_plot_unit_sensitivity"]["paired_difference"]) == ["ws_s1 - dictionary"]
    assert set(m["intervals"]["systems"]) == {"gliner_zero_shot", "dictionary", "ws_s1", "ws_s2"}
    assert list(m["pair_kinds"]) == ["ws_s1 - dictionary"]
    assert m["models"]["models"]["ws"]["seeds"] == [1, 2] and list(m["models"]["pooled_paired"]) == ["ws - gl"]
    assert m["pairs"]["systems"] == ["ws_s1 - dictionary"] and m["pairs"]["models"] == ["ws - gl"]
    # the record says what it holds: a pooled model has mean, range and pooled counts here, no interval of its own
    assert not any("ci" in k or "interval" in k for k in m["models"]["models"]["ws"])
    assert "a pooled model gets no interval of its own in the main scores" in m["pairs"]["rule"]
    assert "every system and model still gets its own interval" not in m["pairs"]["rule"]
    assert "weak build from committed code" in " ".join(m["systems"]["ws_s1"]["checks"])
    nn = m["new_name_recall"]
    assert list(nn["seen_sets"]) == ["mentions<=780;dictionary=all"] and nn["seen_sets"]["mentions<=780;dictionary=all"]["gold"]["new_spans"] == 3
    recall = {s: v["recall"] for s, v in nn["systems"].items()}
    assert recall == {"gliner_zero_shot": 0.3333, "dictionary": 0.0, "ws_s1": 0.6667, "ws_s2": 0.6667}
    assert nn["systems"]["ws_s1"]["precision_among_predictions_with_unseen_text"] == 0.5
    assert nn["models"]["ws"]["pooled"]["recall"] == 0.6667 and nn["models"]["gl"]["pooled"]["recall"] == 0.3333
    # two seeds: pooled tp + fn is 6, the new gold spans are still 3
    assert nn["models"]["ws"]["pooled"]["tp"] + nn["models"]["ws"]["pooled"]["fn"] == 6
    assert nn["models"]["ws"]["pooled"]["new_gold_spans"] == 3 == nn["systems"]["ws_s1"]["new_gold_spans"]
    assert list(nn["paired_difference_recall"]) == ["ws_s1 - dictionary"] and list(nn["pooled_paired_difference_recall"]) == ["ws - gl"]
    assert (out / "predictions_test_ws_s1.jsonl").exists()


@pytest.mark.parametrize("with_new_names,message", [
    (False, "seen_names only sets the new-name view's seen set, and this config has no new_names"),
    (True, "a seen-set rule is"),
])
def test_main_refuses_a_seen_names_rule_that_would_be_recorded_unchecked(tmp_path, monkeypatch, with_new_names, message):
    cfg = e2e_config(tmp_path, monkeypatch)
    if not with_new_names:
        del cfg["new_names"]
    cfg["span_systems"][0]["seen_names"] = {"cutoff_chapter": 999, "bogus": 1}  # unknown key, cutoff after the test chapters
    with pytest.raises(SystemExit) as err:
        run_main(tmp_path, monkeypatch, cfg, "e2e-bad")
    assert message in str(err.value) and "ws_s1" in str(err.value)
    assert not (tmp_path / "runs" / "e2e-bad" / "config.json").exists()  # refused before anything is recorded
    assert not (tmp_path / "runs" / "e2e-bad" / "metrics.json").exists()


def test_main_without_seen_names_or_new_names_still_scores(tmp_path, monkeypatch):
    cfg = e2e_config(tmp_path, monkeypatch)
    del cfg["new_names"]
    m = json.loads((run_main(tmp_path, monkeypatch, cfg) / "metrics.json").read_text(encoding="utf-8"))
    assert "new_name_recall" not in m and "new_name_seen_rule" not in m["systems"]["ws_s1"]
    assert m["scores"]["ws_s1"]["micro"] == {"precision": 0.6, "recall": 0.4286, "f1": 0.5}


def test_main_scores_a_later_snapshot_with_the_training_snapshots_dictionary_builds_and_seen_names(tmp_path, monkeypatch):
    """Stage 3's shape: the answers and the text from the later export "score", the trained systems, the weak build, the
    dictionary and the seen names from the training snapshot "train" (whose dev answers choose GLiNER's threshold)."""
    snapshots = {}
    cfg = e2e_config(tmp_path, monkeypatch, lineage=True, snapshots=snapshots)
    out = run_main(tmp_path, monkeypatch, cfg)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    c = json.loads((out / "config.json").read_text(encoding="utf-8"))
    for rec in (m, c):
        s = rec["snapshots"]
        assert (s["training_snapshot"]["id"], s["scoring_snapshot"]["id"], s["shared_chapters"]) == ("train", "score", 6)
        assert s["shared_text_identical"] is True and s["chapters_only_in_the_scoring_snapshot"] == 2 and "used_for" in s
    assert c["snapshot"]["id"] == "score" and c["config"]["training_snapshot"] == "train"
    # the dictionary system: the training snapshot's one row (汀城) matched in the scoring snapshot's chapters; the row the
    # later export gained (白塔, at ch0781 9-11 and ch0782 3-5) is not matched
    dic = m["systems"]["dictionary"]
    assert dic["outputs"]["outputs"] == 2 and dic["resource"]["entries"] == 1
    assert (dic["snapshot"], dic["from_training_snapshot"]) == ("train", True)
    assert m["per_chapter_counts"]["dictionary"] == {"ch0781": [1, 0, 2], "ch0782": [1, 0, 3]}
    # the seen names come from the training snapshot: 白塔 is still new (3 new gold spans, as on one snapshot)
    nn = m["new_name_recall"]
    assert nn["seen_sets"]["mentions<=780;dictionary=all"]["gold"]["new_spans"] == 3 and "training snapshot's dictionary" in nn["rule"]
    assert {s: v["recall"] for s, v in nn["systems"].items()} == {"gliner_zero_shot": 0.3333, "dictionary": 0.0, "ws_s1": 0.6667, "ws_s2": 0.6667}
    # the trained systems score as on one snapshot, with the lineage in their records
    assert m["scores"]["ws_s1"]["micro"] == {"precision": 0.6, "recall": 0.4286, "f1": 0.5}
    for s in ("ws_s1", "ws_s2"):
        assert m["systems"][s]["training_snapshot"] == "train" and m["systems"][s]["snapshot_lineage"]["model_snapshot"] == "train"
        assert "lineage from the training snapshot train" in m["systems"][s]["checks"][0]
    assert m["models"]["models"]["ws"]["seeds"] == [1, 2] and list(m["models"]["pooled_paired"]) == ["ws - gl"]
    # without dictionary.from_training_snapshot the later export's dictionary would be the resource: refused before any record
    with pytest.raises(SystemExit, match="from_training_snapshot"):
        run_main(tmp_path, monkeypatch, {**cfg, "dictionary": {"min_len": 2}}, "e2e-nodict")
    assert not (tmp_path / "runs" / "e2e-nodict" / "config.json").exists()
    # a scoring snapshot that is not a later export of the training snapshot (a shared chapter's text differs) is refused
    changed = Snapshot("score", None, snapshots["score"].meta,
                       {**snapshots["score"].chapters, "ch0001": Chapter("ch0001", 1, [("p0001", "改过的字。")], "", "h")},
                       snapshots["score"].mentions, snapshots["score"].dictionary)
    monkeypatch.setattr(evaluate, "load_snapshot", lambda sid: {"train": snapshots["train"], "score": changed}[sid])
    with pytest.raises(SystemExit, match="not a later export"):
        run_main(tmp_path, monkeypatch, cfg, "e2e-changed")
    assert not (tmp_path / "runs" / "e2e-changed" / "config.json").exists()


def test_a_training_snapshot_equal_to_the_snapshot_changes_nothing(tmp_path, monkeypatch):
    """With the key absent, equal to the snapshot, or equal with dictionary.from_training_snapshot set, the metrics and the
    saved predictions are the same (the key is a no-op on one snapshot; nothing about the lineage is recorded)."""
    cfg = e2e_config(tmp_path, monkeypatch)

    def metrics_of(cfg, run_id):
        m = json.loads((run_main(tmp_path, monkeypatch, cfg, run_id) / "metrics.json").read_text(encoding="utf-8"))
        assert m.pop("run_id") == run_id
        del m["finished_at"]  # the only fields that differ between runs of one config: the id, the clock and the timings
        for split in m["systems"]["gliner_zero_shot"]["run_stats"].values():
            del split["inference_seconds"]
        return m

    plain = metrics_of(cfg, "e2e-plain")
    same = metrics_of({**cfg, "training_snapshot": "snap"}, "e2e-same")
    flagged = metrics_of({**cfg, "training_snapshot": "snap", "dictionary": {"min_len": 2, "from_training_snapshot": True}}, "e2e-flag")
    assert plain == same == flagged
    assert "snapshots" not in plain and "training_snapshot" not in plain["systems"]["ws_s1"] and plain["systems"]["ws_s1"]["checks"][0] == "snapshot"
    assert "snapshot" not in plain["systems"]["dictionary"] and "training" not in plain["new_name_recall"]["rule"]
    for name in ("dictionary", "ws_s1", "ws_s2", "gliner_zero_shot"):
        digests = {evaluate.sha256(tmp_path / "runs" / rid / f"predictions_test_{name}.jsonl") for rid in ("e2e-plain", "e2e-same", "e2e-flag")}
        assert len(digests) == 1


def test_span_system_recomputes_a_teacher_builds_chapters_from_its_packs(tmp_path, monkeypatch):
    """A teacher labels build (novel_ner.teacher) samples its training chapters: they are recomputed from the packs its
    record lists and the chapters file it hashed, never by weak.split_ids."""
    from novel_ner.teacher import TEACHER_BUILD_KIND

    teach = tmp_path / "configs" / "teach.json"
    write(teach, {"dev_pack": "d01", "packs": {"b01": ["ch0004", "ch0001", "ch0002"], "b02": ["ch0006", "ch0005"], "d01": ["ch0003"]}})

    def change(w):
        w["build"] = {**w["build"], "kind": TEACHER_BUILD_KIND,
                      "config": {"dev_config": "configs/dev.json", "max_train_chapter": 780},  # no test_order: split_ids cannot apply
                      "split": {**w["build"]["split"], "dev_chapters": ["ch0003"]},
                      "teacher": {"packs": ["b01", "b02"]},
                      "chapters_file": {"file": "configs/teach.json", "sha256": evaluate.sha256(teach)}}
    snap, w = world(tmp_path, monkeypatch, change=change)
    _, _, rec, pool = load(snap, w)
    assert rec["weak_build"]["train_chapters"] == 5 and rec["weak_build"]["train_chapter_numbers"] == [1, 6]
    assert rec["weak_build"]["kind"] == TEACHER_BUILD_KIND and pool["weak_build"] == "runs/wb"
    build = json.loads((tmp_path / "runs" / "wb" / "config.json").read_text(encoding="utf-8"))
    for packs, message in ((["b01"], "do not give the sha256"),  # a pack the build did not keep
                           (["b01", "b02", "d01"], "cannot be recomputed"),  # the dev pack is never training material
                           (["b01", "b01"], "cannot be recomputed")):
        write(tmp_path / "runs" / "wb" / "config.json", {**build, "teacher": {"packs": packs}})
        with pytest.raises(SystemExit, match=message):
            load(snap, w)
    write(tmp_path / "runs" / "wb" / "config.json", build)
    write(teach, {"dev_pack": "d01", "packs": {"b01": ["ch0004", "ch0001", "ch0002"], "b02": ["ch0006", "ch0005"], "d01": []}})
    with pytest.raises(SystemExit, match="cannot be recomputed"):  # the chapters file is no longer the one the build hashed
        load(snap, w)


def test_span_system_scores_a_vote_run_by_its_target_predict_runs_chapters(tmp_path, monkeypatch):
    """A C2 vote run (vote.vote_run) records its chapters under predictions_run, not at the top level."""
    snap, w = world(tmp_path, monkeypatch)
    target = tmp_path / "runs" / "p1-test"
    out = tmp_path / "runs" / "v1-c2-test"
    rows = [{**r, "type": "location"} for r in w["runs"][1]["rows"]]  # the vote changed types at the same places
    out.mkdir(parents=True)
    evaluate.write_jsonl(out / "predictions.jsonl", rows)
    rec = json.loads((target / "config.json").read_text(encoding="utf-8"))
    record = {"run_id": "v1-c2-test", "model_run": "m1", "model_run_config_sha256": rec["model_run_config_sha256"],
              "setting": "S3", "seed": 1, "snapshot": {"id": "snap"}, "code": {"commit": "v", "dirty": False},
              "predictions_run": {"run_id": "p1-test", "config_sha256": evaluate.sha256(target / "config.json"),
                                  "chunking": "paragraph500", "chapters": list(TEST_IDS)},
              "material_run": {"run_id": "v1-c2-test-material", "chapters": 5, "chapter_ids_sha256": "m",
                               "equals_model_training_chapters": True, "chunking": "paragraph500"},
              "vote_rule": "majority type over the material", "material_rule": "the training chapters"}
    write(out / "config.json", record)
    write(out / "metrics.json", {"run_id": "v1-c2-test", "outputs": len(rows), "predictions_sha256": evaluate.sha256(out / "predictions.jsonl")})
    spec = {"name": "ws_c2", "predictions_run": "runs/v1-c2-test", "input_condition": "(d') chapter text plus the vote"}
    pred, got, rec2, pool = span_system(spec, snap, w["test_ids"], w["dev_ids"], w["order_ids"])
    assert all(r["type"] == "location" for r in got) and pred["ch0781"]["spans"] == {(9, 11, "location"), (0, 1, "location")}
    assert rec2["vote"]["predictions_run"] == "runs/p1-test" and rec2["vote"]["material_run"] == "runs/v1-c2-test-material"
    assert rec2["vote"]["material_equals_model_training_chapters"] is True and rec2["chunking"] == "paragraph500"
    assert rec2["predict_stats"] is None and pool["seed"] == 1 and any("vote run" in c for c in rec2["checks"])
    write(target / "config.json", {**rec, "started_at": "later"})  # the vote's target must be the file the vote hashed
    with pytest.raises(SystemExit, match="target predict run"):
        span_system(spec, snap, w["test_ids"], w["dev_ids"], w["order_ids"])
    write(out / "config.json", {k: v for k, v in record.items() if k != "vote_rule"})  # not a vote run: no chapters at all
    with pytest.raises(SystemExit, match="chapter list is missing"):
        span_system(spec, snap, w["test_ids"], w["dev_ids"], w["order_ids"])


def test_seeds_that_differ_only_in_the_schedule_length_pool_and_record_it(tmp_path, monkeypatch):
    """The sufficiency rule re-runs a flagged seed under more epochs (configs/train_s1_base_e60.json differs from
    train_s1_base.json in max_epochs and the description only); that run replaces the flagged one in its arm."""
    def change(w):
        w["runs"][1]["model"]["config"]["max_epochs"] = 40
        w["runs"][2]["model"]["config"] = {**w["runs"][2]["model"]["config"], "max_epochs": 60, "description": "the sufficiency re-run"}
    snap, w = world(tmp_path, monkeypatch, seeds=(1, 2), change=change)
    pool = {f"ws_s{s}": load(snap, w, s)[3] for s in (1, 2)}
    assert pool["ws_s1"]["key"] != pool["ws_s2"]["key"] and pool["ws_s1"]["schedule_free_key"] == pool["ws_s2"]["schedule_free_key"]
    about = check_models({"ws": ["ws_s1", "ws_s2"]}, set(pool), pool)
    assert about["ws"]["seeds"] == [1, 2] and about["ws"]["max_epochs_by_run"] == {"ws_s1": 40, "ws_s2": 60}
    assert "sufficiency" in about["ws"]["schedule_note"]
    same = check_models({"ws": ["ws_s1"]}, set(pool), pool)
    assert "max_epochs_by_run" not in same["ws"]  # nothing to record when the configs are identical
    pool["ws_s2"] = {**pool["ws_s2"], "schedule_free_key": "other"}  # anything beyond the schedule still refuses
    with pytest.raises(SystemExit, match="differ in more than the seed"):
        check_models({"ws": ["ws_s1", "ws_s2"]}, set(pool), pool)
