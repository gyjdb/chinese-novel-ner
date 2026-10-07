"""diagnose.py: the error-kind rules, the edge distances, the two what-if post-processings, the seed vote, and a whole
synthetic run on disk (counts only in metrics.json, a sha256 mismatch stops the run). Synthetic data only."""
import hashlib
import json
import re

import pytest

from novel_ner import diagnose
from novel_ner.diagnose import (TrainingIndex, agreement, analyse, bucket_of, ceiling, chunk_info, edge_dist, fn_kind, fp_kind,
                                propagate, seed_vote, type_vote)
from novel_ner.snapshot import Chapter, Snapshot

CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")
TITLES = ["医生", "大师"]
BUCKETS = [{"label": "<20", "min": 0, "max": 20}, {"label": "20-60", "min": 20, "max": 60}, {"label": ">=60", "min": 60, "max": None}]
# 徐0 获1 见2 到3 唐4 医5 生6 。7 徐8 获9 笑10 了11 。12 \n13 唐14 医15 生16 看17 着18 徐19 获20 和21 汀22 城23 。24
DEV = Chapter("ch0002", 2, [("p1", "徐获见到唐医生。徐获笑了。"), ("p2", "唐医生看着徐获和汀城。")], "", "h2")
TEXT = DEV.text


def test_fp_and_fn_kinds_with_the_title_suffix_subkinds():
    gold = {(0, 2, "person"), (4, 7, "person"), (22, 24, "location")}
    assert fp_kind((4, 7, "location"), gold, TEXT, TITLES) == ("type", None)
    assert fp_kind((4, 5, "person"), gold, TEXT, TITLES) == ("boundary", "narrower_by_title(pred=gold-title)")
    assert fp_kind((22, 23, "location"), gold, TEXT, TITLES) == ("boundary", "narrower_other")
    assert fp_kind((21, 24, "location"), gold, TEXT, TITLES) == ("boundary", "wider_other")
    assert fp_kind((1, 3, "person"), gold, TEXT, TITLES) == ("boundary", "crossing")
    assert fp_kind((1, 5, "person"), gold, TEXT, TITLES) == ("boundary", "merged_or_multi")  # overlaps two gold spans
    assert fp_kind((14, 17, "person"), gold, TEXT, TITLES) == ("spurious", None)
    # a prediction that is a gold name plus a title suffix
    assert fp_kind((4, 7, "person"), {(4, 5, "person")}, TEXT, TITLES) == ("boundary", "wider_by_title(pred=gold+title)")
    preds = {(0, 2, "person"), (4, 5, "person"), (22, 23, "location")}
    assert fn_kind((4, 7, "person"), preds, TEXT, TITLES) == ("boundary", "pred=gold-title")
    assert fn_kind((4, 5, "person"), {(4, 7, "person")}, TEXT, TITLES) == ("boundary", "pred=gold+title")
    assert fn_kind((22, 24, "location"), preds, TEXT, TITLES) == ("boundary", "other")
    assert fn_kind((0, 2, "location"), preds, TEXT, TITLES) == ("type", None)
    assert fn_kind((8, 10, "person"), preds, TEXT, TITLES) == ("missed", None)


def test_edge_distances_and_their_buckets():
    ch = Chapter("ch0005", 5, [("p1", "甲" * 30), ("p2", "乙" * 30), ("p3", "丙" * 30)], "", "h")
    chunks, paras = chunk_info(ch, 65)  # p1 + p2 fit in one chunk of 61 characters, p3 is its own
    assert chunks == [(0, 61), (62, 92)] and paras == [(0, 30), (31, 61), (62, 92)]
    n = len(ch.text)
    assert edge_dist((2, 4), chunks, paras, n) == (2, 57, 2)  # the chapter's start is not an internal cut
    assert edge_dist((65, 68), chunks, paras, n) == (3, 3, 3)  # the cut before p3 is
    assert edge_dist((40, 45), chunks, paras, n) == (16, 16, 9)
    assert edge_dist((75, 77), chunks, paras, n) == (13, 13, 13)
    assert [bucket_of(d, BUCKETS) for d in (0, 19, 20, 59, 60, 10 ** 9)] == ["<20", "<20", "20-60", "20-60", ">=60", ">=60"]
    with pytest.raises(SystemExit, match="edge buckets"):
        bucket_of(5, [{"label": "a", "min": 0, "max": 3}, {"label": "b", "min": 10, "max": None}])
    with pytest.raises(SystemExit, match="crosses a chunk edge"):
        edge_dist((60, 63), chunks, paras, n)


def test_type_vote_takes_the_majority_then_the_summed_score_then_the_type_order():
    pred = {(0, 2, "person"), (8, 10, "location"), (19, 21, "person")}
    assert type_vote(pred, TEXT, {}) == {(0, 2, "person"), (8, 10, "person"), (19, 21, "person")}
    two = {(0, 2, "person"), (8, 10, "location")}
    assert type_vote(two, TEXT, {(0, 2, "person"): 0.6, (8, 10, "location"): 0.9}) == {(0, 2, "location"), (8, 10, "location")}
    assert type_vote(two, TEXT, {(0, 2, "person"): 0.7, (8, 10, "location"): 0.7}) == {(0, 2, "person"), (8, 10, "person")}
    assert type_vote({(22, 24, "location")}, TEXT, {}) == {(22, 24, "location")}


def test_propagation_adds_free_exact_occurrences_of_2_plus_character_strings_in_first_place_order():
    pred = {(0, 2, "person"), (9, 11, "item"), (22, 23, "location")}
    # 徐获 recurs at 8 (overlaps the item prediction: skipped) and 19 (added); 汀 is one character: never propagated
    assert propagate(pred, TEXT) == pred | {(19, 21, "person")}
    # strings go in the order of their first predicted place: the earlier one takes a contested place
    text = "白塔医院在白塔。白塔医院很大。"
    both = {(0, 4, "organization"), (5, 7, "location")}
    assert propagate(both, text) == both | {(8, 12, "organization")}
    later = {(0, 2, "location"), (8, 12, "organization")}
    assert propagate(later, text) == later | {(5, 7, "location")}  # 白塔 at 8 is inside the predicted 白塔医院
    # the majority type of the string is propagated; ties go to the type order (organization before location)
    tie = {(0, 2, "location"), (5, 7, "organization")}
    assert propagate(tie, text) == tie | {(8, 10, "organization")}
    assert propagate({(0, 2, "location"), (5, 7, "organization"), (8, 10, "location")}, text) == {
        (0, 2, "location"), (5, 7, "organization"), (8, 10, "location")}  # nothing free to add


def test_seed_vote_counts_and_the_three_combinations():
    gold = {"c": {(0, 2, "person"), (4, 7, "person"), (14, 17, "person")}}
    preds = {"seed1": {"c": {(0, 2, "person"), (4, 7, "person"), (22, 24, "location")}},
             "seed2": {"c": {(0, 2, "person"), (4, 7, "person"), (14, 17, "person")}},
             "seed3": {"c": {(0, 2, "person"), (8, 10, "person"), (22, 24, "location")}}}
    out = seed_vote(["c"], gold, preds)
    assert out["gold_found_by_n_seeds"] == {"0": 0, "1": 1, "2": 1, "3": 1}
    assert out["distinct_predictions_by_n_seeds_counts"] == {"1": {"predictions": 2, "correct": 1}, "2": {"predictions": 2, "correct": 1},
                                                             "3": {"predictions": 1, "correct": 1}}
    assert out["distinct_predictions_by_n_seeds"]["1"] == "1/2 correct"
    assert {k: (v["tp"], v["fp"], v["fn"]) for k, v in ((k, out[k]) for k in ("vote_2of3", "union", "intersection"))} == {
        "vote_2of3": (2, 1, 1), "union": (3, 2, 0), "intersection": (1, 0, 2)}
    assert out["vote_2of3_non_person"]["location"] == {"tp": 0, "fp": 1, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_training_index_seen_status_and_occurrence_profile():
    # 徐0 获1 和2 唐3 医4 生5 到6 了7 汀8 城9 ，10 徐11 获12 笑13 了14 ，15 唐16 医17 生18 没19 笑20 ，21 汀22 城23 很24 大25 。26
    train = Chapter("ch0001", 1, [("p1", "徐获和唐医生到了汀城，徐获笑了，唐医生没笑，汀城很大。")], "", "h1")
    thresholds = {"same_type_high": 0.8, "same_type_low": 0.2}
    rows = [{"chapter_id": "ch0001", "split": "train",
             "spans": [{"start": 0, "end": 2, "type": "person", "text": "徐获"}, {"start": 3, "end": 6, "type": "person", "text": "唐医生"},
                       {"start": 11, "end": 13, "type": "location", "text": "徐获"}, {"start": 16, "end": 19, "type": "person", "text": "唐医生"}],
             "uncertain": [{"start": 8, "end": 10, "text": "汀城", "types": ["location"], "reason": "x"}]}]
    index = TrainingIndex(rows, {"ch0001": train}, thresholds)
    assert index.seen_kind("徐获", "person") == "seen_same_type" and index.seen_kind("徐获", "item") == "seen_other_type_only"
    assert index.seen_kind("汀城", "location") == "unseen"
    assert index.profile("徐获", "person") == {"exact_same_type": 1, "exact_other_type": 1}
    assert index.profile("唐", "person") == {"inside_longer_positive": 2}
    assert index.profile("汀城", "location") == {"uncertain_region": 1, "O": 1}
    assert index.profile("笑", "trait") == {"O": 2}
    assert index.profile("生到了", "item") == {"partial_overlap_positive": 1}
    assert index.profile_class("徐获", "person") == "train_occ_labelled_same_type_20-80%"
    assert index.profile_class("唐医生", "person") == "train_occ_labelled_same_type>=80%"
    assert index.profile_class("唐", "person") == "train_occ_<20%_same_type_mostly_inside_longer_positive"
    assert index.profile_class("没有", "person") == "absent_from_training_text"
    assert index.profile_class("徐获", "item") == "train_occ_<20%_same_type_mostly_exact_other_type"
    assert index.profile_class("汀城", "location") == "train_occ_<20%_same_type_mostly_uncertain_region"  # a tie: the kind met first
    with pytest.raises(SystemExit, match="not the chapter text"):
        TrainingIndex([{**rows[0], "spans": [{"start": 0, "end": 2, "type": "person", "text": "唐获"}]}], {"ch0001": train}, thresholds)


def test_ceiling_and_agreement_place_every_error_against_the_labels():
    ids = ["ch0002"]
    gold = {"ch0002": {(0, 2, "person"), (4, 7, "person"), (14, 17, "person"), (22, 24, "location")}}
    labels = {"ch0002": {(0, 2, "person"), (4, 7, "person"), (14, 15, "person"), (8, 10, "person")}}
    unc = {"ch0002": [(22, 24)]}
    preds = {"seed1": {"ch0002": {(0, 2, "person"), (4, 7, "person"), (8, 10, "person"), (14, 17, "location")}},
             "seed2": {"ch0002": {(0, 2, "person"), (14, 17, "person"), (22, 24, "person")}}}
    c = ceiling(ids, gold, preds, labels)
    assert c["per_type"]["micro"]["labels"] == {"tp": 2, "fp": 2, "fn": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert c["model_recall_by_label_status"]["seed1"] == {"labels_have_it_n": 2, "labels_have_it_found": 2, "labels_miss_it_n": 2,
                                                          "labels_miss_it_found": 0, "label_fp_n": 2, "label_fp_also_predicted_by_model": 1}
    a = agreement(ids, gold, preds, labels, unc)
    assert a["seed1"]["fp_in_labels"] == 1 and a["seed1"]["fp_not_in_labels"] == 1 and a["seed1"]["fn"] == 2
    assert a["seed1"]["fn_labels_miss:labels_other_boundary"] == 1 and a["seed1"]["fn_labels_miss:uncertain_region"] == 1
    assert a["seed2"]["fp_where_labels_uncertain"] == 1 and a["seed2"]["fn_labels_have_it"] == 1
    assert a["pooled"]["errors"] == 7 and a["stable_all3"] == {"fn": 1, "fn_labels_miss:uncertain_region": 1, "errors": 1,
                                                                "errors_shared_with_labels": 1, "share_shared": 1.0,
                                                                "fp_share_in_labels": None, "fn_share_labels_miss": 1.0}
    assert a["base_rates"] == {"gold_spans": 4, "gold_spans_missing_from_labels": 2, "label_positives": 4, "label_positives_wrong": 2}
    assert a["per_chapter_error_correlation"]["pearson"] is None  # one chapter: no variance


# ------------------------------------------------------------------------------- a whole synthetic run on disk

TRAIN = Chapter("ch0001", 1, [("p1", "徐获拿出回音铃。"), ("p2", "唐医生到了汀城。")], "", "h1")
DEV2 = Chapter("ch0004", 4, [("p1", "唐医生见到徐获。")], "", "h4")
TEST = Chapter("ch0781", 781, [("p1", "徐获回到汀城，唐医生笑了。")], "", "h5")  # 徐0 获1 回2 到3 汀4 城5 ，6 唐7 医8 生9 笑10


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def edit_json(path, change):
    """Apply change to the JSON object in path and write it back (a record edited after the fact)."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    change(obj)
    write(path, obj)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_of(chapters, annotations):
    return {"chapters": {c.chapter_id: {"status": "done", "annotations": [{"start": s, "end": e, "type": t, "pending": False, "text": c.text[s:e]}
                                                                           for s, e, t in annotations[c.chapter_id]]} for c in chapters}}


def world(tmp_path, monkeypatch):
    root = tmp_path
    snap = Snapshot("snap", None, {"source_commit": "c"}, {c.chapter_id: c for c in (TRAIN, DEV, DEV2, TEST)}, [], [])
    monkeypatch.setattr(diagnose, "load_snapshot", lambda sid, root=None: snap)
    monkeypatch.setattr(diagnose, "check_gold", lambda export, s: ([], [], {}))
    monkeypatch.setattr(diagnose, "REPO_ROOT", root)
    monkeypatch.setattr(diagnose, "git_state", lambda: {"commit": "c0", "dirty": False})
    write(root / "configs" / "weak.json", {"title_suffixes": TITLES})
    row = lambda cid, split, spans, unc=(): {"chapter_id": cid, "number": int(cid[2:]), "split": split,
                                             "spans": [{"start": s, "end": e, "type": t, "text": snap.chapters[cid].text[s:e]} for s, e, t in spans],
                                             "uncertain": [{"start": s, "end": e, "text": snap.chapters[cid].text[s:e], "types": None, "reason": r}
                                                           for s, e, r in unc]}
    labels = root / "data" / "weak_labels" / "wb" / "labels_S3.jsonl"
    write_rows(labels, [row("ch0001", "train", [(0, 2, "person"), (4, 7, "item"), (9, 12, "person"), (14, 16, "location")]),
                        row("ch0002", "dev", [(0, 2, "person"), (4, 7, "person"), (19, 21, "person")], [(22, 24, "other_chapters_only")]),
                        row("ch0004", "dev", [(0, 3, "person"), (5, 7, "person")])])
    gold_dev = root / "data" / "gold" / "dev.json"
    write(gold_dev, export_of([DEV, DEV2], {"ch0002": [(0, 2, "person"), (4, 7, "person"), (8, 10, "person"), (14, 17, "person"),
                                                        (19, 21, "person"), (22, 24, "location")],
                                             "ch0004": [(0, 3, "person"), (5, 7, "person")]}))
    gold_test = root / "data" / "gold" / "test.json"
    write(gold_test, export_of([TEST], {"ch0781": [(0, 2, "person"), (4, 6, "location"), (7, 10, "person")]}))
    write(root / "runs" / "wb" / "config.json", {"run_id": "wb", "snapshot": {"id": "snap"},
                                                  "config": {"title_suffixes": TITLES, "max_train_chapter": 780},
                                                  "labels": {"S3": {"file": "data/weak_labels/wb/labels_S3.jsonl", "sha256": digest(labels)}},
                                                  "gold_dev": {"file": "data/gold/dev.json", "sha256": digest(gold_dev)},
                                                  "split": {"train_ids_sha256": "t"}})
    write(root / "runs" / "ev" / "config.json", {"run_id": "ev", "gold_test": {"file": "data/gold/test.json", "sha256": digest(gold_test)}})
    # the micro scores the training runs (dev) and the scoring run (test) recorded, worked out by hand from the spans below
    dev_micro = {1: {"tp": 5, "fp": 2, "fn": 3, "micro": {"precision": 0.7143, "recall": 0.625, "f1": 0.6667}},
                 2: {"tp": 5, "fp": 1, "fn": 3, "micro": {"precision": 0.8333, "recall": 0.625, "f1": 0.7143}},
                 3: {"tp": 5, "fp": 1, "fn": 3, "micro": {"precision": 0.8333, "recall": 0.625, "f1": 0.7143}}}
    test_scores = {1: {"micro": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                       "per_type": {"person": {"tp": 2, "fp": 0, "fn": 0}, "location": {"tp": 1, "fp": 0, "fn": 0}}},
                   2: {"micro": {"precision": 0.5, "recall": 0.3333, "f1": 0.4},
                       "per_type": {"person": {"tp": 1, "fp": 1, "fn": 1}, "location": {"tp": 0, "fp": 0, "fn": 1}}},
                   3: {"micro": {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667},
                       "per_type": {"person": {"tp": 1, "fp": 0, "fn": 1}, "location": {"tp": 1, "fp": 0, "fn": 0}, "item": {"tp": 0, "fp": 1, "fn": 0}}}}
    dev_preds = {1: [("ch0002", 0, 2, "person", 0.9), ("ch0002", 4, 7, "person", 0.9), ("ch0002", 8, 10, "person", 0.8),
                     ("ch0002", 14, 17, "dungeon", 0.5), ("ch0002", 22, 23, "location", 0.4), ("ch0004", 0, 3, "person", 0.9), ("ch0004", 5, 7, "person", 0.9)],
                 2: [("ch0002", 0, 2, "person", 0.9), ("ch0002", 4, 7, "person", 0.9), ("ch0002", 14, 17, "person", 0.7),
                     ("ch0002", 19, 21, "person", 0.9), ("ch0004", 0, 3, "person", 0.9), ("ch0004", 5, 6, "person", 0.6)],
                 3: [("ch0002", 0, 2, "person", 0.9), ("ch0002", 4, 5, "person", 0.6), ("ch0002", 19, 21, "person", 0.9),
                     ("ch0002", 22, 24, "location", 0.9), ("ch0004", 0, 3, "person", 0.9), ("ch0004", 5, 7, "person", 0.9)]}
    test_preds = {1: [("ch0781", 0, 2, "person", 0.9), ("ch0781", 4, 6, "location", 0.9), ("ch0781", 7, 10, "person", 0.9)],
                  2: [("ch0781", 0, 2, "person", 0.9), ("ch0781", 7, 8, "person", 0.5)],
                  3: [("ch0781", 0, 2, "item", 0.5), ("ch0781", 4, 6, "location", 0.9), ("ch0781", 7, 10, "person", 0.9)]}
    rows_of = lambda ps: [{"chapter_id": c, "start": s, "end": e, "text": snap.chapters[c].text[s:e], "type": t, "score": sc} for c, s, e, t, sc in ps]
    cfg = {"description": "synthetic", "snapshot": "snap",
           "weak_build": {"run": "runs/wb", "setting": "S3", "labels": "data/weak_labels/wb/labels_S3.jsonl"},
           "title_suffixes_from": "configs/weak.json",
           "dev": {"gold": "data/gold/dev.json", "predictions": []},
           "test": {"gold": "data/gold/test.json", "gold_recorded_in": "runs/ev/config.json", "predictions": []},
           "chunk_max_chars": 500, "edge_buckets": BUCKETS, "training_profile_thresholds": {"same_type_high": 0.8, "same_type_low": 0.2},
           "extras": {"top_strings": 3, "worst_chapters": 2},
           "rules": {k: "in words" for k in (*diagnose.DEV_ANALYSES, *diagnose.EXAM_ANALYSES, "exam_secondary")}}
    for seed in (1, 2, 3):
        t = root / "runs" / f"t{seed}"
        write_rows(t / "predictions_dev.jsonl", rows_of(dev_preds[seed]))
        write(t / "config.json", {"run_id": f"t{seed}", "snapshot": {"id": "snap"}, "weak_labels": {"build": "wb", "sha256": digest(labels)},
                                  "setting": "S3", "seed": seed, "gold_dev": {"sha256": digest(gold_dev)}, "config": {"max_chars": 500},
                                  "code": {"commit": "a", "dirty": False}})
        write(t / "metrics.json", {"run_id": f"t{seed}", "predictions_dev_sha256": digest(t / "predictions_dev.jsonl"),
                                   "dev": {"at_best": dev_micro[seed],
                                           "predict_stats_at_best": {"chunks": 2, "windows": 2, "chunks_over_max_tokens": 0}}})
        p = root / "runs" / f"t{seed}-predict-test"
        write_rows(p / "predictions.jsonl", rows_of(test_preds[seed]))
        write(p / "config.json", {"run_id": p.name, "model_run": f"t{seed}", "model_run_config_sha256": digest(t / "config.json"),
                                  "setting": "S3", "seed": seed, "snapshot": {"id": "snap"}, "code": {"commit": "b", "dirty": False}})
        write(p / "metrics.json", {"run_id": p.name, "predictions_sha256": digest(p / "predictions.jsonl"),
                                   "predict_stats": {"chunks": 1, "windows": 1, "chunks_over_max_tokens": 0}})
        cfg["dev"]["predictions"].append(f"runs/t{seed}/predictions_dev.jsonl")
        cfg["test"]["predictions"].append(f"runs/t{seed}-predict-test/predictions.jsonl")
    write(root / "runs" / "ev" / "metrics.json", {
        "run_id": "ev",
        "systems": {f"sys{seed}": {"predictions_run": f"runs/t{seed}-predict-test",
                                   "predictions_sha256": digest(root / "runs" / f"t{seed}-predict-test" / "predictions.jsonl")} for seed in (1, 2, 3)},
        "scores": {f"sys{seed}": test_scores[seed] for seed in (1, 2, 3)}})
    write(root / "configs" / "diag.json", cfg)
    return root, cfg, snap


def strings_in(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from strings_in(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings_in(v)
    elif isinstance(x, str):
        yield x


def test_a_whole_synthetic_run_writes_counts_only_and_numbers_the_strings(tmp_path, monkeypatch):
    root, cfg, snap = world(tmp_path, monkeypatch)
    a = analyse(cfg, root)
    m = diagnose.build_metrics("r", cfg, a)
    assert not any(CJK.search(s) for s in strings_in(m))  # no string from the book
    assert any(CJK.search(s) for s in strings_in(a["keys"]))  # the numbered strings live in keys.json
    d = m["dev"]
    assert (d["chapters"], d["scored_gold"], d["seeds"]) == (2, 8, ["seed1", "seed2", "seed3"])
    assert d["1_ceiling"]["per_type"]["micro"]["labels"]["tp"] == 5 and d["1_ceiling"]["per_type"]["micro"]["pooled"]["tp"] == 15
    assert d["2_agreement_with_labels"]["base_rates"] == {"gold_spans": 8, "gold_spans_missing_from_labels": 3, "label_positives": 5,
                                                          "label_positives_wrong": 0}
    assert d["3_error_kinds"]["seed1"]["fp_kind"] == {"type": 1, "boundary": 1}
    assert d["3_error_kinds"]["seed1"]["type_confusion_gold>pred"] == {"person>dungeon": 1}
    assert d["3_error_kinds"]["seed3"]["fp_boundary_sub"] == {"narrower_by_title(pred=gold-title)": 1}
    assert d["3_error_kinds"]["seed3"]["fn_boundary_sub"] == {"pred=gold-title": 1}
    assert d["3_error_kinds"]["recall_by_seen_status"] == {"seen_same_type": {"gold": 8, "found_seed_sum": 15, "pooled_recall": 0.625}}
    assert d["4_context"]["chunks"] == {"chunks": 2, "chapters": 2, "mean_chars": 16.5, "single_chunk_chapters": 2, "max_chars": 500}
    assert d["4_context"]["paragraph_edges_control"]["<20"]["gold_spans"] == 8
    w = d["4b_consistency"]["what_if_postprocessing"]["seed1"]
    assert (w["base"]["tp"], w["type_vote"]["tp"]) == (5, 6)  # the dungeon twin of a person prediction is outvoted by its score
    assert d["5_seeds"]["gold_found_by_n_seeds"] == {"0": 0, "1": 3, "2": 3, "3": 2}
    e = d["6_extras"]
    assert e["person_dungeon_type_errors_by_string"]["by_string"][0] == {
        "string": 1, "chars": 3, "errors": 1, "chapters": 1, "training_profile_as_dungeon": {"exact_other_type": 1},
        "training_exact_person_occurrences": 1, "dev_uncertain_reasons": {}}
    assert e["worst_chapters"]["chapters"] == [{"chapter": "ch0002", "pooled_errors": 11}, {"chapter": "ch0004", "pooled_errors": 2}]
    assert e["gold_touching_uncertain"] == {"gold": 1, "found_seed_sum": 1, "pooled_recall": 0.3333}
    assert e["uncertain_reasons"] == {"other_chapters_only": 1}
    assert e["top_fn_strings"]["top_n"] == 3 and len(e["top_fn_strings"]["by_string"]) == 3 and set(a["keys"]["fn_strings"]) == {"1", "2", "3"}
    x = m["exam_secondary"]
    assert x["micro_check"]["seed1"] == {"tp": 3, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert x["6_extras_exam_view"]["gold_found_by_n_seeds_by_string_status"] == {"seen_same_type": {"0": 0, "1": 0, "2": 3, "3": 0}}
    assert x["6_extras_exam_view"]["error_kinds_by_seen_status"]["fp"] == {"unseen": {"boundary": 1}, "seen_other_type_only": {"type": 1}}
    assert x["micro_check_equals_scoring_run"] == {"run": "ev", "systems": {"seed1": "sys1", "seed2": "sys2", "seed3": "sys3"}}
    assert x["micro_check"]["seed2"] == {"tp": 1, "fp": 1, "fn": 2, "precision": 0.5, "recall": 0.3333, "f1": 0.4}
    assert m["checks"]["each_seed_test_micro_equals_the_scoring_run_record_of_the_same_predictions"] is True
    assert m["checks"]["input_files_hashed"] == len(a["files"]) == 16
    # the command line: a scratch run with --out, refused a second time; a formal run needs committed code
    out = tmp_path / "scratch"
    diagnose.main([str(root / "configs" / "diag.json"), "--run-id", "r", "--out", str(out)])
    written = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert written["dev"]["5_seeds"] == m["dev"]["5_seeds"] and (out / "keys.json").exists() and (out / "config.json").exists()
    assert not CJK.search((out / "metrics.json").read_text(encoding="utf-8")) and CJK.search((out / "keys.json").read_text(encoding="utf-8"))
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["config"]["rules"] == cfg["rules"]
    with pytest.raises(SystemExit, match="never overwritten"):
        diagnose.main([str(root / "configs" / "diag.json"), "--run-id", "r", "--out", str(out)])
    monkeypatch.setattr(diagnose, "git_state", lambda: {"commit": "c0", "dirty": True})
    with pytest.raises(SystemExit, match="committed code"):
        diagnose.main([str(root / "configs" / "diag.json"), "--run-id", "r2"])
    assert not (root / "runs" / "r2").exists()


@pytest.mark.parametrize("tamper,message", [
    ("labels", "sha256 the weak build recorded"),
    ("dev_predictions", "sha256 its run recorded"),
    ("test_predictions", "sha256 its run recorded"),
    ("gold_dev", "sha256 the weak build recorded"),
    ("gold_test", "sha256 the scoring run recorded"),
    ("training_config", "model_run and its config.json sha256"),
    ("title_suffixes", "title suffixes"),
    ("max_chars", "max_chars"),
    ("seed_twice", "listed twice"),
    ("missing_rule", "no rule in words"),
    ("chapter_coverage", "cover exactly"),
    ("dev_micro_record", "micro fp 2 is not the recorded 3"),
    ("test_micro_record", "micro f1 0.4 is not the recorded 0.41"),
    ("test_system_sha", "scored other predictions"),
    ("test_system_missing", "records 0 systems"),
    ("dev_chunk_record", "recorded 3 chunks, this run makes 2"),
    ("test_chunk_windows", "split chunks into token windows"),
    ("scoring_run_folder", "do not both name the folder ev")])
def test_inputs_that_do_not_match_stop_the_run(tmp_path, monkeypatch, tamper, message):
    root, cfg, snap = world(tmp_path, monkeypatch)
    append = lambda p: p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    if tamper == "labels":
        append(root / "data" / "weak_labels" / "wb" / "labels_S3.jsonl")
    elif tamper == "dev_predictions":
        append(root / "runs" / "t2" / "predictions_dev.jsonl")
    elif tamper == "test_predictions":
        append(root / "runs" / "t3-predict-test" / "predictions.jsonl")
    elif tamper == "gold_dev":
        append(root / "data" / "gold" / "dev.json")
    elif tamper == "gold_test":
        append(root / "data" / "gold" / "test.json")
    elif tamper == "training_config":
        f = root / "runs" / "t1" / "config.json"
        write(f, {**json.loads(f.read_text(encoding="utf-8")), "note": "edited"})
    elif tamper == "title_suffixes":
        write(root / "configs" / "weak.json", {"title_suffixes": TITLES + ["老师"]})
    elif tamper == "max_chars":
        cfg["chunk_max_chars"] = 400
    elif tamper == "seed_twice":
        cfg["dev"]["predictions"][1] = cfg["dev"]["predictions"][0]
        cfg["test"]["predictions"][1] = cfg["test"]["predictions"][0]
    elif tamper == "missing_rule":
        del cfg["rules"]["4b_consistency"]
    elif tamper == "chapter_coverage":
        f = root / "runs" / "t1" / "predictions_dev.jsonl"
        rows = [r for r in f.read_text(encoding="utf-8").splitlines() if '"ch0004"' not in r]
        f.write_text("\n".join(rows) + "\n", encoding="utf-8")
        edit_json(root / "runs" / "t1" / "metrics.json", lambda m: m.update(predictions_dev_sha256=digest(f)))
    elif tamper == "dev_micro_record":
        edit_json(root / "runs" / "t1" / "metrics.json", lambda m: m["dev"]["at_best"].update(fp=3))
    elif tamper == "test_micro_record":
        edit_json(root / "runs" / "ev" / "metrics.json", lambda m: m["scores"]["sys2"]["micro"].update(f1=0.41))
    elif tamper == "test_system_sha":
        edit_json(root / "runs" / "ev" / "metrics.json", lambda m: m["systems"]["sys3"].update(predictions_sha256="0" * 64))
    elif tamper == "test_system_missing":
        edit_json(root / "runs" / "ev" / "metrics.json", lambda m: m["systems"].pop("sys1"))
    elif tamper == "dev_chunk_record":
        edit_json(root / "runs" / "t2" / "metrics.json", lambda m: m["dev"]["predict_stats_at_best"].update(chunks=3))
    elif tamper == "test_chunk_windows":
        edit_json(root / "runs" / "t2-predict-test" / "metrics.json", lambda m: m["predict_stats"].update(windows=2, chunks_over_max_tokens=1))
    elif tamper == "scoring_run_folder":
        edit_json(root / "runs" / "ev" / "metrics.json", lambda m: m.update(run_id="ev2"))
    with pytest.raises(SystemExit, match=message):
        analyse(cfg, root)
