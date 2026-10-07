"""errors.py: each mechanical rule on tiny synthetic chapters, the checks against evaluate's numbers, and a whole
analysis of a synthetic scoring run on disk. Synthetic data only."""
import hashlib
import json
import random
import shutil
from types import SimpleNamespace

import pytest

from novel_ner import errors
from novel_ner.errors import (a1_words, analyse, analyse_system, build_config, build_metrics, build_report, check_against_eval,
                              check_counts_only, check_new_names_against_eval, check_sums, classify_chapter, counts, llm_system,
                              make_rules, model_counts, model_occurrence, model_per_name, new_name_occurrences, per_name_recall,
                              refuse_existing)
from novel_ner.evaluate import (as_prediction, dictionary_predictions, llm_predictions, new_name_view, strict, summarize, type_counts,
                                unscored_report)
from novel_ner.novelty import novelty
from novel_ner.score import gold_spans
from novel_ner.snapshot import REPO_ROOT, Chapter, Snapshot

TYPES = ("person", "organization", "location", "dungeon", "item", "trait")
TITLES = ["医生", "先生", "队长"]
A1 = ["玩家", "女人", "男人", "他", "您", "富二代", "大学生", "小姑娘"]
GRADE = "^[A-Za-zＡ-Ｚ]级(车票|药剂|道具|副本|玩家)$"


def rules(gold_texts=(), seen=()):
    return make_rules(TITLES, A1, ["车票", "药剂", "道具"], GRADE, gold_texts, seen)


def kinds(fps):
    return {(r["start"], r["end"], r["type"]): (r["category"], r["sub"]) for r in fps}


def test_a1_words_reads_the_quoted_list_of_the_rule():
    guide = "## A\n\n### A1 泛称和代词\n\n**选择：** 不标。“玩家、女人、他”作为身份描述时不标。\n\n### A2 普通物件\n“斧头、手机”\n"
    assert a1_words(guide) == ["玩家", "女人", "他"]
    with pytest.raises(SystemExit):
        a1_words("### A2 普通物件\n")


def test_the_v9_config_copies_the_guides_a1_list_exactly():
    cfg = json.loads((REPO_ROOT / "configs" / "errors_v9.json").read_text(encoding="utf-8"))
    assert cfg["generic_words"] == a1_words((REPO_ROOT / cfg["guide"]).read_text(encoding="utf-8"))


def test_boundary_kinds_title_wider_narrower_crossing_merged():
    text = "唐医生到了汀城站。老唐和何普走了。"  # 唐0 医1 生2 到3 了4 汀5 城6 站7 。8 老9 唐10 和11 何12 普13 走14
    gold = {(0, 1, "person"), (5, 8, "location"), (10, 11, "person"), (12, 14, "person")}
    pred = {(0, 3, "person"),      # 唐医生 = 唐 + a title suffix
            (4, 8, "location"),    # 了汀城站: wider, not by a title
            (5, 7, "location"),    # 汀城 inside 汀城站
            (7, 9, "location"),    # 站。 partly overlaps
            (9, 11, "person"),     # 老唐: the title comes first, so not wider_title
            (10, 14, "person"),    # 唐和何普 covers two gold spans
            (13, 15, "item")}      # 普走 crosses, and its type differs from the gold's
    fps, fns = classify_chapter(gold, pred, [], text, rules())
    assert kinds(fps) == {(0, 3, "person"): ("boundary", "wider_title"), (4, 8, "location"): ("boundary", "wider_other"),
                          (5, 7, "location"): ("boundary", "narrower"), (7, 9, "location"): ("boundary", "crossing"),
                          (9, 11, "person"): ("boundary", "wider_other"), (10, 14, "person"): ("boundary", "merged"),
                          (13, 15, "item"): ("boundary", "crossing")}
    assert [r["type_differs"] for r in fps if r["start"] == 13] == [True]
    assert [r["type_differs"] for r in fps if r["start"] == 0] == [False]
    # every gold span is overlapped by a wrong prediction: the other side of the boundary errors
    assert {(r["start"], r["category"]) for r in fns} == {(0, "wrong_prediction_overlaps"), (5, "wrong_prediction_overlaps"),
                                                         (10, "wrong_prediction_overlaps"), (12, "wrong_prediction_overlaps")}


def test_type_error_and_whether_the_gold_span_there_was_also_found():
    text = "唐和白塔。"
    gold = {(0, 1, "person"), (2, 4, "location")}
    pred = {(0, 1, "item"), (2, 4, "location"), (2, 4, "organization")}  # one place given two types: both kept
    fps, fns = classify_chapter(gold, pred, [], text, rules())
    assert kinds(fps) == {(0, 1, "item"): ("type_error", None), (2, 4, "organization"): ("type_error", None)}
    assert {r["start"]: r["gold_here_found"] for r in fps} == {0: False, 2: True}
    assert [(r["start"], r["category"]) for r in fns] == [(0, "wrong_prediction_overlaps")]
    # score.strict counts a wrong type only against a gold span that was not found
    m = strict({"a": {"spans": gold, "unscored": []}}, {"a": {"spans": pred}})
    assert m["fp_same_span_wrong_type"] == 1


def test_spurious_kinds_in_their_order():
    text = "白塔下，玩家拿着C级道具和车票，门开了，风很大。"
    # 白塔0-2 玩家4-6 C级道具8-12 车票13-15 门16 开了17-19 风20
    pred = {(0, 2, "location"), (4, 6, "person"), (8, 12, "item"), (13, 15, "item"), (16, 17, "item"), (17, 19, "item"),
            (20, 21, "trait")}
    fps, fns = classify_chapter(set(), pred, [], text, rules(gold_texts={"白塔", "风"}))
    assert kinds(fps) == {(0, 2, "location"): ("spurious", "gold_name_elsewhere"), (4, 6, "person"): ("spurious", "generic_word"),
                          (8, 12, "item"): ("spurious", "category_word"), (13, 15, "item"): ("spurious", "category_word"),
                          (16, 17, "item"): ("spurious", "single_char"), (17, 19, "item"): ("spurious", "other"),
                          (20, 21, "trait"): ("spurious", "gold_name_elsewhere")}  # a gold text elsewhere comes first
    assert fns == []


def test_abstained_vs_missed_with_pending_places_from_llm_predictions():
    ch = Chapter("ch0781", 781, [("p0001", "唐医生到了汀城。"), ("p0002", "何普和唐走了。")], "", "h")
    snap = Snapshot("snap", None, {}, {"ch0781": ch}, [], [])
    gold = {"ch0781": {"spans": {(0, 1, "person"), (5, 7, "location"), (9, 11, "person"), (12, 13, "person")}, "unscored": []}}
    answer = {"chapters": [{"chapter_id": "ch0781", "mentions": [
        {"paragraph_id": "p0001", "text": "唐", "occurrence": 1, "pending": True},                         # abstains on 唐
        {"paragraph_id": "p0001", "text": "汀城", "occurrence": 1, "type": None, "pending": False},        # invalid: no type
        {"paragraph_id": "p0002", "text": "何普", "occurrence": 1, "pending": True},                       # pending ...
        {"paragraph_id": "p0002", "text": "何", "occurrence": 1, "type": "person", "pending": False},      # ... and a wrong guess
        {"paragraph_id": "p0002", "text": "白塔", "occurrence": 1, "pending": True}]}]}                    # pending, no place
    preds, book = llm_predictions(answer, snap, ["ch0781"])
    pred = as_prediction(preds, ["ch0781"])
    pred["ch0781"]["errors"] = book["invalid_by_chapter"].get("ch0781", 0)
    pred["ch0781"]["pending"] = [(a, b) for c, a, b in book["pending_places"] if c == "ch0781"]
    result = analyse_system(gold, pred, {"ch0781": ch.text}, rules())
    assert {(r["start"], r["category"]) for r in result["fn"]} == {
        (0, "abstained"), (5, "missed"), (9, "wrong_prediction_overlaps"), (12, "missed")}
    c = counts(result, pending_marks=sum(book["abstained_by_chapter"].values()))
    assert c["fp"]["invalid_no_position"] == 1 and c["fp"]["total"] == 2 and c["fp"]["boundary"]["sub"]["narrower"]["total"] == 1
    assert c["fn"]["wrong_prediction_overlaps"]["of_which_a_pending_mark_also_overlaps"] == 1
    assert c["pending"] == {"pending_marks": 3, "located_places": 2,
                            "places": {"touches_an_unscored_range": 0, "overlaps_a_scored_gold_span": 2, "overlaps_no_scored_gold_span": 0}}
    m = strict(gold, pred)  # evaluate's count of false negatives under a pending mark = abstained + overlapped-and-pending
    assert m["fn_where_draft_was_pending"] == c["fn"]["abstained"]["total"] + 1
    assert (m["tp"], m["fp"], m["fn"]) == (c["tp"]["total"], c["fp"]["total"], c["fn"]["total"])


def test_unscored_ranges_remove_predictions_and_gold_alike():
    text = "唐医生到了汀城。"
    gold = {"a": {"spans": {(0, 1, "person"), (5, 7, "location"), (3, 4, "item")}, "unscored": [(3, 5)]}}
    pred = {"a": {"spans": {(0, 1, "person"), (3, 5, "item"), (4, 7, "location")}}}
    result = analyse_system(gold, pred, {"a": text}, rules())
    c = counts(result)
    assert (c["tp"]["total"], c["fp"]["total"], c["fn"]["total"], c["removed_by_unscored_ranges"]) == (1, 0, 1, 2)
    assert c["fn"]["missed"]["total"] == 1  # a removed prediction counts as nothing ...
    assert c["fn"]["missed"]["of_which_a_prediction_removed_by_an_unscored_range_overlaps"] == 1  # ... but is noted
    m = strict(gold, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)


def random_world(seed):
    rng = random.Random(seed)
    chars = "唐何普白塔汀城灰医生到了下雨走和玩家门风"
    texts, gold, pred = {}, {}, {}
    for i in range(6):
        cid = f"c{i}"
        texts[cid] = "".join(rng.choice(chars) for _ in range(60))
        spans, at = set(), 0
        while at < 55:
            at += rng.randint(1, 5)
            n = rng.randint(1, 4)
            spans.add((at, at + n, rng.choice(TYPES[:5])))
            at += n
        guesses = {s for s in spans if rng.random() < 0.6}
        for _ in range(8):
            a = rng.randint(0, 57)
            guesses.add((a, a + rng.randint(1, 3), rng.choice(TYPES)))
        gold[cid] = {"spans": spans, "unscored": [(20, 23)] if i == 2 else []}
        pred[cid] = {"spans": guesses, "errors": 1 if i == 3 else 0, "pending": [(30, 34)] if i == 4 else []}
    return texts, gold, pred


def eval_like_metrics(name, gold, pred):
    """The parts of evaluate's metrics.json that check_against_eval reads, computed with evaluate's own functions."""
    tc = type_counts(gold, pred)
    m = strict(gold, pred)
    removed = sum(1 for cid, ref in gold.items() for x in pred[cid]["spans"] if errors._touches(x, ref["unscored"]))
    return {"per_chapter_counts": {name: {c: [sum(tc[c][t][i] for t in (*TYPES, "_invalid")) for i in range(3)] for c in gold}},
            "per_chapter_type_counts": {name: tc},
            "scores": {name: {**summarize(list(tc.values())), "strict_diagnostics": {k: m[k] for k in (
                "invalid_predictions", "fp_same_span_wrong_type", "fp_overlapping_gold", "fn_where_draft_was_pending")}}},
            "test": {"unscored": {"predicted_spans_removed": {name: removed}}}}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_categories_sum_to_totals_and_reproduce_evaluates_counts(seed):
    texts, gold, pred = random_world(seed)
    result = analyse_system(gold, pred, texts, rules(gold_texts={"白塔"}, seen={"唐"}))
    c = counts(result, pending_marks=1)  # counts() itself checks every category sums to its total
    metrics = eval_like_metrics("s", gold, pred)
    check_against_eval("s", result, c, metrics)
    assert sum(c["fp"][k]["total"] for k in ("type_error", "boundary", "spurious")) + c["fp"]["invalid_no_position"] == c["fp"]["total"]
    assert sum(c["fn"][k]["total"] for k in ("wrong_prediction_overlaps", "abstained", "missed")) == c["fn"]["total"]
    metrics["per_chapter_counts"]["s"]["c0"][0] += 1
    with pytest.raises(SystemExit):
        check_against_eval("s", result, c, metrics)


def test_a_category_that_does_not_sum_is_caught():
    texts, gold, pred = random_world(5)
    c = counts(analyse_system(gold, pred, texts, rules()))
    c["fp"]["spurious"]["sub"]["other"]["total"] += 1
    with pytest.raises(AssertionError):
        errors.check_sums(c)


def test_a_models_summed_counts_are_checked_and_its_rounded_means_are_display_only():
    c = {}
    for i, seed in enumerate((0, 1, 2)):
        texts, gold, pred = random_world(seed)
        c[f"r{i}"] = counts(analyse_system(gold, pred, texts, rules(gold_texts={"白塔"}, seen={"唐"})), pending_marks=1)
    summed, mean = model_counts(["r0", "r1", "r2"], c)  # check_sums(summed) runs inside and passes
    check_sums(summed)
    assert summed["fp"]["total"] == sum(c[r]["fp"]["total"] for r in c)
    assert summed["fn"]["missed"]["by_novelty"]["new"] == sum(c[r]["fn"]["missed"]["by_novelty"]["new"] for r in c)
    assert mean["fp"]["total"] == round(summed["fp"]["total"] / 3, 2)
    # thirds rounded cell by cell: here the means of the parts do not add up to the mean of the total, which is why the
    # means are for display only (not checked, not in metrics.json)
    with pytest.raises(AssertionError):
        check_sums(mean)
    broken = {**summed, "fp": {**summed["fp"], "total": summed["fp"]["total"] + 1}}
    with pytest.raises(AssertionError):
        model_counts(["x"], {"x": broken})


def test_per_name_recall_counts_distinct_names_and_models_average_runs():
    texts = {"a": "何普和唐，何普看白塔。", "b": "唐见何普，白塔会到了。"}
    # a: 何普0-2 唐3 何普5-7 白塔8-10 / b: 唐0 何普2-4 白塔5-7 (an organization here)
    gold = {"a": {"spans": {(0, 2, "person"), (3, 4, "person"), (5, 7, "person"), (8, 10, "location")}, "unscored": []},
            "b": {"spans": {(0, 1, "person"), (2, 4, "person"), (5, 7, "organization")}, "unscored": []}}
    seen = {"唐"}
    found = {"a": {"spans": {(5, 7, "person"), (8, 10, "location"), (3, 4, "person")}}, "b": {"spans": {(5, 7, "location")}}}
    none = {"a": {"spans": set()}, "b": {"spans": {(2, 3, "person")}}}
    pa, pb = per_name_recall(gold, found, texts, seen), per_name_recall(gold, none, texts, seen)
    assert pa["names"] == {"found": 2, "total": 3, "share": 0.6667}  # 何普/person, 白塔/location of 3 (text, type) names
    assert pa["texts"] == {"found": 2, "total": 2, "share": 1.0}
    assert pa["names_by_type"]["organization"] == {"found": 0, "total": 1}
    assert pb["names"]["found"] == 0
    assert new_name_occurrences(gold, found, texts, seen) == (2, 1, 3)  # 2 of 5 new spans; 白塔 in b is the wrong type
    m = model_per_name(["x", "y"], {"x": pa, "y": pb})
    assert m["names"]["mean_share"] == 0.3333 and m["names"]["range_share"] == [0.0, 0.6667]
    assert m["texts"]["mean_share"] == 0.5 and m["names"]["found_by_run"] == [2, 0]
    # the per-occurrence model numbers are evaluate.new_name_view's
    occ = {"x": new_name_occurrences(gold, found, texts, seen), "y": new_name_occurrences(gold, none, texts, seen)}
    view, _ = new_name_view(gold, {"x": found, "y": none}, texts, {"x": seen, "y": seen}, {"a": "u", "b": "u"},
                            {"n": 20, "seed": 1}, {"m": ["x", "y"]})
    check_new_names_against_eval(occ, {"m": model_occurrence(["x", "y"], occ)}, {"new_name_recall": view})
    with pytest.raises(SystemExit):
        check_new_names_against_eval({"x": (1, 1, 4), "y": occ["y"]}, {}, {"new_name_recall": view})


def test_metrics_may_not_hold_book_text():
    check_counts_only({"systems": {"ws_S3_s1": {"fp": 3}}, "note": "post hoc"})
    with pytest.raises(SystemExit):
        check_counts_only({"systems": {"ws_S3_s1": {"唐": 3}}})


def test_outputs_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "REPO_ROOT", tmp_path)
    refuse_existing("r1")
    (tmp_path / "data" / "analysis" / "r1").mkdir(parents=True)
    (tmp_path / "data" / "analysis" / "r1" / "report.md").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        refuse_existing("r1")


# ------------------------------------------------------------------------------- a whole synthetic scoring run on disk

PARAGRAPHS = {"ch0781": ["唐医生到了汀城。", "白塔很高，玩家来了。", "何普没来。"],  # 唐0 汀城5-7 | 白塔9-11 玩家14-16 | 何普20-22
              "ch0782": ["灰塔和白塔。", "汀城下雨，唐和何普走了。"]}  # 灰塔0-2 白塔3-5 | 汀城7-9 下雨9-11 唐12 何普14-16


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world(tmp_path, monkeypatch):
    """A scoring run 'ev' with gliner, dictionary, one generative system and two seeds of a tagger; its metrics.json
    computed with evaluate's own functions."""
    root, code = tmp_path / "in", tmp_path / "code"
    chapters = {cid: Chapter(cid, int(cid[2:]), [(f"p{i:04d}", t) for i, t in enumerate(ps, 1)], "", "h")
                for cid, ps in PARAGRAPHS.items()}
    snap = Snapshot("snap", None, {}, chapters, [{"chapter_id": "ch0002", "name": "唐", "type": "person"}],
                    [{"name": "汀城", "type": "location", "source": "registry"}])
    monkeypatch.setattr(errors, "load_snapshot", lambda sid, root=None: snap)
    monkeypatch.setattr(errors, "check_gold", lambda export, s: ([], [], {}))
    monkeypatch.setattr(errors, "REPO_ROOT", code)
    ids = list(PARAGRAPHS)
    ann = lambda s, e, t, **k: {"start": s, "end": e, "type": t, **k}
    export = {"chapters": {
        "ch0781": {"status": "done", "annotations": [ann(0, 1, "person"), ann(5, 7, "location"), ann(9, 11, "location"),
                                                     ann(14, 16, "person", pending=True)]},
        "ch0782": {"status": "done", "annotations": [ann(0, 2, "location"), ann(3, 5, "location"), ann(7, 9, "location"),
                                                     ann(12, 13, "person"), ann(14, 16, "person")],
                   "unscored": [{"start": 9, "end": 11, "reason": "r", "text": "下雨"}]}},
        "llm_drafts": {"t1": {}}}
    write(root / "data" / "gold" / "test.json", export)
    gold = gold_spans(export, ids)
    write(root / "data" / "llm_tasks" / "t1" / "task.json", {"chapters": ids})
    m = lambda p, x, **k: {"paragraph_id": p, "text": x, "occurrence": 1, **k}
    answer = {"chapters": [
        {"chapter_id": "ch0781", "mentions": [m("p0001", "唐", type="person"), m("p0001", "汀城", type="location"),
                                              m("p0002", "白塔", pending=True)]},
        {"chapter_id": "ch0782", "mentions": [m("p0001", "灰塔和白塔", type="location"), m("p0002", "唐", type="person"),
                                              m("p0002", "何普", pending=True), m("p0002", "汀城", type="place")]}]}
    write(root / "data" / "llm_tasks" / "t1" / "pass_opus_r1.json", answer)
    write(root / "runs" / "c1" / "claude_calls.json",
          {"calls": [{"task": "t1", "label": "opus_r1", "answer_sha256": digest(root / "data/llm_tasks/t1/pass_opus_r1.json")}]})
    ev = root / "runs" / "ev"
    row = lambda cid, s, e, t, **k: {"chapter_id": cid, "start": s, "end": e, "text": snap.chapters[cid].text[s:e], "type": t, **k}
    gliner = [row("ch0781", 0, 3, "person", score=0.9), row("ch0781", 9, 11, "location", score=0.4),
              row("ch0782", 14, 16, "person", score=0.8), row("ch0782", 9, 11, "item", score=0.7),
              row("ch0781", 14, 16, "person", score=0.95)]
    gliner = [{**r, "kept": r["score"] > 0.5} for r in gliner]
    dictionary, _ = dictionary_predictions(snap.dictionary, [snap.chapters[c] for c in ids], 2)
    ws1 = [row("ch0781", 0, 1, "person"), row("ch0781", 9, 11, "organization"), row("ch0782", 0, 2, "location"),
           row("ch0782", 3, 5, "location"), row("ch0782", 14, 15, "person"), row("ch0782", 16, 17, "person")]
    ws2 = [row("ch0781", 9, 11, "location"), row("ch0781", 14, 16, "person"), row("ch0782", 12, 13, "person"),
           row("ch0782", 4, 5, "location"), row("ch0781", 20, 22, "person")]  # 何普 where the answer has nothing
    llm_preds, book = llm_predictions(answer, snap, ids)
    for name, rows in (("gliner_zero_shot", gliner), ("dictionary", dictionary), ("opus_r1", llm_preds), ("ws_a_s1", ws1),
                       ("ws_a_s2", ws2)):
        write(ev / f"predictions_test_{name}.jsonl", rows)
    systems = {"gliner_zero_shot": as_prediction(gliner, ids, 0.5), "dictionary": as_prediction(dictionary, ids),
               "opus_r1": as_prediction(llm_preds, ids), "ws_a_s1": as_prediction(ws1, ids), "ws_a_s2": as_prediction(ws2, ids)}
    for cid in ids:
        systems["opus_r1"][cid]["errors"] = book["invalid_by_chapter"].get(cid, 0)
        systems["opus_r1"][cid]["pending"] = [(a, b) for c, a, b in book["pending_places"] if c == cid]
    models = {"opus": ["opus_r1"], "ws_a": ["ws_a_s1", "ws_a_s2"]}
    eval_cfg = {"snapshot": "snap", "gold_test": "data/gold/test.json", "dictionary": {"min_len": 2},
                "llm_systems": [{"name": "opus_r1", "label": "opus_r1", "calls": "runs/c1/claude_calls.json"}],
                "span_systems": [{"name": "ws_a_s1"}, {"name": "ws_a_s2"}], "models": models, "new_names": {"cutoff_chapter": 780}}
    write(ev / "config.json", {"run_id": "ev", "config": eval_cfg, "snapshot": {"id": "snap"},
                               "gold_test": {"file": "data/gold/test.json", "sha256": digest(root / "data/gold/test.json"),
                                             "chapters": ids}})
    texts = {c: snap.chapters[c].text for c in ids}
    seen = {"汀城", "唐"}
    metrics = {"run_id": "ev", "per_chapter_counts": {}, "per_chapter_type_counts": {}, "scores": {}}
    for s, p in systems.items():
        part = eval_like_metrics(s, gold, p)
        for k in ("per_chapter_counts", "per_chapter_type_counts", "scores"):
            metrics[k].update(part[k])
    view, _ = new_name_view(gold, systems, texts, {s: seen for s in systems}, {c: "u" for c in ids}, {"n": 20, "seed": 1}, models)
    metrics["new_name_recall"] = {**view, "seen_sets": {"x": {"gold": novelty(gold, texts, seen)}}}
    metrics["test"] = {"unscored": unscored_report(export, gold, ids, snap, systems),
                       "scored_gold_spans": sum(1 for r in gold.values() for x in r["spans"] if not errors._touches(x, r["unscored"]))}
    metrics["systems"] = {
        "gliner_zero_shot": {"threshold": 0.5, "outputs": {"outputs": sum(r["kept"] for r in gliner)}},
        "dictionary": {},
        "opus_r1": {"outputs": {"located_predictions": len(llm_preds), "invalid_predictions": sum(book["invalid_by_chapter"].values()),
                                "abstentions_pending": sum(book["abstained_by_chapter"].values())},
                    "answers_match_logged_sha256": {"checked": 1, "no_hash_logged": 0}},
        "ws_a_s1": {"predictions_sha256": digest(ev / "predictions_test_ws_a_s1.jsonl")},
        "ws_a_s2": {"predictions_sha256": digest(ev / "predictions_test_ws_a_s2.jsonl")}}
    write(ev / "metrics.json", metrics)
    guide = "### A1 泛称和代词\n\n**选择：** 不标。“" + "、".join(A1) + "”作为身份描述时不标。\n\n### A2 x\n"
    (code / "docs").mkdir(parents=True)
    (code / "docs" / "guide.md").write_text(guide, encoding="utf-8")
    write(code / "configs" / "weak.json", {"title_suffixes": TITLES, "category_words": ["车票"], "grade_category_pattern": GRADE})
    cfg = {"eval_run": "runs/ev", "new_names": {"cutoff_chapter": 780}, "guide": "docs/guide.md", "generic_words": A1,
           "rule_words_from": "configs/weak.json",
           "examples": {"top_n": 15, "context_chars": 6, "groups": {"ws_a": ["ws_a_s1", "ws_a_s2"], "opus_r1": ["opus_r1"]}},
           "elsewhere_section": {"label": "ws_a", "systems": ["ws_a_s1", "ws_a_s2"]}}
    write(code / "configs" / "errors_t.json", cfg)
    return SimpleNamespace(root=root, code=code, ev=ev, cfg=cfg, metrics=metrics)


def test_a_whole_synthetic_run_reproduces_evaluate_and_writes_counts_only(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    a = analyse(w.cfg, w.root)
    c = a["counts"]
    assert c["gliner_zero_shot"]["fp"]["boundary"]["sub"]["wider_title"]["total"] == 1   # 唐医生
    assert c["gliner_zero_shot"]["removed_by_unscored_ranges"] == 2                       # 下雨, and 玩家 under the gold pending mark
    assert c["opus_r1"]["fp"]["boundary"]["sub"]["merged"]["total"] == 1                  # 灰塔和白塔
    assert c["opus_r1"]["fp"]["invalid_no_position"] == 1                                 # type "place"
    assert c["opus_r1"]["fn"]["abstained"]["total"] == 2 and c["opus_r1"]["pending"]["pending_marks"] == 2
    assert c["ws_a_s1"]["fp"]["type_error"]["total"] == 1                                 # 白塔 as an organization
    assert c["ws_a_s1"]["fp"]["spurious"]["sub"]["single_char"]["total"] == 1             # 走
    assert c["ws_a_s2"]["fp"]["boundary"]["sub"]["narrower"]["total"] == 1                # 塔 inside 白塔
    assert c["ws_a_s2"]["fp"]["spurious"]["sub"]["gold_name_elsewhere"]["by_type"]["person"] == 1  # 何普 in ch0781
    assert a["gold_side"]["unscored_ranges"] == 1 and a["gold_side"]["gold_pending_marks"] == 1
    assert a["models"]["ws_a"]["summed"]["tp"]["total"] == c["ws_a_s1"]["tp"]["total"] + c["ws_a_s2"]["tp"]["total"]
    assert a["per_name"]["ws_a_s1"]["names"] == {"found": 2, "total": 3, "share": 0.6667}  # 灰塔, 白塔 of 灰塔/白塔/何普
    metrics = build_metrics("r1", w.cfg, a)
    assert "白塔" not in json.dumps(metrics, ensure_ascii=False)
    # only the exact integer sums of a model are published (rounded means need not add up), and they are checked
    assert set(metrics["models"]["ws_a"]) == {"runs", "n_runs", "summed", "mean_per_run_note"}
    check_sums(metrics["models"]["ws_a"]["summed"])
    report = build_report("r1", a["ctx"], a["results"], c, a["per_name"], a["occ"], a["models"], a["gold_side"], w.cfg)
    assert "## 六、小模型（ws_a）误标、但同一字串在别处是答案的名字" in report and "【灰塔和白塔】" in report
    assert "| 「何普」 | 1 | 1/2 | 1（person 1） | ch0781：" in report
    # section 六 is no review list for the answers
    assert "不能拿它决定复查哪些答案" in report and "要不要复查由用户决定" not in report
    assert "看过 v9 分数之后定的" in report and "可能和合计差 0.01" in report
    # an example pooled over runs lists every distinct overlap with its count, not the first occurrence's only
    ws_a = report.split("### ws_a（2 次运行合起来）", 1)[1].split("\n### ", 1)[0]
    assert "「白塔」 location ×2，2/2 个运行｜压在上面的预测：「塔」location ×1；「白塔」organization ×1" in ws_a
    assert build_config("r1", "configs/errors_t.json", w.cfg, a, w.root)["inputs_sha256"]["data/gold/test.json"]
    assert build_config("r1", "configs/errors_t.json", w.cfg, a, w.root, "T0", {"commit": "c"})["started_at"] == "T0"
    assert "timing" in errors.RULES and "system_pending_places" in errors.RULES


def test_main_runs_for_real_only_from_the_checkout_holding_the_inputs(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    cfg_file = str(w.code / "configs" / "errors_t.json")
    # from another checkout: a dry run only; a real run would leave report.md in this checkout's ignored data/
    errors.main([cfg_file, "--run-id", "r0", "--inputs-root", str(w.root), "--dry-run"])
    with pytest.raises(SystemExit, match="dry-run only"):
        errors.main([cfg_file, "--run-id", "r0", "--inputs-root", str(w.root)])
    assert not (w.code / "runs").exists() and not (w.code / "data").exists()
    # from the checkout that holds data/: outputs next to the inputs; the start time is taken before the analysis
    for d in ("docs", "configs"):
        shutil.copytree(w.code / d, w.root / d)
    monkeypatch.setattr(errors, "REPO_ROOT", w.root)
    ticks = iter(range(1000))
    monkeypatch.setattr(errors, "time", SimpleNamespace(strftime=lambda fmt: f"t{next(ticks):03d}"))
    errors.main([cfg_file, "--run-id", "r1"])
    config = json.loads((w.root / "runs" / "r1" / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((w.root / "runs" / "r1" / "metrics.json").read_text(encoding="utf-8"))
    assert (w.root / "data" / "analysis" / "r1" / "report.md").is_file() and config["inputs_root"] == "."
    assert config["started_at"] < metrics["finished_at"]
    with pytest.raises(SystemExit):  # never overwritten
        errors.main([cfg_file, "--run-id", "r1"])
    errors.main([cfg_file, "--run-id", "r2", "--dry-run"])
    assert not (w.root / "runs" / "r2").exists()


@pytest.mark.parametrize("tamper",["span_file", "dictionary_file", "llm_answer", "gliner_kept", "eval_total", "generic_words"])
def test_inputs_that_do_not_match_the_scoring_run_stop_it(tmp_path, monkeypatch, tamper):
    w = world(tmp_path, monkeypatch)
    rows = lambda name: [json.loads(x) for x in (w.ev / f"predictions_test_{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    if tamper == "span_file":
        write(w.ev / "predictions_test_ws_a_s1.jsonl", rows("ws_a_s1")[:-1])
    elif tamper == "dictionary_file":
        write(w.ev / "predictions_test_dictionary.jsonl", rows("dictionary")[:-1])
    elif tamper == "llm_answer":
        f = w.root / "data" / "llm_tasks" / "t1" / "pass_opus_r1.json"
        write(f, {**json.loads(f.read_text(encoding="utf-8")), "note": "edited"})
    elif tamper == "gliner_kept":
        write(w.ev / "predictions_test_gliner_zero_shot.jsonl", [{**r, "kept": True} for r in rows("gliner_zero_shot")])
    elif tamper == "eval_total":
        w.metrics["per_chapter_counts"]["ws_a_s2"]["ch0781"][1] += 1
        write(w.ev / "metrics.json", w.metrics)
    elif tamper == "generic_words":
        w.cfg["generic_words"] = A1 + ["她"]
    with pytest.raises(SystemExit):
        analyse(w.cfg, w.root)
