import pytest

from novel_ner.evaluate import (as_prediction, bootstrap, brief, llm_predictions, output_record, round_stats, unscored_report, chapter_chunks, dictionary_predictions, gliner_predictions,
                                paragraph_bases, restrict, summarize, type_counts)
from novel_ner.score import strict
from novel_ner.snapshot import Chapter

LABELS = {t: t for t in ("person", "organization", "location", "dungeon", "item", "trait")}


def chapter(cid, number, paragraphs):
    return Chapter(cid, number, paragraphs, "", "h")


class FakeGLiNER:
    """Returns fixed entities per chunk text, in chunk offsets, like ChineseGLiNER.predict."""

    def __init__(self, answers):
        self.answers = answers
        self.tokenizer = lambda text: {"input_ids": [1] * len(text)}
        self.tokenizer.unk_token_id = 0

    def configure(self, *args):
        pass

    def token_count(self, text, labels):
        return len(text)

    def predict(self, texts, labels, threshold, batch_size):
        return [[dict(e) for e in self.answers.get(t, [])] for t in texts]


def test_chunks_are_slices_of_the_chapter_and_outputs_get_chapter_offsets():
    ch = chapter("ch0001", 1, [("p0001", "徐获拉开门。"), ("p0002", "唐医生到了汀城。")])
    assert paragraph_bases(ch) == {"p0001": 0, "p0002": 7}
    assert chapter_chunks(ch, 200) == [(0, ch.text)]
    assert chapter_chunks(ch, 8) == [(0, "徐获拉开门。"), (7, "唐医生到了汀城。")]
    # one output inside a paragraph, one running across the paragraph break: both kept, in chapter offsets
    model = FakeGLiNER({ch.text: [{"start": 12, "end": 14, "text": "汀城", "label": "location", "score": 0.95},
                                  {"start": 4, "end": 9, "text": "门。\n唐医", "label": "person", "score": 0.97}]})
    preds, stats = gliner_predictions(model, {"splitter": "char", "space_mode": "per_word"}, LABELS, [ch],
                                      {"chunk_max_chars": 200, "save_threshold": 0.1, "batch_size": 4})
    assert [(p["start"], p["end"], p["type"]) for p in preds] == [(12, 14, "location"), (4, 9, "person")]
    assert stats["outputs_crossing_a_paragraph_break"] == 1
    bad = FakeGLiNER({ch.text: [{"start": 11, "end": 13, "text": "汀城", "label": "location", "score": 0.95}]})
    with pytest.raises(ValueError):
        gliner_predictions(bad, {"splitter": "char", "space_mode": "per_word"}, LABELS, [ch],
                           {"chunk_max_chars": 200, "save_threshold": 0.1, "batch_size": 4})


def test_threshold_keeps_scores_above_it_like_gliner():
    preds = [{"chapter_id": "a", "start": 0, "end": 2, "type": "person", "score": 0.9},
             {"chapter_id": "a", "start": 3, "end": 5, "type": "item", "score": 0.89996},
             {"chapter_id": "a", "start": 6, "end": 8, "type": "item", "score": 0.95}]
    assert as_prediction(preds, ["a"], 0.9)["a"]["spans"] == {(6, 8, "item")}
    assert len(as_prediction(preds, ["a"])["a"]["spans"]) == 3


def test_dictionary_takes_the_longest_name_and_respects_scope_known_and_conflicts():
    rows = [
        {"name": "E27", "type": "location"}, {"name": "E27-014", "type": "location"},
        {"name": "何普", "type": "person", "scope": [1, 5], "known": 1},
        {"name": "周凝", "type": "person", "known": 9},
        {"name": "白塔", "type": "location"}, {"name": "白塔", "type": "organization"},
        {"name": "唐", "type": "person"},
    ]
    ch = chapter("ch0010", 10, [("p0001", "在E27-014和E27，何普、周凝、唐去了白塔。")])
    preds, conflicts = dictionary_predictions(rows, [ch])
    assert [(p["text"], p["type"]) for p in preds] == [("E27-014", "location"), ("E27", "location"), ("周凝", "person")]
    assert conflicts == ["白塔"]
    early = chapter("ch0003", 3, [("p0001", "何普和周凝。")])
    assert [p["text"] for p in dictionary_predictions(rows, [early])[0]] == ["何普"]


GOLD = {"a": {"spans": {(0, 2, "person"), (5, 7, "item")}, "unscored": [(10, 12)]},
        "b": {"spans": {(0, 3, "location")}, "unscored": []}}
PRED = {"a": {"spans": {(0, 2, "person"), (5, 7, "person"), (10, 11, "item")}},
        "b": {"spans": {(0, 2, "location")}}}


def test_type_counts_add_up_to_strict_and_unscored_ranges_count_for_nothing():
    counts = type_counts(GOLD, PRED)
    total = strict(GOLD, PRED)
    pooled = tuple(sum(c[t][i] for c in counts.values() for t in c) for i in range(3))
    assert pooled == (total["tp"], total["fp"], total["fn"]) == (1, 2, 2)
    assert counts["a"]["item"] == [0, 0, 1] and counts["a"]["person"] == [1, 1, 0]


def test_summarize_marks_types_without_samples_and_undefined_precision():
    m = summarize(list(type_counts(GOLD, PRED).values()))
    assert m["per_type"]["trait"]["note"] == "无可评样本" and "f1" not in m["per_type"]["trait"]
    assert m["per_type"]["item"]["precision"] is None and m["per_type"]["item"]["recall"] == 0.0
    assert m["macro_over"] == ["person", "location", "item"]
    assert m["macro_f1"] == round((m["per_type"]["person"]["f1"] + 0 + 0) / 3, 4)
    assert m["non_person_micro"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_restrict_applies_the_same_rule_to_gold_and_predictions():
    g, p = restrict(GOLD, PRED, lambda cid, s: s[2] != "person")
    assert g["a"]["spans"] == {(5, 7, "item")} and p["a"]["spans"] == {(10, 11, "item")}
    assert g["a"]["unscored"] == [(10, 12)]


def test_bootstrap_is_paired_reproducible_and_respects_the_plot_units():
    counts = {"x": {f"c{i}": {"person": [8 + i % 3, 2, 1 + i % 2], "item": [1, 1, 1]} for i in range(20)},
              "y": {f"c{i}": {"person": [5, 4 + i % 2, 4], "item": [0, 0, 2]} for i in range(20)}}
    for sys_counts in counts.values():
        for c in sys_counts.values():
            for t in ("organization", "location", "dungeon", "trait"):
                c[t] = [0, 0, 0]
    strata = {f"c{i}": "u1" if i < 15 else "u2" for i in range(20)}
    one, two = bootstrap(counts, strata, 300, 7), bootstrap(counts, strata, 300, 7)
    assert one == two and one["resampling"]["plot_units"] == {"u1": 15, "u2": 5}
    lo, hi = one["systems"]["x"]["micro_f1"]
    point = summarize(list(counts["x"].values()))["micro"]["f1"]
    assert lo <= point <= hi
    d = one["paired_difference"]["x - y"]["micro_f1"]
    assert d["ci95"][0] > 0 and d["share_of_rounds_above_0"] == 1.0
    units = bootstrap(counts, strata, 300, 7, mode="units")
    assert units["resampling"]["unit"] == "plot unit"


def test_brief_leaves_undefined_scores_undefined_and_empty_groups_say_so():
    m = brief(strict({"a": {"spans": {(0, 2, "person")}, "unscored": []}}, {"a": {"spans": set()}}))
    assert m == {"tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0.0, "f1": 0.0}
    empty = brief(strict({"a": {"spans": set(), "unscored": []}}, {"a": {"spans": set()}}))
    assert empty == {"tp": 0, "fp": 0, "fn": 0, "note": "无可评样本"}


def test_macro_over_a_fixed_type_list_counts_an_absent_type_as_zero():
    rows = [{t: [0, 0, 0] for t in LABELS} for _ in range(2)]
    rows[0]["person"] = [4, 0, 0]
    rows[1]["dungeon"] = [0, 1, 1]
    assert summarize(rows)["macro_f1"] == 0.5  # person 1.0, dungeon 0.0
    assert summarize(rows[:1])["macro_f1"] == 1.0  # dungeon absent: types with support only
    assert summarize(rows[:1], ["person", "dungeon"])["macro_f1"] == 0.5


def test_bootstrap_keeps_the_macro_type_list_fixed_and_counts_rounds_without_one():
    counts = {s: {f"c{i}": {t: [0, 0, 0] for t in LABELS} for i in range(10)} for s in ("x", "y")}
    for s in counts:
        for i in range(10):
            counts[s][f"c{i}"]["person"] = [3, 1, 1]
    counts["x"]["c0"]["dungeon"] = [0, 2, 1]  # the gold is the same for both systems: one dungeon span in c0
    counts["y"]["c0"]["dungeon"] = [0, 0, 1]
    out = bootstrap(counts, {f"c{i}": "u" for i in range(10)}, 200, 3, macro_types=["person", "dungeon"])
    assert out["resampling"]["macro_types"] == ["person", "dungeon"]
    assert 0 < out["resampling"]["rounds_without_gold_of_a_macro_type"] < 200
    assert out["systems"]["x"]["macro_f1"][1] <= 0.375  # person F1 0.75 averaged with dungeon 0, every round


def test_output_record_counts_duplicates_and_conflicting_types_at_one_place():
    preds = [{"chapter_id": "a", "start": 0, "end": 2, "type": "person", "score": 0.95},
             {"chapter_id": "a", "start": 0, "end": 2, "type": "person", "score": 0.95},
             {"chapter_id": "a", "start": 0, "end": 2, "type": "item", "score": 0.95},
             {"chapter_id": "a", "start": 3, "end": 5, "type": "item", "score": 0.5}]
    r = output_record(preds, 0.9)
    assert (r["outputs"], r["exact_duplicates_removed"], r["places_with_conflicting_types"]) == (3, 1, 1)
    assert output_record(preds)["outputs"] == 4


def test_unscored_coverage_counts_overlapping_ranges_once():
    ch = chapter("a", 1, [("p0001", "一二三四五六七八九十")])

    class Snap:
        chapters = {"a": ch}
    export = {"chapters": {"a": {
        "unscored": [{"start": 0, "end": 4, "text": "一二三四", "reason": "r"}],
        "annotations": [{"start": 2, "end": 6, "type": None, "pending": True},
                        {"start": 7, "end": 9, "type": "person", "pending": False}]}}}
    gold = {"a": {"spans": {(7, 9, "person")}, "unscored": [(0, 4), (2, 6)]}}
    r = unscored_report(export, gold, ["a"], Snap(), {"x": {"a": {"spans": {(3, 5, "item"), (7, 9, "person")}}}})
    assert r["characters_covered"] == 6 and r["share_of_test_text"] == 0.6
    assert r["recorded_ranges"] == 1 and r["pending_marks"] == 1 and r["predicted_spans_removed"] == {"x": 1}


def test_generative_answers_follow_appendix_1():
    from novel_ner.snapshot import Snapshot
    ch = chapter("ch0001", 1, [("p0001", "徐获拉开门。"), ("p0002", "唐医生到了汀城，汀城下雨。")])
    snap = Snapshot("s", None, {}, {"ch0001": ch}, [], [])
    mention = lambda **kw: {"paragraph_id": "p0002", "text": "汀城", "occurrence": 1, "type": "location", **kw}
    answer = {"chapters": [{"chapter_id": "ch0001", "mentions": [
        mention(), mention(),  # an exact repeat
        mention(type="organization"),  # the same place, another type: kept
        mention(occurrence=2),
        mention(text="唐", type=None, pending=True),  # abstention
        mention(text="唐", type=None),  # no type and not pending: invalid
        mention(occurrence=True),  # a bool is not an occurrence number: invalid
        mention(text="灰塔"),  # cannot be located: invalid
        mention(type="weapon"),  # not one of the six: invalid
        {"paragraph_id": "p0009", "text": "徐获", "occurrence": 1, "type": "person"},  # no such paragraph: invalid
    ]}, {"chapter_id": "ch0777", "mentions": [mention()]}]}
    preds, book = llm_predictions(answer, snap, ["ch0001", "ch0002"])
    pred = as_prediction(preds, ["ch0001"])
    assert pred["ch0001"]["spans"] == {(12, 14, "location"), (12, 14, "organization"), (15, 17, "location")}
    assert book["invalid_by_chapter"] == {"ch0001": 6} and book["abstained_by_chapter"] == {"ch0001": 1}
    assert book["pending_places"] == [("ch0001", 7, 8)] and book["chapters_missing"] == ["ch0002"]
    assert book["problems"] == {"chapter_not_in_this_task": 1, "mentions_under_a_chapter_not_in_this_task": 1,
                                "could_not_be_located": 3, "type_not_one_of_the_six": 1, "type_missing": 1}
    gold = {"ch0001": {"spans": {(12, 14, "location"), (7, 8, "person")}, "unscored": []}}
    pred["ch0001"]["errors"] = 3
    counts = type_counts(gold, pred)
    m = summarize(list(counts.values()))
    assert m["micro"] == {"precision": round(1 / 6, 4), "recall": 0.5, "f1": 0.25}  # tp 1, fp 2 + 3 invalid, fn 1: 2/(2+5+1)
    total = strict(gold, pred)
    assert (total["tp"], total["fp"], total["fn"]) == (1, 5, 1)


def test_round_stats_equal_the_rounded_point_estimates_and_restrict_can_keep_invalid():
    rows = list(type_counts(GOLD, PRED).values())
    raw, m = round_stats(rows, None), summarize(rows)
    assert round(raw["micro_f1"], 4) == m["micro"]["f1"] and round(raw["macro_f1"], 4) == m["macro_f1"]
    pred = {c: {**v, "errors": 2 if c == "a" else 0} for c, v in PRED.items()}
    g, p = restrict(GOLD, pred, lambda c, s: True, carry_invalid=True)
    assert strict(g, p)["fp"] == strict(GOLD, pred)["fp"]
    g, p = restrict(GOLD, pred, lambda c, s: True)
    assert strict(g, p)["fp"] == strict(GOLD, pred)["fp"] - 2


def test_pending_rules_for_odd_answers():
    from novel_ner.snapshot import Snapshot
    ch = chapter("ch0001", 1, [("p0001", "汀城下雨。")])
    snap = Snapshot("s", None, {}, {"ch0001": ch}, [], [])
    m = {"paragraph_id": "p0001", "text": "汀城", "occurrence": 1}
    answer = {"chapters": [
        {"chapter_id": "ch0001", "mentions": [dict(m, type="location", pending="true")]},
        {"chapter_id": "ch0001", "mentions": [dict(m, type=None, pending=True)]},  # answered twice
        {"chapter_id": "ch0009", "mentions": [dict(m, type=None, pending=True), dict(m, type="location")]},
        {"chapter_id": "ch0010", "mentions": "none"}]}
    preds, book = llm_predictions(answer, snap, ["ch0001"])
    assert [(p["start"], p["type"]) for p in preds] == [(0, "location")]
    assert book["abstained_by_chapter"] == {"ch0001": 1} and book["invalid_by_chapter"] == {"ch0001": 1}
    assert book["problems"] == {"pending_not_true_or_false": 1, "chapter_answered_twice": 1, "chapter_not_in_this_task": 2,
                                "mentions_under_a_chapter_not_in_this_task": 1, "chapter_without_a_mentions_list": 1}


def test_run_agreement_scores_one_run_against_the_other():
    from novel_ner.evaluate import run_agreement
    gold = {"a": {"spans": {(0, 2, "person"), (3, 5, "item")}, "unscored": [(8, 9)]}}
    systems = {"r1": {"a": {"spans": {(0, 2, "person"), (3, 5, "item"), (8, 9, "item")}, "pending": [(6, 7)]}},
               "r2": {"a": {"spans": {(0, 2, "person")}, "pending": [(6, 7), (10, 11)]}}}
    gold["a"]["unscored"] = [(8, 9), (10, 11)]
    out = run_agreement([("r1", "r2")], systems, gold, ["a"])["r1 vs r2"]
    assert (out["output_agreement"]["tp"], out["output_agreement"]["fn"], out["output_agreement"]["fp"]) == (1, 1, 0)
    assert out["abstention_places"] == {"r1": 1, "r2": 1, "same_place_in_both": 1}  # (10, 11) is unscored
    assert out["scores"]["r1"]["micro_f1"] == 1.0 and out["second_minus_first"]["micro_f1"] == round(2 / 3 - 1, 4)


def test_model_summary_pools_runs_per_chapter():
    from novel_ner.evaluate import model_summary
    row = lambda tp, fp, fn: {**{t: [0, 0, 0] for t in LABELS}, "person": [tp, fp, fn], "_invalid": [0, 0, 0]}
    counts = {"m_r1": {"c1": row(8, 2, 0), "c2": row(5, 0, 5)}, "m_r2": {"c1": row(10, 0, 0), "c2": row(4, 0, 6)},
              "n_r1": {"c1": row(6, 0, 4), "c2": row(6, 0, 4)}}
    out = model_summary({"m": ["m_r1", "m_r2"], "n": ["n_r1"]}, counts, {"c1": "u", "c2": "u"}, {"n": 50, "seed": 1})
    m = out["models"]["m"]
    assert m["n_runs"] == 2 and m["range"]["micro_f1"][0] <= m["mean"]["micro_f1"] <= m["range"]["micro_f1"][1]
    assert m["pooled_counts_micro_f1"] == round(2 * 27 / (2 * 27 + 2 + 11), 4)
    assert "m - n" in out["pooled_paired"]
