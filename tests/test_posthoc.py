"""posthoc.py: each analysis on tiny synthetic data, the two-level bootstrap's chapter-only path against evaluate's own
pooled intervals, and a whole synthetic run on disk. Synthetic data only."""
import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from novel_ner import evaluate, posthoc
from novel_ner.evaluate import as_prediction, dictionary_predictions, llm_predictions, model_summary, new_name_view, summarize, type_counts
from novel_ner.novelty import novelty
from novel_ner.posthoc import (analyse, analysis_b, analysis_c, analysis_d, chapter_vectors, check_replication, handling, hits,
                               new_gold, new_name_chapter_counts, split_title, step_record, two_level_bootstrap)
from novel_ner.score import _touches, gold_spans
from novel_ner.snapshot import Chapter, Snapshot
from novel_ner.weak import ids_sha256

TYPES = ("person", "organization", "location", "dungeon", "item", "trait")
BANDS = [{"label": "3_or_more", "min": 3, "max": None}, {"label": "2", "min": 2, "max": 2}, {"label": "1", "min": 1, "max": 1}]


# ------------------------------------------------------------------------------- A: the two-level bootstrap

def random_world(seed, chapters=8, runs=None):
    """Random texts, gold (with an unscored range) and per-run predictions (with invalid outputs) over two plot units."""
    rng = random.Random(seed)
    chars = "唐何普白塔汀城灰医生到了下雨走和门风"
    texts, gold = {}, {}
    for i in range(chapters):
        cid = f"c{i}"
        texts[cid] = "".join(rng.choice(chars) for _ in range(60))
        spans, at = set(), 0
        while at < 55:
            at += rng.randint(1, 5)
            n = rng.randint(1, 3)
            spans.add((at, at + n, rng.choice(TYPES[:5])))
            at += n
        gold[cid] = {"spans": spans, "unscored": [(20, 23)] if i == 2 else []}
    systems = {}
    for r in runs or ["a1", "a2", "a3", "b1", "b2", "c1"]:
        systems[r] = {}
        for cid, ref in gold.items():
            keep = rng.uniform(0.3, 0.9)
            guesses = {s for s in ref["spans"] if rng.random() < keep}
            for _ in range(rng.randint(0, 6)):
                a = rng.randint(0, 57)
                guesses.add((a, a + rng.randint(1, 3), rng.choice(TYPES)))
            systems[r][cid] = {"spans": guesses, "errors": rng.randint(0, 1)}
    strata = {cid: "u1" if i < chapters // 2 else "u2" for i, cid in enumerate(gold)}
    return texts, gold, systems, strata


def vectors_of(gold, systems, texts, seen):
    return {s: chapter_vectors(type_counts(gold, p), new_name_chapter_counts(gold, p, texts, seen)) for s, p in systems.items()}


@pytest.mark.parametrize("seed", [0, 1])
def test_the_chapter_only_level_reproduces_evaluates_pooled_intervals_in_either_direction(seed):
    texts, gold, systems, strata = random_world(seed)
    seen = {"唐", "白塔", "何"}
    models = {"m1": ["a1", "a2", "a3"], "m2": ["b1", "b2"], "m3": ["c1"]}
    b = {"n": 300, "seed": 11}
    counts = {s: type_counts(gold, p) for s, p in systems.items()}
    stored = model_summary(models, counts, strata, b)["pooled_paired"]  # every pair, in models order: m1 - m2, m1 - m3, m2 - m3
    view, _ = new_name_view(gold, systems, texts, {s: seen for s in systems}, strata, b, models)
    pairs = [("m2", "m1"), ("m1", "m3"), ("m3", "m2")]  # two of them stored the other way round
    boot = two_level_bootstrap(vectors_of(gold, systems, texts, seen), models, pairs, strata, b["n"], b["seed"], 99)
    got = check_replication(boot["diffs"]["chapter_only"], pairs, stored, view["pooled_paired_difference_recall"])
    assert {p: {k: v[k] for k in ("stored_key", "stored_direction")} for p, v in got.items()} == {
        ("m2", "m1"): {"stored_key": "m1 - m2", "stored_direction": "reversed"},
        ("m1", "m3"): {"stored_key": "m1 - m3", "stored_direction": "same"},
        ("m3", "m2"): {"stored_key": "m2 - m3", "stored_direction": "reversed"}}
    assert "chapter_only_against_the_stored_interval_negated" not in got[("m1", "m3")]
    # a reversed pair: the interval written for later minus earlier sits at positions int(0.025 (n-1)), int(0.975 (n-1))
    # of its own rounds; the stored one negated sits at n-1 minus those positions; the record counts where they differ
    n = b["n"]
    lo, hi = int(0.025 * (n - 1)), int(0.975 * (n - 1))
    for pair, key in ((("m2", "m1"), "m1 - m2"), (("m3", "m2"), "m2 - m3")):
        ends, shares, largest = 0, 0, 0.0
        for k in (*posthoc.STORED_MAIN, "new_name_recall"):
            d = sorted(boot["diffs"]["chapter_only"][pair][k])
            want = view["pooled_paired_difference_recall"][key] if k == "new_name_recall" else stored[key][k]
            assert [round(-want["ci95"][1], 4), round(-want["ci95"][0], 4)] == [round(d[n - 1 - hi], 4), round(d[n - 1 - lo], 4)]
            written = posthoc.interval(boot["diffs"]["chapter_only"][pair][k])
            assert written["ci95"] == [round(d[lo], 4), round(d[hi], 4)]
            gaps = [abs(round(d[lo], 4) - round(d[n - 1 - hi], 4)), abs(round(d[hi], 4) - round(d[n - 1 - lo], 4))]
            ends += sum(round(g, 4) > 0 for g in gaps)
            largest = max(largest, *(round(g, 4) for g in gaps))
            ties = sum(x == 0 for x in d) / n  # rounds tied at 0 are above 0 in neither direction
            assert abs(written["share_of_rounds_above_0"] - (1 - want["share_of_rounds_above_0"] - ties)) < 1e-3
            shares += written["share_of_rounds_above_0"] != round(1 - want["share_of_rounds_above_0"], 4)
        assert got[pair]["chapter_only_against_the_stored_interval_negated"] == {
            "statistics": 5, "ci95_ends_differing": ends, "largest_ci95_end_difference": largest,
            "shares_above_0_differing_from_1_minus_the_stored": shares}
    # the point difference is the pooled one (all runs' counts added per chapter)
    pooled_f1 = lambda m: summarize(list(evaluate.pooled_counts(models[m], counts).values()))["micro"]["f1"]
    assert round(boot["models"]["m1"]["micro_f1"], 4) == pooled_f1("m1")
    assert round(boot["point"][("m2", "m1")]["micro_f1"], 4) == round(boot["models"]["m2"]["micro_f1"] - boot["models"]["m1"]["micro_f1"], 4)
    # a stored interval that differs by one step in the last place stops the run
    broken = json.loads(json.dumps(stored))
    broken["m1 - m3"]["micro_recall"]["ci95"][0] += 0.0001
    with pytest.raises(SystemExit, match="does not reproduce"):
        check_replication(boot["diffs"]["chapter_only"], pairs, broken, view["pooled_paired_difference_recall"])
    with pytest.raises(SystemExit, match="no pooled interval"):
        check_replication(boot["diffs"]["chapter_only"], [("m1", "zz")], stored, view["pooled_paired_difference_recall"])


def test_run_draws_change_nothing_for_identical_runs_and_widen_when_only_the_runs_differ():
    strata = {f"c{i}": "u1" if i < 3 else "u2" for i in range(6)}
    base = {c: (10 + i, 2 + i % 2, 3, 4, 1, i % 3, 5, 2 + i % 2) for i, c in enumerate(strata)}
    same = {"x1": base, "x2": dict(base), "x3": dict(base), "y1": {c: (8, 1, 5, 2, 2, 1, 3, 4) for c in strata}}
    models = {"x": ["x1", "x2", "x3"], "y": ["y1"]}
    out = two_level_bootstrap(same, models, [("x", "y")], strata, 200, 5, 6)
    assert out["diffs"]["chapter_only"] == out["diffs"]["chapters_and_runs"]  # identical runs: the run draw changes nothing
    assert out["run_draws"] == {"x": True, "y": False}
    # every chapter alike, the runs very different: chapter sampling alone gives a zero-width interval, run draws widen it
    flat = {"x1": {c: (10, 2, 3, 4, 1, 1, 5, 2) for c in strata}, "x2": {c: (2, 9, 11, 0, 4, 4, 1, 6) for c in strata},
            "x3": {c: (6, 5, 7, 2, 2, 2, 3, 4) for c in strata}, "y1": {c: (8, 1, 5, 2, 2, 1, 3, 4) for c in strata}}
    out = two_level_bootstrap(flat, models, [("x", "y")], strata, 200, 5, 6)
    for k in posthoc.STATS:
        lo, hi = posthoc.interval(out["diffs"]["chapter_only"][("x", "y")][k])["ci95"]
        rlo, rhi = posthoc.interval(out["diffs"]["chapters_and_runs"][("x", "y")][k])["ci95"]
        assert hi - lo == 0 and rhi - rlo > 0
    # the chapters drawn do not depend on the run generator's seed
    other = two_level_bootstrap(flat, models, [("x", "y")], strata, 200, 5, 7)
    assert other["diffs"]["chapter_only"] == out["diffs"]["chapter_only"]
    assert other["diffs"]["chapters_and_runs"] != out["diffs"]["chapters_and_runs"]


def test_each_round_draws_k_runs_per_model_and_sums_all_of_them():
    """x has 3 runs, alike on every chapter: x1 only true positives, x2 and x3 only false positives; y is all zeros. So in
    a round x's precision is (times x1 was drawn) / 3: 0, 1/3, 2/3 or 1 with binomial(3, 1/3) shares. Drawing one run
    instead of k, or keeping only the first drawn run, gives only 0 and 1."""
    strata = {f"c{i}": "u1" if i < 2 else "u2" for i in range(4)}
    vec = {"x1": (1, 0, 0, 0, 0, 0, 0, 0), "x2": (0, 1, 0, 0, 0, 0, 0, 0), "x3": (0, 1, 0, 0, 0, 0, 0, 0), "y1": (0,) * 8}
    vectors = {r: {c: v for c in strata} for r, v in vec.items()}
    models = {"x": ["x1", "x2", "x3"], "y": ["y1"]}
    n = 3000
    out = two_level_bootstrap(vectors, models, [("x", "y")], strata, n, 5, 6)
    assert set(out["diffs"]["chapter_only"][("x", "y")]["micro_precision"]) == {1 / 3}
    got = Counter(round(x, 6) for x in out["diffs"]["chapters_and_runs"][("x", "y")]["micro_precision"])
    assert set(got) == {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}
    for j, p in enumerate((8 / 27, 12 / 27, 6 / 27, 1 / 27)):
        assert abs(got[round(j / 3, 6)] / n - p) < 0.03


def test_the_run_draws_follow_the_recorded_generator_order():
    """An independent re-implementation of the rule in RULES: one random.Random(run_seed), round by round, models in the
    order of `models` (single-run models skip it), k draws per model, one choice per draw, the drawn runs' counts summed."""
    texts, gold, systems, strata = random_world(3, runs=["a1", "a2", "a3", "b1", "b2", "c1"])
    seen = {"唐"}
    vectors = vectors_of(gold, systems, texts, seen)
    models = {"m3": ["c1"], "m2": ["b1", "b2"], "m1": ["a1", "a2", "a3"]}
    pairs = [("m1", "m2"), ("m2", "m3")]
    n, cseed, rseed = 80, 11, 12
    out = two_level_bootstrap(vectors, models, pairs, strata, n, cseed, rseed)
    rng = random.Random(rseed)
    want = {p: [] for p in pairs}
    for sample in posthoc.chapter_draws(strata, n, cseed):
        stats = {}
        for m, runs in models.items():
            drawn = runs if len(runs) == 1 else [rng.choice(runs) for _ in runs]
            total = [sum(vectors[r][c][i] for r in drawn for c in sample) for i in range(8)]
            stats[m] = posthoc.stats_of(total)
        for a, b_ in pairs:
            want[(a, b_)].append(stats[a]["micro_f1"] - stats[b_]["micro_f1"])
    for p in pairs:
        assert out["diffs"]["chapters_and_runs"][p]["micro_f1"] == pytest.approx(want[p], abs=1e-12)


def test_interval_side_changes_list_both_directions_with_the_point_sign():
    v = lambda c, r, p: {"point": p, "chapter_only": {"ci95": c}, "chapters_and_runs": {"ci95": r}}
    pairs_out = {"a - b": {"point": {}, "chapter_only": {}, "chapters_and_runs": {}}}
    rows = {"micro_f1": ([0.01, 0.03], [-0.001, 0.04], 0.02), "micro_precision": ([-0.06, -0.01], [-0.07, 0.0], -0.03),
            "micro_recall": ([0.1, 0.2], [0.09, 0.21], 0.15), "non_person_f1": ([-0.01, 0.02], [0.001, 0.02], 0.01),
            "new_name_recall": ([-0.1, 0.1], [-0.2, 0.2], 0.0)}
    for k, (c, r, p) in rows.items():
        x = v(c, r, p)
        pairs_out["a - b"]["point"][k] = x["point"]
        pairs_out["a - b"]["chapter_only"][k] = x["chapter_only"]
        pairs_out["a - b"]["chapters_and_runs"][k] = x["chapters_and_runs"]
    got = posthoc.side_changes(pairs_out)
    assert [(g["statistic"], g["point_sign"], g["chapter_only"], g["chapters_and_runs"]) for g in got] == [
        ("micro_f1", "positive", "above_0", "includes_0"),
        ("micro_precision", "negative", "below_0", "includes_0"),  # an upper end of exactly 0 includes 0
        ("non_person_f1", "positive", "includes_0", "above_0")]


def test_chapter_vectors_count_invalid_outputs_as_false_positives_and_the_new_names_apart():
    tc = {"c": {"person": [3, 1, 2], "organization": [1, 0, 1], "location": [0, 2, 0], "dungeon": [0, 0, 0], "item": [0, 0, 0],
                "trait": [0, 0, 0], "_invalid": [0, 4, 0]}}
    assert chapter_vectors(tc, {"c": [2, 5, 1]}) == {"c": (4, 7, 3, 1, 2, 1, 2, 1)}
    s = posthoc.stats_of((4, 7, 3, 1, 2, 1, 2, 1))
    assert s["micro_precision"] == 4 / 11 and s["micro_recall"] == 4 / 7 and s["new_name_recall"] == 2 / 3
    assert posthoc.stats_of((0,) * 8) == {k: 0.0 for k in posthoc.STATS}  # nothing there: 0, as round_stats counts it


# ------------------------------------------------------------------------------- B and C: new-name breakdowns

TEXTS_B = {"a": "唐先生见何普，何普和白塔。", "b": "唐先生说白塔高，钥匙在何普家。"}
# a: 唐0 先1 生2 见3 何4 普5 ，6 何7 普8 和9 白10 塔11 。12 / b: 唐0 先1 生2 说3 白4 塔5 高6 ，7 钥8 匙9 在10 何11 普12 家13
GOLD_B = {"a": {"spans": {(0, 1, "person"), (4, 6, "person"), (7, 9, "person"), (10, 12, "location")}, "unscored": []},
          "b": {"spans": {(0, 1, "person"), (4, 6, "location"), (8, 10, "item"), (11, 13, "person"), (6, 7, "trait")},
                "unscored": [(6, 8)]}}  # 高 is unscored: not a new gold span
PRED_B = {"x": {"a": {"spans": {(0, 1, "person"), (4, 6, "person"), (10, 12, "organization")}},
                "b": {"spans": {(8, 10, "item"), (11, 13, "person")}}},
          "y": {"a": {"spans": {(7, 9, "person")}}, "b": {"spans": {(4, 6, "location")}}}}
SEEN_B = frozenset({"唐僧", "汀城"})  # 唐 itself is unseen; a seen name starts with it


def test_new_name_breakdowns_by_type_band_single_characters_and_names():
    new = new_gold(GOLD_B, TEXTS_B, SEEN_B)
    assert len(new) == 8 and all(g[4] != "高" for g in new)
    out, found = analysis_b(new, TEXTS_B, PRED_B, {"m": ["x", "y"]}, BANDS, ["先生", "医生"], SEEN_B,
                            {"top_n": 1, "frequent_min_spans": 3})
    x, y = out["systems"]["x"], out["systems"]["y"]
    assert x["all"] == {"spans_found": 4, "recall": 0.5, "names_found": 3, "name_share": 0.75}
    assert x["person"] == {"spans_found": 3, "recall": 0.6, "names_found": 2, "name_share": 1.0}
    assert x["non_person"] == {"spans_found": 1, "recall": 0.3333, "names_found": 1, "name_share": 0.5}
    assert x["type_location"]["spans_found"] == 0 and x["type_item"]["recall"] == 1.0
    assert x["band_3_or_more"] == {"spans_found": 2, "recall": 0.6667, "names_found": 1, "name_share": 1.0}  # 何普 x3
    assert x["band_2"] == {"spans_found": 1, "recall": 0.25, "names_found": 1, "name_share": 0.5}  # 唐 x2, 白塔 x2
    assert x["band_1"]["recall"] == 1.0 and x["single_character"] == {"spans_found": 1, "recall": 0.5, "names_found": 1, "name_share": 1.0}
    assert y["all"] == {"spans_found": 2, "recall": 0.25, "names_found": 2, "name_share": 0.5}
    m = out["models"]["m"]["all"]
    assert (m["recall_mean"], m["recall_range"], m["name_share_mean"], m["name_share_range"]) == (0.375, [0.25, 0.5], 0.625, [0.5, 0.75])
    assert m["spans_found_by_run"] == [4, 2] and m["names_found_by_run"] == [3, 2]
    assert out["groups"]["band_2"] == {"spans": 4, "names": 2, "texts": 2}
    one = out["single_character_facts"]
    assert one["single_character"]["spans"] == 2 and one["followed_by_a_title_suffix"]["spans"] == 2
    assert one["first_character_starts_a_seen_name"]["spans"] == 2 and one["both"]["names"] == 1
    c = out["concentration"]
    assert (c["top1_spans"], c["top_n_spans"], c["frequent_names"], c["frequent_names_spans"], c["names_with_one_span"]) == (3, 3, 1, 3, 1)
    assert c["person"] == {"spans": 5, "share_of_spans": 0.625, "names": 2, "share_of_names": 0.5}
    assert out["spans_whose_band_by_text_type_would_differ"] == 0
    assert found["x"] == hits(new, PRED_B["x"])
    # the per-name totals agree with errors.per_name_recall, which the real run checks
    from novel_ner.errors import per_name_recall
    assert per_name_recall(GOLD_B, PRED_B["x"], TEXTS_B, SEEN_B)["names"]["found"] == x["all"]["names_found"]
    with pytest.raises(SystemExit, match="frequency bands"):
        analysis_b(new, TEXTS_B, PRED_B, {}, BANDS[:2], [], SEEN_B, {"top_n": 1, "frequent_min_spans": 3})


def test_single_character_facts_need_a_seen_name_that_begins_with_the_character():
    texts = {"a": "唐先生见普，普来了，小月。"}
    # 唐0 先1 生2 见3 普4 ，5 普6 来7 了8 ，9 小10 月11 。12
    gold = {"a": {"spans": {(0, 1, "person"), (4, 5, "person"), (6, 7, "person"), (11, 12, "location")}, "unscored": []}}
    seen = frozenset({"小唐", "普洱", "明月"})  # 唐 and 月 are only inside a seen name; 普 begins one
    new = new_gold(gold, texts, seen)
    facts = posthoc.single_character_facts(new, texts, ["先生"], seen)
    assert facts["single_character"]["spans"] == 4
    assert facts["first_character_starts_a_seen_name"] == {
        "spans": 2, "names": 1, "by_type": {"person": 2, "organization": 0, "location": 0, "dungeon": 0, "item": 0, "trait": 0}}
    assert facts["followed_by_a_title_suffix"]["spans"] == 1 and facts["both"]["spans"] == 0


def test_a_text_with_two_types_is_banded_by_its_text_and_the_difference_is_counted():
    texts = {"a": "白塔白塔白塔"}
    gold = {"a": {"spans": {(0, 2, "location"), (2, 4, "location"), (4, 6, "organization")}, "unscored": []}}
    new = new_gold(gold, texts, frozenset())
    out, _ = analysis_b(new, texts, {"x": {"a": {"spans": set()}}}, {}, BANDS, [], frozenset(), {"top_n": 1, "frequent_min_spans": 3})
    assert out["groups"]["band_3_or_more"] == {"spans": 3, "names": 2, "texts": 1}
    assert out["spans_whose_band_by_text_type_would_differ"] == 3  # by (text, type) they would be 2 + 1


def test_never_in_training_text_by_substring_of_the_training_chapters():
    new = new_gold(GOLD_B, TEXTS_B, SEEN_B)
    _, found = analysis_b(new, TEXTS_B, PRED_B, {}, BANDS, [], SEEN_B, {"top_n": 1, "frequent_min_spans": 3})
    train = "\x00".join(["何普在家。", "唐来了。"])  # 唐 and 何普 occur; 白塔 and 钥匙 never do
    out = analysis_c(new, PRED_B, found, {"x": "wb", "y": "wb"}, {"m": ["x", "y"], "u": ["x", "z"]}, {"wb": train})
    assert out["builds"]["wb"]["never_in_training_text"] == {"spans": 3, "names": 2, "texts": 2}
    assert out["builds"]["wb"]["in_training_text_person"] == {"spans": 5, "names": 2, "texts": 2}
    assert out["systems"]["x"]["never_in_training_text"] == {"spans_found": 1, "recall": 0.3333, "names_found": 1, "name_share": 0.5}
    assert out["systems"]["y"]["never_in_training_text_non_person"]["spans_found"] == 1
    assert set(out["models"]) == {"m"}  # a model with an untrained run is skipped
    assert out["models"]["m"]["never_in_training_text"]["names_found_by_run"] == [1, 1]
    assert posthoc.never_in_text(new, "唐\x00普") == [i for i, g in enumerate(new) if g[4] not in ("唐",)]
    assert out["systems"]["x"]["training_text"] == "wb" and out["models"]["m"]["training_text"] == "wb"
    with pytest.raises(SystemExit, match="different training texts"):
        analysis_c(new, PRED_B, found, {"x": "wb", "y": "wc"}, {"m": ["x", "y"]}, {"wb": train, "wc": ""})


# ------------------------------------------------------------------------------- D: title-suffix answers

def test_title_suffix_split_tries_the_longest_suffix_first_and_needs_a_name_before_it():
    assert split_title("公爵夫人", ["夫人", "爵夫人"]) == ("公", "爵夫人")
    assert split_title("冬先生", ["先生"]) == ("冬", "先生")
    assert split_title("先生", ["先生"]) is None and split_title("何普", ["先生"]) is None


def test_title_suffix_handling_in_its_order():
    text = "冬先生和公爵夫人见了章医生。灰先生"
    # 冬0 先1 生2 和3 公4 爵5 夫6 人7 见8 了9 章10 医11 生12 。13 灰14 先15 生16
    gold = {"a": {"spans": {(0, 3, "person"), (4, 8, "person"), (10, 13, "person"), (14, 17, "person"), (8, 9, "item")},
                  "unscored": [(13, 14)]}}
    suffixes = ["先生", "夫人", "医生"]
    span = lambda s, e: ("a", s, e, text[s:e][:-2], text[s:e][-2:], text[s:e])
    assert handling(span(0, 3), {(0, 3, "person")}, []) == "exact_hit"
    assert handling(span(0, 3), {(0, 1, "item")}, [(0, 3)]) == "prediction_equals_name_without_suffix"  # any type; before pending
    assert handling(span(0, 3), {(0, 1, "person"), (1, 3, "person")}, []) == "other_overlap"  # not the only overlap
    assert handling(span(0, 3), {(0, 3, "item")}, []) == "other_overlap"  # the right place with another type
    assert handling(span(4, 8), {(4, 7, "person")}, [(4, 8)]) == "pending"  # a pending mark comes before other overlaps
    assert handling(span(4, 8), {(8, 9, "item")}, []) == "nothing"
    preds = {"r1": {"a": {"spans": {(0, 1, "person"), (4, 8, "person"), (11, 14, "person")}, "pending": [(14, 17)]}},
             "r2": {"a": {"spans": {(0, 3, "person")}}}}  # r1's 医生。 touches the unscored range: counted as nothing
    origin = {"a": {(0, 3, "person"): "user_or_sheet", (4, 8, "person"): "gpt_agreed", (10, 13, "person"): "reviewer",
                    (14, 17, "person"): "user_or_sheet", (8, 9, "item"): "gpt_agreed"}}
    out, keys = analysis_d(gold, {"a": text}, preds, {"m": ["r1", "r2"]}, suffixes, origin, {"a": 0})
    assert out["spans"] == 4 and out["strings"] == 4 and out["title_suffixes"] == 3
    assert out["origin"] == {"gpt_agreed": 1, "user_or_sheet": 2, "reviewer": 1}
    assert out["decided_by_the_user_the_sheet_or_the_reviewer"] == 3
    assert out["systems"]["r1"] == {"exact_hit": 1, "prediction_equals_name_without_suffix": 1, "pending": 1, "other_overlap": 0, "nothing": 1}
    assert out["systems"]["r2"] == {"exact_hit": 1, "prediction_equals_name_without_suffix": 0, "pending": 0, "other_overlap": 0, "nothing": 3}
    assert out["models"]["m"]["exact_hit"] == [1, 1]
    # strings numbered by span count, ties by first place; their text only in the keys (not committed)
    assert [s["string"] for s in out["by_string"]] == [1, 2, 3, 4] and keys["1"] == "冬先生" and keys["4"] == "灰先生"
    assert out["by_string"][1] == {**out["by_string"][1], "chars": 4, "x_chars": 2, "suffix_chars": 2}
    assert "冬" not in json.dumps(out, ensure_ascii=False)
    with pytest.raises(SystemExit, match="origin"):
        analysis_d(gold, {"a": text}, preds, {}, suffixes, {"a": {}}, {"a": 0})


# ------------------------------------------------------------------------------- E: learning-curve steps

def training_record(curve, best, windows=10, batch=4, steps_per_epoch=3, planned=18, run=12):
    cfg = {"run_id": "t", "setting": "S3", "seed": 1, "config": {"batch_size": batch}}
    met = {"run_id": "t", "train": {"chapters": 5, "windows": windows},
           "schedule": {"steps_per_epoch": steps_per_epoch, "steps_planned": planned, "steps_run": run, "stopped_early": run < planned},
           "dev_curve": [{"step": s, "micro": {"precision": None, "recall": 0.0, "f1": f}, "non_person_f1": n} for s, f, n in curve],
           "best": {"step": best}}
    return cfg, met


def test_step_record_counts_windows_exactly_and_checks_the_best_step():
    cfg, met = training_record([(2, 0.5, 0.0), (4, 0.7, 0.0), (6, 0.7, 0.1), (8, 0.6, 0.3)], best=4)
    r = step_record(cfg, met, "t")
    assert (r["best_step"], r["best_step_x_batch_size"], r["windows_seen_at_best_step"]) == (4, 16, 14)  # 10 + 1 x 4
    assert (r["dev_non_person_f1_at_best"], r["dev_non_person_f1_at_last"], r["last_evaluated_step"]) == (0.0, 0.3, 8)
    assert (r["dev_non_person_f1_max"], r["step_of_dev_non_person_f1_max"], r["dev_evaluations"]) == (0.3, 8, 4)
    assert r["stopped_early"] is True and r["steps_run"] == 12 and r["ran_all_planned_steps"] is False
    # patience ran out at the last planned evaluation: the record says stopped early, yet every planned step ran
    cfg, met = training_record([(2, 0.5, 0.0), (4, 0.7, 0.0), (6, 0.6, 0.3)], best=4, run=18)
    met["schedule"]["stopped_early"] = True
    r = step_record(cfg, met, "t")
    assert (r["stopped_early"], r["ran_all_planned_steps"], r["steps_run"], r["steps_planned"]) == (True, True, 18, 18)
    with pytest.raises(SystemExit, match="first evaluation"):  # a tie goes to the earlier step
        step_record(*training_record([(2, 0.5, 0.0), (4, 0.7, 0.0), (6, 0.7, 0.1)], best=6), "t")
    with pytest.raises(SystemExit, match="steps per epoch"):
        step_record(*training_record([(2, 0.5, 0.0)], best=2, steps_per_epoch=2), "t")
    cfg, met = training_record([(2, 0.5, 0.0), (4, 0.7, 0.0)], best=4)
    met["dev"] = {"at_best": {"micro": {"precision": 0.9, "recall": 0.0, "f1": 0.7}}}
    with pytest.raises(SystemExit, match="dev scores at the best step"):
        step_record(cfg, met, "t")


# ------------------------------------------------------------------------------- a whole synthetic run on disk

PARAGRAPHS = {"ch0781": ["唐医生到了汀城。", "白塔很高，冬先生来了。", "何普没来。"],
              # 唐0 汀城5-7 | 白塔9-11 冬先生14-17 | 何普21-23
              "ch0782": ["灰塔和白塔。", "汀城下雨，唐和何普走了。"],
              # 灰塔0-2 白塔3-5 | 汀城7-9 下雨9-11 唐12 何普14-16
              "ch0783": ["冬先生和章医生在白塔。"]}
              # 冬先生0-3 章医生4-7 白塔8-10
TRAINING = {"ch0001": "何普在家。", "ch0002": "唐来了。", "ch0003": "白塔。", "ch0004": "下雨了。", "ch0005": "灰色的塔。",
            "ch0006": "汀城。"}  # ch0003 is a development chapter


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world(tmp_path, monkeypatch, lineage=False):
    """A scoring run 'ev' (GLiNER, the dictionary, one generative run, two seeds of a tagger) with its metrics.json computed
    by evaluate's own functions, the tagger's weak build and training runs, and a posthoc config. lineage: the scoring run
    scored a later export, as stage 3 scores test2: the training snapshot "train" holds the training chapters with the
    dictionary and the mentions; the scoring snapshot "snap" holds every chapter and a dictionary that gained a later
    row (白塔); the weak build and the training runs are from "train"; the dictionary system and the seen names were
    taken from "train", so the saved predictions and every count are the same as without a lineage."""
    root = tmp_path
    chapters = {cid: Chapter(cid, int(cid[2:]), [(f"p{i:04d}", t) for i, t in enumerate(ps, 1)], "", "h")
                for cid, ps in PARAGRAPHS.items()}
    training_chapters = {cid: Chapter(cid, int(cid[2:]), [("p0001", t)], "", "h") for cid, t in TRAINING.items()}
    chapters.update(training_chapters)
    mentions, dict_rows = [{"chapter_id": "ch0002", "name": "唐", "type": "person"}], [{"name": "汀城", "type": "location", "source": "registry"}]
    train = Snapshot("train", None, {}, training_chapters, mentions, list(dict_rows))
    snap = Snapshot("snap", None, {}, chapters, mentions,
                    dict_rows + ([{"name": "白塔", "type": "location", "source": "registry"}] if lineage else []))
    snapshots = {"snap": snap, "train": train}
    train_id = "train" if lineage else "snap"
    monkeypatch.setattr(posthoc, "load_snapshot", lambda sid, root=None: snapshots[sid])
    monkeypatch.setattr(posthoc, "check_gold", lambda export, s: ([], [], {}))
    monkeypatch.setattr(posthoc, "REPO_ROOT", root)
    monkeypatch.setattr(evaluate, "REPO_ROOT", root)  # gold_origin reads the GPT drafts under it
    monkeypatch.setattr(posthoc, "git_state", lambda: {"commit": "c0", "dirty": False})
    ids = list(PARAGRAPHS)
    ann = lambda s, e, t, **k: {"start": s, "end": e, "type": t, **k}
    export = {"chapters": {
        "ch0781": {"status": "done", "annotations": [ann(0, 1, "person"), ann(5, 7, "location"), ann(9, 11, "location"),
                                                     ann(14, 17, "person"), ann(21, 23, "person")]},
        "ch0782": {"status": "done", "annotations": [ann(0, 2, "location"), ann(3, 5, "location"), ann(7, 9, "location"),
                                                     ann(12, 13, "person"), ann(14, 16, "person")],
                   "unscored": [{"start": 9, "end": 11, "reason": "r", "text": "下雨"}]},
        "ch0783": {"status": "done", "annotations": [ann(0, 3, "person"), ann(4, 7, "person", decider="reviewer"),
                                                     ann(8, 10, "location")]}},
        "llm_drafts": {"t1": {}}}
    write(root / "data" / "gold" / "test.json", export)
    gold = gold_spans(export, ids)
    write(root / "data" / "llm_tasks" / "t1" / "task.json", {"chapters": ids})
    drafted = {cid: {"spans": [{**a, "pending": False} for a in c["annotations"] if (a["end"] - a["start"]) != 3]}
               for cid, c in export["chapters"].items()}  # both passes framed 冬先生 / 章医生 differently: decided by the user
    for r in ("A", "B"):
        write(root / "data" / "llm_tasks" / "t1" / f"draft_{r}.json", {"chapters": drafted})
    write(root / "configs" / "order.json", {"order": [{"chapter_id": "ch0781", "unit_id": "u1"}, {"chapter_id": "ch0782", "unit_id": "u1"},
                                                     {"chapter_id": "ch0783", "unit_id": "u2"}]})
    write(root / "configs" / "weak.json", {"title_suffixes": ["先生", "医生", "夫人"]})
    m = lambda p, x, **k: {"paragraph_id": p, "text": x, "occurrence": 1, **k}
    answer = {"chapters": [
        {"chapter_id": "ch0781", "mentions": [m("p0001", "唐", type="person"), m("p0001", "汀城", type="location"),
                                              m("p0002", "白塔", pending=True), m("p0002", "冬", type="person")]},
        {"chapter_id": "ch0782", "mentions": [m("p0001", "灰塔和白塔", type="location"), m("p0002", "唐", type="person"),
                                              m("p0002", "何普", pending=True), m("p0002", "汀城", type="place")]},
        {"chapter_id": "ch0783", "mentions": [m("p0001", "冬先生", pending=True), m("p0001", "章", type="person")]}]}
    write(root / "data" / "llm_tasks" / "t1" / "pass_opus_r1.json", answer)
    write(root / "runs" / "c1" / "claude_calls.json",
          {"calls": [{"task": "t1", "label": "opus_r1", "answer_sha256": digest(root / "data/llm_tasks/t1/pass_opus_r1.json")}]})
    ev = root / "runs" / "ev"
    row = lambda cid, s, e, t, **k: {"chapter_id": cid, "start": s, "end": e, "text": snap.chapters[cid].text[s:e], "type": t, **k}
    gliner = [row("ch0781", 0, 3, "person", score=0.9), row("ch0781", 14, 17, "person", score=0.8),
              row("ch0782", 3, 5, "location", score=0.7), row("ch0782", 9, 11, "item", score=0.7), row("ch0783", 8, 10, "location", score=0.4)]
    gliner = [{**r, "kept": r["score"] > 0.5} for r in gliner]
    dictionary, _ = dictionary_predictions(train.dictionary, [snap.chapters[c] for c in ids], 2)  # = snap's without a lineage
    ws1 = [row("ch0781", 0, 1, "person"), row("ch0781", 9, 11, "organization"), row("ch0781", 14, 15, "person"),
           row("ch0781", 21, 23, "person"), row("ch0782", 0, 2, "location"), row("ch0782", 3, 5, "location"),
           row("ch0782", 14, 16, "person"), row("ch0783", 0, 1, "person"), row("ch0783", 4, 5, "person"), row("ch0783", 8, 10, "location")]
    ws2 = [row("ch0781", 9, 11, "location"), row("ch0781", 14, 17, "person"), row("ch0782", 12, 13, "person"),
           row("ch0782", 4, 5, "location"), row("ch0783", 0, 3, "person"), row("ch0783", 8, 10, "location"), row("ch0783", 4, 7, "item")]
    llm_preds, book = llm_predictions(answer, snap, ids)
    for name, rows in (("gliner_zero_shot", gliner), ("dictionary", dictionary), ("opus_r1", llm_preds), ("ws_a_s1", ws1),
                       ("ws_a_s2", ws2)):
        write(ev / f"predictions_test_{name}.jsonl", rows)
    systems = {"gliner_zero_shot": as_prediction(gliner, ids, 0.5), "dictionary": as_prediction(dictionary, ids),
               "opus_r1": as_prediction(llm_preds, ids), "ws_a_s1": as_prediction(ws1, ids), "ws_a_s2": as_prediction(ws2, ids)}
    for cid in ids:
        systems["opus_r1"][cid]["errors"] = book["invalid_by_chapter"].get(cid, 0)
        systems["opus_r1"][cid]["pending"] = [(a, b) for c, a, b in book["pending_places"] if c == cid]

    # the weak build and the two training runs of the tagger
    train_ids = ["ch0001", "ch0002", "ch0004", "ch0005", "ch0006"]
    write(root / "runs" / "wb" / "config.json", {"run_id": "wb", "snapshot": {"id": train_id},
                                                 "config": {"test_order": "configs/order.json", "max_train_chapter": 780},
                                                 "split": {"dev_chapters": ["ch0003"], "train_ids_sha256": ids_sha256(train_ids)}})
    records = {}
    for seed in (1, 2):
        cfg, met = training_record([(2, 0.5, 0.0), (4, 0.6 + seed / 10, 0.1), (6, 0.6, 0.2)], best=4)
        write(root / "runs" / f"tr{seed}" / "config.json", {**cfg, "run_id": f"tr{seed}", "seed": seed, "snapshot": {"id": train_id},
                                                            "weak_labels": {"build": "wb", "train_ids_sha256": ids_sha256(train_ids)}})
        write(root / "runs" / f"tr{seed}" / "metrics.json", {**met, "run_id": f"tr{seed}"})
        records[f"ws_a_s{seed}"] = {
            "predictions_sha256": digest(ev / f"predictions_test_ws_a_s{seed}.jsonl"), "setting": "S3", "seed": seed,
            "model_run": f"runs/tr{seed}", "model_run_config_sha256": digest(root / "runs" / f"tr{seed}" / "config.json"),
            "weak_build": {"build": "runs/wb", "train_ids_sha256": ids_sha256(train_ids)}}

    models = {"opus": ["opus_r1"], "ws_a": ["ws_a_s1", "ws_a_s2"], "gliner": ["gliner_zero_shot"], "dictionary_match": ["dictionary"]}
    b = {"n": 60, "seed": 3}
    eval_cfg = {"snapshot": "snap", **({"training_snapshot": "train"} if lineage else {}),
                "gold_test": "data/gold/test.json", "test_order": "configs/order.json",
                "dictionary": {"min_len": 2, **({"from_training_snapshot": True} if lineage else {})}, "bootstrap": b,
                "llm_systems": [{"name": "opus_r1", "label": "opus_r1", "calls": "runs/c1/claude_calls.json"}],
                "span_systems": [{"name": "ws_a_s1"}, {"name": "ws_a_s2"}], "models": models, "new_names": {"cutoff_chapter": 780}}
    write(ev / "config.json", {"run_id": "ev", "config": eval_cfg, "snapshot": {"id": "snap"}, "code": {"commit": "e0", "dirty": False},
                               **({"snapshots": {"training_snapshot": {"id": "train", "source_commit": None},
                                                 "scoring_snapshot": {"id": "snap", "source_commit": None},
                                                 "shared_chapters": 6, "shared_text_identical": True,
                                                 "chapters_only_in_the_scoring_snapshot": 3}} if lineage else {}),
                               "gold_test": {"file": "data/gold/test.json", "sha256": digest(root / "data/gold/test.json"), "chapters": ids}})
    texts = {c: snap.chapters[c].text for c in ids}
    seen = {"汀城", "唐"}
    strata = {"ch0781": "u1", "ch0782": "u1", "ch0783": "u2"}
    counts = {s: type_counts(gold, p) for s, p in systems.items()}
    view, _ = new_name_view(gold, systems, texts, {s: seen for s in systems}, strata, b, models)
    metrics = {
        "run_id": "ev", "per_chapter_type_counts": counts,
        "scores": {s: {"micro": summarize(list(counts[s].values()))["micro"]} for s in systems},
        "models": model_summary(models, counts, strata, b),
        "new_name_recall": {**view, "seen_sets": {"x": {"gold": novelty(gold, texts, seen)}}},
        "test": {"plot_units": dict(Counter(strata.values())),
                 "scored_gold_spans": sum(1 for r in gold.values() for x in r["spans"] if not _touches(x, r["unscored"]))},
        "systems": {"gliner_zero_shot": {"threshold": 0.5, "outputs": {"outputs": sum(r["kept"] for r in gliner)}},
                    "dictionary": {},
                    "opus_r1": {"outputs": {"located_predictions": len(llm_preds),
                                            "invalid_predictions": sum(book["invalid_by_chapter"].values()),
                                            "abstentions_pending": sum(book["abstained_by_chapter"].values())},
                                "answers_match_logged_sha256": {"checked": 1, "no_hash_logged": 0}},
                    **records}}
    write(ev / "metrics.json", metrics)
    cfg = {"description": "synthetic", "eval_run": "runs/ev", "new_names": {"cutoff_chapter": 780},
           "title_suffixes_from": "configs/weak.json", "bootstrap": {"rounds": 60, "chapter_seed": 3, "run_seed": 4},
           "pairs": [["ws_a", "opus"], ["ws_a", "gliner"], ["gliner", "dictionary_match"]], "frequency_bands": BANDS,
           "concentration": {"top_n": 2, "frequent_min_spans": 3}, "tiers": {"780": "ws_a"}}
    write(root / "configs" / "posthoc_t.json", cfg)
    return SimpleNamespace(root=root, ev=ev, cfg=cfg, metrics=metrics, cfg_file=str(root / "configs" / "posthoc_t.json"),
                           snapshots=snapshots)


def test_a_whole_synthetic_run_checks_its_inputs_and_writes_counts_only(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    a = analyse(w.cfg)
    assert a["test"] == {"chapters": 3, "scored_gold_spans": 13, "new_gold_spans": 9}
    pair = a["A"]["pairs"]["ws_a - opus"]
    assert pair["stored_key"] == "opus - ws_a" and pair["stored_direction"] == "reversed"
    assert set(pair) >= {"point", "chapter_only", "chapters_and_runs"} and a["A"]["models"]["opus"]["run_draws"] is False
    d = a["D"]  # 冬先生 x2 (decided by the user: neither GPT pass framed it whole), 章医生 (decided by the reviewer)
    assert d["spans"] == 3 and d["strings"] == 2 and d["origin"] == {"gpt_agreed": 0, "user_or_sheet": 2, "reviewer": 1}
    assert d["systems"]["opus_r1"] == {"exact_hit": 0, "prediction_equals_name_without_suffix": 2, "pending": 1, "other_overlap": 0, "nothing": 0}
    assert d["systems"]["ws_a_s1"]["prediction_equals_name_without_suffix"] == 3
    assert d["systems"]["ws_a_s2"] == {"exact_hit": 2, "prediction_equals_name_without_suffix": 0, "pending": 0, "other_overlap": 1, "nothing": 0}
    assert d["systems"]["gliner_zero_shot"]["exact_hit"] == 1
    # 白塔 (dev chapter only), 灰塔, 冬先生, 章医生 never occur in the training chapters; 何普 does
    assert a["C"]["builds"]["runs/wb"]["never_in_training_text"]["names"] == 4 and a["C"]["builds"]["runs/wb"]["training_chapters"] == 5
    assert a["C"]["builds"]["runs/wb"]["from_builds"] == {"runs/wb": 5} and a["C"]["models"]["ws_a"]["training_text"] == "runs/wb"
    assert a["C"]["systems"]["ws_a_s1"]["training_runs"] == ["runs/tr1"]
    assert a["E"]["runs"]["ws_a_s2"]["windows_seen_at_best_step"] == 14 and a["E"]["tiers"]["780"]["best_step"] == [4, 4]
    assert a["E"]["tiers"]["780"]["max_train_chapter"] == 780 and a["E"]["tiers"]["780"]["ran_all_planned_steps"] == [False, False]
    assert pair["chapter_only_rounds_give_the_scoring_runs_stored_pooled_interval_in_its_direction"] is True
    assert pair["chapter_only_against_the_stored_interval_negated"]["statistics"] == 5
    assert a["A"]["interval_side_changes_with_run_draws"] == posthoc.side_changes(a["A"]["pairs"])
    assert a["B"]["systems"]["ws_a_s2"]["all"]["spans_found"] == w.metrics["new_name_recall"]["systems"]["ws_a_s2"]["tp"]
    posthoc.main([w.cfg_file, "--run-id", "r1"])
    config = json.loads((w.root / "runs" / "r1" / "config.json").read_text(encoding="utf-8"))
    text = (w.root / "runs" / "r1" / "metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(text)
    assert "冬" not in text and "白塔" not in text and metrics["post_hoc"].startswith("POST HOC")
    assert config["post_hoc"].startswith("POST HOC") and "POST HOC" in posthoc.__doc__
    assert config["rules"] == posthoc.RULES and config["code"] == {"commit": "c0", "dirty": False}
    assert metrics["checks"]["tier_labels_equal_the_weak_builds_max_train_chapter"] is True
    assert metrics["checks"]["chapter_only_rounds_give_the_scoring_runs_stored_pooled_intervals_in_their_direction"] is True
    assert "interval_side_changes_with_run_draws" in metrics["A_seed_aware_paired_intervals"]
    assert "surname" not in json.dumps(metrics["D_title_suffix_answers"])
    assert config["inputs_sha256"]["runs/ev/predictions_test_ws_a_s1.jsonl"] == w.metrics["systems"]["ws_a_s1"]["predictions_sha256"]
    assert "runs/tr1/config.json" in config["inputs_sha256"] and "runs/wb/config.json" in config["inputs_sha256"]
    keys = json.loads((w.root / "data" / "analysis" / "r1" / "keys.json").read_text(encoding="utf-8"))
    assert keys["D_strings"] == {"1": "冬先生", "2": "章医生"}
    with pytest.raises(SystemExit, match="never overwritten"):
        posthoc.main([w.cfg_file, "--run-id", "r1"])
    monkeypatch.setattr(posthoc, "git_state", lambda: {"commit": "c0", "dirty": True})
    with pytest.raises(SystemExit, match="committed code"):
        posthoc.main([w.cfg_file, "--run-id", "r2"])
    assert not (w.root / "runs" / "r2").exists()


@pytest.mark.parametrize("tamper,message", [
    ("stored_interval", "does not reproduce the scoring run's stored pooled interval"),
    ("span_file", "is not the file the scoring run recorded"),
    ("training_config", "is not the file the scoring run hashed"),
    ("weak_build", "do not give the sha256 the build recorded"),
    ("chapter_seed", "must be the scoring run's"),
    ("new_name_counts", "new-name counts differ"),
    ("pair", "listed twice"),
    ("tier_label", "max_train_chapter"),
    ("tier_name", "max_train_chapter")])
def test_inputs_that_do_not_match_stop_the_run(tmp_path, monkeypatch, tamper, message):
    w = world(tmp_path, monkeypatch)
    if tamper == "stored_interval":
        w.metrics["models"]["pooled_paired"]["opus - ws_a"]["micro_f1"]["share_of_rounds_above_0"] += 0.01
        write(w.ev / "metrics.json", w.metrics)
    elif tamper == "span_file":
        rows = (w.ev / "predictions_test_ws_a_s1.jsonl").read_text(encoding="utf-8").splitlines()
        (w.ev / "predictions_test_ws_a_s1.jsonl").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    elif tamper == "training_config":
        f = w.root / "runs" / "tr1" / "config.json"
        write(f, {**json.loads(f.read_text(encoding="utf-8")), "note": "edited"})
    elif tamper == "weak_build":
        f = w.root / "runs" / "wb" / "config.json"
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["split"]["dev_chapters"] = []  # ch0003 would become a training chapter: another sha256
        write(f, rec)
    elif tamper == "chapter_seed":
        w.cfg["bootstrap"]["chapter_seed"] = 4
    elif tamper == "new_name_counts":
        w.metrics["new_name_recall"]["systems"]["gliner_zero_shot"]["fn"] += 1
        write(w.ev / "metrics.json", w.metrics)
    elif tamper == "pair":
        w.cfg["pairs"].append(["opus", "ws_a"])
    elif tamper == "tier_label":
        w.cfg["tiers"] = {"400": "ws_a"}  # its runs' weak build has max_train_chapter 780
    elif tamper == "tier_name":
        w.cfg["tiers"] = {"all": "ws_a"}
    with pytest.raises(SystemExit, match=message):
        analyse(w.cfg)


WB_TRAIN = ["ch0001", "ch0002", "ch0004", "ch0005", "ch0006"]


def fine_tuned(w, replay=False, stage1_extra=None, seed2_stage1_build="wb"):
    """Make ws_a two-stage, as A3 is: its scored runs tr1/tr2 fine-tuned on a teacher build 'tb' of one chapter (ch0004)
    from the checkpoints of stage 1 runs s1r1/s1r2 trained on 'wb' (the five training chapters); with replay, the scored
    runs also replay the teacher build 'rb' (ch0005, ch0006), as novel_ner.finetune records it."""
    from novel_ner.teacher import TEACHER_BUILD_KIND

    root = w.root
    teach = root / "configs" / "teach.json"
    write(teach, {"dev_pack": "d01", "packs": {"b01": ["ch0004"], "b02": ["ch0005", "ch0006"], "d01": ["ch0003"]}})
    for name, packs, ids in (("tb", ["b01"], ["ch0004"]), ("rb", ["b02"], ["ch0005", "ch0006"])):
        write(root / "runs" / name / "config.json", {
            "run_id": name, "snapshot": {"id": "snap"}, "kind": TEACHER_BUILD_KIND, "config": {"max_train_chapter": 780},
            "teacher": {"packs": packs}, "chapters_file": {"file": "configs/teach.json", "sha256": digest(teach)},
            "split": {"dev_chapters": ["ch0003"], "train_ids_sha256": ids_sha256(ids)}})
    built = {"wb": WB_TRAIN, "rb": ["ch0005", "ch0006"]}
    for seed in (1, 2):
        stage1 = "wb" if seed == 1 else seed2_stage1_build
        s1 = root / "runs" / f"s1r{seed}" / "config.json"
        write(s1, {"run_id": f"s1r{seed}", "setting": "S3", "seed": seed, "snapshot": {"id": "snap"},
                   "weak_labels": {"build": stage1, "train_ids_sha256": ids_sha256(built[stage1])},
                   **((stage1_extra or {}) if seed == 1 else {})})
        tr = root / "runs" / f"tr{seed}" / "config.json"
        rec = json.loads(tr.read_text(encoding="utf-8"))
        rec["weak_labels"] = {"build": "tb", "train_ids_sha256": ids_sha256(["ch0004"])}
        rec["init_from"] = {"run_id": f"s1r{seed}", "config_sha256": digest(s1)}
        if replay:
            rec["replay"] = {"weak_build": "rb", "train_ids_sha256": ids_sha256(["ch0005", "ch0006"])}
        write(tr, rec)
        w.metrics["systems"][f"ws_a_s{seed}"].update(model_run_config_sha256=digest(tr),
                                                     weak_build={"build": "runs/tb", "train_ids_sha256": ids_sha256(["ch0004"])})
    write(w.ev / "metrics.json", w.metrics)


@pytest.mark.parametrize("replay", [False, True])
def test_a_fine_tuned_run_counts_the_text_of_every_stage(tmp_path, monkeypatch, replay):
    """何普 occurs only in ch0001, a stage 1 chapter: with the last stage's build alone (the rule before 2026-10-04
    evening) it would count as never in the training text; over every stage it is in it, as for the one-stage world."""
    w = world(tmp_path, monkeypatch)
    one_stage = analyse(w.cfg)["C"]["builds"]["runs/wb"]
    fine_tuned(w, replay=replay)
    a = analyse(w.cfg)
    key = "runs/tb + runs/rb + runs/wb" if replay else "runs/tb + runs/wb"
    c = a["C"]
    assert set(c["builds"]) == {key} and c["models"]["ws_a"]["training_text"] == key
    assert c["builds"][key]["training_chapters"] == 5
    assert c["builds"][key]["from_builds"] == ({"runs/tb": 1, "runs/rb": 2, "runs/wb": 5} if replay else {"runs/tb": 1, "runs/wb": 5})
    for g in ("never_in_training_text", "in_training_text", "never_in_training_text_person", "in_training_text_non_person"):
        assert c["builds"][key][g] == one_stage[g]
    assert c["builds"][key]["never_in_training_text"]["names"] == 4  # 白塔, 灰塔, 冬先生, 章医生; not 何普
    assert c["systems"]["ws_a_s1"]["training_runs"] == ["runs/tr1", "runs/s1r1"]
    assert {"runs/s1r1/config.json", "runs/s1r2/config.json", "runs/tb/config.json"} <= set(a["files"])
    assert a["E"]["tiers"]["780"]["max_train_chapter"] == 780  # the tier label is still the last stage build's
    # the old rule (the last stage's build alone): ch0004 holds none of the new names, so 何普 would count as never seen
    monkeypatch.setattr(posthoc, "training_lineage", lambda system, root, snapshot, files, what: (
        [(system["weak_build"]["build"], system["weak_build"]["train_ids_sha256"], "last stage")], [system["model_run"]]))
    old = analyse(w.cfg)["C"]["builds"]["runs/tb"]
    assert old["never_in_training_text"]["names"] == 5 and old["in_training_text"]["names"] == 0


@pytest.mark.parametrize("change,message", [
    ("stage1_config", "is not the file runs/tr1's init_from hashed"),
    ("cycle", "comes back to runs/tr1"),
    ("init_record", "does not record the run and config sha256 of its init_from"),
    ("replay_ids", "are not the ones runs/tr1 replay recorded"),
    ("stage1_build", "different training texts")])
def test_an_earlier_stage_that_does_not_match_stops_the_run(tmp_path, monkeypatch, change, message):
    w = world(tmp_path, monkeypatch)
    if change == "stage1_config":
        fine_tuned(w)
        s1 = w.root / "runs" / "s1r1" / "config.json"
        write(s1, {**json.loads(s1.read_text(encoding="utf-8")), "note": "edited"})
    elif change == "cycle":
        fine_tuned(w, stage1_extra={"init_from": {"run_id": "tr1", "config_sha256": "0" * 64}})
    elif change == "init_record":
        fine_tuned(w)
        tr = w.root / "runs" / "tr1" / "config.json"
        rec = json.loads(tr.read_text(encoding="utf-8"))
        rec["init_from"] = {"run_id": "s1r1"}
        write(tr, rec)
        w.metrics["systems"]["ws_a_s1"]["model_run_config_sha256"] = digest(tr)
        write(w.ev / "metrics.json", w.metrics)
    elif change == "replay_ids":
        fine_tuned(w, replay=True)
        tr = w.root / "runs" / "tr1" / "config.json"
        rec = json.loads(tr.read_text(encoding="utf-8"))
        rec["replay"]["train_ids_sha256"] = ids_sha256(["ch0005"])
        write(tr, rec)
        w.metrics["systems"]["ws_a_s1"]["model_run_config_sha256"] = digest(tr)
        write(w.ev / "metrics.json", w.metrics)
    elif change == "stage1_build":  # seed 2's stage 1 on another build: the model's runs no longer share a training text
        fine_tuned(w, seed2_stage1_build="rb")
    with pytest.raises(SystemExit, match=message):
        analyse(w.cfg)


def test_training_chapters_of_a_teacher_build_come_from_its_packs(tmp_path):
    """A teacher labels build (novel_ner.teacher) is recomputed from its packs and the chapters file it hashed, never
    by weak.split_ids; the sha256 check is the same as for a weak.py build."""
    from novel_ner.teacher import TEACHER_BUILD_KIND

    chapters = {f"ch{n:04d}": Chapter(f"ch{n:04d}", n, [("p0001", "x")], "", "h") for n in range(1, 8)}
    snap = Snapshot("snap", None, {}, chapters, [], [])
    teach = tmp_path / "configs" / "teach.json"
    write(teach, {"dev_pack": "d01", "packs": {"b01": ["ch0004", "ch0001"], "b02": ["ch0006"], "d01": ["ch0003"]}})
    build = {"kind": TEACHER_BUILD_KIND, "config": {"max_train_chapter": 780}, "teacher": {"packs": ["b01", "b02"]},
             "chapters_file": {"file": "configs/teach.json", "sha256": digest(teach)},
             "split": {"dev_chapters": ["ch0003"], "train_ids_sha256": ids_sha256(["ch0001", "ch0004", "ch0006"])}}
    assert posthoc.training_chapters(build, snap, tmp_path, "wb") == ["ch0001", "ch0004", "ch0006"]
    with pytest.raises(SystemExit, match="cannot be recomputed"):
        posthoc.training_chapters({**build, "teacher": {"packs": ["b01", "d01"]}}, snap, tmp_path, "wb")
    with pytest.raises(SystemExit, match="do not give the sha256"):
        posthoc.training_chapters({**build, "teacher": {"packs": ["b01"]}}, snap, tmp_path, "wb")
    with pytest.raises(SystemExit, match="cannot be recomputed"):  # a weak.py build still needs its test order
        posthoc.training_chapters({**build, "kind": "weak"}, snap, tmp_path, "wb")


# ------------------------------------------------------------------------------- a scoring run on a later snapshot (stage 3)

def test_a_scoring_run_on_a_later_snapshot_takes_the_dictionary_seen_names_and_builds_from_the_training_snapshot(tmp_path, monkeypatch):
    """The dictionary system was matched from the training snapshot's dictionary, the seen names and the training chapters
    come from it too; the scoring snapshot's dictionary (which gained 白塔) would make the dictionary check and the new-name
    counts fail, so a passing run shows which snapshot each piece came from. The numbers equal the one-snapshot world's."""
    plain = analyse(world(tmp_path, monkeypatch).cfg)
    w = world(tmp_path, monkeypatch, lineage=True)
    a = analyse(w.cfg)
    for key in ("test", "A", "B", "C", "D", "E"):
        assert a[key] == plain[key]
    assert plain["snap_lineage"] is None
    assert (a["snap_lineage"]["training_snapshot"]["id"], a["snap_lineage"]["scoring_snapshot"]["id"]) == ("train", "snap")
    assert (a["snap_lineage"]["shared_chapters"], a["snap_lineage"]["chapters_only_in_the_scoring_snapshot"]) == (6, 3)
    assert a["C"]["builds"]["runs/wb"]["training_chapters"] == 5
    posthoc.main([w.cfg_file, "--run-id", "r-lineage"])
    config = json.loads((w.root / "runs" / "r-lineage" / "config.json").read_text(encoding="utf-8"))
    assert config["eval_run"]["snapshots"]["training_snapshot"]["id"] == "train" and config["eval_run"]["snapshot"] == {"id": "snap"}
    plain_config_keys = {"run", "run_code", "snapshot", "gold_test"}
    assert set(config["eval_run"]) == plain_config_keys | {"snapshots"}


@pytest.mark.parametrize("tamper,message", [
    ("no_snapshots_block", "does not record this training snapshot lineage"),
    ("other_training_snapshot_recorded", "does not record this training snapshot lineage"),
    ("training_run_on_the_scoring_snapshot", "names another run or snapshot"),
    ("weak_build_on_the_scoring_snapshot", "names another run or snapshot"),
    ("not_a_later_export", "not a later export"),
    ("dictionary_from_the_scoring_snapshot", "differ from dictionary matching recomputed")])
def test_a_lineage_that_does_not_hold_stops_the_run(tmp_path, monkeypatch, tamper, message):
    w = world(tmp_path, monkeypatch, lineage=True)
    ev_rec = json.loads((w.ev / "config.json").read_text(encoding="utf-8"))
    if tamper == "no_snapshots_block":
        del ev_rec["snapshots"]
        write(w.ev / "config.json", ev_rec)
    elif tamper == "other_training_snapshot_recorded":
        ev_rec["snapshots"]["training_snapshot"]["id"] = "snap"
        write(w.ev / "config.json", ev_rec)
    elif tamper == "training_run_on_the_scoring_snapshot":
        f = w.root / "runs" / "tr1" / "config.json"
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["snapshot"]["id"] = "snap"
        write(f, rec)
        w.metrics["systems"]["ws_a_s1"]["model_run_config_sha256"] = digest(f)
        write(w.ev / "metrics.json", w.metrics)
    elif tamper == "weak_build_on_the_scoring_snapshot":
        f = w.root / "runs" / "wb" / "config.json"
        write(f, {**json.loads(f.read_text(encoding="utf-8")), "snapshot": {"id": "snap"}})
    elif tamper == "not_a_later_export":
        snap = w.snapshots["snap"]
        changed = replace(snap, chapters={**snap.chapters, "ch0001": Chapter("ch0001", 1, [("p0001", "改过的字。")], "", "h")})
        monkeypatch.setattr(posthoc, "load_snapshot", lambda sid, root=None: {"snap": changed, "train": w.snapshots["train"]}[sid])
    elif tamper == "dictionary_from_the_scoring_snapshot":  # the saved predictions were made from the training snapshot's dictionary
        ev_rec["config"]["dictionary"] = {"min_len": 2}
        ev_rec["config"].pop("training_snapshot")
        del ev_rec["snapshots"]
        write(w.ev / "config.json", ev_rec)
    with pytest.raises(SystemExit, match=message):
        analyse(w.cfg)
