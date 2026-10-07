import inspect
import json

import pytest

from novel_ner import consistency, tagger
from novel_ner.consistency import run_build, transform
from novel_ner.evaluate import sha256, write_jsonl
from novel_ner.snapshot import Chapter, Snapshot
from novel_ner.teacher import TEACHER_BUILD_KIND, build_train_ids
from novel_ner.weak import char_allowed, ids_sha256

ORDER = ["K7_boundary", "K6_type_conflict", "K5_pending_majority", "K8_same_chapter_partial"]
CFG = {"snapshot": "snap-test", "builds": [{"source": "tb", "target": "tbc"}], "rules_in_order": ORDER,
       "K7_boundary": {"rule": "guide B2", "min_len_shorter": 1},
       "K6_type_conflict": {"rule": "minority", "minority_share": 0.2},
       "K5_pending_majority": {"rule": "pending", "pending_share": 0.3333, "min_pending": 2},
       "K8_same_chapter_partial": {"rule": "partial", "min_len": 2}}
DEV = ["ch0007", "ch0008"]
SRC = "teacher_opus_t1"
BOOK = set("林小雨回音铃黑肩章灰塔城灯风车水井甲乙丙门阿")  # every book character the fixtures use


def chapter(n, paras):
    return Chapter(f"ch{n:04d}", n, [(f"p{i:04d}", t) for i, t in enumerate(paras, 1)], f"第{n}章", f"h{n}")


def occ(text, s, k=1):
    """Offset of the k-th occurrence of s in text."""
    at = -1
    for _ in range(k):
        at = text.index(s, at + 1)
    return at


def span(ch, s, k, type_):
    st = occ(ch.text, s, k)
    return {"start": st, "end": st + len(s), "type": type_, "text": s, "sources": [SRC]}


def pend(ch, s, k):
    st = occ(ch.text, s, k)
    return {"start": st, "end": st + len(s), "text": s, "types": None, "reason": "teacher_pending", "sources": [SRC]}


def row(ch, spans=(), uncertain=()):
    return {"chapter_id": ch.chapter_id, "number": ch.number, "split": "train",
            "spans": sorted(spans, key=lambda x: x["start"]), "uncertain": sorted(uncertain, key=lambda x: x["start"])}


def snapshot(chapters):
    extra = [chapter(7, ["开发章。"]), chapter(8, ["开发章。"])]
    return Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in chapters + extra}, [], [])


def run(cfg, chapters, rows):
    s = snapshot(chapters)
    texts = {c.chapter_id: c.text for c in chapters}
    new_rows, counts, keys = transform(cfg, rows, texts)
    return s, {r["chapter_id"]: r for r in new_rows}, counts, keys


def plain(items, *keys):
    return [tuple(x[k] for k in keys) for x in items]


# ---------------------------------------------------------------------------------------------- the rules one by one


def test_k7_extends_to_the_longest_covering_positive_of_the_same_type():
    c1 = chapter(1, ["林小雨走了。小雨回头。大林小雨来了。"])
    c2 = chapter(2, ["林小雨看着回音铃。"])
    c3 = chapter(3, ["大林小雨来了。", "回音响起。"])
    rows = [row(c1, [span(c1, "小雨", 1, "person"), span(c1, "小雨", 2, "person"), span(c1, "小雨", 3, "person")]),
            row(c2, [span(c2, "林小雨", 1, "person"), span(c2, "回音铃", 1, "location")]),
            row(c3, [span(c3, "大林小雨", 1, "person"), span(c3, "回音", 1, "item")])]
    _, out, counts, keys = run(CFG, [c1, c2, c3], rows)
    # the first 小雨 sits in 林小雨, the third in 大林小雨 (the longest wins over 林小雨), the second stands alone
    assert plain(out["ch0001"]["spans"], "start", "end", "text", "type") == [(0, 3, "林小雨", "person"), (6, 8, "小雨", "person"),
                                                                             (11, 15, "大林小雨", "person")]
    assert out["ch0001"]["spans"][0]["sources"] == [SRC, "K7_boundary"] and out["ch0001"]["spans"][1]["sources"] == [SRC]
    assert out["ch0001"]["uncertain"] == []
    # a longer string of another type (回音铃 location) never extends 回音 (item)
    assert plain(out["ch0003"]["spans"], "text", "type") == [("大林小雨", "person"), ("回音", "item")]
    k7 = counts["K7_boundary"]
    assert k7["spans_extended"] == 2 and k7["spans_extended_in_place"] == 2 and k7["extensions_overlapping"] == 0
    assert k7["spans_extended_by_type"]["person"] == 2 and k7["characters_added"] == 3 and k7["chapters_touched"] == 1
    assert k7["distinct_pairs"] == 2 and k7["distinct_shorter_strings"] == 1 and k7["distinct_longer_strings"] == 2
    assert plain(keys["K7_boundary"], "shorter", "longer", "type", "count") == [("小雨", "大林小雨", "person", 1), ("小雨", "林小雨", "person", 1)]
    assert all(counts[r]["spans_made_uncertain"] == 0 for r in ("K6_type_conflict", "K5_pending_majority"))
    assert counts["K8_same_chapter_partial"]["regions_added"] == 0


def test_k7_ties_between_longer_strings_of_one_length_go_to_the_earlier_start():
    c1 = chapter(1, ["林小雨阿来了。"])  # 小雨 at [1,3) sits in both 林小雨 [0,3) and 小雨阿 [1,4), both length 3
    c2 = chapter(2, ["林小雨走了。小雨阿走了。"])
    rows = [row(c1, [span(c1, "小雨", 1, "person")]), row(c2, [span(c2, "林小雨", 1, "person"), span(c2, "小雨阿", 1, "person")])]
    _, out, counts, keys = run(CFG, [c1, c2], rows)
    assert plain(out["ch0001"]["spans"], "start", "end", "text") == [(0, 3, "林小雨")]
    assert plain(keys["K7_boundary"], "shorter", "longer", "count") == [("小雨", "林小雨", 1)]
    # the same with the candidates' insertion order reversed: the tie rule, not the order, decides
    rows2 = [row(c1, [span(c1, "小雨", 1, "person")]), row(c2, [span(c2, "小雨阿", 1, "person"), span(c2, "林小雨", 1, "person")])]
    _, out2, _, _ = run(CFG, [c1, c2], rows2)
    assert plain(out2["ch0001"]["spans"], "start", "end", "text") == [(0, 3, "林小雨")]


def test_k7_overlap_unions_the_extended_span_with_what_it_touches():
    c1 = chapter(1, ["林小雨笑了。", "阿林小雨到了。", "灰塔城很大。"])
    c2 = chapter(2, ["林小雨来了灰塔城。"])
    p2, p3 = occ(c1.text, "阿林小雨"), occ(c1.text, "灰塔城")
    typed = {"start": p3 + 2, "end": p3 + 3, "text": "城", "types": ["organization"], "reason": "other", "sources": [SRC]}
    rows = [row(c1, [span(c1, "林", 1, "person"), span(c1, "小雨", 1, "person"), span(c1, "小雨", 2, "person"),
                     span(c1, "灰塔", 1, "location")], [pend(c1, "阿林", 1), typed]),
            row(c2, [span(c2, "林小雨", 1, "person"), span(c2, "灰塔城", 1, "location")])]
    _, out, counts, keys = run(CFG, [c1, c2], rows)
    assert out["ch0001"]["spans"] == []
    unc = out["ch0001"]["uncertain"]
    assert plain(unc, "start", "end", "text", "types", "reason") == [
        (0, 3, "林小雨", ["person"], "K7_overlap"),  # 林 extended over 小雨 (another span): the union, both persons
        (p2, p2 + 4, "阿林小雨", None, "K7_overlap"),  # 小雨 extended over the pending 阿林: a region of unknown type makes the union unknown
        (p3, p3 + 3, "灰塔城", ["location", "organization"], "K7_overlap")]  # 灰塔 extended over a typed region
    assert unc[0]["sources"] == [SRC, "K7_boundary", "K7_overlap"]
    k7 = counts["K7_boundary"]
    assert k7["spans_extended"] == 3 and k7["extensions_overlapping"] == 3 and k7["spans_extended_in_place"] == 0
    assert k7["spans_made_uncertain"] == 4 and k7["spans_made_uncertain_by_type"] == {"person": 3, "organization": 0, "location": 1,
                                                                                    "dungeon": 0, "item": 0, "trait": 0}
    assert k7["regions_absorbed"] == 2 and k7["regions_absorbed_by_reason"] == {"other": 1, "teacher_pending": 1}
    assert k7["overlap_regions_after_the_rule"] == 3
    assert plain(keys["K7_boundary"], "shorter", "longer", "count") == [("小雨", "林小雨", 1), ("林", "林小雨", 1), ("灰塔", "灰塔城", 1)]
    # the second 小雨 of paragraph 1 was merged by 林's extension and never extended on its own
    assert out["ch0002"]["uncertain"] == [] and len(out["ch0002"]["spans"]) == 2


def test_k6_minority_type_becomes_uncertain_at_the_threshold_and_a_split_is_kept():
    c4 = chapter(4, ["黑肩章在左。黑肩章在右。黑肩章在前。黑肩章在后。黑肩章在中。", "灰塔在左。灰塔在右。灰塔在前。灰塔在后。"])
    spans = [span(c4, "黑肩章", k, "organization") for k in range(1, 5)] + [span(c4, "黑肩章", 5, "item")]
    spans += [span(c4, "灰塔", k, "location") for k in range(1, 4)] + [span(c4, "灰塔", 4, "organization")]
    _, out, counts, keys = run(CFG, [c4], [row(c4, spans)])
    kept = plain(out["ch0004"]["spans"], "text", "type")
    assert kept == [("黑肩章", "organization")] * 4 + [("灰塔", "location")] * 3 + [("灰塔", "organization")]  # 1 of 4 = 0.25 > 0.2: kept
    unc = out["ch0004"]["uncertain"]
    assert plain(unc, "text", "types", "reason") == [("黑肩章", ["item", "organization"], "K6_type_conflict")]  # 1 of 5 = 0.2 <= 0.2
    assert unc[0]["start"] == occ(c4.text, "黑肩章", 5) and unc[0]["sources"] == [SRC, "K6_type_conflict"]
    k6 = counts["K6_type_conflict"]
    assert k6["strings_with_two_or_more_types"] == 2 and k6["strings_affected"] == 1 and k6["strings_kept_as_a_split"] == 1
    assert k6["spans_made_uncertain"] == 1 and k6["spans_made_uncertain_by_type"]["item"] == 1 and k6["regions_added"] == 1
    assert k6["spans_of_strings_with_two_or_more_types"] == 9 and k6["chapters_touched"] == 1
    assert keys["K6_type_conflict"] == [{"text": "黑肩章", "counts_by_type": {"item": 1, "organization": 4}, "minority_types": ["item"],
                                        "spans_made_uncertain": 1}]
    assert counts["K8_same_chapter_partial"]["regions_added"] == 0


def test_k5_pending_majority_at_and_below_the_threshold_and_min_pending():
    c5 = chapter(5, ["灯塔。灯塔。风车。风车。风车。风车。水井。水井。水井。水井。水井。门。"])
    c6 = chapter(6, ["灯塔。灯塔。风车。风车。水井。水井。门。"])
    spans = ([span(c5, "灯塔", k, "organization") for k in (1, 2)] + [span(c5, "风车", k, "organization") for k in range(1, 5)]
             + [span(c5, "水井", k, "organization") for k in range(1, 6)] + [span(c5, "门", 1, "location")])
    pending = ([pend(c6, "灯塔", k) for k in (1, 2)] + [pend(c6, "风车", k) for k in (1, 2)] + [pend(c6, "水井", k) for k in (1, 2)]
               + [pend(c6, "门", 1)])
    _, out, counts, keys = run(CFG, [c5, c6], [row(c5, spans), row(c6, [], pending)])
    # 灯塔 2 pending of 4 (0.5) and 风车 2 of 6 (exactly 0.3333..): converted; 水井 2 of 7: kept; 门 1 pending: under min_pending
    assert plain(out["ch0005"]["spans"], "text", "type") == [("水井", "organization")] * 5 + [("门", "location")]
    assert plain(out["ch0005"]["uncertain"], "text", "types", "reason") == [("灯塔", ["organization"], "K5_pending_majority")] * 2 + \
        [("风车", ["organization"], "K5_pending_majority")] * 4
    assert out["ch0005"]["uncertain"][0]["sources"] == [SRC, "K5_pending_majority"]
    assert out["ch0006"] == row(c6, [], pending)
    k5 = counts["K5_pending_majority"]
    assert k5["strings_with_pending"] == 4 and k5["strings_with_pending_and_positives"] == 4 and k5["strings_affected"] == 2
    assert k5["spans_made_uncertain"] == 6 and k5["spans_made_uncertain_by_type"]["organization"] == 6 and k5["chapters_touched"] == 1
    assert plain(keys["K5_pending_majority"], "text", "pending", "positives", "types") == [("风车", 2, 4, ["organization"]),
                                                                                           ("灯塔", 2, 2, ["organization"])]
    assert keys["K5_pending_majority"][0]["pending_share"] == 0.3333
    assert counts["K8_same_chapter_partial"]["regions_added"] == 0


def test_k8_marks_uncovered_occurrences_left_to_right_and_skips_covered_ones():
    c1 = chapter(1, ["回音铃在桌上。回音铃响了。回音铃铃。", "甲甲甲甲甲。", "门。门。", "小雨和林小雨。"])
    p2 = occ(c1.text, "甲甲甲甲甲")
    rows = [row(c1, [span(c1, "回音铃", 1, "item"), span(c1, "甲甲", 1, "person"), span(c1, "门", 1, "location"),
                     span(c1, "小雨", 1, "person"), span(c1, "林小雨", 1, "person")], [pend(c1, "铃铃", 1)])]
    _, out, counts, keys = run(CFG, [c1], rows)
    assert len(out["ch0001"]["spans"]) == 5  # nothing extended: 小雨 is not covered by 林小雨 at its position
    k8 = [u for u in out["ch0001"]["uncertain"] if u["reason"] == "K8_same_chapter_partial"]
    assert plain(k8, "start", "end", "text", "types") == [
        (occ(c1.text, "回音铃", 2), occ(c1.text, "回音铃", 2) + 3, "回音铃", ["item"]),  # the second occurrence: free
        (p2 + 2, p2 + 4, "甲甲", ["person"])]  # 甲甲 at p2+1 overlaps the span, p2+2 is taken, p2+3 overlaps that one
    # the third 回音铃 overlaps the pending 铃铃; 门 is shorter than min_len; 小雨 inside 林小雨 is covered by that span
    assert k8[0]["sources"] == ["K8_same_chapter_partial"]
    c = counts["K8_same_chapter_partial"]
    assert c["regions_added"] == 2 and c["distinct_strings"] == 2 and c["chapters_touched"] == 1
    assert c["regions_added_by_types"] == {"item": 1, "person": 1} and c["occurrences_skipped_covered"] == 8
    assert plain(keys["K8_same_chapter_partial"], "text", "types", "count") == [("回音铃", ["item"], 1), ("甲甲", ["person"], 1)]


def test_k8_runs_after_k5_and_sees_its_regions():
    c1 = chapter(1, ["甲乙丙。乙丙。甲乙。"])
    c2 = chapter(2, ["甲乙。甲乙。"])
    rows = [row(c1, [span(c1, "甲乙", 1, "person"), span(c1, "乙丙", 2, "item")]), row(c2, [], [pend(c2, "甲乙", 1), pend(c2, "甲乙", 2)])]
    _, out, counts, _ = run(CFG, [c1, c2], rows)
    # K5 turns 甲乙 [0,2) into a region (2 pending of 3); K8 then neither looks for 甲乙 (no longer a positive of the chapter)
    # nor marks 乙丙 at [1,3), which overlaps K5's region
    assert plain(out["ch0001"]["spans"], "start", "end", "text") == [(4, 6, "乙丙")]
    assert plain(out["ch0001"]["uncertain"], "start", "end", "text", "types", "reason") == [(0, 2, "甲乙", ["person"], "K5_pending_majority")]
    assert counts["K8_same_chapter_partial"]["regions_added"] == 0
    # the order is the config's: K8 before K5 would mark the last 甲乙 first
    other = {**CFG, "rules_in_order": ["K7_boundary", "K6_type_conflict", "K8_same_chapter_partial", "K5_pending_majority"]}
    _, out2, counts2, _ = run(other, [c1, c2], rows)
    assert plain(out2["ch0001"]["uncertain"], "start", "end", "reason") == [(0, 2, "K5_pending_majority"), (7, 9, "K8_same_chapter_partial")]
    assert counts2["K8_same_chapter_partial"]["regions_added"] == 1
    # a rule left out of the list is not applied (configs/consistency_v2.json drops K5); duplicates and a wrong
    # rules_not_applied are refused
    _, counts3, _ = transform({**CFG, "rules_in_order": ORDER[:3]}, rows, {c.chapter_id: c.text for c in (c1, c2)})
    assert counts3[ORDER[3]]["regions_added"] == 0
    with pytest.raises(SystemExit, match="rules_in_order"):
        transform({**CFG, "rules_in_order": ORDER + [ORDER[0]]}, rows, {c.chapter_id: c.text for c in (c1, c2)})
    with pytest.raises(SystemExit, match="rules_not_applied"):
        transform({**CFG, "rules_in_order": ORDER[:3], "rules_not_applied": []}, rows, {c.chapter_id: c.text for c in (c1, c2)})


# ---------------------------------------------------------------------------------------------- everything together


def everything():
    """The fixtures of the rule tests in one build of six chapters."""
    c1 = chapter(1, ["林小雨走了。小雨回头。大林小雨来了。", "林小雨笑了。", "阿林小雨到了。", "回音铃在桌上。回音铃响了。回音铃铃。", "甲甲甲甲甲。"])
    c2 = chapter(2, ["林小雨看着回音铃。", "甲乙丙。乙丙。甲乙。"])
    c3 = chapter(3, ["大林小雨来了。", "回音响起。", "甲乙。甲乙。"])
    c4 = chapter(4, ["黑肩章在左。黑肩章在右。黑肩章在前。黑肩章在后。黑肩章在中。", "灰塔在左。灰塔在右。灰塔在前。灰塔在后。"])
    c5 = chapter(5, ["灯塔。灯塔。风车。风车。风车。风车。水井。水井。水井。水井。水井。门。"])
    c6 = chapter(6, ["灯塔。灯塔。风车。风车。水井。水井。门。"])
    rows = [
        row(c1, [span(c1, "小雨", 1, "person"), span(c1, "小雨", 2, "person"), span(c1, "小雨", 3, "person"), span(c1, "林", 3, "person"),
                 span(c1, "小雨", 4, "person"), span(c1, "小雨", 5, "person"), span(c1, "回音铃", 1, "item"), span(c1, "甲甲", 1, "person")],
            [pend(c1, "阿林", 1), pend(c1, "铃铃", 1)]),
        row(c2, [span(c2, "林小雨", 1, "person"), span(c2, "回音铃", 1, "location"), span(c2, "甲乙", 1, "person"), span(c2, "乙丙", 2, "item")]),
        row(c3, [span(c3, "大林小雨", 1, "person"), span(c3, "回音", 1, "item")], [pend(c3, "甲乙", 1), pend(c3, "甲乙", 2)]),
        row(c4, [span(c4, "黑肩章", k, "organization") for k in range(1, 5)] + [span(c4, "黑肩章", 5, "item")]
            + [span(c4, "灰塔", k, "location") for k in range(1, 4)] + [span(c4, "灰塔", 4, "organization")]),
        row(c5, [span(c5, "灯塔", k, "organization") for k in (1, 2)] + [span(c5, "风车", k, "organization") for k in range(1, 5)]
            + [span(c5, "水井", k, "organization") for k in range(1, 6)] + [span(c5, "门", 1, "location")]),
        row(c6, [], [pend(c6, "灯塔", k) for k in (1, 2)] + [pend(c6, "风车", k) for k in (1, 2)] + [pend(c6, "水井", k) for k in (1, 2)]
            + [pend(c6, "门", 1)]),
    ]
    return [c1, c2, c3, c4, c5, c6], rows


def test_rows_are_valid_after_the_rules_and_every_rule_left_its_mark():
    chapters, rows = everything()
    s, out, counts, keys = run(CFG, chapters, rows)
    assert list(out) == [c.chapter_id for c in chapters]
    for c in chapters:
        r = out[c.chapter_id]
        assert r["chapter_id"] == c.chapter_id and r["number"] == c.number and r["split"] == "train"
        assert [x["start"] for x in r["spans"]] == sorted(x["start"] for x in r["spans"])
        assert [x["start"] for x in r["uncertain"]] == sorted(x["start"] for x in r["uncertain"])
        for x in r["spans"] + r["uncertain"]:
            assert c.text[x["start"]:x["end"]] == x["text"] and x["text"]
        allowed = char_allowed(len(c.text), r["spans"], r["uncertain"])  # raises on an overlap
        assert len(allowed) == len(c.text)
    # 回音铃 is item in chapter 1 and location in chapter 2: a 1:1 split, kept; the K8 region in chapter 1 keeps item only
    assert plain([u for u in out["ch0001"]["uncertain"] if u["reason"] == "K8_same_chapter_partial"], "text", "types") == [("回音铃", ["item"]), ("甲甲", ["person"])]
    assert counts["K6_type_conflict"]["strings_with_two_or_more_types"] == 3 and counts["K6_type_conflict"]["strings_affected"] == 1
    assert counts["K7_boundary"]["spans_extended"] == 4 and counts["K7_boundary"]["extensions_overlapping"] == 2
    assert counts["K5_pending_majority"]["strings_affected"] == 3 and counts["K8_same_chapter_partial"]["regions_added"] == 2
    assert {k["text"] for k in keys["K5_pending_majority"]} == {"灯塔", "风车", "甲乙"}
    # the pending counts are the source's: chapter 6 is untouched
    assert out["ch0006"] == rows[5]
    # the tagger's own row check accepts the rows (the dev2 list is passed: the repository's file is not this layout)
    assert len(tagger.training_rows(list(out.values()), s, set(DEV), dev2_ids=frozenset())) == 6


# ---------------------------------------------------------------------------------------------- the build on disk


def make_repo(root, s, rows):
    """A repository layout under root: the source build tb (record, T and W files) and its chapters file."""
    ids = [r["chapter_id"] for r in rows]
    (root / "configs").mkdir()
    chapters_file = root / "configs" / "t.chapters.json"
    chapters_file.write_text(json.dumps({"dev_pack": "teach_dev_b01", "training_packs": ["teach_b01", "teach_b02"],
                                         "packs": {"teach_b01": ids[:3], "teach_b02": ids[3:], "teach_dev_b01": DEV[:1]}}), encoding="utf-8")
    build = root / "data" / "weak_labels" / "tb"
    build.mkdir(parents=True)
    write_jsonl(build / "labels_T.jsonl", rows)
    weak = [{**r, "spans": [{**x, "sources": [["chapter_list", x["type"]]]} for x in r["spans"]], "uncertain": []} for r in rows]
    write_jsonl(build / "labels_W.jsonl", weak)
    record = {
        "run_id": "tb", "config_file": "configs/teach.json", "config": {"snapshot": "snap-test", "max_train_chapter": 780},
        "kind": TEACHER_BUILD_KIND, "snapshot": {"id": "snap-test", "source_commit": "c"},
        "code": {"commit": "abc", "dirty": False}, "environment": {"python": "test"},
        "split": {"max_train_chapter": 780, "train_chapters": len(ids), "train_numbers": [1, 6], "train_ids_sha256": ids_sha256(ids),
                  "dev_chapters": list(DEV), "selection": "all", "rule": "test"},
        "labels": {"T": {"file": "data/weak_labels/tb/labels_T.jsonl", "sha256": sha256(build / "labels_T.jsonl")},
                   "W": {"file": "data/weak_labels/tb/labels_W.jsonl", "sha256": sha256(build / "labels_W.jsonl")}},
        "teacher": {"label": "opus_t1", "source": SRC, "packs": ["teach_b01", "teach_b02"], "pass_files": {}},
        "weak_source": {"build": "wb", "setting": "S3"},
        "chapters_file": {"file": "configs/t.chapters.json", "sha256": sha256(chapters_file)},
        "conversion_rules": ["test"], "material_scope": "test", "started_at": "2026-10-04T00:00:00+0000",
    }
    (root / "runs" / "tb").mkdir(parents=True)
    (root / "runs" / "tb" / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


@pytest.fixture
def repo(tmp_path):
    chapters, rows = everything()
    s = snapshot(chapters)
    record = make_repo(tmp_path, s, rows)
    return tmp_path, s, rows, record


def test_run_build_writes_a_record_accepted_exactly_like_the_source(repo):
    root, s, rows, src = repo
    code = {"commit": "def", "dirty": False}
    metrics, keys = run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code=code, config_sha256="cfg-digest")
    rec = json.loads((root / "runs" / "tbc" / "config.json").read_text(encoding="utf-8"))
    assert set(rec) == set(src) | {"derived_from", "consistency"}
    assert rec["run_id"] == "tbc"
    for key in ("config_file", "config", "kind", "snapshot", "environment", "split", "teacher", "weak_source", "chapters_file",
                "conversion_rules", "material_scope", "started_at"):
        assert rec[key] == src[key], key
    # the record's code is the consistency run's (evaluate reads code.dirty and records code.commit as the build's); the source's
    # code is kept under derived_from
    assert rec["code"] == code and rec["code"] != src["code"] and rec["derived_from"]["source_code"] == src["code"]
    build = root / "data" / "weak_labels" / "tbc"
    assert rec["labels"] == {"T": {"file": "data/weak_labels/tbc/labels_T.jsonl", "sha256": sha256(build / "labels_T.jsonl")},
                             "W": {"file": "data/weak_labels/tbc/labels_W.jsonl", "sha256": sha256(build / "labels_W.jsonl")}}
    assert rec["labels"]["T"]["sha256"] != src["labels"]["T"]["sha256"] and rec["labels"]["W"]["sha256"] == src["labels"]["W"]["sha256"]
    assert (build / "labels_W.jsonl").read_bytes() == (root / "data" / "weak_labels" / "tb" / "labels_W.jsonl").read_bytes()
    assert rec["derived_from"]["source"] == "tb" and rec["derived_from"]["labels_T_sha256"] == src["labels"]["T"]["sha256"]
    assert rec["derived_from"]["config_sha256"] == sha256(root / "runs" / "tb" / "config.json")
    c = rec["consistency"]
    assert c["config_file"] == "configs/consistency.json" and c["config_sha256"] == "cfg-digest" and c["code"] == code
    assert c["rules_in_order"] == ORDER and c["scratch_out"] is None
    assert c["thresholds"] == {"K7_boundary": {"min_len_shorter": 1}, "K6_type_conflict": {"minority_share": 0.2},
                               "K5_pending_majority": {"pending_share": 0.3333, "min_pending": 2}, "K8_same_chapter_partial": {"min_len": 2}}
    assert c["rules"]["K7_boundary"] == "guide B2" and set(c["reading"]) == set(ORDER) | {"after"}
    # accepted like the source: the training chapters recomputed from the record (what evaluate and posthoc do)
    ids = [r["chapter_id"] for r in rows]
    assert build_train_ids(rec, s.chapters, root) == build_train_ids(src, s.chapters, root) == ids
    assert ids_sha256(build_train_ids(rec, s.chapters, root)) == rec["split"]["train_ids_sha256"]
    new_rows = [json.loads(l) for l in (build / "labels_T.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["chapter_id"] for r in new_rows] == ids
    assert len(tagger.training_rows(new_rows, s, set(DEV), dev2_ids=frozenset())) == 6
    # metrics: counts only, no book text; keys.json: the strings
    met = json.loads((root / "runs" / "tbc" / "metrics.json").read_text(encoding="utf-8"))
    assert met == metrics and met["run_id"] == "tbc" and met["source"] == "tb" and met["chapters"] == 6
    assert met["before"]["spans"] == sum(len(r["spans"]) for r in rows) and met["after"]["spans"] == sum(len(r["spans"]) for r in new_rows)
    assert met["before"]["uncertain_by_reason"] == {"teacher_pending": 11}
    assert set(met["after"]["uncertain_by_reason"]) == {"teacher_pending", "K7_overlap", "K6_type_conflict", "K5_pending_majority", "K8_same_chapter_partial"}
    assert met["spans_removed_from_positives"] == met["before"]["spans"] - met["after"]["spans"]
    assert set(met["rules"]) == set(ORDER) and set(met["distinct_strings_affected"]) == set(ORDER)
    for name in ("config.json", "metrics.json"):
        assert not set((root / "runs" / "tbc" / name).read_text(encoding="utf-8")) & BOOK
    k = json.loads((root / "data" / "analysis" / "tbc" / "keys.json").read_text(encoding="utf-8"))
    assert k["run_id"] == "tbc" and k["K7_boundary"] == keys["K7_boundary"] and set(k["K7_boundary"][0]["longer"]) & BOOK
    with pytest.raises(SystemExit, match="never overwritten"):
        run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code=code)


def test_a_label_file_that_differs_from_the_record_stops_the_run(repo):
    root, s, rows, _ = repo
    t = root / "data" / "weak_labels" / "tb" / "labels_T.jsonl"
    original = t.read_bytes()
    t.write_bytes(original + b"\n")
    with pytest.raises(SystemExit, match="sha256"):
        run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code={"commit": "x", "dirty": False})
    assert not (root / "data" / "weak_labels" / "tbc").exists() and not (root / "runs" / "tbc").exists()
    t.write_bytes(original)
    w = root / "data" / "weak_labels" / "tb" / "labels_W.jsonl"
    w.write_bytes(w.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="labels_W.jsonl differs"):
        run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code={"commit": "x", "dirty": False})
    # a record of another kind, or of another snapshot, is refused too
    rec_file = root / "runs" / "tb" / "config.json"
    rec = json.loads(rec_file.read_text(encoding="utf-8"))
    rec_file.write_text(json.dumps({**rec, "kind": "weak labels"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a teacher labels build"):
        run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code={"commit": "x", "dirty": False})
    rec_file.write_text(json.dumps({**rec, "snapshot": {"id": "other"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="snapshot"):
        run_build(CFG, "configs/consistency.json", "tb", "tbc", s, root, code={"commit": "x", "dirty": False})


def test_main_needs_committed_code_unless_out_is_given(repo, monkeypatch):
    root, s, rows, _ = repo
    monkeypatch.setattr(consistency, "REPO_ROOT", root)
    monkeypatch.setattr(consistency, "load_snapshot", lambda sid, **kw: s)
    monkeypatch.setattr(consistency, "git_state", lambda: {"commit": "abc", "dirty": True})
    config = root / "configs" / "consistency.json"
    config.write_text(json.dumps(CFG), encoding="utf-8")
    with pytest.raises(SystemExit, match="uncommitted"):
        consistency.main([str(config)])
    assert not (root / "runs" / "tbc").exists()
    with pytest.raises(SystemExit, match="no build with target"):
        consistency.main([str(config), "--only", "nope", "--out", str(root / "scratch")])
    consistency.main([str(config), "--only", "tbc", "--out", str(root / "scratch")])
    out = root / "scratch"
    assert (out / "runs" / "tbc" / "metrics.json").exists() and (out / "data" / "weak_labels" / "tbc" / "labels_T.jsonl").exists()
    assert (out / "data" / "analysis" / "tbc" / "keys.json").exists() and not (root / "runs" / "tbc").exists()
    rec = json.loads((out / "runs" / "tbc" / "config.json").read_text(encoding="utf-8"))
    assert rec["consistency"]["scratch_out"] == str(out.resolve()) and rec["consistency"]["code"]["dirty"] is True
    assert rec["code"] == {"commit": "abc", "dirty": True}  # a scratch build says it was made from dirty code (evaluate would refuse it)
    assert rec["consistency"]["config_sha256"] == sha256(config) and rec["labels"]["T"]["file"] == "data/weak_labels/tbc/labels_T.jsonl"
    # clean code: the real layout is written
    monkeypatch.setattr(consistency, "git_state", lambda: {"commit": "abc", "dirty": False})
    consistency.main([str(config)])
    assert (root / "runs" / "tbc" / "metrics.json").exists()
    assert sha256(root / "data" / "weak_labels" / "tbc" / "labels_T.jsonl") == sha256(out / "data" / "weak_labels" / "tbc" / "labels_T.jsonl")


def test_the_module_reads_no_answer_file_and_no_system_output():
    """Only the training text and the teacher's own rows: no answer path, no predictions file (source-level, as
    tests/test_teacher.py checks the labels builder)."""
    source = inspect.getsource(consistency).lower()
    assert "gold" not in source and "predictions" not in source
    assert "data/analysis" in source  # the strings go there, never into runs/
