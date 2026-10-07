from collections import Counter

import pytest

from novel_ner import preannotate
from novel_ner.preannotate import compare_chapter, f1, flag_chapter, groups, make_boxes, pick_showcase, title_after

FLAGS = {"vote": 2, "low_score": 0.9, "repeat_min_len": 2}
TITLES = sorted(["先生", "医生", "哥"], key=len, reverse=True)


def test_shown_boxes_need_two_seeds_and_never_overlap():
    s1 = [(0, 2, "person", 0.99), (5, 8, "item", 0.95)]
    s2 = [(0, 2, "person", 0.97), (5, 7, "item", 0.80)]
    s3 = [(0, 2, "person", 0.98), (5, 8, "item", 0.91)]
    shown, minority = make_boxes([s1, s2, s3], 2)
    assert [(b["start"], b["end"], b["type"], b["seeds"]) for b in shown] == [(0, 2, "person", [0, 1, 2]), (5, 8, "item", [0, 2])]
    assert shown[1]["score"] == 0.91  # the lowest among the seeds that output it
    assert [(m["start"], m["end"], m["seeds"]) for m in minority] == [(5, 7, [1])]
    with pytest.raises(ValueError, match="overlap"):  # cannot happen with real seeds; a broken input is refused
        make_boxes([[(0, 3, "person", 1.0), (1, 4, "item", 1.0)]] * 2, 2)


def test_title_word_after_or_inside_a_person_box():
    text = "冬先生和唐医生见了徐获哥。"
    box = lambda s, e, t="person": {"start": s, "end": e, "type": t}
    assert title_after(text, box(0, 1), TITLES) == "先生"        # 【冬】先生
    assert title_after(text, box(0, 3), TITLES) == "先生"        # 【冬先生】 ends with the title
    assert title_after(text, box(4, 5), TITLES) == "医生"
    assert title_after(text, box(9, 11), TITLES) == "哥"
    assert title_after(text, box(4, 5, "item"), TITLES) is None  # only person boxes
    assert title_after("先生来了", box(0, 2), TITLES) is None     # the whole box is the title word


def test_flags_and_miss_candidates():
    text = "徐获拿出回音铃。\n徐获和白塔的冬先生。回音铃响了。"
    seen = {"徐获", "回音铃"}
    s_all = [(0, 2, "person", 0.99), (4, 7, "item", 0.99), (9, 11, "person", 0.99), (12, 14, "location", 0.95),
             (15, 16, "person", 0.95)]
    s3 = [(0, 2, "person", 0.99), (4, 7, "item", 0.99), (9, 11, "person", 0.99), (12, 14, "organization", 0.5),
          (15, 16, "person", 0.85), (19, 22, "person", 0.6)]
    dict_matches = [{"start": 4, "end": 7, "type": "item"}, {"start": 12, "end": 14, "type": "location"}]
    shown, minority, misses = flag_chapter(text, [s_all, s_all, s3], seen, TITLES, dict_matches, FLAGS)
    flags = {text[b["start"]:b["end"]] + str(b["start"]): b["flags"] for b in shown}
    assert flags["徐获0"] == [] and flags["徐获9"] == []
    assert flags["回音铃4"] == ["nonperson"]                        # a seen item: only its type is flagged
    assert flags["白塔12"] == ["seeds", "new", "nonperson"]         # seed 3 said organization (a one-seed box overlaps)
    assert flags["冬15"] == ["low", "new", "title"]                 # all three output it; seed 3's 0.85 is the lowest
    assert shown[-1]["title_word"] == "先生"
    # 回音铃 at 19 is unboxed: another occurrence of a shown text and a one-seed box (person) there
    assert [(m["start"], m["end"], m["kinds"], m["types"]) for m in misses] == [(19, 22, ["repeat", "minority"], ["item", "person"])]
    # the dictionary match at 4 and 12 is covered by shown boxes, so it is no candidate
    assert not any("dictionary" in m["kinds"] for m in misses)


def test_low_and_mixed():
    text = "白塔里有白塔。"
    a = [(0, 2, "location", 0.95), (4, 6, "organization", 0.95)]
    b = [(0, 2, "location", 0.80), (4, 6, "organization", 0.95)]
    shown, _, misses = flag_chapter(text, [a, b, a], {"白塔"}, TITLES, [], FLAGS)
    assert [b["flags"] for b in shown] == [["low", "nonperson", "mixed"], ["nonperson", "mixed"]]
    assert misses == []


def test_groups_route_book_knowledge_to_the_user():
    text = "徐获见了冬先生和白塔。冬先生走了。"
    s = [(0, 2, "person", 0.99), (4, 5, "person", 0.99), (8, 10, "location", 0.99), (11, 12, "person", 0.99)]
    shown, _, misses = flag_chapter(text, [s, s, s], {"徐获", "冬", "白塔"}, TITLES,
                                    [{"start": 14, "end": 16, "type": "item"}], FLAGS)
    g = {x["text"]: x for x in groups([("ch1", shown, misses)])}
    assert g["冬"]["ask_user"] and g["冬"]["flags"] == ["title"] and len(g["冬"]["boxes"]) == 2
    assert not g["白塔"]["ask_user"] and g["白塔"]["flags"] == ["nonperson"]
    assert not g["走了"]["ask_user"] and g["走了"]["misses"] == [("ch1", 14, 16)]  # a candidate only: review
    assert "徐获" not in g                                                       # nothing flagged
    assert list(g)[0] == "冬"                                                   # questions for the user first


def test_compare_counts_against_the_answers():
    shown = [{"start": 0, "end": 2, "type": "person", "flags": []},
             {"start": 4, "end": 5, "type": "person", "flags": ["title"]},     # answer is 4-7 (boundary)
             {"start": 10, "end": 12, "type": "item", "flags": ["nonperson"]},   # not an answer
             {"start": 20, "end": 22, "type": "person", "flags": []}]           # touches an unscored range
    misses = [{"start": 14, "end": 16, "kinds": ["dictionary"]}, {"start": 30, "end": 32, "kinds": ["repeat"]}]
    ref = {"spans": {(0, 2, "person"), (4, 7, "person"), (14, 16, "item"), (40, 42, "location"), (20, 22, "person")},
           "unscored": [(19, 23)]}
    c = compare_chapter(shown, misses, ref)
    assert (c["boxes"], c["boxes_wrong"], c["flagged"], c["flagged_wrong"], c["unflagged_wrong"]) == (3, 2, 2, 2, 0)
    assert c["boxes_touching_unscored"] == 1 and c["gold"] == 4 and c["gold_missed"] == 3
    assert (c["gold_missed_caught_by_flagged_box"], c["gold_missed_caught_by_candidate"], c["gold_missed_nothing_near"]) == (1, 1, 1)
    assert (c["candidates"], c["candidates_on_missed_answer"], c["cand_dictionary_on_missed_answer"]) == (2, 1, 1)
    assert c["flag_title_wrong"] == 1 and c["flag_nonperson_wrong"] == 1


def test_showcase_is_the_chapter_nearest_the_median():
    per = {"ch0801": Counter(boxes=10, boxes_wrong=0, gold_missed=0),   # F1 1.0
           "ch0802": Counter(boxes=10, boxes_wrong=2, gold_missed=2),   # 0.8
           "ch0803": Counter(boxes=10, boxes_wrong=5, gold_missed=5)}   # 0.5
    assert pick_showcase(per, {"ch0801": 0, "ch0802": 1, "ch0803": 9})[:2] == ("ch0802", 0.8)
    per["ch0804"] = Counter(boxes=10, boxes_wrong=2, gold_missed=2)   # tie on F1: more non-person answers wins
    assert pick_showcase(per, {"ch0801": 0, "ch0802": 1, "ch0803": 9, "ch0804": 3})[0] == "ch0804"
    assert f1(8, 2, 2) == 0.8 and f1(0, 0, 3) == 0.0


def test_render_keeps_the_payload_out_of_html(monkeypatch, tmp_path):
    template = tmp_path / "t.html"
    template.write_text("<script>const DATA = /*__DATA__*/;</script>", encoding="utf-8")
    monkeypatch.setattr(preannotate, "TEMPLATE", template)
    page = preannotate.render({"text": "</script><b>"})
    assert "</script><b>" not in page and page.count("</script>") == 1


def test_context_stays_in_its_paragraph():
    text = "甲乙丙丁戊\n己庚辛壬癸"
    assert preannotate.context(text, 8, 9, width=1) == "…庚【辛】壬…"
    assert preannotate.context(text, 6, 7, width=3) == "【己】庚辛壬…"
    assert preannotate.context(text, 3, 5, width=9) == "甲乙丙【丁戊】"


def test_showcase_tie_is_broken_by_the_rule_not_by_float_rounding():
    # F1 1.0, 0.9, 0.7, 0.5: the median 0.8 is exactly 0.1 from 0.9 and 0.7 (in floats 0.0999... vs 0.1000...)
    per = {"ch0801": Counter(boxes=10, boxes_wrong=0, gold_missed=0),
           "ch0802": Counter(boxes=9, boxes_wrong=0, gold_missed=2),     # 2*9/(18+2) = 0.9
           "ch0803": Counter(boxes=7, boxes_wrong=0, gold_missed=6),     # 14/20 = 0.7
           "ch0804": Counter(boxes=5, boxes_wrong=0, gold_missed=10)}    # 10/20 = 0.5
    best, value, median = pick_showcase(per, {"ch0801": 0, "ch0802": 0, "ch0803": 5, "ch0804": 0})
    assert (best, value, median) == ("ch0803", 0.7, 0.8)


def test_a_candidate_on_an_unscored_range_neither_counts_nor_catches():
    ref = {"spans": {(0, 2, "person")}, "unscored": [(2, 4)]}
    c = compare_chapter([], [{"start": 1, "end": 3, "kinds": ["minority"]}], ref)
    assert (c["candidates"], c["candidates_touching_unscored"], c["gold_missed"], c["gold_missed_caught"]) == (0, 1, 1, 0)
    assert c["gold_missed_nothing_near"] == 1
    box = compare_chapter([{"start": 1, "end": 3, "type": "person", "flags": ["low"]}], [], ref)
    assert (box["boxes_touching_unscored"], box["gold_missed_caught"]) == (1, 0)  # the same as a box there


def test_build_refuses_a_run_id_that_is_not_one_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(preannotate, "REPO_ROOT", tmp_path)
    for bad in ("../../leak", "a/b", "C:/x"):
        with pytest.raises(SystemExit, match="one folder name"):
            preannotate.build({}, "c.json", bad)
    assert not any(tmp_path.iterdir())
