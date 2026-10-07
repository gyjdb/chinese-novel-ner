import hashlib

import pytest

from novel_ner.adjudicate import FIXED_SETS, NEW_SETS, SETS, apply_knowledge, empty_export, load_knowledge, merge
from novel_ner.gold import check
from novel_ner.snapshot import Chapter, Snapshot

TEXT = "林小雨去了灰塔城，见到回音铃。"  # 林小雨 [0,3) 灰塔城 [5,8) 回音铃 [11,14)


def span(start, end, type_, pending=False, **extra):
    return {"start": start, "end": end, "type": None if pending else type_, "pending": pending,
            "stuck": [], "candidates": "", "doubt": "", **extra}


def test_merge_accepts_agreement_and_leaves_the_rest_pending():
    a = [span(0, 3, "person"), span(5, 8, "location"), span(11, 14, "item")]
    b = [span(0, 3, "person"), span(5, 8, "organization")]
    out = merge(TEXT, a, b)
    assert [(x["start"], x["end"], x["type"], x["pending"]) for x in out] == [
        (0, 3, "person", False), (5, 8, None, True), (11, 14, None, True)]
    assert out[1]["stuck"] == ["type"] and "A：【灰塔城】地点" in out[1]["candidates"]
    assert out[2]["stuck"] == ["qualification"] and out[2]["candidates"].endswith("B：不标")


def test_merge_joins_overlapping_readings_into_one_pending_mark():
    out = merge(TEXT, [span(5, 8, "location")], [span(5, 7, "location")])
    assert [(x["start"], x["end"], x["pending"]) for x in out] == [(5, 8, True)]
    assert "boundary" in out[0]["stuck"]


def test_merge_keeps_gpt_pending_marks_pending_even_when_both_agree():
    p = span(11, 14, None, pending=True, stuck=["type"], candidates="item / trait", doubt="需要别章证据")
    out = merge(TEXT, [p], [dict(p)])
    assert out[0]["pending"] and out[0]["stuck"] == ["type"] and "需要别章证据" in out[0]["doubt"]


def snap():
    chapters = {"ch0001": Chapter("ch0001", 1, [("p0001", TEXT)], "", "h1"),
                "ch0002": Chapter("ch0002", 2, [("p0001", "店主说回音铃是一件道具。")], "", "h2")}
    return Snapshot("s", None, {}, chapters, [], [])


def test_knowledge_settles_pending_marks_and_flags_skipped_names():
    anns = merge(TEXT, [span(0, 3, "person"), span(11, 14, None, pending=True, stuck=["type"])],
                 [span(0, 3, "person")])
    knowledge = [{"name": "回音铃", "type": "item", "chapter": 2, "note": "道具"},
                 {"name": "灰塔城", "type": "location", "chapter": None, "note": ""}]
    log = apply_knowledge(anns, TEXT, knowledge, snap(), "ch0001")
    bell = next(a for a in anns if a["start"] == 11)
    assert bell["type"] == "item" and not bell["pending"] and bell["cross_chapter"]
    assert bell["evidence"][0]["chapter_id"] == "ch0002" and "回音铃" in bell["evidence"][0]["quote"]
    assert not next(a for a in anns if a["start"] == 0).get("cross_chapter")  # GPT agreed: untouched
    town = next(a for a in anns if a["start"] == 5)  # GPT left it out: the user has to look
    assert town["pending"] and town["stuck"] == ["qualification"]
    assert [entry[0] for entry in log] == ["settled", "unannotated"]


def test_knowledge_sheet_parsing(tmp_path):
    f = tmp_path / "k.tsv"
    f.write_text("名字\t类型\t证据章\t备注\n纸手机\t物品\tch0058\t纸役师做的\n\n回音铃\titem\t\t\n", encoding="utf-8")
    rows = load_knowledge(f)
    assert rows == [{"name": "纸手机", "type": "item", "chapter": 58, "note": "纸役师做的"},
                    {"name": "回音铃", "type": "item", "chapter": None, "note": ""}]
    f.write_text("名字\t类型\n纸手机\t工具\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge(f)


def test_knowledge_without_evidence_chapter_is_not_cross_chapter_and_none_removes_pending():
    anns = merge(TEXT, [span(0, 3, None, pending=True, stuck=["qualification"]), span(11, 14, None, pending=True)],
                 [span(0, 3, None, pending=True), span(11, 14, "item")])
    knowledge = [{"name": "林小雨", "type": "person", "chapter": None, "note": ""},
                 {"name": "回音铃", "type": "none", "chapter": None, "note": ""}]
    log = apply_knowledge(anns, TEXT, knowledge, snap(), "ch0001")
    rain = next(a for a in anns if a["start"] == 0)
    assert rain["type"] == "person" and not rain["pending"] and rain["cross_chapter"] is False
    assert all(a["start"] != 11 for a in anns)  # the pending mark on 回音铃 is gone
    assert sorted(entry[0] for entry in log) == ["settled", "skipped"]


TEXT2 = "徐获请来冬先生作证，冬先生点头。"  # 冬先生 at [4,7) and [10,13)


def test_sheet_widens_inner_readings_to_the_decided_name():
    # first 冬: both passes agreed on 冬 person (B1 reading); second 冬: one pass item, one person (pending)
    anns = merge(TEXT2, [span(4, 5, "person"), span(10, 11, "item")], [span(4, 5, "person"), span(10, 11, "person")])
    log = apply_knowledge(anns, TEXT2, [{"name": "冬先生", "type": "person", "chapter": None, "note": ""}], snap(), "ch0001")
    assert [(a["start"], a["end"], a["type"], a["pending"]) for a in anns] == [(4, 7, "person", False), (10, 13, "person", False)]
    assert [entry[0] for entry in log] == ["extended", "settled"]


def test_sheet_conflicts_and_crossings_go_to_the_user():
    anns = merge(TEXT, [span(0, 3, "location"), span(3, 6, "location")], [span(0, 3, "location"), span(3, 6, "location")])
    knowledge = [{"name": "林小雨", "type": "person", "chapter": None, "note": ""},   # GPT read another type
                 {"name": "灰塔城", "type": "location", "chapter": None, "note": ""}]  # GPT's 去了灰 crosses its edge
    log = apply_knowledge(anns, TEXT, knowledge, snap(), "ch0001")
    rain = next(a for a in anns if a["start"] == 0)
    assert rain["pending"] and rain["stuck"] == ["type"] and "GPT 两遍：【林小雨】地点" in rain["candidates"]
    town = next(a for a in anns if a["start"] == 3)  # one pending mark over 去了灰塔城, GPT's type kept in the candidates
    assert town["pending"] and town["end"] == 8 and town["stuck"] == ["boundary"] and "【去了灰】地点" in town["candidates"]
    assert [entry[0] for entry in log] == ["type_conflict", "overlap"]


def test_sheet_says_not_a_name_but_gpt_was_sure():
    anns = merge(TEXT, [span(0, 3, "person")], [span(0, 3, "person")])
    log = apply_knowledge(anns, TEXT, [{"name": "林小雨", "type": "none", "chapter": None, "note": ""}], snap(), "ch0001")
    assert anns[0]["pending"] and anns[0]["stuck"] == ["qualification"] and log[0][0] == "type_conflict"


def test_a_longer_name_gpt_was_sure_of_is_kept():
    text = "独眼杰克走进来，独眼男人跟着。"
    anns = merge(text, [span(0, 4, "person")], [span(0, 4, "person")])
    log = apply_knowledge(anns, text, [{"name": "独眼", "type": "none", "chapter": None, "note": ""}], snap(), "ch0001")
    assert [(a["start"], a["end"], a["type"], a["pending"]) for a in anns] == [(0, 4, "person", False)]
    assert [entry[0] for entry in log] == ["contained"]


def test_sheet_rows_never_settle_each_other_and_order_does_not_create_answers():
    rows = lambda *pairs: [{"name": n, "type": t, "chapter": None, "note": ""} for n, t in pairs]
    for order in (rows(("冬", "person"), ("冬先生", "person")), rows(("冬先生", "person"), ("冬", "person"))):
        anns = merge(TEXT2, [], [])
        apply_knowledge(anns, TEXT2, order, snap(), "ch0001")
        assert anns and all(a["pending"] for a in anns)  # GPT said nothing: only pending marks, whatever the order
    anns = merge(TEXT2, [span(4, 5, "person")], [span(4, 5, "person")])
    apply_knowledge(anns, TEXT2, rows(("冬先生", "person"), ("先生", "none")), snap(), "ch0001")
    assert (4, 7, "person", False) in [(a["start"], a["end"], a["type"], a["pending"]) for a in anns]  # not reopened


def test_sheet_with_a_name_twice_is_refused(tmp_path):
    f = tmp_path / "k.tsv"
    f.write_text("名字\t类型\n画女\titem\n画女\tperson\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge(f)


def test_evidence_in_the_same_chapter_is_not_cross_chapter():
    anns = merge(TEXT, [span(11, 14, None, pending=True)], [span(11, 14, None, pending=True)])
    apply_knowledge(anns, TEXT, [{"name": "回音铃", "type": "item", "chapter": 1, "note": ""}], snap(), "ch0001")
    bell = next(a for a in anns if a["start"] == 11)
    assert bell["type"] == "item" and bell["cross_chapter"] is False and bell["evidence"] == []



def _result(text, a, b, rows):
    anns = merge(text, a, b)
    apply_knowledge(anns, text, rows, snap(), "ch0001")
    return [(x["start"], x["end"], x["type"], x["pending"]) for x in anns]


def test_sheet_line_order_does_not_change_the_result_with_gpt_readings_present():
    row = lambda n, t: {"name": n, "type": t, "chapter": None, "note": ""}
    cases = [
        (TEXT2, [span(4, 5, "person")], [span(4, 5, None, pending=True)], [row("冬", "person"), row("冬先生", "person")]),
        (TEXT, [span(11, 13, "item")], [span(11, 13, None, pending=True)], [row("回音", "none"), row("回音铃", "item")]),
        (TEXT2, [span(4, 5, "person")], [span(4, 5, "person")], [row("冬", "item"), row("冬先生", "person")]),
    ]
    for text, a, b, rows in cases:
        assert _result(text, a, b, rows) == _result(text, a, b, list(reversed(rows)))
    assert (4, 7, "person", False) in _result(*cases[0])  # the longer name decides


def test_a_replaced_gpt_pending_mark_keeps_its_doubt_and_step():
    # GPT's pending mark 来冬 [3,5) crosses the sheet name 冬先生 [4,7): one pending mark over 来冬先生 keeps GPT's doubt
    a = [span(3, 5, "person")]
    b = [span(3, 5, None, pending=True, stuck=["qualification"], doubt="可能只是称呼")]
    anns = merge(TEXT2, a, b)
    apply_knowledge(anns, TEXT2, [{"name": "冬先生", "type": "person", "chapter": None, "note": ""}], snap(), "ch0001")
    mark = next(x for x in anns if x["start"] == 3)
    assert mark["pending"] and mark["end"] == 7 and "可能只是称呼" in mark["doubt"]
    assert mark["stuck"] == ["qualification", "boundary"]



def test_a_typed_row_crossing_a_bu_biao_row_goes_to_the_user():
    row = lambda n, t: {"name": n, "type": t, "chapter": None, "note": ""}
    for text, rows, gpt in (("甲乙丙", [row("甲乙", "person"), row("乙丙", "none")], span(0, 1, "person")),
                            ("乙甲丙", [row("乙甲", "person"), row("甲丙", "none")], span(0, 1, "person"))):
        out = _result(text, [gpt], [dict(gpt)], rows)
        assert all(p for *_, p in out) and out, (text, out)
    text = "白狐狸精来了"
    a = [span(0, 1, "person"), span(1, 4, None, pending=True)]
    out = _result(text, a, [dict(x) for x in a], [row("白狐", "person"), row("狐狸精", "none")])
    assert out == [(0, 4, None, True)]


def test_a_shorter_typed_row_inside_a_longer_bu_biao_row_is_left_alone():
    text = "他拿出一个小元宝。"
    rows = [{"name": "小元宝", "type": "none", "chapter": None, "note": ""}, {"name": "小元", "type": "person", "chapter": None, "note": ""}]
    assert _result(text, [], [], rows) == []
    anns = merge(text, [span(5, 8, None, pending=True)], [span(5, 8, None, pending=True)])
    log = apply_knowledge(anns, text, rows, snap(), "ch0001")
    assert anns == [] and [entry[0] for entry in log] == ["skipped"]


def test_names_that_overlap_themselves_stay_out_of_the_sheet(tmp_path):
    f = tmp_path / "k.tsv"
    f.write_text("名字\t类型\n乐乐\tperson\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge(f)
    with pytest.raises(ValueError):
        apply_knowledge([], "她喊乐乐乐。", [{"name": "乐乐", "type": "person", "chapter": None, "note": ""}], snap(), "ch0001")


def test_a_crossing_row_shows_what_gpt_said_before_another_row_settled_it():
    rows = [{"name": "乙甲", "type": "person", "chapter": None, "note": ""}, {"name": "甲丙", "type": "item", "chapter": None, "note": ""}]
    anns = merge("乙甲丙", [span(0, 1, "person")], [span(0, 1, "person")])
    apply_knowledge(anns, "乙甲丙", rows, snap(), "ch0001")
    assert [(a["start"], a["end"], a["pending"]) for a in anns] == [(0, 3, True)]
    assert "【乙】人物" in anns[0]["candidates"] and not any(k.startswith("_") for k in anns[0])


def test_empty_export_is_the_pages_blank_state_for_every_listed_chapter():
    assert SETS == ("test", "dev", "dev2", "test2") and NEW_SETS == ("dev2", "test2") and set(NEW_SETS) < set(FIXED_SETS)
    out = empty_export("dev2", snap(), ["ch0002", "ch0001"], "1.1", "T", "configs/annotate_dev2.json")
    assert (out["format"], out["set"], out["snapshot_id"], out["guide_version"], out["current"], out["next_id"]) == (
        "novel-ner-annotation/1", "dev2", "s", "1.1", "ch0002", 1)
    assert list(out["chapters"]) == ["ch0002", "ch0001"] and out["precedents"] == [] and out["updated_at"] is None
    assert out["chapters"]["ch0001"] == {
        "content_hash": "h1", "text_sha256": hashlib.sha256(TEXT.encode("utf-8")).hexdigest(), "status": "todo",
        "provenance": None, "checks": {"occurrences": False, "full_read": False, "records": False},
        "seconds": 0, "annotations": [], "skipped": [], "note": ""}
    assert out["revisions"] == [{"at": "T", "by": "claude", "what": "empty dev2 export: the 2 chapters of configs/annotate_dev2.json "
                                                                  "in that order, no annotations; the base of the set's first prefill"}]
    assert check(out, snap())[0] == []
    for chapters in ([], ["ch0001", "ch0001"], ["ch0001", "ch0009"]):
        with pytest.raises(SystemExit):
            empty_export("test2", snap(), chapters, "1.1", "T", "x")
