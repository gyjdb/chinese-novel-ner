import hashlib
import json

import novel_ner.annotate as annotate
from novel_ner.annotate import PLACEHOLDER, TEMPLATE, chapter_order, render
import pytest

import novel_ner.gold as gold
from novel_ner.gold import FORMAT, SETS, TEST_SETS, check, own_set_path, questions, revise
from novel_ner.snapshot import REPO_ROOT
from novel_ner.score import gold_spans
from novel_ner.snapshot import Chapter, Snapshot

PARAS = [("p0001", "徐获拉开门。"), ("p0002", "唐医生到了汀城。")]  # chapter text: "徐获拉开门。\n唐医生到了汀城。"


def fake_snapshot():
    chapters = {
        "ch0001": Chapter("ch0001", 1, PARAS, "芳香", "h1"),
        "ch0002": Chapter("ch0002", 2, [("p0001", "汀城下雨了，徐获没带伞。")], "雨", "h2"),
    }
    return Snapshot("snap-test", None, {}, chapters, [], [])


def export(annotations, status="doing", checks=True, provenance="manual"):
    text = "\n".join(t for _, t in PARAS)
    return {
        "format": FORMAT, "set": "dev", "snapshot_id": "snap-test", "guide_version": "1.0",
        "chapters": {"ch0001": {
            "content_hash": "h1", "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "status": status, "seconds": 600, "provenance": provenance,
            "checks": {"occurrences": checks, "full_read": checks, "records": checks},
            "annotations": annotations,
        }},
        "precedents": [],
    }


def span(start, end, text, type_="person", **extra):
    return {"start": start, "end": end, "text": text, "type": type_, "pending": False, **extra}


def test_clean_file_passes_and_is_summarised():
    anns = [span(0, 2, "徐获"), span(7, 8, "唐"), span(12, 14, "汀城", "location")]
    errors, warnings, summary = check(export(anns, status="done"), fake_snapshot())
    assert errors == [] and warnings == []
    c = summary["chapters"]["ch0001"]
    assert c["spans"] == 3 and c["by_type"]["person"] == 2 and c["minutes"] == 10.0
    assert summary["done_chapters"] == 1 and summary["minutes_per_hand_annotated_chapter"] == 10.0


def test_wrong_text_overlap_type_and_paragraph_break_are_errors():
    anns = [span(0, 2, "徐荻"), span(1, 3, "获拉"), span(5, 9, "。\n唐医"), span(12, 14, "汀城", "place")]
    errors, _, _ = check(export(anns), fake_snapshot())
    joined = "\n".join(errors)
    assert "does not match" in joined and "overlaps" in joined
    assert "crosses a paragraph break" in joined and "not one of the six" in joined


def test_pending_needs_a_reason_once_the_chapter_is_done():
    pending = {"start": 7, "end": 10, "text": "唐医生", "type": None, "pending": True, "candidates": "", "doubt": ""}
    errors, warnings, _ = check(export([pending]), fake_snapshot())
    assert errors == [] and any("pending" in w for w in warnings)
    errors, _, _ = check(export([pending], status="done"), fake_snapshot())
    assert any("pending" in e for e in errors)


def test_evidence_must_point_at_real_text():
    good = {"chapter_id": "ch0002", "paragraph_id": "p0001", "quote": "汀城下雨了"}
    bad = {"chapter_id": "ch0002", "paragraph_id": "p0001", "quote": "不存在的话"}
    ok = check(export([span(12, 14, "汀城", "location", cross_chapter=True, evidence=[good])]), fake_snapshot())
    assert ok[0] == []
    errors, _, _ = check(export([span(12, 14, "汀城", "location", cross_chapter=True, evidence=[bad])]), fake_snapshot())
    assert any("evidence" in e for e in errors)


def test_provenance_rules():
    data = export([span(0, 2, "徐获")], status="done", provenance=None)
    _, warnings, _ = check(data, fake_snapshot())
    assert any("provenance" in w for w in warnings)
    data["chapters"]["ch0001"]["provenance"] = "model_draft_reviewed"
    errors, warnings, summary = check(data, fake_snapshot())
    assert errors == [] and summary["chapters"]["ch0001"]["provenance"] == "model_draft_reviewed"
    data["set"] = "test"  # test chapters must say how they were made
    data["chapters"]["ch0001"]["provenance"] = None
    assert any("provenance" in e for e in check(data, fake_snapshot())[0])
    data["chapters"]["ch0001"]["provenance"] = "model_double_adjudicated"
    assert check(data, fake_snapshot())[0] == []
    data["chapters"]["ch0001"]["provenance"] = "someone"
    assert any("unknown provenance" in e for e in check(data, fake_snapshot())[0])


def test_questions_quote_the_paragraph_and_the_doubt():
    pending = {"start": 7, "end": 10, "text": "唐医生", "type": None, "pending": True,
               "candidates": "唐 / 唐医生", "doubt": "医生是不是称谓"}
    noted = span(12, 14, "汀城", "location", note="分区还是城市？")
    plain = span(0, 2, "徐获")
    data = export([plain, pending, noted])
    data["exported_at"] = "2026-10-01T00:00:00Z"
    md = questions(data, fake_snapshot())
    assert "「唐医生」：待审（p0002，第 7–10 字）" in md
    assert "原文：【唐医生】到了汀城。" in md and "疑点：医生是不是称谓" in md
    assert "原文：唐医生到了【汀城】。" in md and "备注：分区还是城市？" in md
    assert "「徐获」" not in md  # plain annotations are left out


def test_pending_states_where_it_is_stuck_from_guide_v11():
    pending = {"start": 7, "end": 10, "text": "唐医生", "type": None, "pending": True, "doubt": "医生是不是称谓"}
    data = export([pending])
    assert not any("stuck" in w for w in check(data, fake_snapshot())[1])  # v1.0 did not ask for it
    data["guide_version"] = "1.1"
    assert any("stuck" in w for w in check(data, fake_snapshot())[1])
    pending["stuck"] = ["boundary"]
    assert check(data, fake_snapshot())[1] == []
    assert "卡在：名称边界" in questions(data, fake_snapshot())
    pending["stuck"] = ["mood"]
    assert any("stuck" in e for e in check(data, fake_snapshot())[0])


def test_changed_text_is_refused():
    data = export([span(0, 2, "徐获")])
    data["chapters"]["ch0001"]["content_hash"] = "old"
    errors, _, _ = check(data, fake_snapshot())
    assert any("content_hash" in e for e in errors)


def test_page_payload_cannot_close_the_script_tag():
    assert TEMPLATE.read_text(encoding="utf-8").count(PLACEHOLDER) == 1
    html = render({"chapters": [], "note": "</script><script>alert(1)</script>"})
    payload = html.split('<script id="page-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(payload)["note"] == "</script><script>alert(1)</script>"



def test_revise_settles_pending_marks_and_records_unscored_ranges():
    pending = {"start": 7, "end": 8, "text": "唐", "type": None, "pending": True, "candidates": "人物/不标",
               "doubt": "拿不准", "stuck": ["qualification"], "note": "我不知道"}
    base = export([span(0, 2, "徐获"), pending], status="done")
    decisions = {"decided_by": "user", "decided_at": "2026-10-02",
                 "settle_pending": [{"text": "唐", "chapters": ["ch0001"], "type": "person", "expect": 1}],
                 "unscored": [{"text": "汀城", "reason": "拿不准", "expect": 1}]}
    out, log = revise(base, decisions, fake_snapshot(), now="T")
    a = out["chapters"]["ch0001"]["annotations"][1]
    assert (a["pending"], a["type"]) == (False, "person") and "user 2026-10-02" in a["note"]
    assert out["chapters"]["ch0001"]["unscored"] == [{"start": 12, "end": 14, "text": "汀城", "reason": "拿不准",
                                                      "by": "user 2026-10-02", "at": "T"}]
    assert base["chapters"]["ch0001"]["annotations"][1]["pending"] and "unscored" not in base["chapters"]["ch0001"]
    assert out["revisions"][-1]["by"] == "claude"
    errors, _, summary = check(out, fake_snapshot())
    assert errors == [] and summary["chapters"]["ch0001"]["unscored"] == 1
    assert gold_spans(out, ["ch0001"])["ch0001"]["unscored"] == [(12, 14)]


def test_revise_stops_when_a_decision_touches_an_unexpected_number_of_places():
    base = export([span(0, 2, "徐获")], status="done")
    for decisions in ({"settle_pending": [{"text": "唐", "chapters": ["ch0001"], "type": "person", "expect": 1}]},
                      {"unscored": [{"text": "汀城", "reason": "拿不准", "expect": 2}]},
                      {"unscored": [{"text": "汀城", "reason": "", "expect": 1}]}):
        with pytest.raises(ValueError):
            revise(base, {"decided_by": "user", "decided_at": "d", **decisions}, fake_snapshot())


def test_unscored_ranges_must_match_the_text_and_give_a_reason():
    base = export([], status="done")
    base["chapters"]["ch0001"]["unscored"] = [{"start": 12, "end": 14, "text": "汀州", "reason": "x"},
                                              {"start": 0, "end": 2, "text": "徐获", "reason": ""}]
    errors, _, _ = check(base, fake_snapshot())
    assert any("does not match" in e for e in errors) and any("say why" in e for e in errors)


def test_revise_places_replace_what_is_inside_and_stop_at_crossing_marks():
    base = export([span(0, 2, "徐获"), span(7, 8, "唐"), span(12, 14, "汀城", "location")], status="done")
    decisions = {"decided_by": "user", "decided_at": "2026-10-03", "places": [
        {"chapter_id": "ch0001", "start": 7, "end": 10, "text": "唐医生", "reading": "person", "reason": "整串是称呼"},
        {"chapter_id": "ch0001", "start": 12, "end": 14, "text": "汀城", "reading": "none", "reason": "不标"},
        {"chapter_id": "ch0001", "start": 0, "end": 2, "text": "徐获", "reading": "unscored", "reason": "拿不准"}]}
    out, log = revise(base, decisions, fake_snapshot(), now="T")
    anns = out["chapters"]["ch0001"]["annotations"]
    assert [(a["start"], a["end"], a["text"], a["type"]) for a in anns] == [(7, 10, "唐医生", "person")]
    assert "整串是称呼" in anns[0]["note"] and out["chapters"]["ch0001"]["unscored"][0]["text"] == "徐获"
    assert len(log) == 3 and "唐:person -> person" in log[0]
    errors, _, _ = check(out, fake_snapshot())
    assert errors == []
    for bad in ({"chapter_id": "ch0001", "start": 7, "end": 9, "text": "唐医", "reading": "person", "reason": "x"},
                {"chapter_id": "ch0001", "start": 0, "end": 15, "text": "。", "reading": "person", "reason": "x"},
                {"chapter_id": "ch0001", "start": 12, "end": 14, "text": "汀城", "reading": "place", "reason": "x"},
                {"chapter_id": "ch0001", "start": 12, "end": 14, "text": "汀城", "reading": "none", "reason": ""}):
        with pytest.raises(ValueError):
            revise(export([span(7, 10, "唐医生")], status="done"), {"decided_by": "u", "decided_at": "d", "places": [bad]},
                   fake_snapshot())


def test_revise_place_keeps_the_mark_id_and_evidence_and_records_every_place():
    ev = [{"chapter_id": "ch0002", "paragraph_id": "p0001", "quote": "汀城"}]
    base = export([span(12, 14, "汀城", "organization", id="a1", cross_chapter=True, evidence=ev),
                   span(0, 2, "徐获")], status="done")
    base["chapters"]["ch0001"]["unscored"] = [{"start": 7, "end": 8, "text": "唐", "reason": "拿不准", "by": "user d0"}]
    decisions = {"decided_by": "user", "decided_at": "d", "places": [
        {"chapter_id": "ch0001", "start": 12, "end": 14, "text": "汀城", "reading": "location", "reason": "是地点"},
        {"chapter_id": "ch0001", "start": 7, "end": 8, "text": "唐", "reading": "none", "reason": "泛称",
         "by": "astra d", "decider": "reviewer"}]}
    out, log = revise(base, decisions, fake_snapshot(), now="T")
    chap = out["chapters"]["ch0001"]
    m = next(a for a in chap["annotations"] if a["start"] == 12)
    assert (m["id"], m["type"], m["cross_chapter"], m["evidence"], m["decider"]) == ("a1", "location", True, ev, "user")
    assert "type organization -> location" in log[0] and "唐:不计分(user d0)" in log[1] and "reviewer" in log[1]
    assert "unscored" not in chap or chap["unscored"] == []
    assert [(p["text"], p["reading"], p["decider"]) for p in chap["decided_places"]] == [("汀城", "location", "user"), ("唐", "none", "reviewer")]
    assert "1 places decided by astra d" in out["revisions"][-1]["what"]
    errors, _, _ = check(out, fake_snapshot())
    assert errors == []


def test_silver_standard_sets_require_provenance_on_annotated_chapters():
    assert SETS == ("test", "dev", "dev2", "test2") and TEST_SETS == ("test", "test2", "dev2")
    for name in ("test2", "dev2"):
        data = export([span(0, 2, "徐获")], status="done", provenance=None)
        data["set"] = name
        assert any("must record their provenance" in e for e in check(data, fake_snapshot())[0])
        data["chapters"]["ch0001"]["provenance"] = "model_double_adjudicated"
        assert check(data, fake_snapshot())[0] == []
        blank = export([], status="todo", provenance=None)  # an untouched chapter of a new set's empty base
        blank["set"] = name
        assert check(blank, fake_snapshot())[0] == []
    data = export([span(0, 2, "徐获")], status="done", provenance=None)  # dev keeps its warning
    assert check(data, fake_snapshot())[0] == [] and any("provenance" in w for w in check(data, fake_snapshot())[1])


def test_a_file_under_another_sets_folder_is_refused(tmp_path):
    refused = [("data/gold/test2/incoming/prefilled_test2_b01.json", "dev2"), ("data/gold/dev2/x.json", "test"),
               (tmp_path / "data" / "gold" / "dev" / "y.json", "dev2"), (str(tmp_path / "data" / "gold" / "test" / "y.json"), "test2")]
    for path, set_name in refused:
        with pytest.raises(SystemExit, match="lies under data/gold/"):
            own_set_path(path, set_name, root=tmp_path)
    allowed = [("data/gold/dev2/incoming/prefilled_dev2_b01.json", "dev2"), ("configs/x.json", "test2"),
               (tmp_path / "data" / "gold" / "test2" / "z.json", "test2"), (tmp_path / "elsewhere" / "data" / "gold" / "dev" / "z.json", "test2")]
    for path, set_name in allowed:
        assert own_set_path(path, set_name, root=tmp_path) == path


def test_gold_commands_take_the_set_and_refuse_another_sets_file(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(gold, "REPO_ROOT", tmp_path)
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"set": "test2"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gold", "check", str(f), "--set", "dev2"])
    with pytest.raises(SystemExit, match="is a test2 export, not dev2"):
        gold.main()
    under = tmp_path / "data" / "gold" / "test2" / "x.json"
    under.parent.mkdir(parents=True)
    under.write_text(json.dumps({"set": "dev2"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gold", "check", str(under), "--set", "dev2"])
    with pytest.raises(SystemExit, match="lies under data/gold/test2"):
        gold.main()
    with pytest.raises(SystemExit):  # an unknown set
        monkeypatch.setattr(sys, "argv", ["gold", "check", str(f), "--set", "dev3"])
        gold.main()


def test_chapter_order_takes_a_sampling_configs_list_as_it_stands(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    listed = tmp_path / "configs" / "dev2_x.json"
    good = {"set": "dev2", "n": 3, "chapters": ["ch0809", "ch0981", "ch0825"]}
    listed.write_text(json.dumps(good), encoding="utf-8")
    cfg = {"set": "dev2", "chapters_from_config": "configs/dev2_x.json"}
    assert chapter_order(cfg) == ["ch0809", "ch0981", "ch0825"]
    listed.write_text(json.dumps({"chapters": ["ch0809", "ch0981", "ch0825"]}), encoding="utf-8")  # no set, no n: fine
    assert chapter_order(cfg) == ["ch0809", "ch0981", "ch0825"]
    for bad in ({**good, "n": 2}, {**good, "set": "test2"}, {"chapters": ["ch0809", "ch0809"]}, {"chapters": []}):
        listed.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            chapter_order(cfg)
    order = tmp_path / "configs" / "o.json"  # the order path is as it was
    order.write_text(json.dumps({"order": [{"chapter_id": f"ch{n}"} for n in (1262, 1345, 1207)]}), encoding="utf-8")
    assert chapter_order({"set": "test2", "chapters_from_order": "configs/o.json", "n": 2}) == ["ch1262", "ch1345"]


def test_the_dev2_and_test2_page_configs_resolve_to_their_sets():
    dev2 = json.loads((REPO_ROOT / "configs" / "annotate_dev2.json").read_text(encoding="utf-8"))
    listed = json.loads((REPO_ROOT / "configs" / "dev2_781_990.json").read_text(encoding="utf-8"))
    assert (dev2["set"], dev2["snapshot"], dev2["guide_version"]) == ("dev2", "to0990-79dc77f8d354", "1.1")
    assert chapter_order(dev2) == listed["chapters"] and len(listed["chapters"]) == 25
    test2 = json.loads((REPO_ROOT / "configs" / "annotate_test2.json").read_text(encoding="utf-8"))
    seal = json.loads((REPO_ROOT / "configs" / "test2_seal.json").read_text(encoding="utf-8"))
    assert (test2["set"], test2["snapshot"], test2["guide_version"], test2["n"]) == ("test2", "to1350-4c4b1c91075e", "1.1", seal["n"])
    assert test2["chapters_from_order"] == seal["order"]
    ids = chapter_order(test2)
    assert len(ids) == 40 == len(set(ids)) and all(996 <= int(c[2:]) <= 1350 for c in ids)
    assert not set(ids) & set(chapter_order(dev2))
