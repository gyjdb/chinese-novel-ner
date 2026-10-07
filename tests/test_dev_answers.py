"""The dev answers flow on synthetic data: prefill --set dev (chained), review pending, package, decide, gold revise.
The same flow for the new sets dev2 and test2 (prefill without a base builds the empty base from the set's page config,
then chains) is at the end."""
import copy
import hashlib
import json

import pytest

import novel_ner.adjudicate as adj
import novel_ner.annotate as annotate
import novel_ner.review as rv
from novel_ner.gold import FORMAT, check, revise
from novel_ner.snapshot import Chapter, Snapshot

# ch0001: an earlier dev chapter, done; ch0002-ch0004: prefilled by dev_b01 (ch0002, ch0003) and dev_b02 (ch0004);
# ch0005: a dev chapter no package holds
PARAS = {
    "ch0001": [("p0001", "徐获拉开门。")],
    "ch0002": [("p0001", "林小雨去了灰塔城，见到回音铃。"), ("p0002", "章医生到了。")],  # 灰塔城 [5,8) 回音铃 [11,14) 章医生 [16,19)
    "ch0003": [("p0001", "回音铃又响了，黑肩章来了。")],  # 回音铃 [0,3) 黑肩章 [7,10)
    "ch0004": [("p0001", "冬先生请来灰塔城的人。")],  # 冬先生 [0,3) 灰塔城 [5,8) 人 [9,10)
    "ch0005": [("p0001", "雨停了。")],
}
SHEET = [{"name": "章医生", "type": "person", "chapter": None, "note": "固定叫法"}]


def snap():
    return Snapshot("s", None, {}, {cid: Chapter(cid, int(cid[2:]), ps, f"第{cid}", f"h{cid}") for cid, ps in PARAS.items()},
                    [], [])


def text_of(cid):
    return "\n".join(t for _, t in PARAS[cid])


def dev_base():
    chapters = {}
    for cid in PARAS:
        sha = hashlib.sha256(text_of(cid).encode("utf-8")).hexdigest()
        if cid == "ch0001":
            chapters[cid] = {"content_hash": f"h{cid}", "text_sha256": sha, "status": "done", "provenance": "manual",
                             "checks": {"occurrences": True, "full_read": True, "records": True}, "seconds": 600,
                             "done_at": "t0", "note": "", "skipped": [],
                             "annotations": [{"id": "a1", "start": 0, "end": 2, "text": "徐获", "type": "person",
                                              "pending": False, "note": "", "cross_chapter": False, "evidence": []}]}
        else:
            chapters[cid] = {"content_hash": f"h{cid}", "text_sha256": sha, "status": "todo", "seconds": 0, "note": "",
                             "skipped": [], "annotations": [],
                             "checks": {"occurrences": False, "full_read": False, "records": False}}
    return {"format": FORMAT, "set": "dev", "snapshot_id": "s", "guide_version": "1.1", "current": "ch0001",
            "next_id": 2, "chapters": chapters, "precedents": [], "revisions": []}


def sp(cid, s, e, typ, pending=False, **x):
    return {"start": s, "end": e, "text": text_of(cid)[s:e], "paragraph_id": "p0001", "type": None if pending else typ,
            "pending": pending, "stuck": x.get("stuck", []), "candidates": x.get("candidates", ""),
            "doubt": x.get("doubt", ""), "note": ""}


PASSES = {
    "dev_b01": {
        "A": {"ch0002": [sp("ch0002", 0, 3, "person"), sp("ch0002", 5, 8, "location"), sp("ch0002", 11, 14, "item")],
              "ch0003": [sp("ch0003", 0, 3, "item"), sp("ch0003", 7, 10, "organization")]},
        "B": {"ch0002": [sp("ch0002", 0, 3, "person"), sp("ch0002", 5, 7, "location"),
                         sp("ch0002", 11, 14, None, pending=True, stuck=["type"], candidates="item / trait", doubt="效果像能力")],
              "ch0003": [sp("ch0003", 0, 3, "trait")]},
    },
    "dev_b02": {
        "A": {"ch0004": [sp("ch0004", 0, 3, "person"), sp("ch0004", 5, 8, "location")]},
        "B": {"ch0004": [sp("ch0004", 0, 3, "person"), sp("ch0004", 5, 8, "location"), sp("ch0004", 9, 10, "person")]},
    },
}
GOLD = "data/gold/dev/incoming/prefilled_dev_b02.json"


def make_tasks(root, passes=PASSES):
    for name, runs in passes.items():
        folder = root / "llm_tasks" / name
        folder.mkdir(parents=True)
        chapters = sorted(runs["A"])
        (folder / "task.json").write_text(json.dumps({"name": name, "snapshot_id": "s", "chapters": chapters,
                                                      "guide_version": "1.1"}), encoding="utf-8")
        for run, spans in runs.items():
            (folder / f"draft_{run}.json").write_text(json.dumps({
                "task": name, "run": run, "model": "gpt-test", "source_file": f"pass_{run}.json", "imported_at": "t1",
                "instructions_version": "v1", "chapters": {c: {"spans": s, "errors": []} for c, s in spans.items()}},
                ensure_ascii=False), encoding="utf-8")


def build_world(tmp_path, monkeypatch, passes=PASSES):
    monkeypatch.setattr(adj, "load_snapshot", lambda *a, **k: snap())
    monkeypatch.setattr(rv, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rv, "git_state", lambda: {"commit": "x", "dirty": False})
    make_tasks(tmp_path, passes)
    sheet = tmp_path / "k.tsv"
    sheet.write_text("名字\t类型\t证据章\t备注\n章医生\t人物\t\t固定叫法\n", encoding="utf-8")
    incoming = tmp_path / "data" / "gold" / "dev" / "incoming"
    first, _ = adj.prefill(tmp_path / "llm_tasks" / "dev_b01", dev_base(), sheet, incoming, set_name="dev")
    second, _ = adj.prefill(tmp_path / "llm_tasks" / "dev_b02", json.loads(first.read_text(encoding="utf-8")), sheet,
                            incoming, set_name="dev")
    return tmp_path, sheet, first, second


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def test_prefill_dev_chains_packages_and_keeps_the_other_chapters(world):
    root, sheet, first, second = world
    base, out = dev_base(), json.loads(second.read_text(encoding="utf-8"))
    assert second.name == "prefilled_dev_b02.json" and out["set"] == "dev"
    for cid in ("ch0001", "ch0005"):  # the done chapter and the one no package holds stay exactly as they were
        assert out["chapters"][cid] == base["chapters"][cid]
    assert list(out["chapters"]) == list(base["chapters"])
    for cid in ("ch0002", "ch0003", "ch0004"):
        c = out["chapters"][cid]
        assert (c["provenance"], c["status"], c["checks"]) == ("model_double_adjudicated", "doing", {"adjudicated": False})
    assert set(out["llm_drafts"]) == {"dev_b01", "dev_b02"}
    assert [a["text"] for a in out["chapters"]["ch0004"]["annotations"] if a["pending"]] == ["人"]  # one pass only
    assert check(out, snap())[0] == []
    with pytest.raises(SystemExit):  # dev_b01 again on the chained file: its chapters already have work
        adj.prefill(root / "llm_tasks" / "dev_b01", out, sheet, root / "other", set_name="dev")
    with pytest.raises(SystemExit):  # a dev prefill needs a base
        adj.prefill(root / "llm_tasks" / "dev_b01", None, sheet, root / "other", set_name="dev")
    with pytest.raises(SystemExit):  # the base must be a dev export
        adj.prefill(root / "llm_tasks" / "dev_b01", {**dev_base(), "set": "test"}, sheet, root / "other", set_name="dev")
    small = dev_base()
    del small["chapters"]["ch0004"]
    with pytest.raises(SystemExit):  # a dev package may only hold chapters of the dev set
        adj.prefill(root / "llm_tasks" / "dev_b02", small, sheet, root / "other", set_name="dev")


def test_prefill_default_folders_follow_the_set(world, tmp_path, monkeypatch):
    root, sheet, *_ = world
    monkeypatch.setattr(adj, "REPO_ROOT", tmp_path / "repo")
    target, _ = adj.prefill(root / "llm_tasks" / "dev_b01", dev_base(), sheet, set_name="dev")
    assert target == tmp_path / "repo" / "data" / "gold" / "dev" / "incoming" / "prefilled_dev_b01.json"
    target, _ = adj.prefill(root / "llm_tasks" / "dev_b01", None, sheet)  # the test default: a new test file
    assert target == tmp_path / "repo" / "data" / "gold" / "test" / "incoming" / "prefilled_dev_b01.json"
    assert json.loads(target.read_text(encoding="utf-8"))["set"] == "test"


def gold_of(set_name):
    """The last prefilled file of a set in these fixtures (two packages); GOLD for dev."""
    return f"data/gold/{set_name}/incoming/prefilled_{set_name}_b02.json"


def listing(root, set_name="dev"):
    return rv.list_pending(gold_of(set_name), set_name, "astra", "RID", root=root, snapshot=snap(), task_root=root / "llm_tasks",
                           sheet=SHEET)


def test_pending_lists_every_pending_mark_grouped_by_name(world):
    root, *_ = world
    found, asked, by_sheet = listing(root)
    assert [(p["chapter_id"], p["start"], p["end"]) for p in found] == [
        ("ch0002", 5, 8), ("ch0002", 11, 14), ("ch0002", 16, 19), ("ch0003", 0, 3), ("ch0003", 7, 10), ("ch0004", 9, 10)]
    assert list(asked) == ["回音铃", "人", "灰塔城 / 灰塔", "黑肩章"] and list(by_sheet) == ["章医生"]
    bell = asked["回音铃"]
    assert [p["kinds"] for p in bell] == [["gpt_pending"], ["type"]]
    assert bell[0]["a_says"] == "【回音铃】物品" and bell[0]["b_says"] == "待审【回音铃】（item / trait）"
    assert bell[0]["doubts"] == [{"who": "B 遍", "doubt": "效果像能力", "candidates": "item / trait"}]
    assert asked["灰塔城 / 灰塔"][0]["kinds"] == ["boundary"] and asked["灰塔城 / 灰塔"][0]["names"] == ["灰塔城", "灰塔"]
    assert asked["黑肩章"][0]["kinds"] == ["only_a"] and asked["人"][0]["kinds"] == ["only_b"]
    doctor = by_sheet["章医生"][0]
    assert doctor["kinds"] == ["sheet"] and doctor["sheet_says"] == "【章医生】人物" and doctor["a_says"] == doctor["b_says"] == "不标"
    out = root / "data" / "gold" / "dev" / "review_astra"
    q = (out / "questions.md").read_text(encoding="utf-8")
    assert "## 1. 回音铃（2 处，2 章）" in q and "A 遍：【回音铃】物品；B 遍：待审【回音铃】（item / trait）" in q
    assert "  - B 遍的疑点：效果像能力（候选：item / trait）" in q
    assert "- S1. 章医生：1 处（知识表：【章医生】人物）" in q and "现在的答案" not in q and "复核模型：" not in q
    sheet_md = (out / "answer_sheet.md").read_text(encoding="utf-8")
    assert "| 编号 | 名字 | 处数 | A 遍（第一处） | B 遍（第一处） | 你的回答 |" in sheet_md
    assert "| 1 | 回音铃 | 2 | 【回音铃】物品 | 待审【回音铃】（item / trait） |  |" in sheet_md  # 物品 or 特性
    assert "| S1 | 章医生 | 1 | 不标 | 不标 |  |" in sheet_md
    d = json.loads((out / "disagreements.json").read_text(encoding="utf-8"))
    assert (d["mode"], d["set"], d["gold_file"], d["chapters"]) == ("pending", "dev", GOLD, ["ch0002", "ch0003", "ch0004"])
    assert d["sheet_rows"] == {"章医生": SHEET[0]}
    metrics = (root / "runs" / "RID" / "metrics.json").read_text(encoding="utf-8")
    assert json.loads(metrics)["pending_places"] == 6 and json.loads(metrics)["chapters_still_todo"] == 1
    for name in ("回音铃", "灰塔", "黑肩章", "章医生", "林小雨"):  # counts only, no novel text
        assert name not in metrics and name not in (root / "runs" / "RID" / "config.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):  # a listing is never overwritten
        listing(root)


def test_pending_refuses_drafts_imported_again_after_the_prefill(world):
    root, *_ = world
    draft = root / "llm_tasks" / "dev_b01" / "draft_B.json"
    data = json.loads(draft.read_text(encoding="utf-8"))
    draft.write_text(json.dumps({**data, "imported_at": "t2"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        listing(root)


ANSWERS = {"answered_at": "2026-10-04", "answers": [
    {"no": 1, "name": "回音铃", "places": 2, "answer": "我觉得是物品", "handling": "reviewer"},
    {"no": 2, "name": "人", "places": 1, "answer": "不是名字", "handling": "direct", "reading": "none"},
    {"no": 3, "name": "灰塔城 / 灰塔", "places": 1, "answer": "灰塔城是地点", "handling": "direct", "reading": "location",
     "text": "灰塔城"},
    {"no": 4, "name": "黑肩章", "places": 1, "answer": "制服，不知道咋算", "handling": "reviewer"},
]}
RULINGS = {"q01-1": {"type": "item", "text": "回音铃"}, "q01-2": {"type": "pending", "text": ""},
           "q04-1": {"type": "none", "text": ""}, "s01-1": {"type": "person", "text": "章医生"}}


def adjudicate(root, sheet_groups="reviewer", rulings=RULINGS, answers=ANSWERS, set_name="dev"):
    gold = gold_of(set_name)
    export = json.loads((root / gold).read_text(encoding="utf-8"))
    d = json.loads((root / "data" / "gold" / set_name / "review_astra" / "disagreements.json").read_text(encoding="utf-8"))
    guide = root / "guide.md"
    guide.write_text("规则", encoding="utf-8")
    pkg = root / "llm_tasks" / f"adjudicate_{set_name}_{sheet_groups}"
    places = rv.build_package(answers, d, export, d["chapters"], snap(), pkg, guide,
                              answers_config=f"configs/{set_name}_pending_answers.json", sheet_groups=sheet_groups)
    (pkg / "pass_astra.json").write_text(json.dumps({"decisions": [
        {"id": q["id"], **rulings[q["id"]], "rule": "A1", "reason": "按规则"} for q in places]}, ensure_ascii=False),
        encoding="utf-8")
    decisions = rv.build_decisions(answers, d, gold, export, d["chapters"], snap(), [], [pkg], "astra",
                                   answers_config=f"configs/{set_name}_pending_answers.json", sheet_groups=sheet_groups)
    return export, pkg, places, decisions


def test_package_holds_the_hedged_answers_and_the_sheet_names(world):
    root, *_ = world
    listing(root)
    export, pkg, places, _ = adjudicate(root)
    assert [(q["id"], q["chapter_id"], q["start"], q["name"], q["fixed_type"]) for q in places] == [
        ("q01-1", "ch0002", 11, "回音铃", None), ("q01-2", "ch0003", 0, "回音铃", None),
        ("q04-1", "ch0003", 7, "黑肩章", None), ("s01-1", "ch0002", 16, "章医生", "person")]
    q = (pkg / "questions.md").read_text(encoding="utf-8")
    assert "书的知识（原话）：我觉得是物品" in q and "书的知识（读者的名字知识表）：章医生——人物；备注：固定叫法" in q
    assert "类型已由读者确定：人物" in q and "本段：林小雨去了灰塔城，见到【回音铃】。" in q
    assert "效果像能力" not in q and "item / trait" not in q  # the passes' readings are not shown
    task = json.loads((pkg / "task.json").read_text(encoding="utf-8"))
    assert (task["set"], task["answers_config"], task["listing"]) == ("dev", "configs/dev_pending_answers.json", GOLD)


def test_decide_then_revise_settles_every_pending_place_and_keeps_the_other_chapters(world):
    root, *_ = world
    listing(root)
    export, _, _, decisions = adjudicate(root)
    assert decisions["only_chapters"] == decisions["finish_chapters"] == ["ch0002", "ch0003", "ch0004"]
    assert [(p["chapter_id"], p["start"], p["text"], p["reading"], p["decider"]) for p in decisions["places"]] == [
        ("ch0004", 9, "人", "none", "user"), ("ch0002", 5, "灰塔城", "location", "user"),
        ("ch0002", 11, "回音铃", "item", "reviewer"), ("ch0003", 0, "回音铃", "unscored", "reviewer"),
        ("ch0003", 7, "黑肩章", "none", "reviewer"), ("ch0002", 16, "章医生", "person", "reviewer")]
    assert "用户原话：我觉得是物品" in decisions["places"][2]["reason"]
    assert "名字知识表：章医生——人物" in decisions["places"][5]["reason"]
    out, log = revise(export, decisions, snap(), now="T")
    errors, warnings, summary = check(out, snap())
    assert errors == [] and not [w for w in warnings if "ch2" in w or "ch3" in w or "ch4" in w]
    base = dev_base()
    for cid in ("ch0001", "ch0005"):
        assert out["chapters"][cid] == base["chapters"][cid]
    for cid in ("ch0002", "ch0003", "ch0004"):
        c = out["chapters"][cid]
        assert c["status"] == "done" and c["checks"] == {"adjudicated": True} and c["provenance"] == "model_double_adjudicated"
        assert not [a for a in c["annotations"] if a["pending"]]
    marks = {cid: [(a["start"], a["end"], a["type"]) for a in out["chapters"][cid]["annotations"]] for cid in ("ch0002", "ch0003", "ch0004")}
    assert marks == {"ch0002": [(0, 3, "person"), (5, 8, "location"), (11, 14, "item"), (16, 19, "person")],
                     "ch0003": [], "ch0004": [(0, 3, "person"), (5, 8, "location")]}
    assert out["chapters"]["ch0003"]["unscored"][0]["text"] == "回音铃"
    decided = [(cid, p["start"], p["reading"], p["decider"]) for cid in ("ch0002", "ch0003", "ch0004")
               for p in out["chapters"][cid]["decided_places"]]
    assert sorted(decided) == [("ch0002", 5, "location", "user"), ("ch0002", 11, "item", "reviewer"),
                               ("ch0002", 16, "person", "reviewer"), ("ch0003", 0, "unscored", "reviewer"),
                               ("ch0003", 7, "none", "reviewer"), ("ch0004", 9, "none", "user")]
    assert summary["done_chapters"] == 4 and any(line.startswith("finished ch0004") for line in log)


def test_sheet_groups_can_be_decided_by_the_sheet_row(world):
    root, *_ = world
    listing(root)
    export, _, places, decisions = adjudicate(root, sheet_groups="sheet")
    assert [q["id"] for q in places] == ["q01-1", "q01-2", "q04-1"]
    doctor = next(p for p in decisions["places"] if p["text"] == "章医生")
    assert (doctor["reading"], doctor["decider"]) == ("person", "sheet")
    out, _ = revise(export, decisions, snap(), now="T")
    assert check(out, snap())[0] == []


def test_decide_refuses_incomplete_or_unclear_answers(world):
    root, *_ = world
    listing(root)
    missing = {**ANSWERS, "answers": ANSWERS["answers"][:3]}
    with pytest.raises(SystemExit):  # every asked group needs an answer
        adjudicate(root, answers=missing)
    unclear = copy.deepcopy(ANSWERS)
    del unclear["answers"][2]["text"]  # 地点, but 灰塔城 or 灰塔?
    with pytest.raises(SystemExit):
        adjudicate(root, answers=unclear)
    renumbered = copy.deepcopy(ANSWERS)
    renumbered["answers"][1]["no"] = 3
    with pytest.raises(SystemExit):
        adjudicate(root, answers=renumbered)
    wrong = {**RULINGS, "s01-1": {"type": "item", "text": "章医生"}}  # the sheet fixed the type as person
    with pytest.raises(SystemExit):
        adjudicate(root, rulings=wrong)


def test_revise_keeps_chapters_outside_only_chapters(world):
    root, *_ = world
    listing(root)
    export, _, _, decisions = adjudicate(root)
    stray = {"chapter_id": "ch0001", "start": 0, "end": 2, "text": "徐获", "reading": "none", "reason": "x"}
    with pytest.raises(ValueError):
        revise(export, {**decisions, "places": decisions["places"] + [stray]}, snap(), now="T")
    with pytest.raises(ValueError):  # a chapter cannot be finished while a pending mark is left
        revise(export, {**decisions, "places": decisions["places"][1:]}, snap(), now="T")


def test_decide_stops_when_the_reviewer_frames_a_sheet_name_otherwise(world):
    root, *_ = world
    listing(root)
    shorter = {**RULINGS, "s01-1": {"type": "person", "text": "章"}}  # the sheet row fixes the whole string 章医生
    with pytest.raises(SystemExit, match="the name sheet fixes the whole string 章医生"):
        adjudicate(root, rulings=shorter)
    q = (root / "llm_tasks" / "adjudicate_dev_reviewer" / "questions.md").read_text(encoding="utf-8")
    assert "类型已由读者确定：人物；名字也已由读者确定为整串“章医生”。只裁决这里要不要标" in q
    assert "text 必须是“章医生”" in q and "框多长" not in q  # no hedged answer here has a fixed type
    _, _, _, decisions = adjudicate(root, rulings={**RULINGS, "s01-1": {"type": "none", "text": ""}})
    doctor = next(p for p in decisions["places"] if p["start"] == 16)
    assert (doctor["text"], doctor["reading"], doctor["decider"]) == ("章医生", "none", "reviewer")  # not marking is its call


def test_decide_stops_when_the_reviewer_marks_without_text(world):
    root, *_ = world
    listing(root)
    for qid, kind in (("q01-1", "item"), ("s01-1", "person")):  # a hedged answer's place and a sheet name's place
        with pytest.raises(SystemExit, match="no text"):
            adjudicate(root, rulings={**RULINGS, qid: {"type": kind, "text": ""}})


def test_direct_type_answer_names_its_string_on_a_sheet_or_boundary_place(tmp_path, monkeypatch):
    passes = copy.deepcopy(PASSES)
    for run in ("A", "B"):  # both passes unsure of 章医生到 (the sheet's 章医生 inside) and of where 冬先生 ends
        passes["dev_b01"][run]["ch0002"].append(sp("ch0002", 16, 20, None, pending=True, stuck=["qualification"],
                                                   candidates="person / 不标", doubt="到是不是名字"))
        passes["dev_b02"][run]["ch0004"][0] = sp("ch0004", 0, 3, None, pending=True, stuck=["boundary"],
                                                 candidates="person", doubt="冬还是冬先生")
    root, *_ = build_world(tmp_path, monkeypatch, passes)
    _, asked, by_sheet = listing(root)
    doctor, winter = asked["章医生到"][0], asked["冬先生"][0]
    assert not by_sheet and doctor["names"] == ["章医生到"] and doctor["sheet_says"] == "【章医生】人物"
    assert winter["names"] == ["冬先生"] and winter["stuck"] == ["boundary"] and not winter["sheet_says"]
    numbers = {key: n for n, key in enumerate(asked, 1)}
    rulings = {f"q{numbers[key]:02d}-{k}": RULINGS[f"q{old:02d}-{k}"]
               for key, old in (("回音铃", 1), ("黑肩章", 4)) for k in range(1, len(asked[key]) + 1)}

    def answers(**texts):  # the one string each new direct answer names (none: no text recorded)
        words = {a["name"]: a for a in ANSWERS["answers"]}
        for key in ("章医生到", "冬先生"):
            words[key] = {"answer": f"{key[:3]}是人物", "handling": "direct", "reading": "person"}
            if texts.get(key):
                words[key]["text"] = texts[key]
        return {**ANSWERS, "answers": [{**words[key], "no": numbers[key], "name": key, "places": len(ps)}
                                       for key, ps in asked.items()]}

    for texts in ({"冬先生": "冬先生"}, {"章医生到": "章医生"}):  # the sheet place, then the boundary place, without text
        with pytest.raises(SystemExit, match="several strings, a name-sheet reading or a boundary doubt"):
            adjudicate(root, rulings=rulings, answers=answers(**texts))
    export, _, _, decisions = adjudicate(root, rulings=rulings, answers=answers(章医生到="章医生", 冬先生="冬先生"))
    out, _ = revise(export, decisions, snap(), now="T")
    assert check(out, snap())[0] == []
    assert [(a["start"], a["end"], a["type"]) for a in out["chapters"]["ch0002"]["annotations"]][-1] == (16, 19, "person")
    assert [(a["start"], a["end"], a["type"]) for a in out["chapters"]["ch0004"]["annotations"]][0] == (0, 3, "person")


# ---- the new sets dev2 and test2: no base before the first package; the set's chapters come from its page config ----

NEW_CHAPTERS = ["ch0002", "ch0003", "ch0004", "ch0005"]  # the set in page order; ch0001 is outside it
BLANK_CHECKS = {"occurrences": False, "full_read": False, "records": False}


def blank(cid):
    return {"content_hash": f"h{cid}", "text_sha256": hashlib.sha256(text_of(cid).encode("utf-8")).hexdigest(),
            "status": "todo", "provenance": None, "checks": BLANK_CHECKS, "seconds": 0, "annotations": [], "skipped": [], "note": ""}


def new_set_config(root, set_name, chapters=NEW_CHAPTERS, snapshot="s", guide="1.1", name=None):
    """configs/annotate_<set>.json under root (or configs/<name>.json): dev2 points at a chapter list, test2 at an
    order file with n, as the real configs do."""
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    stem = name or f"annotate_{set_name}"
    if set_name == "test2":
        (configs / f"{stem}.order.json").write_text(json.dumps({"order": [{"chapter_id": c} for c in chapters + ["ch0001"]]}),
                                                     encoding="utf-8")
        cfg = {"set": set_name, "snapshot": snapshot, "guide_version": guide, "chapters_from_order": f"configs/{stem}.order.json",
               "n": len(chapters)}
    else:
        (configs / f"{stem}.list.json").write_text(json.dumps({"set": set_name, "n": len(chapters), "chapters": chapters}),
                                                    encoding="utf-8")
        cfg = {"set": set_name, "snapshot": snapshot, "guide_version": guide, "chapters_from_config": f"configs/{stem}.list.json"}
    path = configs / f"{stem}.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def build_new_world(tmp_path, monkeypatch, set_name, passes=PASSES):
    monkeypatch.setattr(adj, "load_snapshot", lambda *a, **k: snap())
    monkeypatch.setattr(adj, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(annotate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rv, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rv, "git_state", lambda: {"commit": "x", "dirty": False})
    make_tasks(tmp_path, {name.replace("dev_", f"{set_name}_"): runs for name, runs in passes.items()})
    sheet = tmp_path / "k.tsv"
    sheet.write_text("名字\t类型\t证据章\t备注\n章医生\t人物\t\t固定叫法\n", encoding="utf-8")
    new_set_config(tmp_path, set_name)
    incoming = tmp_path / "data" / "gold" / set_name / "incoming"
    first, _ = adj.prefill(tmp_path / "llm_tasks" / f"{set_name}_b01", None, sheet, incoming, set_name=set_name)
    second, _ = adj.prefill(tmp_path / "llm_tasks" / f"{set_name}_b02", json.loads(first.read_text(encoding="utf-8")), sheet,
                            incoming, set_name=set_name)
    return tmp_path, sheet, first, second


@pytest.mark.parametrize("set_name", ["dev2", "test2"])
def test_a_new_set_starts_from_an_empty_base_and_chains(tmp_path, monkeypatch, set_name):
    root, sheet, first, second = build_new_world(tmp_path, monkeypatch, set_name)
    one, out = json.loads(first.read_text(encoding="utf-8")), json.loads(second.read_text(encoding="utf-8"))
    assert first.name == f"prefilled_{set_name}_b01.json" and second == root / "data" / "gold" / set_name / "incoming" / f"prefilled_{set_name}_b02.json"
    assert (out["set"], out["snapshot_id"], out["guide_version"], out["current"]) == (set_name, "s", "1.1", "ch0002")
    assert list(one["chapters"]) == list(out["chapters"]) == NEW_CHAPTERS  # the page config's chapters, in its order
    assert out["revisions"][0] == {"at": one["created_at"], "by": "claude", "what":
        f"empty {set_name} export: the 4 chapters of configs/annotate_{set_name}.json in that order, no annotations; the base of the set's first prefill"}
    assert one["chapters"]["ch0005"] == out["chapters"]["ch0005"] == blank("ch0005")  # untouched, as the page starts it
    assert one["chapters"]["ch0004"] == blank("ch0004")  # b01 did not touch b02's chapter
    assert out["chapters"]["ch0002"] == one["chapters"]["ch0002"]  # chaining carries b01's work as it is
    for cid in ("ch0002", "ch0003", "ch0004"):
        c = out["chapters"][cid]
        assert (c["provenance"], c["status"], c["checks"]) == ("model_double_adjudicated", "doing", {"adjudicated": False})
    assert set(out["llm_drafts"]) == {f"{set_name}_b01", f"{set_name}_b02"} and len(out["revisions"]) == 3
    assert check(out, snap())[0] == []
    with pytest.raises(SystemExit, match="chains onto"):  # a later package without a base: the chain is not restarted
        adj.prefill(root / "llm_tasks" / f"{set_name}_b02", None, sheet, first.parent, set_name=set_name)
    with pytest.raises(SystemExit, match="already has work"):  # the first package again on the chained file
        adj.prefill(root / "llm_tasks" / f"{set_name}_b01", out, sheet, root / "other", set_name=set_name)
    other = "test2" if set_name == "dev2" else "dev2"
    with pytest.raises(SystemExit, match=f"not a {set_name} export"):  # the other new set's file is no base for this one
        adj.prefill(root / "llm_tasks" / f"{set_name}_b02", {**one, "set": other}, sheet, root / "other", set_name=set_name)
    small = json.loads(json.dumps(one))
    del small["chapters"]["ch0004"]
    with pytest.raises(SystemExit, match="are not chapters of"):  # a package may only hold chapters of the set
        adj.prefill(root / "llm_tasks" / f"{set_name}_b02", small, sheet, root / "other", set_name=set_name)
    assert not (root / "other").exists() and not (root / "data" / "gold" / other).exists()


def test_a_new_sets_empty_base_follows_its_page_config(tmp_path, monkeypatch):
    root, sheet, *_ = build_new_world(tmp_path, monkeypatch, "dev2")
    task = root / "llm_tasks" / "dev2_b01"
    wrong_snapshot = new_set_config(root, "dev2", snapshot="other", name="w1")
    with pytest.raises(SystemExit, match="differ from"):
        adj.prefill(task, None, sheet, root / "fresh", set_name="dev2", set_config_path=wrong_snapshot)
    wrong_guide = new_set_config(root, "dev2", guide="1.0", name="w2")
    with pytest.raises(SystemExit, match="differ from"):
        adj.prefill(task, None, sheet, root / "fresh", set_name="dev2", set_config_path=wrong_guide)
    other_set = new_set_config(root, "test2", name="w3")
    with pytest.raises(SystemExit, match="annotation config of 'test2', not dev2"):
        adj.prefill(task, None, sheet, root / "fresh", set_name="dev2", set_config_path=other_set)
    short = new_set_config(root, "dev2", chapters=["ch0002", "ch0003"], name="w4")  # ch0004 is not in the set
    with pytest.raises(SystemExit, match="are not chapters of the dev2 set"):
        adj.prefill(root / "llm_tasks" / "dev2_b02", None, sheet, root / "fresh", set_name="dev2", set_config_path=short)
    assert not (root / "fresh").exists()
    with pytest.raises(SystemExit, match="needs a base"):  # the dev rule is unchanged: no base is built for dev
        adj.prefill(task, None, sheet, root / "fresh", set_name="dev")
    target, _ = adj.prefill(task, None, sheet, root / "fresh", set_name="dev2", set_config_path=short)  # b01 fits the short list
    assert list(json.loads(target.read_text(encoding="utf-8"))["chapters"]) == ["ch0002", "ch0003"]
    with pytest.raises(SystemExit):  # the config is only read without a base, and the default path is configs/annotate_<set>.json
        adj.prefill(task, None, sheet, root / "fresh2", set_name="dev2", set_config_path=root / "configs" / "missing.json")


@pytest.mark.parametrize("set_name", ["dev2", "test2"])
def test_a_new_set_goes_through_pending_package_decide_and_revise(tmp_path, monkeypatch, set_name):
    root, *_ = build_new_world(tmp_path, monkeypatch, set_name)
    found, asked, by_sheet = listing(root, set_name)
    assert list(asked) == ["回音铃", "人", "灰塔城 / 灰塔", "黑肩章"] and list(by_sheet) == ["章医生"] and len(found) == 6
    out = root / "data" / "gold" / set_name / "review_astra"
    d = json.loads((out / "disagreements.json").read_text(encoding="utf-8"))
    assert (d["mode"], d["set"], d["gold_file"], d["chapters"]) == ("pending", set_name, gold_of(set_name), ["ch0002", "ch0003", "ch0004"])
    assert sorted(p.name for p in (root / "data" / "gold").iterdir()) == [set_name]  # nothing of another set was touched
    metrics = json.loads((root / "runs" / "RID" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["pending_places"] == 6 and metrics["chapters_still_todo"] == 1
    assert json.loads((root / "runs" / "RID" / "config.json").read_text(encoding="utf-8"))["set"] == set_name
    export, pkg, places, decisions = adjudicate(root, set_name=set_name)
    task = json.loads((pkg / "task.json").read_text(encoding="utf-8"))
    assert (task["set"], task["listing"], task["answers_config"]) == (set_name, gold_of(set_name), f"configs/{set_name}_pending_answers.json")
    assert [q["id"] for q in places] == ["q01-1", "q01-2", "q04-1", "s01-1"]
    assert decisions["description"].startswith(f"{set_name} answers: the 3 chapters labelled twice by gpt-test")
    assert decisions["base"] == gold_of(set_name) and decisions["only_chapters"] == decisions["finish_chapters"] == ["ch0002", "ch0003", "ch0004"]
    revised, _ = revise(export, decisions, snap(), now="T")
    errors, _, summary = check(revised, snap())
    assert errors == [] and summary["done_chapters"] == 3
    assert revised["chapters"]["ch0005"] == export["chapters"]["ch0005"] == blank("ch0005")
    for cid in ("ch0002", "ch0003", "ch0004"):
        c = revised["chapters"][cid]
        assert (c["status"], c["provenance"], c["checks"]) == ("done", "model_double_adjudicated", {"adjudicated": True})
    marks = {cid: [(a["start"], a["end"], a["type"]) for a in revised["chapters"][cid]["annotations"]] for cid in ("ch0002", "ch0003", "ch0004")}
    assert marks == {"ch0002": [(0, 3, "person"), (5, 8, "location"), (11, 14, "item"), (16, 19, "person")],
                     "ch0003": [], "ch0004": [(0, 3, "person"), (5, 8, "location")]}
    with pytest.raises(SystemExit, match="lies under data/gold/"):  # the file is never read as another set's
        rv.list_pending(gold_of(set_name), "dev", "astra", "RID2", root=root, snapshot=snap(), task_root=root / "llm_tasks", sheet=SHEET)
    assert not (root / "data" / "gold" / "dev").exists() and not (root / "runs" / "RID2").exists()
