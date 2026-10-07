import pytest

import novel_ner.review as rv
from novel_ner.review import answer_sheet_md, by_name, kinds_of, parser, places, questions_md, resolve


def test_places_join_overlapping_spans_and_kinds_name_the_difference():
    answer = {(0, 1, "person"), (10, 13, "item"), (20, 23, "location"), (40, 42, "person")}
    reviewer = {(0, 3, "person"), (10, 13, "trait"), (20, 23, "location"), (30, 33, "item")}
    ps = places(answer, [(50, 52)], reviewer, [(60, 62)])
    got = [(min(s for s, _, _ in p["answer"] | p["reviewer"]) if p["answer"] | p["reviewer"] else None, kinds_of(p)) for p in ps]
    assert got == [(0, ["boundary"]), (10, ["type"]), (20, []), (30, ["only_reviewer"]), (40, ["only_answer"]),
                   (None, ["answer_unscored"]), (None, ["reviewer_pending"])]


def test_a_chain_of_overlaps_is_one_place_and_pending_is_not_called_a_miss():
    ps = places({(0, 4, "item")}, [], {(3, 6, "item"), (5, 8, "item")}, [])
    assert len(ps) == 1 and kinds_of(ps[0]) == ["boundary"]
    p = places({(0, 3, "item")}, [], set(), [(0, 3)])[0]
    assert kinds_of(p) == ["reviewer_pending"]


def test_groups_by_name_and_sheet_names_are_not_asked_again():
    found = [{"names": ["章医生", "章"]}, {"names": ["章医生", "章"]}, {"names": ["琴弦"]}, {"names": ["大章医生", "章医生"]}]
    asked, by_sheet = by_name(found, {"章医生"})
    assert list(by_sheet) == ["章医生 / 章"] and len(by_sheet["章医生 / 章"]) == 2
    assert set(asked) == {"琴弦", "大章医生 / 章医生"}  # a longer string around a sheet name is still asked


def _test_places():
    p = lambda cid, ctx, a, r, kinds, dec=False, doubts=(): {
        "chapter_id": cid, "context": ctx, "answer_says": a, "reviewer_says": r, "kinds": kinds,
        "answer_decided_by_user": dec, "reviewer_doubt": list(doubts), "names": []}
    asked = {"回音铃": [p("ch0001", "拿出【回音铃】。", "【回音铃】物品", "【回音铃】特性", ["type"], True,
                       [{"doubt": "道具还是能力", "candidates": "item / trait"}])],
             "二号": [p("ch0002", "【二号】说", "没标（不计分）", "没标", ["answer_unscored"])] * 4,
             "灰塔城 / 灰塔": [p("ch0001", "去【灰塔城】", "【灰塔】地点", "【灰塔城】地点", ["boundary"])]}
    by_sheet = {"章医生 / 章": [p("ch0002", "【章医生】来了", "【章】人物", "【章医生】人物", ["boundary"])] * 2}
    return asked, by_sheet


def test_the_test_review_wording_is_unchanged_by_default():
    # recorded from the code that wrote the test review's questions (before the dev wording was added)
    asked, by_sheet = _test_places()
    assert questions_md(asked, by_sheet, "astra", "RID") == (
        "# 复核问题清单（astra，RID）\n\n含小说原文，只在本机看，不入库。【】是范围。每组是同一个名字（或同一串字）的所有分歧。\n"
        "请用白话回答“这是什么”就行，比如“章医生是一个人的固定叫法”“这是普通的门，不是名字”“这里是外号，那里是职业”。\n"
        "不确定就写“不知道”，这些会交给复核模型按规则判，判不了的不计分。\n\n## 1. 回音铃（1 处，1 章）\n"
        "- 分歧：类型不同；其中 1 处是你上次定的\n- ch0001：拿出【回音铃】。\u3000现在的答案：【回音铃】物品；复核模型：【回音铃】特性\n"
        "  - 复核模型的疑点：道具还是能力（候选：item / trait）\n- 你的回答：\n\n## 2. 二号（4 处，1 章）\n"
        "- 分歧：答案里这里设为不计分\n" + "- ch0002：【二号】说\u3000现在的答案：没标（不计分）；复核模型：没标\n" * 3 +
        "- 另有 1 处，同样的分歧见 disagreements.json\n- 你的回答：\n\n## 3. 灰塔城 / 灰塔（1 处，1 章）\n- 分歧：框的范围不同\n"
        "- ch0001：去【灰塔城】\u3000现在的答案：【灰塔】地点；复核模型：【灰塔城】地点\n- 你的回答：\n\n"
        "## 已按名字知识表处理（不用回答，列出来供你核对）\n\n- 章医生 / 章：2 处\n")
    header = "| 编号 | 名字 | 处数 | 现在的答案（第一处） | 复核模型（第一处） | 你的回答 |\n|---|---|---|---|---|---|\n"
    assert answer_sheet_md(asked, "astra", "RID") == (
        "# 回答表（astra，RID）\n\n编号和 questions.md 里的一样，原文片段和复核模型的疑点都在那里。只填最后一列，白话就行；不知道就写“不知道”。\n\n"
        "## 一、道具还是特性（书里它是一件道具，还是人物的特性、能力？）\n\n" + header +
        "| 1 | 回音铃 | 1 | 【回音铃】物品 | 【回音铃】特性 |  |\n\n"
        "## 二、是不是名字、是什么（书里是专门的叫法，还是普通说法、编号、口令？）\n\n" + header +
        "| 2 | 二号 | 4 | 没标（不计分） | 没标 |  |\n\n## 三、其他（地点还是组织、名字框多长等）\n\n" + header +
        "| 3 | 灰塔城 / 灰塔 | 1 | 【灰塔】地点 | 【灰塔城】地点 |  |\n\n")


def test_command_defaults_stay_the_test_review_and_dev_names_its_paths():
    args = resolve(parser().parse_args(["package", "--gold", "g.json"]))
    assert (args.set_name, args.answers, args.out) == ("test", "configs/review_astra_answers.json", "data/llm_tasks/adjudicate_v3")
    args = resolve(parser().parse_args(["decide", "--gold", "g.json"]))
    assert (args.answers, args.adjudication, args.sheet_names, args.out) == (
        "configs/review_astra_answers.json", ["data/llm_tasks/adjudicate_v3"], ["章医生", "武博士"], "configs/gold_test_v3_decisions.json")
    assert parser().parse_args(["diff", "--gold", "g.json", "--run-id", "r"]).set_name == "test"
    for argv in (["package", "--set", "dev", "--gold", "g.json", "--answers", "a.json"],
                 ["decide", "--set", "dev", "--gold", "g.json", "--out", "o.json"]):
        with pytest.raises(SystemExit):
            resolve(parser().parse_args(argv))
    args = resolve(parser().parse_args(["decide", "--set", "dev", "--gold", "g.json", "--answers", "a.json", "--out", "o.json"]))
    assert (args.adjudication, args.sheet_names, args.sheet_groups) == ([], [], "reviewer")


def test_the_test_package_and_decisions_are_unchanged_by_default(tmp_path, monkeypatch):
    # a small test-set review (answers v2 vs the reviewer), as the v3 decisions were made; outputs recorded from the
    # code before the dev mode was added
    import json
    import os

    import novel_ner.review as rv
    from novel_ner.snapshot import Chapter, Snapshot
    paras = {"ch0001": [("p0001", "徐获拉开门，看见黑肩章站在门口。"), ("p0002", "二号说：章医生来了。"),
                        ("p0003", "他拿出回音铃，回音铃响了。")],
             "ch0002": [("p0001", "黑肩章又来了，章医生也来了，3号也在。")]}
    snapshot = Snapshot("s", None, {}, {c: Chapter(c, int(c[2:]), ps, "", f"h{c}") for c, ps in paras.items()}, [], [])
    text = {c: "\n".join(t for _, t in ps) for c, ps in paras.items()}

    def at(c, s, k=1):
        i = -1
        for _ in range(k):
            i = text[c].find(s, i + 1)
        return i, i + len(s)

    mark = lambda c, s, t, k=1: {"start": at(c, s, k)[0], "end": at(c, s, k)[1], "text": s, "type": t, "pending": False}
    export = {"set": "test", "snapshot_id": "s", "guide_version": "1.1", "chapters": {
        "ch0001": {"status": "done", "annotations": [mark("ch0001", "徐获", "person"), mark("ch0001", "黑肩章", "organization"),
                                                     mark("ch0001", "章", "person", 2), mark("ch0001", "回音铃", "item")],
                   "unscored": [{**dict(zip(("start", "end"), at("ch0001", "二号"))), "text": "二号", "reason": "x"}]},
        "ch0002": {"status": "done", "annotations": [mark("ch0002", "黑肩章", "organization"), mark("ch0002", "章", "person", 2),
                                                     mark("ch0002", "3号", "person")]}}}
    ids = ["ch0001", "ch0002"]
    reviews = {"ch0001": {"spans": {(*at("ch0001", "徐获"), "person"), (*at("ch0001", "章医生"), "person"),
                                    (*at("ch0001", "回音铃"), "trait"), (*at("ch0001", "回音铃", 2), "item")},
                          "pending": [at("ch0001", "黑肩章")],
                          "doubts": [{"start": 8, "end": 11, "doubt": "制服还是组织", "candidates": "organization / 不标"}]},
               "ch0002": {"spans": {(*at("ch0002", "章医生"), "person")}, "pending": [], "doubts": []}}
    found = rv.collect(export, ids, snapshot, reviews, {"ch0001": {(*at("ch0001", "回音铃"), "item"): "user_or_sheet"}, "ch0002": {}})
    asked, by_sheet = rv.by_name(found, {"章医生"})
    assert list(asked) == ["回音铃", "黑肩章", "3号", "二号"] and list(by_sheet) == ["章医生 / 章"]
    disagreements = json.loads(json.dumps({"gold_file": "g.json", "asked": asked, "by_sheet": by_sheet}))
    words = {"黑肩章": ("制服，代指执法部", "reviewer"), "3号": ("龙套", "reviewer")}
    answers = {"answered_at": "2026-10-03", "answers": [
        {"no": n, "name": k, "places": len(ps), "answer": words.get(k, ("物品",))[0], "handling": words.get(k, ("", "direct"))[1],
         **({} if k in words else {"reading": "item"}), **({"note2": "补充"} if k == "黑肩章" else {})}
        for n, (k, ps) in enumerate(asked.items(), 1)]}
    (tmp_path / "guide.md").write_text("规则", encoding="utf-8")
    pkg = tmp_path / "adj"
    places = rv.build_package(answers, disagreements, export, ids, snapshot, pkg, tmp_path / "guide.md")
    assert [(q["id"], q["no"], q["chapter_id"], q["start"], q["name"]) for q in places] == [
        ("q02-1", 2, "ch0001", 8, "黑肩章"), ("q02-2", 2, "ch0002", 0, "黑肩章"), ("q03-1", 3, "ch0002", 14, "3号")]
    task = json.loads((pkg / "task.json").read_text(encoding="utf-8"))
    assert list(task) == ["name", "kind", "snapshot_id", "instructions", "guide_version", "answers_config", "places", "created_at"]
    assert task["answers_config"] == "configs/review_astra_answers.json" and list(task["places"][0]) == [
        "id", "no", "chapter_id", "start", "end", "name", "paragraph_id", "extra", "fixed_type"]
    q = (pkg / "questions.md").read_text(encoding="utf-8")
    assert "书的知识（原话）：制服，代指执法部\n读者后来补充（原话）：补充\n" in q
    assert "读者希望同类编号统一处理。考卷 60 章里其他“阿拉伯数字＋号”（不含本包要裁决的地方）的现有标法：没有。" in q
    (pkg / "pass_astra.json").write_text(json.dumps({"decisions": [
        {"id": p["id"], "type": t, "text": x, "rule": "A1", "reason": f"r{i}"}
        for i, (p, t, x) in enumerate(zip(places, ("none", "person", "pending"), ("", "黑肩章", "")))]}, ensure_ascii=False),
        encoding="utf-8")
    os.utime(pkg / "pass_astra.json", (1790000000, 1790000000))
    (tmp_path / "base.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rv, "REPO_ROOT", tmp_path)
    sheet = [{"name": "章医生", "type": "person", "chapter": None, "note": "固定叫法"}]
    out = rv.build_decisions(answers, disagreements, "base.json", export, ids, snapshot, ["章医生"], [pkg], "astra", sheet=sheet)
    on = __import__("time").strftime("%Y-%m-%d", __import__("time").localtime(1790000000))
    assert [(p["chapter_id"], p["start"], p["text"], p["reading"], p["by"], p["decider"], p["reason"]) for p in out["places"]] == [
        ("ch0001", 31, "回音铃", "item", "user 2026-10-03", "user", "回答表 1：物品"),
        ("ch0001", 35, "回音铃", "item", "user 2026-10-03", "user", "回答表 1：物品"),
        ("ch0001", 17, "二号", "item", "user 2026-10-03", "user", "回答表 4：物品"),
        ("ch0001", 21, "章医生", "person", "user 2026-10-02", "sheet", "名字知识表：固定叫法"),
        ("ch0002", 7, "章医生", "person", "user 2026-10-02", "sheet", "名字知识表：固定叫法"),
        ("ch0001", 8, "黑肩章", "none", f"astra {on}（按规则 v1.1，依据用户的说明）", "reviewer",
         "A1：r0（q02-1，adj；用户原话：制服，代指执法部；补充：补充）"),
        ("ch0002", 0, "黑肩章", "person", f"astra {on}（按规则 v1.1，依据用户的说明）", "reviewer",
         "A1：r1（q02-2，adj；用户原话：制服，代指执法部；补充：补充）"),
        ("ch0002", 14, "3号", "unscored", f"astra {on}（按规则 v1.1，依据用户的说明）", "reviewer", "A1：r2（q03-1，adj；用户原话：龙套）")]
    assert list(out) == ["description", "base", "base_sha256", "decided_by", "decided_at", "places"]
    assert out["description"].startswith("Answer v3: the user's answers (configs/review_astra_answers.json)")


def test_the_new_sets_take_the_pending_route_and_diff_is_refused(monkeypatch):
    import sys
    for set_name in ("dev2", "test2"):
        monkeypatch.setattr(sys, "argv", ["review", "diff", "--set", set_name, "--gold", "g.json", "--run-id", "r"])
        with pytest.raises(SystemExit, match="pending route"):
            rv.main()
        assert parser().parse_args(["pending", "--set", set_name, "--prefilled", "p.json", "--run-id", "r"]).set_name == set_name
        for argv in (["package", "--set", set_name, "--gold", "g.json", "--answers", "a.json"],
                     ["decide", "--set", set_name, "--gold", "g.json", "--out", "o.json"]):
            with pytest.raises(SystemExit):  # no test defaults for them: every path is named
                resolve(parser().parse_args(argv))
        args = resolve(parser().parse_args(["decide", "--set", set_name, "--gold", "g.json", "--answers", "a.json", "--out", "o.json"]))
        assert (args.adjudication, args.sheet_names, args.sheet_groups) == ([], [], "reviewer")
    with pytest.raises(SystemExit):  # an unknown set
        parser().parse_args(["pending", "--set", "dev3", "--prefilled", "p.json", "--run-id", "r"])


def test_an_adjudication_package_for_test2_is_built_whatever_the_seal_says(tmp_path, monkeypatch):
    """Adjudicating pending places makes the answers: configs/test2_seal.json exempts it (answer_making_exempt), so
    review package --set test2 goes on to read the gold file while the seal is sealed, missing or open."""
    import json
    import sys
    monkeypatch.setattr(rv, "REPO_ROOT", tmp_path)
    seal = tmp_path / "configs" / "test2_seal.json"
    seal.parent.mkdir()
    argv = ["review", "package", "--set", "test2", "--gold", "g.json", "--answers", "a.json", "--out", "p"]
    for status in ("sealed", None, "open"):
        if status:
            seal.write_text(json.dumps({"status": status}), encoding="utf-8")
        elif seal.exists():
            seal.unlink()
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(FileNotFoundError):  # no seal guard here: the (absent) gold file is read next
            rv.main()
        assert not (tmp_path / "p").exists() and not (tmp_path / "data").exists()
    assert not hasattr(rv, "test2_sealed")


# ---- the answer memory (earlier answers reused, new answers cross-checked) ----

def _place(t, reading, decider, by):
    return {"chapter_id": "ch0001", "start": 0, "end": 1, "text": t, "reading": reading, "by": by, "decider": decider, "reason": "x"}


def test_memory_holds_one_entry_per_exact_string_with_answers_rulings_and_the_sheet_row(tmp_path):
    import json
    cfg = tmp_path / "configs"
    cfg.mkdir()
    dump = lambda name, obj: (cfg / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    dump("review_astra_answers.json", {"answered_at": "2026-10-02", "answers": [
        {"no": 7, "name": "画女", "places": 1, "answer": "人物", "handling": "direct", "reading": "person"}]})
    dump("dev_pending_answers.json", {"answered_at": "2026-10-03", "answers": [
        {"no": 1, "name": "画女", "places": 2, "answer": "画女确定为人物吧", "handling": "direct", "reading": "person", "note2": "照办：人物"},
        {"no": 2, "name": "特防部总部 / 特防部", "places": 1, "answer": "特防部总部为一个地点", "handling": "direct", "reading": "location",
         "text": "特防部总部"},
        {"no": 3, "name": "王菌菌丝 / 王菌", "places": 1, "answer": "只标王菌就行吧", "handling": "reviewer"},
        {"no": "S1", "name": "单人通道", "places": 2, "answer": "口令吧", "handling": "reviewer"}]})
    astra = "astra 2026-10-03（按规则 v1.1，依据用户的说明）"
    dump("gold_test_v3_decisions.json", {"places": [_place("画女", "person", "user", "user 2026-10-02"),
                                                   _place("章医生", "person", "sheet", "user 2026-10-02")]})
    dump("gold_dev_v2_decisions.json", {"places": [
        _place("画女", "person", "user", "user 2026-10-03"), _place("画女", "person", "user", "user 2026-10-03"),
        _place("特防部总部", "location", "user", "user 2026-10-03"), _place("王菌", "item", "reviewer", astra),
        _place("单人通道", "none", "reviewer", astra), _place("单人通道", "unscored", "reviewer", astra)]})
    sheet = tmp_path / "sheet.tsv"
    sheet.write_text("名字\t类型\t证据章\t备注\n章医生\tperson\t\t用户 2026-10-02 定\n倒霉相\tperson\t\t用户 2026-10-01 定\n", encoding="utf-8")
    m = rv.build_memory(["configs/review_astra_answers.json", "configs/dev_pending_answers.json"],
                        ["configs/gold_test_v3_decisions.json", "configs/gold_dev_v2_decisions.json"], sheet, root=tmp_path)
    e = {x["text"]: x for x in m["entries"]}
    # a group name is split on " / "; an answer with "text" belongs to that string only (no 特防部 entry)
    assert list(e) == sorted(e) and set(e) == {"画女", "特防部总部", "王菌菌丝", "王菌", "单人通道", "章医生", "倒霉相"}
    assert e["画女"]["carry_over"] == "person" and e["画女"]["carry_over_from"] == "user 2026-10-02 (test v3); user 2026-10-03 (dev v2)"
    assert [(a["source"], a["set"], a["round"], a["no"], a["answered_at"], a["answer"], a["handling"], a["reading"], a.get("note2"))
            for a in e["画女"]["answers"]] == [
        ("configs/review_astra_answers.json", "test", "review", 7, "2026-10-02", "人物", "direct", "person", None),
        ("configs/dev_pending_answers.json", "dev", "pending", 1, "2026-10-03", "画女确定为人物吧", "direct", "person", "照办：人物")]
    assert [(r["source"], r["reading"], r["decider"], r["by"], r["places"]) for r in e["画女"]["rulings"]] == [
        ("configs/gold_test_v3_decisions.json", "person", "user", "user 2026-10-02", 1),
        ("configs/gold_dev_v2_decisions.json", "person", "user", "user 2026-10-03", 2)]
    assert e["单人通道"]["carry_over"] is None and e["单人通道"]["carry_over_from"] is None  # mixed rulings
    assert e["单人通道"]["answers"][0]["no"] == "S1" and len(e["单人通道"]["rulings"]) == 2
    assert e["特防部总部"]["answers"][0]["text"] == "特防部总部" and e["特防部总部"]["carry_over"] == "location"
    assert e["王菌菌丝"]["carry_over"] is None and e["王菌菌丝"]["rulings"] == []  # answered, never ruled
    assert e["王菌"]["carry_over"] == "item" and e["王菌"]["carry_over_from"] == "reviewer astra 2026-10-03 (dev v2)"
    # the sheet row counts as a user ruling: alone (倒霉相) or beside the decisions that applied it (章医生); only the
    # date of its note is kept (a note may quote the evidence sentence, and configs/ is committed)
    assert e["倒霉相"] == {"text": "倒霉相", "answers": [], "sheet": {"type": "person", "evidence": None, "decided_on": "2026-10-01"},
                        "rulings": [], "carry_over": "person", "carry_over_from": "name sheet (user 2026-10-01)", "carry_over_decider": "sheet"}
    assert e["章医生"]["carry_over_from"] == "name sheet, user 2026-10-02 (test v3); name sheet (user 2026-10-02)"
    # the decider a kept ruling is applied with: the one decider, reviewer when a reviewer ruling is among them
    assert {t: e[t]["carry_over_decider"] for t in e} == {"画女": "user", "特防部总部": "user", "王菌菌丝": None, "王菌": "reviewer",
                                                         "单人通道": None, "章医生": "sheet", "倒霉相": "sheet"}
    assert rv.carry_over_decider({"rulings": [{"decider": "user"}, {"decider": "reviewer"}], "sheet": None}) == "reviewer"
    assert rv.carry_over_decider({"rulings": [{"decider": "user"}], "sheet": {"type": "person"}}) == "user"
    assert m["format"] == 2 and all(set(x["sheet"]) == {"type", "evidence", "decided_on"} for x in m["entries"] if x["sheet"])
    assert m["counts"] == {"entries": 7, "with_carry_over": 5, "with_mixed_rulings": 1, "with_answers": 5, "with_rulings": 5,
                           "with_sheet_row": 2, "by_source": {"configs/review_astra_answers.json": 1, "configs/dev_pending_answers.json": 5,
                                                              "configs/gold_test_v3_decisions.json": 2, "configs/gold_dev_v2_decisions.json": 4,
                                                              "sheet.tsv": 2}}
    assert [(s["kind"], s.get("set"), s["path"], len(s["sha256"])) for s in m["sources"]] == [
        ("answers", "test", "configs/review_astra_answers.json", 64), ("answers", "dev", "configs/dev_pending_answers.json", 64),
        ("decisions", "test", "configs/gold_test_v3_decisions.json", 64), ("decisions", "dev", "configs/gold_dev_v2_decisions.json", 64),
        ("sheet", None, "sheet.tsv", 64)]
    assert m["generated_at"] and "chapter" not in json.dumps(m["entries"])
    assert rv.memory_source("dev2=x/y.json", "answers") == {"kind": "answers", "path": "x/y.json", "set": "dev2", "round": "pending"}
    assert rv.memory_source("configs/gold_test2_v1_decisions.json", "decisions")["tag"] == "test2 v1"
    with pytest.raises(SystemExit, match="set=path"):
        rv.memory_source("x/y.json", "decisions")


def _pending_places():
    def p(cid, ctx, a, b, kinds, names, offered, sheet="", doubts=(), stuck=()):
        return {"chapter_id": cid, "context": ctx, "a_says": a, "b_says": b, "sheet_says": sheet, "kinds": kinds, "names": names,
                "offered": offered, "doubts": list(doubts), "stuck": list(stuck), "start": 0, "end": 1}
    asked = {"画女": [p("ch0001", "【画女】笑了", "待审【画女】（person / 不标）", "待审【画女】（person / 不标）", ["gpt_pending"], ["画女"],
                     ["none", "person"], doubts=[{"who": "A 遍", "doubt": "道具还是人", "candidates": "person / 不标"}])] * 2,
             "黑肩章": [p("ch0002", "【黑肩章】来了", "待审【黑肩章】（organization / 不标）", "不标", ["gpt_pending"], ["黑肩章"], ["none", "organization"])],
             "回音铃": [p("ch0001", "拿出【回音铃】", "待审【回音铃】（item / trait）", "【回音铃】物品", ["gpt_pending"], ["回音铃"], ["item", "trait"])],
             "灰塔城 / 灰塔": [p("ch0002", "去【灰塔城】", "【灰塔】地点", "【灰塔城】地点", ["boundary"], ["灰塔城", "灰塔"], ["location"], stuck=["boundary"])],
             "二号": [p("ch0002", "【二号】说", "不标", "待审【二号】（person / 不标）", ["gpt_pending"], ["二号"], ["none", "person"])]}
    by_sheet = {"章医生": [p("ch0002", "【章医生】来了", "不标", "不标", ["sheet"], ["章医生"], ["none", "person"], sheet="【章医生】人物")]}
    return asked, by_sheet


# recorded from the code that wrote the dev2 and test2 listings (2026-10-04, before the memory was added)
TODAY_QUESTIONS = (
    "# 待定处问题清单（astra，RID）\n\n含小说原文，只在本机看，不入库。【】是范围。每组是同一个名字（或同一串字）的所有待定处：GPT 两遍（A 遍、B 遍）"
    "标得不一样，或有一遍拿不准；两遍一致的已直接采用，不在这里。\n请用白话回答“这是什么”就行，比如“章医生是一个人的固定叫法”“这是普通的门，不是名字”"
    "“这里是外号，那里是职业”。\n不确定就写“不知道”，这些会交给复核模型按规则判，判不了的不计分。\n\n## 1. 画女（2 处，1 章）\n- 分歧：GPT 拿不准（待审）\n"
    "- ch0001：【画女】笑了　A 遍：待审【画女】（person / 不标）；B 遍：待审【画女】（person / 不标）\n  - A 遍的疑点：道具还是人（候选：person / 不标）\n"
    "- ch0001：【画女】笑了　A 遍：待审【画女】（person / 不标）；B 遍：待审【画女】（person / 不标）\n  - A 遍的疑点：道具还是人（候选：person / 不标）\n"
    "- 你的回答：\n\n## 2. 黑肩章（1 处，1 章）\n- 分歧：GPT 拿不准（待审）\n- ch0002：【黑肩章】来了　A 遍：待审【黑肩章】（organization / 不标）；B 遍：不标\n"
    "- 你的回答：\n\n## 3. 回音铃（1 处，1 章）\n- 分歧：GPT 拿不准（待审）\n- ch0001：拿出【回音铃】　A 遍：待审【回音铃】（item / trait）；B 遍：【回音铃】物品\n"
    "- 你的回答：\n\n## 4. 灰塔城 / 灰塔（1 处，1 章）\n- 分歧：两遍框的范围不同\n- ch0002：去【灰塔城】　A 遍：【灰塔】地点；B 遍：【灰塔城】地点\n- 你的回答：\n\n"
    "## 5. 二号（1 处，1 章）\n- 分歧：GPT 拿不准（待审）\n- ch0002：【二号】说　A 遍：不标；B 遍：待审【二号】（person / 不标）\n- 你的回答：\n\n"
    "## 名字知识表里的名字（不再问你：GPT 在这些地方和知识表对不上，会连同你在知识表里写的交给复核模型按规则逐处判，判不了的不计分；想自己定就在回答表里按编号补一行）\n\n"
    "- S1. 章医生：1 处（知识表：【章医生】人物）\n")
_HEADER = "| 编号 | 名字 | 处数 | A 遍（第一处） | B 遍（第一处） | 你的回答 |\n|---|---|---|---|---|---|\n"
TODAY_SHEET = (
    "# 回答表（astra，RID）\n\n编号和 questions.md 里的一样，原文片段和GPT 的疑点都在那里。只填最后一列，白话就行；不知道就写“不知道”。\n\n"
    "## 一、道具还是特性（书里它是一件道具，还是人物的特性、能力？）\n\n" + _HEADER + "| 3 | 回音铃 | 1 | 待审【回音铃】（item / trait） | 【回音铃】物品 |  |\n\n"
    "## 二、是不是名字、是什么（书里是专门的叫法，还是普通说法、编号、口令？）\n\n" + _HEADER +
    "| 1 | 画女 | 2 | 待审【画女】（person / 不标） | 待审【画女】（person / 不标） |  |\n| 2 | 黑肩章 | 1 | 待审【黑肩章】（organization / 不标） | 不标 |  |\n"
    "| 5 | 二号 | 1 | 不标 | 待审【二号】（person / 不标） |  |\n\n## 三、其他（地点还是组织、名字框多长等）\n\n" + _HEADER +
    "| 4 | 灰塔城 / 灰塔 | 1 | 【灰塔】地点 | 【灰塔城】地点 |  |\n\n## 四、名字知识表里的名字（可以不填：不填就连同知识表交给复核模型；填了就按你写的）\n\n"
    + _HEADER + "| S1 | 章医生 | 1 | 不标 | 不标 |  |\n\n")


def _memory():
    def entry(t, carry, rulings=(), answers=(), sheet=None):
        e = {"text": t, "answers": list(answers), "sheet": sheet, "rulings": list(rulings), "carry_over": carry}
        return {**e, "carry_over_from": rv.carry_over_from(e) if carry else None,
                "carry_over_decider": rv.carry_over_decider(e) if carry else None}
    ruling = lambda reading, decider, by, tag, n=1: {"source": "d.json", "set": tag.split()[0], "tag": tag, "reading": reading,
                                                      "decider": decider, "by": by, "places": n}
    answer = lambda s, no, words, handling, reading=None, **kw: {"source": "a.json", "set": s, "round": "review", "no": no,
                                                                 "answered_at": "2026-10-03", "answer": words, "handling": handling,
                                                                 "reading": reading, **kw}
    astra = "astra 2026-10-03（按规则 v1.1，依据用户的说明）"
    return {e["text"]: e for e in [
        entry("画女", "person", [ruling("person", "user", "user 2026-10-03", "test v3", 9)],
              [answer("test", 2, "画女都作为人物，虽然她是道具", "direct", "person", note2="照办：人物")]),
        entry("黑肩章", None, [ruling("none", "reviewer", astra, "test v3"), ruling("unscored", "reviewer", astra, "dev v2")],
              [answer("test", 1, "制服", "reviewer")]),
        entry("灰塔城", "location", [ruling("location", "user", "user 2026-10-03", "dev v2")], [answer("dev", 3, "地点", "direct", "location", text="灰塔城")]),
        entry("灰塔", "none", [ruling("none", "reviewer", astra, "dev v2")]),
        entry("二号", None, [], [answer("test", 3, "龙套，我觉得可以不标", "reviewer")]),
        entry("章医生", "person", sheet={"type": "person", "evidence": None, "decided_on": "2026-10-02"})]}


def test_a_memory_carries_ruled_groups_over_quotes_the_rest_and_leaves_the_no_memory_path_byte_for_byte():
    asked, by_sheet = _pending_places()
    assert rv.questions_md(asked, by_sheet, "astra", "RID", rv.PENDING_WORDING) == TODAY_QUESTIONS
    assert rv.answer_sheet_md(asked, "astra", "RID", rv.PENDING_WORDING, by_sheet) == TODAY_SHEET
    still, carried, carry, prior = rv.split_by_memory(asked, _memory())
    assert list(still) == ["黑肩章", "回音铃", "二号"] and list(carried) == ["画女", "灰塔城 / 灰塔"]
    assert carry["画女"]["reading"] == "person" and carry["画女"]["text"] == "画女" and len(carry["画女"]["answers"]) == 1
    # a two-string group carries over when both strings were ruled and at most one is marked: the marked one is the text
    assert {k: v for k, v in carry["灰塔城 / 灰塔"].items() if k != "answers"} == {
        "text": "灰塔城", "reading": "location", "decider": "user",  # the marked string's own decider
        "carry_over_from": "user 2026-10-03 (dev v2); reviewer astra 2026-10-03 (dev v2)"}
    assert carry["画女"]["decider"] == "user"
    assert set(prior) == {"黑肩章", "二号"} and prior["二号"]["rulings"] == ""
    q = rv.questions_md(still, by_sheet, "astra", "RID", rv.PENDING_WORDING, carried=carried, carry=carry, prior=prior)
    assert ("## 1. 黑肩章（1 处，1 章）\n- 分歧：GPT 拿不准（待审）\n- ch0002：【黑肩章】来了　A 遍：待审【黑肩章】（organization / 不标）；B 遍：不标\n"
            "- 你以前答过：test 第1题（2026-10-03）：制服\n- 以前的裁决：不标 1 处（reviewer astra 2026-10-03 (test v3)）、不计分 1 处"
            "（reviewer astra 2026-10-03 (dev v2)）（不一致或没有统一裁决，所以再问一次）\n- 你的回答：\n\n## 2. 回音铃") in q
    assert "## 3. 二号（1 处，1 章）\n- 分歧：GPT 拿不准（待审）\n- ch0002：【二号】说　A 遍：不标；B 遍：待审【二号】（person / 不标）\n- 你以前答过：test 第3题（2026-10-03）：龙套，我觉得可以不标\n- 你的回答：\n\n" in q
    assert ("## 已答过、按以前的裁决处理（不用再答；要改就在这一列写新答案）\n\n" + rv.CARRIED_NOTE + "\n\n### P1. 画女（2 处，1 章）\n"
            "- ch0001：【画女】笑了　A 遍：待审【画女】（person / 不标）；B 遍：待审【画女】（person / 不标）\n"
            "- 你以前答过：test 第2题（2026-10-03）：画女都作为人物，虽然她是道具；后来补充：照办：人物\n"
            "- 以前的裁决：人物（user 2026-10-03 (test v3)）；这次的 2 处都按此处理\n\n### P2. 灰塔城 / 灰塔（1 处，1 章）\n"
            "- ch0002：去【灰塔城】　A 遍：【灰塔】地点；B 遍：【灰塔城】地点\n- 你以前答过：dev 第3题（2026-10-03）：地点\n"
            "- 以前的裁决：地点（user 2026-10-03 (dev v2); reviewer astra 2026-10-03 (dev v2)）；这次的 1 处都按此处理，名字取“灰塔城”\n\n"
            "## 名字知识表里的名字") in q
    s = rv.answer_sheet_md(still, "astra", "RID", rv.PENDING_WORDING, by_sheet, carried=carried, carry=carry)
    assert s.startswith("# 回答表（astra，RID）") and "| 1 | 黑肩章 | 1 |" in s and "| 3 | 二号 | 1 |" in s and "| 2 | 回音铃 | 1 |" in s
    assert s.endswith("| S1 | 章医生 | 1 | 不标 | 不标 |  |\n\n## 已答过、按以前的裁决处理（不用再答；要改就在这一列写新答案）\n\n" + _HEADER +
                      "| P1 | 画女 | 2 | 待审【画女】（person / 不标） | 待审【画女】（person / 不标） | 已答：人物 （user 2026-10-03 (test v3)） |\n"
                      "| P2 | 灰塔城 / 灰塔 | 1 | 【灰塔】地点 | 【灰塔城】地点 | 已答：地点 （user 2026-10-03 (dev v2); reviewer astra 2026-10-03 (dev v2)） |\n\n")
    export = {"chapters": {"ch0001": {"status": "done", "annotations": [{"pending": True}, {"pending": False}]},
                           "ch0002": {"status": "todo", "annotations": []}}}
    found = [p for ps in list(asked.values()) + list(by_sheet.values()) for p in ps]
    plain = rv.pending_metrics(found, asked, by_sheet, export, ["ch0001", "ch0002"])
    assert list(plain) == ["chapters", "chapters_still_todo", "accepted_marks", "pending_places", "places_by_kind", "places_made_by_the_sheet",
                           "name_groups_to_ask", "places_to_ask", "name_groups_by_sheet", "places_by_sheet", "note"]
    assert (plain["pending_places"], plain["name_groups_to_ask"], plain["places_to_ask"], plain["accepted_marks"]) == (7, 5, 6, 1)
    with_memory = rv.pending_metrics(found, still, by_sheet, export, ["ch0001", "ch0002"], carried, prior)
    assert (with_memory["name_groups_to_ask"], with_memory["places_to_ask"], with_memory["groups_carried_over"],
            with_memory["places_carried_over"], with_memory["groups_asked_with_prior_answer"]) == (3, 3, 2, 3, 2)
    assert parser().parse_args(["pending", "--set", "dev2", "--prefilled", "p.json", "--run-id", "r"]).memory is None


def test_check_answers_classifies_every_row_mechanically_and_changes_nothing_else(tmp_path):
    import json
    from novel_ner.evaluate import sha256
    folder = tmp_path / "data" / "gold" / "dev2" / "review_astra"
    folder.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "answer_memory.json").write_text(json.dumps(
        {"generated_at": "2026-10-04T10:00:00-0700", "format": 2, "entries": list(_memory().values())}, ensure_ascii=False), encoding="utf-8")
    asked, by_sheet = _pending_places()
    asked["拱门"] = [dict(asked["二号"][0], names=["拱门"])]
    (folder / "disagreements.json").write_text(json.dumps(
        {"mode": "pending", "set": "dev2", "run_id": "RID0", "asked": asked, "by_sheet": by_sheet}, ensure_ascii=False), encoding="utf-8")
    rows = [("1", "画女", 2, "说了是人物"), ("2", "黑肩章", 1, "组织"), ("3", "回音铃", 1, "物品"), ("4", "灰塔城 / 灰塔", 1, "说了几百遍了，标不标无所谓"),
            ("5", "二号", 1, "龙套"), ("6", "拱门", 1, "是地点，也可以不标"), ("S1", "章医生", 1, "龙套，啥也不是")]
    sheet = "# 回答表\n\n## 一\n\n" + _HEADER + "".join(f"| {n} | {k} | {c} | a | b | {ans} |\n" for n, k, c, ans in rows)
    (folder / "answer_sheet.md").write_text(sheet, encoding="utf-8")
    before = {f.name: sha256(f) for f in folder.iterdir()}
    out, metrics = rv.check_answers("dev2", "data/gold/dev2/review_astra/answer_sheet.md", "data/gold/dev2/review_astra/disagreements.json",
                                    "configs/answer_memory.json", "RID", root=tmp_path)
    verdicts = {r["no"]: (r["verdict"], r["notes"]) for r in out}
    assert list(verdicts) == [1, 2, 3, 4, 5, 6, "S1"]
    assert verdicts[1] == ("consistent", []) and verdicts[3] == ("new", [])  # 回音铃: nothing in the memory
    assert verdicts[2] == ("new", ["以前答过，但没有可沿用的裁决"])  # mixed rulings: nothing to compare with, the earlier answer is quoted
    assert verdicts[5] == ("unclear", ["以前答过，但没有可沿用的裁决"])  # 二号: a hedged answer earlier, never ruled
    assert verdicts[4] == ("repeat_without_type", [])  # 不标 right after 标 is not a type word; the carry-over (地点) stands
    assert verdicts[6] == ("unclear", ["提到了 2 种：地点、不标"])  # two types named = none named
    assert verdicts["S1"] == ("conflict", [])  # the sheet row says 人物, the answer 啥也不是
    assert metrics["verdicts"] == {"consistent": 1, "conflict": 1, "repeat_without_type": 1, "new": 2, "unclear": 2}
    assert (metrics["rows"], metrics["groups_in_listing"], metrics["groups_without_row"], metrics["rows_naming_several_types"]) == (7, 7, 0, 1)
    report = (folder / "answer_check.md").read_text(encoding="utf-8")
    assert "| 2 | 黑肩章 | test 第1题（2026-10-03）：制服 | 不标 1 处（reviewer astra 2026-10-03 (test v3)）、不计分 1 处（reviewer astra 2026-10-03 (dev v2)） | 组织 | new（以前没答过；以前答过，但没有可沿用的裁决） |" in report
    assert "| 4 | 灰塔城 / 灰塔 | dev 第3题（2026-10-03）：地点 | 地点（user 2026-10-03 (dev v2); reviewer astra 2026-10-03 (dev v2)） | 说了几百遍了，标不标无所谓 | repeat_without_type（没说类型，有以前的裁决） |" in report
    assert "| 6 | 拱门 |  |  | 是地点，也可以不标 | unclear（没说类型，以前也没答过；提到了 2 种：地点、不标） |" in report
    assert "| S1 | 章医生 |  | 人物（name sheet (user 2026-10-02)） | 龙套，啥也不是 | conflict（不一致） |" in report
    # only the report was added; the sheet and the listing are untouched
    assert {f.name: sha256(f) for f in folder.iterdir() if f.name != "answer_check.md"} == before
    assert sorted(f.name for f in folder.iterdir()) == ["answer_check.md", "answer_sheet.md", "disagreements.json"]
    config = json.loads((tmp_path / "runs" / "RID" / "config.json").read_text(encoding="utf-8"))
    assert config["sheet_sha256"] == before["answer_sheet.md"] and config["action"] == "check-answers"
    assert json.loads((tmp_path / "runs" / "RID" / "metrics.json").read_text(encoding="utf-8"))["verdicts"] == metrics["verdicts"]
    (folder / "answer_sheet.md").write_text(sheet + "| 7 | 不在清单里 | 1 | a | b | 物品 |\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a group of the listing"):
        rv.check_answers("dev2", "data/gold/dev2/review_astra/answer_sheet.md", "data/gold/dev2/review_astra/disagreements.json",
                         "configs/answer_memory.json", "RID", root=tmp_path)
    assert rv.named_types("画女都作为人物，虽然她是道具") == ["person", "item"] and rv.named_types("口令而已") == ["none"]
    assert rv.named_types("标不标无所谓") == [] and rv.named_types("可以不标") == ["none"] and rv.named_types("特性") == ["trait"]
    assert rv.named_types("已答：不计分 （reviewer astra 2026-10-03 (test v3)）") == ["unscored"]
    (tmp_path / "configs" / "old.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="older memory file"):
        rv.load_memory(tmp_path / "configs" / "old.json")


def test_a_prior_answer_decides_like_a_direct_one_with_the_carried_ruling_s_decider_and_the_package_skips_it(tmp_path, monkeypatch):
    import json
    import os
    import time
    from novel_ner.snapshot import Chapter, Snapshot
    paras = [("p0001", "画女笑了，黑肩章来了。"), ("p0002", "去灰塔城。")]
    snapshot = Snapshot("s", None, {}, {"ch0001": Chapter("ch0001", 1, paras, "", "h")}, [], [])
    text = "\n".join(t for _, t in paras)
    at = lambda s: (text.find(s), text.find(s) + len(s))
    export = {"set": "dev2", "snapshot_id": "s", "guide_version": "1.1", "chapters": {"ch0001": {
        "status": "todo", "provenance": "model_double_adjudicated",
        "annotations": [{"start": at(s)[0], "end": at(s)[1], "text": s, "pending": True} for s in ("画女", "黑肩章", "灰塔城")]}}}
    mk = lambda s, names, stuck=(): {"chapter_id": "ch0001", "start": at(s)[0], "end": at(s)[1], "names": names, "kinds": ["gpt_pending"],
                                     "stuck": list(stuck), "a_says": "x", "b_says": "y", "sheet_says": "", "context": "c", "offered": [],
                                     "doubts": [], "a": [], "b": []}
    listing = {"mode": "pending", "set": "dev2", "gold_file": "base.json", "created_at": "2026-10-04T10:00:00-0700", "chapters": ["ch0001"],
               "sheet_rows": {}, "asked": {"黑肩章": [mk("黑肩章", ["黑肩章"])]}, "by_sheet": {},
               "carried": {"画女": [mk("画女", ["画女"])], "灰塔城 / 灰塔": [mk("灰塔城", ["灰塔城", "灰塔"], ["boundary"])]},
               "carry": {"画女": {"text": "画女", "reading": "person", "decider": "user", "carry_over_from": "user 2026-10-03 (test v3)", "answers": []},
                         "灰塔城 / 灰塔": {"text": "灰塔城", "reading": "location", "decider": "reviewer",
                                        "carry_over_from": "reviewer astra 2026-10-03 (dev v2)", "answers": []}}}
    p1 = {"no": "P1", "name": "画女", "places": 1, "answer": "已答：人物 （user 2026-10-03 (test v3)）", "handling": "prior", "reading": "person",
          "prior": {"carry_over_from": "user 2026-10-03 (test v3)", "answers": [{"set": "test", "no": 2, "answer": "人物"}]}}
    p2 = {"no": "P2", "name": "灰塔城 / 灰塔", "places": 1, "answer": "说了是地点", "handling": "prior", "reading": "location", "text": "灰塔城",
          "prior": {"carry_over_from": "reviewer astra 2026-10-03 (dev v2)"}}
    q1 = {"no": 1, "name": "黑肩章", "places": 1, "answer": "制服", "handling": "reviewer"}
    answers = {"answered_at": "2026-10-04", "answers": [q1, p1, p2]}
    groups = rv.pending_groups(answers, listing)
    assert [(a["gid"], a["no"], a["handling"]) for a, _ in groups] == [("q01", 1, "reviewer"), ("p01", "P1", "prior"), ("p02", "P2", "prior")]
    # the carried ruling's string and decider are filled in from the listing's carry
    assert [(a.get("text"), (a.get("prior") or {}).get("decider")) for a, _ in groups] == [(None, None), ("画女", "user"), ("灰塔城", "reviewer")]
    for bad, why in (({**p1, "reading": "dungeon"}, "carried 画女 over as person"), ({**p1, "reading": "x"}, "needs a reading"),
                     ({**p1, "prior": {}}, "carry_over_from"), ({**p1, "text": "画"}, "with the string 画女"),
                     ({**p1, "prior": {"carry_over_from": "user 2026-10-01 (dev v2)"}}, "over from 'user 2026-10-03 \\(test v3\\)'"),
                     ({**p1, "prior": {**p1["prior"], "decider": "reviewer"}}, "decided by user, not reviewer")):
        with pytest.raises(SystemExit, match=why):
            rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, bad, p2]}, listing)
    with pytest.raises(SystemExit, match=r"carried-over group\(s\) \['P2'\]"):
        rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, p1]}, listing)
    # a prior entry on an ordinary asked group (today's dev2 and test2 sheets were made without a memory) is accepted
    # when the memory carries the group over the same way; without a memory, or without a carry-over there, it is refused
    plain = {**listing, "asked": {**listing["asked"], "画女": listing["carried"]["画女"]}, "carried": {}, "carry": {}}
    mem = _memory()
    got = rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, {**p1, "no": 2}]}, plain, memory=mem)
    assert [(a["gid"], a.get("text"), (a.get("prior") or {}).get("decider")) for a, _ in got] == [("q01", None, None), ("q02", "画女", "user")]
    with pytest.raises(SystemExit, match="checked against the memory"):
        rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, {**p1, "no": 2}]}, plain)
    with pytest.raises(SystemExit, match="no carry-over for 画女"):
        rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, {**p1, "no": 2}]}, plain, memory={k: v for k, v in mem.items() if k != "画女"})
    with pytest.raises(SystemExit, match="the memory carries 画女 over as person, not item"):
        rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, {**p1, "no": 2, "reading": "item"}]}, plain, memory=mem)
    # a two-string prior entry without "text" takes the carried string (the ruling is about that string)
    assert [a["text"] for a, _ in rv.pending_groups({"answered_at": "2026-10-04", "answers": [q1, p1, {k: v for k, v in p2.items() if k != "text"}]},
                                                     listing) if a["handling"] == "prior"] == ["画女", "灰塔城"]
    (tmp_path / "guide.md").write_text("规则", encoding="utf-8")
    pkg = tmp_path / "adj"
    places = rv.build_package(answers, listing, export, ["ch0001"], snapshot, pkg, tmp_path / "guide.md", answers_config="a.json")
    assert [(q["id"], q["name"]) for q in places] == [("q01-1", "黑肩章")]  # the prior groups are not packaged
    (pkg / "pass_astra.json").write_text(json.dumps({"decisions": [{"id": "q01-1", "type": "none", "text": "", "rule": "A1", "reason": "r"}]}),
                                         encoding="utf-8")
    os.utime(pkg / "pass_astra.json", (1790000000, 1790000000))
    (tmp_path / "base.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rv, "REPO_ROOT", tmp_path)
    out = rv.pending_decisions(answers, listing, "base.json", export, snapshot, [pkg], "astra", "a.json")
    on = time.strftime("%Y-%m-%d", time.localtime(1790000000))
    # a kept ruling is applied with its own decider (one gold.py revise accepts and evaluate.gold_origin counts by)
    assert [(p["start"], p["text"], p["reading"], p["by"], p["decider"], p["reason"]) for p in out["places"]] == [
        (0, "画女", "person", "prior ruling: user 2026-10-03 (test v3)", "user",
         "回答表 P1：已答：人物 （user 2026-10-03 (test v3)）；以前的裁决：人物（user 2026-10-03 (test v3)）"),
        (at("灰塔城")[0], "灰塔城", "location", "prior ruling: reviewer astra 2026-10-03 (dev v2)", "reviewer",
         "回答表 P2：说了是地点；以前的裁决：地点（reviewer astra 2026-10-03 (dev v2)）"),
        (at("黑肩章")[0], "黑肩章", "none", f"astra {on}（按规则 v1.1，依据用户的说明）", "reviewer", "A1：r（q01-1，adj；用户原话：制服）")]
    from novel_ner.gold import DECIDERS
    assert all(p["decider"] in DECIDERS for p in out["places"])
    assert ("the user's direct answers (a.json; 0 places), the reviewer astra's rulings on the hedged answers and on the name-sheet "
            "names' places with the sheet's row (1 places;") in out["description"]
    assert "earlier rulings the user kept (2 places, by 'prior ruling: ...' with the carried ruling's decider: reviewer 1, user 1;" in out["description"]
    assert out["decided_at"] == "2026-10-04"
    # the same memory a later decisions file is built into: a place kept by a prior ruling reads as "carried over"
    assert rv.who_decided({"by": "prior ruling: user 2026-10-03 (test v3)", "decider": "user", "tag": "dev2 v1"}) == "carried over (dev2 v1)"


def test_a_later_decisions_file_of_the_same_set_supersedes_the_earlier_ruling_of_the_same_place(tmp_path):
    """test2 v2 re-decided two 天堂之梯 places (none -> location): the memory carries the current version, not a
    mixed ruling; a place of the earlier file that the later file does not decide again is kept."""
    import json
    cfg = tmp_path / "configs"
    cfg.mkdir()
    place = lambda cid, start, text, reading, by: {"chapter_id": cid, "start": start, "end": start + len(text), "text": text,
                                                   "reading": reading, "decider": "user", "by": by, "reason": "r"}
    (cfg / "gold_test2_v1_decisions.json").write_text(json.dumps({"places": [
        place("ch1345", 10, "天堂之梯", "none", "user 2026-10-04"), place("ch1346", 20, "天堂之梯", "none", "user 2026-10-04"),
        place("ch1300", 5, "雅族", "none", "user 2026-10-04")]}, ensure_ascii=False), encoding="utf-8")
    (cfg / "gold_test2_v2_decisions.json").write_text(json.dumps({"places": [
        place("ch1345", 10, "天堂之梯", "location", "user 2026-10-04 (v2)"), place("ch1346", 20, "天堂之梯", "location", "user 2026-10-04 (v2)")]},
        ensure_ascii=False), encoding="utf-8")
    sheet = tmp_path / "sheet.tsv"
    sheet.write_text("名字\t类型\t证据章\t备注\n", encoding="utf-8")
    m = rv.build_memory([], ["test2=configs/gold_test2_v1_decisions.json", "test2=configs/gold_test2_v2_decisions.json"], sheet, root=tmp_path)
    e = {x["text"]: x for x in m["entries"]}
    assert e["天堂之梯"]["carry_over"] == "location" and [r["source"] for r in e["天堂之梯"]["rulings"]] == ["configs/gold_test2_v2_decisions.json"]
    assert e["雅族"]["carry_over"] == "none"
    v1 = next(s for s in m["sources"] if s["path"] == "configs/gold_test2_v1_decisions.json")
    assert v1["places"] == 3 and v1["places_superseded_by_a_later_file_of_the_set"] == 2
    # the order given decides what is later; the same two files the other way round keep v1's reading
    m2 = rv.build_memory([], ["test2=configs/gold_test2_v2_decisions.json", "test2=configs/gold_test2_v1_decisions.json"], sheet, root=tmp_path)
    assert {x["text"]: x for x in m2["entries"]}["天堂之梯"]["carry_over"] == "none"
