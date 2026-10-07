import hashlib
import json
from types import SimpleNamespace

import pytest

from novel_ner import gpt_api, llm_task, qa
from novel_ner.gold import FORMAT
from novel_ner.gpt_api import Budget
from novel_ner.llm_task import locate
from novel_ner.snapshot import Chapter, Snapshot

# chapter text: "林小雨看着林小雨的影子。\n她去了灰塔城，灰塔城很大。\n大学生走了。"
# 林小雨 [0,3) [5,8), 影子 [9,11), 灰塔城 [16,19) [20,23), 大学生 [27,30)
PARAS = [("p0001", "林小雨看着林小雨的影子。"), ("p0002", "她去了灰塔城，灰塔城很大。"), ("p0003", "大学生走了。")]
NUMBERS = list(range(781, 788))  # 7 chapters: two packages of 5 and 2


def chapter(n):
    return Chapter(f"ch{n:04d}", n, PARAS, f"第{n}章", f"h{n}")


def snap():
    return Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in map(chapter, NUMBERS)}, [], [])


def gold_export(s, set_name="dev2"):
    chapters = {}
    for c in s.chapters.values():
        chapters[c.chapter_id] = {
            "content_hash": c.content_hash, "text_sha256": hashlib.sha256(c.text.encode("utf-8")).hexdigest(),
            "status": "done", "provenance": "model_double_adjudicated", "checks": {"adjudicated": True},
            "annotations": [{"start": 0, "end": 3, "text": "林小雨", "type": "person", "pending": False},
                            {"start": 5, "end": 8, "text": "林小雨", "type": "person", "pending": False},
                            {"start": 16, "end": 19, "text": "灰塔城", "type": "location", "pending": False}],
            "unscored": [{"start": 20, "end": 23, "text": "灰塔城", "reason": "判不了", "by": "astra 2026-10-04"}],
            "decided_places": [{"start": 9, "end": 11, "text": "影子", "reading": "none", "decider": "reviewer"},
                               {"start": 20, "end": 23, "text": "灰塔城", "reading": "unscored", "decider": "reviewer"}]}
    chapters["ch0781"]["decided_places"].append({"start": 27, "end": 30, "text": "大学生", "reading": "none", "decider": "user"})
    return {"format": FORMAT, "set": set_name, "snapshot_id": "snap-test", "guide_version": "1.1", "precedents": [],
            "chapters": chapters}


def place(cid, start, end, text, reading, decider):
    by = "astra 2026-10-04（按规则 v1.1，依据用户的说明）" if decider == "reviewer" else "user 2026-10-04"
    return {"chapter_id": cid, "start": start, "end": end, "text": text, "reading": reading, "by": by, "decider": decider,
            "reason": "A2：影子（reviewer ruling text）" if decider == "reviewer" else "回答表 2：身份"}


def decisions(s):
    places = []
    for cid in s.chapters:
        places.append(place(cid, 9, 11, "影子", "none", "reviewer"))
        places.append(place(cid, 0, 3, "林小雨", "person", "user"))  # a typed place: not an exclusion
    places.append(place("ch0781", 20, 23, "灰塔城", "unscored", "reviewer"))
    places.append(place("ch0781", 27, 30, "大学生", "none", "user"))
    return {"description": "test", "base": "data/gold/dev2/incoming/prefilled.json", "base_sha256": "abc", "decided_by": "user",
            "decided_at": "2026-10-04", "places": places}


def answers():
    return {"description": "test", "answered_at": "2026-10-04", "answers": [
        {"no": 1, "name": "影子", "places": 7, "answer": "就是影子，不是名字", "handling": "reviewer", "reading": None,
         "note2": "补充：按 Astra 2026-10-03 的裁决处理"},
        {"no": 2, "name": "大学生 / 学生", "places": 1, "answer": "身份", "handling": "direct", "reading": "none", "text": "大学生"}]}


def listing(s):
    return {"mode": "pending", "set": "dev2", "gold_file": "data/gold/dev2/incoming/prefilled.json", "gold_sha256": "abc",
            "chapters": list(s.chapters),
            "asked": {"影子": [{"chapter_id": cid, "start": 9, "end": 11, "names": ["影子"]} for cid in s.chapters]},
            "by_sheet": {"灰塔城": [{"chapter_id": "ch0781", "start": 20, "end": 23, "names": ["灰塔城"]}]}, "sheet_rows": {}}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository layout under tmp_path: the decisions and answers configs, the dev2 answers file and listing, the QA
    instructions and the guide; nothing under runs/."""
    s = snap()
    monkeypatch.setattr(qa, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(qa, "git_state", lambda: {"commit": "abc", "dirty": False})
    (tmp_path / "docs").mkdir()
    instructions = tmp_path / "docs" / "qa.md"
    instructions.write_text("质检说明 qa_v1", encoding="utf-8")
    monkeypatch.setattr(qa, "QA_INSTRUCTIONS", instructions)
    guide = tmp_path / "docs" / "guide.md"
    guide.write_text("规则 1.1", encoding="utf-8")
    monkeypatch.setitem(llm_task.GUIDES, "1.1", guide)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "decisions.json").write_text(json.dumps(decisions(s), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "configs" / "answers.json").write_text(json.dumps(answers(), ensure_ascii=False), encoding="utf-8")
    gold = tmp_path / "data" / "gold" / "dev2"
    (gold / "review_astra").mkdir(parents=True)
    (gold / "answers.json").write_text(json.dumps(gold_export(s), ensure_ascii=False), encoding="utf-8")
    (gold / "review_astra" / "disagreements.json").write_text(json.dumps(listing(s), ensure_ascii=False), encoding="utf-8")
    return tmp_path, s


def build(root, s, **kw):
    args = {"set_name": "dev2", "decisions_file": "configs/decisions.json", "answers_file": "configs/answers.json",
            "gold_file": "data/gold/dev2/answers.json", "out_dir": "data/llm_tasks/qa_dev2",
            "listing_file": "data/gold/dev2/review_astra/disagreements.json", "snapshot": s, "root": root, **kw}
    return qa.build(**args)


def test_place_to_paragraph_and_occurrence_round_trip():
    ch = chapter(781)
    for start, end, text in ((0, 3, "林小雨"), (5, 8, "林小雨"), (9, 11, "影子"), (16, 19, "灰塔城"), (20, 23, "灰塔城"), (27, 30, "大学生")):
        pid, paragraph, offset = qa.paragraph_at(ch, start, end)
        occurrence = qa.occurrence_at(paragraph, text, offset)
        assert locate(paragraph, text, occurrence) == offset  # what evaluate.llm_predictions / llm_task.convert would find
        assert ch.text[start:end] == text
    assert qa.occurrence_at("她去了灰塔城，灰塔城很大。", "灰塔城", 7) == 2
    assert qa.occurrence_at("AAAA", "AA", 2) == 3  # overlapping starts each count, as locate counts them
    with pytest.raises(ValueError, match="crosses a paragraph break"):
        qa.paragraph_at(ch, 11, 14)
    with pytest.raises(ValueError):
        qa.occurrence_at("她去了灰塔城", "灰塔城", 1)


def test_packages_hold_every_excluded_place_once_and_name_no_decider(repo):
    root, s = repo
    summary = build(root, s)
    assert summary["packages"] == ["qa_dev2_b01", "qa_dev2_b02"] and summary["chapters"] == 7
    assert summary["places"] == 9 and summary["by_ruling"] == {"none": 8, "unscored": 1}  # 7 影子 + 大学生 + 灰塔城; typed places left out
    assert summary["words_from"] == {"listing": 7, "name": 1, "None": 1}  # 大学生 by name; the sheet group has no answer
    assert summary["confirmed_by_decided_places"] == 9 and summary["system_output_read"] is False
    assert summary["decisions_sha256"] and summary["gold_sha256"] and summary["listing_sha256"]
    out = root / "data" / "llm_tasks" / "qa_dev2"
    assert json.loads((out / "build.json").read_text(encoding="utf-8"))["places"] == 9
    seen = []
    for name in summary["packages"]:
        folder = out / name
        assert {f.name for f in folder.iterdir()} == {"instructions.md", "guide.md", "chapters.md", "places.md", "task.json", "README.md"}
        task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        assert task["kind"] == "qa_exclusions" and task["set"] == "dev2" and task["instructions_version"] == "qa_v1"
        assert task["system_output_read"] is False and task["test2_seal"] is None and task["code_commit"] == "abc"
        assert (folder / "chapters.md").read_text(encoding="utf-8") == llm_task.chapters_markdown(s, task["chapters"])
        assert (folder / "instructions.md").read_text(encoding="utf-8") == "质检说明 qa_v1"
        places_md = (folder / "places.md").read_text(encoding="utf-8")
        for p in task["places"]:
            seen.append((p["chapter_id"], p["start"], p["end"]))
            assert p["chapter_id"] in task["chapters"]
            assert places_md.count(f"chapter_id {p['chapter_id']}｜paragraph_id {p['paragraph_id']}｜text 「{p['text']}」｜occurrence {p['occurrence']}") == 1
            assert "decider" not in p and "by" not in p and "reason" not in p
        # nothing of the deciders in what the checker reads (places.md) or in the places record; nothing but the answer string
        for word in ("astra", "reviewer", "decider", "Astra", "补充"):
            assert word not in places_md and word not in json.dumps(task["places"], ensure_ascii=False)
    assert len(seen) == 9 and len(set(seen)) == 9
    b01 = json.loads((out / "qa_dev2_b01" / "task.json").read_text(encoding="utf-8"))
    assert b01["chapters"] == ["ch0781", "ch0782", "ch0783", "ch0784", "ch0785"] and len(b01["places"]) == 7
    by_text = {(p["text"], p["chapter_id"]): p for p in b01["places"]}
    assert by_text[("影子", "ch0781")]["words"] == "就是影子，不是名字"
    assert by_text[("大学生", "ch0781")]["words"] == "身份" and by_text[("大学生", "ch0781")]["words_from"] == "name"
    assert by_text[("灰塔城", "ch0781")] | {} == {**by_text[("灰塔城", "ch0781")], "words": "", "words_from": None, "ruling": "unscored",
                                               "paragraph_id": "p0002", "occurrence": 2}
    places_md = (out / "qa_dev2_b01" / "places.md").read_text(encoding="utf-8")
    assert "- 本段：她去了灰塔城，【灰塔城】很大。" in places_md and "裁决：不计分" in places_md
    assert "- 读者的原话：（没有）" in places_md
    with pytest.raises(SystemExit, match="never overwritten"):
        build(root, s)


def test_a_place_the_answers_file_holds_otherwise_is_refused(repo):
    root, s = repo
    d = decisions(s)
    d["places"].append(place("ch0782", 5, 8, "林小雨", "none", "user"))  # settled as person in the answers file
    (root / "configs" / "decisions.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="ruled none, but the answers file holds a mark"):
        build(root, s)
    d = decisions(s)
    d["places"].append(place("ch0782", 16, 19, "灰塔城", "unscored", "reviewer"))  # a settled mark, not an unscored range
    (root / "configs" / "decisions.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="ruled unscored, but no unscored range"):
        build(root, s)
    assert not (root / "data" / "llm_tasks").exists()  # nothing was written


def test_guards_on_sets_inputs_and_files(repo):
    root, s = repo
    with pytest.raises(SystemExit, match="the QA covers"):
        build(root, s, set_name="dev")
    (root / "runs" / "x").mkdir(parents=True)
    (root / "runs" / "x" / "answers.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="no system output"):
        build(root, s, answers_file="runs/x/answers.json")
    (root / "configs" / "predictions_dev2.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="no system output"):
        build(root, s, answers_file="configs/predictions_dev2.json")
    with pytest.raises(SystemExit, match="lies under data/gold/dev2, not the test2"):
        build(root, s, set_name="test2")
    other = root / "data" / "gold" / "test2"
    other.mkdir()
    (other / "answers.json").write_text(json.dumps(gold_export(s), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="says set 'dev2', not 'test2'"):
        build(root, s, set_name="test2", gold_file="data/gold/test2/answers.json", listing_file=None)
    bad = {**listing(s), "gold_sha256": "other"}
    (root / "configs" / "listing.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="another file"):
        build(root, s, listing_file="configs/listing.json")
    assert not (root / "data" / "llm_tasks").exists()


def reply_for(task, verdict="agree", rule="A2"):
    return {"instructions_version": "qa_v1", "guide_version": "1.1", "places": [
        {"chapter_id": p["chapter_id"], "paragraph_id": p["paragraph_id"], "text": p["text"], "occurrence": p["occurrence"],
         "verdict": "cannot_tell" if p["ruling"] == "unscored" else verdict, "rule": rule,
         "would_mark_as": "item" if verdict == "disagree" else None, "reason": "理由"} for p in task["places"]]}


def test_report_counts_valid_replies_and_flags_invalid_ones(repo):
    root, s = repo
    build(root, s)
    out = root / "data" / "llm_tasks" / "qa_dev2"
    b01 = json.loads((out / "qa_dev2_b01" / "task.json").read_text(encoding="utf-8"))
    (out / "qa_dev2_b01" / "pass_qa.json").write_text(json.dumps(reply_for(b01), ensure_ascii=False), encoding="utf-8")
    (out / "qa_dev2_b01" / "api_qa_attempt1.json").write_text(json.dumps(
        {"record": {"model": "gpt-test", "effort": "xhigh", "seconds": 12.5}, "output_text": "..."}), encoding="utf-8")
    b02 = json.loads((out / "qa_dev2_b02" / "task.json").read_text(encoding="utf-8"))
    bad = reply_for(b02, verdict="disagree")
    bad["places"][0]["verdict"] = "maybe"  # a bad verdict
    bad["places"].append({**bad["places"][1], "paragraph_id": "p0003", "text": "大学生", "occurrence": 1})  # a place not listed
    (out / "qa_dev2_b02" / "pass_qa.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    m = qa.report("dev2", "20261004-qa-dev2", "data/llm_tasks/qa_dev2", root=root)
    assert m["places_listed"] == 9 and m["places_judged"] == 7 and m["places_not_counted"] == 2
    assert m["invalid_replies"] == 1 and m["packages_without_reply"] == 0 and m["packages"] == 2
    assert m["by_verdict"] == {"agree": 6, "cannot_tell": 1} and m["by_ruling"] == {"none": 8, "unscored": 1}
    assert m["by_ruling_and_verdict"] == {"none": {"agree": 6}, "unscored": {"cannot_tell": 1}} and m["would_mark_as"] == {}
    assert m["model"] == "gpt-test" and m["by_set"]["dev2"]["places_judged"] == 7
    run = root / "runs" / "20261004-qa-dev2"
    metrics = (run / "metrics.json").read_text(encoding="utf-8")
    assert json.loads(metrics)["invalid_replies"] == 1
    for word in ("影子", "就是影子", "大学生", "理由"):  # counts only, no text
        assert word not in metrics
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["decisions_sha256"] == b01["decisions_sha256"] and config["inputs_unchanged_since_build"]["decisions_sha256"] is True
    assert [p["status"] for p in config["packages"]] == ["ok", "invalid"] and config["packages"][0]["model"] == "gpt-test"
    assert config["packages"][1]["pass_sha256"] and config["system_output_read"] is False
    assert config["prompt"] == {**qa.QA_PROMPT, "files": list(qa.QA_PROMPT["files"])}
    text = (root / "data" / "gold" / "dev2" / "review_qa" / "qa_report.md").read_text(encoding="utf-8")
    assert "不改任何答案文件" in text and "回复无效" in text and "a place that was not listed" in text and "verdict 'maybe'" in text
    assert "### 影子（7 处；裁决：不标 7；判定：同意 5、不同意 0、说不准 0、未计入 2）" in text
    assert "- 读者的原话：就是影子，不是名字" in text and "| ch0781 | p0002 | 2 | 不计分 | 说不准 | A2 |" in text
    assert "| 不标 | 8 | 6 | 0 | 0 | 2 |" in text and "| 不计分 | 1 | 0 | 0 | 1 | 0 |" in text
    # a package without a reply is listed as such, its places not counted
    (out / "qa_dev2_b02" / "pass_qa.json").unlink()
    m = qa.report("dev2", "20261004-qa-dev2", "data/llm_tasks/qa_dev2", root=root)
    assert m["packages_without_reply"] == 1 and m["invalid_replies"] == 0 and m["places_judged"] == 7


def test_validate_reply_flags_missing_duplicate_and_bad_entries():
    task = {"places": [{"chapter_id": "ch0781", "paragraph_id": "p0001", "text": "影子", "occurrence": 1, "ruling": "none"},
                       {"chapter_id": "ch0781", "paragraph_id": "p0003", "text": "大学生", "occurrence": 1, "ruling": "none"}]}
    ok = {"instructions_version": "qa_v1", "places": [
        {"chapter_id": "ch0781", "paragraph_id": "p0001", "text": "影子", "occurrence": 1, "verdict": "agree", "would_mark_as": None},
        {"chapter_id": "ch0781", "paragraph_id": "p0003", "text": "大学生", "occurrence": 1, "verdict": "disagree", "would_mark_as": "person"}]}
    got, problems = qa.validate_reply(task, ok)
    assert problems == [] and len(got) == 2
    got, problems = qa.validate_reply(task, {"instructions_version": "qa_v1", "places": ok["places"][:1]})
    assert any("without a verdict" in p for p in problems)
    got, problems = qa.validate_reply(task, {"instructions_version": "qa_v1", "places": ok["places"] + ok["places"][:1]})
    assert any("listed twice" in p for p in problems)
    got, problems = qa.validate_reply(task, {"instructions_version": "qa_v1", "places": [{**ok["places"][0], "would_mark_as": "thing"}, ok["places"][1]]})
    assert any("would_mark_as" in p for p in problems)
    assert qa.validate_reply(task, {"places": "none"})[1] == ["no 'places' list in the reply"]
    assert any("instructions_version" in p for p in qa.validate_reply(task, {"instructions_version": "v1", "places": ok["places"]})[1])


class FakeAPI:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        assert kwargs["store"] is False
        self.prompts.append((kwargs["input"], kwargs["prompt_cache_key"]))
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500, input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=300))
        return SimpleNamespace(id="resp_1", model="gpt-test", status="completed", usage=usage, output_text=self.replies.pop(0))


def test_run_sends_the_qa_prompt_through_gpt_api_and_imports_nothing(repo, monkeypatch):
    root, s = repo
    build(root, s)
    out = root / "data" / "llm_tasks" / "qa_dev2"
    task = json.loads((out / "qa_dev2_b01" / "task.json").read_text(encoding="utf-8"))
    api = FakeAPI([json.dumps(reply_for(task), ensure_ascii=False)])
    budget = Budget(cap=10, ledger=root / "runs" / "ledger.json")
    result = gpt_api.run_pass(api, out / "qa_dev2_b01", qa.RUN, "gpt-test", "xhigh", budget, "r1", prompt=qa.QA_PROMPT)
    assert result["status"] == "ok" and (out / "qa_dev2_b01" / "pass_qa.json").exists()
    assert not (out / "qa_dev2_b01" / "draft_qa.json").exists()  # not an annotation answer: nothing imported
    prompt, cache_key = api.prompts[0]
    assert prompt.startswith(qa.QA_PROMPT["request"]) and cache_key == "novel-ner-qa"
    assert prompt.index("===== instructions.md") < prompt.index("===== guide.md") < prompt.index("===== chapters.md") < prompt.index("===== places.md")
    assert "读者的原话：就是影子，不是名字" in prompt and "规则 1.1" in prompt
    assert budget.entries[0]["prompt_version"] == "qa_prompt_v1" and budget.entries[0]["task"] == "qa_dev2_b01"
    # the default prompt (the drafting passes, codex_run and claude_run) is unchanged
    assert gpt_api.build_prompt(out / "qa_dev2_b01").startswith(gpt_api.REQUEST)
    assert "===== places.md" not in gpt_api.build_prompt(out / "qa_dev2_b01")
    # run hands every package of the set to run_many with the QA prompt, in order
    sent = {}
    monkeypatch.setattr(qa.gpt_api, "run_many", lambda jobs, model, effort, cap, run_id, workers, prompt: sent.update(
        jobs=jobs, model=model, cap=cap, run_id=run_id, prompt=prompt))
    qa.run("dev2", "r2", "data/llm_tasks/qa_dev2", model="gpt-6.1-sol", root=root)
    assert [(d.name, r) for d, r in sent["jobs"]] == [("qa_dev2_b01", "qa"), ("qa_dev2_b02", "qa")]
    assert sent["prompt"] is qa.QA_PROMPT and sent["cap"] == 30.0 and sent["model"] == "gpt-6.1-sol"
    with pytest.raises(SystemExit, match="the QA covers"):
        qa.run("dev", "r3", "data/llm_tasks/qa_dev2", root=root)
