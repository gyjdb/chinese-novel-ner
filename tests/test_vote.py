import json

import pytest
from test_tagger import training_setup

from novel_ner import annotate, tagger, vote
from novel_ner.evaluate import sha256, write_jsonl
from novel_ner.snapshot import Chapter, Snapshot, read_jsonl
from novel_ner.vote import vote_run, vote_types
from novel_ner.weak import ids_sha256


def row(cid, start, text, type_, score=0.9):
    return {"chapter_id": cid, "start": start, "end": start + len(text), "text": text, "type": type_, "score": score}


def test_vote_changes_a_type_by_majority_keeps_it_on_a_tie_and_when_the_text_is_unseen():
    material = [row("ch0001", 0, "回音铃", "item"), row("ch0001", 10, "回音铃", "item"), row("ch0003", 0, "回音铃", "item"),
                row("ch0003", 5, "回音铃", "person"),  # item 3, person 1
                row("ch0001", 20, "汀城", "location"), row("ch0003", 9, "汀城", "item"),  # location 1, item 1
                row("ch0001", 30, "唐医生", "organization"),  # organization 1
                row("ch0001", 40, "徐获", "person"), row("ch0003", 20, "徐获", "person"),
                row("ch0003", 30, "徐获", "item"), row("ch0003", 40, "徐获", "item")]  # person 2, item 2
    target = [row("ch0781", 0, "回音铃", "person"),  # item 3 against person 1 + its own 1: item
              row("ch0781", 5, "汀城", "item"),  # item 1 + own 1 against location 1: stays item (no tie)
              row("ch0781", 9, "唐医生", "person"),  # organization 1 against own 1: a tie keeps the model's type
              row("ch0781", 14, "徐获", "item"),  # item 2 + own 1 against person 2: stays item
              row("ch0781", 20, "NPC", "organization"),  # unseen in the material: kept
              row("ch0782", 0, "徐获", "person"), row("ch0782", 5, "徐获", "person"),  # person 4 against item 2: stay
              row("ch0783", 0, "徐获", "item"),  # item 3 against person 2: stays (its own chapter's vote, not ch0782's)
              row("ch0784", 0, "徐获", "person"), row("ch0784", 5, "徐获", "person"), row("ch0784", 10, "徐获", "item"),
              # person 4 against item 3: the item row changes
              row("ch0785", 0, "汀城", "person")]  # location 1, item 1, person 1: a three-way tie keeps person
    before = json.dumps(target)
    out, counts = vote_types(target, material)
    assert json.dumps(target) == before and [r["chapter_id"] for r in out] == [r["chapter_id"] for r in target]
    assert [r["type"] for r in out] == ["item", "item", "person", "item", "organization", "person", "person", "item",
                                        "person", "person", "person", "person"]
    assert all({k: v for k, v in a.items() if k != "type"} == {k: v for k, v in b.items() if k != "type"} for a, b in zip(out, target))
    assert counts == {"spans_seen": 12, "spans_with_material": 11, "spans_unseen": 1, "spans_changed": 2, "ties": 2,
                      "changes_by_type": {"item>person": 1, "person>item": 1}}
    assert vote_types([], material) == ([], {"spans_seen": 0, "spans_with_material": 0, "spans_unseen": 0, "spans_changed": 0,
                                             "ties": 0, "changes_by_type": {}})


def vote_setup(tmp_path, monkeypatch):
    """test_tagger's tiny training setup with the vote module pointed at it; trains run r1."""
    cfg, snapshot, _ = training_setup(tmp_path, monkeypatch)
    order = lambda c: annotate.test_chapters(c["chapters_from_order"], c["n"]) if "chapters_from_order" in c else ["ch0002"]
    monkeypatch.setattr(tagger, "chapter_order", order)
    monkeypatch.setattr(vote, "chapter_order", order)
    monkeypatch.setattr(vote, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(vote, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(vote, "environment", lambda: {"python": "test"})
    tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    return cfg, snapshot


def fake_predict_run(root, run_id, model_run, chapters, rows, after_780=0, after_990=0, chunking=None):
    """A predict run folder as predict_run writes it, with the given rows."""
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    config = {"run_id": run_id, "model_run": model_run, "chapters": chapters, **({"chunking": chunking} if chunking else {})}
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_jsonl(d / "predictions.jsonl", rows)
    (d / "metrics.json").write_text(json.dumps({"run_id": run_id, "chapters_after_780": after_780, "chapters_after_990": after_990,
                                                "predictions_sha256": sha256(d / "predictions.jsonl")}), encoding="utf-8")
    return d


def test_vote_run_predicts_the_material_itself_and_writes_numbers_only_records(tmp_path, monkeypatch):
    cfg, snapshot = vote_setup(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    tagger.predict_run("runs/r1", ["ch0781"], "r2-test", snapshot=snapshot)
    metrics = vote_run("runs/r1", "runs/r2-test", "r3-c2-test", snapshot=snapshot)
    # the material: chapters 1-780 of the snapshot minus the dev chapter ch0002, predicted into its own run folder
    material = json.loads((runs / "r3-c2-test-material" / "config.json").read_text(encoding="utf-8"))
    assert set(material["chapters"]) == {"ch0001", "ch0003"} and material["model_run"] == "r1" and material["chunking"] == "paragraph500"
    assert (runs / "r3-c2-test-material" / "predictions.jsonl").exists() and (runs / "r3-c2-test-material" / "metrics.json").exists()
    rec = json.loads((runs / "r3-c2-test" / "config.json").read_text(encoding="utf-8"))
    assert rec["material_run"]["run_id"] == "r3-c2-test-material" and rec["material_run"]["predicted_here"]
    assert rec["material_run"]["chapters"] == 2 and rec["material_run"]["chapter_ids_sha256"] == ids_sha256(["ch0001", "ch0003"])
    assert rec["material_run"]["equals_model_training_chapters"] and rec["material_run"]["dev_chapters_excluded"] == 1
    assert rec["predictions_run"]["run_id"] == "r2-test" and rec["predictions_run"]["chapters"] == ["ch0781"]
    assert rec["model_run"] == "r1" and rec["model_run_config_sha256"] == sha256(runs / "r1" / "config.json")
    assert "dev2" in rec["material_rule"] and "tie" in rec["vote_rule"]
    for name in ("config.json", "metrics.json"):
        text = (runs / "r3-c2-test" / name).read_text(encoding="utf-8")
        assert not set(text) & set("徐获回音铃唐医生汀城") and tmp_path.name not in text  # no novel text, no local path
    before = read_jsonl(runs / "r2-test" / "predictions.jsonl")
    after = read_jsonl(runs / "r3-c2-test" / "predictions.jsonl")
    assert len(after) == len(before) == metrics["spans_seen"] == metrics["outputs"]
    assert all({k: v for k, v in a.items() if k != "type"} == {k: v for k, v in b.items() if k != "type"} for a, b in zip(after, before))
    assert metrics["spans_with_material"] + metrics["spans_unseen"] == metrics["spans_seen"]
    assert metrics["spans_changed"] == sum(a["type"] != b["type"] for a, b in zip(after, before)) == sum(metrics["changes_by_type"].values())
    assert metrics["material_chapters"] == 2 and metrics["predictions_sha256"] == sha256(runs / "r3-c2-test" / "predictions.jsonl")
    assert set(metrics["outputs_by_type"]) == set(metrics["outputs_by_type_before"]) and sum(metrics["outputs_by_type"].values()) == len(after)
    # the material run can be given to a later vote; the result is the same
    again = vote_run("runs/r1", "runs/r2-test", "r4-c2-test", material_run="runs/r3-c2-test-material", snapshot=snapshot)
    assert again["predictions_sha256"] == metrics["predictions_sha256"]
    assert not json.loads((runs / "r4-c2-test" / "config.json").read_text(encoding="utf-8"))["material_run"]["predicted_here"]
    with pytest.raises(SystemExit, match="already has results"):
        vote_run("runs/r1", "runs/r2-test", "r4-c2-test", material_run="runs/r3-c2-test-material", snapshot=snapshot)


def test_vote_run_applies_the_rule_to_the_files_and_follows_the_target_chunking(tmp_path, monkeypatch):
    cfg, snapshot = vote_setup(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    material = [row("ch0001", 0, "徐获", "person"), row("ch0001", 8, "徐获", "person"), row("ch0003", 0, "汀城", "location")]
    fake_predict_run(tmp_path, "m-ok", "r1", ["ch0001", "ch0003"], material)
    fake_predict_run(tmp_path, "t-rows", "r1", ["ch0781"], [row("ch0781", 0, "徐获", "item"), row("ch0781", 3, "NPC", "organization"),
                                                            row("ch0781", 7, "汀城", "location")], after_780=1)
    m = vote_run("runs/r1", "runs/t-rows", "r9-c2-test", material_run="runs/m-ok", snapshot=snapshot)
    assert (m["spans_seen"], m["spans_with_material"], m["spans_unseen"], m["spans_changed"], m["ties"]) == (3, 2, 1, 1, 0)
    assert m["changes_by_type"] == {"item>person": 1} and m["material_spans"] == 3 and m["material_distinct_texts"] == 2
    assert m["material_texts_with_more_than_one_type"] == 0
    assert m["outputs_by_type_before"]["item"] == 1 and m["outputs_by_type"]["item"] == 0 and m["outputs_by_type"]["person"] == 1
    assert [r["type"] for r in read_jsonl(runs / "r9-c2-test" / "predictions.jsonl")] == ["person", "organization", "location"]
    # a target run made with a chunking variant: the material is predicted the same way, so the run id needs "c1" too
    fake_predict_run(tmp_path, "t-c1", "r1", ["ch0781"], [], after_780=1, chunking="offset250")
    with pytest.raises(SystemExit, match="c1"):
        vote_run("runs/r1", "runs/t-c1", "r8-c2-test", snapshot=snapshot)
    assert not (runs / "r8-c2-test-material").exists() and not (runs / "r8-c2-test").exists()
    vote_run("runs/r1", "runs/t-c1", "r8-c1-c2-test", snapshot=snapshot)
    assert json.loads((runs / "r8-c1-c2-test-material" / "config.json").read_text(encoding="utf-8"))["chunking"] == "offset250"
    assert json.loads((runs / "r8-c1-c2-test" / "config.json").read_text(encoding="utf-8"))["material_run"]["chunking"] == "offset250"
    # a given material run must have the target's chunking, and the "c1" rule holds for a given material run too
    with pytest.raises(SystemExit, match="mix inference variants"):
        vote_run("runs/r1", "runs/t-c1", "r8b-c1-c2-test", material_run="runs/m-ok", snapshot=snapshot)
    with pytest.raises(SystemExit, match="c1"):
        vote_run("runs/r1", "runs/t-c1", "r8c-c2-test", material_run="runs/r8-c1-c2-test-material", snapshot=snapshot)
    with pytest.raises(SystemExit, match="mix inference variants"):
        vote_run("runs/r1", "runs/t-rows", "r8d-c2-test", material_run="runs/r8-c1-c2-test-material", snapshot=snapshot)
    assert not any(p.name.startswith("r8b") or p.name.startswith("r8c") or p.name.startswith("r8d") for p in runs.iterdir())
    vote_run("runs/r1", "runs/t-c1", "r8e-c1-c2-test", material_run="runs/r8-c1-c2-test-material", snapshot=snapshot)


def test_vote_run_refuses_forbidden_material_targets_and_run_ids(tmp_path, monkeypatch):
    real_dev2, real_test, real_seal = tagger.DEV2_FILE, tagger.TEST_FILE, tagger.SEAL_FILE
    cfg, snapshot = vote_setup(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    tagger.predict_run("runs/r1", ["ch0781"], "r2-test", snapshot=snapshot)
    good = [row("ch0001", 0, "徐获", "person"), row("ch0003", 0, "汀城", "location")]
    fake_predict_run(tmp_path, "m-ok", "r1", ["ch0001", "ch0003"], good)
    attempt = lambda run_id="r3-c2-test", material="runs/m-ok", target="runs/r2-test", snap=snapshot: vote_run(
        "runs/r1", target, run_id, material_run=material, snapshot=snap)
    # the run id: "c2" always, "test" for target chapters in 781-990 (the output holds their predictions)
    with pytest.raises(SystemExit, match="c2"):
        attempt("r3-test")
    with pytest.raises(SystemExit, match="'test'"):
        attempt("r3-c2")
    with pytest.raises(SystemExit, match="one folder name"):
        attempt("sub/r3-c2-test")
    # material: a chapter after 780, a dev chapter, the wrong chapter set, another model's run, a tampered file
    # ch1351 is after 780 and outside every real list (dev2 and the test are in 781-990, test2's order in 996-1350)
    beyond = Chapter("ch1351", 1351, [("p1", "汀城。")], "", "h9")
    wide = Snapshot("snap-test", None, {"source_commit": "c"}, {**snapshot.chapters, "ch1351": beyond}, [], [])
    fake_predict_run(tmp_path, "m-late", "r1", ["ch0001", "ch0003", "ch1351"], good)
    with pytest.raises(SystemExit, match="after chapter 780"):
        attempt(material="runs/m-late", snap=wide)
    fake_predict_run(tmp_path, "m-dev", "r1", ["ch0001", "ch0002", "ch0003"], good)
    with pytest.raises(SystemExit, match="dev chapters"):
        attempt(material="runs/m-dev")
    fake_predict_run(tmp_path, "m-short", "r1", ["ch0001"], good[:1])
    with pytest.raises(SystemExit, match="not exactly"):
        attempt(material="runs/m-short")
    fake_predict_run(tmp_path, "m-other", "r9", ["ch0001", "ch0003"], good)
    with pytest.raises(SystemExit, match="not by"):
        attempt(material="runs/m-other")
    fake_predict_run(tmp_path, "m-tampered", "r1", ["ch0001", "ch0003"], good)
    (runs / "m-tampered" / "predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the file"):
        attempt(material="runs/m-tampered")
    with pytest.raises(SystemExit, match="not a finished predict run"):
        attempt(material="runs/r1")  # a training run is not a predict run
    # dev2, the formal test and the sealed test2 by name: the given material and the self-predicted material alike
    dev2 = tmp_path / "dev2.json"
    dev2.write_text(json.dumps({"chapters": ["ch0003"]}), encoding="utf-8")
    monkeypatch.setattr(tagger, "DEV2_FILE", dev2)
    with pytest.raises(SystemExit, match="dev2"):
        attempt()
    with pytest.raises(SystemExit, match="dev2"):
        attempt(material=None)
    assert not (runs / "r3-c2-test-material").exists() and not (runs / "r3-c2-test").exists()
    monkeypatch.setattr(tagger, "DEV2_FILE", real_dev2)
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"order": [{"chapter_id": "ch0003"}]}), encoding="utf-8")
    exam = tmp_path / "exam.json"
    exam.write_text(json.dumps({"chapters_from_order": str(order), "n": 1}), encoding="utf-8")
    monkeypatch.setattr(tagger, "TEST_FILE", exam)
    assert tagger.exam_chapters() == {"ch0003"}
    with pytest.raises(SystemExit, match="formal test"):
        attempt()
    with pytest.raises(SystemExit, match="formal test"):
        attempt(material=None)
    monkeypatch.setattr(tagger, "TEST_FILE", real_test)
    # the dev chapters of annotate_dev.json by name, even when the training config's dev_set says otherwise
    real_dev = tagger.DEV_FILE
    monkeypatch.setattr(tagger, "DEV_FILE", exam)  # the same order file: ch0003 is now a dev chapter by name
    assert tagger.dev_chapters() == {"ch0003"}
    with pytest.raises(SystemExit, match="dev set"):
        attempt()
    with pytest.raises(SystemExit, match="dev set"):
        attempt(material=None)
    assert not (runs / "r3-c2-test-material").exists() and not (runs / "r3-c2-test").exists()
    monkeypatch.setattr(tagger, "DEV_FILE", real_dev)
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed", "order": str(order), "n": 1}), encoding="utf-8")
    monkeypatch.setattr(tagger, "SEAL_FILE", seal)
    with pytest.raises(SystemExit, match="sealed test2"):
        attempt()
    with pytest.raises(SystemExit, match="sealed test2"):
        attempt(material=None)
    # a target chapter in the sealed test2 is refused whatever the run id
    order.write_text(json.dumps({"order": [{"chapter_id": "ch0781"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed test2"):
        attempt("r3-c2-final-test2")
    monkeypatch.setattr(tagger, "SEAL_FILE", real_seal)
    # a target that is a training chapter would count its own predictions twice; a target made by another model
    tagger.predict_run("runs/r1", ["ch0003"], "r5-x", snapshot=snapshot)
    with pytest.raises(SystemExit, match="count twice"):
        attempt("r6-c2", target="runs/r5-x")
    fake_predict_run(tmp_path, "t-other", "r9", ["ch0781"], [], after_780=1)
    with pytest.raises(SystemExit, match="one model"):
        attempt("r7-c2-test", target="runs/t-other")
    assert not any(p.name.startswith(("r3-", "r6-", "r7-")) for p in runs.iterdir())
    # the sound case still works after all that
    assert attempt()["chapters"] == 1
