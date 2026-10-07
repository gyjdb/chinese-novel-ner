import json
import random

import pytest
import torch
from test_tagger import CFG, DEV, LATE, TRAIN, gold_export, region, span, tiny_model, tiny_tokenizer

from novel_ner import finetune, tagger
from novel_ner.evaluate import sha256
from novel_ner.finetune import epoch_sample, fit_mixed, replay_pool_rows, sample_digest
from novel_ner.snapshot import REPO_ROOT, Chapter, Snapshot
from novel_ner.tagger import training_windows
from novel_ner.weak import ids_sha256

QUIET = lambda *_: None
TRAIN2 = Chapter("ch0003", 3, [("p1", "汀城到了。")], "", "h4")
POOL = [Chapter("ch0004", 4, [("p1", "徐获见到唐医生。")], "", "h5"), Chapter("ch0005", 5, [("p1", "回音铃到了汀城。")], "", "h6"),
        Chapter("ch0006", 6, [("p1", "唐医生拿出回音铃。")], "", "h7")]
TEACHER_ROWS = [{"chapter_id": "ch0001", "number": 1, "split": "train", "uncertain": [],
                 "spans": [span(0, 2, "徐获", "person"), span(4, 7, "回音铃", "item"), span(9, 12, "唐医生", "person"), span(14, 16, "汀城", "location")]},
                {"chapter_id": "ch0003", "number": 3, "split": "train", "spans": [span(0, 2, "汀城", "location")], "uncertain": []}]
WEAK_ROWS = [{"chapter_id": "ch0001", "number": 1, "split": "train", "spans": [span(0, 2, "徐获", "person")],
              "uncertain": [region(4, 7, "回音铃", ["item"])]},
             {"chapter_id": "ch0002", "number": 2, "split": "dev", "spans": [], "uncertain": []},
             {"chapter_id": "ch0003", "number": 3, "split": "train", "spans": [], "uncertain": []},
             {"chapter_id": "ch0004", "number": 4, "split": "train", "spans": [span(0, 2, "徐获", "person"), span(4, 7, "唐医生", "person")], "uncertain": []},
             {"chapter_id": "ch0005", "number": 5, "split": "train", "spans": [span(0, 3, "回音铃", "item")], "uncertain": [region(5, 7, "汀城", ["location"])]},
             {"chapter_id": "ch0006", "number": 6, "split": "train", "spans": [span(0, 3, "唐医生", "person"), span(5, 8, "回音铃", "item")], "uncertain": []}]
POOL_IDS = ["ch0004", "ch0005", "ch0006"]


def write_build(root, build, setting, rows, snapshot_id="snap-test"):
    """A weak build's labels file and record runs/<build>/config.json, as weak.build / teacher.labels write them."""
    labels = root / "data" / "weak_labels" / build / f"labels_{setting}.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    record = root / "runs" / build / "config.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"snapshot": {"id": snapshot_id}, "labels": {setting: {"sha256": sha256(labels)}},
                                  "split": {"train_ids_sha256": ids_sha256(r["chapter_id"] for r in rows if r["split"] == "train")}}),
                      encoding="utf-8")
    return labels, record


def setup(tmp_path, monkeypatch, replay_snapshot="snap-test"):
    """A tiny backbone, a snapshot of 7 chapters, a teacher build "teach" (T: ch0001, ch0003), a weak build "weak" (S3:
    those two plus ch0004-ch0006, ch0002 dev), dev answers, and a finished stage 1 run runs/s3-s1 trained on the weak
    build with tagger.train; returns (the replay config, snapshot)."""
    tok = tiny_tokenizer(tmp_path / "vocab")
    tiny_model(tok).save_pretrained(tmp_path / "backbone")
    tok.save_pretrained(tmp_path / "backbone")
    snapshot = Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in (TRAIN, DEV, TRAIN2, LATE, *POOL)}, [], [])
    write_build(tmp_path, "teach", "T", TEACHER_ROWS)
    write_build(tmp_path, "weak", "S3", WEAK_ROWS, replay_snapshot)
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "gold" / "dev.json").write_text(json.dumps(gold_export(DEV, [
        {"start": 5, "end": 7, "text": "徐获", "type": "person", "pending": False}]), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tagger, "REPO_ROOT", tmp_path)
    (tmp_path / "dev_set.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tagger, "chapter_order", lambda dev_cfg: ["ch0002"])
    monkeypatch.setattr(tagger, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(tagger, "environment", lambda: {"python": "test"})
    base = {**CFG, "snapshot": "snap-test", "dev_set": "dev_set.json", "gold_dev": "data/gold/dev.json",
            "backbone": {"id": "tiny", "path": "backbone"}, "max_steps": 4, "deterministic": False}
    if replay_snapshot == "snap-test":  # stage 1 on the weak build (its record must be from this snapshot)
        tagger.train({**base, "weak_build": "weak"}, "S3", 1, "s3-s1", snapshot=snapshot, log=QUIET)
    cfg = {**base, "weak_build": "teach", "init_from_run": "runs/s3-s{seed}",
           "replay": {"weak_build": "weak", "setting": "S3", "ratio": 1, "exclude_teacher_chapters": True, "rule": "test replay"}}
    return cfg, snapshot


def without_replay(cfg):
    return {k: v for k, v in cfg.items() if k != "replay"}


def test_pool_excludes_the_teacher_chapters_and_the_record_describes_it(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch)
    weak_train = tagger.training_rows(WEAK_ROWS, snapshot, {"ch0002"})
    pool, excluded = replay_pool_rows(weak_train, ["ch0001", "ch0003"])
    assert [r["chapter_id"] for r in pool] == POOL_IDS and excluded == 2
    assert replay_pool_rows(weak_train, ["ch0001", "ch0003"], exclude=False) == (weak_train, 0)
    metrics = finetune.train(cfg, "T", 1, "mix1", snapshot=snapshot, log=QUIET)
    run = tmp_path / "runs" / "mix1"
    rec = json.loads((run / "config.json").read_text(encoding="utf-8"))
    r = rec["replay"]
    labels = tmp_path / "data" / "weak_labels" / "weak" / "labels_S3.jsonl"
    assert (r["weak_build"], r["setting"], r["ratio"], r["rule"]) == ("weak", "S3", 1, "test replay")
    assert r["file"] == "data/weak_labels/weak/labels_S3.jsonl" and r["sha256"] == sha256(labels)
    assert r["build_record"] == "runs/weak/config.json" and r["rows_by_split"] == {"train": 5, "dev": 1}
    assert r["train_chapters"] == 5 and r["train_ids_sha256"] == ids_sha256(["ch0001", "ch0003", *POOL_IDS])
    assert r["exclude_teacher_chapters"] is True and r["excluded_teacher_chapters"] == 2
    assert r["pool_chapters"] == 3 and r["pool_ids_sha256"] == ids_sha256(POOL_IDS) and r["pool_chapter_numbers"] == [4, 6]
    assert r["pool_windows"] == 3 and r["sampled_per_epoch"] == 2 and r["windows_per_epoch"] == 4
    assert r["same_snapshot_as_primary"] is True and "random.Random(seed)" in r["sample_rule"]
    m = metrics["replay"]
    assert (m["pool_chapters"], m["pool_windows"], m["sampled_per_epoch"]) == (3, 3, 2)
    assert m["windows_per_epoch"] == {"teacher": 2, "weak": 2, "total": 4} and m["pool_tokens"]["tokens"] == 25
    assert metrics["train"]["chapters"] == 2 and metrics["train"]["windows"] == 2  # the teacher windows alone, as tagger
    for name in ("config.json", "metrics.json"):
        assert not set((run / name).read_text(encoding="utf-8")) & set("徐获回音铃唐医生汀城")  # no novel text
    assert tmp_path.name not in json.dumps(rec)
    assert (run / "predictions_dev.jsonl").exists() and (run / "checkpoints" / "best" / "config.json").exists()
    with pytest.raises(SystemExit, match="already has results"):
        finetune.train(cfg, "T", 1, "mix1", snapshot=snapshot, log=QUIET)


def test_every_epoch_has_teacher_plus_ratio_times_teacher_windows_and_the_draw_follows_the_seed(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch)
    metrics = finetune.train(cfg, "T", 1, "mix1", snapshot=snapshot, log=QUIET)
    # 2 teacher + 2 weak windows per epoch, batch 2 -> 2 steps per epoch; max_steps 4 -> 2 epochs drawn
    assert metrics["schedule"]["steps_per_epoch"] == 2 and metrics["schedule"]["steps_planned"] == 4
    per_epoch = metrics["replay"]["per_epoch"]
    assert metrics["replay"]["epochs_sampled"] == len(per_epoch) == 2
    assert [(p["epoch"], p["teacher"], p["weak"], p["total"]) for p in per_epoch] == [(1, 2, 2, 4), (2, 2, 2, 4)]
    assert all(1 <= p["weak_chapters"] <= 2 for p in per_epoch)  # one window per pool chapter: 2 distinct chapters
    assert all(p["weak_chapters"] == 2 for p in per_epoch)
    stream = random.Random(1)
    assert [p["sampled_sha256"] for p in per_epoch] == [sample_digest(epoch_sample(stream, 3, 2)) for _ in range(2)]
    # the same seed draws the same windows and gives the same losses; the draw depends on the seed
    tok = tiny_tokenizer(tmp_path / "vocab2")
    teacher, _ = training_windows(tok, [(snapshot.chapters[r["chapter_id"]], r) for r in TEACHER_ROWS], CFG)
    pool, _ = training_windows(tok, [(snapshot.chapters[r["chapter_id"]], r) for r in WEAK_ROWS if r["chapter_id"] in POOL_IDS], CFG)
    gold = {"ch0002": {"spans": {(5, 7, "person")}, "unscored": []}}
    dev = {"chapters": [DEV], "gold": gold, "ids": ["ch0002"]}
    fit_cfg = {**CFG, "max_steps": 4}

    def run(seed, ratio=1):
        result = fit_mixed(tiny_model(tok), tok, teacher, pool, ratio * len(teacher), dev, fit_cfg, seed, torch.device("cpu"), log=QUIET)
        return [p["sampled_sha256"] for p in result["replay_epochs"]], [p["train_loss"] for p in result["curve"]]

    assert run(1) == run(1)
    assert run(1)[0] == [p["sampled_sha256"] for p in per_epoch]
    assert epoch_sample(random.Random(1), 1000, 5) != epoch_sample(random.Random(2), 1000, 5)
    assert run(2)[0] != run(1)[0] or run(2)[1] != run(1)[1]
    with pytest.raises(ValueError, match="fewer than"):
        fit_mixed(tiny_model(tok), tok, teacher, pool, 4, dev, fit_cfg, 1, torch.device("cpu"), log=QUIET)
    with pytest.raises(ValueError, match="no teacher windows"):
        fit_mixed(tiny_model(tok), tok, [], pool, 0, dev, fit_cfg, 1, torch.device("cpu"), log=QUIET)
    assert finetune.sampled_per_epoch(3, 500) == 1500 and finetune.sampled_per_epoch(0, 500) == 0 and finetune.sampled_per_epoch(0.5, 3) == 2


def test_record_is_tagger_trains_record_with_the_teacher_build_as_weak_labels_plus_the_replay_block(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch)
    tagger.train(without_replay(cfg), "T", 1, "ref", snapshot=snapshot, log=QUIET)
    finetune.train(cfg, "T", 1, "mix1", snapshot=snapshot, log=QUIET)
    ref = json.loads((tmp_path / "runs" / "ref" / "config.json").read_text(encoding="utf-8"))
    mix = json.loads((tmp_path / "runs" / "mix1" / "config.json").read_text(encoding="utf-8"))
    assert set(mix) == set(ref) | {"replay"}
    for key in ("weak_labels", "init_from", "snapshot", "gold_dev", "backbone", "labels", "setting", "seed", "determinism",
                "device", "code", "environment", "config_file"):
        assert mix[key] == ref[key], key
    assert mix["weak_labels"]["build"] == "teach" and mix["weak_labels"]["train_chapters"] == 2  # what span_system reads
    assert mix["weak_labels"]["train_ids_sha256"] == ids_sha256(["ch0001", "ch0003"])
    assert mix["init_from"]["run_id"] == "s3-s1" and mix["init_from"]["checkpoint"] == "runs/s3-s1/checkpoints/best"
    assert mix["config"] == cfg and without_replay(mix["config"]) == ref["config"]
    assert set(mix["replay"]) >= {"weak_build", "file", "sha256", "setting", "ratio", "pool_windows", "pool_chapters",
                                  "sampled_per_epoch", "excluded_teacher_chapters", "rule"}
    ref_m = json.loads((tmp_path / "runs" / "ref" / "metrics.json").read_text(encoding="utf-8"))
    mix_m = json.loads((tmp_path / "runs" / "mix1" / "metrics.json").read_text(encoding="utf-8"))
    assert set(mix_m) == set(ref_m) | {"replay"} and set(mix_m["schedule"]) == set(ref_m["schedule"])
    assert set(mix_m["sufficiency"]) == set(ref_m["sufficiency"]) and mix_m["train"].keys() == ref_m["train"].keys()


def test_ratio_zero_reproduces_tagger_train_bit_for_bit(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch)
    cfg0 = {**cfg, "replay": {**cfg["replay"], "ratio": 0}}
    ref = tagger.train(without_replay(cfg0), "T", 1, "ref", snapshot=snapshot, log=QUIET)
    mix = finetune.train(cfg0, "T", 1, "mix0", snapshot=snapshot, log=QUIET)
    strip = lambda curve: [{k: v for k, v in p.items() if k != "seconds"} for p in curve]
    assert strip(mix["dev_curve"]) == strip(ref["dev_curve"]) and len(ref["dev_curve"]) == 3  # 1 step per epoch x 3 epochs
    assert [p["train_loss"] for p in mix["dev_curve"]] == [p["train_loss"] for p in ref["dev_curve"]]
    assert mix["predictions_dev_sha256"] == ref["predictions_dev_sha256"]
    assert mix["dev"]["at_best"] == ref["dev"]["at_best"] and mix["best"] == ref["best"]
    drop = lambda s: {k: v for k, v in s.items() if k != "train_seconds"}
    assert drop(mix["schedule"]) == drop(ref["schedule"]) and mix["schedule"]["steps_per_epoch"] == 1
    assert mix["replay"]["sampled_per_epoch"] == 0 and all(p["weak"] == 0 and p["total"] == 2 for p in mix["replay"]["per_epoch"])
    assert sha256(tmp_path / "runs" / "mix0" / "predictions_dev.jsonl") == sha256(tmp_path / "runs" / "ref" / "predictions_dev.jsonl")
    _, a = tagger.load_backbone(tmp_path / "runs" / "ref" / "checkpoints" / "best")
    _, b = tagger.load_backbone(tmp_path / "runs" / "mix0" / "checkpoints" / "best")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_a_replay_build_from_another_snapshot_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch, replay_snapshot="snap-other")
    stage1 = {k: v for k, v in without_replay(cfg).items() if k != "init_from_run"}  # a stage 1 run to start from (on the teacher build)
    tagger.train(stage1, "T", 1, "s3-s1", snapshot=snapshot, log=QUIET)
    with pytest.raises(SystemExit, match="replay weak build weak is from snapshot snap-other, not snap-test"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
    assert not (tmp_path / "runs" / "mix-bad").exists()


def test_replay_block_and_build_checks(tmp_path, monkeypatch):
    cfg, snapshot = setup(tmp_path, monkeypatch)
    cases = [
        (without_replay(cfg), "no replay block"),
        ({**cfg, "replay": {**cfg["replay"], "extra": 1}}, "unknown keys"),
        ({**cfg, "replay": {k: v for k, v in cfg["replay"].items() if k != "rule"}}, "missing keys"),
        ({**cfg, "replay": {**cfg["replay"], "ratio": -1}}, "ratio"),
        ({**cfg, "replay": {**cfg["replay"], "ratio": True}}, "ratio"),
        ({**cfg, "replay": {**cfg["replay"], "exclude_teacher_chapters": "yes"}}, "exclude_teacher_chapters"),
        ({**cfg, "replay": {**cfg["replay"], "weak_build": ""}}, "weak_build"),
        ({k: v for k, v in cfg.items() if k != "init_from_run"}, "init_from_run"),
        ({**cfg, "replay": {**cfg["replay"], "weak_build": "teach", "setting": "T"}}, "primary labels themselves"),
        ({**cfg, "replay": {**cfg["replay"], "ratio": 5}}, "fewer than"),  # 10 > the pool's 3 windows
        ({**cfg, "replay": {**cfg["replay"], "setting": "S9"}}, "no replay weak build labels"),
        ({**cfg, "replay": {**cfg["replay"], "weak_build": "nowhere"}}, "no replay weak build labels"),
    ]
    for bad, word in cases:
        with pytest.raises(SystemExit, match=word):
            finetune.train(bad, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
        assert not (tmp_path / "runs" / "mix-bad").exists(), word
    assert finetune.replay_config(cfg, "T") == cfg["replay"]
    # the replay labels file must be the one its build recorded, and the record must exist
    labels = tmp_path / "data" / "weak_labels" / "weak" / "labels_S3.jsonl"
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the file replay weak build weak recorded"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
    write_build(tmp_path, "weak", "S3", WEAK_ROWS)
    record = tmp_path / "runs" / "weak" / "config.json"
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "split": {"train_ids_sha256": ids_sha256(["ch0001"])}}),
                      encoding="utf-8")
    with pytest.raises(SystemExit, match="training chapters differ from replay weak build weak"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
    record.unlink()
    with pytest.raises(SystemExit, match="no record of replay weak build weak"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
    assert not (tmp_path / "runs" / "mix-bad").exists()
    # the primary build keeps tagger.train's wording
    (tmp_path / "data" / "weak_labels" / "teach" / "labels_T.jsonl").unlink()
    with pytest.raises(SystemExit, match="no weak build labels"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)


def test_round5_configs_are_a3_with_one_change_each_and_recipes_names_them():
    configs = REPO_ROOT / "configs"
    base = json.loads((configs / "train_s1_base_a3.json").read_text(encoding="utf-8"))
    recipes = json.loads((configs / "recipes_v1.json").read_text(encoding="utf-8"))["round5_recipes"]["candidates"]
    assert recipes["lr2e5"]["config"] == "configs/train_s5_a3_lr2e5.json" and recipes["lr5e5"]["config"] == "configs/train_s5_a3_lr5e5.json"
    assert recipes["mix11"]["config"] == "configs/train_s5_a3_mix11.json" and recipes["mix13"]["config"] == "configs/train_s5_a3_mix13.json"
    same = lambda cfg, skip: {k: v for k, v in cfg.items() if k not in skip} == {k: v for k, v in base.items() if k not in skip}
    for name, lr in (("lr2e5", 2e-5), ("lr5e5", 5e-5)):
        cfg = json.loads((REPO_ROOT / recipes[name]["config"]).read_text(encoding="utf-8"))
        assert cfg["lr"] == lr and base["lr"] == 3e-5 and set(cfg) == set(base)
        assert same(cfg, {"description", "arms", "lr"}) and name in cfg["description"] and "tagger train" in cfg["description"]
        with pytest.raises(SystemExit, match="no replay block"):
            finetune.replay_config(cfg, "T")
    for name, ratio in (("mix11", 1), ("mix13", 3)):
        cfg = json.loads((REPO_ROOT / recipes[name]["config"]).read_text(encoding="utf-8"))
        assert set(cfg) == set(base) | {"replay", "trainer"} and cfg["trainer"] == "novel_ner.finetune"
        assert same(cfg, {"description", "arms", "replay", "trainer", "planned_steps_note"})
        assert finetune.replay_config(cfg, "T") == cfg["replay"]
        assert {k: v for k, v in cfg["replay"].items() if k != "rule"} == {
            "weak_build": "20261003-weak_v2", "setting": "S3", "ratio": ratio, "exclude_teacher_chapters": True}
        assert "649" in cfg["replay"]["rule"] and name in cfg["description"] and "novel_ner.finetune" in cfg["description"]
        assert cfg["lr"] == 3e-5 and cfg["weak_build"] == "20261004-teacher100" and cfg["init_from_run"] == base["init_from_run"]
        assert str(-(-(1 + ratio) * 500 // 16)) in cfg["planned_steps_note"]


def test_replay_rows_after_780_or_in_dev_or_dev2_are_refused_before_anything_is_written(tmp_path, monkeypatch):
    """The pool goes through tagger.training_rows like the primary rows: a replay build whose train rows name a chapter
    after 780 (every test, test2 and dev2 chapter is), a dev chapter or a dev2 decision chapter stops the run."""
    cfg, snapshot = setup(tmp_path, monkeypatch)
    late = {"chapter_id": "ch0781", "number": 781, "split": "train", "spans": [span(0, 2, "徐获", "person")], "uncertain": []}
    dev_row = {"chapter_id": "ch0002", "number": 2, "split": "train", "spans": [], "uncertain": []}
    for extra, word in ((late, "must be <= 780"), (dev_row, "not a dev chapter")):
        write_build(tmp_path, "weak", "S3", WEAK_ROWS + [extra])  # the build record's split sha256 covers the extra row
        with pytest.raises(SystemExit, match=word):
            finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
        assert not (tmp_path / "runs" / "mix-bad").exists(), word
    write_build(tmp_path, "weak", "S3", WEAK_ROWS)
    dev2 = tmp_path / "dev2.json"
    dev2.write_text(json.dumps({"chapters": ["ch0005"]}), encoding="utf-8")
    monkeypatch.setattr(tagger, "DEV2_FILE", dev2)
    with pytest.raises(SystemExit, match="dev2 decision chapter"):
        finetune.train(cfg, "T", 1, "mix-bad", snapshot=snapshot, log=QUIET)
    assert not (tmp_path / "runs" / "mix-bad").exists()


def test_the_real_replay_pool_misses_every_dev_dev2_test_and_sealed_test2_chapter():
    """Read-only on the committed records and label files (no training, nothing written): the pool of the
    pre-registered configs = weak_v2 S3 train rows minus the teacher100 chapters = 649 chapters, all <= 780, none in
    dev (31), dev2 (25), the formal test (60) or the sealed test2."""
    root = REPO_ROOT
    teacher = root / "data" / "weak_labels" / "20261004-teacher100" / "labels_T.jsonl"
    weak = root / "data" / "weak_labels" / "20261003-weak_v2" / "labels_S3.jsonl"
    if not teacher.exists() or not weak.exists():
        pytest.skip("the real label files are not in this checkout")
    from novel_ner.snapshot import read_jsonl
    t_train = sorted(r["chapter_id"] for r in read_jsonl(teacher) if r.get("split") == "train")
    w_train = [r for r in read_jsonl(weak) if r.get("split") == "train"]
    pool, excluded = replay_pool_rows(w_train, t_train)
    assert (len(t_train), len(w_train), len(pool), excluded) == (100, 749, 649, 100)
    assert max(r["number"] for r in pool) <= tagger.MAX_TRAIN_CHAPTER
    held_out = tagger.dev_chapters() | tagger.dev2_chapters() | tagger.exam_chapters() | tagger.sealed_test2_chapters()
    assert not {r["chapter_id"] for r in pool} & held_out and not set(t_train) & held_out
