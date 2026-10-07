"""Stage 2 fine-tuning with a replay of weak windows (configs/recipes_v1.json, round 5 candidates mix11 / mix13).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.finetune train configs/train_s5_a3_mix11.json --setting T --seed 1 --run-id <id>

The run is tagger.train's two-stage fine-tuning (init_from_run names the stage 1 checkpoint, "{seed}" filled with the
run's seed; the teacher labels of the config's weak_build and setting are the training rows; the same checks of the
build record, the split and the rows; the same loss, optimizer, schedule, evaluation cadence, checkpoint rule and
sufficiency record) plus a replay: the config's "replay" block names a second weak-label build and setting (the weak
S3 labels of the 749 training chapters), and every epoch trains on ALL teacher windows plus ratio x as many weak
windows, sampled without replacement from a pool, shuffled together. The pool is built once, with the same tokenizer
and chunking as the teacher windows, from the replay build's training rows (split "train", checked by
tagger.training_rows and against the build record's sha256 and split like the primary build) for the chapters that
are not teacher chapters (exclude_teacher_chapters true; the count excluded is recorded). Both builds must be from the
run's snapshot. Every window keeps its own allowed label sets, so the loss is tagger's partial-label NLL over both
sources. steps_per_epoch = ceil((teacher + sampled) / batch_size); warmup and the linear decay are over
steps_per_epoch x max_epochs as in tagger.fit.

tagger.fit takes one fixed list of windows, so fit_mixed below is its loop copied line for line with the per-epoch
window set as the only change (tagger.py is left untouched: a training script imports it while this is written). The
sample is drawn by random.Random(seed).sample from a stream of its own, separate from the batch-order shuffle's
random.Random(seed), so a ratio of 0 gives exactly tagger.train's run (same losses, same predictions; tested).

Records: runs/<id>/config.json is tagger.train's record (the primary weak_labels block is the teacher build, what
evaluate.span_system and posthoc.training_chapters read, so the run scores as a model trained on the teacher
chapters) plus "replay" {weak_build, file, sha256, setting, ratio, pool_windows, pool_chapters, sampled_per_epoch,
excluded_teacher_chapters, rule, ...}; metrics.json is tagger's plus "replay" with the window counts per epoch by
source and the drawn indices' sha256 per epoch (numbers only). The patchable names (REPO_ROOT, load_snapshot,
chapter_order, git_state, environment, init_run, training_rows) are read from the tagger module at call time, so
tests set up like tests/test_tagger.py apply here too.
"""
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from novel_ner import tagger
from novel_ner.evaluate import done_chapters, sha256, write_jsonl
from novel_ner.gold import check as check_gold
from novel_ner.score import gold_spans
from novel_ner.snapshot import read_jsonl
from novel_ner.tagger import (MAX_TRAIN_CHAPTER, TAGS, autocast, backbone_path, collate, default_run_id, dev_scores,
                              load_backbone, micro_batches, partial_label_nll, plain_run_id, predict, seed_everything,
                              sufficiency, training_windows)
from novel_ner.weak import ids_sha256

REPLAY_KEYS = ("weak_build", "setting", "ratio", "exclude_teacher_chapters", "rule")
SAMPLE_RULE = ("every epoch: all teacher windows plus sampled_per_epoch pool windows, drawn without replacement as "
               "random.Random(seed).sample(range(pool_windows), sampled_per_epoch) from a stream of its own (one draw per "
               "epoch, in epoch order), then shuffled together by tagger.fit's batch-order random.Random(seed); the pool is "
               "built once (tagger.training_windows with the config's tokenizer and chunking) from the replay build's "
               "training rows minus the teacher chapters; every window keeps its own allowed label sets (partial-label NLL "
               "over both sources); steps_per_epoch = ceil((teacher + sampled) / batch_size); metrics.json replay.per_epoch "
               "holds the drawn indices' sha256 (of their JSON list) per epoch")


def replay_config(cfg, setting):
    """The config's checked "replay" block: all of REPLAY_KEYS and nothing else, ratio a number >= 0,
    exclude_teacher_chapters a bool, the rest non-empty strings; init_from_run present (this is stage 2); the replay
    labels are not the primary labels themselves."""
    block = cfg.get("replay")
    if not isinstance(block, dict):
        raise SystemExit("the config has no replay block: a plain fine-tuning run is `python -m novel_ner.tagger train`")
    missing, unknown = [k for k in REPLAY_KEYS if k not in block], sorted(set(block) - set(REPLAY_KEYS))
    if missing or unknown:
        raise SystemExit(f"replay block: missing keys {missing}, unknown keys {unknown} (the keys are {list(REPLAY_KEYS)})")
    if not cfg.get("init_from_run"):
        raise SystemExit("replay fine-tuning is stage 2: the config needs init_from_run (the stage 1 checkpoint, "
                         "'{seed}' filled with the run's seed)")
    ratio = block["ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0:
        raise SystemExit(f"replay ratio must be a number >= 0, not {ratio!r}")
    if not isinstance(block["exclude_teacher_chapters"], bool):
        raise SystemExit("replay exclude_teacher_chapters must be true or false")
    for key in ("weak_build", "setting", "rule"):
        if not isinstance(block[key], str) or not block[key].strip():
            raise SystemExit(f"replay {key} must be a non-empty string")
    if (block["weak_build"], block["setting"]) == (cfg["weak_build"], setting):
        raise SystemExit(f"the replay labels ({block['weak_build']} {block['setting']}) are the primary labels themselves")
    return block


def build_files(build, setting, snapshot, what="weak build"):
    """tagger.train's checks of a weak-label build before its rows are read: the labels file and the build record
    exist, the build is from this snapshot, the file is the one the record hashed. Returns (labels file, record
    path, record)."""
    root = tagger.REPO_ROOT
    labels_file = root / "data" / "weak_labels" / build / f"labels_{setting}.jsonl"
    if not labels_file.exists():
        raise SystemExit(f"no {what} labels at {labels_file}")
    build_record = root / "runs" / build / "config.json"
    if not build_record.exists():
        raise SystemExit(f"no record of {what} {build} at {build_record}")
    built = json.loads(build_record.read_text(encoding="utf-8"))
    built_snapshot = built.get("snapshot")
    built_snapshot = built_snapshot.get("id") if isinstance(built_snapshot, dict) else built_snapshot
    if built_snapshot != snapshot.id:
        raise SystemExit(f"{what} {build} is from snapshot {built_snapshot}, not {snapshot.id}")
    if (built.get("labels") or {}).get(setting, {}).get("sha256") != sha256(labels_file):
        raise SystemExit(f"{labels_file.name} differs from the file {what} {build} recorded")
    return labels_file, build_record, built


def build_rows(labels_file, built, build, snapshot, dev_all, what="weak build"):
    """tagger.train's reading of a build's rows: tagger.training_rows (split train, in the snapshot, <= 780, no dev
    or dev2 chapter, texts where they say) and the training chapters' sha256 against the record's split. Returns
    (all rows, training rows, sorted training chapter ids)."""
    rows = read_jsonl(labels_file)
    train_rows = tagger.training_rows(rows, snapshot, dev_all)
    train_ids = sorted(r["chapter_id"] for r in train_rows)
    if ids_sha256(train_ids) != (built.get("split") or {}).get("train_ids_sha256"):
        raise SystemExit(f"the training chapters differ from {what} {build}'s split")
    return rows, train_rows, train_ids


def replay_pool_rows(rows, teacher_ids, exclude=True):
    """(the replay build's training rows that form the pool, how many were left out as teacher chapters)."""
    teacher_ids = set(teacher_ids)
    pool = [r for r in rows if not (exclude and r["chapter_id"] in teacher_ids)]
    return pool, len(rows) - len(pool)


def sampled_per_epoch(ratio, teacher_windows):
    """How many pool windows join each epoch: ratio x the teacher windows, rounded to a whole window."""
    return int(round(ratio * teacher_windows))


def epoch_sample(sampler, pool_size, k):
    """One epoch's draw: k distinct pool indices from sampler (a random.Random), in draw order."""
    return sampler.sample(range(pool_size), k)


def sample_digest(indices):
    """sha256 of the drawn indices as a JSON list (what metrics.json records per epoch)."""
    return hashlib.sha256(json.dumps(list(indices)).encode("utf-8")).hexdigest()


def fit_mixed(model, tokenizer, teacher, pool, k, dev, cfg, seed, device, log=print):
    """tagger.fit's loop on a per-epoch window set: all `teacher` windows plus k windows of `pool` drawn without
    replacement by random.Random(seed).sample (epoch_sample; a stream of its own, so with k = 0 the batch order is
    tagger.fit's and the run is tagger.train's), shuffled together by tagger.fit's random.Random(seed).

    Copied from tagger.fit, which takes one fixed list of windows, with that one change: the optimizer (AdamW, no
    weight decay on biases and LayerNorm), linear warmup then linear decay to 0 over steps_per_epoch x max_epochs
    (max_steps caps it), gradient accumulation (tagger.micro_batches), the partial-label NLL, gradient clipping,
    bf16 autocast on CUDA, dev scoring every eval_every epoch, the best state by dev micro F1 (ties: the earlier
    step) and patience are the same lines. Returns fit's record plus "replay_epochs": per epoch the window counts by
    source, the distinct pool chapters drawn and the drawn indices' sha256 (sample_digest)."""
    from transformers import get_linear_schedule_with_warmup

    if not teacher:
        raise ValueError("no teacher windows")
    if k > len(pool):
        raise ValueError(f"the replay pool has {len(pool)} windows, fewer than the {k} to sample per epoch")
    model.to(device)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
                                   {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=cfg["lr"])
    size = cfg["batch_size"]
    accumulation = int(cfg.get("grad_accumulation", 1))
    if accumulation < 1 or size % accumulation:
        raise ValueError(f"grad_accumulation {accumulation} must be a positive divisor of batch_size {size}")
    per_epoch = math.ceil((len(teacher) + k) / size)
    total = per_epoch * cfg["max_epochs"]
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    warmup = int(cfg["warmup"] * total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total)
    every = max(1, round(per_epoch * cfg["eval_every"]))
    rng = random.Random(seed)  # the batch order, as tagger.fit
    sampler = random.Random(seed)  # the replay draw: its own stream, so k = 0 leaves the batch order tagger.fit's
    curve, best, state, losses, epochs = [], None, None, [], []
    step, waited, stopped, began = 0, 0, False, time.perf_counter()
    while step < total and not stopped:
        picked = epoch_sample(sampler, len(pool), k)
        windows = teacher + [pool[i] for i in picked]
        epochs.append({"epoch": len(epochs) + 1, "teacher": len(teacher), "weak": len(picked), "total": len(windows),
                       "weak_chapters": len({pool[i]["chapter_id"] for i in picked}), "sampled_sha256": sample_digest(picked)})
        order = list(range(len(windows)))
        rng.shuffle(order)
        for b in range(0, len(order), size):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for part, weight in micro_batches([windows[i] for i in order[b:b + size]], accumulation):
                ids, mask, allowed = collate(part, tokenizer.pad_token_id)
                with autocast(cfg, device):
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = partial_label_nll(logits, allowed.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss is {loss.item()} at step {step + 1}")
                (loss if weight == 1.0 else loss * weight).backward()  # weight 1.0: the same graph as without accumulation
                value += loss.item() if weight == 1.0 else loss.item() * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(value)
            if step % every == 0 or step == total:
                rows, counts = predict(model, tokenizer, dev["chapters"], cfg, device)
                scores = dev_scores(dev["gold"], rows, dev["ids"])
                f1 = scores["micro"]["f1"] or 0.0
                point = {"step": step, "epoch": round(step / per_epoch, 3), "lr": scheduler.get_last_lr()[0],
                         "train_loss": round(sum(losses) / len(losses), 5), "micro": scores["micro"],
                         "non_person_f1": scores["non_person_micro"]["f1"], "macro_f1": scores["macro_f1"],
                         "outputs": len(rows), "seconds": round(time.perf_counter() - began, 1)}
                curve.append(point)
                losses = []
                log(f"step {step}/{total} epoch {point['epoch']:.2f} loss {point['train_loss']:.4f} "
                    f"dev P={scores['micro']['precision']} R={scores['micro']['recall']} F1={f1:.4f} ({point['seconds']} s)")
                if best is None or f1 > best["f1"]:
                    best = {"step": step, "epoch": point["epoch"], "f1": f1, "scores": scores, "rows": rows, "predict_stats": counts}
                    state = {k_: v.detach().to("cpu", copy=True) for k_, v in model.state_dict().items()}
                    waited = 0
                else:
                    waited += 1
                    stopped = waited >= cfg["patience"]
            if stopped or step >= total:
                break
    model.load_state_dict(state)
    return {"curve": curve, "best": best, "steps_per_epoch": per_epoch, "steps_planned": total, "warmup_steps": warmup,
            "eval_every_steps": every, "steps_run": step, "epochs_run": round(step / per_epoch, 3),
            "stopped_early": stopped, "grad_accumulation": accumulation, "micro_batch_size": math.ceil(size / accumulation),
            "train_seconds": round(time.perf_counter() - began, 1), "replay_epochs": epochs}


def train(cfg, setting, seed, run_id, config_file=None, snapshot=None, log=print):
    """One replay fine-tuning run: the teacher labels of `setting` from cfg["weak_build"] as the training rows, the
    weak windows of cfg["replay"] replayed every epoch, one seed; writes runs/<run_id>/ (tagger.train's files)."""
    began = time.perf_counter()
    run_id = plain_run_id(run_id)
    replay = replay_config(cfg, setting)
    root = tagger.REPO_ROOT
    out_dir = root / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    if cfg.get("schedule", "linear") != "linear":
        raise SystemExit("only the linear schedule is implemented")
    snapshot = snapshot or tagger.load_snapshot(cfg["snapshot"])
    # the primary (teacher) build, checked as tagger.train checks it, in its order
    labels_file, build_record, built = build_files(cfg["weak_build"], setting, snapshot)
    dev_all = set(tagger.chapter_order(json.loads((root / cfg["dev_set"]).read_text(encoding="utf-8"))))
    if any(c not in snapshot.chapters or snapshot.chapters[c].number > MAX_TRAIN_CHAPTER for c in dev_all):
        raise SystemExit(f"every dev chapter must be in snapshot {snapshot.id} and <= {MAX_TRAIN_CHAPTER}")
    rows, train_rows, train_ids = build_rows(labels_file, built, cfg["weak_build"], snapshot, dev_all)
    # the replay build: the same checks (record, snapshot, sha256, rows, split), then the pool without the teacher chapters
    r_file, r_record, r_built = build_files(replay["weak_build"], replay["setting"], snapshot, "replay weak build")
    r_rows, r_train_rows, r_train_ids = build_rows(r_file, r_built, replay["weak_build"], snapshot, dev_all, "replay weak build")
    pool_rows, excluded = replay_pool_rows(r_train_rows, train_ids, replay["exclude_teacher_chapters"])
    pool_ids = sorted(r["chapter_id"] for r in pool_rows)
    export = json.loads((root / cfg["gold_dev"]).read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the dev answers do not pass the gold check: {errors[:3]}")
    dev_ids = done_chapters(export)
    if not dev_ids or not set(dev_ids) <= dev_all:
        raise SystemExit("the dev answers must hold finished dev chapters")
    gold = gold_spans(export, dev_ids)
    path = backbone_path(cfg)
    init = tagger.init_run(cfg, seed)
    determinism = seed_everything(seed, cfg.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_backbone(path, root / init["checkpoint"])
    # windows: numpy and the tokenizer only, so building the pool here leaves torch's random stream as in tagger.train
    windows, token_stats = training_windows(tokenizer, [(snapshot.chapters[r["chapter_id"]], r) for r in train_rows], cfg)
    pool, pool_stats = training_windows(tokenizer, [(snapshot.chapters[r["chapter_id"]], r) for r in pool_rows], cfg)
    k = sampled_per_epoch(replay["ratio"], len(windows))
    if k > len(pool):
        raise SystemExit(f"the replay pool has {len(pool)} windows ({len(pool_rows)} chapters), fewer than the {k} to sample "
                         f"per epoch (ratio {replay['ratio']} x {len(windows)} teacher windows)")
    record = {
        "run_id": run_id, "config_file": config_file, "config": cfg, "setting": setting, "seed": seed,
        "init_from": init,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
        "weak_labels": {"build": cfg["weak_build"], "file": labels_file.relative_to(root).as_posix(),
                        "sha256": sha256(labels_file), "rows_by_split": dict(Counter(r.get("split") for r in rows)),
                        "train_chapters": len(train_ids), "train_ids_sha256": ids_sha256(train_ids),
                        "train_ids_sha256_rule": "sha256 of the sorted chapter ids joined by \\n (weak.ids_sha256)",
                        "train_chapter_numbers": [min(r["number"] for r in train_rows), max(r["number"] for r in train_rows)],
                        "build_record": build_record.relative_to(root).as_posix()},
        "replay": {"weak_build": replay["weak_build"], "file": r_file.relative_to(root).as_posix(), "sha256": sha256(r_file),
                   "build_record": r_record.relative_to(root).as_posix(), "setting": replay["setting"],
                   "ratio": replay["ratio"], "rule": replay["rule"], "sample_rule": SAMPLE_RULE,
                   "rows_by_split": dict(Counter(r.get("split") for r in r_rows)),
                   "train_chapters": len(r_train_ids), "train_ids_sha256": ids_sha256(r_train_ids),
                   "exclude_teacher_chapters": replay["exclude_teacher_chapters"], "excluded_teacher_chapters": excluded,
                   "pool_chapters": len(pool_rows), "pool_ids_sha256": ids_sha256(pool_ids),
                   "pool_chapter_numbers": ([min(r["number"] for r in pool_rows), max(r["number"] for r in pool_rows)]
                                            if pool_rows else None),
                   "pool_windows": len(pool), "sampled_per_epoch": k, "windows_per_epoch": len(windows) + k,
                   "same_snapshot_as_primary": True},
        "gold_dev": {"file": cfg["gold_dev"], "sha256": sha256(root / cfg["gold_dev"]), "chapters": dev_ids},
        # the config's repo-relative path (tests, smoke runs) or None; never the cache path under the user profile
        "backbone": {"id": cfg["backbone"]["id"], "revision": cfg["backbone"].get("revision"), "path": cfg["backbone"].get("path"),
                     "attn_implementation": getattr(model.config, "_attn_implementation", None)},
        "labels": TAGS, "device": str(device), "determinism": determinism,
        "code": tagger.git_state(), "environment": {**tagger.environment(), "torch_cuda": torch.version.cuda},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    log(f"{run_id}: {len(train_ids)} teacher chapters, {len(windows)} windows, {token_stats['tokens']} tokens; replay pool "
        f"{len(pool_rows)} chapters, {len(pool)} windows, {pool_stats.get('tokens', 0)} tokens, {k} sampled per epoch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dev = {"chapters": [snapshot.chapters[c] for c in dev_ids], "gold": gold, "ids": dev_ids}
    result = fit_mixed(model, tokenizer, windows, pool, k, dev, cfg, seed, device, log)
    best = result["best"]
    again, _ = predict(model, tokenizer, dev["chapters"], cfg, device)
    keys = lambda rs: sorted((r["chapter_id"], r["start"], r["end"], r["type"]) for r in rs)
    best_dir = out_dir / "checkpoints" / "best"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    write_jsonl(out_dir / "predictions_dev.jsonl", best["rows"])
    metrics = {
        "run_id": run_id, "setting": setting, "seed": seed,
        "train": {"chapters": len(train_ids), **token_stats},
        "replay": {"ratio": replay["ratio"], "pool_chapters": len(pool_rows), "pool_windows": len(pool),
                   "pool_tokens": pool_stats, "sampled_per_epoch": k,
                   "windows_per_epoch": {"teacher": len(windows), "weak": k, "total": len(windows) + k},
                   "epochs_sampled": len(result["replay_epochs"]), "per_epoch": result["replay_epochs"],
                   "sample_rule": SAMPLE_RULE},
        "schedule": {k_: result[k_] for k_ in ("steps_per_epoch", "steps_planned", "warmup_steps", "eval_every_steps",
                                               "steps_run", "epochs_run", "stopped_early", "grad_accumulation",
                                               "micro_batch_size", "train_seconds")},
        "dev_curve": result["curve"],
        "best": {"step": best["step"], "epoch": best["epoch"], "rule": "highest dev micro F1; ties: the earlier step"},
        "sufficiency": sufficiency(result),
        "dev": {"chapters": len(dev_ids), "scored_gold_spans": best["scores"]["tp"] + best["scores"]["fn"],
                "at_best": best["scores"], "predict_stats_at_best": best["predict_stats"],
                "saved_state_gives_the_same_predictions": keys(again) == keys(best["rows"])},
        "predictions_dev_sha256": sha256(out_dir / "predictions_dev.jsonl"),
        "wall_seconds": round(time.perf_counter() - began, 1),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    m = best["scores"]["micro"]
    log(f"{run_id}: best at step {best['step']} (epoch {best['epoch']}): dev P={m['precision']} R={m['recall']} F1={m['f1']}; wrote {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train", help="stage 2 fine-tuning with the config's replay block")
    t.add_argument("config")
    t.add_argument("--setting", required=True, help="the teacher labels' setting in the config's weak_build (T)")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run-id", help="<id> or runs/<id>; default: <date>-tag-<setting>-s<seed>-replay")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train(cfg, args.setting, args.seed, args.run_id or default_run_id(args.setting, args.seed, "replay"), Path(args.config).as_posix())


if __name__ == "__main__":
    main()
