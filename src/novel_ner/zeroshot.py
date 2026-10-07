"""Zero-shot (no training) GLiNER runs over sampled chapters, compared with the LLM extraction.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.zeroshot configs/zs_gliner_trial.json

Writes runs/<run_id>/: config.json and metrics.json (committed, no novel text) and
predictions_<variant>.jsonl (not committed).
"""
import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

from novel_ner.compare import agreement, names_from_dictionary, names_from_mentions, names_from_predictions
from novel_ner.snapshot import PARAGRAPH_SEP, REPO_ROOT, load_snapshot


def sample_chapters(snapshot, sample):
    """Chapters the reference model extracted alone, minus excluded ones, sampled with a fixed seed."""
    by_chapter = {}
    for row in snapshot.mentions:
        by_chapter.setdefault(row["chapter_id"], set()).add(row["model"])
    pool = sorted(
        cid for cid, models in by_chapter.items()
        if all(m.startswith(sample["reference_model_prefix"]) for m in models)
        and snapshot.chapters[cid].number <= sample["max_chapter"]
        and snapshot.chapters[cid].number not in sample["exclude_numbers"]
    )
    return sorted(random.Random(sample["seed"]).sample(pool, sample["n"]))


def chunk_chapter(chapter, max_chars):
    """Consecutive paragraphs joined by PARAGRAPH_SEP, each chunk at most max_chars.

    Returns [(text, [(paragraph_id, offset_in_chunk, length)])]. A paragraph longer than
    max_chars becomes its own chunk (the caller checks encoder length separately).
    """
    chunks, cur, spans, size = [], [], [], 0
    for pid, text in chapter.paragraphs:
        extra = len(text) + (len(PARAGRAPH_SEP) if cur else 0)
        if cur and size + extra > max_chars:
            chunks.append((PARAGRAPH_SEP.join(cur), spans))
            cur, spans, size = [], [], 0
            extra = len(text)
        spans.append((pid, size + (len(PARAGRAPH_SEP) if cur else 0), len(text)))
        cur.append(text)
        size += extra
    if cur:
        chunks.append((PARAGRAPH_SEP.join(cur), spans))
    return chunks


def to_paragraph(entity, spans):
    for pid, offset, length in spans:
        if offset <= entity["start"] and entity["end"] <= offset + length:
            return pid, entity["start"] - offset, entity["end"] - offset
    return None  # crosses a paragraph break


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "configs"))}


def environment():
    import gliner
    import tokenizers
    import torch
    import transformers
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "gliner": gliner.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_variant(model, variant, labels, chapters, cfg):
    model.configure(variant["splitter"], variant["space_mode"], variant.get("fix_normalizer", True))
    label_names = list(labels.values())
    back = {v: k for k, v in labels.items()}
    inputs = [(ch.chapter_id, text, spans) for ch in chapters for text, spans in chunk_chapter(ch, cfg["chunk_max_chars"])]
    lengths = [model.token_count(text, label_names) for _, text, _ in inputs]
    unk = sum(model.tokenizer(text)["input_ids"].count(model.tokenizer.unk_token_id) for _, text, _ in inputs)
    start = time.perf_counter()
    out = model.predict([text for _, text, _ in inputs], label_names, cfg["save_threshold"], cfg["batch_size"])
    seconds = time.perf_counter() - start
    predictions, crossing = [], 0
    for (cid, _, spans), entities in zip(inputs, out):
        for e in entities:
            where = to_paragraph(e, spans)
            if where is None:
                crossing += 1
                continue
            pid, s, t = where
            predictions.append({"chapter_id": cid, "paragraph_id": pid, "start": s, "end": t, "text": e["text"],
                                "label": back[e["label"]], "score": round(float(e["score"]), 4)})
    stats = {
        "chunks": len(inputs),
        "max_encoder_tokens": max(lengths),
        "chunks_over_512_tokens": sum(n > 512 for n in lengths),
        "unk_tokens_in_text": unk,
        "dropped_cross_paragraph": crossing,
        "inference_seconds": round(seconds, 2),
    }
    return predictions, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    ap.add_argument("--only", nargs="*", help="run only these variants")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    out_dir = REPO_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(cfg["snapshot"])
    chapter_ids = sample_chapters(snapshot, cfg["sample"])
    chapters = [snapshot.chapters[cid] for cid in chapter_ids]
    ref = names_from_mentions(snapshot.mentions, chapter_ids)
    variants = [v for v in cfg["variants"] if not args.only or v["name"] in args.only]

    from novel_ner.gliner_zh import BACKBONE_ID, BACKBONE_REVISION, MODEL_ID, MODEL_REVISION, ChineseGLiNER

    record = {
        "run_id": run_id,
        "config_file": str(Path(args.config).as_posix()),
        "config": cfg,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta["source_commit"]},
        "chapters": chapter_ids,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "backbone": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION},
        "reference": "weak/mentions.jsonl (LLM extraction; not gold)",
        "code": git_state(),
        "environment": environment(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    metrics = {"run_id": run_id, "n_chapters": len(chapter_ids),
               "dictionary_match": agreement(names_from_dictionary(snapshot.dictionary, chapters), ref),
               "variants": {}}
    model = ChineseGLiNER()
    for variant in variants:
        predictions, stats = run_variant(model, variant, cfg["labels"][variant["labels"]], chapters, cfg)
        with open(out_dir / f"predictions_{variant['name']}.jsonl", "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        by_threshold = {str(t): agreement(names_from_predictions(predictions, chapter_ids, t), ref)
                        for t in cfg["report_thresholds"]}
        best = max(by_threshold, key=lambda t: by_threshold[t]["exact"]["f1"])
        metrics["variants"][variant["name"]] = {"variant": variant, "stats": stats, "best_threshold": best,
                                                "by_threshold": by_threshold}
        e = by_threshold[best]["exact"]
        print(f"{variant['name']:28s} t={best} exact P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f}"
              f"  typed F1={by_threshold[best]['typed']['f1']:.3f}  {stats}", flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
