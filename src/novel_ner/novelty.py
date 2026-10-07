"""How many test answers are names the weak sources never saw (a description of the test data).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.novelty configs/novelty_test.json [--run-id <id>]

A scored test answer is "new" when its exact text is neither a name in the snapshot's dictionary nor a
name in the LLM mentions of chapters up to the cutoff (exact string, any type). This only describes the
test data; nothing is chosen with it. A known character written in a new form (a bare surname, say) also
counts as new, so the share is an upper bound on new characters.

runs/<run_id>/ gets config.json and metrics.json: counts only, no names and no text.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import done_chapters, sha256
from novel_ner.gold import check as check_gold
from novel_ner.score import _touches, gold_spans
from novel_ner.snapshot import REPO_ROOT, load_snapshot
from novel_ner.zeroshot import git_state


def chapter_number(chapter_id):
    return int(chapter_id[2:])


def weak_names(snapshot, cutoff, dictionary_cutoff=None, dev_ids=None, test_ids=None):
    """Names in the dictionary and in the mentions of chapters <= cutoff, plus what the check found.

    dictionary_cutoff = N (a learning-curve tier; None = the whole dictionary, the pre-registered rule) keeps exactly
    the dictionary rows the tier's weak build keeps: weak.clean_dictionary with the cutoff over the training texts of
    chapters <= N (development chapters left out, as in the build) and the first Opus listings up to 780 dating the
    rows without a reveal chapter; its length drop is left out (min_len 1), like the whole-dictionary rule (a name with a
    parenthesis, which it also drops, never equals an answer).
    The mention names are those of every chapter <= cutoff (development chapters too), the same form as the
    pre-registered rule."""
    rows = snapshot.dictionary
    if dictionary_cutoff is not None:
        from novel_ner.annotate import chapter_order
        from novel_ner.weak import clean_dictionary, first_listed, split_ids

        # the development and test chapters the weak builds use (configs/annotate_dev.json, the test order) unless given
        if dev_ids is None:
            dev_ids = chapter_order(json.loads((REPO_ROOT / "configs" / "annotate_dev.json").read_text(encoding="utf-8")))
        if test_ids is None:
            test_ids = [r["chapter_id"] for r in json.loads((REPO_ROOT / "configs" / "test_781_990.order.json")
                                                             .read_text(encoding="utf-8"))["order"]]
        split = split_ids(snapshot.chapters, dev_ids, test_ids, dictionary_cutoff)
        texts = [snapshot.chapters[c].text for c in split["train"]]
        number = {cid: ch.number for cid, ch in snapshot.chapters.items()}
        rows, _ = clean_dictionary(rows, 1, dictionary_cutoff, texts, listed_at=first_listed(snapshot.mentions, number))
    dictionary = {r["name"] for r in rows}
    mentions = {m["name"] for m in snapshot.mentions if chapter_number(m["chapter_id"]) <= cutoff}
    known = [r["known"] for r in snapshot.dictionary if r.get("known") is not None]
    out = {
        "dictionary_distinct_names": len(dictionary),
        "mention_distinct_names_to_cutoff": len(mentions),
        "union_distinct_names": len(dictionary | mentions),
        "mention_rows_after_cutoff_ignored": sum(1 for m in snapshot.mentions if chapter_number(m["chapter_id"]) > cutoff),
        "dictionary_known_max": max(known) if known else None,
        "dictionary_rows_known_after_cutoff": sum(1 for k in known if k > cutoff),
    }
    if dictionary_cutoff is not None:
        out.update({"dictionary_cutoff": dictionary_cutoff, "dictionary_rows_kept": len(rows),
                    "dictionary_rows_dropped": len(snapshot.dictionary) - len(rows)})
    return dictionary, mentions, out


def novelty(gold, texts, seen):
    """gold: score.gold_spans output; texts: chapter id -> chapter text; seen: set of names."""
    spans = [(texts[cid][s:e], t) for cid, ref in gold.items() for s, e, t in ref["spans"]
             if not _touches((s, e), ref["unscored"])]
    new = [(x, t) for x, t in spans if x not in seen]
    by_type = {}
    for t in ENTITY_TYPES:
        all_t = [x for x, tt in spans if tt == t]
        new_t = [x for x, tt in new if tt == t]
        by_type[t] = {"spans": len(all_t), "new_spans": len(new_t),
                      "distinct_names": len(set(all_t)), "new_distinct_names": len(set(new_t))}
    texts_all, texts_new = {x for x, _ in spans}, {x for x, _ in new}
    return {
        "spans": len(spans), "new_spans": len(new),
        "new_share": round(len(new) / len(spans), 4) if spans else None,
        "distinct_texts": len(texts_all), "new_distinct_texts": len(texts_new),
        "distinct_text_type_pairs": len(set(spans)), "new_distinct_text_type_pairs": len(set(new)),
        "by_type": by_type,
        "new_spans_top1_text_share": round(Counter(x for x, _ in new).most_common(1)[0][1] / len(new), 4) if new else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")

    snapshot = load_snapshot(cfg["snapshot"])
    gold_file = REPO_ROOT / cfg["gold_test"]
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the test answers do not pass the gold check: {errors[:3]}")
    ids = done_chapters(export)
    cutoff = cfg["cutoff_chapter"]
    if min(chapter_number(c) for c in ids) <= cutoff:
        raise SystemExit("a test chapter is at or before the cutoff")

    dictionary, mentions, sources = weak_names(snapshot, cutoff)
    counts = novelty(gold_spans(export, ids), {c: snapshot.chapters[c].text for c in ids}, dictionary | mentions)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "config_file": Path(args.config).as_posix(), "config": cfg,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta["source_commit"]},
        "gold_test": {"file": cfg["gold_test"], "sha256": sha256(gold_file), "chapters": len(ids)},
        "code": git_state(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {
        "run_id": run_id,
        "rule": "a scored test answer is new when its exact text is not a dictionary name and not a mention name "
                f"of chapters <= {cutoff} (any type); describes the test data only, nothing is chosen with it",
        "weak_sources": sources, "test": counts,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: counts[k] for k in ("spans", "new_spans", "new_share", "distinct_texts", "new_distinct_texts")}))


if __name__ == "__main__":
    main()
