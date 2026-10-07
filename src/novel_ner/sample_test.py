"""Fixed, stratified order of the formal test chapters.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.sample_test configs/test_781_990.json

Writes <config>.order.json (chapter ids only, committed): every prefix of the order holds each
plot unit in roughly its share of the range, so the test size can be fixed later without
changing which chapters come first.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

from novel_ner.annotate import chapter_order
from novel_ner.snapshot import REPO_ROOT, load_snapshot


def stratified_order(strata_chapters, seed):
    """strata_chapters: {stratum: [chapter ids]} -> one list where each stratum is spread evenly.

    Each stratum is shuffled, its k-th chapter (of n) gets the key (k + u) / n with a random
    offset u per stratum, and all chapters are sorted by key: systematic sampling across strata.
    """
    rng = random.Random(seed)
    keyed = []
    for stratum in sorted(strata_chapters):
        ids = sorted(strata_chapters[stratum])
        rng.shuffle(ids)
        u = rng.random()
        keyed += [((k + u) / len(ids), rng.random(), cid, stratum) for k, cid in enumerate(ids)]
    keyed.sort()
    return [(cid, stratum) for _, _, cid, stratum in keyed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    snapshot = load_snapshot(cfg["snapshot"])
    dev = set(chapter_order(json.loads((REPO_ROOT / cfg["exclude_dev_config"]).read_text(encoding="utf-8"))))
    lo, hi = cfg["range"]
    strata = {s["unit_id"]: [] for s in cfg["strata"]}
    for ch in snapshot.chapters.values():
        if lo <= ch.number <= hi and ch.chapter_id not in dev:
            unit = next(s["unit_id"] for s in cfg["strata"] if s["start"] <= ch.number <= s["end"])
            strata[unit].append(ch.chapter_id)
    order = stratified_order(strata, cfg["seed"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    out = cfg_path.with_suffix(".order.json")
    out.write_text(json.dumps({
        "config": cfg_path.name, "snapshot": snapshot.id, "seed": cfg["seed"], "code_commit": commit,
        "chapters_per_unit": {u: len(ids) for u, ids in strata.items()},
        "order": [{"chapter_id": cid, "number": snapshot.chapters[cid].number, "unit_id": unit} for cid, unit in order],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(order)} chapters; first 20:", " ".join(str(snapshot.chapters[c].number) for c, _ in order[:20]))


if __name__ == "__main__":
    main()
