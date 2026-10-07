"""Build the local annotation page for one chapter set.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.annotate configs/annotate_dev.json
    .venv\\Scripts\\python -m novel_ner.annotate configs/annotate_test2.json     (also annotate_test.json, annotate_dev2.json)

The page holds whole chapters (plus the whole snapshot for cross-chapter search), so it
is written under data/annotate/ (not committed) and only opened locally, never published.
It shows no model or dictionary output, so the gold standard does not lean toward any
system being compared.

A set's chapters come from one of three config keys (chapter_order): chapters_from_order + n (test, test2: the
first n of a stratified order file), chapters_from_config (dev2: the "chapters" list of a sampling config, in its
order) or chapters_from_run + order_seed (dev: a run's chapters, shuffled).
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from novel_ner.snapshot import REPO_ROOT, load_snapshot

TEMPLATE = Path(__file__).with_name("templates") / "annotate.html"
PLACEHOLDER = "/*__DATA__*/"
TEXT_RULE = "chapter text = the snapshot's paragraphs joined by \\n; offsets are [start, end) in Unicode code points"


def text_sha256(chapter):
    return hashlib.sha256(chapter.text.encode("utf-8")).hexdigest()


def test_chapters(order_file, n):
    """The first n chapters of a fixed test order (configs/test_*.order.json)."""
    order = json.loads((REPO_ROOT / order_file).read_text(encoding="utf-8"))["order"]
    return [row["chapter_id"] for row in order[:n]]


def listed_chapters(config_file, set_name=None):
    """The "chapters" list of a sampling config (configs/dev2_781_990.json), in its order; the config's "set" and
    "n", when it has them, must agree with the set and the list."""
    cfg = json.loads((REPO_ROOT / config_file).read_text(encoding="utf-8"))
    ids = list(cfg["chapters"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{config_file}: chapters must be a non-empty list without repeats")
    if set_name is not None and cfg.get("set") not in (None, set_name):
        raise ValueError(f"{config_file} lists the chapters of {cfg.get('set')!r}, not {set_name!r}")
    if cfg.get("n") not in (None, len(ids)):
        raise ValueError(f"{config_file}: n = {cfg.get('n')} but {len(ids)} chapters are listed")
    return ids


def chapter_order(cfg):
    """The set's chapters in a fixed shuffled order: annotate from the top, stop anywhere.

    A test set takes the first n chapters of its stratified order; dev2 takes the chapter list of its sampling
    config as it stands. For the dev set, chapters added later ("extra_chapters") go after the shuffled ones, so
    positions already worked through never move.
    """
    if "chapters_from_order" in cfg:
        return test_chapters(cfg["chapters_from_order"], cfg["n"])
    if "chapters_from_config" in cfg:
        return listed_chapters(cfg["chapters_from_config"], cfg.get("set"))
    run = json.loads((REPO_ROOT / cfg["chapters_from_run"]).read_text(encoding="utf-8"))
    ids = sorted(run["chapters"])
    random.Random(cfg["order_seed"]).shuffle(ids)
    extra = cfg.get("extra_chapters", [])
    if len(set(extra)) != len(extra) or set(extra) & set(ids):
        raise ValueError("extra_chapters must be new and distinct")
    return ids + extra


def page_data(cfg, snapshot, order):
    for cid in order:
        if any(ord(c) > 0xFFFF for c in snapshot.chapters[cid].text):
            # the page indexes JavaScript strings, which count such characters twice
            raise ValueError(f"{cid} has characters outside the BMP; the page would mis-count offsets")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    book = sorted(snapshot.chapters.values(), key=lambda c: c.number)
    return {
        "set": cfg["set"],
        "snapshot_id": snapshot.id,
        "guide_version": cfg["guide_version"],
        "text_rule": TEXT_RULE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit": commit,
        "chapters": [
            {
                "chapter_id": cid,
                "number": snapshot.chapters[cid].number,
                "title": snapshot.chapters[cid].title,
                "content_hash": snapshot.chapters[cid].content_hash,
                "text_sha256": text_sha256(snapshot.chapters[cid]),
                "paragraphs": [list(p) for p in snapshot.chapters[cid].paragraphs],
            }
            for cid in order
        ],
        "book": [
            {"chapter_id": c.chapter_id, "number": c.number, "paragraphs": [list(p) for p in c.paragraphs]}
            for c in book
        ],
    }


def render(data):
    # "<" only occurs inside JSON strings, so escaping it keeps "</script>" out of the page
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("template must contain the data placeholder exactly once")
    return template.replace(PLACEHOLDER, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", help="default: data/annotate/<set>.html")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    snapshot = load_snapshot(cfg["snapshot"])
    order = chapter_order(cfg)
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "annotate" / f"{cfg['set']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(page_data(cfg, snapshot, order)), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("order:", " ".join(f"{k + 1}:{snapshot.chapters[cid].number}" for k, cid in enumerate(order)))


if __name__ == "__main__":
    main()
