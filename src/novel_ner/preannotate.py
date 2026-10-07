"""Pre-annotation demo: the frozen small model's boxes on whole chapters, with fixed mechanical flags for review.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.preannotate predict configs/preannotate_demo.json
    .venv\\Scripts\\python -m novel_ner.preannotate build configs/preannotate_demo.json [--run-id <id>]

`predict` labels the config's chapters after 990 ("new") with each of the three model runs (tagger.predict_run on
the later snapshot; run ids <model run>-<run_suffix>, which contain "preann"). `build` reads the three seeds'
predictions on the 60 test chapters (the existing predict-test runs) and on the new chapters, and applies the
config's fixed rules ("flags.rules"):

- shown boxes: a (start, end, type) at least `vote` of the three seeds output exactly; they never overlap, since one
  seed cannot output two overlapping spans (checked);
- flags on a shown box: seeds (not all three agree, or a one-seed box overlaps it), low (lowest score below
  low_score), new (text not in the dictionary or the Opus lists up to the cutoff), nonperson, title (a person box
  right before a title word, or ending in one), mixed (same text shown with another type in the chapter);
- miss candidates, where no shown box is: repeat (another occurrence of a shown box's text), minority (a one-seed
  box), dictionary (a dictionary match);
- groups: the flags and candidates of one text in one chapter (or one batch); a group goes to the questions for the
  user when a box of it is new or title, and to a reviewing model otherwise.

Nobody judges a box here. On the test chapters the existing answers (only read, never changed) are compared
mechanically, to count how the flags line up with the answers' differences from the boxes; the rules were written
after the error analysis of these chapters, so these counts are post hoc and descriptive.

runs/<id>/ gets config.json and metrics.json (counts and chapter ids only, no text). data/preannotate/<id>/ gets
view.html (the local viewer: chapter text, boxes, flags, and on test chapters the existing answers as a layer) and
questions.md (flagged places with their sentences); both hold novel text, are not committed and are never published.
"""
import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import dictionary_predictions, done_chapters, sha256
from novel_ner.gold import check as check_gold
from novel_ner.novelty import weak_names
from novel_ner.score import _touches, gold_spans
from novel_ner.snapshot import PARAGRAPH_SEP, REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.zeroshot import git_state

TEMPLATE = Path(__file__).with_name("templates") / "preannotate.html"
PLACEHOLDER = "/*__DATA__*/"
BOX_FLAGS = ("seeds", "low", "new", "nonperson", "title", "mixed")
MISS_KINDS = ("repeat", "minority", "dictionary")
TYPE_ZH = {"person": "人物", "organization": "组织", "location": "地点", "dungeon": "副本", "item": "物品", "trait": "特性"}
FLAG_ZH = {
    "seeds": "三次训练不一致", "low": "模型没把握", "new": "没见过的名字", "nonperson": "人物以外的类型",
    "title": "紧挨称谓", "mixed": "同一串字类型不同",
    "repeat": "本章别处标过、这里没标", "minority": "只有一次训练标了", "dictionary": "词典里有、模型没标",
}


# --- inputs -----------------------------------------------------------------------------------------------------

def run_dir(ref):
    name = Path(ref).name
    path = REPO_ROOT / "runs" / name
    if not (path / "metrics.json").exists():
        raise SystemExit(f"runs/{name} has no metrics.json")
    return path


def read_predict_run(ref, model_run, chapter_ids, snapshot_id):
    """One seed's predictions {chapter_id: [(start, end, type, score)]}, after checking the run's record: the file's
    sha256, the model run it came from, its snapshot and that it covers the chapters."""
    path = run_dir(ref)
    rec = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    pred_file = path / "predictions.jsonl"
    if sha256(pred_file) != metrics["predictions_sha256"]:
        raise SystemExit(f"{path.name}: predictions.jsonl differs from the file the run recorded")
    if Path(rec["model_run"]).name != Path(model_run).name:
        raise SystemExit(f"{path.name} comes from {rec['model_run']}, not {model_run}")
    if rec["snapshot"]["id"] != snapshot_id:
        raise SystemExit(f"{path.name} read snapshot {rec['snapshot']['id']}, not {snapshot_id}")
    if not set(chapter_ids) <= set(rec["chapters"]):
        raise SystemExit(f"{path.name} does not cover every chapter asked for")
    out = defaultdict(list)
    for r in read_jsonl(pred_file):
        if r["chapter_id"] in chapter_ids:
            out[r["chapter_id"]].append((r["start"], r["end"], r["type"], r["score"]))
    record = {"run": f"runs/{path.name}", "predictions_sha256": metrics["predictions_sha256"],
              "model_run": f"runs/{Path(model_run).name}", "seed": rec.get("seed"), "code": rec.get("code")}
    return dict(out), record


def title_words(cfg):
    words = json.loads((REPO_ROOT / cfg["flags"]["title_suffixes_from"]).read_text(encoding="utf-8"))
    found = []

    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "title_suffixes":
                    found.append(v)
                walk(v)
    walk(words)
    if len(found) != 1:
        raise SystemExit(f"{cfg['flags']['title_suffixes_from']} must hold one title_suffixes list")
    return sorted(found[0], key=len, reverse=True)


# --- boxes and flags ----------------------------------------------------------------------------------------------

def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def make_boxes(per_seed, vote):
    """(shown, minority) from the seeds' spans of one chapter. A box is {start, end, type, seeds: [seed index],
    score: lowest score among them}."""
    votes = defaultdict(list)
    for i, spans in enumerate(per_seed):
        for s, e, t, score in spans:
            votes[(s, e, t)].append((i, score))
    boxes = [{"start": s, "end": e, "type": t, "seeds": sorted(i for i, _ in v), "score": min(x for _, x in v)}
             for (s, e, t), v in sorted(votes.items())]
    shown = [b for b in boxes if len(b["seeds"]) >= vote]
    minority = [b for b in boxes if len(b["seeds"]) < vote]
    for a, b in zip(shown, shown[1:]):
        if overlaps((a["start"], a["end"]), (b["start"], b["end"])):
            raise ValueError(f"shown boxes overlap at {a['start']} and {b['start']}")
    return shown, minority


def title_after(text, box, titles):
    """The title word right after a person box, or the one its own text ends with (and is longer than)."""
    if box["type"] != "person":
        return None
    own = text[box["start"]:box["end"]]
    for w in titles:
        if text.startswith(w, box["end"]) or (own.endswith(w) and len(own) > len(w)):
            return w
    return None


def find_all(text, needle):
    at = text.find(needle)
    while at != -1:
        yield at
        at = text.find(needle, at + 1)


def flag_chapter(text, per_seed, seen, titles, dict_matches, flags_cfg):
    """Shown boxes with their flags, and miss candidates, of one chapter."""
    shown, minority = make_boxes(per_seed, flags_cfg["vote"])
    n_seeds = len(per_seed)
    types_of = defaultdict(set)
    for b in shown:
        b["text"] = text[b["start"]:b["end"]]
        types_of[b["text"]].add(b["type"])
    for b in shown:
        reasons = []
        if len(b["seeds"]) < n_seeds or any(overlaps((m["start"], m["end"]), (b["start"], b["end"])) for m in minority):
            reasons.append("seeds")
        if b["score"] < flags_cfg["low_score"]:
            reasons.append("low")
        if b["text"] not in seen:
            reasons.append("new")
        if b["type"] != "person":
            reasons.append("nonperson")
        b["title_word"] = title_after(text, b, titles)
        if b["title_word"]:
            reasons.append("title")
        if len(types_of[b["text"]]) > 1:
            reasons.append("mixed")
        b["flags"] = reasons
    covered = [(b["start"], b["end"]) for b in shown]
    misses = {}

    def add(kind, s, e, hint):
        if _touches((s, e), covered):
            return
        m = misses.setdefault((s, e), {"start": s, "end": e, "text": text[s:e], "kinds": [], "types": []})
        if kind not in m["kinds"]:
            m["kinds"].append(kind)
        for t in hint:
            if t not in m["types"]:
                m["types"].append(t)

    for name in sorted(n for n in types_of if len(n) >= flags_cfg["repeat_min_len"]):
        for at in find_all(text, name):
            add("repeat", at, at + len(name), sorted(types_of[name]))
    for m in minority:
        add("minority", m["start"], m["end"], [m["type"]])
    for d in dict_matches:
        add("dictionary", d["start"], d["end"], [d["type"]])
    for m in misses.values():
        m["kinds"].sort(key=MISS_KINDS.index)
    return shown, minority, sorted(misses.values(), key=lambda m: (m["start"], m["end"]))


def groups(chapters):
    """Flagged boxes and miss candidates grouped by text over the given chapters [(chapter_id, shown, misses)]."""
    out = {}
    for cid, shown, misses in chapters:
        for b in shown:
            if not b["flags"]:
                continue
            g = out.setdefault(b["text"], {"text": b["text"], "boxes": [], "misses": [], "flags": set(), "types": Counter()})
            g["boxes"].append((cid, b["start"], b["end"], b["type"]))
            g["flags"].update(b["flags"])
            g["types"][b["type"]] += 1
        for m in misses:
            g = out.setdefault(m["text"], {"text": m["text"], "boxes": [], "misses": [], "flags": set(), "types": Counter()})
            g["misses"].append((cid, m["start"], m["end"]))
            g["flags"].update(m["kinds"])
    for g in out.values():
        g["ask_user"] = bool(g["flags"] & {"new", "title"}) and bool(g["boxes"])
        g["flags"] = [f for f in (*BOX_FLAGS, *MISS_KINDS) if f in g["flags"]]
    return sorted(out.values(), key=lambda g: (not g["ask_user"], -len(g["boxes"]) - len(g["misses"]), g["text"]))


# --- comparison with the existing answers (test chapters) -------------------------------------------------------

def compare_chapter(shown, misses, ref):
    """Counts of one chapter's shown boxes and miss candidates against its existing answers (strict match; boxes and
    answers touching an unscored range are left out, as in scoring)."""
    gold, unscored = ref["spans"], ref["unscored"]
    c = Counter()
    boxes = [b for b in shown if not _touches((b["start"], b["end"]), unscored)]
    c["boxes_touching_unscored"] = len(shown) - len(boxes)
    exact = set()
    for b in boxes:
        key = (b["start"], b["end"], b["type"])
        ok = key in gold
        if ok:
            exact.add(key)
        c["boxes"] += 1
        c["boxes_wrong"] += not ok
        flagged = bool(b["flags"])
        c["flagged"] += flagged
        c["flagged_wrong"] += flagged and not ok
        c["unflagged"] += not flagged
        c["unflagged_wrong"] += not flagged and not ok
        for f in b["flags"]:
            c[f"flag_{f}"] += 1
            c[f"flag_{f}_wrong"] += not ok
    scored_gold = [g for g in gold if not _touches(g[:2], unscored)]
    flagged_spans = [(b["start"], b["end"]) for b in boxes if b["flags"]]
    unflagged_spans = [(b["start"], b["end"]) for b in boxes if not b["flags"]]
    # like boxes, a candidate touching an unscored range is left out (it neither counts nor catches)
    cand_spans = [(m["start"], m["end"]) for m in misses if not _touches((m["start"], m["end"]), unscored)]
    c["gold"] = len(scored_gold)
    for g in scored_gold:
        if g in exact:
            continue
        c["gold_missed"] += 1
        by_flag = _touches(g[:2], flagged_spans)
        by_cand = _touches(g[:2], cand_spans)
        c["gold_missed_caught_by_flagged_box"] += by_flag
        c["gold_missed_caught_by_candidate"] += by_cand
        c["gold_missed_caught"] += by_flag or by_cand
        c["gold_missed_only_unflagged_box"] += not (by_flag or by_cand) and _touches(g[:2], unflagged_spans)
        c["gold_missed_nothing_near"] += not (by_flag or by_cand) and not _touches(g[:2], unflagged_spans)
    missed_spans = [g[:2] for g in scored_gold if g not in exact]
    for m in misses:
        if _touches((m["start"], m["end"]), unscored):
            c["candidates_touching_unscored"] += 1
            continue
        for kind in m["kinds"]:
            c[f"cand_{kind}"] += 1
            c[f"cand_{kind}_on_missed_answer"] += _touches((m["start"], m["end"]), missed_spans)
        c["candidates"] += 1
        c["candidates_on_missed_answer"] += _touches((m["start"], m["end"]), missed_spans)
    return c


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def seed_counts(per_seed_rows, gold, ids):
    """Strict tp/fp/fn of each seed alone (to check the answers are read as the scoring run read them)."""
    out = []
    for rows in per_seed_rows:
        tp = fp = fn = 0
        for cid in ids:
            ref = gold[cid]
            pred = {(s, e, t) for s, e, t, _ in rows.get(cid, []) if not _touches((s, e), ref["unscored"])}
            g = {x for x in ref["spans"] if not _touches(x[:2], ref["unscored"])}
            tp, fp, fn = tp + len(pred & g), fp + len(pred - g), fn + len(g - pred)
        out.append({"tp": tp, "fp": fp, "fn": fn})
    return out


def eval_seed_counts(eval_run, systems):
    m = json.loads((run_dir(eval_run) / "metrics.json").read_text(encoding="utf-8"))
    out = []
    for name in systems:
        per = m["scores"][name]["per_type"]
        out.append({k: sum(per[t][k] for t in ENTITY_TYPES) for k in ("tp", "fp", "fn")})
    return out


def rates(c):
    """Shares from the summed counts (None when the denominator is 0)."""
    share = lambda a, b: round(a / b, 4) if b else None
    out = {
        "shown_boxes_strict": {"tp": c["boxes"] - c["boxes_wrong"], "fp": c["boxes_wrong"], "fn": c["gold_missed"],
                               "precision": share(c["boxes"] - c["boxes_wrong"], c["boxes"]),
                               "recall": share(c["gold"] - c["gold_missed"], c["gold"]),
                               "f1": f1(c["boxes"] - c["boxes_wrong"], c["boxes_wrong"], c["gold_missed"])},
        "flagged_share_of_boxes": share(c["flagged"], c["boxes"]),
        "wrong_share_among_flagged": share(c["flagged_wrong"], c["flagged"]),
        "wrong_share_among_unflagged": share(c["unflagged_wrong"], c["unflagged"]),
        "wrong_boxes_flagged_share": share(c["flagged_wrong"], c["boxes_wrong"]),
        "missed_answers_caught_share": share(c["gold_missed_caught"], c["gold_missed"]),
        "candidates_on_missed_answer_share": share(c["candidates_on_missed_answer"], c["candidates"]),
        "per_flag": {f: {"boxes": c[f"flag_{f}"], "wrong": c[f"flag_{f}_wrong"],
                         "wrong_share": share(c[f"flag_{f}_wrong"], c[f"flag_{f}"])} for f in BOX_FLAGS},
        "per_candidate_kind": {k: {"candidates": c[f"cand_{k}"], "on_missed_answer": c[f"cand_{k}_on_missed_answer"],
                                   "share": share(c[f"cand_{k}_on_missed_answer"], c[f"cand_{k}"])} for k in MISS_KINDS},
        "left_after_a_perfect_review_of_flags": {
            "wrong_unflagged_boxes": c["unflagged_wrong"],
            "missed_answers_not_caught": c["gold_missed"] - c["gold_missed_caught"]},
    }
    return out


def exact_f1(tp, fp, fn):
    return Fraction(2 * tp, 2 * tp + fp + fn) if tp else Fraction(0)


def pick_showcase(per_chapter, nonperson):
    """The chapter whose F1 is nearest the median of all chapters; ties: more non-person answers, then the lower
    chapter number. Exact fractions, so a tie is a tie (with an even count the two middle chapters always are);
    returns the F1 and the median rounded to 4 decimals."""
    values = {cid: exact_f1(c["boxes"] - c["boxes_wrong"], c["boxes_wrong"], c["gold_missed"]) for cid, c in per_chapter.items()}
    median = statistics.median(values.values())
    best = min(values, key=lambda cid: (abs(values[cid] - median), -nonperson[cid], int(cid[2:])))
    return best, round(float(values[best]), 4), round(float(median), 4)


# --- page -----------------------------------------------------------------------------------------------------------

def chapter_view(ch, kind, shown, minority, misses, per_seed, groups_, ref=None):
    """One chapter's data for the page."""
    view = {
        "chapter_id": ch.chapter_id, "number": ch.number, "title": ch.title, "kind": kind,
        "paragraphs": [list(p) for p in ch.paragraphs],
        "boxes": [{"s": b["start"], "e": b["end"], "t": b["type"], "seeds": b["seeds"], "score": round(b["score"], 4),
                   "flags": b["flags"], "title_word": b["title_word"]} for b in shown],
        "minority": [{"s": m["start"], "e": m["end"], "t": m["type"], "seeds": m["seeds"], "score": round(m["score"], 4)}
                     for m in minority],
        "misses": [{"s": m["start"], "e": m["end"], "kinds": m["kinds"], "types": m["types"]} for m in misses],
        "seeds": [[[s, e, t, round(x, 4)] for s, e, t, x in spans] for spans in per_seed],
        "groups": [{"text": g["text"], "ask_user": g["ask_user"], "flags": g["flags"], "types": dict(g["types"]),
                    "boxes": [[s, e] for _, s, e, _ in g["boxes"]], "misses": [[s, e] for _, s, e in g["misses"]]}
                   for g in groups_],
        "gold": None, "unscored": [],
    }
    if ref is not None:
        view["gold"] = sorted([s, e, t] for s, e, t in ref["spans"])
        view["unscored"] = sorted([s, e] for s, e in ref["unscored"])
    return view


def render(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("template must contain the data placeholder exactly once")
    return template.replace(PLACEHOLDER, payload)


def context(text, s, e, width=18):
    """The place in its paragraph, at most `width` characters each side ("…" where the paragraph goes on)."""
    p0 = text.rfind(PARAGRAPH_SEP, 0, s) + 1
    p1 = text.find(PARAGRAPH_SEP, e)
    p1 = len(text) if p1 == -1 else p1
    a, b = max(p0, s - width), min(p1, e + width)
    return f"{'…' if a > p0 else ''}{text[a:s]}【{text[s:e]}】{text[e:b]}{'…' if b < p1 else ''}"


def questions_md(title, batch_groups, texts, numbers, cfg_file, runs):
    ask = [g for g in batch_groups if g["ask_user"]]
    review = [g for g in batch_groups if not g["ask_user"]]
    lines = [f"# {title}", "",
             f"模型：S3 三个种子（只训到第 780 章，{', '.join(runs)}）；挂旗规则：`{cfg_file}`（固定、机械，没人判对错）。",
             "框 = 三次训练里至少两次一模一样的输出。", ""]

    def where(g):
        chs = sorted({numbers[cid] for cid, *_ in g["boxes"]} | {numbers[cid] for cid, *_ in g["misses"]})
        return "、".join(f"第 {n} 章" for n in chs)

    def examples(g, k=3):
        out = [f"  - {context(texts[cid], s, e)}（第 {numbers[cid]} 章）" for cid, s, e, _ in g["boxes"][:k]]
        if len(out) < k:
            out += [f"  - {context(texts[cid], s, e)}（第 {numbers[cid]} 章，模型没标）" for cid, s, e in g["misses"][:k - len(out)]]
        return out

    def typestr(g):
        return "、".join(f"{TYPE_ZH[t]} {n} 处" for t, n in g["types"].most_common()) or "没标"

    lines += [f"## 一、问你（书里的知识）：{len(ask)} 组", "",
              "只需要用白话说这是什么；按规则怎么框由复核模型照你的说法去做。", ""]
    for i, g in enumerate(ask, 1):
        lines.append(f"### {i}. 「{g['text']}」 —— 模型标成：{typestr(g)}；{where(g)}")
        lines.append(f"- 挂旗原因：{'、'.join(FLAG_ZH[f] for f in g['flags'])}")
        q = []
        if "new" in g["flags"]:
            q.append("它是不是一个名字？是谁或是什么（人物、地点、组织、副本、物品、特性）？")
        if "title" in g["flags"]:
            word = next((b for b in g.get("title_words", []) if b), None)
            q.append(f"它和后面的称谓{f'「{word}」' if word else ''}是一个固定称呼（整串是名字），还是姓名加称谓（只框名字）？")
        lines.append(f"- 要回答：{' '.join(q)}")
        lines += examples(g)
        lines.append("")
    lines += [f"## 二、交给复核模型（按规则判）：{len(review)} 组", "",
              "这些是模型拿不准、类型容易错或可能漏标的地方，不需要书外知识，交给复核模型按规则 v1.1 判。", ""]
    for i, g in enumerate(review, 1):
        lines.append(f"- {i}. 「{g['text']}」 —— 模型标成：{typestr(g)}；{where(g)}；"
                     f"原因：{'、'.join(FLAG_ZH[f] for f in g['flags'])}")
        lines += ["  " + x for x in examples(g, 1)]
    return "\n".join(lines) + "\n"


# --- commands -------------------------------------------------------------------------------------------------------

def preann_run_id(model_run, cfg):
    return f"{Path(model_run).name}-{cfg['new']['run_suffix']}"


def predict(cfg):
    from novel_ner.tagger import predict_run

    snapshot = load_snapshot(cfg["new"]["snapshot"])
    for model_run in cfg["model_runs"]:
        run_id = preann_run_id(model_run, cfg)
        if (REPO_ROOT / "runs" / run_id / "metrics.json").exists():
            raise SystemExit(f"runs/{run_id} already has results")
        predict_run(model_run, cfg["new"]["chapters"], run_id, snapshot=snapshot)


def build(cfg, config_file, run_id):
    from novel_ner.tagger import plain_run_id

    run_id = plain_run_id(run_id)  # one folder name: the page must stay under data/ and the record under runs/
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    began = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    flags_cfg, titles = cfg["flags"], title_words(cfg)
    if len(cfg["model_runs"]) != 3 or flags_cfg["vote"] != 2:
        raise SystemExit("the rules are written for three seeds and a vote of two")

    # test chapters: the existing predictions and answers
    sc = cfg["showcase"]
    snap_t = load_snapshot(sc["snapshot"])
    gold_file = REPO_ROOT / sc["gold_test"]
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snap_t)
    if errors:
        raise SystemExit(f"the test answers do not pass the gold check: {errors[:3]}")
    test_ids = done_chapters(export)
    gold = gold_spans(export, test_ids)
    seeds_t, records_t = zip(*(read_predict_run(r, m, test_ids, snap_t.id)
                               for r, m in zip(sc["predict_runs"], cfg["model_runs"])))
    eval_check = None
    if sc.get("eval_run"):
        mine, theirs = seed_counts(seeds_t, gold, test_ids), eval_seed_counts(sc["eval_run"], sc["eval_systems"])
        eval_check = {"eval_run": sc["eval_run"], "systems": sc["eval_systems"], "seed_counts": mine, "same": mine == theirs}
        if mine != theirs:
            raise SystemExit(f"the seeds' counts differ from {sc['eval_run']}: {mine} vs {theirs}")
    dict_names, mention_names, seen_src_t = weak_names(snap_t, sc["seen"]["cutoff_chapter"])
    seen_t = dict_names | mention_names
    dict_t, _ = dictionary_predictions(snap_t.dictionary, [snap_t.chapters[c] for c in test_ids], flags_cfg["dictionary_min_len"])
    dict_by = defaultdict(list)
    for d in dict_t:
        dict_by[d["chapter_id"]].append(d)
    test_views, per_chapter, totals = {}, {}, Counter()
    nonperson = {cid: sum(t != "person" for s, e, t in gold[cid]["spans"] if not _touches((s, e), gold[cid]["unscored"]))
                 for cid in test_ids}
    for cid in test_ids:
        ch = snap_t.chapters[cid]
        per_seed = [sorted(s.get(cid, [])) for s in seeds_t]
        shown, minority, misses = flag_chapter(ch.text, per_seed, seen_t, titles, dict_by[cid], flags_cfg)
        c = compare_chapter(shown, misses, gold[cid])
        per_chapter[cid] = c
        totals.update(c)
        test_views[cid] = (ch, shown, minority, misses, per_seed)
    showcase, showcase_f1, median = pick_showcase(per_chapter, nonperson)

    # new chapters: the preann predictions
    nw = cfg["new"]
    snap_n = load_snapshot(nw["snapshot"])
    new_ids = nw["chapters"]
    if any(snap_n.chapters[c].number <= 990 for c in new_ids):
        raise SystemExit("the new chapters must be after 990")
    seeds_n, records_n = zip(*(read_predict_run(preann_run_id(m, cfg), m, new_ids, snap_n.id) for m in cfg["model_runs"]))
    dict_names, mention_names, seen_src_n = weak_names(snap_n, nw["seen"]["cutoff_chapter"])
    seen_n = dict_names | mention_names
    dict_n, _ = dictionary_predictions(snap_n.dictionary, [snap_n.chapters[c] for c in new_ids], flags_cfg["dictionary_min_len"])
    dict_nby = defaultdict(list)
    for d in dict_n:
        dict_nby[d["chapter_id"]].append(d)
    new_views, new_counts = {}, {}
    for cid in new_ids:
        ch = snap_n.chapters[cid]
        per_seed = [sorted(s.get(cid, [])) for s in seeds_n]
        shown, minority, misses = flag_chapter(ch.text, per_seed, seen_n, titles, dict_nby[cid], flags_cfg)
        new_views[cid] = (ch, shown, minority, misses, per_seed)
        new_counts[cid] = burden(shown, misses)
    batch = groups([(cid, v[1], v[3]) for cid, v in new_views.items()])
    words = defaultdict(set)
    for cid, (_, shown, _, _, _) in new_views.items():
        for b in shown:
            if b["title_word"]:
                words[b["text"]].add(b["title_word"])
    for g in batch:
        g["title_words"] = sorted(words.get(g["text"], ()))

    # page and questions
    for ch in [snap_t.chapters[c] for c in test_ids] + [snap_n.chapters[c] for c in new_ids]:
        if any(ord(x) > 0xFFFF for x in ch.text):
            raise ValueError(f"{ch.chapter_id} has characters outside the BMP; the page would mis-count offsets")
    order = [showcase] + [c for c in new_ids] + sorted((c for c in test_ids if c != showcase), key=lambda c: snap_t.chapters[c].number)
    chapters = []
    for cid in order:
        if cid in new_views:
            ch, shown, minority, misses, per_seed = new_views[cid]
            chapters.append(chapter_view(ch, "new", shown, minority, misses, per_seed, groups([(cid, shown, misses)])))
        else:
            ch, shown, minority, misses, per_seed = test_views[cid]
            chapters.append(chapter_view(ch, "showcase" if cid == showcase else "test", shown, minority, misses, per_seed,
                                         groups([(cid, shown, misses)]), gold[cid]))
    test_rates = rates(totals)
    data = {
        "run_id": run_id, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config_file": config_file,
        "code": git_state(), "model_runs": [f"runs/{Path(m).name}" for m in cfg["model_runs"]],
        "snapshots": {"test": snap_t.id, "new": snap_n.id}, "gold_test": sc["gold_test"],
        "low_score": flags_cfg["low_score"], "seen_cutoff": {"test": sc["seen"]["cutoff_chapter"], "new": nw["seen"]["cutoff_chapter"]},
        "type_zh": TYPE_ZH, "flag_zh": FLAG_ZH,
        "showcase": {"chapter_id": showcase, "f1": showcase_f1, "median_f1": median},
        "test_summary": {"chapters": len(test_ids), "counts": dict(totals), "rates": test_rates},
        "chapters": chapters,
    }
    page_dir = REPO_ROOT / "data" / "preannotate" / run_id
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "view.html").write_text(render(data), encoding="utf-8")
    texts = {cid: snap_n.chapters[cid].text for cid in new_ids}
    numbers = {cid: snap_n.chapters[cid].number for cid in new_ids}
    first, last = numbers[new_ids[0]], numbers[new_ids[-1]]
    (page_dir / "questions.md").write_text(questions_md(
        f"第 {first}–{last} 章初标：要检查的地方（按名字归组）", batch, texts, numbers, config_file,
        [r["run"] for r in records_n]), encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "config_file": config_file, "config": cfg, "code": git_state(), "started_at": began,
        "snapshots": {"test": {"id": snap_t.id, "source_commit": snap_t.meta.get("source_commit")},
                      "new": {"id": snap_n.id, "source_commit": snap_n.meta.get("source_commit")}},
        "gold_test": {"file": sc["gold_test"], "sha256": sha256(gold_file), "chapters": len(test_ids)},
        "predict_runs": {"test": list(records_t), "new": list(records_n)},
        "title_words": len(titles),
        "seen_sources": {"test": seen_src_t, "new": seen_src_n},
        "outputs_not_committed": [f"data/preannotate/{run_id}/view.html", f"data/preannotate/{run_id}/questions.md"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {
        "run_id": run_id,
        "note": "counts only. The flag rules were fixed (configs/preannotate_demo.json, committed) before they were "
                "computed here, but written after the error analysis of the same test chapters: the test counts are post "
                "hoc and descriptive, likely optimistic for later chapters. The answers are the silver standard v3.",
        "seed_check_against_scoring": eval_check,
        "showcase": {"chapter_id": showcase, "shown_boxes_f1": showcase_f1, "median_over_test_chapters": median,
                     "rule": sc["pick_rule"], "counts": dict(per_chapter[showcase])},
        "test": {"chapters": len(test_ids), "counts": dict(totals), "rates": test_rates,
                 "per_chapter_mean": burden_mean([burden_from_views(test_views[c]) for c in test_ids]),
                 "groups_over_the_60_chapters": group_counts(groups([(c, test_views[c][1], test_views[c][3]) for c in test_ids]))},
        "new": {"chapters": new_ids, "per_chapter": new_counts, "batch_groups": group_counts(batch)},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"showcase {showcase} (F1 {showcase_f1}, median {median}); wrote {out_dir} and {page_dir}")
    return metrics


def burden(shown, misses):
    c = Counter(boxes=len(shown), flagged=sum(bool(b["flags"]) for b in shown), candidates=len(misses),
                all_three_seeds=sum(len(b["seeds"]) == 3 for b in shown))
    for b in shown:
        c[f"type_{b['type']}"] += 1
        for f in b["flags"]:
            c[f"flag_{f}"] += 1
    for m in misses:
        for k in m["kinds"]:
            c[f"cand_{k}"] += 1
    g = groups([("x", shown, misses)])
    c["groups_ask_user"] = sum(x["ask_user"] for x in g)
    c["groups_review"] = sum(not x["ask_user"] for x in g)
    return dict(c)


def burden_from_views(view):
    _, shown, _, misses, _ = view
    return burden(shown, misses)


def burden_mean(rows):
    keys = sorted({k for r in rows for k in r})
    return {k: round(sum(r.get(k, 0) for r in rows) / len(rows), 2) for k in keys}


def group_counts(gs):
    return {"groups": len(gs), "ask_user": sum(g["ask_user"] for g in gs), "review": sum(not g["ask_user"] for g in gs),
            "ask_user_by_flag": dict(Counter(f for g in gs if g["ask_user"] for f in g["flags"] if f in ("new", "title"))),
            "places_in_ask_user_groups": sum(len(g["boxes"]) + len(g["misses"]) for g in gs if g["ask_user"]),
            "places_in_review_groups": sum(len(g["boxes"]) + len(g["misses"]) for g in gs if not g["ask_user"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["predict", "build"])
    ap.add_argument("config")
    ap.add_argument("--run-id", help="build only; default: <date>-preannotate-demo")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.action == "predict":
        predict(cfg)
    else:
        build(cfg, Path(args.config).as_posix(), args.run_id or f"{time.strftime('%Y%m%d')}-preannotate-demo")


if __name__ == "__main__":
    main()
