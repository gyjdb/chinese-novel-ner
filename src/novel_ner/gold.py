"""Check annotation files exported from the annotation page, and keep them.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.gold check     <exported.json>
    .venv\\Scripts\\python -m novel_ner.gold save      <exported.json>
    .venv\\Scripts\\python -m novel_ner.gold questions <exported.json> [--chapters ch0046 ...]
    .venv\\Scripts\\python -m novel_ner.gold revise    <saved.json> --decisions configs/<decisions>.json [--set <set>]

`save` copies a file that passes the check to data/gold/<set>/ under a new name; earlier
files are never overwritten (a revision is a new file). `revise` applies the user's recorded
decisions (a committed config) to a saved file -- or, for the dev, dev2 and test2 answers, to the last prefilled
file (review.py pending) -- mechanically and saves the result as a new version under data/gold/<set>/:
pending marks the user settled get their type, and words the user could not decide become unscored
ranges (guide appendix 1: anything touching one counts for nothing, for every system). `questions` writes the pending items
and annotations with notes, each with its paragraph, as a Markdown list for discussing the
rules (it quotes the novel, so it goes under data/ and is never committed). data/ is not committed.

Sets (SETS): test (60 chapters, 781-990), dev (31 chapters), dev2 (25 decision chapters, 781-990 outside the test;
stage 0) and test2 (40 sealed chapters, 996-1350). Each set's files live under data/gold/<set>/ and never
cross: --set, where a command takes it, refuses a file that says it is another set's or that lies under another set's
folder (own_set_path). TEST_SETS are the sets made as a silver standard (every chapter drafted twice by GPT and
adjudicated; see the README), whose annotated chapters must record their provenance (an error, not a warning):
test, test2 and dev2. dev2 follows the test rules rather than the dev rules because it is drafted exactly like the test
(two GPT passes, the user's name questions, the reviewer on the rest) and nothing of it is hand-annotated, so the
stricter rule costs it nothing (every prefilled chapter carries model_double_adjudicated) and catches a chapter changed
on the page without saying who made it; the dev set keeps the lenient rule it has had (a warning on a done chapter
without provenance).
"""
import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.snapshot import REPO_ROOT, load_snapshot

FORMAT = "novel-ner-annotation/1"
# who produced a chapter's annotations (project annotation protocol, 2026-10-01): the formal test is a silver standard,
# GPT annotates each chapter twice and the user decides where the two passes differ
PROVENANCE = ("manual", "manual_model_checked", "model_draft_reviewed", "model_double_adjudicated")
# the chapter sets with answer files under data/gold/<set>/ (see the module docstring)
SETS = ("test", "dev", "dev2", "test2")
# the silver-standard sets: annotated chapters must record their provenance (dev2 included, see the docstring)
TEST_SETS = ("test", "test2", "dev2")
# guide v1.1 E2: where a pending mark is stuck
STUCK = {"qualification": "名称资格", "boundary": "名称边界", "type": "实体类型"}


def own_set_path(path, set_name, root=None):
    """Refuse a file under another set's folder: a path inside data/gold/<other>/ (relative to root, the repository
    by default) stops the run, so nothing of one set is ever read or written as another's. A path outside data/gold/
    is allowed. Returns path."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(Path(root or REPO_ROOT).resolve())
    except ValueError:
        rel = None if p.is_absolute() else p
    parts = rel.parts if rel is not None else ()
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "gold" and parts[2] != set_name:
        raise SystemExit(f"{path} lies under data/gold/{parts[2]}, not the {set_name} set's folder")
    return path


def check(export, snapshot):
    """(errors, warnings, summary) for one exported annotation file."""
    errors, warnings = [], []
    if export.get("format") != FORMAT:
        return [f"format is {export.get('format')!r}, expected {FORMAT!r}"], warnings, {}
    if export.get("snapshot_id") != snapshot.id:
        errors.append(f"snapshot {export.get('snapshot_id')} != {snapshot.id}")
    summary = {"set": export.get("set"), "guide_version": export.get("guide_version"), "chapters": {}}
    for cid, chap in export.get("chapters", {}).items():
        where = cid
        if cid not in snapshot.chapters:
            errors.append(f"{where}: not in the snapshot")
            continue
        source = snapshot.chapters[cid]
        text = source.text
        where = f"ch{source.number}"
        if chap.get("content_hash") != source.content_hash:
            errors.append(f"{where}: content_hash differs from the snapshot (the text changed)")
            continue
        if chap.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            errors.append(f"{where}: text_sha256 does not match the chapter text")
        annotations = sorted(chap.get("annotations", []), key=lambda a: (a["start"], a["end"]))
        previous = None
        for a in annotations:
            label = f"{where} [{a['start']},{a['end']}) {a.get('text')!r}"
            if not (0 <= a["start"] < a["end"] <= len(text)):
                errors.append(f"{label}: offsets out of range")
                continue
            if text[a["start"]:a["end"]] != a.get("text"):
                errors.append(f"{label}: text does not match the chapter at these offsets")
            if "\n" in text[a["start"]:a["end"]]:
                errors.append(f"{label}: crosses a paragraph break")
            if previous and a["start"] < previous["end"]:
                errors.append(f"{label}: overlaps {previous.get('text')!r} (flat annotation)")
            previous = a
            if a.get("pending"):
                if not (a.get("candidates", "").strip() or a.get("doubt", "").strip()):
                    (errors if chap.get("status") == "done" else warnings).append(
                        f"{label}: pending without candidate readings or doubt")
                if any(k not in STUCK for k in a.get("stuck", [])):
                    errors.append(f"{label}: unknown stuck step {a.get('stuck')}")
                elif not a.get("stuck") and export.get("guide_version", "1.0") != "1.0":
                    warnings.append(f"{label}: pending without the step it is stuck at (guide v1.1 E2)")
            elif a.get("type") not in ENTITY_TYPES:
                errors.append(f"{label}: type {a.get('type')!r} is not one of the six")
            for e in a.get("evidence", []):
                other = snapshot.chapters.get(e.get("chapter_id"))
                paragraph = dict(other.paragraphs).get(e.get("paragraph_id")) if other else None
                if paragraph is None or e.get("quote", "") not in paragraph:
                    errors.append(f"{label}: evidence {e.get('chapter_id')}/{e.get('paragraph_id')} not found")
            if a.get("cross_chapter") and not a.get("evidence"):
                warnings.append(f"{label}: marked cross-chapter but has no evidence")
        for u in chap.get("unscored", []):
            label = f"{where} unscored [{u.get('start')},{u.get('end')}) {u.get('text')!r}"
            if not (isinstance(u.get("start"), int) and isinstance(u.get("end"), int) and 0 <= u["start"] < u["end"] <= len(text)):
                errors.append(f"{label}: offsets out of range")
            elif text[u["start"]:u["end"]] != u.get("text"):
                errors.append(f"{label}: text does not match the chapter at these offsets")
            if not u.get("reason", "").strip():
                errors.append(f"{label}: an unscored range must say why")
        if chap.get("status") == "done" and not all(chap.get("checks", {}).values()):
            warnings.append(f"{where}: marked done without all three checks")
        provenance = chap.get("provenance")
        if provenance is not None and provenance not in PROVENANCE:
            errors.append(f"{where}: unknown provenance {provenance!r}")
        elif provenance is None and chap.get("status") == "done":
            warnings.append(f"{where}: done but no provenance recorded")
        if export.get("set") in TEST_SETS and annotations and provenance is None:
            errors.append(f"{where}: {export.get('set')} chapters (a silver standard) must record their provenance")
        types = Counter(a["type"] for a in annotations if not a.get("pending"))
        summary["chapters"][cid] = {
            "number": source.number,
            "status": chap.get("status"),
            "provenance": provenance,
            "minutes": round(chap.get("seconds", 0) / 60, 1),
            "spans": sum(types.values()),
            "by_type": {t: types[t] for t in ENTITY_TYPES},
            "pending": sum(1 for a in annotations if a.get("pending")),
            "unscored": len(chap.get("unscored", [])),
            "cross_chapter": sum(1 for a in annotations if a.get("cross_chapter")),
            "distinct_names": len({a["text"] for a in annotations if not a.get("pending")}),
        }
    done = [c for c in summary["chapters"].values() if c["status"] == "done"]
    summary["done_chapters"] = len(done)
    if done:
        summary["spans_per_done_chapter"] = round(sum(c["spans"] for c in done) / len(done), 1)
    # annotation time only means something for chapters the user annotated by hand
    timed = [c for c in done if c["provenance"] in ("manual", "manual_model_checked")]
    summary["hand_annotated_done"] = len(timed)
    if timed:
        summary["minutes_per_hand_annotated_chapter"] = round(sum(c["minutes"] for c in timed) / len(timed), 1)
    summary["precedents"] = len(export.get("precedents", []))
    return errors, warnings, summary


TYPE_ZH = {"person": "人物", "organization": "组织", "location": "地点", "dungeon": "副本", "item": "物品", "trait": "特性"}


def questions(export, snapshot, chapters=None):
    """Markdown list of pending items and annotations with notes, quoting their paragraph."""
    lines = [
        "# 开发章试标疑难清单",
        "",
        f"规则 v{export.get('guide_version')} · 快照 {export.get('snapshot_id')} · 导出于 {export.get('exported_at')}",
        "",
        "本清单含小说原文片段：只用于和 GPT 讨论规则（开发章例外），不发布、不入库。【】是当前划的范围。",
        "",
        f"请按《中文网络小说名字识别标注规则 v{export.get('guide_version')}》逐条讨论：能用现有规则裁决的，说明依据哪一条、应怎样标；"
        "现有规则说不清的，提出补进 v1.1 的条文；同类情况合并讨论。最后由我决定。",
    ]
    for cid, chap in export["chapters"].items():
        if chapters and cid not in chapters:
            continue
        source = snapshot.chapters[cid]
        items = [a for a in sorted(chap.get("annotations", []), key=lambda a: a["start"])
                 if a.get("pending") or any(a.get(k, "").strip() for k in ("note", "doubt", "candidates"))]
        if not items:
            continue
        lines += ["", f"## 第{source.number}章 {source.title}（{cid}）"]
        starts, offset = [], 0
        for pid, text in source.paragraphs:
            starts.append((offset, pid, text))
            offset += len(text) + 1
        for a in items:
            base, pid, text = next(s for s in reversed(starts) if s[0] <= a["start"])
            s, e = a["start"] - base, a["end"] - base
            label = "待审" if a.get("pending") else TYPE_ZH.get(a.get("type"), a.get("type"))
            lines.append(f"- 「{a['text']}」：{label}（{pid}，第 {a['start']}–{a['end']} 字）")
            lines.append(f"  - 原文：{text[:s]}【{text[s:e]}】{text[e:]}")
            if a.get("stuck"):
                lines.append("  - 卡在：" + "、".join(STUCK[k] for k in a["stuck"] if k in STUCK))
            for key, name in (("candidates", "候选解释"), ("doubt", "疑点"), ("note", "备注")):
                if a.get(key, "").strip():
                    lines.append(f"  - {name}：{a[key].strip()}")
            if a.get("evidence"):
                lines.append("  - 跨章证据：" + "；".join(f"第{ev['chapter_number']}章「{ev['quote']}」" for ev in a["evidence"]))
    return "\n".join(lines) + "\n"


def revise(export, decisions, snapshot, now=None):
    """Apply the user's recorded decisions to a saved answer file; returns (new export, log lines).

    decisions: {"settle_pending": [{"text", "chapters", "type", "expect"}],
                "unscored": [{"text", "reason", "expect"}], "places": [see revise_place],
                "finish_chapters": [chapter ids], "only_chapters": [chapter ids],
                "decided_by", "decided_at", "note"}.
    Every entry states how many places it expects to touch; a different count stops the revision,
    so a decision never lands somewhere the user did not mean. Nothing else changes.
    finish_chapters (the dev answers, review.py pending): chapters whose every pending place the decisions settle;
    each must have no pending mark left, and is then marked done (check "adjudicated" ticked), as the user did on
    the annotation page for the test. only_chapters: the chapters the decisions may change; any other chapter that
    comes out different stops the revision (the dev answers' earlier chapters stay exactly as they are).
    """
    now = now or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    out = json.loads(json.dumps(export))
    log = []
    by = f"{decisions['decided_by']} {decisions['decided_at']}"
    for d in decisions.get("settle_pending", []):
        if d["type"] not in ENTITY_TYPES:
            raise ValueError(f"settle_pending {d['text']}: type {d['type']!r} is not one of the six")
        hits = [a for cid in d["chapters"] for a in out["chapters"][cid]["annotations"]
                if a.get("pending") and a["text"] == d["text"]]
        if len(hits) != d["expect"]:
            raise ValueError(f"settle_pending {d['text']}: expected {d['expect']} pending marks, found {len(hits)}")
        for a in hits:
            a.update({"pending": False, "type": d["type"], "updated_at": now,
                      "note": (a.get("note", "") + f" 【{by} 裁决：{TYPE_ZH[d['type']]}】").strip()})
        log.append(f"settled {len(hits)} pending {d['text']} in {', '.join(d['chapters'])} as {d['type']}")
    for d in decisions.get("unscored", []):
        if not d.get("reason", "").strip():
            raise ValueError(f"unscored {d['text']}: give the reason")
        found = 0
        for cid, chap in out["chapters"].items():
            text = snapshot.chapters[cid].text
            at = text.find(d["text"])
            while at >= 0:
                stop = at + len(d["text"])
                inside = [a["text"] for a in chap["annotations"] if a["start"] < stop and at < a["end"]]
                if inside:
                    log.append(f"note: unscored {d['text']} in {cid} at {at} touches marks {inside}; they are left out of scoring too")
                chap.setdefault("unscored", []).append({"start": at, "end": stop, "text": d["text"], "reason": d["reason"],
                                                        "by": by, "at": now})
                found += 1
                at = text.find(d["text"], stop)
        if found != d["expect"]:
            raise ValueError(f"unscored {d['text']}: expected {d['expect']} occurrences, found {found}")
        log.append(f"unscored {found} occurrences of {d['text']}")
    deciders = Counter()
    for d in decisions.get("places", []):
        log.append(revise_place(out, d, snapshot, by, now))
        deciders[d.get("by", by)] += 1
    keep = decisions.get("only_chapters")
    for cid in decisions.get("finish_chapters", []):
        chap = out["chapters"][cid]
        left = sum(1 for a in chap["annotations"] if a.get("pending"))
        if left:
            raise ValueError(f"finish {cid}: {left} pending mark(s) left; the decisions must settle every pending place")
        chap.update({"status": "done", "checks": {**chap.get("checks", {}), "adjudicated": True}, "done_at": now})
        log.append(f"finished {cid} (no pending marks left)")
    for cid, chap in out["chapters"].items():
        if "unscored" in chap and (keep is None or cid in keep):
            chap["unscored"].sort(key=lambda u: (u["start"], u["end"]))
    if keep is not None:
        unknown = [cid for cid in keep if cid not in out["chapters"]]
        changed = [cid for cid in out["chapters"] if cid not in keep and out["chapters"][cid] != export["chapters"].get(cid)]
        if unknown or changed:
            raise ValueError(f"only_chapters: not in the file {unknown}; changed outside them {changed}")
    summary = "; ".join(f"{n} places decided by {who}" for who, n in deciders.items())
    note = decisions.get("note") or decisions.get("description")
    out.setdefault("revisions", []).append({"at": now, "by": "claude", "what": (
        f"applied recorded decisions ({by}) mechanically" + (f" ({summary})" if summary else "") + ": "
        + "; ".join(x for x in log if not x.startswith("note:")) + (f". {note}" if note else ""))})
    out["updated_at"] = now
    return out, log


DECIDERS = ("user", "sheet", "reviewer")


def revise_place(out, d, snapshot, by, now):
    """One decided place: {"chapter_id", "start", "end", "text", "reading", "reason"[, "by", "decider",
    "occurrence"]}.

    [start, end) is the place as listed for the user (every mark and unscored range there lies inside it).
    reading is one of the six types (the name `text` is marked there), "none" (nothing is marked) or
    "unscored" (the name's range is left out of scoring). decider says who decided it: the user, the user's
    name sheet, or the reviewer model on the user's notes; it is written on the mark and in the chapter's
    "decided_places" record (kept for "none" too, so every place leaves a trace). Everything inside the place is
    replaced; a mark already at exactly the name's position keeps its id, evidence and cross-chapter flag (only
    its type changes), and the cross-chapter evidence of other removed marks moves onto the new mark. A mark or
    unscored range crossing the place's edge, or a name not found exactly once (or at the given occurrence,
    counted from 1 inside the place), stops the revision."""
    cid, start, end, name, reading = d["chapter_id"], d["start"], d["end"], d["text"], d["reading"]
    by = d.get("by", by)  # who decided this place, with the date
    decider = d.get("decider", "user")
    if reading not in ENTITY_TYPES and reading not in ("none", "unscored"):
        raise ValueError(f"place {cid} {start}: reading {reading!r} is not a type, none or unscored")
    if decider not in DECIDERS:
        raise ValueError(f"place {cid} {start}: decider {decider!r} is not one of {DECIDERS}")
    if not d.get("reason", "").strip():
        raise ValueError(f"place {cid} {start}: give the reason")
    chap = out["chapters"][cid]
    text = snapshot.chapters[cid].text
    hits = [i for i in range(start, end - len(name) + 1) if text.startswith(name, i)] if name else []
    k = d.get("occurrence")
    if (k is None and len(hits) != 1) or (k is not None and not 1 <= k <= len(hits)):
        raise ValueError(f"place {cid} {start}-{end}: {name!r} occurs {len(hits)} times there")
    at = hits[0] if k is None else hits[k - 1]
    stop = at + len(name)
    for kind, items in (("mark", chap["annotations"]), ("unscored range", chap.get("unscored", []))):
        for a in items:
            if a["start"] < end and start < a["end"] and not (start <= a["start"] and a["end"] <= end):
                raise ValueError(f"place {cid} {start}-{end}: a {kind} {a.get('text')!r} crosses its edge")
    removed = [a for a in chap["annotations"] if start <= a["start"] and a["end"] <= end]
    chap["annotations"] = [a for a in chap["annotations"] if a not in removed]
    dropped = [u for u in chap.get("unscored", []) if start <= u["start"] and u["end"] <= end]
    if "unscored" in chap:
        chap["unscored"] = [u for u in chap["unscored"] if u not in dropped]
    notes = []
    if reading in ENTITY_TYPES:
        same = next((a for a in removed if (a["start"], a["end"]) == (at, stop)), None)
        mark = dict(same) if same else {"start": at, "end": stop, "text": name, "id": f"rev-{cid}-{at}",
                                        "created_at": now, "guide_version": out.get("guide_version"),
                                        "cross_chapter": False, "evidence": []}
        if same and same.get("type") != reading:
            notes.append(f"type {same.get('type')} -> {reading}")
        for a in removed:  # cross-chapter evidence of a mark this place replaces is kept on the new mark
            if a is not same and a.get("cross_chapter"):
                mark["cross_chapter"] = True
                mark["evidence"] = list(mark.get("evidence") or []) + list(a.get("evidence") or [])
                notes.append(f"kept the cross-chapter evidence of {a['text']}")
        mark.update({"type": reading, "pending": False, "updated_at": now, "candidates": "", "doubt": "", "stuck": [],
                     "decided_by": by, "decider": decider,
                     "note": (mark.get("note", "") + f" 【{by}：{d['reason']}】").strip()})
        mark.setdefault("cross_chapter", False)
        mark.setdefault("evidence", [])
        chap["annotations"].append(mark)
        chap["annotations"].sort(key=lambda a: (a["start"], a["end"]))
    elif reading == "unscored":
        chap.setdefault("unscored", []).append({"start": at, "end": stop, "text": name, "reason": d["reason"],
                                                "by": by, "at": now})
    was = [f"{a['text']}:{a.get('type')}" for a in removed] + [f"{u['text']}:不计分({u.get('by')})" for u in dropped]
    chap.setdefault("decided_places", []).append({"start": at, "end": stop, "text": name, "reading": reading, "by": by,
                                                  "decider": decider, "reason": d["reason"], "was": was, "at": now})
    return (f"place {cid} {at}-{stop} {name}: {'、'.join(was) or 'nothing'} -> {reading} [{decider}: {by}]"
            + (f" ({'; '.join(notes)})" if notes else ""))


def print_report(errors, warnings, summary):
    for cid, c in summary.get("chapters", {}).items():
        if c["status"] == "todo" and not c["spans"] and not c["pending"]:
            continue
        kinds = " ".join(f"{t}={n}" for t, n in c["by_type"].items() if n)
        print(f"ch{c['number']:<4} {c['status']:<5} {c['minutes']:>5} min  spans={c['spans']:<3} pending={c['pending']}"
              + (f" unscored={c['unscored']}" if c["unscored"] else "") +
              f" cross={c['cross_chapter']} distinct={c['distinct_names']}  {kinds}  [{c['provenance'] or 'no provenance'}]")
    if summary.get("done_chapters"):
        print(f"done: {summary['done_chapters']} chapters, {summary['spans_per_done_chapter']} spans per chapter")
    if summary.get("hand_annotated_done"):
        print(f"hand-annotated: {summary['hand_annotated_done']} chapter(s), "
              f"{summary['minutes_per_hand_annotated_chapter']} min per chapter")
    print(f"precedents: {summary.get('precedents', 0)}")
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:", e)
    print("OK" if not errors else f"{len(errors)} error(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "save", "questions", "revise"])
    ap.add_argument("file")
    ap.add_argument("--chapters", nargs="*", help="questions: only these chapter ids")
    ap.add_argument("--decisions", help="revise: the user's decisions (a JSON config)")
    ap.add_argument("--set", dest="set_name", choices=SETS, default=None,
                    help="the set the file belongs to; refused if the file says another set or lies under another set's folder")
    args = ap.parse_args()
    export = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if args.set_name is not None:
        own_set_path(args.file, args.set_name)
        if export.get("set") != args.set_name:
            raise SystemExit(f"{args.file} is a {export.get('set')} export, not {args.set_name}")
    snapshot = load_snapshot(export.get("snapshot_id", ""))
    errors, warnings, summary = check(export, snapshot)
    print_report(errors, warnings, summary)
    if errors:
        raise SystemExit(1)
    if args.action == "questions":
        out_dir = REPO_ROOT / "data" / "gold" / export["set"]
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"questions_{time.strftime('%Y%m%d-%H%M%S')}.md"
        target.write_text(questions(export, snapshot, args.chapters), encoding="utf-8")
        print(f"wrote {target}")
    if args.action == "revise":
        decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
        source = Path(args.file).resolve()
        expected = (REPO_ROOT / decisions["base"]).resolve()
        if source != expected:
            raise SystemExit(f"the decisions are for {decisions['base']}, not {args.file}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != decisions["base_sha256"]:
            raise SystemExit(f"{decisions['base']} changed since the decisions were written down")
        revised, log = revise(export, decisions, snapshot)
        for line in log:
            print(line)
        errors, warnings, summary = check(revised, snapshot)
        print_report(errors, warnings, summary)
        if errors:
            raise SystemExit(1)
        out_dir = REPO_ROOT / "data" / "gold" / export["set"]
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_guide-v{revised['guide_version']}.json"
        if target.exists():
            raise SystemExit(f"{target} exists; not overwriting")
        target.write_text(json.dumps(revised, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {target}")
    if args.action == "save":
        out_dir = REPO_ROOT / "data" / "gold" / export["set"]
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_guide-v{export['guide_version']}.json"
        if target.exists():
            raise SystemExit(f"{target} exists; not overwriting")
        shutil.copyfile(args.file, target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
