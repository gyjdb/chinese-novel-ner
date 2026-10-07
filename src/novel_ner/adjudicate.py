"""Merge two GPT passes into a prefilled annotation file the user only has to settle (silver standard).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/test_b01 [--base <latest test export>]
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev_b01 --set dev --base <dev answers or previous prefilled>
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev2_b01 --set dev2            (first package: no base)
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev2_b02 --set dev2 --base data/gold/dev2/incoming/prefilled_dev2_b01.json
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/test2_b01 --set test2          (the same for test2)

Dev answers (the 26 unanswered dev chapters are made the same way as the test answers): --set dev
needs --base, and the packages are chained, each merged onto the previous one's output, so the last file holds
every chapter: dev_b01 onto the current dev answers (data/gold/dev/20261001-215315_guide-v1.1.json: 31 chapters,
5 done), dev_b02 onto data/gold/dev/incoming/prefilled_dev_b01.json, dev_b03 onto prefilled_dev_b02.json, and so on
to dev_b06. A dev package may only hold chapters the base lists (the dev set is fixed); a chapter with any work in
the base (the 5 done ones, or one an earlier package prefilled) is refused, never overwritten, and every other
chapter of the base is carried over as it is. review.py pending then turns what is left pending into questions.

dev2 and test2 (stage 0; NEW_SETS) chain the same way, but nothing of them exists before the first
package: without --base, the base is built here as an empty export of the set in the annotation page's format (every
chapter of configs/annotate_<set>.json, in the page's order, status todo, no annotations; the chapter list and the
snapshot are the config's, checked against the task), and the first package is prefilled onto it. Every later package
takes --base, the previous package's prefilled file; without it, a set whose incoming folder already holds a prefilled
file is refused, so a chain is never restarted by mistake. The set's files stay under data/gold/<set>/: a base under
another set's folder, or one whose "set" is another set's, is refused (gold.own_set_path).

Rules (project annotation protocol, 2026-10-01):
- a span both passes give with the same boundaries and type, overlapping nothing else, is accepted;
- anything else (one pass only, other boundaries or type, pending in either) becomes one pending
  mark over the overlapping group, with each pass's reading as a candidate, for the user to settle;
- the user's name knowledge sheet (data/gold/name_knowledge.tsv) settles pending marks that sit exactly
  on one of its names (as a cross-chapter decision, with evidence from the chapter it names), and turns
  occurrences GPT left unannotated or typed differently into pending marks for the user.

Writes data/gold/<set>/incoming/prefilled_<task>.json in the annotation page's format (for the test: import it in
data/annotate/test.html) and a short report next to it. Neither is committed: they quote the novel.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.gold import SETS, own_set_path
from novel_ner.snapshot import REPO_ROOT, load_snapshot

FORMAT = "novel-ner-annotation/1"
# sets whose base lists every chapter of the set (a package may only hold chapters of the set, and the packages
# chain file to file); the test's packages instead grow the latest page export
FIXED_SETS = ("dev", "dev2", "test2")
# sets with nothing before their first package: its base is an empty export built from configs/annotate_<set>.json
NEW_SETS = ("dev2", "test2")
KNOWLEDGE = REPO_ROOT / "data" / "gold" / "name_knowledge.tsv"
TYPE_NAMES = {"人物": "person", "组织": "organization", "地点": "location", "副本": "dungeon",
              "物品": "item", "道具": "item", "特性": "trait", "不标": "none", "none": "none",
              **{t: t for t in ENTITY_TYPES}}
TYPE_ZH = {"person": "人物", "organization": "组织", "location": "地点", "dungeon": "副本", "item": "物品", "trait": "特性",
           "none": "不标"}
TEXT_RULE = "chapter text = the snapshot's paragraphs joined by \\n; offsets are [start, end) in Unicode code points"


def load_knowledge(path=KNOWLEDGE):
    """Rows of the user's sheet: name, type, evidence chapter number (optional), note (optional)."""
    rows = []
    if not Path(path).exists():
        return rows
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        cells = [c.strip() for c in line.split("\t")]
        if n == 1 or not cells[0] or cells[0].startswith("#"):
            continue
        if len(cells) < 2 or cells[1] not in TYPE_NAMES:
            raise ValueError(f"{path} line {n}: type must be one of {sorted(TYPE_NAMES)}")
        chapter = cells[2] if len(cells) > 2 else ""
        digits = chapter.lower().removeprefix("ch").lstrip("0")
        if chapter and not digits.isdigit():
            raise ValueError(f"{path} line {n}: evidence chapter must be a number like 58, got {chapter!r}")
        if overlaps_itself(cells[0]):
            raise ValueError(f"{path} line {n}: {cells[0]} can overlap itself (like 乐乐 in 乐乐乐), so where it is meant"
                             " cannot be read off the text; leave it out of the sheet and decide it place by place")
        if any(r["name"] == cells[0] for r in rows):
            raise ValueError(f"{path} line {n}: {cells[0]} is listed twice; keep one row (a name used in different"
                             " ways stays out of the sheet and is decided place by place)")
        rows.append({"name": cells[0], "type": TYPE_NAMES[cells[1]],
                     "chapter": int(digits) if chapter else None,
                     "note": cells[3] if len(cells) > 3 else ""})
    return rows


def evidence_for(snapshot, name, chapter_number, here):
    """A quote of the name in the chapter the sheet points to; None if it is not there."""
    chapter = next((c for c in snapshot.chapters.values() if c.number == chapter_number), None)
    if chapter is None or chapter.chapter_id == here:
        return None
    for pid, text in chapter.paragraphs:
        i = text.find(name)
        if i >= 0:
            return {"chapter_id": chapter.chapter_id, "chapter_number": chapter.number, "paragraph_id": pid,
                    "start": i, "end": i + len(name), "quote": text[max(0, i - 12):i + len(name) + 12]}
    return None


def _readings(annotations, text):
    """How the GPT-derived marks read a stretch of text, for a pending mark's candidates."""
    out = []
    for a in annotations:
        if a["pending"]:
            out.append(f"待审【{text[a['start']:a['end']]}】（{a['candidates'] or '无候选'}）")
        else:
            out.append(f"【{text[a['start']:a['end']]}】{TYPE_ZH[a['type']]}")
    return "、".join(out)


def _reading(span, text):
    if span is None:
        return "不标"
    if span["pending"]:
        return f"待审【{text[span['start']:span['end']]}】（{span['candidates'] or '无候选'}）"
    return f"【{text[span['start']:span['end']]}】{TYPE_ZH[span['type']]}"


def merge(text, spans_a, spans_b):
    """Annotations for one chapter: agreed spans accepted, every other overlapping group pending."""
    tagged = sorted([("A", s) for s in spans_a] + [("B", s) for s in spans_b], key=lambda x: (x[1]["start"], x[1]["end"]))
    groups, end = [], -1
    for run, s in tagged:
        if groups and s["start"] < end:
            groups[-1].append((run, s))
            end = max(end, s["end"])
        else:
            groups.append([(run, s)])
            end = s["end"]
    out = []
    for group in groups:
        a = [s for run, s in group if run == "A"]
        b = [s for run, s in group if run == "B"]
        if (len(a) == 1 and len(b) == 1 and not a[0]["pending"] and not b[0]["pending"]
                and (a[0]["start"], a[0]["end"], a[0]["type"]) == (b[0]["start"], b[0]["end"], b[0]["type"])):
            s = a[0]
            out.append({"start": s["start"], "end": s["end"], "type": s["type"], "pending": False, "note": ""})
            continue
        start, stop = min(s["start"] for _, s in group), max(s["end"] for _, s in group)
        stuck = set()
        for _, s in group:
            stuck.update(s.get("stuck", []))
        bounds = {(s["start"], s["end"]) for _, s in group}
        kinds = {s["type"] for _, s in group if not s["pending"]}
        if len(bounds) > 1:
            stuck.add("boundary")
        if len(kinds) > 1:
            stuck.add("type")
        if not a or not b:
            stuck.add("qualification")
        doubts = [f"{run}：{s['doubt']}" for run, s in group if s.get("doubt")]
        candidates = "；".join(f"{run}：" + ("、".join(_reading(s, text) for s in side) if side else "不标")
                               for run, side in (("A", a), ("B", b)))
        out.append({"start": start, "end": stop, "type": None, "pending": True, "candidates": candidates,
                    "doubt": "两遍不一致或待审。" + " ".join(doubts),
                    "stuck": [k for k in ("qualification", "boundary", "type") if k in stuck] or ["qualification"],
                    "note": ""})
    return out


def overlaps_itself(name):
    """True for names like 乐乐 whose occurrences can overlap (乐乐乐 holds two)."""
    return any(name[k:] == name[:len(name) - k] for k in range(1, len(name)))


def occurrences(text, name):
    """Starts of a name, left to right. Sheet names never overlap themselves, so these never overlap."""
    starts, at = [], text.find(name)
    while at >= 0:
        starts.append(at)
        at = text.find(name, at + len(name))
    return starts


def apply_knowledge(annotations, text, knowledge, snapshot, chapter_id):
    """Apply the user's name knowledge sheet to one chapter's merged annotations. Returns a log.

    Rows are taken longest name first (ties by name), so the result does not depend on the sheet's line order
    and a shorter name inside a longer one meets the longer row's decision. Sheet names never overlap themselves
    (load_knowledge refuses them), so their occurrences are disjoint. For each occurrence:
    - GPT already agrees (one span, same boundaries and type): nothing changes;
    - a longer confident GPT span contains the name (e.g. a full name holding a 不标 word): left alone;
    - nothing annotated there: a pending mark for the user (or nothing, if the sheet says 不标);
    - GPT's readings lie inside the name and none contradicts the sheet (pending marks, or confident
      spans of the same type): replaced by the sheet's reading over the whole name -- this settles
      pending marks and widens a confident span to the boundary the user decided (e.g. 冬 -> 冬先生);
      a 不标 row removes pending marks;
    - GPT confidently read something the sheet contradicts, or GPT's span crosses the name's edge:
      one pending mark over the whole stretch, listing GPT's readings and the sheet's.
    Marks the sheet itself made, including a hidden placeholder for every 不标 decision, are never taken for
    GPT's readings: an occurrence inside a name another row already decided is left alone (the user decided
    the longer name); one that touches another row's decision in any other way ends up as a pending mark
    listing both rows and the GPT readings involved.
    A row with an evidence chapter (another chapter) is whole-book knowledge: what it settles is marked
    cross-chapter, with a quote from that chapter. A row without one is the user's reading of the name here.
    """
    log = []
    number = snapshot.chapters[chapter_id].number

    def placeholder(start, stop, name, removed):
        return {"start": start, "end": stop, "type": "none", "pending": False, "_sheet": True, "_none": True,
                "_name": name, "_gpt": removed, "candidates": "", "doubt": "", "stuck": []}

    def pending_over(group, start, stop, candidates, stuck, doubt):
        steps = set(stuck)
        doubts = [doubt]
        for a in group:
            a["_drop"] = True
            if a["pending"]:
                steps.update(a.get("stuck", []))
                if a.get("doubt"):
                    doubts.append(a["doubt"])
        annotations.append({"start": start, "end": stop, "type": None, "pending": True, "candidates": candidates,
                            "doubt": " ".join(doubts), "note": "", "_sheet": True,
                            "stuck": [k for k in ("qualification", "boundary", "type") if k in steps]})

    for row in sorted(knowledge, key=lambda r: (-len(r["name"]), r["name"])):
        name, kind = row["name"], row["type"]
        sheet_reading = f"知识表：【{name}】{TYPE_ZH[kind]}"
        if overlaps_itself(name):
            raise ValueError(f"{name} can overlap itself; the sheet must not contain it")
        for at in occurrences(text, name):
            stop = at + len(name)
            touching = [a for a in annotations if not a.get("_drop") and a["start"] < stop and at < a["end"]]
            made = [a for a in touching if a.get("_sheet")]
            gpt = [a for a in touching if not a.get("_sheet")]
            accepted = [a for a in gpt if not a["pending"]]
            cover = next((a for a in made if a["start"] <= at and stop <= a["end"]), None)
            if cover is not None:
                if cover["pending"] and sheet_reading not in cover["candidates"]:
                    cover["candidates"] += f"；{sheet_reading}"
                    log.append(("sheet_conflict", at, stop, name, kind, None))
            elif made:
                removed = [g for a in made for g in a.get("_gpt", [])]  # GPT marks a 不标 decision took away
                group = touching + removed
                lo = min([at] + [a["start"] for a in group])
                hi = max([stop] + [a["end"] for a in group])
                parts = [f"GPT 两遍：{_readings(gpt + removed, text)}"] if gpt or removed else []
                parts += [a["candidates"] if a["pending"] else f"知识表：{_readings([a], text)}" for a in made]
                parts.append(sheet_reading)
                pending_over(group, lo, hi, "；".join(parts), ["boundary"],
                             "知识表的裁决在这里互相交叉（可能是两个名字，也可能是同一个名字的两处），看这一处该按哪个。")
                log.append(("sheet_conflict", lo, hi, name, kind, None))
            elif not gpt:
                if kind == "none":
                    annotations.append(placeholder(at, stop, name, []))
                else:
                    pending_over([], at, stop, f"{sheet_reading}；GPT 两遍：不标", ["qualification"],
                                 "名字知识表里有这个名字，GPT 两遍都没标；看这一处是不是这个名字的用法。")
                    log.append(("unannotated", at, stop, name, kind, None))
            elif (len(gpt) == 1 and (gpt[0]["start"], gpt[0]["end"]) == (at, stop)
                  and accepted and accepted[0]["type"] == kind):
                pass  # GPT already agrees with the sheet
            elif (len(gpt) == 1 and accepted and accepted[0]["start"] <= at and stop <= accepted[0]["end"]
                  and (accepted[0]["start"], accepted[0]["end"]) != (at, stop)):
                log.append(("contained", at, stop, name, kind, None))  # a longer name GPT was sure of: keep it
            elif (all(at <= a["start"] and a["end"] <= stop for a in gpt) and all(a["type"] == kind for a in accepted)
                  and not (kind == "none" and accepted)):
                widened = bool(accepted)
                for a in gpt:
                    a["_drop"] = True
                if kind == "none":
                    annotations.append(placeholder(at, stop, name, gpt))
                else:
                    ev = evidence_for(snapshot, name, row["chapter"], chapter_id) if row["chapter"] else None
                    elsewhere = bool(row["chapter"]) and row["chapter"] != number
                    annotations.append({"start": at, "end": stop, "type": kind, "pending": False, "_sheet": True, "_gpt": gpt,
                                        "cross_chapter": elsewhere, "evidence": [ev] if ev else [],
                                        "note": f"按名字知识表裁决（{TYPE_ZH[kind]}）。{row['note']}".strip(),
                                        "candidates": "", "doubt": "", "stuck": []})
                    log.append(("extended" if widened else "settled", at, stop, name, kind,
                                "（证据章里没找到这个名字，证据待补）" if elsewhere and not ev else ""))
            else:
                lo = min([at] + [a["start"] for a in gpt])
                hi = max([stop] + [a["end"] for a in gpt])
                inside = (lo, hi) == (at, stop)
                stuck = (["qualification"] if kind == "none" else ["type"]) if inside else ["boundary"]
                if inside and any((a["start"], a["end"]) != (at, stop) for a in gpt):
                    stuck.append("boundary")
                doubt = ("名字知识表和 GPT 两遍的标法不同，看这一处是哪种用法。" if inside
                         else "名字知识表的名字和 GPT 标的范围交叉，看这一处该怎么框。")
                pending_over(gpt, lo, hi, f"GPT 两遍：{_readings(gpt, text)}；{sheet_reading}", stuck, doubt)
                log.append(("type_conflict" if inside else "overlap", lo, hi, name, kind, None))
    for a in annotations:  # a 不标 decision that still stands and took GPT marks away is reported as skipped
        if a.get("_none") and not a.get("_drop") and a["_gpt"]:
            log.append(("skipped", a["start"], a["end"], a["_name"], "none", ""))
    annotations[:] = [a for a in annotations if not a.get("_drop") and not a.get("_none")]
    for a in annotations:
        for key in ("_sheet", "_gpt", "_name"):
            a.pop(key, None)
    annotations.sort(key=lambda a: (a["start"], a["end"]))
    return log


def empty_export(set_name, snapshot, chapters, guide_version, now, source):
    """An export of set_name in the annotation page's format with every chapter untouched, as the page itself starts
    (status todo, no provenance, no annotations): the base of a new set's first prefill. chapters: the set's chapter
    ids in the page's order; source: where the list came from, for the revision record."""
    if not chapters or len(set(chapters)) != len(chapters):
        raise SystemExit(f"the {set_name} set's chapter list is empty or repeats a chapter")
    missing = [cid for cid in chapters if cid not in snapshot.chapters]
    if missing:
        raise SystemExit(f"{missing} are not in snapshot {snapshot.id}")
    return {
        "format": FORMAT, "set": set_name, "snapshot_id": snapshot.id, "guide_version": guide_version,
        "text_rule": TEXT_RULE, "page_generated_at": None, "created_at": now, "updated_at": None, "exported_at": None,
        "current": chapters[0], "next_id": 1,
        "chapters": {cid: {"content_hash": snapshot.chapters[cid].content_hash,
                           "text_sha256": hashlib.sha256(snapshot.chapters[cid].text.encode("utf-8")).hexdigest(),
                           "status": "todo", "provenance": None,
                           "checks": {"occurrences": False, "full_read": False, "records": False},
                           "seconds": 0, "annotations": [], "skipped": [], "note": ""} for cid in chapters},
        "precedents": [],
        "revisions": [{"at": now, "by": "claude",
                       "what": f"empty {set_name} export: the {len(chapters)} chapters of {source} in that order, "
                               "no annotations; the base of the set's first prefill"}],
    }


def set_config(set_name, path=None):
    """(config, chapter order, path) of the set's annotation page config, configs/annotate_<set>.json by default."""
    path = Path(path) if path else REPO_ROOT / "configs" / f"annotate_{set_name}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: the {set_name} set's annotation page config (configs/annotate_<set>.json)")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("set") != set_name:
        raise SystemExit(f"{path} is the annotation config of {cfg.get('set')!r}, not {set_name}")
    return cfg, chapter_order(cfg), path


def prefill(task_dir, base=None, knowledge_path=KNOWLEDGE, out_dir=None, set_name="test", set_config_path=None):
    """Merge one package's two passes into the base, an export of set_name (without a base: a new test file, or for
    a set of NEW_SETS the empty export of its chapters built from its annotation config, set_config_path)."""
    if set_name not in SETS:
        raise ValueError(f"set must be one of {SETS}, not {set_name!r}")
    if base is None and set_name == "dev":
        raise SystemExit(f"a {set_name} prefill needs a base: the current {set_name} answers file for the first package, "
                         "then the previous package's prefilled file")
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    drafts = {run: json.loads((task_dir / f"draft_{run}.json").read_text(encoding="utf-8")) for run in ("A", "B")}
    snapshot = load_snapshot(task["snapshot_id"], verify=False)
    knowledge = load_knowledge(knowledge_path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())  # the page's timestamp format (UTC)
    for run, d in drafts.items():
        extra = set(d["chapters"]) - set(task["chapters"])
        if extra:
            raise SystemExit(f"pass {run} answered for chapters outside the task: {sorted(extra)}; look at it first")
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "data" / "gold" / set_name / "incoming"
    if base is None and set_name in NEW_SETS:
        earlier = sorted(out_dir.glob("prefilled_*.json")) if out_dir.exists() else []
        if earlier:
            raise SystemExit(f"{out_dir} already holds {earlier[-1].name}; a later {set_name} package chains onto the "
                             "previous package's prefilled file (--base)")
        cfg, order, cfg_path = set_config(set_name, set_config_path)
        if cfg.get("snapshot") != task["snapshot_id"] or str(cfg.get("guide_version")) != str(task["guide_version"]):
            raise SystemExit(f"{cfg_path}: snapshot {cfg.get('snapshot')} / guide {cfg.get('guide_version')} differ from "
                             f"the task's {task['snapshot_id']} / {task['guide_version']}")
        try:
            source = cfg_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            source = cfg_path.name
        base = empty_export(set_name, snapshot, order, task["guide_version"], now, source)
    state = base or {
        "format": FORMAT, "set": set_name, "snapshot_id": task["snapshot_id"], "guide_version": task["guide_version"],
        "text_rule": TEXT_RULE, "page_generated_at": None, "created_at": now, "updated_at": now, "exported_at": None,
        "current": task["chapters"][0], "next_id": 1, "chapters": {}, "precedents": [], "revisions": [],
    }
    if state.get("set") != set_name or state.get("snapshot_id") != task["snapshot_id"]:
        raise SystemExit(f"the base file is not a {set_name} export of this snapshot")
    if set_name in FIXED_SETS:  # the set is fixed: the base lists all its chapters, with the snapshot's text
        missing = [cid for cid in task["chapters"] if cid not in state["chapters"]]
        if missing:
            raise SystemExit(f"{missing} are not chapters of the {set_name} set (not in the base file)")
        changed = [cid for cid in task["chapters"]
                   if state["chapters"][cid].get("content_hash") != snapshot.chapters[cid].content_hash]
        if changed:
            raise SystemExit(f"{changed}: the base file was made on other text than the snapshot")
    report = [f"# 预填结果：{task['name']}", "", "含小说原文，不入库。", ""]
    for cid in task["chapters"]:
        old = state["chapters"].get(cid)
        if old and (old.get("annotations") or old.get("skipped") or old.get("note") or old.get("seconds")
                    or old.get("provenance") or any(old.get("checks", {}).values()) or old.get("status") != "todo"):
            raise SystemExit(f"{cid} already has work in the base file; not overwriting it")
        for run, d in drafts.items():
            if d["chapters"][cid]["errors"]:
                raise SystemExit(f"pass {run} has unlocated mentions in {cid}; look at them before merging")
        chapter = snapshot.chapters[cid]
        text = chapter.text
        anns = merge(text, drafts["A"]["chapters"][cid]["spans"], drafts["B"]["chapters"][cid]["spans"])
        log = apply_knowledge(anns, text, knowledge, snapshot, cid)
        skipped = [{"start": s, "end": e, "text": text[s:e], "at": now, "by": "name knowledge sheet"}
                   for kind_, s, e, *_ in log if kind_ == "skipped"]
        for a in anns:
            a.update({"id": f"a{state['next_id']}", "text": text[a["start"]:a["end"]], "guide_version": task["guide_version"],
                      "created_at": now, "updated_at": now})
            for key, empty in (("candidates", ""), ("doubt", ""), ("stuck", []), ("cross_chapter", False), ("evidence", []), ("note", "")):
                a.setdefault(key, empty)
            state["next_id"] += 1
        pending = [a for a in anns if a["pending"]]
        state["chapters"][cid] = {
            "content_hash": chapter.content_hash, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "status": "doing", "provenance": "model_double_adjudicated", "checks": {"adjudicated": False},
            "seconds": 0, "annotations": anns, "skipped": skipped,
            "note": f"GPT 两遍合并（{task['name']}）：采用 {len(anns) - len(pending)} 处，待你裁决 {len(pending)} 处。",
        }
        report.append(f"## 第{chapter.number}章 {chapter.title}（{cid}）：采用 {len(anns) - len(pending)} 处，待裁决 {len(pending)} 处")
        for kind_, s, e, name, typ, has_ev in log:
            what = {"settled": f"按知识表定为{TYPE_ZH[typ]}{has_ev or ''}",
                    "skipped": "按知识表定为不标（去掉待审）",
                    "extended": f"按知识表把 GPT 一致标的范围扩成整串，{TYPE_ZH[typ]}{has_ev or ''}",
                    "type_conflict": f"知识表说{TYPE_ZH[typ]}，GPT 两遍标的不一样 → 待审",
                    "unannotated": f"知识表说{TYPE_ZH[typ]}，GPT 没标 → 待审",
                    "overlap": "知识表的名字和 GPT 的范围交叉 → 待审",
                    "contained": "知识表的名字在 GPT 一致标的更长名字里，保留 GPT 的标法",
                    "sheet_conflict": "知识表的裁决在这里交叉 → 待审"}[kind_]
            left, right = text[max(0, s - 12):s].split("\n")[-1], text[e:e + 12].split("\n")[0]
            report.append(f"- {what}：{left}【{text[s:e]}】{right}")
        report.append("")
    state["updated_at"] = now
    state.setdefault("llm_drafts", {})[task["name"]] = {
        run: {"model": d["model"], "source_file": d["source_file"], "imported_at": d["imported_at"],
              "instructions_version": d["instructions_version"]} for run, d in drafts.items()}
    state.setdefault("revisions", []).append({
        "at": now, "by": "claude",
        "what": f"prefilled {', '.join(task['chapters'])} from two GPT passes ({task['name']}) and the user's name knowledge sheet "
                f"({len(knowledge)} rows); agreed spans accepted, the rest left pending for the user"})
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"prefilled_{task['name']}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; not overwriting")
    target.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"prefilled_{task['name']}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return target, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--base", help="the latest export of the set, so earlier batches are kept (dev: required; the dev "
                                   "answers file for dev_b01, then the previous package's prefilled file; dev2, test2: "
                                   "left out for the first package, then the previous package's prefilled file)")
    ap.add_argument("--knowledge", default=str(KNOWLEDGE))
    ap.add_argument("--set", dest="set_name", choices=SETS, default="test")
    ap.add_argument("--set-config", default=None,
                    help="dev2, test2 without --base: the set's annotation page config (default configs/annotate_<set>.json)")
    args = ap.parse_args()
    base = json.loads(Path(own_set_path(args.base, args.set_name, REPO_ROOT)).read_text(encoding="utf-8")) if args.base else None
    target, report = prefill(args.task, base, args.knowledge, set_name=args.set_name, set_config_path=args.set_config)
    print("\n".join(line for line in report if line.startswith("## ")))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
