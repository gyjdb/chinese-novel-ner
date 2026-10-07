"""Independent quality check of the answers' exclusions (configs/teacher_train300.json "exclusion_qa":
安排独立答案质检核对排除项，不看系统预测和成绩，也不把这些裁决传入训练).

    set PYTHONPATH=src
    rem 1. the packages: every place of the decisions file ruled none (不标) or unscored (不计分), whoever decided it,
    rem    checked against the answers file, grouped by chapter, at most 5 chapters per package (qa_<set>_b01, b02, ...)
    .venv\\Scripts\\python -m novel_ner.qa build --set dev2 --decisions configs/gold_dev2_v1_decisions.json --answers configs/dev2_pending_answers.json --gold data/gold/dev2/20261004-130201_guide-v1.1.json --out data/llm_tasks/qa_dev2
    .venv\\Scripts\\python -m novel_ner.qa build --set test2 --decisions configs/gold_test2_v1_decisions.json --answers configs/test2_pending_answers.json --gold data/gold/test2/20261004-130350_guide-v1.1.json --out data/llm_tasks/qa_test2
    rem 2. one fresh GPT-6.1-Sol request per package through the API (gpt_api's mechanics: the ledger, the 30 USD cap on
    rem    all runs, store=False, a retry on invalid JSON; the reply is kept as pass_qa.json, nothing is imported)
    .venv\\Scripts\\python -m novel_ner.qa run --set dev2 --run-id 20261004-qa-dev2 [--model gpt-6.1-sol --effort xhigh --workers 2]
    .venv\\Scripts\\python -m novel_ner.qa run --set test2 --run-id 20261004-qa-test2
    rem 3. the report for the user (data/gold/<set>/review_qa/qa_report.md, with the novel's text, not committed) and the
    rem    counts (runs/<run id>/metrics.json, config.json with the hashes; no text)
    .venv\\Scripts\\python -m novel_ner.qa report --set dev2 --run-id 20261004-qa-dev2
    .venv\\Scripts\\python -m novel_ner.qa report --set test2 --run-id 20261004-qa-test2

What the checker gets (docs/llm_annotation/qa_exclusions_v1.md): the guide, the chapters' full text (llm_task.chapters_markdown,
paragraphs with ids) and places.md: every excluded place with its chapter id, paragraph id, the exact string, its occurrence
number in the paragraph (counted as llm_task.locate counts, from the place's chapter offsets through evaluate.paragraph_bases,
so a reply locates exactly as evaluate.llm_predictions does), the ruling (不标 or 不计分) and the user's words about the name:
only the verbatim "answer" string of the answers file, linked to the place as review.decide links them (the listing
data/gold/<set>/review_astra/disagreements.json: the place's listed bounds -> group key -> the entry with that name; a place
the listing does not hold is matched by name). It judges by the guide and the chapter text alone: agree / disagree (with the
clause and the type it would give) / cannot_tell, one entry per listed place.

What it never gets: who made the ruling (decider, dates, the reviewer's reasons), any system's prediction or score, any
comparison result. build reads the decisions file, the answers file, the answers config and the listing only; an input path
under runs/ or named like a predictions or metrics file is refused (refuse_system_input), and task.json records that fact.
Sets: dev2 and test2 (the exclusion_qa targets) and test as an explicit option; anything else is refused. The gold file must
lie in its own set's folder (gold.own_set_path) and say that set. test2 packages are allowed: making and checking the answers
is exempt from the seal (configs/test2_seal.json answer_making_exempt lists novel_ner.qa, added in its own commit 566452d);
the checker is not a compared system and sees no system's output.

Effect: none by itself. The report changes no answer file; the user decides, and a change is a new answers version by the
existing flow (review decide -> gold revise; old versions untouched). The QA verdicts enter no training label (teacher labels
come from Opus on training chapters only) and are shown to no system under test.
"""
import argparse
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner import gpt_api
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import paragraph_bases, sha256
from novel_ner.gold import check as check_gold
from novel_ner.gold import own_set_path
from novel_ner.llm_task import GUIDES, chapters_markdown
from novel_ner.score import _context
from novel_ner.snapshot import REPO_ROOT, load_snapshot
from novel_ner.zeroshot import git_state

ALLOWED_SETS = ("dev2", "test2", "test")  # test is allowed as an explicit option, never a default
EXCLUDED = ("none", "unscored")
RULING_ZH = {"none": "不标", "unscored": "不计分"}
VERDICTS = ("agree", "disagree", "cannot_tell")
VERDICT_ZH = {"agree": "同意", "disagree": "不同意", "cannot_tell": "说不准"}
TYPE_ZH = {"person": "人物", "organization": "组织", "location": "地点", "dungeon": "副本", "item": "物品", "trait": "特性"}
QA_VERSION = "qa_v1"
QA_INSTRUCTIONS = REPO_ROOT / "docs" / "llm_annotation" / "qa_exclusions_v1.md"
KIND = "qa_exclusions"
RUN = "qa"  # the pass name: pass_qa.json next to each package
DEFAULT_MODEL = "gpt-6.1-sol"
QA_PROMPT = {"version": "qa_prompt_v1", "request": "请按 instructions.md 核对 places.md 里的每一处，只输出 JSON。",
             "files": ("instructions.md", "guide.md", "chapters.md", "places.md"), "cache_key": "novel-ner-qa", "import": False}
NO_SYSTEM_OUTPUT = ("built from the decisions file, the answers file, the answers config and the review listing only; nothing "
                    "under runs/ and no predictions file was read (refuse_system_input); the package names no decider and no system")
SEAL_NOTE = "answer-making is exempt from the test2 seal (configs/test2_seal.json answer_making_exempt lists novel_ner.qa)"
PLACES_INTRO = (
    "下面是现在的答案**没有标为名字**的地方：裁决为“不标”的，答案认为这里不是名字；裁决为“不计分”的，当时按规则和材料判不了，"
    "这一段对所有系统都不计分。请按 instructions.md，只依据 guide.md 和 chapters.md 的原文，逐处判断这个排除站不站得住。"
    "每处给出章节编号、段落编号、原文字串和它在该段里第几次出现（数法见 instructions.md），并引出本段原文，【】标出这一处。\n\n"
    "“读者的原话”是读过全书的人对这个名字说的话（原话照抄），当作有出处的跨章证据（规则 E1）；原话里对标法的倾向要按规则核对，"
    "不能直接当结论。")
README = """# 排除项质检任务包：{name}

集合 {set}，章节：{chapters}。质检说明 {instructions}，规则 v{guide}，快照 {snapshot}。要核对 {n} 处（不标 {none} 处，不计分 {unscored} 处）。

这个包由 `novel_ner.qa build` 生成，只含：规则（guide.md）、质检说明（instructions.md）、章节原文（chapters.md）、要核对的排除处和读者的原话（places.md）。
不含任何系统的预测或成绩，也不写明每处是谁裁决的。

跑法（通过 API，一包一次全新请求，回复存成 pass_qa.json）：

    .venv\\Scripts\\python -m novel_ner.qa run --set {set} --run-id <id>
    .venv\\Scripts\\python -m novel_ner.qa report --set {set} --run-id <id>

质检结果只写进 data/gold/{set}/review_qa/qa_report.md 和 runs/<id>/（计数）；不改任何答案文件，也不进入训练。
这些文件含小说原文：只在本机和这次请求里用，不要发到别处。
"""


def refuse_system_input(path, root=None):
    """An input of the QA may never be a system's output: a path under runs/ (where every prediction run lives) or a file
    named like a predictions or metrics file is refused, whatever it holds. Returns path."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(Path(root or REPO_ROOT).resolve())
    except ValueError:
        rel = p
    parts = [x.lower() for x in rel.parts]
    if "runs" in parts or any(word in p.name.lower() for word in ("predictions", "metrics")):
        raise SystemExit(f"{path}: the QA reads no system output (nothing under runs/, no predictions or metrics file)")
    return path


def paragraph_at(chapter, start, end):
    """(paragraph id, paragraph text, offset inside it) of a chapter span [start, end), through evaluate.paragraph_bases;
    a span crossing a paragraph break or outside the text is refused."""
    bases = paragraph_bases(chapter)
    for pid, text in chapter.paragraphs:
        base = bases[pid]
        if base <= start < base + len(text):
            if end > base + len(text):
                raise ValueError(f"{chapter.chapter_id} [{start},{end}) crosses a paragraph break")
            return pid, text, start - base
    raise ValueError(f"{chapter.chapter_id} [{start},{end}) is outside the chapter text")


def occurrence_at(paragraph, text, offset):
    """The occurrence number (from 1) of text at offset in the paragraph, counting every earlier start as llm_task.locate
    counts them (overlapping starts each count once), so llm_task.locate(paragraph, text, result) == offset."""
    if not text or not paragraph.startswith(text, offset):
        raise ValueError(f"{text!r} is not at {offset} of the paragraph")
    n, at = 0, -1
    while True:
        at = paragraph.find(text, at + 1)
        n += 1
        if at == offset:
            return n
        if at < 0 or at > offset:  # cannot happen after the check above; kept so a bug never loops
            raise ValueError(f"{text!r} is not at {offset} of the paragraph")


def name_in_place(text, place):
    """[at, stop) of the place's name inside its listed bounds [start, end), found as gold.revise_place finds it: exactly
    once, or at the place's "occurrence" counted inside the bounds."""
    name, start, end = place["text"], place["start"], place["end"]
    hits = [i for i in range(start, end - len(name) + 1) if text.startswith(name, i)] if name else []
    k = place.get("occurrence")
    if (k is None and len(hits) != 1) or (k is not None and not 1 <= k <= len(hits)):
        raise SystemExit(f"{place['chapter_id']} {start}-{end}: {name!r} occurs {len(hits)} times there")
    at = hits[0] if k is None else hits[k - 1]
    return at, at + len(name)


def excluded_places(decisions):
    """Every place of a decisions file ruled none or unscored, whoever decided it (the bulk lists settle_pending and
    unscored of older decisions files are not places and are not checked here; their counts are returned too)."""
    if not isinstance(decisions.get("places"), list):
        raise SystemExit("the decisions file has no 'places' list; only place decisions are checked")
    places = [p for p in decisions["places"] if p.get("reading") in EXCLUDED]
    bulk = {k: len(decisions.get(k, [])) for k in ("settle_pending", "unscored") if decisions.get(k)}
    return places, bulk


def locate_places(export, snapshot, places):
    """The excluded places located in their chapters and checked against the answers file: a none place has no mark and no
    unscored range over the name; an unscored place lies inside a recorded unscored range. Returns (places with the name's
    [start, end), paragraph id, occurrence, ruling and short context; how many the chapter's decided_places record confirms).
    A place the file does not hold as the decisions say stops the build."""
    out, seen, confirmed, problems = [], set(), 0, []
    for p in places:
        cid = p["chapter_id"]
        if cid not in export["chapters"] or cid not in snapshot.chapters:
            raise SystemExit(f"{cid}: not in the answers file or the snapshot")
        chap, ch = export["chapters"][cid], snapshot.chapters[cid]
        at, stop = name_in_place(ch.text, p)
        if (cid, at, stop) in seen:
            raise SystemExit(f"{cid} {at}-{stop} {p['text']!r}: decided twice")
        seen.add((cid, at, stop))
        marks = [a for a in chap["annotations"] if a["start"] < stop and at < a["end"]]
        ranges = [u for u in chap.get("unscored", []) if u["start"] < stop and at < u["end"]]
        label = f"{cid} {at}-{stop} {p['text']!r}"
        if p["reading"] == "none":
            if marks:
                problems.append(f"{label}: ruled none, but the answers file holds a mark there (settled or pending): "
                                f"{[(a.get('text'), a.get('type')) for a in marks]}")
            elif ranges:
                problems.append(f"{label}: ruled none, but the answers file has an unscored range there")
        elif not any(u["start"] <= at and stop <= u["end"] for u in ranges):
            problems.append(f"{label}: ruled unscored, but no unscored range of the answers file covers it")
        if any((d.get("start"), d.get("end"), d.get("reading")) == (at, stop, p["reading"]) for d in chap.get("decided_places", [])):
            confirmed += 1
        pid, paragraph, offset = paragraph_at(ch, at, stop)
        out.append({"chapter_id": cid, "number": ch.number, "start": at, "end": stop, "text": p["text"],
                    "place_start": p["start"], "place_end": p["end"], "paragraph_id": pid,
                    "occurrence": occurrence_at(paragraph, p["text"], offset), "ruling": p["reading"],
                    "context": _context(ch.text, at, stop)})
    if problems:
        raise SystemExit("the answers file does not hold these places as the decisions say:\n" + "\n".join(problems[:10]))
    return out, confirmed


def group_keys(listing):
    """{(chapter_id, start, end): group key} of a review listing: the places of its asked, by_sheet and carried groups, the
    bounds review.decide writes into the decisions file."""
    out = {}
    for part in ("asked", "by_sheet", "carried"):
        for key, ps in (listing.get(part) or {}).items():
            for p in ps:
                out[(p["chapter_id"], p["start"], p["end"])] = key
    return out


def user_words(answers, listing, places):
    """The user's verbatim "answer" string for every place, as review.decide links answers to places: the place's listed
    bounds -> the listing's group key -> the answers entry with that name; a place the listing does not hold (or without a
    listing) takes the entries whose text is the place's string or whose group name holds it. Nothing else of an entry is
    taken: no note2, no prior record, no handling, reading or decider. Sets words and words_from on each place."""
    by_name = {a["name"]: a["answer"] for a in answers["answers"]}
    keys = group_keys(listing) if listing else {}
    for p in places:
        key = keys.get((p["chapter_id"], p["place_start"], p["place_end"]))
        if key is not None and key in by_name:
            p["words"], p["words_from"] = by_name[key], "listing"
            continue
        matches = [a["answer"] for a in answers["answers"]
                   if a.get("text") == p["text"] or p["text"] in a["name"].split(" / ")]
        p["words"] = "；".join(dict.fromkeys(matches))
        p["words_from"] = "name" if matches else None
    return places


def places_markdown(set_name, name, snapshot, places):
    """places.md of one package: the excluded places by chapter and position, each with its chapter id, paragraph id,
    string, occurrence, ruling, the user's words and the paragraph with the place marked. No decider, no system."""
    lines = [f"# 要核对的排除处（{set_name}，{name}）", "", PLACES_INTRO, ""]
    n = 0
    for cid in sorted({p["chapter_id"] for p in places}, key=lambda c: snapshot.chapters[c].number):
        ch = snapshot.chapters[cid]
        lines += [f"## {cid} 第{ch.number}章 {ch.title}", ""]
        for p in sorted((p for p in places if p["chapter_id"] == cid), key=lambda p: p["start"]):
            n += 1
            pid, paragraph, offset = paragraph_at(ch, p["start"], p["end"])
            stop = offset + len(p["text"])
            lines.append(f"### {n}. chapter_id {cid}｜paragraph_id {pid}｜text 「{p['text']}」｜occurrence {p['occurrence']}"
                         f"｜裁决：{RULING_ZH[p['ruling']]}")
            lines.append(f"- 读者的原话：{p['words'] or '（没有）'}")
            lines.append(f"- 本段：{paragraph[:offset]}【{paragraph[offset:stop]}】{paragraph[stop:]}")
            lines.append("")
    return "\n".join(lines)


def public_place(p):
    """A place as task.json keeps it: position, string, paragraph id, occurrence, ruling, the user's words, context."""
    return {k: p[k] for k in ("chapter_id", "start", "end", "text", "place_start", "place_end", "paragraph_id", "occurrence",
                              "ruling", "words", "words_from", "context")}


def build(set_name, decisions_file, answers_file, gold_file, out_dir, listing_file=None, batch_size=5, snapshot=None, root=None):
    """The QA packages of one set under out_dir (qa_<set>_b01, ...): at most batch_size chapters each, every excluded place
    once. Refuses a set outside ALLOWED_SETS, an input under runs/ or named like a system output, a gold file outside the
    set's folder or saying another set, a listing made on another file than the decisions, an answers file that does not
    pass gold.check, a place the answers file does not hold as the decisions say, and an existing package (never
    overwritten). Returns the build summary (counts and hashes, also written to out_dir/build.json)."""
    root = Path(root or REPO_ROOT)
    if set_name not in ALLOWED_SETS:
        raise SystemExit(f"set {set_name!r}: the QA covers {ALLOWED_SETS} (dev2 and test2 by the plan; test on request)")
    for f in (decisions_file, answers_file, gold_file, listing_file):
        if f:
            refuse_system_input(f, root)
    own_set_path(gold_file, set_name, root)
    decisions = json.loads((root / decisions_file).read_text(encoding="utf-8"))
    answers = json.loads((root / answers_file).read_text(encoding="utf-8"))
    export = json.loads((root / gold_file).read_text(encoding="utf-8"))
    if export.get("set") != set_name:
        raise SystemExit(f"{gold_file} says set {export.get('set')!r}, not {set_name!r}")
    listing = json.loads((root / listing_file).read_text(encoding="utf-8")) if listing_file else None
    if listing is not None:
        if listing.get("set") != set_name:
            raise SystemExit(f"{listing_file} says set {listing.get('set')!r}, not {set_name!r}")
        if listing.get("gold_sha256") != decisions.get("base_sha256"):
            raise SystemExit(f"{listing_file} was made on another file than {decisions_file} decided (gold_sha256 != base_sha256)")
    snapshot = snapshot or load_snapshot(export["snapshot_id"])
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"{gold_file} does not pass the gold check: {errors[:3]}")
    guide = GUIDES[export["guide_version"]]
    places, bulk = excluded_places(decisions)
    located, confirmed = locate_places(export, snapshot, places)
    user_words(answers, listing, located)
    chapters = sorted({p["chapter_id"] for p in located}, key=lambda c: snapshot.chapters[c].number)
    names = [f"qa_{set_name}_b{k + 1:02d}" for k in range(math.ceil(len(chapters) / batch_size))]
    out = root / out_dir
    existing = [n for n in names if (out / n).exists()]
    if existing:
        raise SystemExit(f"{existing} exist under {out_dir}; packages are never overwritten (pick another --out)")
    files = {"decisions_file": decisions_file, "decisions_sha256": sha256(root / decisions_file),
             "answers_file": answers_file, "answers_sha256": sha256(root / answers_file),
             "gold_file": gold_file, "gold_sha256": sha256(root / gold_file),
             "listing_file": listing_file, "listing_sha256": sha256(root / listing_file) if listing_file else None}
    commit = git_state()["commit"]
    for k, name in enumerate(names):
        ids = chapters[k * batch_size:(k + 1) * batch_size]
        mine = sorted((p for p in located if p["chapter_id"] in ids), key=lambda p: (p["number"], p["start"]))
        folder = out / name
        folder.mkdir(parents=True)
        shutil.copyfile(QA_INSTRUCTIONS, folder / "instructions.md")
        shutil.copyfile(guide, folder / "guide.md")
        (folder / "chapters.md").write_text(chapters_markdown(snapshot, ids), encoding="utf-8")
        (folder / "places.md").write_text(places_markdown(set_name, name, snapshot, mine), encoding="utf-8")
        rulings = Counter(p["ruling"] for p in mine)
        task = {"name": name, "kind": KIND, "set": set_name, "snapshot_id": snapshot.id, "instructions_version": QA_VERSION,
                "guide_version": export["guide_version"], "chapters": ids, "places": [public_place(p) for p in mine], **files,
                "system_output_read": False, "system_output_note": NO_SYSTEM_OUTPUT,
                "test2_seal": SEAL_NOTE if set_name == "test2" else None,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit": commit}
        (folder / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / "README.md").write_text(README.format(
            name=name, set=set_name, chapters="、".join(f"第{snapshot.chapters[c].number}章" for c in ids),
            instructions=QA_VERSION, guide=export["guide_version"], snapshot=snapshot.id, n=len(mine),
            none=rulings["none"], unscored=rulings["unscored"]), encoding="utf-8")
    summary = {"set": set_name, "kind": KIND, "out_dir": out_dir, "packages": names, "batch_size": batch_size,
               "chapters": len(chapters), "places": len(located), "distinct_strings": len({p["text"] for p in located}),
               "by_ruling": dict(Counter(p["ruling"] for p in located)),
               "words_from": dict(Counter(str(p["words_from"]) for p in located)),
               "confirmed_by_decided_places": confirmed, "bulk_decisions_not_covered": bulk, **files,
               "snapshot_id": snapshot.id, "guide_version": export["guide_version"], "instructions_version": QA_VERSION,
               "system_output_read": False, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit": commit}
    (out / "build.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def packages(set_name, out_dir, root=None):
    """The set's package folders under out_dir, in order; each must be a QA package of that set."""
    out = Path(root or REPO_ROOT) / out_dir
    dirs = sorted(d for d in out.glob(f"qa_{set_name}_b*") if d.is_dir())
    if not dirs:
        raise SystemExit(f"no qa_{set_name}_b* package under {out_dir}; run build first")
    for d in dirs:
        task = json.loads((d / "task.json").read_text(encoding="utf-8"))
        if task.get("kind") != KIND or task.get("set") != set_name:
            raise SystemExit(f"{d.name} is not a {KIND} package of {set_name}")
    return dirs


def run(set_name, run_id, out_dir, model=DEFAULT_MODEL, effort="xhigh", budget=30.0, workers=2, root=None):
    """One fresh request per package through gpt_api.run_many with the QA prompt (its ledger, budget cap and retry; the
    reply is kept as pass_qa.json and not imported as an annotation answer)."""
    if set_name not in ALLOWED_SETS:
        raise SystemExit(f"set {set_name!r}: the QA covers {ALLOWED_SETS}")
    jobs = [(d, RUN) for d in packages(set_name, out_dir, root)]
    return gpt_api.run_many(jobs, model, effort, budget, run_id, workers, prompt=QA_PROMPT)


def validate_reply(task, reply):
    """({place key: entry}, problems) of one reply against its package: every listed place once, no other place, verdicts
    among VERDICTS, would_mark_as null or one of the six types, instructions_version qa_v1. Any problem makes the whole
    reply invalid: it is flagged and none of its verdicts is counted."""
    problems = []
    listed = {(p["chapter_id"], p["paragraph_id"], p["text"], p["occurrence"]) for p in task["places"]}
    entries = reply.get("places") if isinstance(reply, dict) else None
    if not isinstance(entries, list):
        return {}, ["no 'places' list in the reply"]
    if reply.get("instructions_version") != QA_VERSION:
        problems.append(f"instructions_version {reply.get('instructions_version')!r}, expected {QA_VERSION!r}")
    got = {}
    for e in entries:
        if not isinstance(e, dict):
            problems.append("an entry is not an object")
            continue
        key = (e.get("chapter_id"), e.get("paragraph_id"), e.get("text"), e.get("occurrence"))
        if key not in listed:
            problems.append(f"a place that was not listed: {key}")
            continue
        if key in got:
            problems.append(f"listed twice: {key}")
            continue
        if e.get("verdict") not in VERDICTS:
            problems.append(f"{key}: verdict {e.get('verdict')!r} is none of {VERDICTS}")
        if e.get("would_mark_as") is not None and e.get("would_mark_as") not in ENTITY_TYPES:
            problems.append(f"{key}: would_mark_as {e.get('would_mark_as')!r} is neither null nor one of the six")
        got[key] = e
    missing = [k for k in listed if k not in got]
    if missing:
        problems.append(f"{len(missing)} listed place(s) without a verdict, e.g. {sorted(missing)[:3]}")
    return got, problems


def api_record(folder):
    """Model, effort and attempts of the package's request, from the api_qa_attempt*.json files gpt_api keeps."""
    records = [json.loads(f.read_text(encoding="utf-8"))["record"] for f in sorted(folder.glob(f"api_{RUN}_attempt*.json"))]
    if not records:
        return {"model": None, "effort": None, "attempts": 0, "seconds": None}
    return {"model": records[-1].get("model"), "effort": records[-1].get("effort"), "attempts": len(records),
            "seconds": round(sum(r.get("seconds") or 0 for r in records), 1)}


def run_folder(root, run_id):
    """<root>/runs/<id> for "<id>" or "runs/<id>"; a nested or relative id is refused."""
    plain = run_id[5:] if run_id.startswith("runs/") else run_id
    if not plain or "/" in plain or "\\" in plain or plain in (".", ".."):
        raise SystemExit(f"run id {run_id!r} must be one folder name under runs/")
    return Path(root) / "runs" / plain


def cell(x):
    return str("" if x is None else x).replace("|", "｜").replace("\n", " ")


def report_markdown(set_name, run_id, rows, pkgs, m):
    lines = [f"# 排除项质检报告（{set_name}，{run_id}）", "", "含小说原文，只在本机看，不入库。", "", "## 结论", ""]
    lines.append(f"- 核对了 {m['places_listed']} 处（{m['distinct_strings']} 个字符串，{m['chapters']} 章）："
                 + "、".join(f"{RULING_ZH[r]} {n} 处" for r, n in m["by_ruling"].items()))
    lines.append(f"- 质检员（{m['model'] or '模型未记录'}，说明 {QA_VERSION}）只看规则、章节原文和你的原话；"
                 "不知道每处是谁裁决的，也没看任何系统的预测或成绩。")
    v = m["by_verdict"]
    uncounted = ""
    if m["places_not_counted"]:
        uncounted = (f" 无效回复 {m['invalid_replies']} 包、没有回复 {m['packages_without_reply']} 包，"
                     f"共 {m['places_not_counted']} 处没有计入。")
    lines.append(f"- 有效判定 {m['places_judged']} 处：同意 {v.get('agree', 0)}、不同意 {v.get('disagree', 0)}、"
                 f"说不准 {v.get('cannot_tell', 0)}。{uncounted}")
    lines += ["", "| 裁决 | 处数 | 同意 | 不同意 | 说不准 | 未计入 |", "|---|---|---|---|---|---|"]
    for r, n in m["by_ruling"].items():
        rv = m["by_ruling_and_verdict"].get(r, {})
        lines.append(f"| {RULING_ZH[r]} | {n} | {rv.get('agree', 0)} | {rv.get('disagree', 0)} | {rv.get('cannot_tell', 0)} "
                     f"| {n - sum(rv.values())} |")
    if m["would_mark_as"]:
        lines.append("")
        lines.append("- 不同意的地方质检员会标成：" + "、".join(f"{TYPE_ZH.get(t, t)} {n}" for t, n in m["would_mark_as"].items()))
    lines += ["", "这份报告不改任何答案文件；是否采纳由你决定。要改，走现有流程出新版本（review decide → gold revise；旧版本不动）。"
              "这些核对结果不进入任何训练标签，也不给任何被比较的系统看。", "",
              "## 怎么读", "",
              "质检员的判定只是另一个读法，不是新的裁决；下面只说每格的意思，改不改、怎么改都由你定。", "",
              "- 不标 × 同意：质检员也认为这里不是名字。", "- 不标 × 不同意：质检员认为按规则该标，“会标成”给了类型。",
              "- 不标 × 说不准：质检员只凭本章文字判不了。", "- 不计分 × 说不准：和裁决一致。",
              "- 不计分 × 同意：质检员认为这里不是名字。", "- 不计分 × 不同意：质检员认为这里是名字。", ""]
    bad = [p for p in pkgs if p["status"] != "ok"]
    if bad:
        lines += ["## 没有计入的包", ""]
        for p in bad:
            lines.append(f"- {p['name']}（{p['places']} 处）：{'没有回复' if p['status'] == 'no_reply' else '回复无效'}"
                         + ("；".join([""] + p["problems"][:5]) if p.get("problems") else ""))
        lines.append("")
    groups = defaultdict(list)
    for r in rows:
        groups[r["text"]].append(r)

    def order(item):
        rs = item[1]
        c = Counter(r["verdict"] for r in rs)
        return (-c["disagree"], -c["cannot_tell"], -len(rs), item[0])

    lines += ["## 逐串", ""]
    for text, rs in sorted(groups.items(), key=order):
        c = Counter(r["verdict"] for r in rs)
        rulings = "、".join(f"{RULING_ZH[r]} {n}" for r, n in Counter(r["ruling"] for r in rs).items())
        lines.append(f"### {text}（{len(rs)} 处；裁决：{rulings}；判定：同意 {c['agree']}、不同意 {c['disagree']}、"
                     f"说不准 {c['cannot_tell']}" + (f"、未计入 {c[None]}" if c[None] else "") + "）")
        for words in dict.fromkeys(r["words"] for r in rs if r["words"]):
            lines.append(f"- 读者的原话：{words}")
        lines += ["", "| 章 | 段 | 第几次 | 裁决 | 判定 | 依据 | 会标成 | 理由 | 原文 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (r["chapter_id"], r["start"])):
            verdict = VERDICT_ZH.get(r["verdict"], "未计入")
            lines.append(f"| {r['chapter_id']} | {r['paragraph_id']} | {r['occurrence']} | {RULING_ZH[r['ruling']]} | {verdict} "
                         f"| {cell(r['rule'])} | {cell(TYPE_ZH.get(r['would_mark_as'], r['would_mark_as']))} | {cell(r['reason'])} "
                         f"| {cell(r['context'])} |")
        lines.append("")
    return "\n".join(lines)


def report(set_name, run_id, out_dir, root=None, report_dir=None):
    """Reads every package's pass_qa.json, validates it, writes data/gold/<set>/review_qa/qa_report.md (the places with
    the verdicts, grouped by string, a summary first) and runs/<run id>/metrics.json (counts) + config.json (hashes).
    Changes no answer file. Returns the metrics."""
    root = Path(root or REPO_ROOT)
    if set_name not in ALLOWED_SETS:
        raise SystemExit(f"set {set_name!r}: the QA covers {ALLOWED_SETS}")
    rows, pkgs, hashes = [], [], None
    for d in packages(set_name, out_dir, root):
        task = json.loads((d / "task.json").read_text(encoding="utf-8"))
        hashes = hashes or {k: task.get(k) for k in ("decisions_file", "decisions_sha256", "answers_file", "answers_sha256",
                                                      "gold_file", "gold_sha256", "listing_file", "listing_sha256")}
        pass_file = d / f"pass_{RUN}.json"
        info = {"name": d.name, "places": len(task["places"]), "chapters": task["chapters"], "task_sha256": sha256(d / "task.json"),
                "instructions_sha256": sha256(d / "instructions.md"), "guide_sha256": sha256(d / "guide.md"), **api_record(d)}
        if not pass_file.exists():
            pkgs.append({**info, "status": "no_reply", "pass_sha256": None, "problems": []})
            got, valid = {}, False
        else:
            try:
                got, problems = validate_reply(task, json.loads(pass_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError as e:
                got, problems = {}, [f"pass_{RUN}.json is not JSON: {e}"]
            valid = not problems
            pkgs.append({**info, "status": "ok" if valid else "invalid", "pass_sha256": sha256(pass_file), "problems": problems})
        for p in task["places"]:
            e = got.get((p["chapter_id"], p["paragraph_id"], p["text"], p["occurrence"])) if valid else None
            rows.append({**p, "package": d.name, "verdict": e["verdict"] if e else None,
                         "rule": e.get("rule") if e else None, "would_mark_as": e.get("would_mark_as") if e else None,
                         "reason": e.get("reason") if e else None})
    judged = [r for r in rows if r["verdict"] is not None]
    by_rv = defaultdict(Counter)
    for r in judged:
        by_rv[r["ruling"]][r["verdict"]] += 1
    models = sorted({p["model"] for p in pkgs if p["model"]})
    metrics = {
        "run_id": run_id, "set": set_name, "kind": KIND, "instructions_version": QA_VERSION,
        "model": "、".join(models) if models else None,
        "packages": len(pkgs), "packages_replied": sum(p["status"] != "no_reply" for p in pkgs),
        "invalid_replies": sum(p["status"] == "invalid" for p in pkgs),
        "packages_without_reply": sum(p["status"] == "no_reply" for p in pkgs),
        "places_listed": len(rows), "places_judged": len(judged), "places_not_counted": len(rows) - len(judged),
        "chapters": len({r["chapter_id"] for r in rows}), "distinct_strings": len({r["text"] for r in rows}),
        "by_ruling": dict(Counter(r["ruling"] for r in rows)),
        "by_verdict": dict(Counter(r["verdict"] for r in judged)),
        "by_ruling_and_verdict": {k: dict(v) for k, v in by_rv.items()},
        "would_mark_as": dict(Counter(r["would_mark_as"] for r in judged if r["verdict"] == "disagree" and r["would_mark_as"])),
        "by_set": {set_name: {"places_listed": len(rows), "places_judged": len(judged),
                              "by_verdict": dict(Counter(r["verdict"] for r in judged))}},
        "note": "the QA changes no answer file and enters no training label; a change is the user's decision, made as a new "
                "answers version by review decide and gold revise",
    }
    folder = root / (report_dir or f"data/gold/{set_name}/review_qa")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qa_report.md").write_text(report_markdown(set_name, run_id, rows, pkgs, metrics), encoding="utf-8")
    out = run_folder(root, run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    still = {k: (sha256(root / hashes[k.replace("_sha256", "_file")]) == hashes[k]) if hashes.get(k.replace("_sha256", "_file")) else None
             for k in ("decisions_sha256", "answers_sha256", "gold_sha256", "listing_sha256")}
    config = {"run_id": run_id, "set": set_name, "kind": KIND, "instructions_version": QA_VERSION, "prompt": QA_PROMPT,
              "out_dir": out_dir, "report": (folder / "qa_report.md").relative_to(root).as_posix() if folder.is_relative_to(root) else str(folder),
              **hashes, "inputs_unchanged_since_build": still,
              "packages": [{k: p[k] for k in ("name", "status", "places", "chapters", "task_sha256", "pass_sha256",
                                              "instructions_sha256", "guide_sha256", "model", "effort", "attempts", "seconds")}
                           for p in pkgs],
              "system_output_read": False, "system_output_note": NO_SYSTEM_OUTPUT, "code": git_state(),
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="the QA packages of one set")
    b.add_argument("--set", required=True, choices=ALLOWED_SETS, dest="set_name")
    b.add_argument("--decisions", required=True, help="configs/gold_<set>_v1_decisions.json")
    b.add_argument("--answers", required=True, help="configs/<set>_pending_answers.json")
    b.add_argument("--gold", required=True, help="the answers file of the set (data/gold/<set>/<time>_guide-v1.1.json)")
    b.add_argument("--out", help="default data/llm_tasks/qa_<set>")
    b.add_argument("--listing", help="the review listing the decisions came from; default data/gold/<set>/review_astra/disagreements.json when it exists")
    b.add_argument("--batch-size", type=int, default=5)
    r = sub.add_parser("run", help="one GPT request per package (gpt_api mechanics)")
    r.add_argument("--set", required=True, choices=ALLOWED_SETS, dest="set_name")
    r.add_argument("--run-id", required=True)
    r.add_argument("--out", help="default data/llm_tasks/qa_<set>")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--effort", default="xhigh")
    r.add_argument("--budget", type=float, default=30.0, help="cap on the estimated spend of all runs, USD (gpt_api's ledger)")
    r.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("report", help="the report and the counts; changes no answer file")
    p.add_argument("--set", required=True, choices=ALLOWED_SETS, dest="set_name")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", help="default data/llm_tasks/qa_<set>")
    args = ap.parse_args()
    out_dir = args.out or f"data/llm_tasks/qa_{args.set_name}"
    if args.action == "build":
        listing = args.listing
        if listing is None and (REPO_ROOT / f"data/gold/{args.set_name}/review_astra/disagreements.json").exists():
            listing = f"data/gold/{args.set_name}/review_astra/disagreements.json"
        summary = build(args.set_name, args.decisions, args.answers, args.gold, out_dir, listing, args.batch_size)
        print(json.dumps({k: summary[k] for k in ("set", "packages", "chapters", "places", "distinct_strings", "by_ruling",
                                                   "words_from", "confirmed_by_decided_places", "bulk_decisions_not_covered")},
                         ensure_ascii=False, indent=1))
        print(f"wrote {len(summary['packages'])} packages under {out_dir} (no system output read; no answer file changed)")
    elif args.action == "run":
        run(args.set_name, args.run_id, out_dir, args.model, args.effort, args.budget, args.workers)
    else:
        m = report(args.set_name, args.run_id, out_dir)
        print(json.dumps({k: m[k] for k in ("places_listed", "places_judged", "by_verdict", "by_ruling_and_verdict",
                                            "invalid_replies", "packages_without_reply")}, ensure_ascii=False, indent=1))
        print(f"wrote data/gold/{args.set_name}/review_qa/qa_report.md and runs/{args.run_id}/ (no answer file changed)")


if __name__ == "__main__":
    main()
