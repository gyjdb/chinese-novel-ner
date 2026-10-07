"""Where the reviewer (GPT-6-Astra) and the current test answers disagree, grouped by name into plain questions
for the user (models label, the user supplies knowledge of the book; Claude only organizes).
The same steps finish the dev answers (pending, below).

Runbook, dev answers (the 26 unanswered dev chapters made like the test answers; see the README):

    set PYTHONPATH=src
    rem 1. merge each package's two GPT-6.1-Sol passes, chained: dev_b01 onto the dev answers, then each onto the last
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev_b01 --set dev --base data/gold/dev/20261001-215315_guide-v1.1.json
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev_b02 --set dev --base data/gold/dev/incoming/prefilled_dev_b01.json
    rem    ... dev_b03 to dev_b06 the same way; then check the last file against the passes and the name sheet
    .venv\\Scripts\\python -m novel_ner.verify_prefill data/gold/dev/incoming/prefilled_dev_b06.json
    rem 2. what is still pending, as plain questions per name: data/gold/dev/review_astra/questions.md, answer_sheet.md
    .venv\\Scripts\\python -m novel_ner.review pending --set dev --prefilled data/gold/dev/incoming/prefilled_dev_b06.json --run-id <date>-dev-pending
    rem 3. the user's answers, verbatim, into configs/dev_pending_answers.json ("no" and "name" as listed; handling direct
    rem    or reviewer by the answer's wording, like configs/review_astra_answers.json; a direct answer naming a type on a
    rem    group with several strings, a name-sheet reading or a boundary doubt says which string in "text"); Claude
    rem    decides no label
    rem 4. the hedged answers' places, and the name-sheet names' places, for GPT-6-Astra (skip 4 if nothing goes there)
    .venv\\Scripts\\python -m novel_ner.review package --set dev --gold data/gold/dev/incoming/prefilled_dev_b06.json --answers configs/dev_pending_answers.json --out data/llm_tasks/adjudicate_dev
    .venv\\Scripts\\python -m novel_ner.codex_run run data/llm_tasks/adjudicate_dev --kind adjudicate --label astra --run-id <date>-adjudicate-dev --role "adjudicator of the dev answers' pending places (not a compared system)"
    rem 5. the decisions file, then the new dev answers version (data/gold/dev/<time>_guide-v1.1.json; old files unchanged)
    .venv\\Scripts\\python -m novel_ner.review decide --set dev --gold data/gold/dev/incoming/prefilled_dev_b06.json --answers configs/dev_pending_answers.json --adjudication data/llm_tasks/adjudicate_dev --out configs/gold_dev_v2_decisions.json
    .venv\\Scripts\\python -m novel_ner.gold revise data/gold/dev/incoming/prefilled_dev_b06.json --decisions configs/gold_dev_v2_decisions.json

In the new version every place that was pending is marked, removed or unscored, and records who decided it (user:
a direct answer; reviewer: Astra on the user's words, or on the name sheet's row for a sheet name; sheet: only with
--sheet-groups sheet); the prefilled chapters are marked done; the 5 earlier dev chapters stay exactly as they were.

Runbook, dev2 answers (stage 0: the 25 decision chapters of configs/annotate_dev2.json,
packages dev2_b01..b05, made like the test; nothing of dev2 exists before the first package):

    set PYTHONPATH=src
    rem 1. merge each package's two GPT-6.1-Sol passes, chained: dev2_b01 onto an empty dev2 export that adjudicate
    rem    builds from configs/annotate_dev2.json (no --base), then each package onto the previous prefilled file
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev2_b01 --set dev2
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/dev2_b02 --set dev2 --base data/gold/dev2/incoming/prefilled_dev2_b01.json
    rem    ... dev2_b03 to dev2_b05 the same way; then check the last file against the passes and the name sheet
    .venv\\Scripts\\python -m novel_ner.verify_prefill data/gold/dev2/incoming/prefilled_dev2_b05.json --set dev2
    rem 2. what is pending, as plain questions per name: data/gold/dev2/review_astra/questions.md, answer_sheet.md
    .venv\\Scripts\\python -m novel_ner.review pending --set dev2 --prefilled data/gold/dev2/incoming/prefilled_dev2_b05.json --run-id 20261004-dev2-pending
    rem 3. the user's answers, verbatim, into configs/dev2_pending_answers.json (exactly as step 3 of the dev runbook)
    rem 4. the hedged answers' places and the name-sheet names' places, for GPT-6-Astra (skip 4 if nothing goes there)
    .venv\\Scripts\\python -m novel_ner.review package --set dev2 --gold data/gold/dev2/incoming/prefilled_dev2_b05.json --answers configs/dev2_pending_answers.json --out data/llm_tasks/adjudicate_dev2
    .venv\\Scripts\\python -m novel_ner.codex_run run data/llm_tasks/adjudicate_dev2 --kind adjudicate --label astra --run-id 20261004-adjudicate-dev2 --role "adjudicator of the dev2 answers' pending places (not a compared system)"
    rem 5. the decisions file, then the first dev2 answers version (data/gold/dev2/<time>_guide-v1.1.json)
    .venv\\Scripts\\python -m novel_ner.review decide --set dev2 --gold data/gold/dev2/incoming/prefilled_dev2_b05.json --answers configs/dev2_pending_answers.json --adjudication data/llm_tasks/adjudicate_dev2 --out configs/gold_dev2_v1_decisions.json
    .venv\\Scripts\\python -m novel_ner.gold revise data/gold/dev2/incoming/prefilled_dev2_b05.json --decisions configs/gold_dev2_v1_decisions.json --set dev2

Runbook, test2 answers (the 40 sealed chapters of configs/annotate_test2.json, snapshot to1350, packages
test2_b01..b08): the five steps of the dev2 runbook with dev2 replaced by test2 -- the last prefilled file is
prefilled_test2_b08.json, the run ids 20261004-test2-pending and 20261004-adjudicate-test2, the answers
configs/test2_pending_answers.json, the package data/llm_tasks/adjudicate_test2, the decisions
configs/gold_test2_v1_decisions.json, the answers file data/gold/test2/<time>_guide-v1.1.json:

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/test2_b01 --set test2
    .venv\\Scripts\\python -m novel_ner.adjudicate data/llm_tasks/test2_b02 --set test2 --base data/gold/test2/incoming/prefilled_test2_b01.json
    rem    ... test2_b03 to test2_b08 the same way
    .venv\\Scripts\\python -m novel_ner.verify_prefill data/gold/test2/incoming/prefilled_test2_b08.json --set test2
    .venv\\Scripts\\python -m novel_ner.review pending --set test2 --prefilled data/gold/test2/incoming/prefilled_test2_b08.json --run-id 20261004-test2-pending
    rem    the user's answers into configs/test2_pending_answers.json, then (if any group goes to the reviewer)
    .venv\\Scripts\\python -m novel_ner.review package --set test2 --gold data/gold/test2/incoming/prefilled_test2_b08.json --answers configs/test2_pending_answers.json --out data/llm_tasks/adjudicate_test2
    .venv\\Scripts\\python -m novel_ner.codex_run run data/llm_tasks/adjudicate_test2 --kind adjudicate --label astra --run-id 20261004-adjudicate-test2 --role "adjudicator of the test2 answers' pending places (not a compared system)"
    .venv\\Scripts\\python -m novel_ner.review decide --set test2 --gold data/gold/test2/incoming/prefilled_test2_b08.json --answers configs/test2_pending_answers.json --adjudication data/llm_tasks/adjudicate_test2 --out configs/gold_test2_v1_decisions.json
    .venv\\Scripts\\python -m novel_ner.gold revise data/gold/test2/incoming/prefilled_test2_b08.json --decisions configs/gold_test2_v1_decisions.json --set test2

The test2 seal (configs/test2_seal.json, the material-scope rules) keeps every system under test away from these
chapters; making the answers is exempt: the GPT drafting passes and the adjudication of pending places by GPT-6-Astra
(codex_run --kind adjudicate; the adjudicator is not a compared system and sees no system's output), so the test2
answers are made exactly like the test's and dev2's. Annotation passes (codex_run --kind annotate, claude_run,
tagger predict) on test2 chapters are refused until the seal is opened in stage 3.

dev2 and test2 have no earlier answers, so they take the pending route only (diff is refused for them), and every
file of a set stays under data/gold/<set>/ (gold.own_set_path: a file under another set's folder is refused).

Runbook, answer memory (the dev2 and test2 sheets asked 22 of 83 name groups the user had already
answered, because "review pending" consulted only the name sheet; earlier answers are reused and the new answers are
cross-checked against them; nothing here judges an answer -- every rule is mechanical and the user decides conflicts):

    set PYTHONPATH=src
    rem a. the memory, configs/answer_memory.json: one entry per exact string from the earlier answer files (verbatim),
    rem    the decisions files (rulings per reading and decider) and the name sheet (a row counts as a user ruling);
    rem    carry_over = the one reading when every ruling of the string agrees, else null. Later files are appended:
    rem    --answers configs/dev2_pending_answers.json --decisions configs/gold_dev2_v1_decisions.json ([set=]path)
    .venv\\Scripts\\python -m novel_ner.review memory
    rem b. a filled sheet against the memory: data/gold/<set>/review_astra/answer_check.md (no, name, 以前的回答,
    rem    以前的裁决, 这次的回答, 判定 = consistent / conflict / repeat_without_type / new / unclear, by the type words
    rem    人物 组织 地点 副本 物品 道具 特性 不标 口令 啥也不是 不计分), counts in runs/<run-id>/; it changes no answer
    .venv\\Scripts\\python -m novel_ner.review check-answers --set dev2 --sheet data/gold/dev2/review_astra/answer_sheet.md --listing data/gold/dev2/review_astra/disagreements.json --memory configs/answer_memory.json --run-id 20261004-dev2-answer-check
    rem c. a listing made with the memory: a group whose every string has a carry_over is not asked -- it is listed
    rem    under 已答过 (rows P1, P2, ...; the sheet's cell pre-filled 已答：<type> （<deciders and dates>）), and a group
    rem    with earlier answers but no carry_over is asked with them quoted (你以前答过). Without --memory nothing changes.
    .venv\\Scripts\\python -m novel_ner.review pending --set <set> --prefilled <file> --run-id <id> --memory configs/answer_memory.json
    rem d. in the answers file, a group kept as before takes a prior entry: {"no", "name", "places", "answer": the user's
    rem    words this time (or the pre-filled cell), "handling": "prior", "reading": the carry_over, "prior":
    rem    {"carry_over_from": ..., "answers": [...]}} ("text" as for a direct answer when the group has several
    rem    strings; left out, it is the carried string). package skips it; decide checks it against the listing's carry
    rem    (a listing made with --memory) or else against the memory (--memory, default configs/answer_memory.json:
    rem    a sheet made without one, like the dev2 and test2 sheets of 2026-10-04) -- the reading, the string and
    rem    carry_over_from must be the carry-over's -- and applies it with the carried ruling's own decider (user,
    rem    sheet or reviewer: what gold.py revise accepts and evaluate.gold_origin counts by; one decider when the
    rem    rulings agree on it, reviewer when a reviewer ruling is among them, user when user and sheet mix) and
    rem    by "prior ruling: <carry_over_from>".

Test answers:

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.review diff --gold data/gold/test/<answers>.json --label astra --run-id <id>

For every test chapter the answers' settled spans, their unscored regions, and the reviewer's spans and pending
marks are put together; spans that touch one another by a character form one place. A place where the answers
and the reviewer hold exactly the same spans (and nothing is pending or unscored there) is agreed; any other
place is listed with what kind of difference it is. Places are grouped by the names written there, so the user
answers once per name. A place whose names are all in the name sheet (data/gold/name_knowledge.tsv) is listed
under the sheet instead of asked again. Nothing here changes any answer file; it only writes the list.

Outputs: data/gold/test/review_<label>/questions.md (with short quotes of the novel, not committed),
disagreements.json (the places, for applying the user's answers later, not committed), and
runs/<run-id>/metrics.json + config.json (counts only). --set dev reads the dev packages and writes under data/gold/dev/.

pending lists, for the chapters a prefill added (provenance model_double_adjudicated, not done), every pending mark
as a place: the strings and readings of the two passes there (from their drafts, checked against what the prefill
wrote in the mark), the name sheet's reading when the sheet made the mark, the passes' doubts, and short context.
Places are grouped by name the same way; the groups of a name-sheet name are listed, not asked (the sheet is the
user's word on the name; the prefill already settled what the sheet could), and go to the reviewer with the
sheet's row. Same outputs as diff (questions.md, answer_sheet.md, disagreements.json; counts in runs/<run-id>/).
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.adjudicate import NEW_SETS, SETS, TYPE_NAMES, TYPE_ZH, _reading, load_knowledge
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import done_chapters, gold_origin, sha256
from novel_ner.gold import check as check_gold
from novel_ner.gold import own_set_path
from novel_ner.llm_task import TASK_ROOT, convert
from novel_ner.score import _context, draft_spans, gold_spans, strict
from novel_ner.snapshot import REPO_ROOT, load_snapshot
from novel_ner.zeroshot import git_state

KIND_ZH = {
    "type": "类型不同",
    "boundary": "框的范围不同",
    "only_answer": "答案标了，复核模型没标",
    "only_reviewer": "复核模型标了，答案没标",
    "reviewer_pending": "复核模型拿不准（待审）",
    "answer_unscored": "答案里这里设为不计分",
}
# pending (dev answers): how the two GPT passes differ at a place the prefill left pending
PENDING_KIND_ZH = {
    "type": "两遍类型不同",
    "boundary": "两遍框的范围不同",
    "only_a": "A 遍标了，B 遍没标",
    "only_b": "B 遍标了，A 遍没标",
    "gpt_pending": "GPT 拿不准（待审）",
    "sheet": "和名字知识表对不上",
    "other": "其他",
}


def places(answer_spans, unscored, reviewer_spans, reviewer_pending):
    """Group every span of one chapter into places: spans that overlap (directly or through others) are one place."""
    items = ([("answer", s) for s in answer_spans] + [("unscored", (s, e, None)) for s, e in unscored]
             + [("reviewer", s) for s in reviewer_spans] + [("pending", (s, e, None)) for s, e in reviewer_pending])
    items.sort(key=lambda x: (x[1][0], x[1][1]))
    groups, end = [], -1
    for item in items:
        if groups and item[1][0] < end:
            groups[-1].append(item)
            end = max(end, item[1][1])
        else:
            groups.append([item])
            end = item[1][1]
    out = []
    for g in groups:
        by = defaultdict(set)
        for who, span in g:
            by[who].add(span)
        out.append({"answer": by["answer"], "unscored": {(s, e) for s, e, _ in by["unscored"]},
                    "reviewer": by["reviewer"], "pending": {(s, e) for s, e, _ in by["pending"]}})
    return out


def kinds_of(place):
    """What kind of difference a place shows; [] when the answers and the reviewer agree there."""
    a, r = place["answer"], place["reviewer"]
    kinds = []
    if place["unscored"]:
        kinds.append("answer_unscored")
    if place["pending"]:
        kinds.append("reviewer_pending")
    if a != r:
        if a and r:
            kinds.append("type" if {(s, e) for s, e, _ in a} == {(s, e) for s, e, _ in r} else "boundary")
        elif a and not place["pending"]:
            kinds.append("only_answer")
        elif r and not place["unscored"]:
            kinds.append("only_reviewer")
    return kinds


def describe(spans, text):
    return "、".join(f"【{text[s:e]}】{TYPE_ZH.get(t, t)}" for s, e, t in sorted(spans)) or "没标"


def collect(export, ids, snapshot, reviews, origin):
    """Every disagreeing place of the test chapters, with its names, kinds and short context."""
    gold = gold_spans(export, ids)
    out = []
    for cid in ids:
        text = snapshot.chapters[cid].text
        rev = reviews[cid]
        for p in places(gold[cid]["spans"], gold[cid]["unscored"], rev["spans"], rev["pending"]):
            kinds = kinds_of(p)
            if not kinds:
                continue
            all_spans = [(s, e) for s, e, _ in p["answer"] | p["reviewer"]] + list(p["unscored"] | p["pending"])
            start, end = min(s for s, _ in all_spans), max(e for _, e in all_spans)
            names = sorted({text[s:e] for s, e in all_spans}, key=lambda x: (-len(x), x))
            out.append({
                "chapter_id": cid, "start": start, "end": end, "names": names, "kinds": kinds,
                "answer": sorted(p["answer"]), "reviewer": sorted(p["reviewer"]),
                "unscored": sorted(p["unscored"]), "pending": sorted(p["pending"]),
                "answer_decided_by_user": any(origin.get(cid, {}).get(s) == "user_or_sheet" for s in p["answer"]),
                "context": _context(text, start, end),
                "answer_says": describe(p["answer"], text) + ("（不计分）" if p["unscored"] else ""),
                "reviewer_says": describe(p["reviewer"], text) + ("（待审）" if p["pending"] else ""),
                "reviewer_doubt": [d for d in rev["doubts"] if any(s < d["end"] and d["start"] < e for s, e in p["pending"])],
            })
    return out


def pending_chapters(export):
    """The chapters a prefill added and nobody finished: provenance model_double_adjudicated, not done (file order)."""
    return [cid for cid, c in export["chapters"].items()
            if c.get("provenance") == "model_double_adjudicated" and c.get("status") != "done"]


def pass_readings(export, task_root=TASK_ROOT):
    """{chapter_id: {"A": spans, "B": spans}}: the two GPT passes the prefill merged (data/llm_tasks/<task>/draft_A.json
    and draft_B.json, recorded in the export's llm_drafts); a draft imported again after the prefill is refused."""
    out = {}
    for task, runs in export.get("llm_drafts", {}).items():
        for run in ("A", "B"):
            d = json.loads((Path(task_root) / task / f"draft_{run}.json").read_text(encoding="utf-8"))
            if (d.get("source_file"), d.get("imported_at")) != (runs[run]["source_file"], runs[run]["imported_at"]):
                raise SystemExit(f"{task}/draft_{run}.json was imported again after the prefill; prefill again")
            for cid, c in d["chapters"].items():
                out.setdefault(cid, {})[run] = c["spans"]
    return out


SHEET_READING = re.compile(r"知识表：【([^】]+)】(人物|组织|地点|副本|物品|特性|不标)")


def collect_pending(export, ids, snapshot, readings):
    """Every pending mark of the given chapters as a place: the strings of the mark and of the two passes' readings
    there, each pass's reading as adjudicate.merge wrote it into the mark's candidates ("A：...；B：...", checked against
    the drafts), the name sheet's readings when the sheet made the mark, the passes' doubts, and short context."""
    out = []
    for cid in ids:
        text = snapshot.chapters[cid].text
        if cid not in readings or set(readings[cid]) != {"A", "B"}:
            raise SystemExit(f"{cid}: the GPT passes the prefill merged are not recorded in the file")
        marks = sorted((a for a in export["chapters"][cid]["annotations"] if a.get("pending")),
                       key=lambda a: (a["start"], a["end"]))
        for m in marks:
            s, e = m["start"], m["end"]
            side = {run: [x for x in readings[cid][run] if x["start"] < e and s < x["end"]] for run in ("A", "B")}
            if any(x["start"] < s or e < x["end"] for run in side for x in side[run]):
                raise SystemExit(f"{cid} {s}-{e}: a pass's span reaches past the pending mark; the prefill and the drafts differ")
            says = {run: "、".join(_reading(x, text) for x in side[run]) or "不标" for run in side}
            candidates = m.get("candidates") or ""
            merged = candidates.startswith("A：")  # made by merge; the sheet's marks start otherwise
            if merged and candidates != f"A：{says['A']}；B：{says['B']}":
                raise SystemExit(f"{cid} {s}-{e}: the pending mark's candidates are not the two passes' readings; prefill again")
            sheet = [] if merged else list(dict.fromkeys(SHEET_READING.findall(candidates)))
            sure = {run: {(x["start"], x["end"], x["type"]) for x in side[run] if not x["pending"]} for run in side}
            unsure = {run: [x for x in side[run] if x["pending"]] for run in side}
            kinds = ["sheet"] if sheet else []
            if unsure["A"] or unsure["B"]:
                kinds.append("gpt_pending")
            a, b = sure["A"], sure["B"]
            if a != b:
                if a and b:
                    kinds.append("type" if {(x, y) for x, y, _ in a} == {(x, y) for x, y, _ in b} else "boundary")
                elif a and not unsure["B"]:
                    kinds.append("only_a")
                elif b and not unsure["A"]:
                    kinds.append("only_b")
            doubts = [x for run in side for x in unsure[run]]
            offered = {t for run in side for _, _, t in sure[run]} | {TYPE_NAMES[t] for _, t in sheet}
            offered |= {t for x in doubts for t in ENTITY_TYPES if t in (x.get("candidates") or "")}
            if not side["A"] or not side["B"] or any("不标" in (x.get("candidates") or "") for x in doubts):
                offered.add("none")
            names = sorted({text[s:e]} | {text[x["start"]:x["end"]] for run in side for x in side[run]},
                           key=lambda n: (-len(n), n))
            out.append({
                "chapter_id": cid, "start": s, "end": e, "mark_id": m.get("id"), "names": names, "kinds": kinds or ["other"],
                "stuck": m.get("stuck", []),
                "a": [[x["start"], x["end"], x["type"], x["pending"]] for x in side["A"]],
                "b": [[x["start"], x["end"], x["type"], x["pending"]] for x in side["B"]],
                "a_says": says["A"], "b_says": says["B"],
                "sheet_says": "、".join(f"【{n}】{t}" for n, t in sheet), "offered": sorted(offered),
                "context": _context(text, s, e),
                "doubts": [{"who": f"{run} 遍", "doubt": x.get("doubt") or "", "candidates": x.get("candidates") or ""}
                           for run in side for x in unsure[run]],
            })
    return out


def by_name(found, sheet_names):
    groups = defaultdict(list)
    for place in found:
        groups[" / ".join(place["names"])].append(place)
    asked, by_sheet = {}, {}
    for key, ps in groups.items():
        # the sheet settles a place when its longest name is a sheet name (the sheet fixes that whole string;
        # a shorter name there, like 章 inside 章医生, is the boundary the sheet already decides)
        target = by_sheet if ps[0]["names"][0] in sheet_names else asked
        target[key] = ps
    order = lambda d: dict(sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0])))
    return order(asked), order(by_sheet)


def questions_md(asked, by_sheet, label, run_id, wording=None, carried=None, carry=None, prior=None):
    """The questions, one per name group. wording: TEST_WORDING (the default: answers vs the reviewer) or
    PENDING_WORDING (dev answers: pass A vs pass B). With a memory (split_by_memory): carried groups are listed under
    已答过 with the earlier answers and the ruling (not asked), and prior[key] quotes the earlier answers under a
    question that is still asked. Without them the output is exactly as before."""
    w = wording or TEST_WORDING
    lines = [f"# {w['title']}（{label}，{run_id}）", "",
             f"含小说原文，只在本机看，不入库。【】是范围。{w['scope']}",
             "请用白话回答“这是什么”就行，比如“章医生是一个人的固定叫法”“这是普通的门，不是名字”“这里是外号，那里是职业”。",
             "不确定就写“不知道”，这些会交给复核模型按规则判，判不了的不计分。", ""]
    for n, (key, ps) in enumerate(asked.items(), 1):
        kinds = sorted({k for p in ps for k in p["kinds"]}, key=list(w["kinds"]).index)
        chapters = sorted({p["chapter_id"] for p in ps})
        decided = sum(p.get("answer_decided_by_user", False) for p in ps)
        lines.append(f"## {n}. {key}（{len(ps)} 处，{len(chapters)} 章）")
        lines.append(f"- 分歧：{'；'.join(w['kinds'][k] for k in kinds)}" + (f"；其中 {decided} 处是你上次定的" if decided else ""))
        for p in ps[:3]:
            says = "；".join(f"{who}：{p[k]}" for k, who in w["says"] if p.get(k))
            lines.append(f"- {p['chapter_id']}：{p['context']}　{says}")
            for d in p[w["doubts"]][:w["doubts_shown"]]:
                lines.append(f"  - {d.get('who', w['doubt_who'])}的疑点：{d['doubt']}（候选：{d['candidates']}）")
        if len(ps) > 3:
            lines.append(f"- 另有 {len(ps) - 3} 处，同样的分歧见 disagreements.json")
        if prior and key in prior:
            lines += prior_lines(prior[key])
        lines += ["- 你的回答：", ""]
    if carried:
        lines += [f"## {CARRIED_HEADING}", "", CARRIED_NOTE, ""]
        for k, (key, ps) in enumerate(carried.items(), 1):
            c = carry[key]
            chapters = sorted({p["chapter_id"] for p in ps})
            lines.append(f"### P{k}. {key}（{len(ps)} 处，{len(chapters)} 章）")
            p = ps[0]
            says = "；".join(f"{who}：{p[x]}" for x, who in w["says"] if p.get(x))
            lines.append(f"- {p['chapter_id']}：{p['context']}　{says}")
            lines += prior_lines(c, rulings=False)
            lines.append(f"- 以前的裁决：{READING_ZH[c['reading']]}（{c['carry_over_from']}）；这次的 {len(ps)} 处都按此处理"
                         + (f"，名字取“{c['text']}”" if len(key.split(" / ")) > 1 else ""))
            lines.append("")
    if by_sheet:
        lines += [f"## {w['sheet_heading']}", ""]
        for k, (key, ps) in enumerate(by_sheet.items(), 1):
            lines.append(w["sheet_line"](k, key, ps))
    return "\n".join(lines) + "\n"


def candidate_types(ps):
    """The types the reviewer offered in its pending marks for a group (from the candidates text)."""
    seen = set()
    for p in ps:
        for d in p["reviewer_doubt"]:
            text = d.get("candidates") or ""
            seen |= {t for t in ("person", "organization", "location", "dungeon", "item", "trait") if t in text}
            if "不标" in text:
                seen.add("none")
    return seen


def offered_types(ps):
    """The types the two passes (and the name sheet) offered at a group's pending places, "none" for 不标."""
    return {t for p in ps for t in p["offered"]}


TEST_WORDING = {
    "title": "复核问题清单", "scope": "每组是同一个名字（或同一串字）的所有分歧。",
    "says": (("answer_says", "现在的答案"), ("reviewer_says", "复核模型")),
    "doubts": "reviewer_doubt", "doubt_who": "复核模型", "doubts_shown": 1, "kinds": KIND_ZH,
    "name_kinds": {"only_answer", "only_reviewer", "answer_unscored"}, "types": candidate_types,
    "sheet_heading": "已按名字知识表处理（不用回答，列出来供你核对）",
    "sheet_line": lambda k, key, ps: f"- {key}：{len(ps)} 处",
    "sheet_doubts": "复核模型的疑点",
}
PENDING_WORDING = {
    "title": "待定处问题清单",
    "scope": "每组是同一个名字（或同一串字）的所有待定处：GPT 两遍（A 遍、B 遍）标得不一样，或有一遍拿不准；两遍一致的已直接采用，不在这里。",
    "says": (("a_says", "A 遍"), ("b_says", "B 遍"), ("sheet_says", "名字知识表")),
    "doubts": "doubts", "doubt_who": "GPT", "doubts_shown": 2, "kinds": PENDING_KIND_ZH,
    "name_kinds": {"only_a", "only_b"}, "types": offered_types,
    "sheet_heading": "名字知识表里的名字（不再问你：GPT 在这些地方和知识表对不上，会连同你在知识表里写的交给复核模型按规则逐处判，"
                     "判不了的不计分；想自己定就在回答表里按编号补一行）",
    "sheet_line": lambda k, key, ps: f"- S{k}. {key}：{len(ps)} 处" + (f"（知识表：{ps[0]['sheet_says']}）" if ps[0].get("sheet_says") else ""),
    "sheet_doubts": "GPT 的疑点",
    "sheet_part": "四、名字知识表里的名字（可以不填：不填就连同知识表交给复核模型；填了就按你写的）",
}


def answer_sheet_md(asked, label, run_id, wording=None, by_sheet=None, carried=None, carry=None):
    """A short fill-in table per question kind; the full context stays in questions.md. With by_sheet and a wording
    that has "sheet_part" (pending), the name-sheet groups follow as an optional part; with carried groups (a memory),
    they follow under 已答过 with the answer cell pre-filled 已答：<type> （<deciders and dates>）."""
    w = wording or TEST_WORDING
    (left_key, left), (right_key, right) = w["says"][:2]
    item_or_trait, is_it_a_name, other = [], [], []
    for n, (key, ps) in enumerate(asked.items(), 1):
        cands = w["types"](ps)
        kinds = {k for p in ps for k in p["kinds"]}
        row = (n, key, len(ps), ps[0][left_key], ps[0][right_key], "")
        if cands and cands <= {"item", "trait"}:
            item_or_trait.append(row)
        elif "none" in cands or kinds & w["name_kinds"]:
            is_it_a_name.append(row)
        else:
            other.append(row)
    lines = [f"# 回答表（{label}，{run_id}）", "",
             f"编号和 questions.md 里的一样，原文片段和{w['sheet_doubts']}都在那里。只填最后一列，白话就行；不知道就写“不知道”。", ""]
    parts = [("一、道具还是特性（书里它是一件道具，还是人物的特性、能力？）", item_or_trait),
             ("二、是不是名字、是什么（书里是专门的叫法，还是普通说法、编号、口令？）", is_it_a_name),
             ("三、其他（地点还是组织、名字框多长等）", other)]
    if by_sheet and w.get("sheet_part"):
        parts.append((w["sheet_part"], [(f"S{k}", key, len(ps), ps[0][left_key], ps[0][right_key], "")
                                        for k, (key, ps) in enumerate(by_sheet.items(), 1)]))
    if carried:
        parts.append((CARRIED_HEADING, [(f"P{k}", key, len(ps), ps[0][left_key], ps[0][right_key], prefilled_cell(carry[key]))
                                        for k, (key, ps) in enumerate(carried.items(), 1)]))
    for title, rows in parts:
        if not rows:
            continue
        lines += [f"## {title}", "", f"| 编号 | 名字 | 处数 | {left}（第一处） | {right}（第一处） | 你的回答 |", "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {key} | {k} | {a} | {r} | {ans} |" for n, key, k, a, r, ans in rows]
        lines.append("")
    return "\n".join(lines) + "\n"


# ---- answer memory: what the user already answered, how it was ruled, per exact string ----
MEMORY_ANSWERS = ("configs/review_astra_answers.json", "configs/dev_pending_answers.json")
MEMORY_DECISIONS = ("configs/gold_test_v3_decisions.json", "configs/gold_dev_v2_decisions.json")
MEMORY_OUT = "configs/answer_memory.json"
READINGS = (*ENTITY_TYPES, "none", "unscored")
READING_ZH = {**TYPE_ZH, "unscored": "不计分"}
# the type words an answer may name (adjudicate.TYPE_NAMES' Chinese keys, plus 口令 and 啥也不是 as none)
TYPE_WORDS = {**{zh: t for zh, t in TYPE_NAMES.items() if not zh.isascii()}, "口令": "none", "啥也不是": "none",
              "不计分": "unscored"}
PRIOR_BY = "prior ruling: "  # the "by" of a decided place that kept an earlier ruling
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CARRIED_HEADING = "已答过、按以前的裁决处理（不用再答；要改就在这一列写新答案）"
CARRIED_NOTE = "这些名字你以前答过，而且以前的裁决一致，这次不再问：回答表里这些行（P1、P2……）的最后一列已经填好“已答：…”，不动就按它处理；想改就把那一格换成新答案。"
PREFILLED = "已答："


def memory_source(spec, kind):
    """One source of the memory, "[set=]path": the set is read from the file name when not given
    (review_astra_answers -> test, <set>_pending_answers, gold_<set>_v<n>_decisions)."""
    set_name, path = spec.split("=", 1) if "=" in spec else (None, spec)
    stem = Path(path).stem
    if kind == "answers":
        m = re.fullmatch(r"(\w+?)_pending_answers", stem)
        guess = "test" if stem == "review_astra_answers" else m.group(1) if m else None
        extra = {"round": "review" if stem == "review_astra_answers" else "pending"}
    else:
        m = re.fullmatch(r"gold_(\w+?)_v(\d+)_decisions", stem)
        guess = m.group(1) if m else None
        extra = {"tag": f"{m.group(1)} v{m.group(2)}" if m else stem}
    set_name = set_name or guess
    if set_name not in SETS:
        raise SystemExit(f"{spec}: name the file's set as set=path (one of {SETS})")
    return {"kind": kind, "path": path, "set": set_name, **extra}


def who_decided(r):
    """A ruling as 'decider date (file tag)': user 2026-10-03 (test v3), reviewer astra 2026-10-03 (dev v2),
    name sheet, user 2026-10-02 (test v3); a ruling that was itself carried over names only the round."""
    head = r["by"].split("（")[0].strip()
    if r["by"].startswith(PRIOR_BY):
        head = "carried over"
    elif r["decider"] == "reviewer":
        head = f"reviewer {head}"
    elif r["decider"] == "sheet":
        head = f"name sheet, {head}"
    return f"{head} ({r['tag']})"


def carry_over_from(entry):
    """A short English string naming the deciders and dates behind an entry's rulings and sheet row."""
    parts = [who_decided(r) for r in entry["rulings"]]
    if entry["sheet"]:
        on = entry["sheet"].get("decided_on")
        parts.append("name sheet" + (f" (user {on})" if on else ""))
    return "; ".join(dict.fromkeys(parts))


def carry_over_decider(entry):
    """The decider a kept ruling is applied with (one of gold.DECIDERS, which evaluate.gold_origin counts by): the one
    decider when the rulings and the sheet row agree on it; reviewer when a reviewer ruling is among them (the place
    is then not counted as decided by the user); user when only user and sheet rulings mix."""
    deciders = {r["decider"] for r in entry["rulings"]} | ({"sheet"} if entry["sheet"] else set())
    if len(deciders) == 1:
        return next(iter(deciders))
    return "reviewer" if "reviewer" in deciders else "user"


def build_memory(answers=MEMORY_ANSWERS, decisions=MEMORY_DECISIONS, sheet_path=None, root=REPO_ROOT):
    """The memory: {"generated_at", "sources" (with sha256), "counts", "entries": [{"text", "answers" (verbatim, per
    answers file), "sheet": {type, evidence (chapter number), decided_on (the date in the row's note)} or null,
    "rulings" (per decisions file, reading and decider), "carry_over": the one reading when every ruling of the string
    and the sheet row agree, else null, "carry_over_from", "carry_over_decider"}]}. A group name like "A / B" is two
    strings; an answer with "text" belongs to that string only. Names only, no chapter text (the sheet's note is not
    copied: a row's note may quote its evidence sentence; only the date in it is kept)."""
    root = Path(root)
    entries = defaultdict(lambda: {"answers": [], "sheet": None, "rulings": []})
    sources = []
    for spec in answers:
        src = memory_source(spec, "answers")
        data = json.loads((root / src["path"]).read_text(encoding="utf-8"))
        for a in data["answers"]:
            for t in ([a["text"]] if a.get("text") else a["name"].split(" / ")):
                entries[t]["answers"].append({
                    "source": src["path"], "set": src["set"], "round": src["round"], "no": a["no"],
                    "answered_at": data["answered_at"], "answer": a["answer"], "handling": a["handling"],
                    "reading": a.get("reading"), **({"text": a["text"]} if a.get("text") else {}),
                    **({"note2": a["note2"]} if a.get("note2") else {})})
        sources.append({**src, "sha256": sha256(root / src["path"]), "answered_at": data["answered_at"],
                        "answers": len(data["answers"])})
    # a later decisions file of the same set (a later answers version) supersedes an earlier one at the places it
    # decides again: the earlier ruling of that place is dropped, so the memory carries the current version's reading
    loaded = []
    for spec in decisions:
        src = memory_source(spec, "decisions")
        data = json.loads((root / src["path"]).read_text(encoding="utf-8"))
        loaded.append((src, data))
    superseded = Counter()
    latest = {}  # (set, chapter_id, start, end) -> index of the last file (in the given order) deciding that place
    for i, (src, data) in enumerate(loaded):
        for p in data["places"]:
            latest[(src["set"], p["chapter_id"], p["start"], p["end"])] = i
    for i, (src, data) in enumerate(loaded):
        groups = {}
        for p in data["places"]:
            if latest[(src["set"], p["chapter_id"], p["start"], p["end"])] != i:
                superseded[src["path"]] += 1
                continue
            g = groups.setdefault((p["text"], p["reading"], p["decider"]), {"by": [], "places": 0})
            g["places"] += 1
            if p["by"] not in g["by"]:
                g["by"].append(p["by"])
        for (t, reading, decider), g in groups.items():
            entries[t]["rulings"].append({"source": src["path"], "set": src["set"], "tag": src["tag"], "reading": reading,
                                          "decider": decider, "by": "; ".join(g["by"]), "places": g["places"]})
        sources.append({**src, "sha256": sha256(root / src["path"]), "places": len(data["places"]),
                        "places_superseded_by_a_later_file_of_the_set": superseded[src["path"]]})
    sheet_file = Path(sheet_path) if sheet_path else root / "data" / "gold" / "name_knowledge.tsv"
    rows = load_knowledge(sheet_file)
    for r in rows:
        on = DATE.search(r["note"] or "")
        entries[r["name"]]["sheet"] = {"type": r["type"], "evidence": r["chapter"], "decided_on": on.group(0) if on else None}
    rel = sheet_file.resolve()
    sources.append({"kind": "sheet", "path": (rel.relative_to(root.resolve()) if rel.is_relative_to(root.resolve()) else rel).as_posix(),
                    "sha256": sha256(sheet_file), "rows": len(rows)})
    out = []
    for t in sorted(entries):
        e = entries[t]
        readings = {r["reading"] for r in e["rulings"]} | ({e["sheet"]["type"]} if e["sheet"] else set())
        carry = next(iter(readings)) if len(readings) == 1 else None
        out.append({"text": t, "answers": e["answers"], "sheet": e["sheet"], "rulings": e["rulings"],
                    "carry_over": carry, "carry_over_from": carry_over_from(e) if carry else None,
                    "carry_over_decider": carry_over_decider(e) if carry else None})
    by_source = Counter()
    for e in out:
        for s in {a["source"] for a in e["answers"]} | {r["source"] for r in e["rulings"]} | ({sources[-1]["path"]} if e["sheet"] else set()):
            by_source[s] += 1
    return {"description": "Answer memory: for every exact string the user answered before (verbatim, with how the answer "
            "was handled), the rulings recorded on it (per decisions file, reading and decider) and its name-sheet row; "
            "carry_over is the one reading when every ruling and the sheet row agree, else null. Built mechanically by "
            "review.py memory from the sources below; names only, no chapter text; nothing here judges an answer.",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": sources, "format": 2,
            "counts": {"entries": len(out), "with_carry_over": sum(1 for e in out if e["carry_over"]),
                       "with_mixed_rulings": sum(1 for e in out if not e["carry_over"] and (e["rulings"] or e["sheet"])),
                       "with_answers": sum(1 for e in out if e["answers"]), "with_rulings": sum(1 for e in out if e["rulings"]),
                       "with_sheet_row": sum(1 for e in out if e["sheet"]), "by_source": dict(by_source)},
            "entries": out}


def load_memory(path):
    """The memory file as {"entries": {text: entry}, ...}; a file of an older format (no carry_over_decider) is refused."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != 2:
        raise SystemExit(f"{path} is an older memory file; build it again (review memory)")
    return {**data, "entries": {e["text"]: e for e in data["entries"]}}


def prior_of(key, memory):
    """What the memory holds for a group of a listing: (entries by string, carry, answers). carry is
    {"text", "reading", "decider", "carry_over_from"} when every string of the group has a carry_over and at most one
    of them is marked (the others 不标; the decider is the marked string's, or the first string's when none is marked),
    else None; answers are the earlier answers of its strings, verbatim, each once."""
    strings = key.split(" / ")
    found = {t: memory[t] for t in strings if t in memory}
    answers = list({(a["source"], str(a["no"])): a for t in strings for a in found.get(t, {}).get("answers", [])}.values())
    carry = None
    if found and all(t in found and found[t]["carry_over"] for t in strings):
        marked = [t for t in strings if found[t]["carry_over"] != "none"]
        if len(marked) <= 1:
            t = marked[0] if marked else strings[0]
            carry = {"text": t, "reading": found[t]["carry_over"], "decider": found[t]["carry_over_decider"],
                     "carry_over_from": "; ".join(dict.fromkeys(found[s]["carry_over_from"] for s in strings))}
    return found, carry, answers


def rulings_words(found):
    """The rulings and sheet rows of a group's strings in Chinese, for quoting: 不标 19 处（reviewer astra ...）."""
    out = []
    for t, e in found.items():
        words = [f"{READING_ZH[r['reading']]} {r['places']} 处（{who_decided(r)}）" for r in e["rulings"]]
        if e["sheet"]:
            words.append(f"知识表 {READING_ZH[e['sheet']['type']]}")
        if words:
            out.append((f"{t}：" if len(found) > 1 else "") + "、".join(words))
    return "；".join(out)


def split_by_memory(asked, memory):
    """(asked, carried, carry, prior): a group whose strings all carry over leaves asked for carried (carry[key] from
    prior_of, with the earlier answers); a group with memory entries but no carry-over stays asked, with its earlier
    answers and rulings in prior[key] to quote."""
    still, carried, carry, prior = {}, {}, {}, {}
    for key, ps in asked.items():
        found, c, answers = prior_of(key, memory)
        if c:
            carried[key], carry[key] = ps, {**c, "answers": answers}
        else:
            still[key] = ps
            if found:
                prior[key] = {"answers": answers, "rulings": rulings_words(found)}
    return still, carried, carry, prior


def prior_lines(info, rulings=True):
    lines = [f"- 你以前答过：{a['set']} 第{a['no']}题（{a['answered_at']}）：{a['answer']}"
             + (f"；后来补充：{a['note2']}" if a.get("note2") else "") for a in info["answers"]]
    if rulings and info.get("rulings"):
        lines.append(f"- 以前的裁决：{info['rulings']}（不一致或没有统一裁决，所以再问一次）")
    return lines


def prefilled_cell(c):
    return f"{PREFILLED}{READING_ZH[c['reading']]} （{c['carry_over_from']}）"


def named_types(answer):
    """The readings an answer names by its type words, in order of first appearance, each once; 不标 right after 标
    (标不标: whether to mark) is not a type word."""
    hits = []
    for word, t in TYPE_WORDS.items():
        for m in re.finditer(re.escape(word), answer):
            if word == "不标" and m.start() > 0 and answer[m.start() - 1] == "标":
                continue
            hits.append((m.start(), t))
    return list(dict.fromkeys(t for _, t in sorted(hits)))


def read_sheet(path):
    """The rows of a filled answer sheet: every table row (not the header or the rule) as {no, name, places, a, b,
    answer}; the last column is the user's answer."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "编号" or set(cells[0]) <= {"-", ":"}:
            continue
        if len(cells) != 6 or not cells[2].isdigit():
            raise SystemExit(f"{path}: not a sheet row of six cells: {line}")
        rows.append({"no": int(cells[0]) if cells[0].isdigit() else cells[0], "name": cells[1], "places": int(cells[2]),
                     "a": cells[3], "b": cells[4], "answer": cells[5]})
    return rows


VERDICT_ZH = {"consistent": "一致", "conflict": "不一致", "repeat_without_type": "没说类型，有以前的裁决",
              "new": "以前没答过", "unclear": "没说类型，以前也没答过"}


def verdict(key, answer, memory):
    """How a new answer stands to the memory, mechanically: consistent (names the carry-over reading, or the sheet type
    when there is no carry-over), conflict (names another), repeat_without_type (names none, a reference exists), new
    (no memory entry), unclear (names none, no memory entry). An answer naming several types names none; a group with
    entries but no reference (mixed rulings, a reviewer-handled answer without a ruling) is new or unclear, noted."""
    found, carry, answers = prior_of(key, memory)
    reference = carry["reading"] if carry else next(
        (e["sheet"]["type"] for t in key.split(" / ") for e in [found.get(t)] if e and e["sheet"]), None)
    named = named_types(answer)
    new = named[0] if len(named) == 1 else None
    if reference is not None:
        v = "consistent" if new == reference else "conflict" if new else "repeat_without_type"
    else:
        v = "new" if new else "unclear"
    notes = []
    if len(named) > 1:
        notes.append(f"提到了 {len(named)} 种：{'、'.join(READING_ZH[t] for t in named)}")
    if reference is None and found:
        notes.append("以前答过，但没有可沿用的裁决")
    return {"verdict": v, "named": named, "reference": reference, "notes": notes, "found": found, "carry": carry,
            "prior_answers": "；".join(f"{a['set']} 第{a['no']}题（{a['answered_at']}）：{a['answer']}"
                                      + (f"；后来补充：{a['note2']}" if a.get("note2") else "") for a in answers),
            "prior_ruling": f"{READING_ZH[carry['reading']]}（{carry['carry_over_from']}）" if carry else rulings_words(found)}


def check_answers(set_name, sheet_file, listing_file, memory_file, run_id, root=REPO_ROOT):
    """review check-answers: every row of a filled sheet against the memory (verdict), written as
    <listing folder>/answer_check.md and counted in runs/<run-id>/; no answer is changed, no conflict decided."""
    root = Path(root)
    listing_path = own_set_path(root / listing_file, set_name, root)
    listing = json.loads(listing_path.read_text(encoding="utf-8"))
    if listing.get("set") != set_name or listing.get("mode") != "pending":
        raise SystemExit(f"{listing_file} is not a pending listing of {set_name}")
    sheet_path = own_set_path(root / sheet_file, set_name, root)
    rows = read_sheet(sheet_path)
    mem = load_memory(root / memory_file)
    groups = {**listing["asked"], **listing["by_sheet"], **listing.get("carried", {})}
    numbers = {key: str(n) for n, key in enumerate(listing["asked"], 1)}
    numbers.update({key: f"S{k}" for k, key in enumerate(listing["by_sheet"], 1)})
    numbers.update({key: f"P{k}" for k, key in enumerate(listing.get("carried", {}), 1)})
    out, counts, seen = [], Counter(), set()
    several = kept = 0
    for r in rows:
        key = r["name"]
        if key not in numbers or numbers[key] != str(r["no"]) or len(groups[key]) != r["places"]:
            raise SystemExit(f"{sheet_file}: row {r['no']} {key} ({r['places']} places) is not a group of the listing as numbered")
        if key in seen:
            raise SystemExit(f"{sheet_file}: {key} is listed twice")
        seen.add(key)
        v = verdict(key, r["answer"], mem["entries"])
        counts[v["verdict"]] += 1
        several += len(v["named"]) > 1
        kept += r["answer"].startswith(PREFILLED)
        out.append({**r, **v})
    missing = [numbers[k] for k in groups if k not in seen]
    lines = [f"# 回答核对（{set_name}，{run_id}）", "",
             f"回答表 {sheet_file} 的每一行，和以前的回答、裁决（{memory_file}）机械比对：判定只看回答里有没有类型词"
             f"（{'、'.join(TYPE_WORDS)}），不判对错；冲突由你定。consistent＝一致，conflict＝不一致，"
             "repeat_without_type＝没说类型但有以前的裁决，new＝以前没答过，unclear＝没说类型、以前也没答过。", "",
             "| 编号 | 名字 | 以前的回答 | 以前的裁决 | 这次的回答 | 判定 |", "|---|---|---|---|---|---|"]
    for r in out:
        v = r["verdict"] + "（" + "；".join([VERDICT_ZH[r["verdict"]], *r["notes"]]) + "）"
        lines.append(f"| {r['no']} | {r['name']} | {r['prior_answers']} | {r['prior_ruling']} | {r['answer']} | {v} |")
    if missing:
        lines += ["", f"回答表里没有这些组的行：{'、'.join(missing)}"]
    (sheet_path.parent / "answer_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "action": "check-answers", "set": set_name, "sheet": sheet_file, "sheet_sha256": sha256(sheet_path),
        "listing": listing_file, "listing_sha256": sha256(listing_path), "listing_run_id": listing.get("run_id"),
        "memory": memory_file, "memory_sha256": sha256(root / memory_file), "memory_generated_at": mem.get("generated_at"),
        "code": git_state(), "created_at": created}, ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {"run_id": run_id, "set": set_name, "rows": len(out), "groups_in_listing": len(groups),
               "groups_without_row": len(missing), "verdicts": {k: counts.get(k, 0) for k in VERDICT_ZH},
               "rows_naming_several_types": several, "rows_prefilled_kept": kept,
               "note": "the filled sheet against the answer memory, mechanically by type words; no answer was changed, "
                       "no conflict decided"}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, metrics


ADJUDICATE_INSTRUCTIONS = REPO_ROOT / "docs" / "llm_annotation" / "adjudicate_v1.md"
NUMBERED = re.compile(r"(?<![0-9A-Za-z])[0-9]{1,4}号")


def paragraph_of(chapter, start, end):
    """(index, paragraph id, paragraph text, local start, local end) of the paragraph holding [start, end)."""
    offset = 0
    for i, (pid, text) in enumerate(chapter.paragraphs):
        if offset <= start and end <= offset + len(text):
            return i, pid, text, start - offset, end - offset
        offset += len(text) + 1
    raise ValueError(f"{chapter.chapter_id} {start}-{end} is not inside one paragraph")


def numbered_convention(export, ids, snapshot, skip):
    """How the current answers mark "Arabic number + 号" strings in the test chapters, leaving out the places being
    adjudicated (skip: {chapter_id: [(start, end)]}): {string: Counter(type, "none" or "unscored")}."""
    out = defaultdict(Counter)
    for cid in ids:
        text = snapshot.chapters[cid].text
        chap = export["chapters"][cid]
        for m in NUMBERED.finditer(text):
            s, e = m.start(), m.end()
            if any(a < e and s < b for a, b in skip.get(cid, [])):
                continue
            marks = [a.get("type") for a in chap["annotations"] if (a["start"], a["end"]) == (s, e) and not a.get("pending")]
            unscored = any(u["start"] < e and s < u["end"] for u in chap.get("unscored", []))
            out[m.group(0)][marks[0] if marks else "unscored" if unscored else "none"] += 1
    return out


def other_occurrences(name, ids, snapshot, export, taken):
    """Occurrences of a name in the test chapters outside the given places (taken: {chapter_id: [(start, end)]}) and
    outside any longer mark around them, with the current answers' marks there."""
    out = []
    for cid in ids:
        text = snapshot.chapters[cid].text
        chap = export["chapters"][cid]
        at = text.find(name)
        while at >= 0:
            stop = at + len(name)
            if not any(a < stop and at < b for a, b in taken.get(cid, [])):
                touch = [a for a in chap["annotations"] if a["start"] < stop and at < a["end"]]
                longer = [a for a in touch if a["start"] <= at and stop <= a["end"] and (a["start"], a["end"]) != (at, stop)]
                unscored = [u for u in chap.get("unscored", []) if u["start"] < stop and at < u["end"]]
                around = touch + unscored  # the place is the occurrence widened to what touches it
                out.append({"chapter_id": cid, "start": min([at] + [x["start"] for x in around]),
                            "end": max([stop] + [x["end"] for x in around]), "at": at,
                            "inside_longer_mark": bool(longer),
                            "marks": [(a["text"], a.get("type")) for a in touch], "unscored": bool(unscored),
                            "context": _context(text, at, stop)})
            at = text.find(name, stop)
    return out


def places_of(groups):
    taken = defaultdict(list)
    for ps in groups:
        for p in ps:
            taken[p["chapter_id"]].append((p["start"], p["end"]))
    return taken


def reviewer_groups(answers, asked):
    groups = []
    for a in answers["answers"]:
        if a["handling"] != "reviewer":
            continue
        if a["name"] not in asked or len(asked[a["name"]]) != a["places"]:
            raise SystemExit(f"answer {a['no']} {a['name']}: its places are not in the list as answered")
        groups.append((a, asked[a["name"]]))
    return groups


def group_id(a):
    """q<no> for an answered group (as in the test package), s<k> for a name-sheet group of a pending listing."""
    return a.get("gid") or f"q{a['no']:02d}"


def sheet_words(row):
    """A name-sheet row as book knowledge: the name, the type the user gave it, the user's note."""
    return f"{row['name']}——{TYPE_ZH[row['type']]}" + (f"；备注：{row['note']}" if row.get("note") else "")


SHEET_GROUPS = ("reviewer", "sheet")


def pending_groups(answers, disagreements, sheet_groups="reviewer", memory=None):
    """(answer, places) for every group of a pending listing, asked groups first, in listing order. Every asked group
    needs the user's answer with "no" and "name" as listed. A name-sheet group (S1, S2, ...) the user did not answer
    goes to the reviewer with the sheet's row as the book knowledge and the row's type as fixed (sheet_groups="sheet":
    it is decided by the row as it stands). Answers decide their listed places only (no scope: the passes agreed
    everywhere else). A prior answer (an earlier ruling kept) is checked against the listing's carry, or, for a group
    the listing did not carry over, against the memory's entries (memory: load_memory(...)["entries"], required then):
    its reading, string and carry_over_from must be the carry-over's, and it is given the carry-over's decider."""
    if sheet_groups not in SHEET_GROUPS:
        raise SystemExit(f"sheet groups go to one of {SHEET_GROUPS}, not {sheet_groups!r}")
    asked, by_sheet = disagreements["asked"], disagreements["by_sheet"]
    carried, carry = disagreements.get("carried", {}), disagreements.get("carry", {})
    numbers = {key: n for n, key in enumerate(asked, 1)}
    numbers.update({key: f"S{k}" for k, key in enumerate(by_sheet, 1)})
    numbers.update({key: f"P{k}" for k, key in enumerate(carried, 1)})
    answered = {}
    for a in answers["answers"]:
        key = a["name"]
        if key not in numbers:
            raise SystemExit(f"answer {a['no']}: {key} is not a group of the listing")
        if key in answered or a["no"] != numbers[key]:
            raise SystemExit(f"answer {a['no']}: {key} is answered twice, or the listing numbers it {numbers[key]}")
        ps = asked.get(key) or by_sheet.get(key) or carried[key]
        if a["places"] != len(ps):
            raise SystemExit(f"answer {a['no']} {key}: expected {a['places']} places, found {len(ps)}")
        if a["handling"] not in ("direct", "reviewer", "prior"):
            raise SystemExit(f"answer {a['no']}: handling {a['handling']!r} is none of direct, reviewer, prior")
        if a["handling"] in ("direct", "prior") and a.get("reading") not in READINGS:
            raise SystemExit(f"answer {a['no']}: a {a['handling']} answer needs a reading (one of the six types, none or unscored)")
        if a["handling"] == "prior":
            if not isinstance(a.get("prior"), dict) or not a["prior"].get("carry_over_from"):
                raise SystemExit(f"answer {a['no']}: a prior answer needs prior.carry_over_from (the earlier deciders and dates)")
            if key in carry:
                ref, where = carry[key], "the listing carried"
            elif memory is not None:
                ref, where = prior_of(key, memory)[1], "the memory carries"
            else:
                raise SystemExit(f"answer {a['no']}: the listing did not carry {key} over; a prior answer on it is checked "
                                 "against the memory (decide/package --memory)")
            if ref is None:
                raise SystemExit(f"answer {a['no']}: the memory has no carry-over for {key} (no ruling, or mixed rulings); "
                                 "answer it direct or give it to the reviewer, not prior")
            if a["reading"] != ref["reading"]:
                raise SystemExit(f"answer {a['no']}: {where} {key} over as {ref['reading']}, not {a['reading']}; "
                                 "a changed answer is direct or reviewer, not prior")
            if a.get("text", ref["text"]) != ref["text"]:
                raise SystemExit(f"answer {a['no']}: {where} {key} over with the string {ref['text']}, not {a['text']}")
            if a["prior"]["carry_over_from"] != ref["carry_over_from"]:
                raise SystemExit(f"answer {a['no']}: {where} {key} over from {ref['carry_over_from']!r}, "
                                 f"not {a['prior']['carry_over_from']!r}")
            if a["prior"].get("decider", ref["decider"]) != ref["decider"]:
                raise SystemExit(f"answer {a['no']}: the carried ruling on {key} was decided by {ref['decider']}, "
                                 f"not {a['prior']['decider']}")
            a = {**a, "text": ref["text"], "prior": {**a["prior"], "decider": ref["decider"]}}
        if a.get("scope") or a.get("exclude"):
            raise SystemExit(f"answer {a['no']}: a pending answer decides its listed places only (no scope or exclude)")
        answered[key] = a
    missing = [numbers[key] for key in list(asked) + list(carried) if key not in answered]
    if missing:
        raise SystemExit(f"no answer for the asked or carried-over group(s) {missing}"
                         + " (a carried-over group kept as before takes a prior entry from the listing's carry)" * any(
                             key in carried and key not in answered for key in carried))
    groups = []
    for key, ps in list(asked.items()) + list(carried.items()) + list(by_sheet.items()):
        gid = (f"q{numbers[key]:02d}" if key in asked else f"p{int(numbers[key][1:]):02d}" if key in carried
               else f"s{int(numbers[key][1:]):02d}")
        if key in answered:
            groups.append(({**answered[key], "gid": gid}, ps))
            continue
        row = disagreements["sheet_rows"][ps[0]["names"][0]]
        fixed = row["type"] if sheet_groups == "reviewer" and row["type"] in ENTITY_TYPES else None
        groups.append(({"no": numbers[key], "gid": gid, "name": key, "places": len(ps), "answer": sheet_words(row),
                        "handling": sheet_groups, "sheet_row": row, "fixed_type": fixed}, ps))
    return groups


def sentences_with(name, chapter):
    """The sentences of a chapter that contain the name (split at 。！？ inside each paragraph)."""
    out = []
    for _, para in chapter.paragraphs:
        for s in re.findall(r"[^。！？]*[。！？]?", para):
            if name in s and s.strip():
                out.append(s.strip())
    return out


def build_package(answers, disagreements, export, ids, snapshot, out_dir, guide_file, only=None,
                  answers_config="configs/review_astra_answers.json", sheet_groups="reviewer", memory=None):
    """The adjudication package for the reviewer: rules, instructions, and every place of the hedged answers with
    the user's words as book knowledge. Neither the current answers' nor the reviewer's earlier reading is shown;
    for numbered names the current marking of the other numbered names is given, because the user asked for one
    consistent treatment of them. only: answer numbers to package (a second ruling on some groups); a group may
    carry later words of the user (note3) and evidence the user pointed to (sentences of the chapters they found,
    from the snapshot, and lines transcribed from their screenshots for chapters the snapshot does not hold).

    A pending listing (dev answers) packages the groups pending_groups sends to the reviewer: the hedged answers,
    and the name-sheet groups the user did not answer (the sheet's row as the book knowledge, its type and its whole
    string fixed, the sentences of its evidence chapter). Only the listed places go in: the passes agreed everywhere else, and the
    numbered-name convention was the user's request for the test. The passes' readings are not shown either.
    only takes answer numbers, so a repackage leaves the name-sheet groups out."""
    pending = disagreements.get("mode") == "pending"
    by_name = {a["name"]: a for a in answers["answers"]}
    by_no = {a["no"]: a for a in answers["answers"]}
    by_number = {c.number: c for c in snapshot.chapters.values()}
    if pending:
        groups = [(a, ps) for a, ps in pending_groups(answers, disagreements, sheet_groups, memory)
                  if a["handling"] == "reviewer" and (only is None or a["no"] in only)]
        extra, convention = {group_id(a): [] for a, _ in groups}, None
    else:
        groups = [(a, ps) for a, ps in reviewer_groups(answers, disagreements["asked"]) if only is None or a["no"] in only]
        asked_places = places_of(disagreements["asked"].values())
        # the same names elsewhere in the test chapters go in too, so one ruling covers every use (the user asked for
        # consistency); places inside a longer mark or inside another listed place are left alone
        extra = {group_id(a): [{**o, "names": [ps[0]["names"][0]], "extra": True}
                               for o in other_occurrences(ps[0]["names"][0], ids, snapshot, export, asked_places)
                               if not o["inside_longer_mark"]] for a, ps in groups}
        skip = places_of([ps for _, ps in groups] + list(extra.values()))
        convention = numbered_convention(export, ids, snapshot, skip)
    zh = {**TYPE_ZH, "none": "不标", "unscored": "不计分"}
    lines = ["# 要裁决的地方", "",
             "每组先给“书的知识”（读过全书的人写的原话），再逐处给原文：前一段、本段（【】标出要裁决的那段原文）、后一段。", ""]
    places = []
    for a, ps in groups:
        gid = group_id(a)
        row = a.get("sheet_row")
        lines += [f"## {gid}　{a['name']}", "", f"书的知识（{'读者的名字知识表' if row else '原话'}）：{a['answer']}"]
        if row and row.get("chapter") in by_number:
            found = sentences_with(row["name"], by_number[row["chapter"]])
            if found:
                lines += ["", f"名字知识表给的证据章里含“{row['name']}”的句子（跨章证据，规则 E1）："]
                lines += [f"- 第{row['chapter']}章：{s}" for s in found]
        if a.get("note2"):
            lines.append(f"读者后来补充（原话）：{a['note2']}")
        for n3 in a.get("note3", []):
            lines.append(f"读者再补充（原话）：{n3}")
        if a.get("evidence_chapters") or a.get("evidence_transcribed"):
            lines += ["", f"读者在全书里搜索“{a['name']}”找到的句子（跨章证据，规则 E1）："]
            for n in a.get("evidence_chapters", []):
                for s in sentences_with(a["name"], by_number[n]):
                    lines.append(f"- 第{n}章：{s}")
            for s in a.get("evidence_transcribed", []):
                lines.append(f"- （读者截图转录，第 990 章以后，不在本次原文里）{s}")
        refs = [b for other, b in by_name.items() if other != a["name"] and (other in a["answer"] or other in a["name"])
                and not (pending and len(other) < 2)]  # a one-character group (人) would match words like 人物
        refs += [by_no[n] for n in a.get("refers_to", []) if by_no[n] not in refs]
        for b in refs:
            lines.append(f"（原话里说到的“{b['name']}”，那一条的原话：{b['answer']}）")
        if a.get("fixed_type") and row:  # a name-sheet row fixes the whole string too (decide stops on another frame)
            lines.append(f"类型已由读者确定：{zh[a['fixed_type']]}；名字也已由读者确定为整串“{row['name']}”。只裁决这里要不要标；"
                         f"如果要标，type 必须是 {a['fixed_type']}，text 必须是“{row['name']}”。")
        elif a.get("fixed_type"):
            lines.append(f"类型已由读者确定：{zh[a['fixed_type']]}。只裁决要不要标、框多长（规则 B 部分）；"
                         f"如果要标，type 必须是 {a['fixed_type']}。")
        if convention is not None and NUMBERED.fullmatch(ps[0]["names"][0]):
            marked = "；".join(f"{k} {sum(v.values())} 处，" + "、".join(f"{zh.get(t, t)} {n}" for t, n in v.items())
                              for k, v in sorted(convention.items(), key=lambda kv: -sum(kv[1].values())))
            lines.append(f"读者希望同类编号统一处理。考卷 60 章里其他“阿拉伯数字＋号”（不含本包要裁决的地方）的现有标法："
                         f"{marked or '没有'}。")
        lines.append("")
        for k, p in enumerate(ps + extra[gid], 1):
            ch = snapshot.chapters[p["chapter_id"]]
            i, pid, para, s, e = paragraph_of(ch, p["start"], p["end"])
            qid = f"{gid}-{k}"
            places.append({"id": qid, "no": a["no"], "chapter_id": p["chapter_id"], "start": p["start"],
                           "end": p["end"], "name": p["names"][0], "paragraph_id": pid,
                           "extra": bool(p.get("extra")), "fixed_type": a.get("fixed_type")})
            if pending:
                places[-1]["group"] = a["name"]
            lines.append(f"### {qid}（{p['chapter_id']}，{pid}）")
            if i > 0:
                lines.append(f"前一段：{ch.paragraphs[i - 1][1]}")
            lines.append(f"本段：{para[:s]}【{para[s:e]}】{para[e:]}")
            if i + 1 < len(ch.paragraphs):
                lines.append(f"后一段：{ch.paragraphs[i + 1][1]}")
            lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(ADJUDICATE_INSTRUCTIONS, out_dir / "instructions.md")
    shutil.copyfile(guide_file, out_dir / "guide.md")
    task = {"name": out_dir.name, "kind": "adjudicate", "snapshot_id": export["snapshot_id"], "instructions": "adjudicate_v1",
            "guide_version": export["guide_version"], "answers_config": answers_config,
            "places": places, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if pending:
        task.update({"set": export["set"], "listing": disagreements["gold_file"], "sheet_groups": sheet_groups})
    (out_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8")
    return places


def sheet_places(export, ids, snapshot, names, sheet):
    """Every occurrence in the test chapters of the given name-sheet rows (decided after the answers were made),
    as places. Marks and unscored ranges wholly inside the name (like 章 in 章医生) are replaced; anything that
    reaches past the name's edges stops the run, to be looked at (the sheet only fixes the whole string)."""
    rows = {r["name"]: r for r in sheet}
    out = []
    for name in names:
        row = rows[name]
        for cid in ids:
            chap = export["chapters"][cid]
            text = snapshot.chapters[cid].text
            at = text.find(name)
            while at >= 0:
                stop = at + len(name)
                touch = [x for x in chap["annotations"] + chap.get("unscored", []) if x["start"] < stop and at < x["end"]]
                wider = [x for x in touch if x["start"] < at or stop < x["end"]]
                if wider:
                    raise SystemExit(f"{name} in {cid} at {at}: {[x.get('text') for x in wider]} reach past it; decide that place by hand")
                out.append({"chapter_id": cid, "start": at, "end": stop, "text": name, "reading": row["type"],
                            "by": "user 2026-10-02", "decider": "sheet", "reason": f"名字知识表：{row['note']}"})
                at = text.find(name, stop)
    return out


def read_rulings(adjudication_dir, label):
    """The reviewer's rulings: (package dirs, the first package's task, {place id: ruling}, {place id: (date, dir)}).
    The first package is the full ruling; a later package rules again on some of the same places (the same ids
    and positions) and replaces the earlier ruling there."""
    dirs = adjudication_dir if isinstance(adjudication_dir, (list, tuple)) else [adjudication_dir]
    task, by_id, ruled = None, {}, {}
    for k, folder in enumerate(dirs):
        t = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        pass_file = folder / f"pass_{label}.json"
        ruling = {d.get("id"): d for d in json.loads(pass_file.read_text(encoding="utf-8")).get("decisions", [])}
        asked = {q["id"]: q for q in t["places"]}
        if set(ruling) != set(asked):
            raise SystemExit(f"{folder.name}: the ruling does not answer exactly the asked places: {sorted(set(ruling) ^ set(asked))}")
        if k == 0:
            task = t
        else:
            base = {q["id"]: q for q in task["places"]}
            moved = [i for i, q in asked.items() if i not in base or (base[i]["chapter_id"], base[i]["start"], base[i]["end"])
                     != (q["chapter_id"], q["start"], q["end"])]
            if moved:
                raise SystemExit(f"{folder.name}: places not in the first package as they are: {moved[:5]}")
        on = time.strftime("%Y-%m-%d", time.localtime(pass_file.stat().st_mtime))
        for i, d in ruling.items():
            by_id[i], ruled[i] = d, (on, folder.relative_to(REPO_ROOT).as_posix())
    return dirs, task, by_id, ruled


def ruled_reading(q, d):
    """(text, reading) of one reviewer ruling: a type marks its text, none removes, pending leaves it unscored."""
    kind = d.get("type")
    if kind not in ENTITY_TYPES and kind not in ("none", "pending"):
        raise SystemExit(f"{q['id']}: type {kind!r}")
    if q.get("fixed_type") and kind in ENTITY_TYPES and kind != q["fixed_type"]:
        raise SystemExit(f"{q['id']}: the reviewer gave {kind}, but the user fixed the type as {q['fixed_type']}; ask the user")
    return (d.get("text") or q["name"]) if kind in ENTITY_TYPES else q["name"], "unscored" if kind == "pending" else kind


def check_no_overlap(places):
    spans = sorted((p["chapter_id"], p["start"], p["end"]) for p in places)
    for x, y in zip(spans, spans[1:]):
        if x[0] == y[0] and y[1] < x[2]:
            raise SystemExit(f"two decided places overlap in {x[0]}: {x[1:]} and {y[1:]}")


def occurs_once(snapshot, p, name):
    text = snapshot.chapters[p["chapter_id"]].text
    return name and sum(text.startswith(name, i) for i in range(p["start"], p["end"] - len(name) + 1)) == 1


def pending_decisions(answers, disagreements, base_file, export, snapshot, adjudication_dir, label, answers_config,
                      sheet_groups="reviewer", memory=None):
    """The decisions for a pending listing (dev answers): every listed place gets exactly one reading -- the user's
    direct answer (decider user), the reviewer's ruling on the user's words or on the name sheet's row (reviewer;
    pending -> unscored; a ruling that gives a type without text, or frames a sheet name otherwise than as its whole
    string, stops the run), the sheet row itself (sheet, only with sheet_groups="sheet"), or an earlier ruling the
    user kept (a prior answer: the carried ruling's reading, string and decider, by "prior ruling: ...") -- and the
    chapters of the listing are finished; nothing else may change (only_chapters)."""
    chapters = disagreements["chapters"]
    groups = pending_groups(answers, disagreements, sheet_groups, memory)
    listed = sorted((p["chapter_id"], p["start"], p["end"]) for _, ps in groups for p in ps)
    marks = sorted((cid, a["start"], a["end"]) for cid in chapters for a in export["chapters"][cid]["annotations"] if a.get("pending"))
    if marks != listed:
        raise SystemExit("the listed places are not the file's pending marks; list them again (review pending)")
    places = []
    for a, ps in groups:
        if a["handling"] in ("direct", "prior"):
            # the other string of a boundary doubt or of a name-sheet row may be in no pass's span, so not in names
            if a["reading"] in ENTITY_TYPES and not a.get("text") and any(
                    len(p["names"]) > 1 or p.get("sheet_says") or "boundary" in p["stuck"] for p in ps):
                raise SystemExit(f"answer {a['no']}: its places hold several strings, a name-sheet reading or a boundary "
                                 f"doubt ({a['name']}); write the one the answer names as text, or give the group to the reviewer")
            prior = a["handling"] == "prior"  # an earlier ruling kept: decided as it was, not by the user's date
            for p in ps:
                name = a.get("text") or p["names"][0]
                if not occurs_once(snapshot, p, name):
                    raise SystemExit(f"answer {a['no']}: {name} is not once in the place {p['chapter_id']} {p['start']}-{p['end']}")
                places.append({"chapter_id": p["chapter_id"], "start": p["start"], "end": p["end"], "text": name,
                               "reading": a["reading"],
                               "by": f"{PRIOR_BY}{a['prior']['carry_over_from']}" if prior else f"user {answers['answered_at']}",
                               "decider": a["prior"]["decider"] if prior else "user",
                               "reason": f"回答表 {a['no']}：{a['answer']}" + (f"；补充：{a['note2']}" if a.get("note2") else "")
                               + (f"；以前的裁决：{READING_ZH[a['reading']]}（{a['prior']['carry_over_from']}）" if prior else "")})
        elif a["handling"] == "sheet":
            for p in ps:
                places.append({"chapter_id": p["chapter_id"], "start": p["start"], "end": p["end"], "text": a["sheet_row"]["name"],
                               "reading": a["sheet_row"]["type"], "by": f"user {disagreements['created_at'][:10]}",
                               "decider": "sheet", "reason": f"名字知识表：{a['answer']}"})
    reviewer = {group_id(a): (a, ps) for a, ps in groups if a["handling"] == "reviewer"}
    dirs = [] if not adjudication_dir else adjudication_dir if isinstance(adjudication_dir, (list, tuple)) else [adjudication_dir]
    if bool(reviewer) != bool(dirs):
        raise SystemExit("the groups that go to the reviewer need its ruling (package, codex_run, then --adjudication)"
                         if reviewer else "no group goes to the reviewer; leave out --adjudication")
    if reviewer:
        dirs, task, by_id, ruled = read_rulings(dirs, label)
        want = sorted((p["chapter_id"], p["start"], p["end"]) for _, ps in reviewer.values() for p in ps)
        if want != sorted((q["chapter_id"], q["start"], q["end"]) for q in task["places"]):
            raise SystemExit(f"{dirs[0].name} does not hold exactly the places of the groups that go to the reviewer; package again")
        for q in task["places"]:
            a, _ = reviewer[q["id"].rsplit("-", 1)[0]]
            d = by_id[q["id"]]
            if d.get("type") in ENTITY_TYPES and not d.get("text"):
                raise SystemExit(f"{q['id']}: the reviewer gave {d['type']} but no text; ask it again (the code picks no span)")
            ruled_on, ruled_in = ruled[q["id"]]
            text, reading = ruled_reading(q, d)
            sheet = bool(a.get("sheet_row"))
            if sheet and reading in ENTITY_TYPES and text != a["sheet_row"]["name"]:
                raise SystemExit(f"{q['id']}: the reviewer framed {text}, but the name sheet fixes the whole string "
                                 f"{a['sheet_row']['name']}; ask the user")
            words = (f"名字知识表：{a['answer']}" if sheet else f"用户原话：{a['answer']}"
                     + (f"；补充：{a['note2']}" if a.get("note2") else ""))
            places.append({"chapter_id": q["chapter_id"], "start": q["start"], "end": q["end"], "text": text, "reading": reading,
                           "by": f"{label} {ruled_on}（按规则 v{export['guide_version']}，依据用户的{'名字知识表' if sheet else '说明'}）",
                           "decider": "reviewer",
                           "reason": f"{d.get('rule', '')}：{d.get('reason', '')}（{q['id']}，{ruled_in}；{words}）"})
    if sorted((p["chapter_id"], p["start"], p["end"]) for p in places) != listed:
        raise SystemExit("the decisions do not cover every listed place exactly once")
    check_no_overlap(places)
    kept = Counter(p["decider"] for p in places if p["by"].startswith(PRIOR_BY))
    who = Counter(p["decider"] for p in places if not p["by"].startswith(PRIOR_BY))
    models = sorted({r["model"] for runs in export.get("llm_drafts", {}).values() for r in runs.values()})
    return {"description": f"{export['set']} answers: the {len(chapters)} chapters labelled twice by {', '.join(models)} and merged "
            f"by adjudicate.py prefill ({base_file}). Each of the {len(listed)} places the two passes left pending is decided "
            f"once: the user's direct answers ({answers_config}; {who['user']} places), the reviewer {label}'s rulings on the "
            f"hedged answers and on the name-sheet names' places with the sheet's row ({who['reviewer']} places; "
            + (", ".join(f.relative_to(REPO_ROOT).as_posix() for f in dirs) if reviewer else "none")
            + f"; by guide v{export['guide_version']}; what it could not decide is unscored)"
            + (f", the name sheet's rows ({who['sheet']} places)" if who["sheet"] else "")
            + (f", earlier rulings the user kept ({sum(kept.values())} places, by 'prior ruling: ...' with the carried "
               f"ruling's decider: {', '.join(f'{d} {n}' for d, n in sorted(kept.items()))}; the answers file's prior "
               "entries name the deciders and dates)" if kept else "")
            + ". The listed chapters are then marked done; the other chapters stay as they are. Applied mechanically by "
            "gold.py revise; Claude decided no label.",
            "base": base_file, "base_sha256": hashlib.sha256((REPO_ROOT / base_file).read_bytes()).hexdigest(),
            "decided_by": "user", "decided_at": answers["answered_at"],
            "only_chapters": chapters, "finish_chapters": chapters, "places": places}


def build_decisions(answers, disagreements, base_file, export, ids, snapshot, sheet_names, adjudication_dir, label,
                    sheet=None, answers_config=None, sheet_groups="reviewer", memory=None):
    """The v3 decisions: the user's direct answers on their listed places, the newer sheet rows everywhere in the
    test chapters, and the reviewer's rulings on the hedged places (pending -> unscored). A pending listing (dev
    answers) goes to pending_decisions."""
    if disagreements.get("mode") == "pending":
        if sheet_names:
            raise SystemExit("a pending listing takes no --sheet-names: the prefill applied the name sheet")
        return pending_decisions(answers, disagreements, base_file, export, snapshot, adjudication_dir, label,
                                 answers_config, sheet_groups, memory)
    places = []
    asked_places = places_of(disagreements["asked"].values())
    for a in answers["answers"]:
        if a["handling"] != "direct":
            continue
        ps = disagreements["asked"].get(a["name"], [])
        if len(ps) != a["places"]:
            raise SystemExit(f"answer {a['no']} {a['name']}: expected {a['places']} places, found {len(ps)}")
        if a.get("scope") == "all":  # the user said the answer holds for every use of the name in the test chapters
            skip = {(x["chapter_id"], x["at"]) for x in a.get("exclude", [])}
            others = [o for o in other_occurrences(ps[0]["names"][0], ids, snapshot, export, asked_places)
                      if not o["inside_longer_mark"]]
            if not skip <= {(o["chapter_id"], o["at"]) for o in others}:
                raise SystemExit(f"answer {a['no']}: an excluded place is not among the other uses")
            ps = ps + [{**o, "names": [ps[0]["names"][0]]} for o in others if (o["chapter_id"], o["at"]) not in skip]
        for p in ps:
            places.append({"chapter_id": p["chapter_id"], "start": p["start"], "end": p["end"], "text": p["names"][0],
                           "reading": a["reading"], "by": f"user {answers['answered_at']}", "decider": "user",
                           "reason": f"回答表 {a['no']}：{a['answer']}" + (f"；补充：{a['note2']}" if a.get("note2") else "")})
    # the sheet is read only when rows are to be applied (a test may pass its own sheet)
    places += sheet_places(export, ids, snapshot, sheet_names,
                           (load_knowledge() if sheet is None else sheet) if sheet_names else [])
    dirs, task, by_id, ruled = read_rulings(adjudication_dir, label)
    note = {a["no"]: a["answer"] for a in answers["answers"]}
    note2 = {a["no"]: a.get("note2") for a in answers["answers"]}
    for q in task["places"]:
        d = by_id[q["id"]]
        ruled_on, ruled_in = ruled[q["id"]]
        text, reading = ruled_reading(q, d)
        places.append({"chapter_id": q["chapter_id"], "start": q["start"], "end": q["end"], "text": text, "reading": reading,
                       "by": f"{label} {ruled_on}（按规则 v1.1，依据用户的说明）", "decider": "reviewer",
                       "reason": f"{d.get('rule', '')}：{d.get('reason', '')}（{q['id']}，{ruled_in}；用户原话：{note[q['no']]}"
                                 + (f"；补充：{note2[q['no']]}" if note2.get(q["no"]) else "") + "）"})
    check_no_overlap(places)
    return {"description": "Answer v3: the user's answers (configs/review_astra_answers.json) on the places where GPT-6-Astra "
            "and answer v2 disagreed, the name-sheet rows added 2026-10-02 (everywhere in the test chapters), and GPT-6-Astra's "
            f"rulings on the hedged answers ({', '.join(f.relative_to(REPO_ROOT).as_posix() for f in dirs)}, later packages "
            "replacing earlier rulings on the same places; by guide v1.1 on the user's "
            "notes; what it could not decide is unscored). Applied mechanically by gold.py revise; Claude decided no label.",
            "base": base_file, "base_sha256": hashlib.sha256((REPO_ROOT / base_file).read_bytes()).hexdigest(),
            "decided_by": "user", "decided_at": answers["answered_at"], "places": places}


def pending_metrics(found, asked, by_sheet, export, ids, carried=None, prior=None):
    """The counts of a pending listing; the memory's two counts only when a memory was used (carried, prior given)."""
    anns = [a for cid in ids for a in export["chapters"][cid]["annotations"]]
    kinds = Counter(k for p in found for k in p["kinds"])
    out = {"chapters": len(ids),
           "chapters_still_todo": sum(1 for c in export["chapters"].values() if c.get("status") == "todo"),
           "accepted_marks": sum(1 for a in anns if not a.get("pending")), "pending_places": len(found),
           "places_by_kind": dict(kinds), "places_made_by_the_sheet": sum(1 for p in found if "sheet" in p["kinds"]),
           "name_groups_to_ask": len(asked), "places_to_ask": sum(len(v) for v in asked.values()),
           "name_groups_by_sheet": len(by_sheet), "places_by_sheet": sum(len(v) for v in by_sheet.values())}
    if carried is not None:
        out.update({"groups_carried_over": len(carried), "places_carried_over": sum(len(v) for v in carried.values()),
                    "groups_asked_with_prior_answer": len(prior or {})})
    out["note"] = "the places the two GPT passes left pending in the prefilled chapters; nothing was changed"
    return out


def list_pending(gold, set_name, label, run_id, root=REPO_ROOT, snapshot=None, task_root=TASK_ROOT, sheet=None, memory=None):
    """review pending: the pending places of a prefilled export as questions per name (see the module docstring).
    gold is the prefilled file's path relative to root; memory, the answer memory's path relative to root (optional:
    without it the outputs are exactly as before). Returns (found, asked, by_sheet)."""
    root = Path(root)
    gold_file = own_set_path(root / gold, set_name, root)
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    if export.get("set") != set_name:
        raise SystemExit(f"{gold} is a {export.get('set')} export, not {set_name}")
    snapshot = snapshot or load_snapshot(export["snapshot_id"])
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the prefilled file does not pass the gold check: {errors[:3]}")
    ids = pending_chapters(export)
    if not ids:
        raise SystemExit(f"{gold} has no prefilled chapters to finish (provenance model_double_adjudicated, not done)")
    out_dir = root / "data" / "gold" / set_name / f"review_{label}"
    if (out_dir / "disagreements.json").exists():
        raise SystemExit(f"{out_dir} already holds a listing; not overwriting it")
    found = collect_pending(export, ids, snapshot, pass_readings(export, task_root))
    sheet = load_knowledge() if sheet is None else sheet
    rows = {r["name"]: r for r in sheet}
    asked, by_sheet = by_name(found, set(rows))
    carried = carry = prior = None
    if memory is not None:
        mem = load_memory(root / memory)
        asked, carried, carry, prior = split_by_memory(asked, mem["entries"])
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = {"mode": "pending", "set": set_name, "gold_file": gold, "gold_sha256": sha256(gold_file), "label": label,
               "run_id": run_id, "created_at": created, "chapters": ids,
               "sheet_rows": {ps[0]["names"][0]: rows[ps[0]["names"][0]] for ps in by_sheet.values()},
               "asked": asked, "by_sheet": by_sheet}
    if memory is not None:
        listing.update({"memory": {"path": memory, "sha256": sha256(root / memory), "generated_at": mem.get("generated_at")},
                        "carried": carried, "carry": carry, "prior": prior})
    (out_dir / "disagreements.json").write_text(json.dumps(listing, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "questions.md").write_text(questions_md(asked, by_sheet, label, run_id, PENDING_WORDING, carried=carried,
                                                       carry=carry, prior=prior), encoding="utf-8")
    (out_dir / "answer_sheet.md").write_text(answer_sheet_md(asked, label, run_id, PENDING_WORDING, by_sheet, carried=carried,
                                                             carry=carry), encoding="utf-8")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "action": "pending", "set": set_name, "gold_file": gold, "gold_sha256": sha256(gold_file),
        "label": label, "chapters": ids,
        "drafts": {t: {r: sha256(Path(task_root) / t / f"draft_{r}.json") for r in ("A", "B")} for t in export.get("llm_drafts", {})},
        "name_sheet_rows": len(sheet), **({"memory": listing["memory"]} if memory is not None else {}),
        "code": git_state(), "created_at": created}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"run_id": run_id, **pending_metrics(found, asked, by_sheet, export, ids, carried, prior)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return found, asked, by_sheet


# the test review's defaults (answers v3); the dev answers name every path explicitly
TEST_DEFAULTS = {"answers": "configs/review_astra_answers.json", "package_out": "data/llm_tasks/adjudicate_v3",
                 "adjudication": ["data/llm_tasks/adjudicate_v3"], "sheet_names": ["章医生", "武博士"],
                 "decide_out": "configs/gold_test_v3_decisions.json"}


def resolve(args):
    """Fill the options left out: the test review's paths for --set test; for another set they must be given."""
    if args.action not in ("package", "decide"):
        return args
    test = args.set_name == "test"
    out_key = "package_out" if args.action == "package" else "decide_out"
    for opt, key in (("answers", "answers"), ("out", out_key)):
        if getattr(args, opt) is None:
            if not test:
                raise SystemExit(f"--set {args.set_name}: give --{opt}")
            setattr(args, opt, TEST_DEFAULTS[key])
    if args.action == "decide":
        if args.adjudication is None:
            args.adjudication = TEST_DEFAULTS["adjudication"] if test else []
        if args.sheet_names is None:
            args.sheet_names = TEST_DEFAULTS["sheet_names"] if test else []
    return args


def parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--gold", required=True)
    d.add_argument("--label", default="astra")
    d.add_argument("--run-id", required=True)
    n = sub.add_parser("pending", help="the pending places of a prefilled export as questions per name")
    n.add_argument("--prefilled", required=True, help="the last prefilled file (adjudicate.py), relative to the repo")
    n.add_argument("--label", default="astra", help="names the folder review_<label> (the reviewer's label)")
    n.add_argument("--run-id", required=True)
    n.add_argument("--memory", default=None, help=f"the answer memory ({MEMORY_OUT}): groups ruled before are not asked")
    m = sub.add_parser("memory", help=f"build {MEMORY_OUT} from the earlier answers, decisions and the name sheet")
    m.add_argument("--answers", nargs="*", default=[], help=f"answers files to add to {' '.join(MEMORY_ANSWERS)} ([set=]path)")
    m.add_argument("--decisions", nargs="*", default=[], help=f"decisions files to add to {' '.join(MEMORY_DECISIONS)} ([set=]path)")
    m.add_argument("--sheet", default=None, help="the name sheet (default data/gold/name_knowledge.tsv)")
    m.add_argument("--out", default=MEMORY_OUT)
    h = sub.add_parser("check-answers", help="a filled answer sheet against the memory: answer_check.md, counts in runs/")
    h.add_argument("--sheet", required=True, help="the filled answer_sheet.md, relative to the repo")
    h.add_argument("--listing", required=True, help="the disagreements.json the sheet was made from")
    h.add_argument("--memory", default=MEMORY_OUT)
    h.add_argument("--run-id", required=True)
    h.add_argument("--set", dest="set_name", choices=SETS, required=True)
    k = sub.add_parser("package", help="the adjudication package for the hedged answers")
    c = sub.add_parser("decide", help="write the decisions file for gold.py revise (test default: answers v3)")
    for x in (k, c):
        x.add_argument("--gold", required=True, help="the answers listed against (pending: the prefilled file)")
        x.add_argument("--answers", default=None, help=f"the user's answers (test default {TEST_DEFAULTS['answers']})")
        x.add_argument("--label", default="astra")
        x.add_argument("--sheet-groups", choices=SHEET_GROUPS, default="reviewer",
                       help="pending listings: what decides the name-sheet groups the user did not answer")
        x.add_argument("--memory", default=MEMORY_OUT,
                       help="the answer memory a prior answer is checked against when the listing did not carry its group over")
    for x in (d, n, k, c):
        x.add_argument("--set", dest="set_name", choices=SETS, default="test")
    k.add_argument("--out", default=None, help=f"the package folder (test default {TEST_DEFAULTS['package_out']})")
    k.add_argument("--only", nargs="*", type=int, help="answer numbers to package again")
    c.add_argument("--adjudication", nargs="+", default=None,
                   help="the first package, then any later ones whose rulings replace it on the same places "
                        f"(test default {TEST_DEFAULTS['adjudication'][0]})")
    c.add_argument("--sheet-names", nargs="*", default=None,
                   help=f"test: name-sheet rows to apply everywhere (default {' '.join(TEST_DEFAULTS['sheet_names'])})")
    c.add_argument("--out", default=None, help=f"the decisions file (test default {TEST_DEFAULTS['decide_out']})")
    return ap


def main():
    args = resolve(parser().parse_args())
    if args.action == "diff" and args.set_name in NEW_SETS:
        raise SystemExit(f"{args.set_name} has no earlier answers to diff against; its answers take the pending route "
                         "(review pending, package, decide; see the runbook)")
    if args.action == "memory":
        memory = build_memory([*MEMORY_ANSWERS, *args.answers], [*MEMORY_DECISIONS, *args.decisions], args.sheet)
        (REPO_ROOT / args.out).write_text(json.dumps(memory, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{memory['counts']['entries']} strings, {memory['counts']['with_carry_over']} with a carry-over, "
              f"{memory['counts']['with_mixed_rulings']} with mixed rulings -> {args.out}")
        return
    if args.action == "check-answers":
        rows, metrics = check_answers(args.set_name, args.sheet, args.listing, args.memory, args.run_id)
        print(f"{len(rows)} rows: " + ", ".join(f"{k} {n}" for k, n in metrics["verdicts"].items())
              + f"; wrote answer_check.md next to the sheet and runs/{args.run_id}")
        return
    if args.action == "pending":
        found, asked, by_sheet = list_pending(args.prefilled, args.set_name, args.label, args.run_id, memory=args.memory)
        print(f"{len(found)} pending places, {len(asked)} name groups to ask, {len(by_sheet)} name-sheet groups; "
              f"wrote data/gold/{args.set_name}/review_{args.label}")
        return
    if args.action in ("package", "decide"):
        export = json.loads(own_set_path(REPO_ROOT / args.gold, args.set_name, REPO_ROOT).read_text(encoding="utf-8"))
        if export.get("set") != args.set_name:
            raise SystemExit(f"{args.gold} is a {export.get('set')} export, not {args.set_name}")
        snapshot = load_snapshot(export["snapshot_id"])
        ids = done_chapters(export)
        answers = json.loads((REPO_ROOT / args.answers).read_text(encoding="utf-8"))
        disagreements = json.loads((REPO_ROOT / "data" / "gold" / args.set_name / f"review_{args.label}" / "disagreements.json")
                                   .read_text(encoding="utf-8"))
        if disagreements["gold_file"] != args.gold:
            raise SystemExit(f"the disagreements were listed against {disagreements['gold_file']}")
        if disagreements.get("mode") == "pending":
            if sha256(REPO_ROOT / args.gold) != disagreements["gold_sha256"]:
                raise SystemExit(f"{args.gold} changed since the pending places were listed; list them again")
            ids = disagreements["chapters"]
        elif args.set_name != "test":
            raise SystemExit("for a set other than test, package and decide take a pending listing (review pending)")
        memory = None  # read only when a prior answer sits on a group the listing did not carry over
        if any(a.get("handling") == "prior" and a["name"] not in disagreements.get("carry", {}) for a in answers["answers"]):
            memory = load_memory(REPO_ROOT / args.memory)["entries"]
        if args.action == "package":
            out = REPO_ROOT / args.out
            if args.set_name != "test" and (out / "task.json").exists():
                raise SystemExit(f"{args.out} already holds a package; pick a new folder (packages are never overwritten)")
            places = build_package(answers, disagreements, export, ids, snapshot, out,
                                   TASK_ROOT / f"{args.set_name}_b01" / "guide.md", only=set(args.only) if args.only else None,
                                   answers_config=args.answers, sheet_groups=args.sheet_groups, memory=memory)
            print(f"{len(places)} places in {args.out}")
        else:
            decisions = build_decisions(answers, disagreements, args.gold, export, ids, snapshot, args.sheet_names,
                                        [REPO_ROOT / a for a in args.adjudication], args.label,
                                        answers_config=args.answers, sheet_groups=args.sheet_groups, memory=memory)
            (REPO_ROOT / args.out).write_text(json.dumps(decisions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{len(decisions['places'])} places -> {args.out}")
        return

    gold_file = own_set_path(REPO_ROOT / args.gold, args.set_name, REPO_ROOT)
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    if export.get("set") != args.set_name:
        raise SystemExit(f"{args.gold} is a {export.get('set')} export, not {args.set_name}")
    snapshot = load_snapshot(export["snapshot_id"])
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the answers do not pass the gold check: {errors[:3]}")
    ids = done_chapters(export)
    reviews, packages = {}, {}
    for task_dir in sorted(TASK_ROOT.glob(f"{args.set_name}_b*")):
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        pass_file = task_dir / f"pass_{args.label}.json"
        if not pass_file.exists():
            raise SystemExit(f"{task_dir.name} has no {pass_file.name} yet")
        converted = convert(json.loads(pass_file.read_text(encoding="utf-8")), snapshot, task["chapters"])
        (task_dir / f"review_{args.label}.json").write_text(json.dumps(
            {"task": task["name"], "run": args.label, "source_file": pass_file.name, "chapters": converted},
            ensure_ascii=False, indent=1), encoding="utf-8")
        spans = draft_spans({"chapters": converted}, task["chapters"])
        for cid in task["chapters"]:
            doubts = [s for s in converted[cid]["spans"] if s["pending"]]
            reviews[cid] = {**spans[cid], "doubts": doubts}
            packages[cid] = {"task": task["name"], "answer_sha256": sha256(pass_file), "unlocated": spans[cid]["errors"]}
    if set(ids) - set(reviews):
        raise SystemExit(f"no review for {sorted(set(ids) - set(reviews))}")
    origin, _ = gold_origin(export, ids)
    found = collect(export, ids, snapshot, reviews, origin)
    sheet = load_knowledge()
    asked, by_sheet = by_name(found, {r["name"] for r in sheet})

    out_dir = REPO_ROOT / "data" / "gold" / args.set_name / f"review_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "disagreements.json").write_text(json.dumps(
        {"gold_file": args.gold, "label": args.label, "run_id": args.run_id, "asked": asked, "by_sheet": by_sheet},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "questions.md").write_text(questions_md(asked, by_sheet, args.label, args.run_id), encoding="utf-8")
    (out_dir / "answer_sheet.md").write_text(answer_sheet_md(asked, args.label, args.run_id), encoding="utf-8")

    gold = gold_spans(export, ids)
    vs = strict(gold, {c: {"spans": reviews[c]["spans"], "pending": reviews[c]["pending"], "errors": reviews[c]["errors"]}
                       for c in ids})
    kinds = defaultdict(int)
    for p in found:
        for k in p["kinds"]:
            kinds[k] += 1
    run_dir = REPO_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "gold_file": args.gold, "gold_sha256": sha256(gold_file), "label": args.label,
        "packages": packages, "name_sheet_rows": len(sheet), "code": git_state(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": args.run_id, "chapters": len(ids),
        "reviewer_vs_answers": {k: vs[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
        "reviewer_vs_answers_per_type": vs["per_type"],
        "reviewer_unlocated": sum(r["errors"] for r in reviews.values()),
        "reviewer_pending_marks": sum(len(r["pending"]) for r in reviews.values()),
        "disagreeing_places": len(found), "places_by_kind": dict(kinds),
        "places_where_the_user_had_decided": sum(p["answer_decided_by_user"] for p in found),
        "name_groups_to_ask": len(asked), "places_to_ask": sum(len(v) for v in asked.values()),
        "name_groups_by_sheet": len(by_sheet), "places_by_sheet": sum(len(v) for v in by_sheet.values()),
        "places_on_gpt_agreed_answer_spans": sum(1 for p in found if p["answer"] and not p["answer_decided_by_user"]),
        "note": "the reviewer against the current answers; neither is ground truth here; nothing was changed",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(found)} places, {len(asked)} name groups to ask, {len(by_sheet)} by the sheet; wrote {out_dir}")


if __name__ == "__main__":
    main()
