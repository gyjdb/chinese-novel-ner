"""Error analysis of a scoring run: 误标 (false positives), 漏标 (false negatives) and 待审 (abstentions), each place
put in a category by a fixed mechanical rule.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.errors configs/errors_v9.json [--run-id ID] [--inputs-root PATH] [--dry-run]

No one judges whether any answer is right. Claude wrote this program and must not judge whether an answer is right (Opus
is one of the compared models), so every category below comes from the rules alone, applied by the program. "False
positive" and "false negative" are scoring terms relative to the answer file (the reference): a place where a system and
the answer file disagree counts against the system. A possible problem in the answer file is never called an error of the
answer; the "gold_name_elsewhere" kind only says that the same string is an answer somewhere else.

Timing: the categories, their order and their sub-kinds were written after the v9 scores were seen (the study plan named
only example kinds: generic words taken as names, a title inside the boundary, new items not recognised). They describe;
they are not a pre-registered test. Only the per-occurrence new-name recall was set in advance.

Inputs. The scoring run named by "eval_run": its config.json gives the scoring config, the answer file (checked against
the sha256 it recorded), the snapshot and the test chapters; its metrics.json gives the totals that every count here
must reproduce; its predictions_test_<system>.jsonl are the predictions, each checked before use:
- trained span systems: the file's sha256 is the one the scoring run recorded (predictions_sha256);
- GLiNER: the rows flagged kept are exactly those scoring above the recorded threshold, as many as the run counted;
- the dictionary: the file equals dictionary matching recomputed from the snapshot with the scoring config's min_len;
- generative systems (Opus, Sonnet): their answers (data/llm_tasks/<task>/pass_<label>.json) are read again with
  evaluate.llm_predictions, task by task as evaluate does (answers checked against the call log's sha256 the same way),
  because the saved predictions hold no pending marks and no invalid outputs; the recomputed located predictions must
  equal the saved file row for row.
--dry-run computes and checks everything, builds the report in memory and prints counts only; it writes nothing.
--inputs-root reads the inputs (eval run, data/) from another checkout of this repository and works only with --dry-run:
outputs go to the checkout the code runs from, and report.md would land in that checkout's git-ignored data/, which a
plain `git worktree remove` deletes without warning, leaving the committed metrics.json pointing at a report that is
gone. A real run is made from the checkout that holds data/, so it records that checkout's code state and the report
stays next to its inputs.

Scoring regions are evaluate's: a gold or predicted span touching one of the answer file's unscored ranges or pending
marks counts for nothing (score.gold_spans, score._touches); exact duplicate predictions count once
(evaluate.as_prediction). Then, per system and chapter, with G the scored gold spans and P the scored predicted spans
(start, end, type), "overlap" meaning at least one shared character and gold positions taken as their distinct
(start, end):

误标 (false positive): each p in P that is not in G; the first rule that applies:
  1. type_error: a gold span in G has p's start and end (so it has another type).
  2. boundary: p overlaps a gold span in G. Sub-kind, the first that applies:
       merged: p contains two or more gold spans;
       wider_title: p contains exactly one gold span g, overlaps no other, starts where g starts, and the rest of p
         (from g's end to p's end) is one of title_suffixes;
       wider_other: p contains exactly one gold span and overlaps no other (and is not wider_title);
       narrower: p lies inside one gold span;
       crossing: anything else (p partly overlaps a gold span, neither containing it nor lying inside it).
  3. spurious: p overlaps no gold span in G. Sub-kind, the first that applies:
       gold_name_elsewhere: p's text is the text of a scored gold span somewhere in the test chapters (any type);
       generic_word: p's text is one of generic_words (the list of the annotation guide's rule A1, copied exactly and
         checked against the guide when the program runs);
       category_word: p's text is one of category_words or matches grade_category_pattern (configs/weak_v2.json);
       single_char: p's text is one character;
       other.
  Invalid outputs (a generative system's mention with no position or no valid type) are counted on their own: they
  are false positives in the totals, as in evaluate, and have no type and no place.

漏标 (false negative): each g in G that is not in P; the first rule that applies:
  1. wrong_prediction_overlaps: a span in P overlaps g (with flat gold this is a false positive: the other side of a
     type or boundary error);
  2. abstained (待审): no span in P overlaps g and a pending mark of the system (its located place) overlaps g; only the
     generative systems can mark a place pending, the taggers cannot;
  3. missed: nothing of the system overlaps g. A prediction removed by an unscored range counts as nothing.
  Every false-negative count is split new vs seen by the pre-registered seen rule (a gold span is new when its exact
  text is neither a dictionary name nor a mention name of chapters <= 780: novelty.weak_names, any type); missed spans
  are also counted when they are one character.

待审 on the answer side: the answer file's unscored ranges (how many, and how many predictions of each system they
removed) and its pending marks (expected 0 in the final answers).

Pending places of a system (generative systems only): each located pending mark is put in the first category that
applies: touches_an_unscored_range (it shares a character with one of the answer file's unscored ranges or pending
marks); overlaps_a_scored_gold_span (it overlaps a span in G); overlaps_no_scored_gold_span. A place filed under
touches_an_unscored_range can still make an overlapping scored gold span "abstained", as in score.strict, so
overlaps_a_scored_gold_span need not equal the abstained false negatives.

Per type: every count is split by type, the predicted type for false positives and the gold type for false negatives.

Checks (the run stops if one fails): per system, the TP/FP/FN of every chapter and type equal the scoring run's
metrics.json, as do the micro P/R/F1, the invalid outputs, the predictions the unscored ranges removed, score.strict's
fp_same_span_wrong_type (= type_error minus those where the gold span at that place was also found) and
fn_where_draft_was_pending (= abstained plus wrong_prediction_overlaps with a pending mark too), and the pre-registered
new-name recall (per occurrence) of every system and model; every category sums to its total, checked on every system
and on every model's summed counts (exact integers over its runs). A model's mean per run is summed / n_runs: the report
shows it with each cell rounded to 2 decimals on its own, so a row of means can miss its total by 0.01; it is not checked
and not written to metrics.json (only the summed counts are).

New names, two ways. Per occurrence (pre-registered; evaluate's new_name_recall, re-derived here and
checked): recall over the new gold spans. Per distinct name (post hoc, chosen after the scores were seen): over the new
gold spans, the distinct (text, type) names and the distinct texts; a name (text) is found if the system has at least
one exact true positive on a gold span with that name (text). For a model with several runs (seeds): the mean over its
runs of each run's share, and the range; never "found by any run".

Outputs: runs/<run_id>/config.json (inputs with sha256, the rules in words, code git state) and metrics.json (numbers
only: no string from the book; checked) are committed; data/analysis/<run_id>/report.md holds strings with short
contexts from the novel and is not committed. An existing output is never overwritten.
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import _prf_defined, as_prediction, dictionary_predictions, done_chapters, llm_predictions, sha256
from novel_ner.gold import check as check_gold
from novel_ner.novelty import weak_names
from novel_ner.score import _context, _touches, gold_spans
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.zeroshot import git_state

FP_KINDS = ("type_error", "boundary", "spurious")
FP_SUB = {"boundary": ("merged", "wider_title", "wider_other", "narrower", "crossing"),
          "spurious": ("gold_name_elsewhere", "generic_word", "category_word", "single_char", "other")}
FN_KINDS = ("wrong_prediction_overlaps", "abstained", "missed")
PENDING_PLACES = ("touches_an_unscored_range", "overlaps_a_scored_gold_span", "overlaps_no_scored_gold_span")

RULES = {
    "authority": ("every category is assigned by the fixed mechanical rules below, applied by the program; nobody judged "
                  "whether an answer is right (Claude wrote the program and must not: Opus is a compared model). False "
                  "positive / negative are scoring terms relative to the answer file; a possible problem in the answer "
                  "file is never called an error of the answer"),
    "regions": ("evaluate's: a gold or predicted span touching an unscored range or a gold pending mark counts for "
                "nothing; exact duplicate predictions count once. G = scored gold spans, P = scored predicted spans of a "
                "chapter; overlap = at least one shared character; gold positions = distinct (start, end)"),
    "false_positive": ("each p in P not in G, first rule that applies: type_error = a gold span has p's start and end; "
                       "boundary = p overlaps a gold span (merged: contains two or more gold spans; wider_title: contains "
                       "exactly one gold span g and overlaps no other, starts at g's start, and p's text after g's end is a "
                       "title suffix; wider_other: contains exactly one and overlaps no other; narrower: inside one gold "
                       "span; crossing: anything else); spurious = overlaps no gold span (gold_name_elsewhere: p's text "
                       "is the text of a scored gold span anywhere in the test chapters, any type; generic_word: in the "
                       "guide's A1 list; category_word: a category word or matching the grade+category pattern; "
                       "single_char: one character; other). Invalid outputs (no position) are counted on their own"),
    "false_negative": ("each g in G not in P, first rule that applies: wrong_prediction_overlaps = a span in P overlaps g; "
                       "abstained (pending) = no span in P overlaps g and a pending mark of the system overlaps g; missed = "
                       "nothing overlaps g (a prediction removed by an unscored range counts as nothing). Split new / seen "
                       "by the pre-registered seen rule; missed one-character spans counted too"),
    "answer_side_pending": "the answer file's unscored ranges (count, predictions removed per system) and pending marks",
    "system_pending_places": ("generative systems only: each located pending mark of a system, first that applies: "
                              "touches_an_unscored_range (shares a character with an unscored range or a gold pending mark); "
                              "overlaps_a_scored_gold_span (overlaps a span in G); overlaps_no_scored_gold_span. A place "
                              "touching an unscored range can still make an overlapping scored gold span abstained (as in "
                              "score.strict), so overlaps_a_scored_gold_span need not equal the abstained false negatives"),
    "per_type": "every count split by type: predicted type for false positives, gold type for false negatives",
    "new_names": ("per occurrence (pre-registered, the study plan) = evaluate's new_name_recall, re-derived and checked; per "
                  "distinct name (post hoc): over the new gold spans' distinct (text, type) names and distinct texts, a "
                  "name is found when the system has at least one exact true positive on it; a model: mean over runs of "
                  "each run's share, and the range"),
    "model_counts": ("a model's counts are summed over its runs (exact integers, checked like a run's); the mean per run is "
                     "summed / n_runs, shown in the report with each cell rounded to 2 decimals on its own (a row of means "
                     "can miss its total by 0.01) and not stored"),
    "timing": ("the categories, their order and their sub-kinds were written after the v9 scores were seen (the study plan "
               "named only example kinds: generic words taken as names, a title inside the boundary, new items not "
               "recognised); descriptive only, not a pre-registered test; only the per-occurrence new-name recall "
               " was set in advance"),
}


# ---------------------------------------------------------------------------------------------- the rules themselves

def a1_words(guide_text):
    """The words of the annotation guide's rule A1, in order: the quoted list in the rule's 选择 line."""
    m = re.search(r"^### A1[^\n]*\n(.*?)^### ", guide_text, re.S | re.M)
    if not m:
        raise SystemExit("the guide has no section A1")
    line = next((x for x in m.group(1).splitlines() if x.startswith("**选择：**")), None)
    quoted = re.search(r"“([^”]+)”", line or "")
    if not quoted:
        raise SystemExit("the guide's A1 has no quoted list in its 选择 line")
    return quoted.group(1).split("、")


def make_rules(title_suffixes, generic_words, category_words, grade_pattern, gold_texts, seen):
    return {"title_suffixes": frozenset(title_suffixes), "generic_words": frozenset(generic_words),
            "category_words": frozenset(category_words), "grade_pattern": re.compile(grade_pattern),
            "gold_texts": frozenset(gold_texts), "seen": frozenset(seen)}


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def classify_chapter(gold, pred, pending, text, rules, removed=()):
    """One chapter of one system, by the module's rules.

    gold, pred: sets of scored (start, end, type) (the unscored regions already left out); pending: the system's located
    pending places [(start, end)]; removed: predicted spans the unscored regions removed. Returns (false positives,
    false negatives), each a list of dicts with the place, its text and its category."""
    at = defaultdict(set)  # gold position -> its types
    for s, e, t in gold:
        at[(s, e)].add(t)
    fps, fns = [], []
    for s, e, t in sorted(pred - gold):
        x = text[s:e]
        over = sorted(b for b in at if _overlaps(b, (s, e)))
        rec = {"start": s, "end": e, "type": t, "text": x, "gold": [(b[0], b[1], u) for b in over for u in sorted(at[b])]}
        if (s, e) in at:
            rec.update(category="type_error", sub=None, gold_here_found=all((s, e, u) in pred for u in at[(s, e)]))
        elif over:
            contained = [b for b in over if s <= b[0] and b[1] <= e]
            if len(contained) >= 2:
                sub = "merged"
            elif len(contained) == 1 and len(over) == 1:
                g0, g1 = contained[0]
                sub = "wider_title" if g0 == s and text[g1:e] in rules["title_suffixes"] else "wider_other"
            elif len(over) == 1 and over[0][0] <= s and e <= over[0][1]:
                sub = "narrower"
            else:
                sub = "crossing"
            rec.update(category="boundary", sub=sub, type_differs=all(t not in at[b] for b in over))
        else:
            if x in rules["gold_texts"]:
                sub = "gold_name_elsewhere"
            elif x in rules["generic_words"]:
                sub = "generic_word"
            elif x in rules["category_words"] or rules["grade_pattern"].fullmatch(x):
                sub = "category_word"
            elif len(x) == 1:
                sub = "single_char"
            else:
                sub = "other"
            rec.update(category="spurious", sub=sub)
        fps.append(rec)
    for s, e, t in sorted(gold - pred):
        x = text[s:e]
        over_pred = sorted(q for q in pred if _overlaps(q, (s, e)))
        over_pending = [q for q in pending if _overlaps(q, (s, e))]
        kind = "wrong_prediction_overlaps" if over_pred else "abstained" if over_pending else "missed"
        fns.append({"start": s, "end": e, "type": t, "text": x, "category": kind, "new": x not in rules["seen"],
                    "predictions": over_pred, "pending_overlaps": bool(over_pending),
                    "removed_prediction_overlaps": any(_overlaps(q, (s, e)) for q in removed)})
    return fps, fns


def analyse_system(gold, prediction, texts, rules):
    """Every scored place of one system, categorised. gold: score.gold_spans output; prediction: evaluate's per-chapter
    {"spans", optional "errors" (invalid outputs), optional "pending" (located pending places)}."""
    out = {"tp": [], "fp": [], "fn": [], "invalid": 0, "removed": 0, "pending_places": Counter(), "pending_located": 0,
           "per_chapter": {}, "per_chapter_type": {}}
    for cid, ref in gold.items():
        unscored = ref["unscored"]
        g = {s for s in ref["spans"] if not _touches(s, unscored)}
        everything = prediction[cid]["spans"]
        p = {s for s in everything if not _touches(s, unscored)}
        removed = sorted(s for s in everything if _touches(s, unscored))
        pending = list(prediction[cid].get("pending", []))
        fps, fns = classify_chapter(g, p, pending, texts[cid], rules, removed)
        tps = [{"start": s, "end": e, "type": t, "text": texts[cid][s:e], "new": texts[cid][s:e] not in rules["seen"]}
               for s, e, t in sorted(g & p)]
        for r in tps + fps + fns:
            r["chapter_id"] = cid
        invalid = prediction[cid].get("errors", 0)
        out["tp"] += tps
        out["fp"] += fps
        out["fn"] += fns
        out["invalid"] += invalid
        out["removed"] += len(removed)
        out["per_chapter"][cid] = [len(tps), len(fps) + invalid, len(fns)]
        out["per_chapter_type"][cid] = {t: [sum(r["type"] == t for r in rs) for rs in (tps, fps, fns)] for t in ENTITY_TYPES}
        for q in pending:
            out["pending_located"] += 1
            out["pending_places"]["touches_an_unscored_range" if _touches(q, unscored) else
                                  "overlaps_a_scored_gold_span" if any(_overlaps(q, x) for x in g) else
                                  "overlaps_no_scored_gold_span"] += 1
    return out


def _tally(records, novelty=False):
    out = {"total": len(records), "by_type": {t: sum(r["type"] == t for r in records) for t in ENTITY_TYPES}}
    if novelty:
        out["by_novelty"] = {"new": sum(r["new"] for r in records), "seen": sum(not r["new"] for r in records)}
    return out


def counts(result, pending_marks=None):
    """The numbers of one system's analysis (no strings)."""
    fp, fn = result["fp"], result["fn"]
    of = lambda rs, k, v: [r for r in rs if r[k] == v]
    out = {
        "tp": _tally(result["tp"], True),
        "fp": {"total": len(fp) + result["invalid"], "located": len(fp), "invalid_no_position": result["invalid"],
               "by_type": _tally(fp)["by_type"]},
        "fn": {**_tally(fn, True)},
        "removed_by_unscored_ranges": result["removed"],
    }
    errs = of(fp, "category", "type_error")
    out["fp"]["type_error"] = {**_tally(errs), "of_which_the_gold_span_there_was_also_found": sum(r["gold_here_found"] for r in errs)}
    for kind in ("boundary", "spurious"):
        rs = of(fp, "category", kind)
        out["fp"][kind] = {**_tally(rs), "sub": {k: _tally(of(rs, "sub", k)) for k in FP_SUB[kind]}}
    out["fp"]["boundary"]["of_which_type_differs_from_every_overlapped_gold"] = sum(
        r["type_differs"] for r in of(fp, "category", "boundary"))
    for kind in FN_KINDS:
        out["fn"][kind] = _tally(of(fn, "category", kind), True)
    out["fn"]["wrong_prediction_overlaps"]["of_which_a_pending_mark_also_overlaps"] = sum(
        r["pending_overlaps"] for r in of(fn, "category", "wrong_prediction_overlaps"))
    missed = of(fn, "category", "missed")
    out["fn"]["missed"]["single_char"] = _tally([r for r in missed if len(r["text"]) == 1], True)
    out["fn"]["missed"]["of_which_a_prediction_removed_by_an_unscored_range_overlaps"] = sum(
        r["removed_prediction_overlaps"] for r in missed)
    if pending_marks is not None:
        out["pending"] = {"pending_marks": pending_marks, "located_places": result["pending_located"],
                          "places": {k: result["pending_places"][k] for k in PENDING_PLACES}}
    check_sums(out)
    return out


def check_sums(c):
    """Every category sums to its total, every split to its whole."""
    def whole(t, what):
        if sum(t["by_type"].values()) != t["total"]:
            raise AssertionError(f"{what}: types do not sum to the total")
        if "by_novelty" in t and sum(t["by_novelty"].values()) != t["total"]:
            raise AssertionError(f"{what}: new + seen is not the total")
    fp, fn = c["fp"], c["fn"]
    if sum(fp[k]["total"] for k in FP_KINDS) != fp["located"] or fp["located"] + fp["invalid_no_position"] != fp["total"]:
        raise AssertionError("false-positive categories do not sum to the false positives")
    for k in FP_KINDS:
        whole(fp[k], k)
    if {t: sum(fp[k]["by_type"][t] for k in FP_KINDS) for t in ENTITY_TYPES} != fp["by_type"]:
        raise AssertionError("false-positive categories do not sum to the false positives of each type")
    for kind, subs in FP_SUB.items():
        for s in subs:
            whole(fp[kind]["sub"][s], f"{kind}/{s}")
        if sum(fp[kind]["sub"][s]["total"] for s in subs) != fp[kind]["total"]:
            raise AssertionError(f"{kind}: sub-kinds do not sum to it")
        if {t: sum(fp[kind]["sub"][s]["by_type"][t] for s in subs) for t in ENTITY_TYPES} != fp[kind]["by_type"]:
            raise AssertionError(f"{kind}: sub-kinds do not sum to it per type")
    whole(fn, "fn")
    whole(c["tp"], "tp")
    for k in FN_KINDS:
        whole(fn[k], k)
    if sum(fn[k]["total"] for k in FN_KINDS) != fn["total"]:
        raise AssertionError("false-negative categories do not sum to the false negatives")
    for split in ("by_type", "by_novelty"):
        if {x: sum(fn[k][split][x] for k in FN_KINDS) for x in fn[split]} != fn[split]:
            raise AssertionError(f"false-negative categories do not sum per {split}")
    if "pending" in c and sum(c["pending"]["places"].values()) != c["pending"]["located_places"]:
        raise AssertionError("pending places do not sum")


# ---------------------------------------------------------------------------------------------- new names

def new_name_occurrences(gold, prediction, texts, seen):
    """(tp, fp, fn) of evaluate's new-name view: gold and predictions restricted alike to unseen texts, unscored left out."""
    tp = fp = fn = 0
    for cid, ref in gold.items():
        keep = lambda x: not _touches(x, ref["unscored"]) and texts[cid][x[0]:x[1]] not in seen
        g = {x for x in ref["spans"] if keep(x)}
        p = {x for x in prediction[cid]["spans"] if keep(x)}
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    return tp, fp, fn


def per_name_recall(gold, prediction, texts, seen):
    """Post hoc: the new gold names, distinct by (text, type) and by text; found = at least one exact true positive."""
    names, found_names, by_text, found_texts = set(), set(), set(), set()
    for cid, ref in gold.items():
        p = prediction[cid]["spans"]
        for s, e, t in ref["spans"]:
            x = texts[cid][s:e]
            if _touches((s, e), ref["unscored"]) or x in seen:
                continue
            names.add((x, t))
            by_text.add(x)
            if (s, e, t) in p:
                found_names.add((x, t))
                found_texts.add(x)
    share = lambda a, b: round(len(a) / len(b), 4) if b else None
    return {"names": {"found": len(found_names), "total": len(names), "share": share(found_names, names)},
            "texts": {"found": len(found_texts), "total": len(by_text), "share": share(found_texts, by_text)},
            "names_by_type": {t: {"found": sum(1 for _, u in found_names if u == t), "total": sum(1 for _, u in names if u == t)}
                              for t in ENTITY_TYPES}}


def _mean_range(values):
    known = [v for v in values if v is not None]
    if not known:
        return None, None
    return round(sum(known) / len(known), 4), [round(min(known), 4), round(max(known), 4)]


def model_per_name(runs, per_name):
    """A model's per-name result: the mean over its runs of each run's share, and the range (min, max)."""
    out = {"runs": list(runs)}
    for k in ("names", "texts"):
        mean, rng = _mean_range([per_name[r][k]["found"] / per_name[r][k]["total"] if per_name[r][k]["total"] else None
                                 for r in runs])
        out[k] = {"mean_share": mean, "range_share": rng, "total": per_name[runs[0]][k]["total"],
                  "found_by_run": [per_name[r][k]["found"] for r in runs]}
    return out


def model_occurrence(runs, occ):
    """evaluate.new_name_view's model numbers: recall by run (rounded), mean and range of the unrounded recalls."""
    recall = [occ[r][0] / (occ[r][0] + occ[r][2]) if occ[r][0] + occ[r][2] else None for r in runs]
    mean, rng = _mean_range(recall)
    return {"runs": list(runs), "recall_by_run": {r: (round(x, 4) if x is not None else None) for r, x in zip(runs, recall)},
            "mean_recall": mean, "range_recall": rng}


# ---------------------------------------------------------------------------------------------- pooling numeric trees

def add_trees(a, b):
    if isinstance(a, dict):
        return {k: add_trees(a[k], b[k]) for k in a}
    return a + b


def scale_tree(a, n):
    if isinstance(a, dict):
        return {k: scale_tree(v, n) for k, v in a.items()}
    return round(a / n, 2)


def model_counts(runs, c):
    """A model's counts: (summed, mean_per_run). summed adds its runs' trees (exact integers) and is checked like a run's
    (check_sums); mean_per_run = summed / n_runs with each cell rounded to 2 decimals on its own, for display only: its
    cells need not add up to their totals, so it is neither checked nor written to metrics.json."""
    summed = c[runs[0]]
    for r in runs[1:]:
        summed = add_trees(summed, c[r])
    check_sums(summed)
    return summed, scale_tree(summed, len(runs))


# ---------------------------------------------------------------------------------------------- checks against the scoring run

def check_against_eval(system, result, c, metrics):
    """Stop unless one system's analysis gives exactly the scoring run's numbers (no strings quoted)."""
    problems = []
    want = metrics["per_chapter_counts"][system]
    if {k: list(v) for k, v in want.items()} != result["per_chapter"]:
        problems.append("per-chapter TP/FP/FN")
    totals = [sum(v[i] for v in want.values()) for i in range(3)]
    if totals != [c["tp"]["total"], c["fp"]["total"], c["fn"]["total"]]:
        problems.append("TP/FP/FN totals")
    if metrics.get("per_chapter_type_counts", {}).get(system) is not None:
        ptc = metrics["per_chapter_type_counts"][system]
        if any(ptc[cid][t] != result["per_chapter_type"][cid][t] for cid in ptc for t in ENTITY_TYPES):
            problems.append("per-chapter per-type counts")
    score = metrics["scores"][system]
    for t in ENTITY_TYPES:
        got = (c["tp"]["by_type"][t], c["fp"]["by_type"][t], c["fn"]["by_type"][t])
        if got != tuple(score["per_type"][t][k] for k in ("tp", "fp", "fn")):
            problems.append(f"{t} TP/FP/FN")
    if _prf_defined(c["tp"]["total"], c["fp"]["total"], c["fn"]["total"]) != score["micro"]:
        problems.append("micro P/R/F1")
    diag = score["strict_diagnostics"]
    if diag["invalid_predictions"] != c["fp"]["invalid_no_position"]:
        problems.append("invalid outputs")
    te = c["fp"]["type_error"]
    if diag["fp_same_span_wrong_type"] != te["total"] - te["of_which_the_gold_span_there_was_also_found"]:
        problems.append("fp_same_span_wrong_type")
    if diag["fn_where_draft_was_pending"] != (c["fn"]["abstained"]["total"]
                                              + c["fn"]["wrong_prediction_overlaps"]["of_which_a_pending_mark_also_overlaps"]):
        problems.append("fn_where_draft_was_pending")
    if metrics["test"]["unscored"]["predicted_spans_removed"][system] != c["removed_by_unscored_ranges"]:
        problems.append("predictions removed by the unscored ranges")
    if problems:
        raise SystemExit(f"{system}: this analysis does not reproduce the scoring run's {', '.join(problems)}")


def check_new_names_against_eval(occ, model_occ, metrics):
    view = metrics["new_name_recall"]
    for s, (tp, fp, fn) in occ.items():
        v = view["systems"][s]
        if (v["tp"], v["fp"], v["fn"]) != (tp, fp, fn):
            raise SystemExit(f"{s}: the per-occurrence new-name counts differ from the scoring run's")
    for m, v in model_occ.items():
        e = view["models"][m]
        if (e["recall_by_run"], e["mean_recall"], e["range_recall"]) != (v["recall_by_run"], v["mean_recall"], v["range_recall"]):
            raise SystemExit(f"model {m}: the per-occurrence new-name recall differs from the scoring run's")


CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")


def check_counts_only(metrics):
    """metrics.json is committed: it may hold numbers and fixed English words, never a string from the book."""
    if CJK.search(json.dumps(metrics, ensure_ascii=False)):
        raise SystemExit("metrics.json would hold Chinese text; it must hold counts only")


# ---------------------------------------------------------------------------------------------- loading (all checked)

def llm_system(spec, root, snapshot, test_ids, task_of, files):
    """A generative system read again from its answers, task by task as evaluate.main does. Returns (prediction with
    "errors" and "pending" per chapter, located predictions, bookkeeping)."""
    preds, pending, problems, missing = [], [], Counter(), []
    invalid, abstained = {}, {}
    checked = {"checked": 0, "no_hash_logged": 0}
    calls_file = root / spec["calls"]
    calls = json.loads(calls_file.read_text(encoding="utf-8"))
    files[spec["calls"]] = sha256(calls_file)
    for task in sorted({task_of[c] for c in test_ids}):
        task_dir = root / "data" / "llm_tasks" / task
        ids = [c for c in json.loads((task_dir / "task.json").read_text(encoding="utf-8"))["chapters"] if c in set(test_ids)]
        answer_file = task_dir / f"pass_{spec['label']}.json"
        if not answer_file.exists():
            raise SystemExit(f"{spec['name']}: no answer for {task}")
        digest = sha256(answer_file)
        files[f"data/llm_tasks/{task}/{answer_file.name}"] = digest
        got, b = llm_predictions(json.loads(answer_file.read_text(encoding="utf-8")), snapshot, ids)
        preds += got
        invalid.update(b["invalid_by_chapter"])
        abstained.update(b["abstained_by_chapter"])
        pending += b["pending_places"]
        problems.update(b["problems"])
        missing += b["chapters_missing"]
        logged = [c.get("answer_sha256") for c in calls["calls"] if c.get("task") == task and c.get("answer_sha256")
                  and c.get("label", spec["label"]) == spec["label"]]
        if not logged:
            checked["no_hash_logged"] += 1
        elif digest != logged[-1]:
            raise SystemExit(f"{spec['name']}: the answer for {task} is not the one its call log recorded")
        else:
            checked["checked"] += 1
    pred = as_prediction(preds, test_ids)
    for cid in test_ids:
        pred[cid]["errors"] = invalid.get(cid, 0)
        pred[cid]["pending"] = [(a, b) for c, a, b in pending if c == cid]
    book = {"located": len(preds), "invalid": sum(invalid.values()), "pending_marks": sum(abstained.values()),
            "pending_places": len(pending), "problems": dict(problems), "chapters_missing": missing,
            "answers_match_logged_sha256": checked}
    return pred, preds, book


def load_systems(names, eval_cfg, eval_metrics, eval_dir, root, snapshot, test_ids, test_export, files):
    """{system: prediction} for the named systems of the scoring run, every file checked as the module docstring says."""
    llm = {s["name"]: s for s in eval_cfg.get("llm_systems", [])}
    spans = {s["name"] for s in eval_cfg.get("span_systems", [])}
    task_of = {}
    for task in test_export.get("llm_drafts", {}):
        for cid in json.loads((root / "data" / "llm_tasks" / task / "task.json").read_text(encoding="utf-8"))["chapters"]:
            task_of[cid] = task
    out, pending_marks = {}, {}
    rel = lambda p: p.relative_to(root).as_posix()
    wanted = set(test_ids)
    for s in names:
        f = eval_dir / f"predictions_test_{s}.jsonl"
        if not f.is_file():
            raise SystemExit(f"{s}: {rel(f)} does not exist")
        digest = sha256(f)
        files[rel(f)] = digest
        rows = read_jsonl(f)
        record = eval_metrics["systems"][s]
        for r in rows:
            text = snapshot.chapters[r["chapter_id"]].text if r.get("chapter_id") in snapshot.chapters else None
            if r.get("chapter_id") not in wanted or text is None or text[r["start"]:r["end"]] != r["text"] \
                    or r.get("type") not in ENTITY_TYPES:
                raise SystemExit(f"{s}: a saved prediction is outside the test chapters, has another type or another text")
        if s == "gliner_zero_shot":
            threshold = record["threshold"]
            if any(r["kept"] != (r["score"] > threshold) for r in rows):
                raise SystemExit("gliner_zero_shot: a row's kept flag disagrees with the recorded threshold")
            if sum(r["kept"] for r in rows) != record["outputs"]["outputs"]:
                raise SystemExit("gliner_zero_shot: the kept rows are not as many as the scoring run counted")
            out[s] = as_prediction(rows, test_ids, threshold)
        elif s == "dictionary":
            again, _ = dictionary_predictions(snapshot.dictionary, [snapshot.chapters[c] for c in test_ids],
                                              eval_cfg["dictionary"]["min_len"])
            if again != rows:
                raise SystemExit("dictionary: the saved predictions differ from dictionary matching recomputed from the snapshot")
            out[s] = as_prediction(rows, test_ids)
        elif s in llm:
            pred, preds, book = llm_system(llm[s], root, snapshot, test_ids, task_of, files)
            if preds != rows:
                raise SystemExit(f"{s}: the predictions read again from its answers differ from the saved file")
            o = record["outputs"]
            if (book["located"], book["invalid"], book["pending_marks"]) != (
                    o["located_predictions"], o["invalid_predictions"], o["abstentions_pending"]):
                raise SystemExit(f"{s}: located, invalid or pending counts differ from the scoring run's")
            if book["answers_match_logged_sha256"] != record["answers_match_logged_sha256"]:
                raise SystemExit(f"{s}: the answer hash checks differ from the scoring run's")
            out[s] = pred
            pending_marks[s] = book["pending_marks"]
        elif s in spans:
            if digest != record.get("predictions_sha256"):
                raise SystemExit(f"{s}: {rel(f)} is not the file the scoring run recorded")
            out[s] = as_prediction(rows, test_ids)
        else:
            raise SystemExit(f"{s} is not a system of the scoring run")
    return out, pending_marks


# ---------------------------------------------------------------------------------------------- the report (has book text)

ZH_FP = {"type_error": "类型错（起止相同、类型不同）", "boundary": "边界错（和答案重叠、起止不同）",
         "spurious": "凭空多标（不和任何答案重叠）",
         "merged": "合并（盖住两个以上答案）", "wider_title": "多带称谓（答案＋称谓后缀）", "wider_other": "偏宽（其他）",
         "narrower": "偏窄（在一个答案里面）", "crossing": "交错（部分重叠）",
         "gold_name_elsewhere": "同一字串在别处是答案", "generic_word": "泛称代词（A1 列表）",
         "category_word": "类别词（车票/药剂/道具、等级＋类别）", "single_char": "单字", "other": "其他"}
ZH_FN = {"wrong_prediction_overlaps": "有不对的预测压在上面（类型错或边界错的另一面）",
         "abstained": "待审（系统标了待审，没给答案）", "missed": "完全没标（什么都没压在上面）"}
ZH_RULES = [
    "计分范围和评分程序一样：碰到答案文件的不计分范围或待审标记的答案和预测都不算；完全相同的预测只算一次。"
    "下面 G 是一章里计分的答案，P 是计分的预测；“重叠”指至少共用一个字；答案的位置按不同的（起点，终点）算。",
    "误标（P 里不在 G 里的每一处），按顺序取第一条符合的：① 类型错：有答案和它起止完全相同；"
    "② 边界错：和某个答案重叠。再分：合并＝盖住两个以上答案；多带称谓＝只盖住一个答案、不碰别的、起点相同、多出来的尾巴正好是称谓后缀表里的词；"
    "偏宽（其他）＝只盖住一个答案、不碰别的；偏窄＝落在一个答案里面；交错＝其余；"
    "③ 凭空多标：不和任何答案重叠。再分：同一字串在别处是答案（考卷任何地方、任何类型）；泛称代词（A1 列表）；"
    "类别词（类别词表，或“等级＋类别”的式子）；单字；其他。"
    "无位置的输出（模型给了、但定位不到或没有合法类型）单独计：算误标，没有类型和位置。",
    "漏标（G 里不在 P 里的每一处），按顺序：① 有预测压在上面（答案不重叠时，那个预测必定是误标：类型错或边界错的另一面）；"
    "② 待审：没有预测压着，但系统的待审标记压着（只有 Opus、Sonnet 能标待审）；③ 完全没标：什么都没压着（被不计分范围删掉的预测当作没有）。"
    "每一类再按新名字 / 见过的分；完全没标的另记单字。",
    "答案这边的待审：答案文件的不计分范围（几处、各系统被删掉几个预测）和待审标记。",
    "系统的待审标记（只有 Opus、Sonnet）：能定位的每一处按顺序取第一条：碰到不计分范围（和答案文件的不计分范围或待审标记共用一个字）；"
    "压在计分答案上；不压任何计分答案。碰到不计分范围的那处若同时压着计分答案，那个答案仍记为待审漏标，"
    "所以“压在计分答案上”的数不一定等于待审漏标数。",
    "类型：误标按系统给的类型，漏标按答案的类型。",
    "新名字：按出现次数（事先定）＝评分运行的 new_name_recall，这里重算并核对一致；按不同名字（事后）：新名字处里不同的（字串，类型）"
    "和不同字串，系统至少有一处完全标对就算找到；多次运行的模型取各次比例的平均和范围。",
    "检查：每个系统逐章、逐类型的 TP/FP/FN、micro P/R/F1、无位置输出数、被不计分范围删掉的预测数、新名字按出现次数的数字都必须和评分运行的 "
    "metrics.json 一致，每一类加起来必须等于总数（每个系统和每个模型各次运行的合计都查）；有一处不符程序就停。"
    "模型的“平均”＝合计÷运行数，每格分别四舍五入到两位小数，不查、也不写进 metrics.json。",
    "时间：这些类别、先后顺序和再分的小类是看过 v9 分数之后定的，只作描述，不是事先定的检验；"
    "事先定的只有新名字按出现次数的召回。",
]


def _cell(t):
    return [t["total"], *(t["by_type"][x] for x in ENTITY_TYPES)]


def category_table(c):
    lines = ["| 类别 | 合计 | " + " | ".join(ENTITY_TYPES) + " | 新名字 | 见过的 |", "|" + "---|" * (len(ENTITY_TYPES) + 4)]

    def row(label, t):
        novelty = [t["by_novelty"]["new"], t["by_novelty"]["seen"]] if "by_novelty" in t else ["", ""]
        lines.append("| " + " | ".join(str(v) for v in [label, *_cell(t), *novelty]) + " |")
    row("**对（TP）**", c["tp"])
    row("**误标（FP）合计**", {"total": c["fp"]["total"], "by_type": c["fp"]["by_type"]})
    for k in FP_KINDS:
        row(f"误标·{ZH_FP[k]}", c["fp"][k])
        for s in FP_SUB.get(k, ()):
            row(f"　└ {ZH_FP[s]}", c["fp"][k]["sub"][s])
    lines.append("| " + " | ".join(["误标·无位置的输出", str(c["fp"]["invalid_no_position"]), *("–" for _ in ENTITY_TYPES), "", ""]) + " |")
    row("**漏标（FN）合计**", c["fn"])
    for k in FN_KINDS:
        row(f"漏标·{ZH_FN[k]}", c["fn"][k])
    row("　└ 完全没标里的单字", c["fn"]["missed"]["single_char"])
    notes = [f"- 类型错里，该处答案（另一类型）也被找到的：{c['fp']['type_error']['of_which_the_gold_span_there_was_also_found']}；"
             f"边界错里，类型和所有重叠答案都不同的：{c['fp']['boundary']['of_which_type_differs_from_every_overlapped_gold']}。",
             f"- 有不对的预测压着的漏标里，同时也有待审标记的：{c['fn']['wrong_prediction_overlaps']['of_which_a_pending_mark_also_overlaps']}；"
             f"完全没标里，有一个被不计分范围删掉的预测压着的：{c['fn']['missed']['of_which_a_prediction_removed_by_an_unscored_range_overlaps']}。",
             f"- 不计分范围删掉的预测：{c['removed_by_unscored_ranges']}。"]
    if "pending" in c:
        p = c["pending"]
        notes.append(f"- 系统的待审标记 {p['pending_marks']} 个，能定位的 {p['located_places']} 处（按顺序取第一条，规则见文末）：碰到不计分范围 "
                     f"{p['places']['touches_an_unscored_range']}，压在计分答案上 {p['places']['overlaps_a_scored_gold_span']}，"
                     f"不压任何计分答案 {p['places']['overlaps_no_scored_gold_span']}。")
    return lines + [""] + notes


def _example_groups(results, systems, order, text_of):
    """{category key: {(text, type): {"n", "systems", "first", "overlaps"}}} over the given systems; "overlaps" counts
    every distinct set of overlapping answers (false positives) or predictions (false negatives) over all occurrences."""
    groups = defaultdict(dict)
    for s in systems:
        for r in results[s]["fp"]:
            key = ("fp", r["category"], r["sub"])
            _add(groups[key], (r["text"], r["type"]), s, r, order, text_of, r["gold"])
        for r in results[s]["fn"]:
            key = ("fn", r["category"], ("new" if r["new"] else "seen") if r["category"] == "missed" else None)
            _add(groups[key], (r["text"], r["type"]), s, r, order, text_of, r["predictions"])
    return groups


def _add(group, key, system, rec, order, text_of, overlapping):
    g = group.setdefault(key, {"n": 0, "systems": set(), "first": None, "overlaps": Counter()})
    g["n"] += 1
    g["systems"].add(system)
    if overlapping:
        t = text_of[rec["chapter_id"]]
        g["overlaps"][tuple((t[a:b], u) for a, b, u in overlapping)] += 1
    place = (order[rec["chapter_id"]], rec["start"], rec["end"])
    if g["first"] is None or place < g["first"][0]:
        g["first"] = (place, rec)


def _example_line(text_of, key, g, n_systems, width, side):
    x, t = key
    rec = g["first"][1]
    cid = rec["chapter_id"]
    text = text_of[cid]
    extra = ""
    if g["overlaps"]:
        label = "重叠的答案" if side == "fp" else "压在上面的预测"
        extra = f"｜{label}：" + "；".join("、".join(f"「{y}」{u}" for y, u in sig) + f" ×{n}"
                                          for sig, n in sorted(g["overlaps"].items(), key=lambda kv: (-kv[1], kv[0])))
    seeds = f"，{len(g['systems'])}/{n_systems} 个运行" if n_systems > 1 else ""
    return f"- 「{x}」 {t} ×{g['n']}{seeds}{extra}｜例（{cid}）：{_context(text, rec['start'], rec['end'], width)}"


def build_report(run_id, ctx, results, c, per_name, occ, models, gold_side, cfg):
    text_of, order = ctx["texts"], {cid: i for i, cid in enumerate(ctx["test_ids"])}
    width, top_n = cfg["examples"]["context_chars"], cfg["examples"]["top_n"]
    L = [f"# 错误分析 {run_id}", "",
         "含小说原文和模型输出的字串，在 data/ 下，不入库。", "",
         "## 先读这里", "",
         "- 每一处的类别都由程序按固定的机械规则分出（规则见文末，和 `src/novel_ner/errors.py` 开头写的一样）。"
         "没有人、也没有 Claude 判断任何一处答案对不对：Claude 写了这个程序，但 Opus 是被比较的模型之一，所以 Claude 不判答案。",
         "- “误标”“漏标”只是计分上的说法，以答案文件为准：系统和答案文件不一致，就记作系统的误标或漏标。"
         "答案文件本身可能有问题的地方，这里一律不叫“答案的错误”，也不改答案。这份分析不是复查答案的清单。",
         "- 分类规则（有哪些类、先后顺序、再分的小类）是看过 v9 分数之后定的，只作描述，不是事先定的检验；"
         "事先定的只有新名字按出现次数的召回。the study plan 只举过几类例子：泛称当名字、边界多框称谓、新道具认不出。",
         "- “同一字串在别处是答案”只说明：系统在这里标的这串字，在考卷别的地方被答案标成了名字（任何类型）；这里没有答案。程序不判断哪边对。",
         f"- 来源：评分运行 `{cfg['eval_run']}`（答案文件 `{ctx['gold_file']}`，sha256 {ctx['gold_sha'][:12]}…；快照 {ctx['snapshot']}；"
         f"考卷 {len(ctx['test_ids'])} 章）。每个系统逐章、逐类型的 TP/FP/FN 都和这次评分的 metrics.json 核对一致（程序里的检查，不一致就停）。",
         f"- 泛称代词表照抄规则 v1.1 的 A1（程序运行时从 `{cfg['guide']}` 读出来核对）：{'、'.join(cfg['generic_words'])}。"
         f"称谓后缀、类别词和“等级＋类别”的式子取自 `{cfg['rule_words_from']}`（弱标签的转换规则用的同一份）。",
         "- 新名字的“见过”按事先定的规则：前 780 章的词典名和 Opus 名单里出现过的字串（字面完全一样，任何类型）；没出现过的叫新名字。", "",
         "## 名词", "",
         "- **误标（FP）**：系统给出、但答案里没有完全相同（同章、同起止、同类型）的一处。",
         "- **漏标（FN）**：答案里有、系统没有完全相同地给出的一处。",
         "- **待审**：系统这边，只有 Opus、Sonnet 能标“待审”（这处不给答案）；小模型不能。答案这边，是答案文件里的不计分范围和待审标记。", "",
         "## 一、答案这边的待审", "",
         f"- 答案文件记录的不计分范围 {gold_side['unscored_ranges']} 处，答案里的待审标记 {gold_side['gold_pending_marks']} 处"
         f"（定稿答案应为 0），两者一共涉及 {gold_side['chapters_with_them']} 章；碰到它们的已定答案 {gold_side['settled_gold_spans_touching_them']} 处（不计分）。",
         "- 它们删掉的系统预测（不计分，也不算误标）：" + "，".join(f"{s} {n}" for s, n in gold_side["predicted_spans_removed"].items()) + "。", "",
         "## 二、总表", "",
         "每个运行一行；“模型”行是该模型各次运行（种子）的平均，每格分别四舍五入到两位小数，所以一行里各类相加可能和合计差 0.01。",
         "", "| 系统 | 对 TP | 误标 FP | 类型错 | 边界错 | 其中多带称谓 | 凭空多标 | 其中别处是答案 | 无位置 | 漏标 FN | 有预测压着 | 待审 | 完全没标 | 其中新名字 |",
         "|" + "---|" * 14]
    row = lambda name, x: L.append(
        f"| {name} | {x['tp']['total']} | {x['fp']['total']} | {x['fp']['type_error']['total']} | {x['fp']['boundary']['total']} | "
        f"{x['fp']['boundary']['sub']['wider_title']['total']} | {x['fp']['spurious']['total']} | "
        f"{x['fp']['spurious']['sub']['gold_name_elsewhere']['total']} | {x['fp']['invalid_no_position']} | {x['fn']['total']} | "
        f"{x['fn']['wrong_prediction_overlaps']['total']} | {x['fn']['abstained']['total']} | {x['fn']['missed']['total']} | "
        f"{x['fn']['missed']['by_novelty']['new']} |")
    for s in c:
        row(s, c[s])
    for m, v in models.items():
        if v["n_runs"] > 1:
            row(f"模型 {m}（{v['n_runs']} 次平均）", v["mean_per_run"])
    L += ["", "## 三、新名字：按出现次数和按不同名字", "",
          "- 按出现次数（事先定的，就是评分运行里的 new_name_recall，这里重算核对过）：新名字的每一处算一次。",
          "- 按不同名字（**事后**加的，看过分数之后才定）：同一个名字（字串＋类型）不管出现几次只算一个，"
          "只要系统至少有一处完全标对就算找到；“不同字串”不看类型。模型给各次运行比例的平均和范围，不是“任何一次找到就算”。", "",
          f"新名字：{ctx['new_gold']['spans']} 处，{ctx['new_gold']['names']} 个不同名字（字串＋类型），{ctx['new_gold']['texts']} 个不同字串。", "",
          "| 系统 | 按出现次数召回（事先定） | 按不同名字找到（事后） | 按不同字串找到（事后） |", "|---|---|---|---|"]
    share = lambda a, b: f"{a}/{b} = {a / b:.3f}" if b else f"{a}/{b}"
    mean_range = lambda mean, rng: f"平均 {mean:.3f}（{rng[0]:.3f}–{rng[1]:.3f}）" if mean is not None else "–"
    for s in c:
        tp, _, fn = occ[s]
        pn = per_name[s]
        L.append(f"| {s} | {share(tp, tp + fn)} | {share(pn['names']['found'], pn['names']['total'])} | "
                 f"{share(pn['texts']['found'], pn['texts']['total'])} |")
    for m, v in models.items():
        if v["n_runs"] > 1:
            o, n = v["new_names_per_occurrence"], v["new_names_per_name_post_hoc"]
            L.append(f"| 模型 {m}（{v['n_runs']} 次） | {mean_range(o['mean_recall'], o['range_recall'])} | "
                     f"{mean_range(n['names']['mean_share'], n['names']['range_share'])} | "
                     f"{mean_range(n['texts']['mean_share'], n['texts']['range_share'])} |")
    L += ["", "按出现次数时，出现很多次的名字（比如主角）权重大；按不同名字时，每个名字一样重，出现一两次的新名字分量就上来了。"
          "两种看法回答的问题不同，两个都留着。", "", "## 四、各系统分类表", "",
          "类型一栏：误标按系统给的类型，漏标按答案的类型。“新名字 / 见过的”只对对、漏标两类有意义。", ""]
    for s in c:
        L += [f"### {s}", "", *category_table(c[s]), ""]
    L += ["## 五、主要系统的例子", "",
          f"每类按出现次数取前 {top_n} 个字串（字串＋类型）。“重叠的答案”“压在上面的预测”列出所有出现处里各不相同的重叠情况和各自次数"
          "（×n）；只有末尾那段上下文取自第一次出现处（【】是范围）。几次运行合起来看时，“k/n 个运行”是有几次运行出现过。", ""]
    for label, systems in cfg["examples"]["groups"].items():
        groups = _example_groups(results, systems, order, text_of)
        L += [f"### {label}" + (f"（{len(systems)} 次运行合起来）" if len(systems) > 1 else ""), ""]
        keys = [("fp", k, s) for k in FP_KINDS for s in (FP_SUB.get(k) or (None,))] + \
               [("fn", "wrong_prediction_overlaps", None), ("fn", "abstained", None), ("fn", "missed", "new"), ("fn", "missed", "seen")]
        for key in keys:
            g = groups.get(key, {})
            if not g:
                continue
            side, kind, sub = key
            title = (f"误标·{ZH_FP[kind]}" + (f"·{ZH_FP[sub]}" if sub else "")) if side == "fp" else \
                    (f"漏标·{ZH_FN[kind]}" + ({"new": "·新名字", "seen": "·见过的名字"}[sub] if sub else ""))
            L += [f"#### {title}（{sum(v['n'] for v in g.values())} 处，{len(g)} 个字串）", ""]
            top = sorted(g.items(), key=lambda kv: (-kv[1]["n"], kv[0]))[:top_n]
            L += [_example_line(text_of, k, v, len(systems), width, side) for k, v in top] + [""]
    sect = cfg["elsewhere_section"]
    L += [f"## 六、小模型（{sect['label']}）误标、但同一字串在别处是答案的名字", "",
          "规则（机械的）：小模型在这里标的一串字不和任何答案重叠，但同一串字在考卷别处被答案标成了名字（任何类型）。"
          "下面只列事实：这里标了几次、几次运行有、答案在别处标了几处什么类型，各给一段上下文。"
          f"程序不判断哪边对。这张表只按一个被比较的系统（{sect['label']}）的输出挑出来，而且是看过分数之后才加的，"
          "所以不能拿它决定复查哪些答案（项目的标注规程规定 Opus、Sonnet、GLiNER、词典的结果不能当答案草稿；小模型也是被比较的系统，同一道理）。"
          "如果仍因此改了某个名字：属于事后修订，要按规则对同一字串的所有出现一起改，不看是哪个系统标的；"
          "另存为新的答案版本，新旧两版都评分、都报告。", "",
          "| 字串 | 这里误标次数 | 运行数 | 答案在别处：处数（类型） | 误标处例 | 答案处例 |", "|---|---|---|---|---|---|"]
    elsewhere = defaultdict(lambda: {"n": 0, "systems": set(), "first": None})
    for s in sect["systems"]:
        for r in results[s]["fp"]:
            if r["sub"] == "gold_name_elsewhere":
                g = elsewhere[r["text"]]
                g["n"] += 1
                g["systems"].add(s)
                place = (order[r["chapter_id"]], r["start"])
                if g["first"] is None or place < g["first"][0]:
                    g["first"] = (place, r)
    gold_places = ctx["gold_places_by_text"]
    for x, g in sorted(elsewhere.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        r = g["first"][1]
        places = gold_places[x]
        kinds = Counter(t for _, _, _, t in places)
        cid, s0, e0, _ = places[0]
        L.append(f"| 「{x}」 | {g['n']} | {len(g['systems'])}/{len(sect['systems'])} | {len(places)}（"
                 + "、".join(f"{t} {n}" for t, n in sorted(kinds.items())) + f"） | {r['chapter_id']}：{_context(text_of[r['chapter_id']], r['start'], r['end'], width)} | "
                 f"{cid}：{_context(text_of[cid], s0, e0, width)} |")
    if not elsewhere:
        L.append("| （没有） | | | | | |")
    L += ["", "## 七、规则全文", "", f"（英文原文在 runs/{run_id}/config.json 的 rules，和 `src/novel_ner/errors.py` 开头的说明。）", ""]
    L += [f"- {x}" for x in ZH_RULES]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------- main

def outputs(run_id):
    return REPO_ROOT / "runs" / run_id, REPO_ROOT / "data" / "analysis" / run_id / "report.md"


def refuse_existing(run_id):
    run_dir, report = outputs(run_id)
    taken = [p for p in (run_dir / "config.json", run_dir / "metrics.json", report) if p.exists()]
    if taken:
        raise SystemExit(f"{taken[0]} already exists; use a new run id (outputs are never overwritten)")


def analyse(cfg, root):
    """Everything, read and checked; returns what the outputs are made of. root: where the inputs are read from."""
    files = {}
    eval_dir = root / cfg["eval_run"]
    eval_rec = json.loads((eval_dir / "config.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    files[f"{cfg['eval_run']}/config.json"] = sha256(eval_dir / "config.json")
    files[f"{cfg['eval_run']}/metrics.json"] = sha256(eval_dir / "metrics.json")
    if not (eval_rec["run_id"] == eval_metrics["run_id"] == Path(cfg["eval_run"]).name):
        raise SystemExit("the scoring run's config.json and metrics.json do not both name its folder")
    eval_cfg = eval_rec["config"]

    # the rule words: A1 from the guide (checked), the rest from the weak-label config
    guide_text = (REPO_ROOT / cfg["guide"]).read_text(encoding="utf-8")
    if a1_words(guide_text) != cfg["generic_words"]:
        raise SystemExit("generic_words is not the guide's A1 list, word for word")
    weak = json.loads((REPO_ROOT / cfg["rule_words_from"]).read_text(encoding="utf-8"))
    rule_files = {cfg["guide"]: sha256(REPO_ROOT / cfg["guide"]), cfg["rule_words_from"]: sha256(REPO_ROOT / cfg["rule_words_from"])}

    snapshot = load_snapshot(eval_cfg["snapshot"], root=root / "data" / "snapshots")
    if snapshot.id != eval_rec["snapshot"]["id"]:
        raise SystemExit("the snapshot is not the scoring run's")
    gold_file = root / eval_cfg["gold_test"]
    gold_sha = sha256(gold_file)
    if gold_sha != eval_rec["gold_test"]["sha256"]:
        raise SystemExit("the answer file is not the one the scoring run hashed")
    files[eval_cfg["gold_test"]] = gold_sha
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the test answers do not pass the gold check ({len(errors)} problems)")
    test_ids = done_chapters(export)
    if test_ids != eval_rec["gold_test"]["chapters"] or len(test_ids) != len(export["chapters"]):
        raise SystemExit("the test chapters are not the scoring run's")
    gold = gold_spans(export, test_ids)
    texts = {cid: snapshot.chapters[cid].text for cid in test_ids}

    if cfg["new_names"] != eval_cfg.get("new_names"):
        raise SystemExit("new_names is not the scoring run's pre-registered rule")
    if any("seen_names" in s for s in eval_cfg.get("span_systems", [])):
        raise SystemExit("a span system of the scoring run has its own seen set; this analysis uses the pre-registered one only")
    dictionary, mentions, seen_sources = weak_names(snapshot, cfg["new_names"]["cutoff_chapter"])
    seen = dictionary | mentions

    names = list(eval_metrics["systems"])
    systems = cfg.get("systems") or names
    if set(systems) - set(names) or len(set(systems)) != len(systems):
        raise SystemExit("systems must be systems of the scoring run, each once")
    systems = [s for s in names if s in set(systems)]
    shown = [s for runs in cfg["examples"]["groups"].values() for s in runs] + cfg["elsewhere_section"]["systems"]
    if set(shown) - set(systems):
        raise SystemExit("examples and elsewhere_section may only name analysed systems")
    models ={m: runs for m, runs in (eval_cfg.get("models") or {}).items() if set(runs) <= set(systems)}
    predictions, pending_marks = load_systems(systems, eval_cfg, eval_metrics, eval_dir, root, snapshot, test_ids, export, files)

    scored = [(cid, s, e, t) for cid, ref in gold.items() for s, e, t in sorted(ref["spans"]) if not _touches((s, e), ref["unscored"])]
    gold_places_by_text = defaultdict(list)
    for cid, s, e, t in scored:
        gold_places_by_text[texts[cid][s:e]].append((cid, s, e, t))
    rules = make_rules(weak["title_suffixes"], cfg["generic_words"], weak["category_words"], weak["grade_category_pattern"],
                       gold_places_by_text, seen)

    results, c, occ, per_name = {}, {}, {}, {}
    for s in systems:
        results[s] = analyse_system(gold, predictions[s], texts, rules)
        c[s] = counts(results[s], pending_marks.get(s))
        check_against_eval(s, results[s], c[s], eval_metrics)
        occ[s] = new_name_occurrences(gold, predictions[s], texts, seen)
        per_name[s] = per_name_recall(gold, predictions[s], texts, seen)
        if (c[s]["tp"]["by_novelty"]["new"], c[s]["fn"]["by_novelty"]["new"]) != (occ[s][0], occ[s][2]):
            raise SystemExit(f"{s}: new-name true positives / false negatives differ between the two counts")
    model_occ = {m: model_occurrence(runs, occ) for m, runs in models.items()}
    check_new_names_against_eval(occ, model_occ, eval_metrics)

    model_out = {}
    for m, runs in models.items():
        summed, mean = model_counts(runs, c)
        model_out[m] = {"runs": runs, "n_runs": len(runs), "summed": summed, "mean_per_run": mean,
                        "new_names_per_occurrence": model_occ[m], "new_names_per_name_post_hoc": model_per_name(runs, per_name)}

    positions = Counter((cid, s, e) for cid, s, e, _ in scored)
    unscored_ranges = [(cid, u) for cid in test_ids for u in export["chapters"][cid].get("unscored", [])]
    gold_pending = [(cid, a) for cid in test_ids for a in export["chapters"][cid]["annotations"] if a.get("pending")]
    gold_side = {"unscored_ranges": len(unscored_ranges), "gold_pending_marks": len(gold_pending),
                 "chapters_with_them": len({cid for cid, _ in unscored_ranges} | {cid for cid, _ in gold_pending}),
                 "settled_gold_spans_touching_them": sum(1 for ref in gold.values() for x in ref["spans"] if _touches(x, ref["unscored"])),
                 "predicted_spans_removed": {s: c[s]["removed_by_unscored_ranges"] for s in systems}}
    u = eval_metrics["test"]["unscored"]
    if (u["recorded_ranges"], u["pending_marks"], u["settled_gold_spans_overlapping_them"]) != (
            gold_side["unscored_ranges"], gold_side["gold_pending_marks"], gold_side["settled_gold_spans_touching_them"]):
        raise SystemExit("the answer side's unscored ranges or pending marks differ from the scoring run's")
    if len(scored) != eval_metrics["test"]["scored_gold_spans"]:
        raise SystemExit("the scored gold spans are not as many as the scoring run's")

    new_spans = [(cid, s, e, t) for cid, s, e, t in scored if texts[cid][s:e] not in seen]
    new_gold = {"spans": len(new_spans), "names": len({(texts[c_][s:e], t) for c_, s, e, t in new_spans}),
                "texts": len({texts[c_][s:e] for c_, s, e, _ in new_spans}),
                "names_by_type": {t: len({texts[c_][s:e] for c_, s, e, u_ in new_spans if u_ == t}) for t in ENTITY_TYPES}}
    gold_seen = eval_metrics["new_name_recall"]["seen_sets"]
    if len(gold_seen) != 1:
        raise SystemExit("the scoring run used more than one seen set")
    g_ = next(iter(gold_seen.values()))["gold"]
    if (g_["new_spans"], g_["new_distinct_text_type_pairs"], g_["new_distinct_texts"]) != (new_gold["spans"], new_gold["names"], new_gold["texts"]):
        raise SystemExit("the new gold spans or names differ from the scoring run's")

    return {"files": files, "rule_files": rule_files, "eval_rec": eval_rec, "systems": systems, "models": model_out,
            "results": results, "counts": c, "occ": occ, "per_name": per_name, "gold_side": gold_side, "new_gold": new_gold,
            "seen_sources": seen_sources,
            "ctx": {"texts": texts, "test_ids": test_ids, "gold_file": eval_cfg["gold_test"], "gold_sha": gold_sha,
                    "snapshot": snapshot.id, "new_gold": new_gold, "gold_places_by_text": gold_places_by_text},
            "test": {"chapters": len(test_ids), "scored_gold_spans": len(scored),
                     "gold_places_with_two_types": sum(1 for n in positions.values() if n > 1)}}


def build_metrics(run_id, cfg, a):
    m = {
        "run_id": run_id, "eval_run": cfg["eval_run"],
        "test": a["test"],
        "answer_side_pending": a["gold_side"],
        "systems": a["counts"],
        "models": {k: {**{x: v[x] for x in ("runs", "n_runs", "summed")},
                       "mean_per_run_note": ("mean = summed / n_runs; not stored because cells rounded one by one need not "
                                             "add up")}
                   for k, v in a["models"].items()},
        "new_names": {
            "seen_rule": cfg["new_names"], "seen_sources": a["seen_sources"],
            "new_gold": {k: v for k, v in a["new_gold"].items()},
            "per_occurrence_pre_registered": {
                "note": "evaluate's new_name_recall, re-derived and checked equal",
                "systems": {s: {"tp": tp, "fp": fp, "fn": fn, "recall": round(tp / (tp + fn), 4) if tp + fn else None}
                            for s, (tp, fp, fn) in a["occ"].items()},
                "models": {k: v["new_names_per_occurrence"] for k, v in a["models"].items()}},
            "per_name_post_hoc": {
                "note": ("post hoc, chosen after the scores were seen: distinct new (text, type) names and distinct new texts; "
                         "found = at least one exact true positive; a model = mean over its runs of each run's share, and the range"),
                "systems": a["per_name"],
                "models": {k: v["new_names_per_name_post_hoc"] for k, v in a["models"].items()}},
        },
        "checks": {"per_chapter_and_type_counts_equal_the_scoring_run": True, "categories_sum_to_totals": True,
                   "categories_sum_to_totals_checked_on": "every system and every model's summed counts",
                   "new_name_recall_equals_the_scoring_run": True, "input_files_hashed": len(a["files"])},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    check_counts_only(m)
    return m


def build_config(run_id, config_file, cfg, a, root, started=None, code=None):
    """started, code: the start time and git state taken before the analysis ran (main passes both); taken now if not given."""
    return {"run_id": run_id, "config_file": config_file, "config": cfg,
            "inputs_root": "." if root == REPO_ROOT else root.as_posix(),
            "eval_run": {"run": cfg["eval_run"], "run_code": a["eval_rec"].get("code"), "snapshot": a["eval_rec"]["snapshot"],
                         "gold_test": {k: a["eval_rec"]["gold_test"][k] for k in ("file", "sha256")}},
            "inputs_sha256": a["files"], "rule_files_sha256": a["rule_files"],
            "systems": a["systems"], "models": {k: v["runs"] for k, v in a["models"].items()},
            "rules": RULES, "code": code if code is not None else git_state(),
            "started_at": started or time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    ap.add_argument("--inputs-root", help="with --dry-run only: read the scoring run and data/ from this checkout "
                                          "(default: this one)")
    ap.add_argument("--dry-run", action="store_true", help="compute and check everything, print counts only, write nothing")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    root = Path(args.inputs_root).resolve() if args.inputs_root else REPO_ROOT
    if not args.dry_run and root.resolve() != Path(REPO_ROOT).resolve():
        raise SystemExit("--inputs-root is for --dry-run only: report.md would go to this checkout's git-ignored data/, "
                         "which `git worktree remove` deletes; run for real from the checkout that holds data/")
    if not args.dry_run:
        refuse_existing(run_id)
    started, code = time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_state()
    a = analyse(cfg, root)
    metrics = build_metrics(run_id, cfg, a)
    config = build_config(run_id, Path(args.config).as_posix(), cfg, a, root, started, code)
    report = build_report(run_id, a["ctx"], a["results"], a["counts"], a["per_name"], a["occ"], a["models"], a["gold_side"], cfg)
    for s, x in a["counts"].items():
        fp, fn = x["fp"], x["fn"]
        print(f"{s:14s} TP {x['tp']['total']:5d} FP {fp['total']:4d} (type {fp['type_error']['total']}, boundary "
              f"{fp['boundary']['total']} [title {fp['boundary']['sub']['wider_title']['total']}], spurious {fp['spurious']['total']} "
              f"[gold elsewhere {fp['spurious']['sub']['gold_name_elsewhere']['total']}], invalid {fp['invalid_no_position']}) "
              f"FN {fn['total']:4d} (overlapped {fn['wrong_prediction_overlaps']['total']}, pending {fn['abstained']['total']}, "
              f"missed {fn['missed']['total']} [new {fn['missed']['by_novelty']['new']}]) | new names: per occurrence "
              f"{a['occ'][s][0]}/{a['occ'][s][0] + a['occ'][s][2]}, per name {a['per_name'][s]['names']['found']}/"
              f"{a['per_name'][s]['names']['total']}, per text {a['per_name'][s]['texts']['found']}/{a['per_name'][s]['texts']['total']}")
    for m, v in a["models"].items():
        if v["n_runs"] < 2:
            continue
        x, o, n = v["mean_per_run"], v["new_names_per_occurrence"], v["new_names_per_name_post_hoc"]
        print(f"model {m:9s} mean of {v['n_runs']}: TP {x['tp']['total']} FP {x['fp']['total']} (type {x['fp']['type_error']['total']}, "
              f"boundary {x['fp']['boundary']['total']}, spurious {x['fp']['spurious']['total']}) FN {x['fn']['total']} (overlapped "
              f"{x['fn']['wrong_prediction_overlaps']['total']}, pending {x['fn']['abstained']['total']}, missed {x['fn']['missed']['total']}) "
              f"| new names: per occurrence {o['mean_recall']} {o['range_recall']}, per name {n['names']['mean_share']} "
              f"{n['names']['range_share']}, per text {n['texts']['mean_share']} {n['texts']['range_share']}")
    print(f"test: {a['test']['chapters']} chapters, {a['test']['scored_gold_spans']} scored gold spans, "
          f"{a['test']['gold_places_with_two_types']} gold places with two types; new gold: {a['new_gold']['spans']} spans, "
          f"{a['new_gold']['names']} (text, type) names, {a['new_gold']['texts']} texts")
    print(f"answer side: {a['gold_side']['unscored_ranges']} unscored ranges, {a['gold_side']['gold_pending_marks']} pending marks; "
          f"{len(a['files'])} input files hashed; report {report.count(chr(10))} lines")
    if args.dry_run:
        print("dry run: nothing written")
        return
    run_dir, report_file = outputs(run_id)
    refuse_existing(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    report_file.write_text(report, encoding="utf-8")
    print(f"wrote {run_dir} and {report_file}")


if __name__ == "__main__":
    main()
