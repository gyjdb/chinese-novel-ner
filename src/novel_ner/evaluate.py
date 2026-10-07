"""Formal scoring on the test answers (annotation guide v1.1, appendix 1).

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.evaluate configs/eval_test_v3.json [--run-id <id>]

Strict position-level scoring: a prediction counts only if chapter, start, end and type all match
a gold span. Gold pending marks and the gold file's unscored ranges are left out for every system:
any gold or predicted span overlapping one by a character counts for nothing (score.gold_spans).
Settings are chosen on the development answers only (the GLiNER threshold); the test set is scored
once with them.

Reported, per appendix 1: micro P/R/F1; each type with its support, predictions and distinct names
(a type with nothing to score is marked so, not given a 0); micro over the non-person types and the
macro average over types; the unscored ranges (reasons, coverage, what they removed); the cross-chapter
group and the rest under one fixed region rule; span-only scores as an auxiliary view. Uncertainty:
paired bootstrap over chapters, resampled within the plot units the test set was stratified by, and,
as a sensitivity check, over whole plot units (only a few, unequal, so that interval is unstable).

Systems, with their input conditions recorded:
- gliner_zero_shot (local): GLiNER multi v2.1, no training, chapter text only, fed character by character;
- dictionary: chapter text plus the snapshot's name dictionary (from the novel project; every name in it
  first appears by chapter 780, which the run checks and records);
- generative models listed under "llm_systems" in the config (e.g. Opus and Sonnet run by claude_run.py): their
  answers to the same prompt GPT got, located by paragraph, text and occurrence. Appendix 1: an output that
  cannot be located or has no valid type is an invalid prediction (counted in the overall FP and listed);
  a pending mark is an abstention (no prediction); exact repeats count once; one place given two types
  keeps both;
- trained systems listed under "span_systems": the test predictions of a tagger predict run
  (tagger.py predict), checked before anything is scored (span_system) and scored like every other system.

Optional config keys (a config without them scores exactly as before):
- "span_systems": [{"name", "predictions_run": "runs/<id>", "input_condition"[, "seen_names", "note"]}];
  "dev_set" (default configs/annotate_dev.json) names the development chapters no training chapter may be.
- "models": {"name": [system, ...]}: several runs of one model (two Opus runs, or three seeds of one weak-supervision
  setting) pooled by model_summary; a model of trained systems must hold runs that differ only in the seed.
- "pairs": [[a, b], ...]: compute only these paired differences (a - b). Both names are systems (main intervals)
  or both are models (pooled comparison); a single system goes into a pooled pair as a one-run model. Every
  system (each run, each seed) still gets its own main interval; a pooled model gets mean, range and pooled counts
  in "models" (no interval of its own there) and its own pooled interval only in the new-name view. Without
  "pairs" every pair is computed, as before.
- "new_names": {"cutoff_chapter": 780}: the pre-registered new-name recall for every system and
  model: a scored gold span is new when its exact text is neither a dictionary name nor a mention name of chapters
  <= cutoff (novelty.weak_names, any type). A span system may carry its own "seen_names" rule
  ({"cutoff_chapter": N, "dictionary_cutoff": N}: mentions of chapters <= N, dictionary rows revealed by N whose
  name occurs in a chapter <= N) for a learning-curve tier, where "new" is counted from what that tier could use;
  the main table keeps the pre-registered rule. A "seen_names" in a config without "new_names" governs nothing and
  is refused.
- "training_snapshot": the snapshot the trained systems, the weak builds, the dictionary and the seen names come from
  when "snapshot" (the answers and the chapter text) is a later export of the book (stage 3: test2 read from to1350,
  the systems trained on to0990). Default: "snapshot" itself, and then nothing below applies and the run is the same as
  before this key existed. Otherwise every chapter of the training snapshot must be in the scoring snapshot with the
  same number and text (snapshot_lineage, recorded as "snapshots" in config.json and metrics.json); a span system's
  predict run must be on the scoring snapshot and record a snapshot_lineage from the training snapshot with
  shared_text_identical true (tagger predict --snapshot), while its training run and weak build are from the training
  snapshot (whose chapters recompute the build's training chapters); the dictionary system must take the training
  snapshot's dictionary ("dictionary": {"min_len": 2, "from_training_snapshot": true}; refused otherwise, since the later
  export's dictionary holds names from later chapters) and match it in the scoring snapshot's chapters; GLiNER reads the
  scoring snapshot's chapters; the new-name rule's seen names (dictionary + mentions <= cutoff) come from the training
  snapshot; the dev answers are checked against the snapshot they name (either of the two; the dev chapters are the same
  text in both), the test answers against the scoring snapshot.

runs/<run_id>/ gets config.json and metrics.json (numbers only, committed) and
predictions_<split>_<system>.jsonl (spans with text, not committed).
"""
import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.gold import check as check_gold
from novel_ner.llm_task import locate
from novel_ner.score import _prf, _touches, gold_spans, strict
from novel_ner.snapshot import PARAGRAPH_SEP, REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.zeroshot import chunk_chapter, environment, git_state

NON_PERSON = tuple(t for t in ENTITY_TYPES if t != "person")
DEV_SET = "configs/annotate_dev.json"  # the development chapters (31); no training chapter may be one
SPAN_SPEC_KEYS = {"name", "predictions_run", "input_condition", "seen_names", "note"}
SEEN_RULE_KEYS = {"cutoff_chapter", "dictionary_cutoff"}
SCHEDULE_ONLY_KEYS = {"max_epochs", "description"}  # the training-config keys in which one model's seeds may differ: a run the
# sufficiency rule flagged is re-run under a longer schedule and replaces the flagged run in its arm 


def done_chapters(export):
    """Chapters the user finished, in file order."""
    return [cid for cid, c in export["chapters"].items() if c.get("status") == "done"]


def paragraph_bases(chapter):
    """{paragraph_id: offset of the paragraph in the chapter text}."""
    bases, at = {}, 0
    for pid, text in chapter.paragraphs:
        bases[pid] = at
        at += len(text) + len(PARAGRAPH_SEP)
    return bases


def chapter_chunks(chapter, max_chars):
    """[(offset in the chapter text, chunk text)]: whole paragraphs joined as in the chapter text."""
    bases = paragraph_bases(chapter)
    out = []
    for text, spans in chunk_chapter(chapter, max_chars):
        start = bases[spans[0][0]]
        if chapter.text[start:start + len(text)] != text:
            raise ValueError(f"{chapter.chapter_id}: chunk at {start} is not a slice of the chapter text")
        out.append((start, text))
    return out


def gliner_predictions(model, variant, labels, chapters, cfg):
    """All GLiNER outputs scoring above cfg["save_threshold"], in chapter offsets, with raw float scores.

    A chunk is a contiguous slice of the chapter text, so every output has a chapter position; one that
    runs across a paragraph break is kept and scored like any other prediction (appendix 1: outputs are
    never dropped quietly). Decoding is flat and greedy by score, so keeping the outputs above a higher
    threshold afterwards gives exactly what decoding at that threshold would.
    """
    model.configure(variant["splitter"], variant["space_mode"], variant.get("fix_normalizer", True))
    names = [labels[t] for t in ENTITY_TYPES]
    back = {labels[t]: t for t in ENTITY_TYPES}
    inputs = [(ch, start, text) for ch in chapters for start, text in chapter_chunks(ch, cfg["chunk_max_chars"])]
    lengths = [model.token_count(text, names) for _, _, text in inputs]
    unk = sum(model.tokenizer(text)["input_ids"].count(model.tokenizer.unk_token_id) for _, _, text in inputs)
    began = time.perf_counter()
    out = model.predict([text for _, _, text in inputs], names, cfg["save_threshold"], cfg["batch_size"])
    seconds = time.perf_counter() - began
    predictions = []
    for (ch, base, _), entities in zip(inputs, out):
        for e in entities:
            start, end = base + e["start"], base + e["end"]
            if ch.text[start:end] != e["text"]:
                raise ValueError(f"{ch.chapter_id} [{start},{end}) does not hold {e['text']!r}")
            predictions.append({"chapter_id": ch.chapter_id, "start": start, "end": end, "text": e["text"],
                                "type": back[e["label"]], "score": float(e["score"])})
    stats = {"chunks": len(inputs), "max_encoder_tokens": max(lengths), "chunks_over_512_tokens": sum(n > 512 for n in lengths),
             "unk_tokens_in_text": unk, "inference_seconds": round(seconds, 2),
             "outputs_crossing_a_paragraph_break": sum(PARAGRAPH_SEP in p["text"] for p in predictions)}
    return predictions, stats


def dictionary_predictions(dictionary, chapters, min_len=2):
    """Longest-match-first, non-overlapping matches of dictionary names, left to right.

    Names shorter than min_len are skipped (single characters match everywhere). A scoped alias only
    counts inside its chapter range, and an alias with a reveal chapter ("known") only from that chapter
    on (化名、伪装按揭示章节生效). A name listed with more than one type is skipped: plain matching has no
    way to choose. Returns (predictions, names skipped for conflicting types).
    """
    out, conflicts = [], set()
    for ch in chapters:
        types = defaultdict(set)
        for row in dictionary:
            scope, known = row.get("scope"), row.get("known")
            if len(row["name"]) < min_len or (scope and not scope[0] <= ch.number <= scope[1]):
                continue
            if known and ch.number < known:
                continue
            types[row["name"]].add(row["type"])
        usable = {n: next(iter(t)) for n, t in types.items() if len(t) == 1}
        conflicts.update(n for n, t in types.items() if len(t) > 1)
        by_length = sorted(usable, key=len, reverse=True)
        text, at = ch.text, 0
        while at < len(text):
            name = next((n for n in by_length if text.startswith(n, at)), None)
            if name is None:
                at += 1
                continue
            out.append({"chapter_id": ch.chapter_id, "start": at, "end": at + len(name), "text": name,
                        "type": usable[name]})
            at += len(name)
    return out, sorted(conflicts)


def llm_predictions(answer, snapshot, chapter_ids):
    """A generative model's answer (drafting instructions v1 JSON) -> (predictions, bookkeeping), per appendix 1.

    pending true = an abstention (no prediction). A mention without a valid type (missing, null or not one of
    the six) that is not pending, one that cannot be located (unknown paragraph, missing text, occurrence that
    is not a positive integer, text not found), and every mention listed under a chapter that is not in this
    task are invalid predictions; the last kind is booked against the task's first chapter, since it has no
    place of its own. A chapter answered twice is read in full both times and flagged.
    """
    wanted = set(chapter_ids)
    preds, pending = [], []
    invalid, abstained, problems = Counter(), Counter(), Counter()
    answered = set()
    chapters = answer.get("chapters") if isinstance(answer, dict) else None
    for chapter in chapters if isinstance(chapters, list) else []:
        cid = chapter.get("chapter_id") if isinstance(chapter, dict) else None
        if cid not in wanted:
            problems["chapter_not_in_this_task"] += 1
            stray = chapter.get("mentions") if isinstance(chapter, dict) else None
            if stray is not None and not isinstance(stray, list):
                problems["chapter_without_a_mentions_list"] += 1
            stray = [m for m in stray if not (isinstance(m, dict) and m.get("pending") is True)] if isinstance(stray, list) else []
            if stray and chapter_ids:
                invalid[chapter_ids[0]] += len(stray)
                problems["mentions_under_a_chapter_not_in_this_task"] += len(stray)
            continue
        if cid in answered:
            problems["chapter_answered_twice"] += 1
        answered.add(cid)
        mentions = chapter.get("mentions")
        if not isinstance(mentions, list):
            problems["chapter_without_a_mentions_list"] += 1
            continue
        starts, offset = {}, 0
        for pid, text in snapshot.chapters[cid].paragraphs:
            starts[pid] = (offset, text)
            offset += len(text) + len(PARAGRAPH_SEP)
        for m in mentions:
            if not isinstance(m, dict):
                invalid[cid] += 1
                problems["mention_not_an_object"] += 1
                continue
            text, occurrence = m.get("text"), m.get("occurrence")
            where = starts.get(m.get("paragraph_id"))
            at = None
            if (where is not None and isinstance(text, str) and text and isinstance(occurrence, int)
                    and not isinstance(occurrence, bool) and occurrence >= 1):
                at = locate(where[1], text, occurrence)
            if "pending" in m and not isinstance(m["pending"], bool):
                problems["pending_not_true_or_false"] += 1  # only true abstains; the mention is read as not pending
            if m.get("pending") is True:
                abstained[cid] += 1
                if at is not None:
                    pending.append((cid, where[0] + at, where[0] + at + len(text)))
                continue
            if m.get("type") not in ENTITY_TYPES:
                invalid[cid] += 1
                problems["type_missing" if m.get("type") is None else "type_not_one_of_the_six"] += 1
                continue
            if at is None:
                invalid[cid] += 1
                problems["could_not_be_located"] += 1
                continue
            preds.append({"chapter_id": cid, "start": where[0] + at, "end": where[0] + at + len(text), "text": text,
                          "type": m["type"]})
    return preds, {"invalid_by_chapter": dict(invalid), "abstained_by_chapter": dict(abstained), "pending_places": pending,
                   "problems": dict(problems), "chapters_missing": sorted(wanted - answered)}


def dictionary_record(dictionary, snapshot, min_len):
    """What the dictionary system may use (appendix 1: declare a dictionary resource and its build range)."""
    ordered = sorted(snapshot.chapters.values(), key=lambda c: c.number)
    first = {}
    for row in dictionary:
        first[row["name"]] = next((c.number for c in ordered if row["name"] in c.text), None)
    seen = [n for n in first.values() if n is not None]
    return {
        "entries": len(dictionary),
        "by_source": dict(Counter(r["source"] for r in dictionary)),
        "by_type": {t: sum(1 for r in dictionary if r["type"] == t) for t in ENTITY_TYPES},
        "built_from": f"the novel project's alias registry, item library and unit names; its LLM extraction covers chapters <= {snapshot.meta.get('extracted_chapters')}",
        "latest_first_appearance_of_a_name": max(seen) if seen else None,
        "names_never_in_the_text": sum(1 for n in first.values() if n is None),
        "latest_reveal_chapter": max((r.get("known") or 0 for r in dictionary), default=0),
        "names_shorter_than_min_len_skipped": sum(1 for r in dictionary if len(r["name"]) < min_len),
    }


def output_record(predictions, threshold=None):
    """Appendix 1 bookkeeping for the outputs that count: invalid ones (no position), exact duplicates
    (removed), and one place given two types (both kept and scored)."""
    kept = [p for p in predictions if threshold is None or p["score"] > threshold]
    keys = [(p["chapter_id"], p["start"], p["end"], p["type"]) for p in kept]
    places = Counter(k[:3] for k in set(keys))
    return {"outputs": len(kept), "invalid_no_position": 0, "exact_duplicates_removed": len(keys) - len(set(keys)),
            "places_with_conflicting_types": sum(1 for n in places.values() if n > 1),
            "note": "every output carries a chapter position by construction; a position whose text did not match would stop the run"}


def as_prediction(predictions, chapter_ids, threshold=None):
    """{chapter_id: {"spans": {(start, end, type)}}} for scoring; keeps score > threshold if given (GLiNER's rule)."""
    out = {cid: {"spans": set()} for cid in chapter_ids}
    for p in predictions:
        if p["chapter_id"] in out and (threshold is None or p["score"] > threshold):
            out[p["chapter_id"]]["spans"].add((p["start"], p["end"], p["type"]))
    return out


def restrict(gold, prediction, keep, carry_invalid=False):
    """The same scoring problem limited to spans where keep(chapter_id, span) holds, applied alike to gold and predictions.

    Invalid predictions have no position, so a region view cannot place them; carry_invalid keeps them (as FP)
    for views that only remove some names, not regions of text."""
    g = {cid: {"spans": {s for s in ref["spans"] if keep(cid, s)}, "unscored": ref["unscored"]} for cid, ref in gold.items()}
    p = {cid: {"spans": {s for s in prediction[cid]["spans"] if keep(cid, s)},
               **({"errors": prediction[cid].get("errors", 0)} if carry_invalid else {})} for cid in gold}
    return g, p


def type_counts(gold, prediction):
    """{chapter_id: {type: [tp, fp, fn], "_invalid": [0, n, 0]}} under strict matching with the unscored ranges
    left out; "_invalid" holds outputs that could not be placed (prediction["errors"]), which only the overall
    micro counts (as FP)."""
    out = {}
    for cid, ref in gold.items():
        unscored = ref["unscored"]
        g = {s for s in ref["spans"] if not _touches(s, unscored)}
        p = {s for s in prediction[cid]["spans"] if not _touches(s, unscored)}
        c = {t: [0, 0, 0] for t in ENTITY_TYPES}
        for s in g & p:
            c[s[2]][0] += 1
        for s in p - g:
            c[s[2]][1] += 1
        for s in g - p:
            c[s[2]][2] += 1
        c["_invalid"] = [0, prediction[cid].get("errors", 0), 0]
        out[cid] = c
    return out


def _prf_defined(tp, fp, fn):
    """_prf, but precision (recall) is None when nothing was predicted (nothing was there): undefined, not 0.
    F1 = 2TP / (2TP + FP + FN) is defined whenever there is anything to count (0 if TP is 0)."""
    m = _prf(tp, fp, fn)
    if tp + fp == 0:
        m["precision"] = None
    if tp + fn == 0:
        m["recall"] = None
    m["f1"] = round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None
    return m


def summarize(rows, macro_types=None):
    """Scores from a list of per-chapter {type: [tp, fp, fn]}: micro, non-person micro, per type, macro.

    The macro is averaged over macro_types when given (a fixed list, so every bootstrap round averages the
    same types; a listed type with nothing in a round counts F1 0 there), otherwise over the types with
    support in rows.
    """
    total = {t: [sum(r[t][i] for r in rows) for i in range(3)] for t in ENTITY_TYPES}
    invalid = sum(r.get("_invalid", [0, 0, 0])[1] for r in rows)
    micro = _prf_defined(*(sum(total[t][i] for t in ENTITY_TYPES) + (invalid if i == 1 else 0) for i in range(3)))
    non_person = _prf_defined(*(sum(total[t][i] for t in NON_PERSON) for i in range(3)))
    per_type = {}
    for t, (tp, fp, fn) in total.items():
        if tp + fn == 0:
            per_type[t] = {"tp": tp, "fp": fp, "fn": fn, "support": 0, "predicted": tp + fp, "note": "无可评样本"}
            continue
        m = _prf(tp, fp, fn)
        if tp + fp == 0:
            m["precision"] = None  # nothing predicted: precision is undefined, not 0
        per_type[t] = {**m, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn, "predicted": tp + fp}
    scored = list(macro_types) if macro_types is not None else [t for t in ENTITY_TYPES if per_type[t]["support"]]
    macro = round(sum(_raw_f1(*total[t]) for t in scored) / len(scored), 4) if scored else None
    return {"micro": micro, "non_person_micro": non_person, "macro_f1": macro, "macro_over": scored, "per_type": per_type}


def brief(m):
    """tp/fp/fn/P/R/F1 of a score.strict result; precision is None when nothing was predicted (undefined, not 0)."""
    out = {k: m[k] for k in ("tp", "fp", "fn")}
    if out["tp"] + out["fp"] + out["fn"] == 0:
        return {**out, "note": "无可评样本"}
    return {**out, **_prf_defined(out["tp"], out["fp"], out["fn"])}


def _raw_f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def round_stats(rows, macro_types):
    """Unrounded statistics for one bootstrap round (rounding happens once, on the interval ends)."""
    total = {t: [sum(r[t][i] for r in rows) for i in range(3)] for t in ENTITY_TYPES}
    invalid = sum(r.get("_invalid", [0, 0, 0])[1] for r in rows)
    tp, fp, fn = (sum(total[t][i] for t in ENTITY_TYPES) for i in range(3))
    fp += invalid
    np_tp, np_fp, np_fn = (sum(total[t][i] for t in NON_PERSON) for i in range(3))
    types = macro_types if macro_types is not None else [t for t in ENTITY_TYPES if total[t][0] + total[t][2]]
    return {"micro_f1": _raw_f1(tp, fp, fn), "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0, "non_person_f1": _raw_f1(np_tp, np_fp, np_fn),
            "macro_f1": sum(_raw_f1(*total[t]) for t in types) / len(types) if types else 0.0,
            "_support": {t: total[t][0] + total[t][2] for t in ENTITY_TYPES}}


def _quantile_interval(values):
    values = sorted(values)
    return [round(values[int(0.025 * (len(values) - 1))], 4), round(values[int(0.975 * (len(values) - 1))], 4)]


def bootstrap(counts, strata, n, seed, mode="within_strata", macro_types=None, pairs=None):
    """Paired percentile bootstrap. counts = {system: {chapter_id: {type: [tp, fp, fn]}}}; strata = {chapter_id: unit}.

    mode "within_strata": each round draws, inside every plot unit, as many chapters as it has, with
    replacement (the test set was sampled stratified by unit; appendix 1). mode "units": each round draws
    whole plot units with replacement (a sensitivity check for correlation inside a storyline; few units).
    All systems use the same draw in a round, so differences are paired. Every statistic is recomputed from
    the pooled counts of the round, never by averaging chapter F1. The macro uses the fixed macro_types in
    every round; how many rounds drew no gold span of one of them is recorded. pairs = [(a, b)] limits the
    paired differences to a - b for those pairs; None means every pair in counts order. The draws depend only
    on seed, strata and n, so every call with the same three draws the same chapters.
    """
    systems = list(counts)
    units = defaultdict(list)
    for cid in sorted(strata):
        units[strata[cid]].append(cid)
    unit_ids = sorted(units)
    rng = random.Random(seed)
    stats = ("micro_f1", "micro_precision", "micro_recall", "non_person_f1", "macro_f1")
    draws = {s: {k: [] for k in stats} for s in systems}
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(systems) for b in systems[i + 1:]]
    else:
        pairs = [tuple(p) for p in pairs]
        if any(len(p) != 2 or p[0] == p[1] or p[0] not in counts or p[1] not in counts for p in pairs):
            raise ValueError("every pair must name two different counted systems")
    diffs = {(a, b): {k: [] for k in ("micro_f1", "micro_precision", "micro_recall", "non_person_f1", "macro_f1")}
             for a, b in pairs}
    absent = 0
    for _ in range(n):
        if mode == "within_strata":
            sample = [rng.choice(units[u]) for u in unit_ids for _ in units[u]]
        elif mode == "units":
            sample = [cid for u in (rng.choice(unit_ids) for _ in unit_ids) for cid in units[u]]
        else:
            raise ValueError(mode)
        value = {}
        for s in systems:
            # unrounded; a round where nothing was predicted (or nothing was there) counts P/R as 0
            value[s] = round_stats([counts[s][c] for c in sample], macro_types)
            for k in stats:
                draws[s][k].append(value[s][k])
        if macro_types and any(value[systems[-1]]["_support"][t] == 0 for t in macro_types):
            absent += 1  # support is the gold count, the same for every system in a round
        for a, b in pairs:
            for k in diffs[(a, b)]:
                diffs[(a, b)][k].append(value[a][k] - value[b][k])
    return {
        "resampling": {"mode": mode, "unit": "chapter within plot unit" if mode == "within_strata" else "plot unit",
                       "rounds": n, "seed": seed, "interval": "percentile 2.5% / 97.5%",
                       "plot_units": {u: len(units[u]) for u in unit_ids},
                       "macro_types": list(macro_types) if macro_types is not None else "types with support in each round",
                       "rounds_without_gold_of_a_macro_type": absent},
        "systems": {s: {k: _quantile_interval(v) for k, v in draws[s].items()} for s in systems},
        "paired_difference": {f"{a} - {b}": {k: {"ci95": _quantile_interval(v),
                                                 "share_of_rounds_above_0": round(sum(x > 0 for x in v) / len(v), 4)}
                                             for k, v in diffs[(a, b)].items()} for a, b in pairs},
    }


def unscored_report(export, gold, chapter_ids, snapshot, predictions):
    """Appendix 1: the excluded ranges, why, how much text they cover, and what they removed from each system."""
    ranges = [(cid, u) for cid in chapter_ids for u in export["chapters"][cid].get("unscored", [])]
    pending = [(cid, a) for cid in chapter_ids for a in export["chapters"][cid]["annotations"] if a.get("pending")]
    characters = sum(len(snapshot.chapters[cid].text) for cid in chapter_ids)
    covered = 0
    for cid in chapter_ids:  # characters under any range or pending mark, overlaps counted once
        spots = set()
        for x in export["chapters"][cid].get("unscored", []) + [a for a in export["chapters"][cid]["annotations"] if a.get("pending")]:
            spots.update(range(x["start"], x["end"]))
        covered += len(spots)
    return {
        "recorded_ranges": len(ranges), "pending_marks": len(pending),
        "by_reason": dict(Counter(u["reason"] for _, u in ranges)),
        "by_text": dict(Counter(u["text"] for _, u in ranges)),
        "chapters_touched": len({cid for cid, _ in ranges} | {cid for cid, _ in pending}),
        "characters_covered": covered, "share_of_test_text": round(covered / characters, 5) if characters else None,
        "settled_gold_spans_overlapping_them": sum(1 for cid, ref in gold.items() for s in ref["spans"] if _touches(s, ref["unscored"])),
        "predicted_spans_removed": {s: sum(1 for cid, ref in gold.items() for x in p[cid]["spans"] if _touches(x, ref["unscored"]))
                                    for s, p in predictions.items()},
    }


def gold_origin(export, chapter_ids):
    """{chapter_id: {(start, end, type): "gpt_agreed" | "user_or_sheet" | "reviewer"}} for the settled gold spans:
    whether the reviewer model decided it on the user's notes (answer v3 on), both GPT passes gave it identically, or
    the user (or the name sheet) decided it. Also returns {chapter_id: task}."""
    origin, seen = {}, {}
    wanted = set(chapter_ids)
    for task in export.get("llm_drafts", {}):
        drafts = [json.loads((REPO_ROOT / "data" / "llm_tasks" / task / f"draft_{r}.json").read_text(encoding="utf-8"))
                  for r in ("A", "B")]
        if set(drafts[0]["chapters"]) != set(drafts[1]["chapters"]):
            raise SystemExit(f"{task}: the two passes cover different chapters")
        for cid in sorted(set(drafts[0]["chapters"]) & wanted):
            if cid in seen:
                raise SystemExit(f"{cid} is drafted in both {seen[cid]} and {task}")
            seen[cid] = task
            both = set.intersection(*({(s["start"], s["end"], s["type"]) for s in d["chapters"][cid]["spans"] if not s["pending"]}
                                      for d in drafts))
            origin[cid] = {(a["start"], a["end"], a["type"]): "reviewer" if a.get("decider") == "reviewer"
                           else "gpt_agreed" if (a["start"], a["end"], a["type"]) in both
                           else "user_or_sheet" for a in export["chapters"][cid]["annotations"] if not a.get("pending")}
    return origin, seen


def gold_record(export, chapter_ids, snapshot, calibration_metrics):
    """How the silver-standard answers were made: provenance, how many spans both GPT passes gave, the calibration."""
    origin, seen = gold_origin(export, chapter_ids)
    kinds = Counter(o for c in origin.values() for o in c.values())
    agreed, settled = kinds["gpt_agreed"], kinds["user_or_sheet"]
    doubled = [c for c in chapter_ids if export["chapters"][c].get("provenance") == "model_double_adjudicated"]
    if set(doubled) - set(seen):
        raise SystemExit(f"no GPT drafts recorded for {sorted(set(doubled) - set(seen))[:5]}")
    calib = json.loads((REPO_ROOT / calibration_metrics).read_text(encoding="utf-8"))
    drafts_info = export.get("llm_drafts", {}).values()
    return {
        "guide_version": export.get("guide_version"),
        "drafting": {"models": sorted({r["model"] for t in drafts_info for r in t.values()}),
                     "instructions_versions": sorted({str(r["instructions_version"]) for t in drafts_info for r in t.values()})},
        "provenance": dict(Counter(export["chapters"][c].get("provenance") for c in chapter_ids)),
        "settled_spans_given_identically_by_both_gpt_passes": agreed,
        "settled_spans_decided_by_the_user_or_the_name_sheet": settled,
        "settled_spans_decided_by_the_reviewer_on_the_users_notes": kinds["reviewer"],
        "cross_chapter_flags": sum(1 for c in chapter_ids for a in export["chapters"][c]["annotations"]
                                   if a.get("cross_chapter") and not a.get("pending")),
        "calibration": {"run": calibration_metrics, "chapters": calib["chapters"], "gold_file": calib["gold_file"],
                        "passes_vs_user": {r: {k: d["vs_gold"][k] for k in ("precision", "recall", "f1")}
                                           for r, d in calib["drafts"].items()}},
        "note": "silver standard: GPT annotated each chapter twice, the user decided disagreements, pending marks and cross-chapter names",
    }


def run_agreement(pairs, systems, gold, chapter_ids):
    """Two runs of one model with identical settings: how far their outputs agree (one run scored against the
    other) and how far apart their gold scores are (second minus first, unrounded until the end). The gold's
    unscored ranges are left out everywhere, abstention places included. A small score gap can hide changes
    that cancel out; read it with the output agreement."""
    out = {}
    for a, b in pairs:
        ref = {c: {"spans": systems[a][c]["spans"], "unscored": gold[c]["unscored"]} for c in chapter_ids}
        other = {c: {"spans": systems[b][c]["spans"]} for c in chapter_ids}
        places = {s: {c: {x for x in systems[s][c].get("pending", []) if not _touches(x, gold[c]["unscored"])} for c in chapter_ids}
                  for s in (a, b)}
        stats = {s: round_stats(list(type_counts(gold, systems[s]).values()), None) for s in (a, b)}
        out[f"{a} vs {b}"] = {
            "output_agreement": brief(strict(ref, other)),
            "abstention_places": {a: sum(len(v) for v in places[a].values()), b: sum(len(v) for v in places[b].values()),
                                  "same_place_in_both": sum(len(places[a][c] & places[b][c]) for c in chapter_ids)},
            "scores": {s: {k: round(stats[s][k], 4) for k in ("micro_f1", "micro_precision", "micro_recall", "non_person_f1", "macro_f1")}
                       for s in (a, b)},
            "second_minus_first": {k: round(stats[b][k] - stats[a][k], 4)
                                   for k in ("micro_f1", "micro_precision", "micro_recall", "non_person_f1", "macro_f1")},
        }
    return out


def pooled_counts(runs, counts):
    """One model's runs (or seeds) pooled: every run's TP/FP/FN added per chapter and type."""
    return {c: {t: [sum(counts[r][c][t][i] for r in runs) for i in range(3)] for t in counts[runs[0]][c]}
            for c in counts[runs[0]]}


def model_summary(models, counts, strata, b, macro_types=None, pairs=None, about=None):
    """Several runs of one model (or seeds of one trained setting): each run, the mean and range (min, max) over
    runs, pooled counts (all runs' TP/FP/FN added per chapter) and a paired chapter bootstrap between the pooled
    models (only the differences are kept; a pooled model gets no interval of its own here). Run-to-run variation
    is still not in that interval; the runs are fixed outputs. pairs = [(a, b)] of
    model names limits the pooled differences (None: every pair); about = {model: fields} adds facts about a model
    (seeds and setting of a trained one)."""
    pooled = {}
    out = {}
    for name, runs in models.items():
        pooled[name] = pooled_counts(runs, counts)
        per_run = {r: round_stats(list(counts[r].values()), None) for r in runs}
        stat = lambda k: [per_run[r][k] for r in runs]
        out[name] = {"runs": runs, "n_runs": len(runs), **(about or {}).get(name, {}),
                     "mean": {k: round(sum(stat(k)) / len(runs), 4) for k in ("micro_f1", "non_person_f1", "macro_f1")},
                     "range": {k: [round(min(stat(k)), 4), round(max(stat(k)), 4)] for k in ("micro_f1", "non_person_f1", "macro_f1")},
                     "pooled_counts_micro_f1": round(round_stats(list(pooled[name].values()), None)["micro_f1"], 4)}
    wanted = len(pooled) > 1 if pairs is None else bool(pairs)
    ci = bootstrap(pooled, strata, b["n"], b["seed"], "within_strata", macro_types, pairs) if wanted else None
    return {"models": out, "pooled_paired": ci["paired_difference"] if ci else (None if pairs is None else {}),
            "note": ("pooled = every run's counts added per chapter, so each model counts once per run; the paired interval "
                     "covers chapter sampling only; macro over the same fixed type list as the main intervals")}


def resolve_pairs(pairs, systems, models):
    """The config's "pairs" -> (system pairs, model pairs), each [(a, b)]; (None, None) without it (every pair).

    Both names of a pair must be systems (main intervals) or both models (pooled comparison); a name that is both is
    ambiguous. A pair listed twice, in either order, is refused."""
    if pairs is None:
        return None, None
    systems, models = set(systems), set(models or {})
    out = {"system": [], "model": []}
    seen = set()
    for pair in pairs:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair) and pair[0] != pair[1]):
            raise SystemExit(f"pairs: {pair!r} is not two different names")
        kinds = []
        for name in pair:
            if name in systems and name in models:
                raise SystemExit(f"pairs: {name!r} is both a system and a model; rename the model")
            if name not in systems and name not in models:
                raise SystemExit(f"pairs: {name!r} is neither a system nor a model in this config")
            kinds.append("system" if name in systems else "model")
        if kinds[0] != kinds[1]:
            raise SystemExit(f"pairs: {pair!r} joins a system and a model; list the system as a one-run model to compare it pooled")
        if frozenset(pair) in seen:
            raise SystemExit(f"pairs: {pair!r} is listed twice")
        seen.add(frozenset(pair))
        out[kinds[0]].append(tuple(pair))
    return out["system"], out["model"]


def run_folder(ref, what):
    """REPO_ROOT/runs/<id> for "runs/<id>" or "<id>" (one folder name; anything nested or relative is refused)."""
    parts = Path(ref).parts if isinstance(ref, str) and ref else ()
    if len(parts) == 2 and parts[0] == "runs":
        parts = parts[1:]
    if len(parts) != 1 or parts[0] in (".", "..") or Path(parts[0]).name != parts[0]:
        raise SystemExit(f"{what}: {ref!r} must be runs/<id> or <id>")
    return REPO_ROOT / "runs" / parts[0]


def read_record(path, what):
    if not path.is_file():
        raise SystemExit(f"{what}: {path.relative_to(REPO_ROOT).as_posix()} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def span_system(spec, snapshot, test_ids, dev_ids, order_ids, training=None):
    """A trained system's test predictions from a tagger predict run (tagger.predict_run), checked before scoring.

    training: the snapshot the training run and the weak build are from when `snapshot` (the scored text) is a later
    export of it (config "training_snapshot"); default `snapshot` itself, which changes nothing below.
    Stops with a message (never quoting a prediction's text) unless:
    - the predict run's config.json and metrics.json both name the run's own folder (run_id), so neither file was
      copied in from another run;
    - the predict run is from this snapshot and, when the training snapshot is another, records a snapshot_lineage from
      it with shared_text_identical true (tagger predict --snapshot); its predictions.jsonl is the file its metrics.json
      hashed (and holds as many rows as it counted);
    - its chapter list is exactly the test chapters scored here, no more, no fewer, each once;
    - every row is in one of those chapters, its [start, end) holds exactly its text in the chapter text, and its
      type is one of the six;
    - the training run it names exists, is the config.json the predict run hashed, finished (metrics.json), is from
      the training snapshot and has the predict run's setting and seed;
    - the predict run, the training run and the weak build were made from committed code (code.dirty false);
    - the training chapters are all <= 780 and share no chapter with the scored test chapters, the test order or
      the development chapters (dev_ids): the weak build's training chapters are recomputed from the build's own
      record (weak.split_ids on its dev set, test order and max_train_chapter) and must give the sha256 that the
      build and the training run recorded; the run's label file is the one the build recorded for its setting;
    - the dev answers that chose the training run's checkpoint are development chapters.
    Returns (prediction, rows, record for metrics.json, facts for pooling seeds).
    """
    from novel_ner.teacher import TEACHER_BUILD_KIND, build_train_ids
    from novel_ner.weak import LAST_TRAINING_CHAPTER, ids_sha256, split_ids

    name = spec.get("name") if isinstance(spec, dict) else None
    what = f"span system {name!r}"
    training = training or snapshot
    if not isinstance(name, str) or not name:
        raise SystemExit("every span system needs a name")
    unknown = set(spec) - SPAN_SPEC_KEYS
    if unknown:
        raise SystemExit(f"{what}: unknown keys {sorted(unknown)} (allowed: {sorted(SPAN_SPEC_KEYS)})")
    if not isinstance(spec.get("input_condition"), str) or not spec["input_condition"].strip():
        raise SystemExit(f"{what}: give its input_condition")
    run_dir = run_folder(spec.get("predictions_run"), what)
    rec = read_record(run_dir / "config.json", what)
    met = read_record(run_dir / "metrics.json", what)
    if not (rec.get("run_id") == met.get("run_id") == run_dir.name):
        raise SystemExit(f"{what}: runs/{run_dir.name}'s config.json and metrics.json name runs {rec.get('run_id')!r} and "
                         f"{met.get('run_id')!r}; both must be {run_dir.name!r} (files copied between runs?)")
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.is_file():
        raise SystemExit(f"{what}: runs/{run_dir.name}/predictions.jsonl does not exist")
    if (rec.get("snapshot") or {}).get("id") != snapshot.id:
        raise SystemExit(f"{what}: the predict run is from snapshot {(rec.get('snapshot') or {}).get('id')}, not {snapshot.id}")
    lineage = rec.get("snapshot_lineage")  # tagger.predict_run's record when it read a later export than the model's own
    if training.id != snapshot.id and not (isinstance(lineage, dict) and lineage.get("model_snapshot") == training.id
                                           and lineage.get("shared_text_identical") is True):
        raise SystemExit(f"{what}: the predict run on snapshot {snapshot.id} does not record a lineage from the training snapshot "
                         f"{training.id} with identical shared text (snapshot_lineage = {lineage})")
    digest = sha256(pred_file)
    if digest != met.get("predictions_sha256"):
        raise SystemExit(f"{what}: runs/{run_dir.name}/predictions.jsonl is not the file its metrics.json hashed")
    listed = rec.get("chapters")
    vote = rec.get("predictions_run") if "vote_rule" in rec else None  # a C2 vote run (vote.vote_run) revises a predict run's types
    if listed is None and isinstance(vote, dict):
        listed = vote.get("chapters")  # the vote's chapters are its target predict run's
    if not isinstance(listed, list) or len(set(listed)) != len(listed):
        raise SystemExit(f"{what}: the predict run's chapter list is missing or names a chapter twice")
    extra, missing = set(listed) - set(test_ids), set(test_ids) - set(listed)
    if extra or missing:
        raise SystemExit(f"{what}: the predict run covers {len(extra)} chapters not scored here and misses {len(missing)} "
                         f"scored ones; it must cover exactly the {len(test_ids)} test chapters")
    if (rec.get("code") or {}).get("dirty") is not False:
        raise SystemExit(f"{what}: the predict run was not made from committed code (code.dirty is not false)")
    if vote is not None:
        target_cfg = run_folder(vote.get("run_id"), f"{what} (vote target)") / "config.json"
        if not target_cfg.is_file() or sha256(target_cfg) != vote.get("config_sha256"):
            raise SystemExit(f"{what}: the vote's target predict run runs/{target_cfg.parent.name}/config.json is not the file "
                             "the vote hashed")
    rows = read_jsonl(pred_file)
    if met.get("outputs") != len(rows):
        raise SystemExit(f"{what}: predictions.jsonl holds {len(rows)} rows, its metrics.json counted {met.get('outputs')}")
    wanted = set(test_ids)
    for i, r in enumerate(rows, 1):
        cid, s, e, text = r.get("chapter_id"), r.get("start"), r.get("end"), r.get("text")
        ok = (cid in wanted and r.get("type") in ENTITY_TYPES and isinstance(text, str)
              and all(isinstance(x, int) and not isinstance(x, bool) for x in (s, e)) and 0 <= s < e
              and e - s == len(text) and snapshot.chapters[cid].text[s:e] == text)
        if not ok:
            raise SystemExit(f"{what}: prediction row {i} is outside the scored chapters, has a type that is not one of "
                             "the six, or its text is not the chapter text at its offsets")

    model_dir = run_folder(rec.get("model_run"), f"{what} (model_run)")
    model_cfg = model_dir / "config.json"
    if not model_cfg.is_file():
        raise SystemExit(f"{what}: the training run runs/{model_dir.name} has no config.json")
    if sha256(model_cfg) != rec.get("model_run_config_sha256"):
        raise SystemExit(f"{what}: runs/{model_dir.name}/config.json is not the file the predict run hashed")
    trained = json.loads(model_cfg.read_text(encoding="utf-8"))
    trained_metrics = read_record(model_dir / "metrics.json", f"{what} (training run not finished)")
    if (trained.get("snapshot") or {}).get("id") != training.id:
        raise SystemExit(f"{what}: the training run is from snapshot {(trained.get('snapshot') or {}).get('id')}, not {training.id}")
    if (trained.get("code") or {}).get("dirty") is not False:
        raise SystemExit(f"{what}: the training run runs/{model_dir.name} was not made from committed code (code.dirty is not false)")
    if (trained.get("setting"), trained.get("seed")) != (rec.get("setting"), rec.get("seed")):
        raise SystemExit(f"{what}: the predict run's setting and seed are not its training run's")
    weak = trained.get("weak_labels") or {}
    numbers = weak.get("train_chapter_numbers")
    if not (isinstance(numbers, list) and len(numbers) == 2 and 1 <= numbers[0] <= numbers[1] <= LAST_TRAINING_CHAPTER):
        raise SystemExit(f"{what}: the training run does not record training chapters within 1-{LAST_TRAINING_CHAPTER}")
    build_dir = run_folder(weak.get("build"), f"{what} (weak build)")
    build = read_record(build_dir / "config.json", f"{what} (weak build record)")
    if (build.get("snapshot") or {}).get("id") != training.id:
        raise SystemExit(f"{what}: weak build {build_dir.name} is from snapshot {(build.get('snapshot') or {}).get('id')}, not {training.id}")
    if (build.get("code") or {}).get("dirty") is not False:
        raise SystemExit(f"{what}: weak build {build_dir.name} was not made from committed code (code.dirty is not false)")
    split = build.get("split") or {}
    if not split.get("train_ids_sha256") or split["train_ids_sha256"] != weak.get("train_ids_sha256"):
        raise SystemExit(f"{what}: the training run's chapters (sha256) are not weak build {build_dir.name}'s")
    if ((build.get("labels") or {}).get(trained.get("setting")) or {}).get("sha256") != weak.get("sha256"):
        raise SystemExit(f"{what}: the training run's label file is not the one weak build {build_dir.name} recorded for "
                         f"setting {trained.get('setting')}")
    bcfg = build.get("config") or {}
    try:
        if build.get("kind") == TEACHER_BUILD_KIND:  # a sample of the training chapters: recomputed from the build's packs
            train = build_train_ids(build, training.chapters, REPO_ROOT)
        else:
            build_dev = chapter_order(json.loads((REPO_ROOT / bcfg["dev_config"]).read_text(encoding="utf-8")))
            build_order = [r["chapter_id"] for r in json.loads((REPO_ROOT / bcfg["test_order"]).read_text(encoding="utf-8"))["order"]]
            train = split_ids(training.chapters, build_dev, build_order, bcfg["max_train_chapter"])["train"]
    except (KeyError, TypeError, OSError, ValueError) as err:
        raise SystemExit(f"{what}: weak build {build_dir.name}'s training chapters cannot be recomputed from its record "
                         f"({type(err).__name__}: {err})")
    if ids_sha256(train) != split["train_ids_sha256"]:
        raise SystemExit(f"{what}: weak build {build_dir.name}'s training chapters, recomputed from its record, do not give "
                         "the sha256 it recorded")
    if len(train) != weak.get("train_chapters"):
        raise SystemExit(f"{what}: the training run counts {weak.get('train_chapters')} training chapters, the build gives {len(train)}")
    late = sum(1 for c in train if training.chapters[c].number > LAST_TRAINING_CHAPTER)
    on_test = len(set(train) & (set(test_ids) | set(order_ids)))
    on_dev = len(set(train) & set(dev_ids))
    if late or on_test or on_dev:
        raise SystemExit(f"{what}: training chapters after chapter {LAST_TRAINING_CHAPTER}: {late}; in the test chapters or "
                         f"test order: {on_test}; development chapters: {on_dev}")
    dev_answers = trained.get("gold_dev") or {}
    if not set(dev_answers.get("chapters") or []) <= set(dev_ids):
        raise SystemExit(f"{what}: the dev answers that chose the training run's checkpoint are not all development chapters")
    at_best = ((trained_metrics.get("dev") or {}).get("at_best") or {}).get("micro") or {}
    snapshot_check = "snapshot" if training.id == snapshot.id else (
        f"predict run on the scoring snapshot {snapshot.id} with a recorded lineage from the training snapshot {training.id} "
        "(shared text identical); training run and weak build from the training snapshot")
    record = {
        "input_condition": spec["input_condition"],
        "predictions_run": f"runs/{run_dir.name}", "predictions_sha256": digest,
        "predict_code": rec.get("code"), "predict_stats": met.get("predict_stats"),
        "model_run": f"runs/{model_dir.name}", "model_run_config_sha256": rec["model_run_config_sha256"],
        "model_code": trained.get("code"), "setting": trained.get("setting"), "seed": trained.get("seed"),
        "training_config_file": trained.get("config_file"),
        "backbone": {k: (trained.get("backbone") or {}).get(k) for k in ("id", "revision")},
        "chunking": rec.get("chunking") if vote is None else vote.get("chunking"),
        "weak_build": {"build": f"runs/{build_dir.name}", "kind": build.get("kind"), "labels_sha256": weak.get("sha256"),
                       "build_code": build.get("code"),
                       "max_train_chapter": bcfg.get("max_train_chapter"), "dictionary_cutoff": bcfg.get("dictionary_cutoff"),
                       "train_chapters": len(train), "train_ids_sha256": split["train_ids_sha256"],
                       "train_chapter_numbers": [min(training.chapters[c].number for c in train),
                                                 max(training.chapters[c].number for c in train)]},
        "dev_answers": {"file": dev_answers.get("file"), "sha256": dev_answers.get("sha256"),
                        "chapters": len(dev_answers.get("chapters") or [])},
        "dev_at_best": {k: at_best.get(k) for k in ("precision", "recall", "f1")},
        "checks": [snapshot_check, "predictions sha256 = predict run's record", "chapters = the scored test chapters",
                   "every row's text at its offsets, type one of the six", "training run config sha256 = predict run's record",
                   "predict run, training run and weak build from committed code",
                   "predict run's config.json and metrics.json name the run's own folder",
                   f"training chapters recomputed from the weak build record: all <= {LAST_TRAINING_CHAPTER}, none in the "
                   "test chapters, the test order or the development chapters",
                   "checkpoint chosen on development chapters"],
        "outputs": output_record(rows),
        "outputs_crossing_a_paragraph_break": sum(1 for r in rows if PARAGRAPH_SEP in r["text"]),
    }
    if spec.get("note"):
        record["note"] = spec["note"]
    if training.id != snapshot.id:
        record["training_snapshot"] = training.id
        record["snapshot_lineage"] = lineage
    if "seen_names" in spec:
        record["new_name_seen_rule"] = spec["seen_names"]
    if vote is not None:
        material = rec.get("material_run") or {}
        record["vote"] = {"rule": rec.get("vote_rule"), "material_rule": rec.get("material_rule"),
                          "predictions_run": f"runs/{vote.get('run_id')}", "material_run": f"runs/{material.get('run_id')}",
                          "material_chapters": material.get("chapters"), "material_chapter_ids_sha256": material.get("chapter_ids_sha256"),
                          "material_equals_model_training_chapters": material.get("equals_model_training_chapters"),
                          "material_chunking": material.get("chunking")}
        record["checks"].append("a vote run (C2): its chapters are its target predict run's, whose config.json is the file the vote hashed")
    config = trained.get("config") or {}
    key = lambda cfg: hashlib.sha256(json.dumps([trained.get("setting"), cfg, weak.get("sha256"), split["train_ids_sha256"]],
                                                sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    pool = {"setting": trained.get("setting"), "seed": trained.get("seed"), "weak_build": f"runs/{build_dir.name}",
            "key": key(trained.get("config")),
            "schedule_free_key": key({k: v for k, v in config.items() if k not in SCHEDULE_ONLY_KEYS}),
            "max_epochs": config.get("max_epochs")}
    return as_prediction(rows, test_ids), rows, record, pool


def check_models(models, names, pool):
    """Check the config's "models" against the systems; returns {model: facts} for the models made of trained runs.

    Every listed run must be a system, listed once. A model with a trained span system must hold only trained
    span systems that differ in nothing but the seed (setting, training config, label file, training chapters),
    each seed once; training configs may differ in SCHEDULE_ONLY_KEYS alone (a seed re-run under a longer schedule
    by the sufficiency rule), which is then recorded per run (max_epochs_by_run)."""
    about = {}
    for model, runs in (models or {}).items():
        if not isinstance(runs, list) or not runs or len(set(runs)) != len(runs):
            raise SystemExit(f"model {model!r}: list its runs, each once")
        unknown = [r for r in runs if r not in names]
        if unknown:
            raise SystemExit(f"model {model!r}: {unknown} are not systems in this config")
        trained = [r for r in runs if r in pool]
        if not trained:
            continue
        if len(trained) != len(runs):
            raise SystemExit(f"model {model!r} mixes trained span systems with other systems")
        schedules = None
        if len({pool[r]["key"] for r in runs}) != 1:
            if len({pool[r].get("schedule_free_key") for r in runs}) != 1:
                raise SystemExit(f"model {model!r}: its runs differ in more than the seed (setting, training config, labels or chapters)")
            schedules = {r: pool[r].get("max_epochs") for r in runs}
        seeds = [pool[r]["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"model {model!r}: two of its runs have the same seed")
        about[model] = {"setting": pool[runs[0]]["setting"], "weak_build": pool[runs[0]]["weak_build"], "seeds": seeds}
        if schedules:
            about[model]["max_epochs_by_run"] = schedules
            about[model]["schedule_note"] = (f"the runs' training configs differ only in {sorted(SCHEDULE_ONLY_KEYS)}: a run the "
                                             "sufficiency rule flagged was re-run under a longer schedule and replaces the flagged "
                                             "run in its arm ; the flagged run is not in this model")
    return about


def seen_rule_label(rule):
    d = rule.get("dictionary_cutoff")
    return f"mentions<={rule['cutoff_chapter']};dictionary={'all' if d is None else f'<={d}'}"


def seen_names(snapshot, rule, first_test_chapter, what):
    """(label, names, record) for a seen-set rule {"cutoff_chapter": N[, "dictionary_cutoff": M]} (novelty.weak_names)."""
    from novel_ner.novelty import weak_names

    if not isinstance(rule, dict) or set(rule) - SEEN_RULE_KEYS or "cutoff_chapter" not in rule:
        raise SystemExit(f"{what}: a seen-set rule is {{'cutoff_chapter': N}} with an optional 'dictionary_cutoff'")
    for key in SEEN_RULE_KEYS:
        v = rule.get(key)
        if v is None and key == "dictionary_cutoff":
            continue
        if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < first_test_chapter:
            raise SystemExit(f"{what}: {key} must be a chapter number before the first test chapter ({first_test_chapter})")
    dictionary, mentions, sources = weak_names(snapshot, rule["cutoff_chapter"], rule.get("dictionary_cutoff"))
    return seen_rule_label(rule), frozenset(dictionary | mentions), {"rule": dict(rule), "sources": sources}


def _new_name_point(tp, fp, fn, interval=None):
    m = _prf_defined(tp, fp, fn)
    out = {"tp": tp, "fp": fp, "fn": fn, "new_gold_spans": tp + fn, "recall": m["recall"],
           "precision_among_predictions_with_unseen_text": m["precision"]}
    if interval is not None:
        out["recall_ci95"] = interval["micro_recall"]
        out["precision_among_predictions_with_unseen_text_ci95"] = interval["micro_precision"] if m["precision"] is not None else None
    return out


def new_name_view(gold, systems, texts, seen_of, strata, b, models=None, system_pairs=None, model_pairs=None):
    """Recall on the gold spans whose exact text is unseen (the pre-registered extra analysis).

    seen_of = {system: set of seen names}. Gold and predictions are restricted alike (restrict) to spans whose text
    is not seen, so recall = TP_new / (TP_new + FN_new) exactly: a hit has its gold span's text, so restricting the
    predictions never removes one. The view's precision is another number, precision among the predictions whose
    text is unseen, and is named so. Invalid outputs (no position) are left out. Intervals: the paired chapter
    bootstrap within plot units (same seed and rounds as the main intervals, so the same draws); a model's runs are
    pooled per chapter as in model_summary, so its pooled tp, fp and fn are summed over its runs, while its
    new_gold_spans is the number of new gold spans, counted once (its runs must share a seen set).
    Returns (result, {system: counts}).
    """
    counts, points = {}, {}
    for s, p in systems.items():
        names = seen_of[s]
        g, q = restrict(gold, p, lambda cid, x: texts[cid][x[0]:x[1]] not in names)
        counts[s] = type_counts(g, q)
        m = strict(g, q)
        points[s] = tuple(sum(c[t][i] for c in counts[s].values() for t in (*ENTITY_TYPES, "_invalid")) for i in range(3))
        assert points[s] == (m["tp"], m["fp"], m["fn"])
    ci = bootstrap(counts, strata, b["n"], b["seed"], "within_strata", None, system_pairs)
    out = {"systems": {s: _new_name_point(*points[s], ci["systems"][s]) for s in systems},
           "paired_difference_recall": {k: v["micro_recall"] for k, v in ci["paired_difference"].items()},
           "models": None, "pooled_paired_difference_recall": None}
    if models:
        pooled = {m: pooled_counts(runs, counts) for m, runs in models.items()}
        mci = bootstrap(pooled, strata, b["n"], b["seed"], "within_strata", None, model_pairs)
        out["models"] = {}
        for m, runs in models.items():
            recall = [points[r][0] / (points[r][0] + points[r][2]) if points[r][0] + points[r][2] else None for r in runs]
            known = [x for x in recall if x is not None]
            total = tuple(sum(points[r][i] for r in runs) for i in range(3))
            point = _new_name_point(*total, mci["systems"][m])
            per_run_new = {points[r][0] + points[r][2] for r in runs}
            assert len(per_run_new) == 1, f"model {m!r}: its runs disagree on the new gold spans (different seen sets)"
            point["new_gold_spans"] = per_run_new.pop()  # the gold count, once; tp, fp and fn stay summed over the runs
            out["models"][m] = {"runs": runs, "recall_by_run": {r: (round(x, 4) if x is not None else None) for r, x in zip(runs, recall)},
                                "mean_recall": round(sum(known) / len(known), 4) if known else None,
                                "range_recall": [round(min(known), 4), round(max(known), 4)] if known else None,
                                "pooled": point}
        out["pooled_paired_difference_recall"] = {k: v["micro_recall"] for k, v in mci["paired_difference"].items()}
    return out, counts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def snapshot_lineage(training, scoring):
    """The facts that let systems from the training snapshot be scored on a later export of the book (the scoring
    snapshot): None when the two are one; otherwise every chapter of the training snapshot must be in the scoring
    snapshot with the same number and text (the run stops if not), and the record says what each snapshot was used for."""
    if training.id == scoring.id:
        return None
    shared = [c for c in training.chapters if c in scoring.chapters]
    changed = [c for c in shared if (training.chapters[c].number, training.chapters[c].text)
               != (scoring.chapters[c].number, scoring.chapters[c].text)]
    if changed or len(shared) != len(training.chapters):
        raise SystemExit(f"snapshot {scoring.id} is not a later export of {training.id}: {len(training.chapters) - len(shared)} "
                         f"chapters missing, {len(changed)} with another text or number")
    return {"training_snapshot": {"id": training.id, "source_commit": training.meta.get("source_commit")},
            "scoring_snapshot": {"id": scoring.id, "source_commit": scoring.meta.get("source_commit")},
            "shared_chapters": len(shared), "shared_text_identical": True,
            "chapters_only_in_the_scoring_snapshot": len(scoring.chapters) - len(shared),
            "used_for": {"scoring_snapshot": ("the test answers and the test chapters' text: GLiNER and the dictionary match "
                                              "read them there, the trained systems' predict runs were made on it (with a "
                                              "recorded lineage from the training snapshot) and the generative systems' "
                                              "answers are located in it"),
                         "training_snapshot": ("the trained systems' training runs and weak builds (whose training chapters "
                                               "are recomputed from its chapters), the dictionary system's name dictionary "
                                               "(dictionary.from_training_snapshot), the new-name rule's seen names "
                                               "(dictionary + mentions <= cutoff), and the dev answers when they name it")}}


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
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(cfg["snapshot"])
    training_id = cfg.get("training_snapshot") or cfg["snapshot"]
    training = snapshot if training_id == snapshot.id else load_snapshot(training_id)
    lineage = snapshot_lineage(training, snapshot)  # None when the two are one; stops unless snapshot is a later export of training
    d = cfg["dictionary"]
    if lineage and not d.get("from_training_snapshot"):
        raise SystemExit("training_snapshot differs from snapshot: set dictionary.from_training_snapshot to true (the scoring "
                         "snapshot's dictionary holds names from chapters after the training range)")
    dict_source = training if d.get("from_training_snapshot") else snapshot  # the same object without a lineage
    files = {k: REPO_ROOT / cfg[k] for k in ("gold_test", "gold_dev")}
    exports = {k: json.loads(files[k].read_text(encoding="utf-8")) for k in files}
    # the dev answers may be the training snapshot's (chapters <= 780, the same text in both); the test answers are this snapshot's
    dev_snapshot = training if exports["gold_dev"].get("snapshot_id") == training.id else snapshot
    text_of = {"test": snapshot, "dev": dev_snapshot}
    for name, export in exports.items():
        against = text_of[name.split("_")[1]]
        errors, _, _ = check_gold(export, against)  # offsets, texts and chapter versions against the snapshot the answers name
        if errors:
            raise SystemExit(f"{name} does not pass the gold check: {errors[:3]}")
    test_export, dev_export = exports["gold_test"], exports["gold_dev"]
    test_ids, dev_ids = done_chapters(test_export), done_chapters(dev_export)
    if len(test_ids) != len(test_export["chapters"]):
        raise SystemExit("the test answers have unfinished chapters")
    if set(test_ids) & set(dev_ids):
        raise SystemExit("a chapter is in both the development and the test answers")
    order = json.loads((REPO_ROOT / cfg["test_order"]).read_text(encoding="utf-8"))
    strata = {r["chapter_id"]: r["unit_id"] for r in order["order"] if r["chapter_id"] in set(test_ids)}
    if set(strata) != set(test_ids):
        raise SystemExit("some test chapters are not in the test order file")
    test_gold, dev_gold = gold_spans(test_export, test_ids), gold_spans(dev_export, dev_ids)
    name_counts_all = Counter(snapshot.chapters[cid].text[s:e] for cid, ref in test_gold.items() for s, e, _ in ref["spans"]
                              if not _touches((s, e), ref["unscored"]))
    top_name_all = name_counts_all.most_common(1)[0][0]

    # trained systems, models, pairs and the new-name rule are all checked before anything runs
    span_specs = cfg.get("span_systems") or []
    span_loaded, span_pool = {}, {}
    if span_specs:
        dev_set_ids = chapter_order(json.loads((REPO_ROOT / cfg.get("dev_set", DEV_SET)).read_text(encoding="utf-8")))
        order_ids = [r["chapter_id"] for r in order["order"]]
        for spec in span_specs:
            pred, rows, rec, pool = span_system(spec, snapshot, test_ids, dev_set_ids, order_ids, training)
            if spec["name"] in span_loaded:
                raise SystemExit(f"span system {spec['name']!r} is listed twice")
            span_loaded[spec["name"]] = (pred, rows, rec)
            span_pool[spec["name"]] = pool
    names = ["gliner_zero_shot", "dictionary", *(s["name"] for s in cfg.get("llm_systems", [])), *span_loaded]
    if len(set(names)) != len(names):
        raise SystemExit("two systems share a name")
    model_about = check_models(cfg.get("models"), set(names), span_pool)
    system_pairs, model_pairs = resolve_pairs(cfg.get("pairs"), names, cfg.get("models"))
    seen_rules = None
    if cfg.get("new_names") is None:
        unused = [s["name"] for s in span_specs if "seen_names" in s]
        if unused:
            raise SystemExit(f"span systems {unused}: seen_names only sets the new-name view's seen set, and this config "
                             "has no new_names")
    else:
        first_test = min(snapshot.chapters[c].number for c in test_ids)
        default = seen_names(training, cfg["new_names"], first_test, "new_names")  # the weak sources of the training snapshot
        seen_rules = {s: default for s in names}
        for spec in span_specs:
            if "seen_names" in spec:
                seen_rules[spec["name"]] = seen_names(training, spec["seen_names"], first_test, f"span system {spec['name']!r}")
        for model, runs in (cfg.get("models") or {}).items():
            if len({seen_rules[r][0] for r in runs}) != 1:
                raise SystemExit(f"model {model!r}: its runs count new names under different seen-set rules")

    from novel_ner.gliner_zh import BACKBONE_ID, BACKBONE_REVISION, MODEL_ID, MODEL_REVISION, ChineseGLiNER

    g = cfg["gliner"]
    if min(g["thresholds"]) < g["save_threshold"]:
        raise SystemExit("every threshold on the grid must be at or above save_threshold (lower outputs are never kept)")
    record = {
        "run_id": run_id, "config_file": Path(args.config).as_posix(), "config": cfg,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta["source_commit"]},
        **({"snapshots": lineage} if lineage else {}),
        "gold_test": {"file": cfg["gold_test"], "sha256": sha256(files["gold_test"]), "chapters": test_ids},
        "gold_dev": {"file": cfg["gold_dev"], "sha256": sha256(files["gold_dev"]), "chapters": dev_ids},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "backbone": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION},
        "code": git_state(), "environment": environment(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    # 1) GLiNER: choose the threshold on the development answers, then run the test chapters once
    model = ChineseGLiNER()
    raw, stats = {}, {}
    for split, ids in (("dev", dev_ids), ("test", test_ids)):
        raw[split], stats[split] = gliner_predictions(model, g["variant"], g["labels"], [text_of[split].chapters[c] for c in ids], g)
    dev_sweep = {t: strict(dev_gold, as_prediction(raw["dev"], dev_ids, t)) for t in g["thresholds"]}
    chosen = max(g["thresholds"], key=lambda t: (dev_sweep[t]["f1"], -t))  # ties: the lower threshold
    gliner = as_prediction(raw["test"], test_ids, chosen)

    # 2) dictionary matching
    dict_preds, conflicts = dictionary_predictions(dict_source.dictionary, [snapshot.chapters[c] for c in test_ids], d["min_len"])
    dictionary = as_prediction(dict_preds, test_ids)

    for split in raw:
        write_jsonl(out_dir / f"predictions_{split}_gliner_zero_shot.jsonl",
                    [{**p, "kept": p["score"] > chosen} for p in raw[split]])
    write_jsonl(out_dir / "predictions_test_dictionary.jsonl", dict_preds)

    systems = {"gliner_zero_shot": gliner, "dictionary": dictionary}
    llm_records = {}
    task_of = {}
    for task in test_export.get("llm_drafts", {}):
        for cid in json.loads((REPO_ROOT / "data" / "llm_tasks" / task / "task.json").read_text(encoding="utf-8"))["chapters"]:
            task_of[cid] = task
    for spec in cfg.get("llm_systems", []):
        preds, book = [], {"invalid_by_chapter": {}, "abstained_by_chapter": {}, "pending_places": [], "problems": Counter(),
                           "chapters_missing": []}
        for task in sorted({task_of[c] for c in test_ids}):
            task_dir = REPO_ROOT / "data" / "llm_tasks" / task
            ids = [c for c in json.loads((task_dir / "task.json").read_text(encoding="utf-8"))["chapters"] if c in set(test_ids)]
            answer_file = task_dir / f"pass_{spec['label']}.json"
            if not answer_file.exists():
                raise SystemExit(f"{spec['name']}: no answer for {task} ({answer_file.name})")
            got, b = llm_predictions(json.loads(answer_file.read_text(encoding="utf-8")), snapshot, ids)
            preds += got
            for k in ("invalid_by_chapter", "abstained_by_chapter"):
                book[k].update(b[k])
            book["pending_places"] += b["pending_places"]
            book["problems"].update(b["problems"])
            book["chapters_missing"] += b["chapters_missing"]
        pred = as_prediction(preds, test_ids)
        for cid in test_ids:
            pred[cid]["errors"] = book["invalid_by_chapter"].get(cid, 0)
            pred[cid]["pending"] = [(a, b) for c, a, b in book["pending_places"] if c == cid]
        systems[spec["name"]] = pred
        write_jsonl(out_dir / f"predictions_test_{spec['name']}.jsonl", preds)
        calls = json.loads((REPO_ROOT / spec["calls"]).read_text(encoding="utf-8"))
        answers_checked = {"checked": 0, "no_hash_logged": 0}
        for task in sorted({task_of[c] for c in test_ids}):
            answer_file = REPO_ROOT / "data" / "llm_tasks" / task / f"pass_{spec['label']}.json"
            logged = [c.get("answer_sha256") for c in calls["calls"] if c.get("task") == task and c.get("answer_sha256")
                      and c.get("label", spec["label"]) == spec["label"]]
            if not logged:
                answers_checked["no_hash_logged"] += 1  # runs logged before claude_run recorded hashes
            elif sha256(answer_file) != logged[-1]:
                raise SystemExit(f"{answer_file} is not the answer its call log recorded")
            else:
                answers_checked["checked"] += 1
        headers = calls.get("invocations") or [calls]  # older logs kept these fields at the top level
        if calls.get("invocations") and "effort" in calls:
            headers = [calls] + headers  # an old-format log later extended by a new invocation
        mine = [c for c in calls["calls"] if c.get("label", spec["label"]) == spec["label"]]
        settings = lambda key: sorted({str(h.get(key)) for h in headers if h.get(key) is not None})
        guide = (REPO_ROOT / "data" / "llm_tasks" / task_of[test_ids[0]] / "guide.md").read_text(encoding="utf-8")
        in_guide = {n for n in name_counts_all if n in guide}
        keys = [(p_["chapter_id"], p_["start"], p_["end"], p_["type"]) for p_ in preds]
        places = Counter(k[:3] for k in set(keys))
        llm_records[spec["name"]] = {
            "input_condition": ("(c) 5 test chapters per call + annotation guide v1.1 (with in-book example names) + drafting "
                                "instructions v1 with a pending (abstain) option: the exact prompt GPT got when drafting the "
                                "silver standard (api_prompt_v1); generative; no dictionary, no retrieval"),
            "positioning": ("each mention gives paragraph id, text and occurrence number and the program finds the position "
                            "(appendix 1's alternative to a locating snippet); fixed before the runs"),
            "retry_rule": "one call per package; an error reply or one without valid JSON is retried once; fixed before the runs",
            "run_of_model": spec.get("run_note", "single run"),
            "sampling": "Claude Code CLI defaults; temperature, top-p and seed cannot be set; not reproducible run for run",
            "gold_names_written_in_the_guide": {
                "rule": "a scored gold name that occurs anywhere in guide.md as a substring (an upper bound)",
                "distinct_names": len(in_guide), "scored_spans": sum(name_counts_all[n] for n in in_guide),
                "of_which_the_most_frequent_name": name_counts_all[top_name_all] if top_name_all in in_guide else 0,
                "of_which_one_character_names": sum(name_counts_all[n] for n in in_guide if len(n) == 1),
                "names_of_two_or_more_characters_other_than_the_most_frequent": {
                    "names": sum(1 for n in in_guide if len(n) > 1 and n != top_name_all),
                    "scored_spans": sum(name_counts_all[n] for n in in_guide if len(n) > 1 and n != top_name_all)}},
            "label": spec["label"], "calls_record": spec["calls"],
            "models_used": sorted({m for c in mine for m in c.get("models_used", [])}),
            "effort": settings("effort"), "system_prompt": settings("system_prompt"), "isolation": settings("isolation"),
            "claude_cli": settings("claude_cli"), "invocations": len(headers), "calls": len(mine),
            "calls_after_the_first_for_a_package": sum(n - 1 for n in Counter(c["task"] for c in mine).values()),
            "packages_called_in_more_than_one_invocation": sum(  # each invocation starts a package at attempt 1
                1 for n in Counter(c["task"] for c in mine if c.get("attempt") == 1).values() if n > 1),
            "answers_match_logged_sha256": answers_checked,
            "failed_calls": sum(1 for c in mine if c.get("failed") or c.get("is_error") or c.get("invalid_json")),
            "seconds_total": round(sum(c["seconds"] for c in mine), 1),
            "list_price_estimate_usd": round(sum(c.get("list_price_estimate_usd") or 0 for c in mine), 4),
            "outputs": {"located_predictions": len(preds), "exact_duplicates_removed": len(keys) - len(set(keys)),
                        "places_with_conflicting_types": sum(1 for n in places.values() if n > 1),
                        "invalid_predictions": sum(book["invalid_by_chapter"].values()),
                        "abstentions_pending": sum(book["abstained_by_chapter"].values()),
                        "problems": dict(book["problems"]), "chapters_missing": book["chapters_missing"]},
            "caveat": ("the silver standard was drafted by GPT from this same prompt, so systems given it share its "
                       "conventions (see by_gold_origin_post_hoc); GPT itself is not compared. Only these systems may abstain, "
                       "which raises their precision (abstentions_pending). The models may have seen the novel in pretraining; "
                       "the instructions forbid using it, which cannot be checked"),
        }
    span_records = {}
    for s, (pred, rows, rec) in span_loaded.items():  # trained systems, checked above, scored like every other system
        systems[s] = pred
        write_jsonl(out_dir / f"predictions_test_{s}.jsonl", rows)
        span_records[s] = rec
    counts = {s: type_counts(test_gold, p) for s, p in systems.items()}
    repeated = run_agreement(cfg.get("run_pairs", []), systems, test_gold, test_ids)
    supported = [t for t in ENTITY_TYPES if any(c[t][0] + c[t][2] for c in counts["dictionary"].values())]
    scores = {s: summarize(list(counts[s].values()), supported) for s in systems}
    models = model_summary(cfg.get("models", {}), counts, strata, cfg["bootstrap"], supported, model_pairs,
                           model_about) if cfg.get("models") else None
    for s, p in systems.items():  # the per-type pooling must agree with score.strict
        m = strict(test_gold, p)
        assert (m["tp"], m["fp"], m["fn"]) == tuple(sum(c[t][i] for c in counts[s].values() for t in (*ENTITY_TYPES, "_invalid"))
                                                    for i in range(3))
    b = cfg["bootstrap"]
    main_ci = bootstrap(counts, strata, b["n"], b["seed"], "within_strata", supported, system_pairs)
    unit_ci = bootstrap(counts, strata, b["n"], b["seed"], "units", supported, system_pairs)
    reported_pairs = system_pairs if system_pairs is not None else [(a, b_) for i, a in enumerate(systems) for b_ in list(systems)[i + 1:]]

    new_names = None
    if seen_rules is not None:  # the pre-registered extra analysis
        from novel_ner.novelty import novelty
        texts_all = {cid: snapshot.chapters[cid].text for cid in test_ids}
        view, _ = new_name_view(test_gold, systems, texts_all, {s: seen_rules[s][1] for s in systems}, strata, b,
                                cfg.get("models"), system_pairs, model_pairs)
        seen_sets = {}
        for s in systems:
            label, names_seen, info = seen_rules[s]
            if label not in seen_sets:
                seen_sets[label] = {**info, "gold": novelty(test_gold, texts_all, names_seen), "systems": []}
            seen_sets[label]["systems"].append(s)
            view["systems"][s] = {"seen_set": label, **view["systems"][s]}
            assert view["systems"][s]["new_gold_spans"] == seen_sets[label]["gold"]["new_spans"]
        new_names = {
            "rule": (f"pre-registered: a scored gold span is new when its exact text is not a name in the "
                     f"{'training ' if lineage else ''}snapshot's dictionary and not a mention name of chapters <= "
                     f"{cfg['new_names']['cutoff_chapter']} (any type; "
                     "a known character written a new way counts as new). Gold and predictions are restricted alike, so recall = "
                     "TP_new / (TP_new + FN_new); a hit has its gold span's text, so no hit is lost. Unscored ranges are left out"),
            "precision_label": ("precision_among_predictions_with_unseen_text = TP / predictions whose exact text is unseen; it is not "
                                "precision on new names (predictions of seen strings are left out, any unseen string that is not gold "
                                "counts against it). Invalid outputs (no position) are in neither number"),
            "seen_sets": seen_sets, **view,
            "intervals": ("paired chapter bootstrap within plot units, the same seed and rounds as the main intervals (the same "
                          "draws); a round with no such prediction counts precision 0; models pooled per chapter as in 'models'"),
            "models_pooled": ("a model's pooled tp, fp and fn are summed over its runs (each run counts once); its new_gold_spans "
                              "is the number of new gold spans, counted once, as for a system"),
            "pairs": "the configured pairs" if system_pairs is not None else "every pair",
        }

    # cross-chapter group: a span (gold or predicted) belongs to it if it overlaps a gold span flagged cross-chapter
    regions = {cid: [(a["start"], a["end"]) for a in test_export["chapters"][cid]["annotations"]
                     if a.get("cross_chapter") and not a.get("pending")] for cid in test_ids}
    in_group = lambda cid, s: _touches(s, regions[cid])
    groups = {}
    for s, p in systems.items():
        inside, rest = restrict(test_gold, p, in_group), restrict(test_gold, p, lambda c, x: not in_group(c, x))
        groups[s] = {"cross_chapter": brief(strict(*inside)), "rest": brief(strict(*rest))}
    gold_in_group = sum(1 for cid, ref in test_gold.items() for x in ref["spans"] if in_group(cid, x) and not _touches(x, ref["unscored"]))

    # post hoc (chosen after the v4 results): gold the user or the name sheet decided vs gold both GPT passes gave alike;
    # a gold or predicted span is in "user_or_sheet" if it overlaps a settled gold span the user or the sheet decided
    origin, _ = gold_origin(test_export, test_ids)
    # from answer v3 on, spans the reviewer decided on the user's notes count with the decided ones (not GPT-agreed)
    decided = {cid: [(a, b) for (a, b, _), o in origin.get(cid, {}).items() if o in ("user_or_sheet", "reviewer")]
               for cid in test_ids}
    in_decided = lambda cid, x: _touches(x, decided[cid])
    if set(origin) != set(test_ids):
        raise SystemExit("the gold-origin view needs GPT drafts for every test chapter")
    origin_views = {s: {"user_or_sheet": brief(strict(*restrict(test_gold, p, in_decided))),
                        "rest": brief(strict(*restrict(test_gold, p, lambda c, x: not in_decided(c, x))))}
                    for s, p in systems.items()}

    # auxiliary views (diagnostics, not the main metric); span-only scoring gives every span one label
    one_type = lambda p: {cid: {"spans": {(s, e, "person") for s, e, _ in v["spans"]},
                                **{k: v[k] for k in ("unscored", "errors") if k in v}} for cid, v in p.items()}
    texts = {cid: snapshot.chapters[cid].text for cid in test_ids}
    name_counts = Counter(texts[cid][s:e] for cid, ref in test_gold.items() for s, e, _ in ref["spans"] if not _touches((s, e), ref["unscored"]))
    top_name, top_count = name_counts.most_common(1)[0]
    without_top = lambda cid, x: texts[cid][x[0]:x[1]] != top_name
    scored_spans = [(c, sp) for c, ref in test_gold.items() for sp in ref["spans"] if not _touches(sp, ref["unscored"])]
    by_type_names = {t: Counter(texts[c][sp[0]:sp[1]] for c, sp in scored_spans if sp[2] == t) for t in ENTITY_TYPES}
    concentration = {t: {"spans": sum(n.values()), "distinct_names": len(n),
                         "top1_share": round(n.most_common(1)[0][1] / sum(n.values()), 4) if n else None,
                         "top3_share": round(sum(k for _, k in n.most_common(3)) / sum(n.values()), 4) if n else None,
                         "chapters": len({c for c, sp in scored_spans if sp[2] == t})}
                     for t, n in by_type_names.items()}
    per_unit = {}
    for u in sorted(set(strata.values())):
        ids = [c for c in test_ids if strata[c] == u]
        per_unit[u] = {"chapters": len(ids),
                       "scored_gold_spans": sum(1 for c, _ in scored_spans if c in ids),
                       **{s: _prf_defined(*(sum(counts[s][c][t][i] for c in ids for t in (*ENTITY_TYPES, "_invalid"))
                                            for i in range(3)))["f1"]
                          for s in systems}}

    gold_n = {t: sum(1 for x in test_gold.values() for s in x["spans"] if s[2] == t and not _touches(s, x["unscored"]))
              for t in ENTITY_TYPES}
    metrics = {
        "run_id": run_id,
        **({"snapshots": lineage} if lineage else {}),
        "test": {"chapters": len(test_ids), "plot_units": dict(Counter(strata.values())),
                 "scored_gold_spans": sum(name_counts.values()),
                 "gold_by_type": gold_n,
                 "names_by_type": concentration,
                 "unscored": unscored_report(test_export, test_gold, test_ids, snapshot, systems),
                 "gold": gold_record(test_export, test_ids, snapshot, cfg["calibration_metrics"])},
        "dev": {"chapters": len(dev_ids), "gold_spans": sum(len(x["spans"]) for x in dev_gold.values()),
                "recorded_unscored_ranges": sum(len(dev_export["chapters"][c].get("unscored", [])) for c in dev_ids),
                "pending_marks": sum(1 for c in dev_ids for a in dev_export["chapters"][c]["annotations"] if a.get("pending")),
                "gold_by_type": {t: sum(1 for x in dev_gold.values() for sp in x["spans"] if sp[2] == t and not _touches(sp, x["unscored"]))
                                 for t in ENTITY_TYPES},
                "provenance": dict(Counter(dev_export["chapters"][c].get("provenance") for c in dev_ids))},
        "systems": {
            "gliner_zero_shot": {
                "input_condition": "(a) chapter text only, six English label words; no rules, no dictionary, no training",
                "input": ("chapter text only; no training; no dictionary. Chunks of whole consecutive paragraphs up to chunk_max_chars "
                          "characters, no overlap (a single longer paragraph is a chunk of its own)"),
                "longest_chunk_characters": max(len(t) for c in test_ids for _, t in chapter_chunks(snapshot.chapters[c], g["chunk_max_chars"])),
                "model_limits": {"max_len_words": getattr(model.model.config, "max_len", None),
                                 "max_width_words": getattr(model.model.config, "max_width", None)},
                "variant_chosen_in": "runs/20260930-zs_gliner_trial (development chapters <= 780, compared with the Opus extraction, not gold)",
                "outputs": output_record(raw["test"], chosen),
                "threshold": chosen, "threshold_rule": "keep outputs scoring above the threshold (raw model score, as GLiNER decodes)",
                "threshold_chosen_on": "dev answers only: best strict micro F1 over the grid (ties: lower threshold)",
                "dev_sweep": {str(t): {k: m[k] for k in ("precision", "recall", "f1")} for t, m in dev_sweep.items()},
                "run_stats": stats,
                "kept_outputs_crossing_a_paragraph_break": sum(1 for p in raw["test"] if p["score"] > chosen and PARAGRAPH_SEP in p["text"]),
                "test_sweep_for_reference_only_not_used_to_choose": {
                    str(t): {k: v for k, v in strict(test_gold, as_prediction(raw["test"], test_ids, t)).items()
                             if k in ("precision", "recall", "f1")} for t in g["thresholds"]},
            },
            "dictionary": {
                "input_condition": "(b) chapter text + the novel project's name dictionary (names first seen by chapter 780)",
                "input": ("chapter text plus the novel project's name dictionary (a different input condition from GLiNER's); "
                          "longest match first, left to right, no overlaps"),
                "outputs": output_record(dict_preds),
                "min_len": d["min_len"], "names_skipped_for_conflicting_types": len(conflicts),
                "resource": dictionary_record(dict_source.dictionary, dict_source, d["min_len"]),
                **({"snapshot": dict_source.id, "from_training_snapshot": True} if lineage else {}),
            },
            **llm_records,
            **span_records,
        },
        "scores": {s: {**scores[s],
                       "strict_diagnostics": {k: v for k, v in strict(test_gold, p).items()
                                              if k in ("fp_same_span_wrong_type", "fp_overlapping_gold", "invalid_predictions",
                                                       "fn_where_draft_was_pending")},
                       "span_only_auxiliary": brief(strict(one_type(test_gold), one_type(p))),
                       "cross_chapter_group": groups[s]["cross_chapter"], "rest": groups[s]["rest"],
                       "without_the_most_frequent_name_post_hoc": brief(strict(*restrict(test_gold, p, without_top, carry_invalid=True))),
                       "by_gold_origin_post_hoc": origin_views[s]}
                   for s, p in systems.items()},
        "cross_chapter_group": {"rule": "a gold or predicted span is in the group if it overlaps a settled gold span flagged cross_chapter",
                                "gold_spans": gold_in_group, "share_of_scored_gold": round(gold_in_group / sum(name_counts.values()), 4),
                                "caveat": "the user flagged cross-chapter names only partly (2026-10-02), so the group is incomplete"},
        "most_frequent_name": {"share_of_scored_gold": round(top_count / sum(name_counts.values()), 4), "spans": top_count,
                               "note": "post-hoc sensitivity view chosen after seeing the per-name counts; the name is not recorded in this file"},
        "per_plot_unit_micro_f1_post_hoc": per_unit,
        "repeated_runs": repeated,
        "models": models,
        "pair_kinds": {f"{a} - {b}": ("run-noise diagnostic (two runs of one model), not a comparison"
                                      if any(a in runs and b in runs for runs in (cfg.get("models") or {}).values())
                                      else "comparison") for a, b in reported_pairs},
        "by_gold_origin_post_hoc_rule": ("post hoc, added after the v4 results: 'user_or_sheet' = any gold or predicted span "
                                         "overlapping a settled gold span that both GPT passes did not give identically (the user "
                                         "or the name sheet decided it) or that the reviewer decided on the user's notes (from "
                                         "answer v3 on, even where both passes gave it identically); 'rest' = everything else: "
                                         "settled spans both GPT passes "
                                         "gave identically, plus places where a GPT pass proposed a name the user did not keep. "
                                         "The groups come from how GPT behaved on the same prompt, so they are not neutral; "
                                         "invalid predictions have no place and are in neither"),
        "intervals_note": ("test-chapter sampling only; each generative run and each training seed is treated as fixed output, "
                           "so run-to-run and seed-to-seed variation is in no interval (see repeated_runs and models). Macro F1 "
                           "hangs on tiny types: one span of a type with n scored spans can move that type's F1 by about 1/n, so "
                           "macro F1 by about 1/(6n): " + ", ".join(f"{t} n={n} (~{1 / (6 * n):.3f})" for t, n in sorted(gold_n.items())
                                                                     if 0 < n < 50)),
        "intervals": main_ci,
        "intervals_by_whole_plot_unit_sensitivity": {**unit_ci, "caveat": "only a few plot units of very unequal size: unstable"},
        "per_chapter_counts": {s: {c: [sum(counts[s][c][t][i] for t in (*ENTITY_TYPES, "_invalid")) for i in range(3)]
                                   for c in test_ids} for s in systems},
        "per_chapter_type_counts": counts,
        **({"pairs": {"systems": [f"{a} - {b}" for a, b in system_pairs], "models": [f"{a} - {b}" for a, b in model_pairs],
                      "rule": ("only these paired differences are computed (config 'pairs', fixed before scoring); every system "
                               "(each run and seed) still gets its own interval; a pooled model gets no interval of its own in "
                               "the main scores (mean, range and pooled counts in 'models'), only in new_name_recall when "
                               "that view is configured")}} if system_pairs is not None else {}),
        **({"new_name_recall": new_names} if new_names is not None else {}),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in systems:
        m, ci = scores[s], main_ci["systems"][s]
        print(f"{s:18s} micro P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f} F1={m['micro']['f1']:.3f} {ci['micro_f1']}"
              f" | non-person F1={m['non_person_micro']['f1']:.3f} {ci['non_person_f1']} | macro F1={m['macro_f1']:.3f} {ci['macro_f1']}")
    print(f"GLiNER threshold {chosen} (chosen on {len(dev_ids)} dev chapters)")
    for k, v in main_ci["paired_difference"].items():
        print(f"{k} (within units): {v}")
    for k, v in unit_ci["paired_difference"].items():
        print(f"{k} (whole units, sensitivity): {v}")
    if new_names is not None:
        for s, v in new_names["systems"].items():
            print(f"{s:18s} new-name recall {v['recall']} {v['recall_ci95']} ({v['new_gold_spans']} new gold spans, {v['seen_set']})")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
