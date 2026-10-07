"""Post-hoc analyses of a scoring run (v10): seed-aware intervals, new-name breakdowns, names never in the training
text, title-suffix answers, learning-curve steps.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.posthoc configs/posthoc_v10.json [--run-id ID]

POST HOC. Every analysis in this module was chosen after the v9 and v10 test scores (and those of the learning-curve
tiers) were seen; each was first computed in scratch while those runs were being verified. This program makes them a
recorded, tested run so that a write-up cites only stored numbers. None is a pre-registered test and nothing about any
system is chosen with them. The one pre-registered quantity they build on is the per-occurrence new-name recall of
the study plan by the fixed rule (a scored gold span is new when its exact text is neither a dictionary name nor a mention
name of chapters <= 780: novelty.weak_names, any type), which is re-derived here and checked equal to the scoring run's.

Nobody judges whether an answer is right. Claude wrote this program and must not (Opus is a compared model): every count
comes from the fixed mechanical rules below, applied by the program to the answer file and the systems' outputs. A
disagreement between a system and the answer file is never called an error of the answer.

Inputs, all checked (the run stops on any mismatch):
- the scoring run "eval_run": its config.json and metrics.json, both naming its folder; the answer file it hashed, read
  with score.gold_spans exactly as evaluate (gold pending marks and the recorded unscored ranges are unscored: a gold or
  predicted span touching one counts for nothing); its test chapters; the plot units of its test order; its bootstrap
  rounds and seed (the config must name the same);
- every system's test predictions (predictions_test_<system>.jsonl of the scoring run), read and checked as
  errors.load_systems does: trained systems by the sha256 the scoring run recorded; GLiNER by its kept rows against the
  recorded threshold; the dictionary recomputed from the snapshot; the generative systems' answers read again with
  evaluate.llm_predictions (checked against the call log's sha256), which also gives their pending places. Each system's
  per-chapter, per-type TP/FP/FN must equal the scoring run's per_chapter_type_counts, its micro scores its scores, and
  its per-occurrence new-name TP/FP/FN its new_name_recall;
- for trained systems, the weak build each was trained on (training chapters recomputed with weak.split_ids from the
  build's record and checked against the sha256 that the build and the scoring run recorded) and the training run (its
  config.json the file the scoring run hashed, its setting and seed the scoring run's); for C also every build of the
  run's earlier stages: the replay build of a replay fine-tuning run and, through init_from, the builds of the run whose
  checkpoint it started from, recursively (each config.json the file its child recorded);
- when the scoring run scored a later export (its config's training_snapshot differs from its snapshot; stage 3's test2):
  the training snapshot is loaded too, the scoring run must record the lineage ("snapshots"), every training chapter
  must be in the scoring snapshot with the same text (evaluate.snapshot_lineage), and the training runs, the weak builds,
  the training text of C, the dictionary system's name dictionary (dictionary.from_training_snapshot) and the seen names
  of the fixed rule are taken from the training snapshot, as the scoring run took them.

The analyses (rules in words in RULES, written to config.json):
A. Seed-aware paired intervals (two-level bootstrap) for configured pairs of pooled models, later minus earlier: each
   round draws chapters within plot units exactly as evaluate.bootstrap does (same strata, seed and rounds) and, from a
   separate generator, each multi-run model's runs with replacement; the same machinery with the runs fixed gives the
   scoring run's stored pooled intervals in the direction each was stored, which is asserted (for a pair stored the
   other way round, the interval written here for later minus earlier can differ from the stored one negated at an
   end, by the gap between two neighbouring sorted round values, and in the share above 0: see RULES).
B. New-name recall (fixed rule) by type, by frequency band of the text, for single-character spans, per occurrence and per
   distinct (text, type) name, per run and per model (mean and range over runs), with the concentration facts.
C. For trained systems: recall on the new names whose text never occurs in any of the system's training chapters, of
   every stage (a fine-tuned run's init_from chain and replay build included).
D. Title-suffix whole-string answers: how each run handles gold person spans whose text is X + a title suffix, and who
   decided those answers (evaluate.gold_origin). Mechanical; it does not judge who is right.
E. Learning-curve steps from each tier run's training record.

Outputs: runs/<run_id>/config.json (inputs with sha256, the rules in words, code state) and metrics.json (numbers only,
no string from the book; checked) are committed; data/analysis/<run_id>/keys.json (not committed) maps the numbered
strings of D to their text. Nothing is overwritten, and a real run needs committed code (src/ and configs/ clean).
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from novel_ner.compare import ENTITY_TYPES
from novel_ner.errors import check_counts_only, load_systems, per_name_recall
from novel_ner.evaluate import (NON_PERSON, _prf_defined, _quantile_interval, _raw_f1, done_chapters, gold_origin, sha256,
                                snapshot_lineage, type_counts)
from novel_ner.gold import check as check_gold
from novel_ner.novelty import weak_names
from novel_ner.score import _touches, gold_spans
from novel_ner.snapshot import REPO_ROOT, load_snapshot
from novel_ner.teacher import TEACHER_BUILD_KIND, build_train_ids
from novel_ner.weak import ids_sha256, split_ids
from novel_ner.zeroshot import git_state

STATS = ("micro_f1", "micro_precision", "micro_recall", "non_person_f1", "new_name_recall")
STORED_MAIN = ("micro_f1", "micro_precision", "micro_recall", "non_person_f1")  # evaluate's model_summary pooled_paired
LEVELS = ("chapter_only", "chapters_and_runs")
HANDLING = ("exact_hit", "prediction_equals_name_without_suffix", "pending", "other_overlap", "nothing")
ORIGINS = ("gpt_agreed", "user_or_sheet", "reviewer")

POST_HOC = ("POST HOC: every analysis in this run was chosen after the v9 and v10 test scores (including the learning-curve "
            "tiers') were seen, and first computed in scratch while verifying those runs; descriptive, not pre-registered "
            "tests; nothing about any system is chosen with them. Only the per-occurrence new-name recall by the fixed rule "
            " was set in advance; it is re-derived here and checked equal to the scoring run's")

RULES = {
    "timing": POST_HOC,
    "authority": ("every count comes from fixed mechanical rules applied by the program; nobody judged whether an answer is "
                  "right (Claude wrote the program and must not: Opus is a compared model). A disagreement between a system "
                  "and the answer file is never called an error of the answer"),
    "regions": ("evaluate's: the answer file read with score.gold_spans; a gold or predicted span touching a gold pending "
                "mark or a recorded unscored range counts for nothing; exact duplicate predictions count once"),
    "A_seed_aware_intervals": (
        "each of the rounds draws test chapters within plot units exactly as evaluate.bootstrap (mode within_strata) does, "
        "with the scoring run's seed, so every round draws the same chapters as the scoring run's intervals. The run draws "
        "use their own generator, random.Random(run_seed) (run_seed 20261003 in configs/posthoc_v10.json), created once and "
        "consumed round by round and, within a round, model by model in the order of the scoring run's models (only models "
        "that are in some pair): a model with k >= 2 runs makes k draws, each one call of choice on its list of runs (so with "
        "replacement), and its counts in the round are the sum over all k drawn runs (a run drawn twice counts twice) of "
        "their TP/FP/FN on the drawn chapters; a single-run model (GLiNER, the dictionary) keeps its one run and does not "
        "touch the generator. The statistics are recomputed from the round's pooled counts as evaluate.round_stats does: "
        "micro F1, precision and recall (invalid outputs count as false positives), non-person micro F1, and new-name recall "
        "by the fixed rule (gold and predictions restricted alike to spans whose text is unseen by chapter 780). Pairs are "
        "later minus earlier (the config lists [later, earlier]). point = the statistic of each model's counts pooled over "
        "all its runs and all test chapters (every run's TP/FP/FN added up, then one ratio; as the scoring run's "
        "pooled_counts_micro_f1), later minus earlier; for micro F1 and the other ratios this is not the difference of the "
        "run-mean scores. The interval is evaluate's percentile (sorted round values at int(0.025 (n-1)) and "
        "int(0.975 (n-1)), rounded to 4 places); share_of_rounds_above_0 counts the rounds with a difference strictly above "
        "0. Level chapter_only: the same machinery with every run kept (no run draws). Its rounds must give the scoring run's "
        "stored pooled interval of the pair (models.pooled_paired; new_name_recall.pooled_paired_difference_recall) in the "
        "direction it was stored, or the run stops: a pair stored the same way round is compared directly (and then equals "
        "it exactly); a pair stored the other way round (earlier minus later) is compared with the negated round values "
        "(exact in floating point). For such a reversed pair the chapter_only interval written here (later minus earlier) "
        "is NOT the stored interval negated: negating the stored [lo, hi] gives this pair's sorted round values at positions "
        "n-1-int(0.975 (n-1)) and n-1-int(0.025 (n-1)) (250 and 9750 for 10000 rounds) instead of int(0.025 (n-1)) and "
        "int(0.975 (n-1)) (249 and 9749), so an end can differ by the gap between two neighbouring sorted round values "
        "(0.0001 or 0.0002 in posthoc_v10; largest_ci95_end_difference records it); and the share above 0 written here equals 1 "
        "minus the stored share only when no round is exactly 0 (a round tied at 0 is above 0 in neither direction). For "
        "each reversed pair, chapter_only_against_the_stored_interval_negated counts these differences; a write-up that "
        "cites both runs should, for a given pair, quote only one of the two. Level chapters_and_runs adds the run draws; "
        "width_minus_chapter_only_width = its interval width minus chapter_only's. interval_side_changes_with_run_draws "
        "lists every pair and statistic whose interval lies on a different side of 0 at the two levels (above_0: stored "
        "lower end > 0; below_0: stored upper end < 0; otherwise includes_0), with the sign of the point. With 2-3 runs per "
        "model the run draw is a tiny sample of run-to-run variation: a rough widening, not an estimate of seed variance"),
    "B_new_name_breakdowns": (
        "new gold spans by the fixed rule (scored, text unseen by chapter 780: novelty.weak_names, any type). A run finds a "
        "new gold span when it predicts exactly its chapter, start, end and type. Per occurrence: found spans / new spans in "
        "the group. Per name: a name is a distinct (text, type) among the group's new spans, found when at least one of its "
        "spans is found in that run. Groups: all; person; non_person (the five other types); each type; frequency band of "
        "the span's text = the number of new gold spans with the same text, any type (bands from the config); single "
        "character spans. A model: the mean over its runs of each run's share and the range (min, max), never 'found by "
        "any run'. Single-character facts: followed_by_a_title_suffix = the chapter text right after the span starts with "
        "one of the title suffixes (configs/weak_v2.json); first_character_starts_a_seen_name = some seen name (fixed rule) "
        "begins with the span's character. Concentration: spans of the top_n names by span count (ties do not change the "
        "sum), names with at least frequent_min_spans spans and their spans, the person share of new spans and of new names"),
    "C_never_in_training_text": (
        "trained systems only. Their training chapters are the union over every stage of their weights: the weak build "
        "the scoring run recorded (the last stage's labels), the replay build of a replay fine-tuning run (the training "
        "record's replay block, novel_ner.finetune), and, when the training run started from another run's checkpoint "
        "(its record's init_from), that run's builds, recursively. Each training run's config.json must be the file the "
        "scoring run hashed (the scored run) or the one its child recorded in init_from (config_sha256), name its own "
        "folder and the training snapshot (the scoring run's snapshot, or its training_snapshot when it scored a later "
        "export); each build's training chapters are recomputed from its record "
        "(weak.split_ids on the record's development chapters, test order and max_train_chapter, or a teacher build's "
        "packs) and must give the sha256 the build and the training run that used it recorded. The key of a training "
        "text in builds is the build itself for a one-stage run and the builds joined by ' + ' (last stage first) "
        "otherwise; from_builds gives each build's chapter count, training_runs each system's chain of runs. (Before "
        "2026-10-04 evening only the last stage's build was used, so a two-stage run, e.g. A3 or R2 from the 749-chapter "
        "S3 checkpoint, counted the new names of the S3 chapters outside its teacher chapters as never in its training "
        "text: posthoc-dev2-v1 to v3 hold that C for A3, R2, A3c and R2c.) A new gold name (text, type; fixed rule) is "
        "never in the training text when its text occurs nowhere, as a "
        "substring, in the text of any training chapter. Recall per occurrence and per name (as in B) on those names and, "
        "for contrast, on the other new names (their text occurs in some training chapter), each split person / non_person. "
        "It only rules out string lookup: a name never in the training text cannot have been labelled there; a name that "
        "is in it was not necessarily labelled"),
    "D_title_suffix_answers": (
        "mechanical; it does not judge who is right, the answer file or a system. A title-suffix whole-string answer is a "
        "scored gold person span whose text is X + a title suffix (configs/weak_v2.json title_suffixes, tried longest first; "
        "X not empty). Each run's handling of each such span, the first that applies, in this order: exact_hit = the run "
        "predicts exactly the span with type person; prediction_equals_name_without_suffix = exactly one scored prediction "
        "of the run overlaps the span and it is X's place (the span's start to start + len(X)), any type (X is only the span "
        "minus its title suffix, 1 or more characters; the rule does not say what X is, e.g. whether it is "
        "a surname); pending = a located pending mark of the run overlaps the span (generative systems only; places "
        "recomputed from their answers with evaluate.llm_predictions, as errors.py does); other_overlap = some other "
        "scored prediction overlaps it; nothing. A prediction removed by an unscored range counts as nothing. Who decided each answer: evaluate.gold_origin (gpt_agreed = both GPT passes gave "
        "it identically; user_or_sheet; reviewer). Strings are numbered by span count, ties by first place in the test "
        "order; their text is not in metrics.json"),
    "E_learning_curve_steps": (
        "from each tier run's training record (runs/<training run>/metrics.json, and config.json, the file the scoring run "
        "hashed): steps planned and run, steps per epoch, stopped early, dev evaluations. stopped_early is the record's flag "
        "as written: tagger.fit sets it when patience runs out, even at the last planned evaluation, so it can be true for a "
        "run that ran every planned step; ran_all_planned_steps = steps run equal steps planned, and only a run with "
        "ran_all_planned_steps false stopped before its schedule ended. Each tier's label must equal the max_train_chapter "
        "of the weak build of every run of its model (read from the build's config.json), or the run stops. Best step = the "
        "record's (checked: "
        "the first evaluation with the highest dev micro F1, the record's checkpoint rule); dev micro and non-person F1 at "
        "the best step and at the last evaluation, and the highest dev non-person F1 on the curve with its first step; "
        "windows seen at the best step = best step x batch size, as asked (an upper bound: the last batch of an epoch can be "
        "smaller), and exactly = full epochs x windows per epoch + remaining steps x batch size. Dev numbers are scores on "
        "the 31 development chapters (up to chapter 777), not test scores"),
}


# ---------------------------------------------------------------------------------------------- shared pieces

def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _share(a, b):
    return round(a / b, 4) if b else None


def _mean_range(values):
    known = [v for v in values if v is not None]
    if not known:
        return None, None
    return round(sum(known) / len(known), 4), [round(min(known), 4), round(max(known), 4)]


def new_name_chapter_counts(gold, prediction, texts, seen):
    """{chapter: [tp, fp, fn]} of evaluate's new-name view: gold and predictions restricted alike to scored spans whose
    text is unseen (invalid outputs have no place and are left out)."""
    out = {}
    for cid, ref in gold.items():
        keep = lambda x: not _touches(x, ref["unscored"]) and texts[cid][x[0]:x[1]] not in seen
        g = {x for x in ref["spans"] if keep(x)}
        p = {x for x in prediction[cid]["spans"] if keep(x)}
        out[cid] = [len(g & p), len(p - g), len(g - p)]
    return out


def new_gold(gold, texts, seen):
    """The scored new gold spans, chapter by chapter in gold order then by place: [(cid, start, end, type, text)]."""
    out = []
    for cid, ref in gold.items():
        for s, e, t in sorted(ref["spans"]):
            x = texts[cid][s:e]
            if not _touches((s, e), ref["unscored"]) and x not in seen:
                out.append((cid, s, e, t, x))
    return out


def hits(spans, prediction):
    """Indices of the spans [(cid, start, end, type, ...)] that the prediction has exactly."""
    return {i for i, (cid, s, e, t, *_) in enumerate(spans) if (s, e, t) in prediction[cid]["spans"]}


# ---------------------------------------------------------------------------------------------- A: two-level bootstrap

def chapter_vectors(tc, new_counts):
    """{chapter: (tp, fp, fn, np_tp, np_fp, np_fn, new_tp, new_fn)} from evaluate.type_counts rows (fp includes the
    invalid outputs, as round_stats counts them) and the new-name counts."""
    out = {}
    for cid, c in tc.items():
        tp, fp, fn = (sum(c[t][i] for t in ENTITY_TYPES) for i in range(3))
        fp += c.get("_invalid", [0, 0, 0])[1]
        np_ = tuple(sum(c[t][i] for t in NON_PERSON) for i in range(3))
        out[cid] = (tp, fp, fn, *np_, new_counts[cid][0], new_counts[cid][2])
    return out


def stats_of(v):
    """evaluate.round_stats' statistics (unrounded, same formulas) from one pooled vector, plus new-name recall."""
    tp, fp, fn, a, b, c, ntp, nfn = (int(x) for x in v)
    return {"micro_f1": _raw_f1(tp, fp, fn), "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0, "non_person_f1": _raw_f1(a, b, c),
            "new_name_recall": ntp / (ntp + nfn) if ntp + nfn else 0.0}


def chapter_draws(strata, n, seed):
    """The chapter samples of evaluate.bootstrap(mode="within_strata"), round by round: same generator, same order."""
    units = defaultdict(list)
    for cid in sorted(strata):
        units[strata[cid]].append(cid)
    unit_ids = sorted(units)
    rng = random.Random(seed)
    for _ in range(n):
        yield [rng.choice(units[u]) for u in unit_ids for _ in units[u]]


def interval(values):
    return {"ci95": _quantile_interval(values), "share_of_rounds_above_0": round(sum(x > 0 for x in values) / len(values), 4)}


def two_level_bootstrap(vectors, models, pairs, strata, n, chapter_seed, run_seed):
    """vectors = {run: {chapter: 8-tuple}}; models = {model: [runs]}; pairs = [(later, earlier)] of model names.

    Returns {"point": {pair: {stat: diff}}, "models": {model: pooled stats}, "diffs": {level: {pair: {stat: [diff per
    round]}}}}. point is the difference of the statistics of counts pooled over all runs and chapters (not of run-mean
    scores). Level chapter_only keeps every run of a model; chapters_and_runs also draws each multi-run model's runs with
    replacement from one random.Random(run_seed): round by round, models in the order of `models`, k = number of runs
    draws per model, one rng.choice per draw, the model's counts the sum over all k drawn runs. The chapter draws come
    from their own generator (chapter_draws), so both levels see the same chapters in every round."""
    chapters = sorted(strata)
    at = {c: i for i, c in enumerate(chapters)}
    used = [m for m in models if any(m in p for p in pairs)]
    run_list = list(dict.fromkeys(r for m in used for r in models[m]))
    pos = {r: i for i, r in enumerate(run_list)}
    table = np.array([[vectors[r][c] for c in chapters] for r in run_list], dtype=np.int64)  # (runs, chapters, 8)
    rows = {m: [pos[r] for r in models[m]] for m in used}
    whole = table.sum(axis=1)
    pooled = {m: stats_of(whole[rows[m]].sum(axis=0)) for m in used}
    rng = random.Random(run_seed)
    diffs = {level: {p: {k: [] for k in STATS} for p in pairs} for level in LEVELS}
    for sample in chapter_draws(strata, n, chapter_seed):
        weight = np.bincount([at[c] for c in sample], minlength=len(chapters)).astype(np.int64)
        totals = np.einsum("c,rck->rk", weight, table)  # every run's counts on this round's chapters
        value = {level: {} for level in LEVELS}
        for m in used:
            idx = rows[m]
            drawn = idx if len(idx) == 1 else [rng.choice(idx) for _ in idx]
            value["chapter_only"][m] = stats_of(totals[idx].sum(axis=0))
            value["chapters_and_runs"][m] = stats_of(totals[drawn].sum(axis=0))
        for level in LEVELS:
            for a, b in pairs:
                for k in STATS:
                    diffs[level][(a, b)][k].append(value[level][a][k] - value[level][b][k])
    point = {(a, b): {k: pooled[a][k] - pooled[b][k] for k in STATS} for a, b in pairs}
    return {"point": point, "models": pooled, "diffs": diffs, "run_draws": {m: len(models[m]) > 1 for m in used}}


def _plus0(x):
    return round(x, 4) + 0.0  # no -0.0


def check_replication(chapter_only, pairs, stored_main, stored_new):
    """Stop unless every pair's chapter-only rounds give the scoring run's stored pooled interval (and share of rounds
    above 0) in the direction it was stored: a reversed one is compared with the negated rounds. Returns {pair:
    {"stored_key", "stored_direction"}}, plus for a reversed pair "chapter_only_against_the_stored_interval_negated": how
    far the interval written for later minus earlier is from the stored one negated (the percentile positions are not
    symmetric and rounds tied at 0 are above 0 in neither direction; see RULES)."""
    out = {}
    for a, b in pairs:
        same, other = f"{a} - {b}", f"{b} - {a}"
        if same in stored_main:
            key, sign = same, 1
        elif other in stored_main:
            key, sign = other, -1
        else:
            raise SystemExit(f"the scoring run stored no pooled interval for {a} and {b}; the replication cannot be checked")
        if key not in stored_new:
            raise SystemExit(f"the scoring run stored no pooled new-name interval for {key}")
        ends, largest, shares = 0, 0.0, 0
        for k in (*STORED_MAIN, "new_name_recall"):
            values = chapter_only[(a, b)][k] if sign == 1 else [-x for x in chapter_only[(a, b)][k]]
            want = stored_new[key] if k == "new_name_recall" else stored_main[key][k]
            if interval(values) != {"ci95": list(want["ci95"]), "share_of_rounds_above_0": want["share_of_rounds_above_0"]}:
                raise SystemExit(f"{a} - {b}: the chapter-only {k} interval does not reproduce the scoring run's stored "
                                 f"pooled interval ({key})")
            if sign == -1:
                written = interval(chapter_only[(a, b)][k])
                negated = [_plus0(-want["ci95"][1]), _plus0(-want["ci95"][0])]
                gaps = [_plus0(abs(x - y)) for x, y in zip(written["ci95"], negated)]
                ends += sum(1 for g in gaps if g)
                largest = max(largest, *gaps)
                shares += written["share_of_rounds_above_0"] != _plus0(1 - want["share_of_rounds_above_0"])
        out[(a, b)] = {"stored_key": key, "stored_direction": "same" if sign == 1 else "reversed"}
        if sign == -1:
            out[(a, b)]["chapter_only_against_the_stored_interval_negated"] = {
                "statistics": len(STORED_MAIN) + 1, "ci95_ends_differing": ends, "largest_ci95_end_difference": largest,
                "shares_above_0_differing_from_1_minus_the_stored": shares}
    return out


def side_of_0(ci95):
    """Where a stored (rounded) interval lies: above_0 (lower end > 0), below_0 (upper end < 0) or includes_0."""
    return "above_0" if ci95[0] > 0 else "below_0" if ci95[1] < 0 else "includes_0"


def side_changes(pairs_out):
    """Every pair and statistic whose interval lies on a different side of 0 at the two levels, in pair then STATS order."""
    out = []
    for pair, v in pairs_out.items():
        for k in STATS:
            c, r = v["chapter_only"][k]["ci95"], v["chapters_and_runs"][k]["ci95"]
            if side_of_0(c) != side_of_0(r):
                p = v["point"][k]
                out.append({"pair": pair, "statistic": k, "point": p,
                            "point_sign": "positive" if p > 0 else "negative" if p < 0 else "zero",
                            "chapter_only": side_of_0(c), "chapters_and_runs": side_of_0(r),
                            "chapter_only_ci95": c, "chapters_and_runs_ci95": r})
    return out


def analysis_a(result, pairs, replication):
    out = {}
    for a, b in pairs:
        out[f"{a} - {b}"] = {
            "point": {k: round(v, 4) for k, v in result["point"][(a, b)].items()},
            **{level: {k: interval(result["diffs"][level][(a, b)][k]) for k in STATS} for level in LEVELS},
            "chapter_only_rounds_give_the_scoring_runs_stored_pooled_interval_in_its_direction": True,
            **replication[(a, b)],
        }
        for k in STATS:
            lo, hi = out[f"{a} - {b}"]["chapters_and_runs"][k]["ci95"]
            clo, chi = out[f"{a} - {b}"]["chapter_only"][k]["ci95"]
            out[f"{a} - {b}"]["chapters_and_runs"][k]["width_minus_chapter_only_width"] = round((hi - lo) - (chi - clo), 4)
    return out


# ---------------------------------------------------------------------------------------------- B: new-name breakdowns

def band_of(count, bands):
    found = [b["label"] for b in bands if count >= b["min"] and (b.get("max") is None or count <= b["max"])]
    if len(found) != 1:
        raise SystemExit(f"a text with {count} new spans falls in {len(found)} frequency bands; the bands must not overlap or leave gaps")
    return found[0]


def groups(new, bands):
    """{group: [indices into new]} in a fixed order; the band of a span is that of its text's count among new spans."""
    text_count = Counter(x for *_, x in new)
    out = {"all": list(range(len(new))),
           "person": [i for i, g in enumerate(new) if g[3] == "person"],
           "non_person": [i for i, g in enumerate(new) if g[3] != "person"]}
    for t in ENTITY_TYPES:
        out[f"type_{t}"] = [i for i, g in enumerate(new) if g[3] == t]
    for b in bands:
        out[f"band_{b['label']}"] = [i for i, g in enumerate(new) if band_of(text_count[g[4]], bands) == b["label"]]
    out["single_character"] = [i for i, g in enumerate(new) if len(g[4]) == 1]
    return out


def group_sizes(new, members):
    return {"spans": len(members), "names": len({(new[i][4], new[i][3]) for i in members}),
            "texts": len({new[i][4] for i in members})}


def group_result(new, members, found):
    """One run on one group: per occurrence and per distinct (text, type) name."""
    names = {(new[i][4], new[i][3]) for i in members}
    got = {(new[i][4], new[i][3]) for i in members if i in found}
    n_found = sum(1 for i in members if i in found)
    return {"spans_found": n_found, "recall": _share(n_found, len(members)),
            "names_found": len(got), "name_share": _share(len(got), len(names))}


def model_group(runs, results, sizes):
    """A model on one group: the mean over its runs of each run's share (unrounded), the range, and the counts by run."""
    rec = [results[r]["spans_found"] / sizes["spans"] if sizes["spans"] else None for r in runs]
    nam = [results[r]["names_found"] / sizes["names"] if sizes["names"] else None for r in runs]
    (rm, rr), (nm, nr) = _mean_range(rec), _mean_range(nam)
    return {"recall_mean": rm, "recall_range": rr, "name_share_mean": nm, "name_share_range": nr,
            "spans_found_by_run": [results[r]["spans_found"] for r in runs],
            "names_found_by_run": [results[r]["names_found"] for r in runs]}


def concentration(new, top_n, frequent_min):
    names = Counter((x, t) for *_, t, x in new)
    counts = sorted(names.values(), reverse=True)
    frequent = [n for n in counts if n >= frequent_min]
    person_spans = sum(1 for g in new if g[3] == "person")
    person_names = sum(1 for (_, t) in names if t == "person")
    return {"new_spans": len(new), "names": len(names), "texts": len({g[4] for g in new}),
            "top1_spans": counts[0] if counts else 0, "top1_share": _share(counts[0], len(new)) if counts else None,
            "top_n": top_n, "top_n_spans": sum(counts[:top_n]), "top_n_share": _share(sum(counts[:top_n]), len(new)),
            "frequent_min_spans": frequent_min, "frequent_names": len(frequent), "frequent_names_spans": sum(frequent),
            "frequent_names_share": _share(sum(frequent), len(new)),
            "names_with_one_span": sum(1 for n in counts if n == 1),
            "person": {"spans": person_spans, "share_of_spans": _share(person_spans, len(new)),
                       "names": person_names, "share_of_names": _share(person_names, len(names))}}


def single_character_facts(new, texts, suffixes, seen):
    one = [g for g in new if len(g[4]) == 1]
    starts = {n[0] for n in seen if n}
    followed = [g for g in one if any(texts[g[0]][g[2]:g[2] + len(x)] == x for x in suffixes)]
    shares = [g for g in one if g[4] in starts]
    part = lambda gs: {"spans": len(gs), "names": len({(g[4], g[3]) for g in gs}),
                       "by_type": {t: sum(1 for g in gs if g[3] == t) for t in ENTITY_TYPES}}
    return {"single_character": part(one), "followed_by_a_title_suffix": part(followed),
            "first_character_starts_a_seen_name": part(shares),
            "both": part([g for g in followed if g[4] in starts])}


def analysis_b(new, texts, predictions, models, bands, suffixes, seen, conc):
    members = groups(new, bands)
    sizes = {g: group_sizes(new, m) for g, m in members.items()}
    found = {s: hits(new, p) for s, p in predictions.items()}
    systems = {s: {g: group_result(new, m, found[s]) for g, m in members.items()} for s in predictions}
    model_out = {m: {"runs": runs, **{g: model_group(runs, {r: systems[r][g] for r in runs}, sizes[g]) for g in members}}
                 for m, runs in models.items()}
    text_count, name_count = Counter(g[4] for g in new), Counter((g[4], g[3]) for g in new)
    differ = sum(1 for g in new if band_of(text_count[g[4]], bands) != band_of(name_count[(g[4], g[3])], bands))
    bands_out = {b["label"]: {"min_spans": b["min"], "max_spans": b.get("max")} for b in bands}
    return {"groups": sizes, "bands": bands_out, "spans_whose_band_by_text_type_would_differ": differ,
            "concentration": concentration(new, conc["top_n"], conc["frequent_min_spans"]),
            "single_character_facts": single_character_facts(new, texts, suffixes, seen),
            "systems": systems, "models": model_out}, found


# ---------------------------------------------------------------------------------------------- C: never in training text

def training_chapters(build, snapshot, root, what):
    """A weak build's training chapters, recomputed from its record (weak.split_ids; a teacher labels build from its
    packs, teacher.build_train_ids) and checked against its sha256."""
    bcfg, split = build.get("config") or {}, build.get("split") or {}
    try:
        if build.get("kind") == TEACHER_BUILD_KIND:
            train = build_train_ids(build, snapshot.chapters, root)
        else:
            order = [r["chapter_id"] for r in json.loads((root / bcfg["test_order"]).read_text(encoding="utf-8"))["order"]]
            train = split_ids(snapshot.chapters, split["dev_chapters"], order, bcfg["max_train_chapter"])["train"]
    except (KeyError, TypeError, OSError, ValueError) as err:
        raise SystemExit(f"{what}: the training chapters cannot be recomputed from the build record ({type(err).__name__})")
    if ids_sha256(train) != split.get("train_ids_sha256"):
        raise SystemExit(f"{what}: the recomputed training chapters do not give the sha256 the build recorded")
    return train


def training_lineage(system, root, snapshot, files, what):
    """Every labels build a trained system's weights were trained on, last stage first: per training run its
    weak_labels build and, for a replay fine-tuning run (novel_ner.finetune), its replay build; then the run its
    init_from names (the checkpoint it started from), and so on. Each run's config.json is checked: the scored run's
    against the sha256 the scoring run recorded, an earlier run's against the config_sha256 its child kept; each names
    its own folder and the snapshot. Returns ([(build, the train_ids_sha256 the run recorded for it, where)], [runs])."""
    run, want, hashed_by = system.get("model_run"), system.get("model_run_config_sha256"), "the scoring run"
    lineage, runs = [], []
    while run:
        if run in runs:
            raise SystemExit(f"{what}: the init_from chain comes back to {run}")
        runs.append(run)
        cfg_file = root / run / "config.json"
        rec = read_json(cfg_file, f"{what}: training run {run}")
        if sha256(cfg_file) != want:
            raise SystemExit(f"{what}: {run}/config.json is not the file {hashed_by} hashed")
        if rec.get("run_id") != Path(run).name or (rec.get("snapshot") or {}).get("id") != snapshot.id:
            raise SystemExit(f"{what}: training run {run} names another run or snapshot")
        files[f"{run}/config.json"] = want
        for block, key, where in (("weak_labels", "build", "weak_labels"), ("replay", "weak_build", "replay")):
            b = rec.get(block)
            if b is None and block == "replay":
                continue
            if not (isinstance(b, dict) and b.get(key) and b.get("train_ids_sha256")):
                raise SystemExit(f"{what}: training run {run} does not record its {block} build and training chapters")
            lineage.append((f"runs/{b[key]}", b["train_ids_sha256"], f"{run} {where}"))
        init = rec.get("init_from")
        if init is None:
            break
        if not (isinstance(init, dict) and init.get("run_id") and init.get("config_sha256")):
            raise SystemExit(f"{what}: training run {run} does not record the run and config sha256 of its init_from")
        run, want, hashed_by = f"runs/{init['run_id']}", init["config_sha256"], f"{run}'s init_from"
    return lineage, runs


def never_in_text(new, train_text):
    """Indices of the new spans whose text occurs nowhere in train_text (training chapters joined by a NUL)."""
    return [i for i, g in enumerate(new) if g[4] not in train_text]


def analysis_c(new, predictions, found, trained, models, build_train):
    """trained = {system: training text key}; build_train = {key: joined training text} (a key is one build, or the
    builds of every stage of a fine-tuned run, see RULES)."""
    by_build = {}
    for build, text in build_train.items():
        never = set(never_in_text(new, text))
        rest = set(range(len(new))) - never
        parts = {"never_in_training_text": never, "in_training_text": rest}
        sub = {}
        for name, idx in parts.items():
            sub[name] = sorted(idx)
            sub[f"{name}_person"] = sorted(i for i in idx if new[i][3] == "person")
            sub[f"{name}_non_person"] = sorted(i for i in idx if new[i][3] != "person")
        by_build[build] = sub
    out = {"builds": {b: {g: group_sizes(new, m) for g, m in sub.items()} for b, sub in by_build.items()},
           "systems": {}, "models": {}}
    for s, build in trained.items():
        out["systems"][s] = {"training_text": build, **{g: group_result(new, m, found[s]) for g, m in by_build[build].items()}}
    for m, runs in models.items():
        if not all(r in trained for r in runs):
            continue
        build = trained[runs[0]]
        if any(trained[r] != build for r in runs):
            raise SystemExit(f"model {m}: its runs come from different training texts")
        sizes = out["builds"][build]
        out["models"][m] = {"runs": runs, "training_text": build,
                            **{g: model_group(runs, {r: out["systems"][r][g] for r in runs}, sizes[g]) for g in sizes}}
    return out


# ---------------------------------------------------------------------------------------------- D: title-suffix answers

def split_title(text, suffixes):
    """(X, suffix) for a text that is X + a title suffix, suffixes tried longest first (X not empty); else None."""
    for x in sorted(suffixes, key=len, reverse=True):
        if text.endswith(x) and len(text) > len(x):
            return text[:-len(x)], x
    return None


def title_answers(gold, texts, suffixes, chapter_rank):
    """[(cid, start, end, X, suffix, text)] for the scored gold person spans whose text is X + a title suffix."""
    out = []
    for cid, ref in gold.items():
        for s, e, t in sorted(ref["spans"]):
            if t != "person" or _touches((s, e), ref["unscored"]):
                continue
            hit = split_title(texts[cid][s:e], suffixes)
            if hit:
                out.append((cid, s, e, hit[0], hit[1], texts[cid][s:e]))
    return sorted(out, key=lambda g: (chapter_rank[g[0]], g[1]))


def handling(span, scored, pending):
    """How one run handles one title-suffix answer: the first of HANDLING that applies (see RULES)."""
    _, s, e, x, *_ = span
    if (s, e, "person") in scored:
        return "exact_hit"
    over = [q for q in scored if _overlaps(q, (s, e))]
    if len(over) == 1 and (over[0][0], over[0][1]) == (s, s + len(x)):
        return "prediction_equals_name_without_suffix"
    if any(_overlaps(q, (s, e)) for q in pending):
        return "pending"
    return "other_overlap" if over else "nothing"


def analysis_d(gold, texts, predictions, models, suffixes, origin, chapter_rank):
    spans = title_answers(gold, texts, suffixes, chapter_rank)
    scored = {s: {cid: {x for x in p[cid]["spans"] if not _touches(x, gold[cid]["unscored"])} for cid in gold}
              for s, p in predictions.items()}
    kind = {s: [handling(g, scored[s][g[0]], predictions[s][g[0]].get("pending", [])) for g in spans] for s in predictions}
    strings = Counter(g[5] for g in spans)
    first = {}
    for i, g in enumerate(spans):
        first.setdefault(g[5], i)
    order = sorted(strings, key=lambda x: (-strings[x], first[x]))
    number = {x: k for k, x in enumerate(order, 1)}
    origin_of = [origin[g[0]].get((g[1], g[2], "person")) for g in spans]
    if any(o not in ORIGINS for o in origin_of):
        raise SystemExit("a title-suffix answer has no recorded origin")

    def tally(idx):
        return {s: {h: sum(1 for i in idx if kind[s][i] == h) for h in HANDLING} for s in predictions}

    def summary(idx):
        per_run = tally(idx)
        return {"spans": len(idx),
                "origin": {o: sum(1 for i in idx if origin_of[i] == o) for o in ORIGINS},
                "decided_by_the_user_the_sheet_or_the_reviewer": sum(1 for i in idx if origin_of[i] != "gpt_agreed"),
                "systems": per_run,
                "models": {m: {h: [per_run[r][h] for r in runs] for h in HANDLING} for m, runs in models.items()}}

    out = {"title_suffixes": len(set(suffixes)), "strings": len(order), **summary(range(len(spans))), "by_string": []}
    for x in order:
        idx = [i for i, g in enumerate(spans) if g[5] == x]
        _, _, _, head, tail, _ = spans[idx[0]]
        out["by_string"].append({"string": number[x], "chars": len(x), "x_chars": len(head), "suffix_chars": len(tail),
                                 **summary(idx)})
    for s in predictions:
        if sum(out["systems"][s].values()) != len(spans):
            raise AssertionError(f"{s}: the handling kinds do not sum to the title-suffix answers")
    keys = {str(number[x]): x for x in order}
    return out, keys


# ---------------------------------------------------------------------------------------------- E: learning-curve steps

def step_record(train_cfg, train_metrics, what):
    """The step facts of one training run (see RULES)."""
    sched, curve, best = train_metrics["schedule"], train_metrics["dev_curve"], train_metrics["best"]["step"]
    if not curve:
        raise SystemExit(f"{what}: no dev evaluations")
    f1 = [p["micro"]["f1"] or 0.0 for p in curve]
    first_max = curve[f1.index(max(f1))]["step"]
    if best != first_max:
        raise SystemExit(f"{what}: the recorded best step is not the first evaluation with the highest dev micro F1")
    at_best = next(p for p in curve if p["step"] == best)
    if (train_metrics.get("dev") or {}).get("at_best", {}).get("micro") not in (None, at_best["micro"]):
        raise SystemExit(f"{what}: the dev scores at the best step differ from the curve's")
    batch = train_cfg["config"]["batch_size"]
    windows = train_metrics["train"]["windows"]
    per_epoch = sched["steps_per_epoch"]
    if per_epoch != -(-windows // batch):
        raise SystemExit(f"{what}: steps per epoch is not the windows divided by the batch size, rounded up")
    nonp = [p["non_person_f1"] or 0.0 for p in curve]
    top = nonp.index(max(nonp))
    return {"train_chapters": train_metrics["train"]["chapters"], "windows_per_epoch": windows, "batch_size": batch,
            "steps_per_epoch": per_epoch, "steps_planned": sched["steps_planned"], "steps_run": sched["steps_run"],
            "stopped_early": sched["stopped_early"], "ran_all_planned_steps": sched["steps_run"] == sched["steps_planned"],
            "dev_evaluations": len(curve), "best_step": best,
            "best_step_x_batch_size": best * batch,
            "windows_seen_at_best_step": (best // per_epoch) * windows + (best % per_epoch) * batch,
            "dev_micro_f1_at_best": at_best["micro"]["f1"], "dev_non_person_f1_at_best": at_best["non_person_f1"],
            "last_evaluated_step": curve[-1]["step"], "dev_micro_f1_at_last": curve[-1]["micro"]["f1"],
            "dev_non_person_f1_at_last": curve[-1]["non_person_f1"],
            "dev_non_person_f1_max": curve[top]["non_person_f1"], "step_of_dev_non_person_f1_max": curve[top]["step"]}


def analysis_e(tiers, models, eval_metrics, root, files, max_train_chapter):
    """max_train_chapter = {trained system: its weak build's max_train_chapter}; every tier label must equal it."""
    runs_out, tiers_out = {}, {}
    for tier, model in tiers.items():
        if model not in models:
            raise SystemExit(f"tier {tier}: {model} is not a model of the scoring run")
        found = sorted({max_train_chapter.get(r) for r in models[model]}, key=str)
        if not tier.isdigit() or found != [int(tier)]:
            raise SystemExit(f"tier {tier}: the weak builds of {model}'s runs have max_train_chapter {found}, not the tier's label")
        for r in models[model]:
            rec = eval_metrics["systems"][r]
            what = f"tier {tier} run {r}"
            if "model_run" not in rec:
                raise SystemExit(f"{what} is not a trained system")
            tdir = root / rec["model_run"]
            cfg_file, met_file = tdir / "config.json", tdir / "metrics.json"
            if sha256(cfg_file) != rec["model_run_config_sha256"]:
                raise SystemExit(f"{what}: {rec['model_run']}/config.json is not the file the scoring run hashed")
            tcfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            tmet = json.loads(met_file.read_text(encoding="utf-8"))
            if not (tcfg.get("run_id") == tmet.get("run_id") == tdir.name):
                raise SystemExit(f"{what}: the training run's records do not both name its folder")
            if (tcfg.get("setting"), tcfg.get("seed")) != (rec.get("setting"), rec.get("seed")):
                raise SystemExit(f"{what}: the training run's setting and seed are not the scoring run's")
            files[f"{rec['model_run']}/config.json"] = sha256(cfg_file)
            files[f"{rec['model_run']}/metrics.json"] = sha256(met_file)
            runs_out[r] = {"tier": tier, "training_run": rec["model_run"], "seed": rec["seed"], **step_record(tcfg, tmet, what)}
        rs = [runs_out[r] for r in models[model]]
        tiers_out[tier] = {"model": model, "runs": models[model],
                           **{k: [x[k] for x in rs] for k in ("steps_planned", "steps_run", "best_step", "windows_seen_at_best_step",
                                                              "dev_non_person_f1_at_best", "dev_non_person_f1_at_last")},
                           "max_train_chapter": int(tier), "stopped_early": [x["stopped_early"] for x in rs],
                           "ran_all_planned_steps": [x["ran_all_planned_steps"] for x in rs]}
    return {"runs": runs_out, "tiers": tiers_out}


# ---------------------------------------------------------------------------------------------- loading (all checked)

def read_json(path, what):
    if not path.is_file():
        raise SystemExit(f"{what}: {path.name} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def check_config(cfg, eval_cfg):
    b = cfg["bootstrap"]
    if {"n": b["rounds"], "seed": b["chapter_seed"]} != {k: eval_cfg["bootstrap"][k] for k in ("n", "seed")}:
        raise SystemExit("bootstrap rounds and chapter_seed must be the scoring run's (so the chapter draws are its draws)")
    if b["run_seed"] == b["chapter_seed"]:
        raise SystemExit("run_seed must differ from chapter_seed")
    if cfg["new_names"] != eval_cfg.get("new_names"):
        raise SystemExit("new_names is not the scoring run's pre-registered rule")
    if any("seen_names" in s for s in eval_cfg.get("span_systems", [])):
        raise SystemExit("a span system of the scoring run has its own seen set; these analyses use the fixed rule only")
    models = eval_cfg.get("models") or {}
    seen = set()
    for pair in cfg["pairs"]:
        if not (isinstance(pair, list) and len(pair) == 2 and pair[0] != pair[1] and all(p in models for p in pair)):
            raise SystemExit(f"pairs: {pair!r} must be two different models of the scoring run")
        if frozenset(pair) in seen:
            raise SystemExit(f"pairs: {pair!r} is listed twice")
        seen.add(frozenset(pair))
    if set(cfg["tiers"].values()) - set(models):
        raise SystemExit("tiers must name models of the scoring run")


def analyse(cfg, root=None):
    """Everything, read and checked; returns what the outputs are made of."""
    root = root or REPO_ROOT
    files = {}
    eval_dir = root / cfg["eval_run"]
    rec = read_json(eval_dir / "config.json", "the scoring run")
    met = read_json(eval_dir / "metrics.json", "the scoring run")
    if not (rec.get("run_id") == met.get("run_id") == eval_dir.name):
        raise SystemExit("the scoring run's config.json and metrics.json do not both name its folder")
    for name in ("config.json", "metrics.json"):
        files[f"{cfg['eval_run']}/{name}"] = sha256(eval_dir / name)
    eval_cfg = rec["config"]
    check_config(cfg, eval_cfg)
    weak = json.loads((root / cfg["title_suffixes_from"]).read_text(encoding="utf-8"))
    suffixes = list(weak["title_suffixes"])
    files[cfg["title_suffixes_from"]] = sha256(root / cfg["title_suffixes_from"])

    snapshot = load_snapshot(eval_cfg["snapshot"], root=root / "data" / "snapshots")
    if snapshot.id != rec["snapshot"]["id"]:
        raise SystemExit("the snapshot is not the scoring run's")
    # a scoring run on a later export (stage 3): the trained systems, the weak builds, the dictionary and the seen names
    # come from its training snapshot, every chapter of which must be in the scoring snapshot with the same text
    training_id = eval_cfg.get("training_snapshot") or eval_cfg["snapshot"]
    training = snapshot if training_id == snapshot.id else load_snapshot(training_id, root=root / "data" / "snapshots")
    snap_lineage = snapshot_lineage(training, snapshot)
    recorded = rec.get("snapshots")
    if snap_lineage and not (isinstance(recorded, dict) and (recorded.get("training_snapshot") or {}).get("id") == training.id
                             and recorded.get("shared_chapters") == snap_lineage["shared_chapters"]
                             and recorded.get("shared_text_identical") is True):
        raise SystemExit("the scoring run does not record this training snapshot lineage (snapshots)")
    gold_file = root / eval_cfg["gold_test"]
    if sha256(gold_file) != rec["gold_test"]["sha256"]:
        raise SystemExit("the answer file is not the one the scoring run hashed")
    files[eval_cfg["gold_test"]] = rec["gold_test"]["sha256"]
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    problems, _, _ = check_gold(export, snapshot)
    if problems:
        raise SystemExit(f"the test answers do not pass the gold check ({len(problems)} problems)")
    test_ids = done_chapters(export)
    if test_ids != rec["gold_test"]["chapters"] or len(test_ids) != len(export["chapters"]):
        raise SystemExit("the test chapters are not the scoring run's")
    gold = gold_spans(export, test_ids)
    texts = {cid: snapshot.chapters[cid].text for cid in test_ids}
    order = json.loads((root / eval_cfg["test_order"]).read_text(encoding="utf-8"))["order"]
    files[eval_cfg["test_order"]] = sha256(root / eval_cfg["test_order"])
    strata = {r["chapter_id"]: r["unit_id"] for r in order if r["chapter_id"] in set(test_ids)}
    if set(strata) != set(test_ids) or dict(Counter(strata.values())) != met["test"]["plot_units"]:
        raise SystemExit("the plot units of the test chapters are not the scoring run's")
    chapter_rank = {cid: i for i, cid in enumerate(test_ids)}

    names = list(met["systems"])
    models = eval_cfg.get("models") or {}
    if any(set(runs) - set(names) for runs in models.values()):
        raise SystemExit("a model of the scoring run names a system it did not score")
    # the dictionary system matched the training snapshot's dictionary in the scoring snapshot's chapters when the scoring
    # run took it from there; errors.load_systems recomputes it from one snapshot object, so it is given that view
    from_training = bool((eval_cfg.get("dictionary") or {}).get("from_training_snapshot"))
    view = replace(snapshot, dictionary=training.dictionary) if snap_lineage and from_training else snapshot
    predictions, _ = load_systems(names, eval_cfg, met, eval_dir, root, view, test_ids, export, files)
    tc = {}
    for s in names:
        tc[s] = type_counts(gold, predictions[s])
        if json.loads(json.dumps(tc[s])) != met["per_chapter_type_counts"][s]:
            raise SystemExit(f"{s}: the per-chapter, per-type counts differ from the scoring run's")
        tp, fp, fn = (sum(c[t][i] for c in tc[s].values() for t in (*ENTITY_TYPES, "_invalid")) for i in range(3))
        if _prf_defined(tp, fp, fn) != met["scores"][s]["micro"]:
            raise SystemExit(f"{s}: the micro scores differ from the scoring run's")

    dictionary, mentions, seen_sources = weak_names(training, cfg["new_names"]["cutoff_chapter"])  # the training snapshot's weak sources
    seen = frozenset(dictionary | mentions)
    new_counts = {s: new_name_chapter_counts(gold, predictions[s], texts, seen) for s in names}
    view = met["new_name_recall"]
    for s in names:
        got = tuple(sum(v[i] for v in new_counts[s].values()) for i in range(3))
        if got != tuple(view["systems"][s][k] for k in ("tp", "fp", "fn")):
            raise SystemExit(f"{s}: the new-name counts differ from the scoring run's")
    new = new_gold(gold, texts, seen)
    gold_seen = [v["gold"] for v in view["seen_sets"].values()]
    if len(gold_seen) != 1 or gold_seen[0]["new_spans"] != len(new):
        raise SystemExit("the new gold spans differ from the scoring run's")

    # A
    vectors = {s: chapter_vectors(met["per_chapter_type_counts"][s], new_counts[s]) for s in names}
    pairs = [tuple(p) for p in cfg["pairs"]]
    b = cfg["bootstrap"]
    boot = two_level_bootstrap(vectors, models, pairs, strata, b["rounds"], b["chapter_seed"], b["run_seed"])
    replication = check_replication(boot["diffs"]["chapter_only"], pairs, met["models"]["pooled_paired"],
                                    view["pooled_paired_difference_recall"])
    a_pairs = analysis_a(boot, pairs, replication)
    a_out = {"pairs": a_pairs,
             "models": {m: {"runs": models[m], "run_draws": boot["run_draws"][m],
                            "pooled": {k: round(v, 4) for k, v in boot["models"][m].items()}} for m in boot["models"]},
             "interval_side_changes_with_run_draws": side_changes(a_pairs)}

    # B
    conc = cfg["concentration"]
    b_out, found = analysis_b(new, texts, predictions, models, cfg["frequency_bands"], suffixes, seen, conc)
    for s in names:
        pn = per_name_recall(gold, predictions[s], texts, seen)["names"]
        if (b_out["systems"][s]["all"]["spans_found"], b_out["systems"][s]["all"]["names_found"]) != (
                view["systems"][s]["tp"], pn["found"]):
            raise SystemExit(f"{s}: the breakdown's totals differ from the new-name recall")

    # C
    trained, final_build, chains, build_train, build_max, texts_of = {}, {}, {}, {}, {}, {}

    def build_chapters(build):
        if build not in build_train:
            bdir = root / build
            brec = read_json(bdir / "config.json", f"weak build {build}")
            if brec.get("run_id") != bdir.name or (brec.get("snapshot") or {}).get("id") != training.id:
                raise SystemExit(f"weak build {build}: its record names another run or snapshot")
            files[f"{build}/config.json"] = sha256(bdir / "config.json")
            build_train[build] = training_chapters(brec, training, root, f"weak build {build}")
            build_max[build] = brec["config"]["max_train_chapter"]
        return build_train[build]

    for s in names:
        wb = met["systems"][s].get("weak_build")
        if not wb:
            continue
        build = wb["build"]
        if ids_sha256(build_chapters(build)) != wb["train_ids_sha256"]:
            raise SystemExit(f"{s}: its training chapters are not the ones the scoring run recorded")
        lineage, chains[s] = training_lineage(met["systems"][s], root, training, files, s)
        if lineage[0][0] != build:
            raise SystemExit(f"{s}: its training run's weak labels are not the build the scoring run recorded")
        for b, want, where in lineage:
            if ids_sha256(build_chapters(b)) != want:
                raise SystemExit(f"{s}: the training chapters of {b} are not the ones {where} recorded")
        builds = list(dict.fromkeys(b for b, _, _ in lineage))
        key = " + ".join(builds)
        texts_of.setdefault(key, builds)
        trained[s], final_build[s] = key, build
    union = {k: sorted(set().union(*(build_train[b] for b in bs))) for k, bs in texts_of.items()}
    c_out = analysis_c(new, predictions, found, trained, models,
                       {k: "\x00".join(training.chapters[c].text for c in ids) for k, ids in union.items()})
    for key, builds in texts_of.items():
        c_out["builds"][key]["training_chapters"] = len(union[key])
        c_out["builds"][key]["from_builds"] = {b: len(build_train[b]) for b in builds}
    for s, runs in chains.items():
        c_out["systems"][s]["training_runs"] = runs

    # D
    origin, _ = gold_origin(export, test_ids)
    d_out, keys = analysis_d(gold, texts, predictions, models, suffixes, origin, chapter_rank)

    # E
    e_out = analysis_e(cfg["tiers"], models, met, root, files, {s: build_max[b] for s, b in final_build.items()})

    return {"files": files, "eval_rec": rec, "systems": names, "models": models, "seen_sources": seen_sources,
            "snap_lineage": snap_lineage,
            "test": {"chapters": len(test_ids), "scored_gold_spans": met["test"]["scored_gold_spans"], "new_gold_spans": len(new)},
            "A": a_out, "B": b_out, "C": c_out, "D": d_out, "E": e_out, "keys": {"D_strings": keys}}


# ---------------------------------------------------------------------------------------------- outputs

def build_metrics(run_id, cfg, a):
    m = {
        "run_id": run_id, "post_hoc": POST_HOC, "eval_run": cfg["eval_run"], "test": a["test"],
        "fixed_new_name_rule": {"rule": cfg["new_names"], "seen_sources": a["seen_sources"]},
        "A_seed_aware_paired_intervals": {
            "rounds": cfg["bootstrap"]["rounds"], "chapter_seed": cfg["bootstrap"]["chapter_seed"],
            "run_seed": cfg["bootstrap"]["run_seed"], "levels": list(LEVELS), "statistics": list(STATS), **a["A"]},
        "B_new_name_breakdowns": a["B"],
        "C_never_in_training_text": a["C"],
        "D_title_suffix_answers": a["D"],
        "E_learning_curve_steps": a["E"],
        "checks": {"per_chapter_type_counts_and_micro_equal_the_scoring_run": True,
                   "new_name_counts_equal_the_scoring_run": True,
                   "chapter_only_rounds_give_the_scoring_runs_stored_pooled_intervals_in_their_direction": True,
                   "breakdown_totals_equal_the_new_name_recall_and_errors_per_name_count": True,
                   "training_chapters_sha256_equal_the_weak_builds_and_the_scoring_run": True,
                   "earlier_stage_builds_through_init_from_and_replay_checked_and_in_the_training_text": True,
                   "training_run_configs_are_the_files_the_scoring_run_hashed": True,
                   "tier_labels_equal_the_weak_builds_max_train_chapter": True,
                   "input_files_hashed": len(a["files"])},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    check_counts_only(m)
    return m


def build_config(run_id, config_file, cfg, a, started, code):
    out = {"run_id": run_id, "config_file": config_file, "config": cfg, "post_hoc": POST_HOC,
           "eval_run": {"run": cfg["eval_run"], "run_code": a["eval_rec"].get("code"), "snapshot": a["eval_rec"]["snapshot"],
                        **({"snapshots": a["eval_rec"]["snapshots"]} if a.get("snap_lineage") else {}),
                        "gold_test": {k: a["eval_rec"]["gold_test"][k] for k in ("file", "sha256")}},
           "inputs_sha256": a["files"], "systems": a["systems"], "models": a["models"], "rules": RULES,
           "code": code, "started_at": started}
    check_counts_only(out)
    return out


def outputs(run_id):
    return REPO_ROOT / "runs" / run_id, REPO_ROOT / "data" / "analysis" / run_id / "keys.json"


def refuse_existing(run_id):
    run_dir, keys = outputs(run_id)
    taken = [p for p in (run_dir / "config.json", run_dir / "metrics.json", keys) if p.exists()]
    if taken:
        raise SystemExit(f"{taken[0]} already exists; use a new run id (outputs are never overwritten)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    refuse_existing(run_id)
    started, code = time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_state()
    if code.get("dirty") is not False:
        raise SystemExit("src/ or configs/ has uncommitted changes; a recorded run is made from committed code")
    a = analyse(cfg)
    metrics = build_metrics(run_id, cfg, a)
    config = build_config(run_id, Path(args.config).as_posix(), cfg, a, started, code)
    for pair, v in a["A"]["pairs"].items():
        print(f"{pair:28s} " + " | ".join(
            f"{k} {v['point'][k]:+.4f} chapters {v['chapter_only'][k]['ci95']} +runs {v['chapters_and_runs'][k]['ci95']}"
            for k in ("micro_f1", "new_name_recall")))
    changes = a["A"]["interval_side_changes_with_run_draws"]
    print(f"A intervals on a different side of 0 once runs are drawn: {len(changes)}")
    for c in changes:
        print(f"  {c['pair']:28s} {c['statistic']:16s} point {c['point']:+.4f}: chapters {c['chapter_only_ci95']} "
              f"({c['chapter_only']}) -> +runs {c['chapters_and_runs_ci95']} ({c['chapters_and_runs']})")
    conc = a["B"]["concentration"]
    print(f"new gold: {conc['new_spans']} spans, {conc['names']} names; top {conc['top_n']} {conc['top_n_spans']} spans; "
          f"{conc['frequent_names']} names with >= {conc['frequent_min_spans']} spans carry {conc['frequent_names_spans']}")
    for m, v in a["B"]["models"].items():
        print(f"B {m:16s} recall {v['all']['recall_mean']} {v['all']['recall_range']} | per name {v['all']['name_share_mean']} "
              f"{v['all']['name_share_range']} | person {v['person']['recall_mean']} non-person {v['non_person']['recall_mean']}")
    for m, v in a["C"]["models"].items():
        print(f"C {m:16s} never in training text: recall {v['never_in_training_text']['recall_mean']} "
              f"{v['never_in_training_text']['recall_range']}, names {v['never_in_training_text']['name_share_mean']}")
    d = a["D"]
    print(f"D {d['spans']} title-suffix answers, {d['strings']} strings, decided by user/sheet/reviewer "
          f"{d['decided_by_the_user_the_sheet_or_the_reviewer']}")
    for t, v in a["E"]["tiers"].items():
        print(f"E tier {t}: steps run {v['steps_run']} of {v['steps_planned']} (all planned {v['ran_all_planned_steps']}, "
              f"stopped_early flag {v['stopped_early']}), best {v['best_step']}, dev non-person F1 at best "
              f"{v['dev_non_person_f1_at_best']} at last {v['dev_non_person_f1_at_last']}")
    refuse_existing(run_id)
    run_dir, keys_file = outputs(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    keys_file.write_text(json.dumps({"run_id": run_id, "note": "not committed: holds strings from the book", **a["keys"]},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {run_dir} and {keys_file}")


if __name__ == "__main__":
    main()
