"""Diagnostics of the S3 tagger (weak supervision, three seeds) on the 31 development chapters (answers v2), with the 60
test chapters as a secondary view: the scratch diagnostics of 2026-10-04 (stage 0) as a recorded, tested run.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.diagnose configs/diagnose_dev_v1.json [--run-id ID] [--out DIR]

Without --out the run goes to runs/<run_id>/ (config.json and metrics.json, committed) and data/analysis/<run_id>/keys.json
(not committed: the book's strings behind the numbered keys of 6_extras), and needs committed code (src/ and configs/
clean). With --out DIR all three files go to DIR (a scratch run, no clean-code requirement). Nothing is overwritten.

DESCRIPTIVE, chosen after scores were seen: every analysis was first computed in scratch on 2026-10-04 while the route was
being drafted (the research notes cite them); none is a pre-registered test, and nothing about any system is
chosen with them. They separate what the tagger's errors have in common with its own weak labels (L), what they look
like (M) and whether chunk edges matter (C), on the development chapters the checkpoints were picked on. The test
chapters are a secondary view that has been looked at many times.

Nobody judges whether an answer is right. Claude wrote this program and must not (Opus is a compared model): every count
comes from the fixed mechanical rules in the config's "rules" block, applied by the program to the answer files, the
weak labels and the three seeds' outputs. A disagreement between the tagger (or its labels) and an answer file is never
called an error of the answer. The two what-if post-processings of 4b are measured on the same chapters they are
described on, with no parameter fitted: indicative only, not a result.

Inputs, all checked (the run stops on any mismatch):
- the snapshot (load_snapshot verifies its files) and the weak build: its record runs/<build>/config.json must name its
  folder and the snapshot, the labels file of the setting must have the sha256 the record holds, the title suffixes read
  from title_suffixes_from must equal the ones the build was made with, and every label span's text must be the chapter
  text at its place;
- the development answers: the gold check passes; the file has the sha256 the weak build and every training run recorded;
  every answered chapter has a dev row in the labels and no training row is an answered chapter of either set;
- the three training runs: config.json and metrics.json name their folder; snapshot, weak build, setting and the labels'
  sha256 are the config's; max_chars (the chunk size of 4_context) is theirs; the seeds are distinct; each
  predictions_dev.jsonl has the sha256 its metrics.json recorded, covers exactly the answered chapters, and every row's
  text is the chapter text at its place;
- the test answers (gold check; the sha256 the scoring run named by test.gold_recorded_in holds) and the three predict
  runs: model_run is the training run of the same seed, model_run_config_sha256 is that run's config.json, same setting,
  seed and snapshot; predictions.jsonl checked like the dev files;
- the scores this run computes agree with the records: each seed's strict micro on the development chapters equals its
  training run's dev.at_best (tp, fp, fn, P, R, F1), and on the test chapters the scoring run's (the metrics.json beside
  test.gold_recorded_in) scores of the system whose predictions_run is that predict run, with the same predictions
  sha256; the chunks of 4_context are the tagger's: the runs' recorded chunk counts equal this run's, no chunk was split
  into token windows (windows = chunks, chunks_over_max_tokens = 0).

Outputs: config.json (inputs with sha256, the config with its rules, the runs' code states, this code's state) and
metrics.json (numbers, chapter ids and fixed English words only; no string from the book, checked) and keys.json
(the strings behind the numbered keys, under data/analysis/, never committed).
"""
import argparse
import json
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.errors import check_counts_only
from novel_ner.evaluate import as_prediction, chapter_chunks, done_chapters, sha256
from novel_ner.gold import check as check_gold
from novel_ner.score import _prf, _touches, gold_spans
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.zeroshot import git_state

DEV_ANALYSES = ("1_ceiling", "2_agreement_with_labels", "3_error_kinds", "4_context", "4b_consistency", "5_seeds", "6_extras")
EXAM_ANALYSES = ("micro_check", "3_error_kinds", "4_context", "4b_consistency", "5_seeds", "6_extras_exam_view")
SEEN_KINDS = ("unseen", "seen_same_type", "seen_other_type_only")
EDGE_VIEWS = (("all_chunk_edges", 0), ("internal_chunk_cuts_only", 1), ("paragraph_edges_control", 2))
SHA_KEY = {"predictions_dev.jsonl": "predictions_dev_sha256", "predictions.jsonl": "predictions_sha256"}
SECONDARY = "exam already looked at many times; descriptive only"


# ---------------------------------------------------------------------------------------------- shared pieces

def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}


def scores(ids, gold, pred):
    """Strict micro and per-type counts and scores of one prediction {chapter: {(start, end, type)}}."""
    c = {t: Counter() for t in ("micro", *ENTITY_TYPES)}
    for cid in ids:
        g, p = gold[cid], pred[cid]
        for s in g & p:
            c[s[2]]["tp"] += 1
            c["micro"]["tp"] += 1
        for s in p - g:
            c[s[2]]["fp"] += 1
            c["micro"]["fp"] += 1
        for s in g - p:
            c[s[2]]["fn"] += 1
            c["micro"]["fn"] += 1
    return {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in c.items()}


def pooled(score_list):
    """Counts added over the seeds, one ratio; mean_seed_f1 = the mean of the seeds' (rounded) F1."""
    out = {}
    for k in score_list[0]:
        tp, fp, fn = (sum(s[k][i] for s in score_list) for i in ("tp", "fp", "fn"))
        out[k] = {**prf(tp, fp, fn), "mean_seed_f1": round(sum(s[k]["f1"] for s in score_list) / len(score_list), 4)}
    return out


def bucket_of(d, buckets):
    """The label of the one bucket with min <= d < max (max null = no upper end); the buckets must tile the distances."""
    found = [b["label"] for b in buckets if b["min"] <= d and (b.get("max") is None or d < b["max"])]
    if len(found) != 1:
        raise SystemExit(f"a distance of {d} falls in {len(found)} edge buckets; the buckets must not overlap or leave gaps")
    return found[0]


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else None


def ranks(a):
    """Average ranks (0-based) with ties sharing the mean rank."""
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for m in range(i, j + 1):
            r[order[m]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def _round(x, places):
    return None if x is None else round(x, places)


# ---------------------------------------------------------------------------------------------- the training labels

class TrainingIndex:
    """What the training labels (the weak build's train rows) say about a string: which types it is a positive of, and
    how every occurrence of it in the training text is labelled."""

    def __init__(self, train_rows, chapters, thresholds):
        self.pos_types = defaultdict(Counter)
        self.chapters = []
        self.high, self.low = thresholds["same_type_high"], thresholds["same_type_low"]
        self._cache = {}
        for r in train_rows:
            t = chapters[r["chapter_id"]].text
            spans = []
            cover = [-1] * len(t)
            for i, x in enumerate(r["spans"]):
                if t[x["start"]:x["end"]] != x["text"]:
                    raise SystemExit(f"{r['chapter_id']}: a label span's text is not the chapter text at its place")
                spans.append((x["start"], x["end"], x["type"]))
                self.pos_types[x["text"]][x["type"]] += 1
                for j in range(x["start"], x["end"]):
                    cover[j] = i
            unc = bytearray(len(t))
            for x in r["uncertain"]:
                for j in range(x["start"], x["end"]):
                    unc[j] = 1
            self.chapters.append((t, cover, spans, unc))

    def seen_kind(self, x, typ):
        c = self.pos_types.get(x)
        if not c:
            return "unseen"
        return "seen_same_type" if c[typ] else "seen_other_type_only"

    def profile(self, x, typ):
        """Counter of every occurrence of x in the training text (overlapping occurrences included) by how the labels treat
        it: exact_same_type, exact_other_type, inside_longer_positive, partial_overlap_positive, uncertain_region or O,
        in the order the kinds were first met (profile_class breaks ties by it)."""
        key = (x, typ)
        if key in self._cache:
            return self._cache[key]
        c = Counter()
        for t, cover, spans, unc in self.chapters:
            at = t.find(x)
            while at != -1:
                s, e = at, at + len(x)
                idx = set(cover[s:e]) - {-1}
                if len(idx) == 1:
                    ss, se, st = spans[next(iter(idx))]
                    if (ss, se) == (s, e):
                        kind = "exact_same_type" if st == typ else "exact_other_type"
                    elif ss <= s and e <= se:
                        kind = "inside_longer_positive"
                    else:
                        kind = "partial_overlap_positive"
                elif len(idx) > 1:
                    kind = "partial_overlap_positive"
                elif any(unc[s:e]):
                    kind = "uncertain_region"
                else:
                    kind = "O"
                c[kind] += 1
                at = t.find(x, at + 1)
        self._cache[key] = c
        return c

    def profile_class(self, x, typ):
        """One class per instance from the string's training occurrences (not weighted by how often the error occurs)."""
        c = self.profile(x, typ)
        n = sum(c.values())
        if n == 0:
            return "absent_from_training_text"
        f = c["exact_same_type"] / n
        lo, hi = f"{self.low * 100:g}", f"{self.high * 100:g}"
        if f >= self.high:
            return f"train_occ_labelled_same_type>={hi}%"
        if f >= self.low:
            return f"train_occ_labelled_same_type_{lo}-{hi}%"
        rest = Counter({k: v for k, v in c.items() if k != "exact_same_type"})
        top = max(rest, key=rest.get)  # ties: the kind met first in the training scan (the Counter's insertion order)
        return f"train_occ_<{lo}%_same_type_mostly_" + top


# ---------------------------------------------------------------------------------------------- error kinds

def fp_kind(p, gold_set, text, titles):
    """(kind, boundary subkind) of a false positive against the chapter's scored gold spans (see rules 3)."""
    s, e, _ = p
    if any((a, b) == (s, e) for a, b, _ in gold_set):
        return "type", None
    ov = sorted(g for g in gold_set if overlaps(g, p))
    if not ov:
        return "spurious", None
    sub = None
    for g in ov:
        if g[0] == s and e > g[1] and text[g[1]:e] in titles:
            sub = "wider_by_title(pred=gold+title)"
        elif g[0] == s and e < g[1] and text[e:g[1]] in titles:
            sub = "narrower_by_title(pred=gold-title)"
    if sub is None:
        if len(ov) > 1:
            sub = "merged_or_multi"
        else:
            g = ov[0]
            sub = "narrower_other" if g[0] <= s and e <= g[1] else "wider_other" if s <= g[0] and g[1] <= e else "crossing"
    return "boundary", sub


def fn_kind(g, pred_set, text, titles):
    """(kind, boundary subkind) of a false negative against the chapter's scored predictions (see rules 3)."""
    s, e, _ = g
    ov = sorted(p for p in pred_set if overlaps(p, g))
    if not ov:
        return "missed", None
    if any((p[0], p[1]) == (s, e) for p in ov):
        return "type", None
    sub = "other"
    for p in ov:
        if p[0] == s and p[1] < e and text[p[1]:e] in titles:
            sub = "pred=gold-title"
        elif p[0] == s and p[1] > e and text[e:p[1]] in titles:
            sub = "pred=gold+title"
    return "boundary", sub


# ---------------------------------------------------------------------------------------------- 1, 2: the labels

def ceiling(ids, gold, preds, labels):
    lab = scores(ids, gold, labels)
    per_seed = {k: scores(ids, gold, p) for k, p in preds.items()}
    pool = pooled(list(per_seed.values()))
    rows = {}
    for t in ("micro", *ENTITY_TYPES):
        rows[t] = {"labels": lab[t], **{k: per_seed[k][t] for k in preds}, "pooled": pool[t],
                   "pooled_f1_minus_labels_f1": round(pool[t]["f1"] - lab[t]["f1"], 4),
                   "pooled_P_minus_labels_P": round(pool[t]["precision"] - lab[t]["precision"], 4),
                   "pooled_R_minus_labels_R": round(pool[t]["recall"] - lab[t]["recall"], 4)}
    split = {}
    for k, p in preds.items():
        c = Counter()
        for cid in ids:
            for g in gold[cid]:
                side = "labels_have_it" if g in labels[cid] else "labels_miss_it"
                c[side + "_n"] += 1
                c[side + "_found"] += g in p[cid]
            for x in labels[cid] - gold[cid]:
                c["label_fp_n"] += 1
                c["label_fp_also_predicted_by_model"] += x in p[cid]
        split[k] = dict(c)
    return {"per_type": rows, "model_recall_by_label_status": split}


def fp_place(x, lab, unc):
    return "fp_in_labels" if x in lab else ("fp_where_labels_uncertain" if _touches(x, unc) else "fp_not_in_labels")


def fn_place(x, lab, unc):
    if x in lab:
        return "fn_labels_have_it"
    if _touches(x, unc):
        return "fn_labels_miss:uncertain_region"
    if any(overlaps(x, y) for y in lab):
        same = any((y[0], y[1]) == (x[0], x[1]) for y in lab)
        return "fn_labels_miss:labels_other_type" if same else "fn_labels_miss:labels_other_boundary"
    return "fn_labels_miss:labels_O"


def agreement(ids, gold, preds, labels, uncertain):
    tallies = {k: Counter() for k in (*preds, "pooled", "stable_all3")}
    by_type = defaultdict(Counter)
    for k, p in preds.items():
        c = tallies[k]
        for cid in ids:
            g, lab, unc = gold[cid], labels[cid], uncertain[cid]
            for x in p[cid] - g:
                where = fp_place(x, lab, unc)
                c[where] += 1
                c["fp"] += 1
                by_type[x[2]][where] += 1
            for x in g - p[cid]:
                where = fn_place(x, lab, unc)
                c[where] += 1
                c["fn"] += 1
                by_type[x[2]][where] += 1
        tallies["pooled"].update(c)
    stable = tallies["stable_all3"]
    for cid in ids:
        g, lab, unc = gold[cid], labels[cid], uncertain[cid]
        in_all = set.intersection(*(p[cid] for p in preds.values()))
        in_any = set.union(*(p[cid] for p in preds.values()))
        for x in in_all - g:
            stable["fp"] += 1
            stable[fp_place(x, lab, unc)] += 1
        for x in g - in_any:
            stable["fn"] += 1
            stable[fn_place(x, lab, unc)] += 1
    out = {}
    for k, c in tallies.items():
        shared = c["fp_in_labels"] + sum(v for kk, v in c.items() if kk.startswith("fn_labels_miss"))
        errors = c["fp"] + c["fn"]
        out[k] = {**dict(c), "errors": errors, "errors_shared_with_labels": shared,
                  "share_shared": round(shared / errors, 4) if errors else None,
                  "fp_share_in_labels": round(c["fp_in_labels"] / c["fp"], 4) if c["fp"] else None,
                  "fn_share_labels_miss": round((c["fn"] - c["fn_labels_have_it"]) / c["fn"], 4) if c["fn"] else None}
    out["by_type_pooled"] = {t: dict(v) for t, v in by_type.items()}
    out["base_rates"] = {"gold_spans": sum(len(gold[c]) for c in ids),
                         "gold_spans_missing_from_labels": sum(len(gold[c] - labels[c]) for c in ids),
                         "label_positives": sum(len(labels[c]) for c in ids),
                         "label_positives_wrong": sum(len(labels[c] - gold[c]) for c in ids)}
    lab_err = [len(labels[c] ^ gold[c]) for c in ids]
    mod_err = [sum(len(p[c] ^ gold[c]) for p in preds.values()) / len(preds) for c in ids]
    out["per_chapter_error_correlation"] = {
        "pearson": _round(pearson(lab_err, mod_err), 3), "spearman": _round(spearman(lab_err, mod_err), 3),
        "top5_chapters_share_of_model_errors": round(sum(sorted(mod_err, reverse=True)[:5]) / sum(mod_err), 3) if sum(mod_err) else None,
        "top5_chapters_share_of_label_errors": round(sum(sorted(lab_err, reverse=True)[:5]) / sum(lab_err), 3) if sum(lab_err) else None}
    return out


# ---------------------------------------------------------------------------------------------- 3: error kinds

def error_kinds(ids, gold, preds, text, index, titles):
    out = {}
    pool = defaultdict(Counter)
    for k, p in preds.items():
        c = defaultdict(Counter)
        for cid in ids:
            g, pk, txt = gold[cid], p[cid], text[cid]
            for x in sorted(pk - g):
                kind, sub = fp_kind(x, g, txt, titles)
                c["fp_kind"][kind] += 1
                c["fp_kind_by_type"][f"{x[2]}:{kind}"] += 1
                if sub:
                    c["fp_boundary_sub"][sub] += 1
                if kind == "type":
                    for a, b, t in sorted(g):
                        if (a, b) == (x[0], x[1]):
                            c["type_confusion_gold>pred"][f"{t}>{x[2]}"] += 1
                            break
                s = txt[x[0]:x[1]]
                if kind == "spurious":
                    if any(txt[a:b] == s for a, b, _ in g):
                        c["fp_spurious_sub"]["string_is_gold_elsewhere_in_chapter"] += 1
                    elif len(s) == 1:
                        c["fp_spurious_sub"]["single_char"] += 1
                    else:
                        c["fp_spurious_sub"]["other"] += 1
                c["fp_string_in_training_positives"][index.seen_kind(s, x[2])] += 1
                c["fp_training_occurrence_profile"][index.profile_class(s, x[2])] += 1
            for x in sorted(g - pk):
                kind, sub = fn_kind(x, pk, txt, titles)
                c["fn_kind"][kind] += 1
                c["fn_kind_by_type"][f"{x[2]}:{kind}"] += 1
                if sub:
                    c["fn_boundary_sub"][sub] += 1
                s = txt[x[0]:x[1]]
                c["fn_string_in_training_positives"][index.seen_kind(s, x[2])] += 1
                c["fn_training_occurrence_profile"][index.profile_class(s, x[2])] += 1
            for x in sorted(g & pk):
                c["tp_training_occurrence_profile"][index.profile_class(txt[x[0]:x[1]], x[2])] += 1
        out[k] = {kk: dict(v) for kk, v in c.items()}
        for kk, v in c.items():
            pool[kk].update(v)
    out["pooled"] = {kk: dict(v) for kk, v in pool.items()}
    rec = defaultdict(Counter)
    for cid in ids:
        for x in gold[cid]:
            sk = index.seen_kind(text[cid][x[0]:x[1]], x[2])
            rec[sk]["gold"] += 1
            rec[sk]["found_seed_sum"] += sum(x in p[cid] for p in preds.values())
    out["recall_by_seen_status"] = {sk: {**dict(v), "pooled_recall": round(v["found_seed_sum"] / (len(preds) * v["gold"]), 4)}
                                    for sk, v in rec.items()}
    fn_str, fp_str = Counter(), Counter()
    for p in preds.values():
        for cid in ids:
            for x in gold[cid] - p[cid]:
                fn_str[text[cid][x[0]:x[1]]] += 1
            for x in p[cid] - gold[cid]:
                fp_str[text[cid][x[0]:x[1]]] += 1
    top = lambda c: sum(sorted(c.values(), reverse=True)[:10])
    out["string_concentration_pooled"] = {
        "fn_errors": sum(fn_str.values()), "fn_distinct_strings": len(fn_str),
        "fn_top10_strings_share": round(top(fn_str) / max(1, sum(fn_str.values())), 3),
        "fp_errors": sum(fp_str.values()), "fp_distinct_strings": len(fp_str),
        "fp_top10_strings_share": round(top(fp_str) / max(1, sum(fp_str.values())), 3)}
    return out


# ---------------------------------------------------------------------------------------------- 4: chunk edges

def chunk_info(chapter, max_chars):
    """([(chunk start, chunk end)], [(paragraph start, paragraph end)]) in chapter-text offsets."""
    chunks = [(s, s + len(t)) for s, t in chapter_chunks(chapter, max_chars)]
    paras, at = [], 0
    for _, t in chapter.paragraphs:
        paras.append((at, at + len(t)))
        at += len(t) + 1
    return chunks, paras


def edge_dist(span, chunks, paras, n):
    """(distance to the nearest edge of the span's chunk, to the nearest internal cut (the chapter's own start and end
    do not count; a huge number when the chunk has none), to the nearest edge of the span's paragraph)."""
    s, e = span[0], span[1]
    i = bisect_right([a for a, _ in chunks], s) - 1
    cs, ce = chunks[i]
    if not (cs <= s and e <= ce):
        raise SystemExit("a span crosses a chunk edge; chunks are whole paragraphs and spans never cross a paragraph break")
    left, right = s - cs, ce - e
    far = 10 ** 9
    d_int = min(left if cs > 0 else far, right if ce < n else far)
    j = bisect_right([a for a, _ in paras], s) - 1
    ps, pe = paras[j]
    return min(left, right), d_int, min(s - ps, pe - e)


def context(ids, gold, preds, text, chapters, max_chars, buckets, labels=None):
    info = {c: chunk_info(chapters[c], max_chars) for c in ids}
    res = {}
    for name, idx in EDGE_VIEWS:
        gb, pb, lb = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for cid in ids:
            chunks, paras = info[cid]
            n = len(text[cid])
            for x in gold[cid]:
                b = bucket_of(edge_dist(x, chunks, paras, n)[idx], buckets)
                gb[b]["gold"] += 1
                for k, p in preds.items():
                    gb[b][f"fn_{k}"] += x not in p[cid]
                if labels is not None:
                    lb[b]["gold"] += 1
                    lb[b]["labels_fn"] += x not in labels[cid]
            for k, p in preds.items():
                for x in p[cid]:
                    b = bucket_of(edge_dist(x, chunks, paras, n)[idx], buckets)
                    pb[b][f"pred_{k}"] += 1
                    pb[b][f"fp_{k}"] += x not in gold[cid]
            if labels is not None:
                for x in labels[cid]:
                    b = bucket_of(edge_dist(x, chunks, paras, n)[idx], buckets)
                    lb[b]["label_pos"] += 1
                    lb[b]["label_fp"] += x not in gold[cid]
        table = {}
        for bk in buckets:
            b = bk["label"]
            g, p = gb[b], pb[b]
            row = {"gold_spans": g["gold"]}
            for k in preds:
                row[f"fn_rate_{k}"] = f"{g[f'fn_{k}']}/{g['gold']}"
                row[f"fp_rate_{k}"] = f"{p[f'fp_{k}']}/{p[f'pred_{k}']}"
                row[f"fn_{k}"], row[f"fp_{k}"], row[f"predictions_{k}"] = g[f"fn_{k}"], p[f"fp_{k}"], p[f"pred_{k}"]
            fn_sum = sum(g[f"fn_{k}"] for k in preds)
            fp_sum = sum(p[f"fp_{k}"] for k in preds)
            pr_sum = sum(p[f"pred_{k}"] for k in preds)
            row["fn_rate_pooled"] = round(fn_sum / (len(preds) * g["gold"]), 4) if g["gold"] else None
            row["fn_pooled_counts"] = f"{fn_sum}/{len(preds) * g['gold']}"
            row["fp_rate_pooled"] = round(fp_sum / pr_sum, 4) if pr_sum else None
            row["fp_pooled_counts"] = f"{fp_sum}/{pr_sum}"
            row["fn_pooled"], row["fp_pooled"], row["predictions_pooled"] = fn_sum, fp_sum, pr_sum
            if labels is not None:
                row["labels_fn_rate"] = f"{lb[b]['labels_fn']}/{lb[b]['gold']}"
                row["labels_fp_rate"] = f"{lb[b]['label_fp']}/{lb[b]['label_pos']}"
                row["labels_fn"], row["labels_positives"], row["labels_fp"] = lb[b]["labels_fn"], lb[b]["label_pos"], lb[b]["label_fp"]
            table[b] = row
        res[name] = table
    sizes = [b - a for c in ids for a, b in info[c][0]]
    res["chunks"] = {"chunks": len(sizes), "chapters": len(ids), "mean_chars": round(sum(sizes) / len(sizes), 1),
                     "single_chunk_chapters": sum(len(info[c][0]) == 1 for c in ids), "max_chars": max_chars}
    return res


# ---------------------------------------------------------------------------------------------- 4b: consistency

def type_vote(pred, text, score):
    """Every prediction of a string in a chapter takes the type most of them have; ties by the summed score, then by the
    type order."""
    by_str = defaultdict(list)
    for x in sorted(pred):
        by_str[text[x[0]:x[1]]].append(x)
    out = set()
    for s, xs in by_str.items():
        cnt, sc = Counter(), Counter()
        for x in xs:
            cnt[x[2]] += 1
            sc[x[2]] += score.get(x, 0)
        best = max(sorted(cnt), key=lambda t: (cnt[t], sc[t], -ENTITY_TYPES.index(t)))
        out |= {(x[0], x[1], best) for x in xs}
    return out


def propagate(pred, text):
    """Every other exact occurrence of a predicted string of 2+ characters that overlaps no prediction and nothing
    already added gets the string's majority type (ties by the type order). Strings in order of their first predicted
    place; occurrences left to right."""
    by_str = defaultdict(list)
    for x in sorted(pred):
        by_str[text[x[0]:x[1]]].append(x)
    out = set(pred)
    occupied = [(x[0], x[1]) for x in sorted(pred)]
    for s, xs in by_str.items():
        if len(s) < 2:
            continue
        cnt = Counter(x[2] for x in xs)
        t = max(sorted(cnt), key=lambda t: (cnt[t], -ENTITY_TYPES.index(t)))
        at = text.find(s)
        while at != -1:
            span = (at, at + len(s))
            if "\n" not in s and not any(a < span[1] and span[0] < b for a, b in occupied):
                out.add((span[0], span[1], t))
                occupied.append(span)
            at = text.find(s, at + 1)
    return out


def consistency(ids, gold, preds, text, scores_of):
    res = {}
    pool = Counter()
    for k, p in preds.items():
        c = Counter()
        for cid in ids:
            g, pk, txt = gold[cid], p[cid], text[cid]
            tp_by_str, pred_types = defaultdict(set), defaultdict(set)
            for x in g & pk:
                tp_by_str[txt[x[0]:x[1]]].add(x[2])
            for x in pk:
                pred_types[txt[x[0]:x[1]]].add(x[2])
            for x in g - pk:
                s = txt[x[0]:x[1]]
                c["fn"] += 1
                c["fn_same_string_correct_elsewhere_same_type"] += x[2] in tp_by_str.get(s, ())
                c["fn_string_predicted_elsewhere_any_type"] += s in pred_types
                c["fn_string_predicted_with_2plus_types_in_chapter"] += len(pred_types.get(s, ())) >= 2
            for x in pk - g:
                s = txt[x[0]:x[1]]
                c["fp"] += 1
                others = tp_by_str.get(s, set())
                c["fp_same_string_correct_elsewhere_other_type"] += bool(others - {x[2]})
                c["fp_same_string_correct_elsewhere_same_type"] += x[2] in others
                c["fp_string_predicted_with_2plus_types_in_chapter"] += len(pred_types[s]) >= 2
        res[k] = dict(c)
        pool.update(c)
    res["pooled"] = dict(pool)
    whatif = {}
    for k, p in preds.items():
        voted = {cid: type_vote(p[cid], text[cid], scores_of[k][cid]) for cid in ids}
        prop = {cid: propagate(p[cid], text[cid]) for cid in ids}
        whatif[k] = {"base": scores(ids, gold, p)["micro"], "type_vote": scores(ids, gold, voted)["micro"],
                     "propagate_exact_strings": scores(ids, gold, prop)["micro"]}
    res["what_if_postprocessing"] = whatif
    return res


# ---------------------------------------------------------------------------------------------- 5: seeds

def seed_vote(ids, gold, preds):
    n_seeds = len(preds)
    found, votes, correct = Counter(), Counter(), Counter()
    vote2, union, inter = {}, {}, {}
    for cid in ids:
        for x in gold[cid]:
            found[sum(x in p[cid] for p in preds.values())] += 1
        allp = set.union(*(p[cid] for p in preds.values()))
        count = {x: sum(x in p[cid] for p in preds.values()) for x in allp}
        vote2[cid] = {x for x, n in count.items() if n >= 2}
        union[cid] = allp
        inter[cid] = {x for x, n in count.items() if n == n_seeds}
        for x, n in count.items():
            votes[n] += 1
            correct[n] += x in gold[cid]
    v2 = scores(ids, gold, vote2)
    return {"seeds": n_seeds,
            "gold_found_by_n_seeds": {str(n): found[n] for n in range(n_seeds + 1)},
            "distinct_predictions_by_n_seeds": {str(n): f"{correct[n]}/{votes[n]} correct" for n in sorted(votes)},
            "distinct_predictions_by_n_seeds_counts": {str(n): {"predictions": votes[n], "correct": correct[n]} for n in sorted(votes)},
            "single_seeds": {k: scores(ids, gold, p)["micro"] for k, p in preds.items()},
            "vote_2of3": v2["micro"], "vote_2of3_non_person": {t: v2[t] for t in ENTITY_TYPES},
            "union": scores(ids, gold, union)["micro"], "intersection": scores(ids, gold, inter)["micro"]}


# ---------------------------------------------------------------------------------------------- 6: extras

def number_strings(first, counts):
    """{string: number} by count (desc), ties by first place (chapter rank, start)."""
    order = sorted(counts, key=lambda s: (-counts[s], first[s]))
    return {s: i for i, s in enumerate(order, 1)}


def uncertain_status(x, regions):
    return "uncertain" if any(x[0] < u[1] and u[0] < x[1] for u in regions) else None


def extras(ids, gold, preds, labels, regions, text, index, chapter_rank, titles, top_n, worst_n):
    """regions = {chapter: [(start, end, reason)]} of the dev labels' uncertain regions."""
    keys = {}
    # 1. gold person predicted as dungeon, by string
    clu, chap, first = Counter(), defaultdict(set), {}
    for p in preds.values():
        for cid in ids:
            for x in sorted(p[cid] - gold[cid]):
                if x[2] == "dungeon" and (x[0], x[1], "person") in gold[cid]:
                    s = text[cid][x[0]:x[1]]
                    clu[s] += 1
                    chap[s].add(cid)
                    first[s] = min(first.get(s, (10 ** 9, 0)), (chapter_rank[cid], x[0]))
    number = number_strings(first, clu)
    by_string = []
    for s in sorted(clu, key=number.get):
        reasons = Counter(r for c in chap[s] for a, b, r in regions[c] if text[c][a:b] == s)
        by_string.append({"string": number[s], "chars": len(s), "errors": clu[s], "chapters": len(chap[s]),
                          "training_profile_as_dungeon": dict(index.profile(s, "dungeon")),
                          "training_exact_person_occurrences": index.profile(s, "person")["exact_same_type"],
                          "dev_uncertain_reasons": dict(reasons)})
    keys["person_dungeon_strings"] = {str(number[s]): s for s in clu}
    pd_out = {"errors": sum(clu.values()), "distinct_strings": len(clu), "by_string": by_string}
    # 2. recall by (seen status, label status)
    rec = defaultdict(lambda: defaultdict(Counter))
    for cid in ids:
        for x in gold[cid]:
            sk = index.seen_kind(text[cid][x[0]:x[1]], x[2])
            lab = "labels_have" if x in labels[cid] else "labels_uncertain" if uncertain_status(x, regions[cid]) else "labels_miss"
            rec[sk][lab]["gold"] += 1
            rec[sk][lab]["found_seed_sum"] += sum(x in p[cid] for p in preds.values())
    rec_out = {sk: {lab: {**dict(v), "pooled_recall": round(v["found_seed_sum"] / (len(preds) * v["gold"]), 4)}
                    for lab, v in rec[sk].items()} for sk in SEEN_KINDS if sk in rec}
    # 3. top FN and FP strings
    def top_strings(side):
        count, first, info = Counter(), {}, defaultdict(lambda: {"chapters": set(), "types": Counter(), "kinds": Counter(), "label": Counter()})
        for p in preds.values():
            for cid in ids:
                errs = gold[cid] - p[cid] if side == "fn" else p[cid] - gold[cid]
                for x in sorted(errs):
                    s = text[cid][x[0]:x[1]]
                    count[s] += 1
                    first[s] = min(first.get(s, (10 ** 9, 0)), (chapter_rank[cid], x[0]))
                    i = info[s]
                    i["chapters"].add(cid)
                    i["types"][x[2]] += 1
                    if side == "fn":
                        i["kinds"][fn_kind(x, p[cid], text[cid], titles)[0]] += 1
                        i["label"]["have" if x in labels[cid] else "miss"] += 1
                    else:
                        i["kinds"][fp_kind(x, gold[cid], text[cid], titles)[0]] += 1
                        i["label"]["in_labels" if x in labels[cid] else uncertain_status(x, regions[cid]) or "not"] += 1
        number = number_strings(first, count)
        rows = []
        for s in sorted(count, key=number.get)[:top_n]:
            i = info[s]
            main_type = max(sorted(i["types"]), key=lambda t: (i["types"][t], -ENTITY_TYPES.index(t)))
            rows.append({"string": number[s], "chars": len(s), "errors": count[s], "chapters": len(i["chapters"]),
                         "types": dict(i["types"]), "kinds": dict(i["kinds"]), "labels": dict(i["label"]),
                         "seen": index.seen_kind(s, main_type)})
        keys[f"{side}_strings"] = {str(number[s]): s for s in sorted(count, key=number.get)[:top_n]}
        return {"errors": sum(count.values()), "distinct_strings": len(count), "top_n": top_n, "by_string": rows}
    # 4. worst chapters
    per = Counter()
    for p in preds.values():
        for cid in ids:
            per[cid] += len(p[cid] ^ gold[cid])
    worst = sorted(per, key=lambda c: (-per[c], chapter_rank[c]))[:worst_n]
    # 5. gold touching an uncertain region
    n = f = 0
    for cid in ids:
        for x in gold[cid]:
            if uncertain_status(x, regions[cid]):
                n += 1
                f += sum(x in p[cid] for p in preds.values())
    out = {"person_dungeon_type_errors_by_string": pd_out, "recall_by_seen_and_label_status": rec_out,
           "top_fn_strings": top_strings("fn"), "top_fp_strings": top_strings("fp"),
           "worst_chapters": {"top_n": worst_n, "pooled_errors": sum(per.values()),
                              "chapters": [{"chapter": c, "pooled_errors": per[c]} for c in worst]},
           "gold_touching_uncertain": {"gold": n, "found_seed_sum": f,
                                       "pooled_recall": round(f / (len(preds) * n), 4) if n else None},
           "uncertain_reasons": dict(Counter(r for c in ids for _, _, r in regions[c]))}
    return out, keys


def exam_view(ids, gold, preds, text, index, titles):
    """Seed agreement and error kinds split by whether the string is a same-type positive in the training labels."""
    status = lambda s, t: "seen_same_type" if index.seen_kind(s, t) == "seen_same_type" else "not_seen_same_type"
    found = defaultdict(Counter)
    pv = defaultdict(lambda: defaultdict(Counter))
    kinds = defaultdict(lambda: defaultdict(Counter))
    for cid in ids:
        for x in gold[cid]:
            found[status(text[cid][x[0]:x[1]], x[2])][sum(x in p[cid] for p in preds.values())] += 1
        for x in set.union(*(p[cid] for p in preds.values())):
            sk = status(text[cid][x[0]:x[1]], x[2])
            n = sum(x in p[cid] for p in preds.values())
            pv[sk][n]["predictions"] += 1
            pv[sk][n]["correct"] += x in gold[cid]
        for p in preds.values():
            for x in gold[cid] - p[cid]:
                kinds["fn"][index.seen_kind(text[cid][x[0]:x[1]], x[2])][fn_kind(x, p[cid], text[cid], titles)[0]] += 1
            for x in p[cid] - gold[cid]:
                kinds["fp"][index.seen_kind(text[cid][x[0]:x[1]], x[2])][fp_kind(x, gold[cid], text[cid], titles)[0]] += 1
    return {"gold_found_by_n_seeds_by_string_status": {sk: {str(n): found[sk][n] for n in range(len(preds) + 1)} for sk in sorted(found)},
            "distinct_predictions_by_string_status_and_n_seeds": {sk: {str(n): dict(v[n]) for n in sorted(v)} for sk, v in sorted(pv.items())},
            "error_kinds_by_seen_status": {side: {sk: dict(kinds[side][sk]) for sk in SEEN_KINDS if sk in kinds[side]} for side in ("fn", "fp")}}


# ---------------------------------------------------------------------------------------------- loading (all checked)

def read_json(path, what):
    if not path.is_file():
        raise SystemExit(f"{what}: {path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def load_answers(root, gold_file, snapshot, files, what):
    """(export, answered chapter ids, scored gold spans {chapter: set}, unscored {chapter: [(s, e)]}); gold check passed."""
    path = root / gold_file
    export = read_json(path, what)
    problems, _, _ = check_gold(export, snapshot)
    if problems:
        raise SystemExit(f"{what}: the answers do not pass the gold check ({len(problems)} problems)")
    ids = done_chapters(export)
    if not ids:
        raise SystemExit(f"{what}: no answered chapter")
    gold = gold_spans(export, ids)
    scored = {c: {s for s in gold[c]["spans"] if not _touches(s, gold[c]["unscored"])} for c in ids}
    files[gold_file] = sha256(path)
    return export, ids, scored, {c: gold[c]["unscored"] for c in ids}


def load_predictions(root, file, ids, unscored, chapters, files, what):
    """(scored predictions {chapter: set}, {chapter: {(start, end, type): score}}); the rows must cover exactly the
    answered chapters and every row's text must be the chapter text at its place."""
    path = root / file
    if not path.is_file():
        raise SystemExit(f"{what}: {file} does not exist")
    rows = read_jsonl(path)
    if {r["chapter_id"] for r in rows} != set(ids):
        raise SystemExit(f"{what}: {file} does not cover exactly the answered chapters")
    for r in rows:
        if chapters[r["chapter_id"]].text[r["start"]:r["end"]] != r["text"]:
            raise SystemExit(f"{what}: a row of {file} is not the chapter text at its place")
    ap = as_prediction(rows, ids)
    scored = {c: {s for s in ap[c]["spans"] if not _touches(s, unscored[c])} for c in ids}
    score = {c: {} for c in ids}
    for r in rows:
        score[r["chapter_id"]][(r["start"], r["end"], r["type"])] = r["score"]
    files[file] = sha256(path)
    return scored, score


def check_run_record(run_dir, what):
    rec, met = read_json(run_dir / "config.json", what), read_json(run_dir / "metrics.json", what)
    if not (rec.get("run_id") == met.get("run_id") == run_dir.name):
        raise SystemExit(f"{what}: config.json and metrics.json do not both name the folder {run_dir.name}")
    return rec, met


def check_predictions_sha(root, file, met, what, files):
    key = SHA_KEY.get(Path(file).name)
    if key is None or key not in met:
        raise SystemExit(f"{what}: the run's metrics.json records no sha256 for {Path(file).name}")
    if sha256(root / file) != met[key]:
        raise SystemExit(f"{what}: {file} does not have the sha256 its run recorded ({key})")
    files[file] = met[key]


def check_recorded_micro(got, rec, what):
    """This run's strict micro of a seed (tp, fp, fn, precision, recall, f1) must equal the one a run recorded; rec may
    hold the counts and the scores flat or the scores under "micro" (a training run's dev.at_best)."""
    flat = {**rec, **(rec.get("micro") or {})}
    for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
        if key not in flat:
            raise SystemExit(f"{what}: the record has no {key} to check this run's micro against")
        if flat[key] != got[key]:
            raise SystemExit(f"{what}: this run's micro {key} {got[key]} is not the recorded {flat[key]}")


def check_chunk_record(stats, chunks, what):
    """The 4_context chunks are the tagger's only if the run saw the same number of chunks and split none into windows."""
    stats = stats or {}
    if stats.get("chunks") != chunks:
        raise SystemExit(f"{what}: the run recorded {stats.get('chunks')} chunks, this run makes {chunks}")
    if stats.get("windows") != chunks or stats.get("chunks_over_max_tokens") != 0:
        raise SystemExit(f"{what}: the run split chunks into token windows (windows {stats.get('windows')}, over max_tokens "
                         f"{stats.get('chunks_over_max_tokens')}); 4_context's chunks would not be the tagger's")


def scoring_run_system(emet, predict_run, predictions_sha256, what):
    """The system of the scoring run whose predictions_run is this predict run, with the same predictions sha256; returns
    its recorded micro (tp, fp, fn from the per-type counts added, P, R, F1 from scores.micro)."""
    names = [n for n, s in (emet.get("systems") or {}).items() if s.get("predictions_run") == predict_run]
    if len(names) != 1:
        raise SystemExit(f"{what}: the scoring run records {len(names)} systems with predictions_run {predict_run}, need one")
    sysrec = emet["systems"][names[0]]
    if sysrec.get("predictions_sha256") != predictions_sha256:
        raise SystemExit(f"{what}: the scoring run scored other predictions of {predict_run} (predictions sha256 differ)")
    sc = (emet.get("scores") or {}).get(names[0]) or {}
    counts = {key: sum((t.get(key) or 0) for t in (sc.get("per_type") or {}).values()) for key in ("tp", "fp", "fn")}
    return names[0], {**counts, **(sc.get("micro") or {})}


def check_config(cfg):
    for part in ("dev", "test"):
        if len(cfg[part]["predictions"]) < 2:
            raise SystemExit(f"{part}: at least two seeds' prediction files are needed")
    if len(cfg["dev"]["predictions"]) != len(cfg["test"]["predictions"]):
        raise SystemExit("dev and test must list the same seeds' runs, in the same order")
    rules = cfg.get("rules") or {}
    missing = [a for a in (*DEV_ANALYSES, *EXAM_ANALYSES, "exam_secondary") if a not in rules]
    if missing:
        raise SystemExit(f"rules: no rule in words for {missing}")
    for b in cfg["edge_buckets"]:
        if b.get("max") is not None and b["max"] <= b["min"]:
            raise SystemExit(f"edge_buckets: {b['label']} has max <= min")
    th = cfg["training_profile_thresholds"]
    if not 0 < th["same_type_low"] < th["same_type_high"] <= 1:
        raise SystemExit("training_profile_thresholds: need 0 < same_type_low < same_type_high <= 1")


def analyse(cfg, root=None):
    """Everything, read and checked; returns what the outputs are made of."""
    root = root or REPO_ROOT
    check_config(cfg)
    files, code_of_runs = {}, {}
    snapshot = load_snapshot(cfg["snapshot"], root=root / "data" / "snapshots")
    chapters = snapshot.chapters

    # the weak build and its labels
    wb = cfg["weak_build"]
    build_dir = root / wb["run"]
    brec = read_json(build_dir / "config.json", f"weak build {wb['run']}")
    if brec.get("run_id") != build_dir.name or (brec.get("snapshot") or {}).get("id") != snapshot.id:
        raise SystemExit(f"weak build {wb['run']}: its record names another run or snapshot")
    labels_rec = (brec.get("labels") or {}).get(wb["setting"])
    if not labels_rec or labels_rec.get("file") != wb["labels"]:
        raise SystemExit(f"weak build {wb['run']}: setting {wb['setting']} is not recorded with the labels file {wb['labels']}")
    labels_sha = sha256(root / wb["labels"])
    if labels_sha != labels_rec["sha256"]:
        raise SystemExit(f"{wb['labels']} does not have the sha256 the weak build recorded")
    files[f"{wb['run']}/config.json"] = sha256(build_dir / "config.json")
    files[wb["labels"]] = labels_sha
    weak_cfg = read_json(root / cfg["title_suffixes_from"], "title_suffixes_from")
    titles = list(weak_cfg["title_suffixes"])
    if titles != list((brec.get("config") or {}).get("title_suffixes") or []):
        raise SystemExit("the title suffixes are not the ones the weak build was made with")
    files[cfg["title_suffixes_from"]] = sha256(root / cfg["title_suffixes_from"])
    rows = read_jsonl(root / wb["labels"])
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = {r["chapter_id"]: r for r in rows if r["split"] == "dev"}
    index = TrainingIndex(train_rows, chapters, cfg["training_profile_thresholds"])

    # answers
    gold_dev_file = cfg["dev"]["gold"]
    if (brec.get("gold_dev") or {}).get("file") != gold_dev_file:
        raise SystemExit("the development answers are not the file the weak build scored its labels on")
    dev_export, dev_ids, G, dev_unscored = load_answers(root, gold_dev_file, snapshot, files, "development answers")
    if files[gold_dev_file] != brec["gold_dev"]["sha256"]:
        raise SystemExit("the development answers do not have the sha256 the weak build recorded")
    if set(dev_ids) - set(dev_rows):
        raise SystemExit("an answered development chapter has no dev row in the labels")
    eval_dir = (root / cfg["test"]["gold_recorded_in"]).parent
    test_rec, eval_met = check_run_record(eval_dir, f"scoring run {eval_dir.name} (test.gold_recorded_in)")
    if (test_rec.get("gold_test") or {}).get("file") != cfg["test"]["gold"]:
        raise SystemExit("test.gold_recorded_in does not record the test answers file")
    files[cfg["test"]["gold_recorded_in"]] = sha256(eval_dir / "config.json")
    files[f"{Path(cfg['test']['gold_recorded_in']).parent.as_posix()}/metrics.json"] = sha256(eval_dir / "metrics.json")
    test_export, test_ids, TG, test_unscored = load_answers(root, cfg["test"]["gold"], snapshot, files, "test answers")
    if files[cfg["test"]["gold"]] != test_rec["gold_test"]["sha256"]:
        raise SystemExit("the test answers do not have the sha256 the scoring run recorded")
    train_ids = {r["chapter_id"] for r in train_rows}
    if train_ids & (set(dev_ids) | set(test_ids)):
        raise SystemExit("a training row of the labels is an answered development or test chapter")

    # the training runs and their test predictions
    P, S, TP, TS = {}, {}, {}, {}
    seeds, records, scored_as = [], {}, {}
    for dev_file, test_file in zip(cfg["dev"]["predictions"], cfg["test"]["predictions"]):
        run_dir = (root / dev_file).parent
        what = f"training run {run_dir.name}"
        rec, met = check_run_record(run_dir, what)
        if (rec.get("snapshot") or {}).get("id") != snapshot.id:
            raise SystemExit(f"{what}: its snapshot is not {snapshot.id}")
        wl = rec.get("weak_labels") or {}
        if wl.get("build") != build_dir.name or rec.get("setting") != wb["setting"] or wl.get("sha256") != labels_sha:
            raise SystemExit(f"{what}: it was not trained on {build_dir.name} setting {wb['setting']} with these labels")
        if (rec.get("gold_dev") or {}).get("sha256") != files[gold_dev_file]:
            raise SystemExit(f"{what}: its development answers are not this file")
        if rec["config"]["max_chars"] != cfg["chunk_max_chars"]:
            raise SystemExit(f"{what}: its max_chars is not chunk_max_chars")
        seed = rec["seed"]
        if seed in seeds:
            raise SystemExit(f"{what}: seed {seed} is listed twice")
        seeds.append(seed)
        k = f"seed{seed}"
        check_predictions_sha(root, dev_file, met, what, files)
        files[f"{run_dir.name}/config.json"] = sha256(run_dir / "config.json")
        code_of_runs[run_dir.name] = rec.get("code")
        P[k], S[k] = load_predictions(root, dev_file, dev_ids, dev_unscored, chapters, files, what)
        tdir = (root / test_file).parent
        twhat = f"predict run {tdir.name}"
        trec, tmet = check_run_record(tdir, twhat)
        if trec.get("model_run") != run_dir.name or trec.get("model_run_config_sha256") != files[f"{run_dir.name}/config.json"]:
            raise SystemExit(f"{twhat}: it is not the test prediction of {run_dir.name} (model_run and its config.json sha256)")
        if (trec.get("setting"), trec.get("seed")) != (wb["setting"], seed) or (trec.get("snapshot") or {}).get("id") != snapshot.id:
            raise SystemExit(f"{twhat}: setting, seed or snapshot differ from the training run's")
        check_predictions_sha(root, test_file, tmet, twhat, files)
        code_of_runs[tdir.name] = trec.get("code")
        TP[k], TS[k] = load_predictions(root, test_file, test_ids, test_unscored, chapters, files, twhat)
        scored_as[k], test_micro_rec = scoring_run_system(eval_met, f"runs/{tdir.name}", tmet["predictions_sha256"], twhat)
        records[k] = {"dev_micro": (met.get("dev") or {}).get("at_best") or {}, "test_micro": test_micro_rec,
                      "dev_stats": (met.get("dev") or {}).get("predict_stats_at_best"), "test_stats": tmet.get("predict_stats"),
                      "what": what, "twhat": twhat}

    # the dev labels, scored like predictions
    L = {c: {(x["start"], x["end"], x["type"]) for x in dev_rows[c]["spans"] if not _touches((x["start"], x["end"]), dev_unscored[c])}
         for c in dev_ids}
    U = {c: [(x["start"], x["end"]) for x in dev_rows[c]["uncertain"]] for c in dev_ids}
    regions = {c: [(x["start"], x["end"], x.get("reason")) for x in dev_rows[c]["uncertain"]] for c in dev_ids}
    text = {c: chapters[c].text for c in (*dev_ids, *test_ids)}
    buckets, max_chars = cfg["edge_buckets"], cfg["chunk_max_chars"]
    ex = cfg["extras"]

    dev = {"chapters": len(dev_ids), "scored_gold": sum(len(G[c]) for c in dev_ids), "seeds": list(P)}
    dev["1_ceiling"] = ceiling(dev_ids, G, P, L)
    dev["2_agreement_with_labels"] = agreement(dev_ids, G, P, L, U)
    dev["3_error_kinds"] = error_kinds(dev_ids, G, P, text, index, titles)
    dev["4_context"] = context(dev_ids, G, P, text, chapters, max_chars, buckets, L)
    dev["4b_consistency"] = consistency(dev_ids, G, P, text, S)
    dev["5_seeds"] = seed_vote(dev_ids, G, P)
    dev["6_extras"], keys = extras(dev_ids, G, P, L, regions, text, index, {c: i for i, c in enumerate(dev_ids)}, titles,
                                   ex["top_strings"], ex["worst_chapters"])

    exam = {"chapters": len(test_ids), "scored_gold": sum(len(TG[c]) for c in test_ids), "seeds": list(TP), "SECONDARY": SECONDARY}
    exam["micro_check"] = {k: scores(test_ids, TG, p)["micro"] for k, p in TP.items()}
    exam["3_error_kinds"] = error_kinds(test_ids, TG, TP, text, index, titles)
    exam["4_context"] = context(test_ids, TG, TP, text, chapters, max_chars, buckets)
    exam["4b_consistency"] = consistency(test_ids, TG, TP, text, TS)
    exam["5_seeds"] = seed_vote(test_ids, TG, TP)
    exam["6_extras_exam_view"] = exam_view(test_ids, TG, TP, text, index, titles)

    # the scores and chunks of this run must be the ones the training, predict and scoring runs recorded
    for k, r in records.items():
        check_recorded_micro(dev["1_ceiling"]["per_type"]["micro"][k], r["dev_micro"], f"{r['what']}: development chapters")
        check_recorded_micro(exam["micro_check"][k], r["test_micro"], f"{r['twhat']}: test chapters, scoring run {eval_dir.name}")
        check_chunk_record(r["dev_stats"], dev["4_context"]["chunks"]["chunks"], f"{r['what']}: development chapters")
        check_chunk_record(r["test_stats"], exam["4_context"]["chunks"]["chunks"], f"{r['twhat']}: test chapters")
    exam["micro_check_equals_scoring_run"] = {"run": eval_dir.name, "systems": scored_as}

    build_info = {"run": build_dir.name, "setting": wb["setting"], "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                  "max_train_chapter": (brec.get("config") or {}).get("max_train_chapter"),
                  "train_ids_sha256": (brec.get("split") or {}).get("train_ids_sha256"),
                  "title_suffixes": len(titles), "distinct_positive_strings": len(index.pos_types)}
    return {"files": files, "code_of_runs": code_of_runs, "seeds": seeds, "weak_build": build_info,
            "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
            "dev": dev, "exam_secondary": exam, "keys": keys}


# ---------------------------------------------------------------------------------------------- outputs

def build_metrics(run_id, cfg, a):
    m = {"run_id": run_id, "note": cfg["description"], "snapshot": a["snapshot"]["id"], "weak_build": a["weak_build"],
         "seeds": a["seeds"], "dev": a["dev"], "exam_secondary": a["exam_secondary"],
         "checks": {"weak_build_record_labels_sha256_and_title_suffixes": True,
                    "answers_pass_the_gold_check_and_have_the_recorded_sha256": True,
                    "training_runs_name_this_build_setting_labels_and_answers": True,
                    "predict_runs_name_their_training_run_and_its_config_sha256": True,
                    "prediction_files_have_the_recorded_sha256_and_cover_the_answered_chapters": True,
                    "no_training_row_is_an_answered_chapter": True,
                    "each_seed_dev_micro_equals_its_training_run_record": True,
                    "each_seed_test_micro_equals_the_scoring_run_record_of_the_same_predictions": True,
                    "chunk_counts_equal_the_runs_records_and_no_chunk_was_split_into_windows": True,
                    "input_files_hashed": len(a["files"])},
         "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    check_counts_only(m)
    return m


def build_config(run_id, config_file, cfg, a, started, code):
    out = {"run_id": run_id, "config_file": config_file, "config": cfg, "snapshot": a["snapshot"],
           "inputs_sha256": a["files"], "runs_code": a["code_of_runs"], "code": code, "started_at": started}
    check_counts_only(out)
    return out


def outputs(run_id, out_dir=None):
    if out_dir is not None:
        d = Path(out_dir)
        return d, d / "keys.json"
    return REPO_ROOT / "runs" / run_id, REPO_ROOT / "data" / "analysis" / run_id / "keys.json"


def refuse_existing(run_id, out_dir=None):
    run_dir, keys = outputs(run_id, out_dir)
    taken = [p for p in (run_dir / "config.json", run_dir / "metrics.json", keys) if p.exists()]
    if taken:
        raise SystemExit(f"{taken[0]} already exists; use a new run id or directory (outputs are never overwritten)")


def write_outputs(run_id, out_dir, config, metrics, keys):
    refuse_existing(run_id, out_dir)
    run_dir, keys_file = outputs(run_id, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    keys_file.write_text(json.dumps({"run_id": run_id, "note": "not committed: holds strings from the book", **keys},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    return run_dir, keys_file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    ap.add_argument("--out", help="scratch run: write config.json, metrics.json and keys.json to this directory "
                                  "(default: runs/<run_id>/ and data/analysis/<run_id>/, from committed code only)")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    refuse_existing(run_id, args.out)
    started, code = time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_state()
    if args.out is None and code.get("dirty") is not False:
        raise SystemExit("src/ or configs/ has uncommitted changes; a recorded run is made from committed code (use --out for a scratch run)")
    a = analyse(cfg)
    metrics = build_metrics(run_id, cfg, a)
    config = build_config(run_id, Path(args.config).as_posix(), cfg, a, started, code)
    d = a["dev"]
    c1 = d["1_ceiling"]["per_type"]["micro"]
    print(f"dev {d['chapters']} chapters, {d['scored_gold']} scored gold spans; labels F1 {c1['labels']['f1']}, pooled model F1 "
          f"{c1['pooled']['f1']} (P {c1['pooled']['precision']}, R {c1['pooled']['recall']})")
    ag = d["2_agreement_with_labels"]["pooled"]
    print(f"2 errors {ag['errors']}, shared with the labels {ag['errors_shared_with_labels']} ({ag['share_shared']}); FP in labels "
          f"{ag['fp_share_in_labels']}, FN the labels miss {ag['fn_share_labels_miss']}; per-chapter error correlation pearson "
          f"{d['2_agreement_with_labels']['per_chapter_error_correlation']['pearson']}")
    ek = d["3_error_kinds"]["pooled"]
    print(f"3 fp kinds {ek['fp_kind']} fn kinds {ek['fn_kind']}")
    for name, _ in EDGE_VIEWS:
        print(f"4 {name}: " + " | ".join(f"{b} fn {r['fn_rate_pooled']} fp {r['fp_rate_pooled']}" for b, r in d["4_context"][name].items()))
    for k, v in d["4b_consistency"]["what_if_postprocessing"].items():
        print(f"4b {k}: base F1 {v['base']['f1']} type vote {v['type_vote']['f1']} propagate {v['propagate_exact_strings']['f1']}")
    s5 = d["5_seeds"]
    print(f"5 gold found by n seeds {s5['gold_found_by_n_seeds']}; vote 2of3 F1 {s5['vote_2of3']['f1']}, union {s5['union']['f1']}, "
          f"intersection {s5['intersection']['f1']}")
    e6 = d["6_extras"]
    print(f"6 person>dungeon errors {e6['person_dungeon_type_errors_by_string']['errors']} on "
          f"{e6['person_dungeon_type_errors_by_string']['distinct_strings']} strings; gold touching uncertain "
          f"{e6['gold_touching_uncertain']['gold']} recall {e6['gold_touching_uncertain']['pooled_recall']}")
    x = a["exam_secondary"]
    print(f"exam (secondary) {x['chapters']} chapters: " + " ".join(f"{k} F1 {v['f1']}" for k, v in x["micro_check"].items()))
    run_dir, keys_file = write_outputs(run_id, args.out, config, metrics, a["keys"])
    print(f"wrote {run_dir} and {keys_file}")


if __name__ == "__main__":
    main()
