"""Teacher-label self-consistency (round 3, configs/consistency_v1.json): the rules K7, K6, K5 and K8 applied to a
teacher labels build's T rows, reading only the training chapters' text and the teacher's own label rows. The W rows are
copied unchanged, and the new build keeps the source record's shape so tagger.train, evaluate.span_system and
posthoc.training_chapters accept it exactly like the source (teacher.build_train_ids on the new record must give the
source's chapters; checked on the source record before anything is written and on the new record after). Only run_id,
labels and code (this run's git state; the source's under derived_from.source_code) differ from the source record.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.consistency configs/consistency_v1.json [--only <target build id>] [--out <dir>]

Without --out a run writes data/weak_labels/<target>/ (labels_T.jsonl, labels_W.jsonl), runs/<target>/ (config.json and
metrics.json, counts only) and data/analysis/<target>/keys.json (the affected strings; not committed), and needs
committed code (src/ and configs/ clean) like posthoc. With --out <dir> everything goes under <dir> instead (a scratch
run; the record says so and the code may be dirty). Builds are never overwritten.

The rules, in the config's order (the config's rule texts are binding; READING below says how they are read here):
- K7_boundary: for each positive span, the candidate longer strings are the positive texts of the same type anywhere in
  the build (all T rows of the source) that contain the span text as a strict substring and occur in the chapter text at
  a position covering the span; the longest is taken, ties by the earlier start; the span is extended (type kept,
  sources + K7_boundary). When the extended span would overlap another span or an uncertain region, the extended span
  and everything it overlaps become one uncertain region over their union, reason K7_overlap, types = every type
  involved (a region of unknown type, types null, makes the union unknown). Counted per (shorter, longer) pair.
- K6_type_conflict: over the whole build (after K7), a text with two or more types among the positives: every type
  whose share of the text's positive occurrences is at most minority_share becomes uncertain at its occurrences, with
  types = all types the text has among the positives, reason K6_type_conflict.
- K5_pending_majority: pending occurrences of a text = the source rows' teacher_pending regions with that text (before
  K7/K6); positive occurrences = the positive spans with that text after K7/K6. When pending >= min_pending and
  pending / (pending + positives) >= pending_share, every positive occurrence becomes uncertain with types = the types
  the text has among those positives, reason K5_pending_majority.
- K8_same_chapter_partial: per chapter, the distinct positive texts of at least min_len characters (after K5) are looked
  up in the chapter text; the occurrences are taken in one left-to-right pass (the longest at a start first), each one
  that overlaps no span, no uncertain region and no occurrence already taken becomes an uncertain region with types =
  the types the text has among the chapter's positives, reason K8_same_chapter_partial.
Shares are compared exactly (fractions). After the rules the spans and regions are sorted, every text is checked
against the chapter text and weak.char_allowed must accept the row (no overlaps).

Material: the snapshot's text of the build's own training chapters, runs/<source>/config.json and the two label files of
the source build (their sha256 checked against the record). No answer file, no name sheet, no dev2, test or test2
chapter and no system output is read; tests/test_consistency.py checks the source of this module for that.
"""
import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import sha256, write_jsonl
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.teacher import TEACHER_BUILD_KIND, build_train_ids
from novel_ner.weak import char_allowed, ids_sha256
from novel_ner.zeroshot import git_state

RULES = ("K7_boundary", "K6_type_conflict", "K5_pending_majority", "K8_same_chapter_partial")
PENDING = "teacher_pending"
K7_OVERLAP = "K7_overlap"
KIND = "teacher labels consistency build (novel_ner.consistency)"
READING = {
    "K7_boundary": "candidates = positive texts of the same type in the whole source build that contain the span text as a "
                   "strict substring (span text at least min_len_shorter characters) and occur in the chapter text at a start st "
                   "with st <= start and end <= st + len(longer); the longest wins, ties the earlier st; the extended span keeps "
                   "its type and gets K7_boundary in its sources. An extended span overlapping other spans or regions becomes, "
                   "with all of them, one region over their union, reason K7_overlap, types = every type involved; a member "
                   "region of unknown type (types null) makes the union unknown (types null). Spans are visited in their source "
                   "order; a span already merged into a region is not extended",
    "K6_type_conflict": "occurrences per (text, type) over all positive spans after K7; for a text with two or more types, a "
                        "type with count / total <= minority_share (exact fraction) becomes uncertain at every occurrence, "
                        "types = sorted types of the text among the positives (the minority type included)",
    "K5_pending_majority": "pending = source regions with reason teacher_pending and that text; positives = spans with that "
                           "text after K7 and K6; when pending >= min_pending and pending / (pending + positives) >= pending_share "
                           "(exact fraction) and positives >= 1, every such span becomes uncertain, types = sorted types of the "
                           "text among those positives",
    "K8_same_chapter_partial": "per chapter: every occurrence of every distinct positive text of at least min_len characters "
                               "(after K5) in the chapter text, sorted by start then longest first, one left-to-right pass: an "
                               "occurrence overlapping a span, a region or an occurrence already taken is skipped, the others "
                               "become regions with types = sorted types of the text among the chapter's positives",
    "after": "spans and regions sorted by start; every text equals the chapter text at its offsets; weak.char_allowed accepts "
             "the row; the chapter ids and their order are the source's",
}


def _overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _type_counts(spans):
    return {t: sum(1 for s in spans if s["type"] == t) for t in ENTITY_TYPES}


def summary(rows):
    """Counts of a set of rows: spans by type, uncertain regions by reason and by their types."""
    spans = [s for r in rows for s in r["spans"]]
    unc = [u for r in rows for u in r["uncertain"]]
    return {"chapters": len(rows), "spans": len(spans), "spans_by_type": _type_counts(spans),
            "uncertain": len(unc), "uncertain_by_reason": dict(sorted(Counter(u["reason"] for u in unc).items())),
            "uncertain_by_types": dict(sorted(Counter("unknown" if u["types"] is None else "+".join(u["types"]) for u in unc).items()))}


# ---------------------------------------------------------------------------------------------- K7


def longer_strings_by_type(rows, min_len_shorter):
    """{type: {text: [longer positive texts of that type containing text as a strict substring, longest first, then
    by string]}} over all positive spans of rows; texts shorter than min_len_shorter get no candidates."""
    texts = defaultdict(set)
    for r in rows:
        for s in r["spans"]:
            texts[s["type"]].add(s["text"])
    out = {}
    for t, ts in texts.items():
        by_len = sorted(ts, key=lambda x: (-len(x), x))
        out[t] = {short: [L for L in by_len if len(L) > len(short) and short in L]
                  for short in ts if len(short) >= min_len_shorter}
    return out


def k7_extension(text, span, longer):
    """(start, end, longer string) for the longest string of `longer` (sorted longest first) that occurs in text at a
    position covering the span, ties by the earlier start; None when none does."""
    best = None
    for L in longer:
        n = len(L)
        if best is not None and n < len(best[2]):
            break
        lo, hi = max(0, span["end"] - n), min(span["start"], len(text) - n)
        for st in range(lo, hi + 1):
            if text[st:st + n] == L:
                if best is None or (n, -st) > (len(best[2]), -best[0]):
                    best = (st, st + n, L)
                break
    return best


def apply_k7(work, longer_by_type, counts, pairs):
    """K7 on one chapter's work ({text, spans, uncertain}); counts and pairs ({(shorter, longer, type): n}) updated."""
    text = work["text"]
    touched = False
    for span in list(work["spans"]):
        if not any(x is span for x in work["spans"]):
            continue  # merged into a K7_overlap region by an earlier extension
        ext = k7_extension(text, span, longer_by_type.get(span["type"], {}).get(span["text"], ()))
        if ext is None:
            continue
        st, en, L = ext
        new = {"start": st, "end": en, "type": span["type"], "text": L, "sources": list(span["sources"]) + ["K7_boundary"]}
        touched = True
        counts["spans_extended"] += 1
        counts["spans_extended_by_type"][span["type"]] += 1
        counts["characters_added"] += (en - st) - (span["end"] - span["start"])
        pairs[(span["text"], L, span["type"])] += 1
        hit_spans = [x for x in work["spans"] if x is not span and _overlaps(new, x)]
        hit_regions = [x for x in work["uncertain"] if _overlaps(new, x)]
        if not hit_spans and not hit_regions:
            work["spans"][next(i for i, x in enumerate(work["spans"]) if x is span)] = new
            counts["spans_extended_in_place"] += 1
            continue
        members = [new] + hit_spans + hit_regions
        start, end = min(m["start"] for m in members), max(m["end"] for m in members)
        types, unknown = {new["type"]} | {x["type"] for x in hit_spans}, False
        for x in hit_regions:
            if x["types"] is None:
                unknown = True
            else:
                types |= set(x["types"])
        sources = list(dict.fromkeys(s for m in members for s in (m.get("sources") or [])))  # in order, once each
        region = {"start": start, "end": end, "text": text[start:end], "types": None if unknown else sorted(types),
                  "reason": K7_OVERLAP, "sources": sources + [K7_OVERLAP]}
        work["spans"] = [x for x in work["spans"] if x is not span and not any(x is h for h in hit_spans)]
        work["uncertain"] = [x for x in work["uncertain"] if not any(x is h for h in hit_regions)] + [region]
        counts["extensions_overlapping"] += 1
        counts["spans_made_uncertain"] += 1 + len(hit_spans)
        counts["spans_made_uncertain_by_type"][span["type"]] += 1
        for x in hit_spans:
            counts["spans_made_uncertain_by_type"][x["type"]] += 1
        counts["regions_absorbed"] += len(hit_regions)
        for x in hit_regions:
            counts["regions_absorbed_by_reason"][x["reason"]] += 1
    if touched:
        counts["chapters_touched"] += 1


# ---------------------------------------------------------------------------------------------- K6


def apply_k6(works, minority_share, counts, keys):
    """K6 over the whole build; keys gets one entry per affected text."""
    occ = defaultdict(Counter)
    for w in works:
        for s in w["spans"]:
            occ[s["text"]][s["type"]] += 1
    share = Fraction(str(minority_share))
    minority = {}
    for text, c in occ.items():
        if len(c) < 2:
            continue
        counts["strings_with_two_or_more_types"] += 1
        counts["spans_of_strings_with_two_or_more_types"] += sum(c.values())
        total = sum(c.values())
        mins = {t for t, n in c.items() if Fraction(n, total) <= share}
        if mins:
            minority[text] = mins
            keys.append({"text": text, "counts_by_type": dict(sorted(c.items())), "minority_types": sorted(mins),
                         "spans_made_uncertain": sum(c[t] for t in mins)})
        else:
            counts["strings_kept_as_a_split"] += 1
    counts["strings_affected"] = len(minority)
    for w in works:
        kept, before = [], counts["spans_made_uncertain"]
        for s in w["spans"]:
            mins = minority.get(s["text"])
            if mins and s["type"] in mins:
                w["uncertain"].append({"start": s["start"], "end": s["end"], "text": s["text"], "types": sorted(occ[s["text"]]),
                                       "reason": "K6_type_conflict", "sources": list(s["sources"]) + ["K6_type_conflict"]})
                counts["spans_made_uncertain"] += 1
                counts["spans_made_uncertain_by_type"][s["type"]] += 1
                counts["regions_added"] += 1
            else:
                kept.append(s)
        w["spans"] = kept
        counts["chapters_touched"] += counts["spans_made_uncertain"] > before
    keys.sort(key=lambda k: (-k["spans_made_uncertain"], k["text"]))


# ---------------------------------------------------------------------------------------------- K5


def apply_k5(works, source_rows, pending_share, min_pending, counts, keys):
    """K5 over the whole build; the pending counts come from the source rows."""
    pending = Counter(u["text"] for r in source_rows for u in r["uncertain"] if u["reason"] == PENDING)
    positives = defaultdict(Counter)
    for w in works:
        for s in w["spans"]:
            positives[s["text"]][s["type"]] += 1
    share = Fraction(str(pending_share))
    hit = {}
    counts["strings_with_pending"] = len(pending)
    for text, p in pending.items():
        n = sum(positives[text].values()) if text in positives else 0
        if n:
            counts["strings_with_pending_and_positives"] += 1
        if n and p >= min_pending and Fraction(p, p + n) >= share:
            hit[text] = sorted(positives[text])
            keys.append({"text": text, "pending": p, "positives": n, "pending_share": round(p / (p + n), 4), "types": hit[text]})
    counts["strings_affected"] = len(hit)
    for w in works:
        kept, before = [], counts["spans_made_uncertain"]
        for s in w["spans"]:
            types = hit.get(s["text"])
            if types:
                w["uncertain"].append({"start": s["start"], "end": s["end"], "text": s["text"], "types": types,
                                       "reason": "K5_pending_majority", "sources": list(s["sources"]) + ["K5_pending_majority"]})
                counts["spans_made_uncertain"] += 1
                counts["spans_made_uncertain_by_type"][s["type"]] += 1
                counts["regions_added"] += 1
            else:
                kept.append(s)
        w["spans"] = kept
        counts["chapters_touched"] += counts["spans_made_uncertain"] > before
    keys.sort(key=lambda k: (-k["positives"], k["text"]))


# ---------------------------------------------------------------------------------------------- K8


def apply_k8(work, min_len, counts, strings):
    """K8 on one chapter; strings ({(text, types key): n}) updated."""
    text = work["text"]
    types_of = defaultdict(set)
    for s in work["spans"]:
        if len(s["text"]) >= min_len:
            types_of[s["text"]].add(s["type"])
    if not types_of:
        return
    taken = bytearray(len(text))
    for x in work["spans"] + work["uncertain"]:
        taken[x["start"]:x["end"]] = b"\x01" * (x["end"] - x["start"])
    cands = []
    for s in types_of:
        i = text.find(s)
        while i >= 0:
            cands.append((i, i + len(s), s))
            i = text.find(s, i + 1)
    cands.sort(key=lambda c: (c[0], c[0] - c[1], c[2]))
    added = 0
    for st, en, s in cands:
        if any(taken[st:en]):
            counts["occurrences_skipped_covered"] += 1
            continue
        taken[st:en] = b"\x01" * (en - st)
        types = sorted(types_of[s])
        work["uncertain"].append({"start": st, "end": en, "text": s, "types": types, "reason": "K8_same_chapter_partial",
                                  "sources": ["K8_same_chapter_partial"]})
        counts["regions_added"] += 1
        counts["regions_added_by_types"]["+".join(types)] += 1
        strings[(s, "+".join(types))] += 1
        added += 1
    if added:
        counts["chapters_touched"] += 1


# ---------------------------------------------------------------------------------------------- the transformation


SUB_COUNTERS = {"K7_boundary": ("spans_extended_by_type", "spans_made_uncertain_by_type", "regions_absorbed_by_reason"),
                "K6_type_conflict": ("spans_made_uncertain_by_type",), "K5_pending_majority": ("spans_made_uncertain_by_type",),
                "K8_same_chapter_partial": ("regions_added_by_types",)}
COUNTS = {"K7_boundary": ("spans_extended", "spans_extended_in_place", "extensions_overlapping", "spans_made_uncertain", "regions_absorbed",
                          "characters_added", "distinct_pairs", "distinct_shorter_strings", "distinct_longer_strings",
                          "overlap_regions_after_the_rule", "chapters_touched"),
          "K6_type_conflict": ("strings_with_two_or_more_types", "spans_of_strings_with_two_or_more_types", "strings_kept_as_a_split",
                               "strings_affected", "spans_made_uncertain", "regions_added", "chapters_touched"),
          "K5_pending_majority": ("strings_with_pending", "strings_with_pending_and_positives", "strings_affected", "spans_made_uncertain",
                                  "regions_added", "chapters_touched"),
          "K8_same_chapter_partial": ("regions_added", "distinct_strings", "occurrences_skipped_covered", "chapters_touched")}


def _plain(counts):
    """A rule's counts (a Counter of ints and of Counters) as plain sorted dicts; a *_by_type dict lists the six types."""
    out = {}
    for k in sorted(counts):
        v = counts[k]
        if isinstance(v, Counter):
            out[k] = {t: v[t] for t in ENTITY_TYPES} if k.endswith("_by_type") else dict(sorted(v.items()))
        else:
            out[k] = v
    return out


def _finish(row, work):
    text = work["text"]
    spans = sorted(work["spans"], key=lambda x: (x["start"], x["end"]))
    unc = sorted(work["uncertain"], key=lambda x: (x["start"], x["end"]))
    for x in spans + unc:
        if not x["text"] or text[x["start"]:x["end"]] != x["text"]:
            raise ValueError(f"{row['chapter_id']} [{x['start']},{x['end']}) does not hold its text")
    if any(s["type"] not in ENTITY_TYPES for s in spans):
        raise ValueError(f"{row['chapter_id']}: a span has a type outside the six")
    if any(u["types"] is not None and (not u["types"] or any(t not in ENTITY_TYPES for t in u["types"])) for u in unc):
        raise ValueError(f"{row['chapter_id']}: a region has an empty type list or a type outside the six")
    char_allowed(len(text), spans, unc)  # raises on an overlap
    return {**row, "spans": spans, "uncertain": unc}


def transform(cfg, rows, texts):
    """(new rows, {rule: counts}, {rule: strings}) for the source T rows under cfg (rules_in_order and thresholds);
    texts = {chapter_id: chapter text}. Pure: nothing is read or written."""
    order = list(cfg["rules_in_order"])
    # every listed rule is one of the four, listed once; a rule left out is not applied (its counts stay zero and the
    # record names it under rules_not_applied), so a control like configs/consistency_v2.json can drop one rule
    if any(rule not in RULES for rule in order) or len(set(order)) != len(order) or not order:
        raise SystemExit(f"rules_in_order must list rules among {RULES}, each once, not {order}")
    left_out = [rule for rule in RULES if rule not in order]
    if sorted(cfg.get("rules_not_applied", left_out)) != sorted(left_out):
        raise SystemExit(f"rules_not_applied must name exactly the rules left out of rules_in_order: {left_out}")
    works = [{"text": texts[r["chapter_id"]], "spans": json.loads(json.dumps(r["spans"])),
              "uncertain": json.loads(json.dumps(r["uncertain"]))} for r in rows]
    for r, w in zip(rows, works):
        for x in r["spans"] + r["uncertain"]:
            if w["text"][x["start"]:x["end"]] != x["text"]:
                raise SystemExit(f"{r['chapter_id']} [{x['start']},{x['end']}): the source row does not match the chapter text")
    counts = {rule: Counter({key: Counter() for key in SUB_COUNTERS[rule]}) for rule in RULES}
    for rule in RULES:
        for key in COUNTS[rule]:
            counts[rule][key] = 0
    keys = {"K7_boundary": Counter(), "K6_type_conflict": [], "K5_pending_majority": [], "K8_same_chapter_partial": Counter()}
    for rule in order:
        if rule == "K7_boundary":
            longer = longer_strings_by_type(rows, cfg[rule]["min_len_shorter"])
            for w in works:
                apply_k7(w, longer, counts[rule], keys[rule])
            counts[rule]["distinct_pairs"] = len(keys[rule])
            counts[rule]["distinct_shorter_strings"] = len({p[0] for p in keys[rule]})
            counts[rule]["distinct_longer_strings"] = len({p[1] for p in keys[rule]})
            counts[rule]["overlap_regions_after_the_rule"] = sum(1 for w in works for u in w["uncertain"] if u["reason"] == K7_OVERLAP)
        elif rule == "K6_type_conflict":
            apply_k6(works, cfg[rule]["minority_share"], counts[rule], keys[rule])
        elif rule == "K5_pending_majority":
            apply_k5(works, rows, cfg[rule]["pending_share"], cfg[rule]["min_pending"], counts[rule], keys[rule])
        else:
            for w in works:
                apply_k8(w, cfg[rule]["min_len"], counts[rule], keys[rule])
            counts[rule]["distinct_strings"] = len({s for s, _ in keys[rule]})
    new_rows = [_finish(r, w) for r, w in zip(rows, works)]
    out_counts = {rule: _plain(counts[rule]) for rule in RULES}
    out_keys = {
        "K7_boundary": [{"shorter": s, "longer": L, "type": t, "count": n}
                        for (s, L, t), n in sorted(keys["K7_boundary"].items(), key=lambda kv: (-kv[1], kv[0]))],
        "K6_type_conflict": keys["K6_type_conflict"],
        "K5_pending_majority": keys["K5_pending_majority"],
        "K8_same_chapter_partial": [{"text": s, "types": t.split("+"), "count": n}
                                    for (s, t), n in sorted(keys["K8_same_chapter_partial"].items(), key=lambda kv: (-kv[1], kv[0]))],
    }
    return new_rows, out_counts, out_keys


# ---------------------------------------------------------------------------------------------- the build


def thresholds(cfg):
    return {rule: {k: v for k, v in cfg[rule].items() if k != "rule"} for rule in cfg["rules_in_order"]}


def read_source(record_file, root, source, snapshot_id):
    """(record, {setting: path}) of the source build after the checks: a teacher labels build of snapshot_id whose two
    label files are the ones its record hashed."""
    if not record_file.exists():
        raise SystemExit(f"source build {source}: {record_file} is missing")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if record.get("kind") != TEACHER_BUILD_KIND:
        raise SystemExit(f"source build {source} is not a teacher labels build (kind {record.get('kind')!r})")
    built = record.get("snapshot")
    built = built.get("id") if isinstance(built, dict) else built
    if built != snapshot_id:
        raise SystemExit(f"source build {source} is from snapshot {built}, not {snapshot_id}")
    paths = {}
    for setting in ("T", "W"):
        ref = (record.get("labels") or {}).get(setting) or {}
        expected = f"data/weak_labels/{source}/labels_{setting}.jsonl"
        if ref.get("file") != expected:
            raise SystemExit(f"source build {source}: its record names {ref.get('file')!r} for setting {setting}, not {expected}")
        path = root / expected
        if not path.exists():
            raise SystemExit(f"source build {source}: {expected} is missing")
        if sha256(path) != ref.get("sha256"):
            raise SystemExit(f"source build {source}: labels_{setting}.jsonl differs from the file its record hashed (sha256)")
        paths[setting] = path
    return record, paths


def check_rows(rows, record, snapshot, source):
    """The source T rows are training rows of the snapshot's chapters and give the record's train_ids_sha256."""
    ids = [r["chapter_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"source build {source}: a chapter has two rows")
    for r in rows:
        ch = snapshot.chapters.get(r["chapter_id"])
        if ch is None or ch.number != r["number"] or r.get("split") != "train":
            raise SystemExit(f"source build {source}: {r['chapter_id']} is not a training chapter of snapshot {snapshot.id}")
    if ids_sha256(ids) != (record.get("split") or {}).get("train_ids_sha256"):
        raise SystemExit(f"source build {source}: the rows' chapters do not give the sha256 its record recorded")
    return ids


def target_record(record, source, target, record_file, files, cfg, config_file, config_sha256, code, out_root):
    rec = json.loads(json.dumps(record))
    rec["run_id"] = target
    rec["labels"] = files
    rec["code"] = code  # the code that made these labels (evaluate reads code.dirty and records code.commit as the build's)
    rec["derived_from"] = {
        "source": source, "record": f"runs/{source}/config.json", "config_sha256": sha256(record_file),
        "labels_T_sha256": record["labels"]["T"]["sha256"], "labels_W_sha256": record["labels"]["W"]["sha256"],
        "source_code": record.get("code"),
        "rule": "the source record with run_id, the labels' sha256 and code (the consistency run's git state; the source's is "
                "source_code here) changed and derived_from and consistency added; every other key (snapshot, split, kind, "
                "config, teacher, chapters_file, weak_source, started_at, ...) is the source's, so the build is accepted exactly "
                "like the source; W is the source's file byte for byte",
    }
    rec["consistency"] = {
        "config_file": config_file, "config_sha256": config_sha256,
        "rules_in_order": list(cfg["rules_in_order"]), "rules_not_applied": [r for r in RULES if r not in cfg["rules_in_order"]],
        "thresholds": thresholds(cfg),
        "rules": {rule: cfg[rule]["rule"] for rule in cfg["rules_in_order"]}, "reading": READING,
        "code": code, "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scratch_out": None if out_root is None else str(out_root),
        "material": "the snapshot text of the build's training chapters, the source record and its two label files; no answer "
                    "file, name sheet, dev2, test or test2 chapter or system output (tests/test_consistency.py checks the source)",
    }
    return rec


def run_build(cfg, config_file, source, target, snapshot, root=None, out_root=None, code=None, config_sha256=None):
    """One (source, target) pair: read, transform, check, write; returns (metrics, keys). config_file is the config's
    repo-relative path for the record and config_sha256 its digest (main computes it from the file it read)."""
    root = Path(root or REPO_ROOT)
    out = Path(out_root) if out_root else root
    code = code or git_state()
    record_file = root / "runs" / source / "config.json"
    record, paths = read_source(record_file, root, source, cfg["snapshot"])
    if snapshot.id != cfg["snapshot"]:
        raise SystemExit(f"the snapshot loaded is {snapshot.id}, the config says {cfg['snapshot']}")
    rows = read_jsonl(paths["T"])
    ids = check_rows(rows, record, snapshot, source)
    texts = {c: snapshot.chapters[c].text for c in ids}
    new_rows, counts, keys = transform(cfg, rows, texts)
    if [r["chapter_id"] for r in new_rows] != ids:
        raise SystemExit("the transformed rows are not the source's chapters in the source's order")
    # accepted exactly like the source: the training chapters recomputed from the record (what evaluate and posthoc do)
    # are the rows' chapters; checked on the source record before anything is written (the target copies every key
    # build_train_ids reads) and again on the target record below
    source_ids = build_train_ids(record, snapshot.chapters, root)
    if sorted(source_ids) != sorted(ids) or ids_sha256(source_ids) != (record.get("split") or {}).get("train_ids_sha256"):
        raise SystemExit(f"source build {source}: teacher.build_train_ids on its record does not give its rows' chapters")
    label_dir, run_dir, analysis_dir = out / "data" / "weak_labels" / target, out / "runs" / target, out / "data" / "analysis" / target
    for d in (label_dir, run_dir, analysis_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; builds are never overwritten (use a new target id)")
    label_dir.mkdir(parents=True)
    write_jsonl(label_dir / "labels_T.jsonl", new_rows)
    shutil.copyfile(paths["W"], label_dir / "labels_W.jsonl")
    files = {s: {"file": f"data/weak_labels/{target}/labels_{s}.jsonl", "sha256": sha256(label_dir / f"labels_{s}.jsonl")} for s in ("T", "W")}
    if files["W"]["sha256"] != record["labels"]["W"]["sha256"]:
        raise SystemExit("the copied labels_W.jsonl does not have the source's sha256")
    rec = target_record(record, source, target, record_file, files, cfg, config_file, config_sha256, code, out_root)
    new_ids = build_train_ids(rec, snapshot.chapters, root)
    if new_ids != source_ids or ids_sha256(new_ids) != rec["split"]["train_ids_sha256"]:
        raise SystemExit("teacher.build_train_ids on the new record does not give the source's training chapters")
    before, after = summary(rows), summary(new_rows)
    metrics = {
        "run_id": target, "kind": KIND, "source": source, "chapters": len(new_rows),
        "rules_in_order": list(cfg["rules_in_order"]), "rules_not_applied": [r for r in RULES if r not in cfg["rules_in_order"]],
        "thresholds": thresholds(cfg),
        "before": before, "after": after,
        "spans_removed_from_positives": before["spans"] - after["spans"],
        "uncertain_regions_added_net": after["uncertain"] - before["uncertain"],
        "rules": counts,
        "distinct_strings_affected": {"K7_boundary": counts["K7_boundary"]["distinct_shorter_strings"],
                                      "K6_type_conflict": counts["K6_type_conflict"]["strings_affected"],
                                      "K5_pending_majority": counts["K5_pending_majority"]["strings_affected"],
                                      "K8_same_chapter_partial": counts["K8_same_chapter_partial"]["distinct_strings"]},
        "keys_file": f"data/analysis/{target}/keys.json", "scratch_out": None if out_root is None else str(out_root),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    run_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (analysis_dir / "keys.json").write_text(json.dumps({"run_id": target, "source": source,
                                                         "note": "not committed: holds strings from the book", **keys},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    report(target, metrics, keys)
    print(f"wrote {label_dir}, {run_dir} and {analysis_dir}")
    return metrics, keys


def report(target, metrics, keys, top=10):
    """Console summary (the strings go to the console only, never to runs/)."""
    b, a = metrics["before"], metrics["after"]
    print(f"{target}: {metrics['chapters']} chapters; spans {b['spans']} -> {a['spans']}, uncertain {b['uncertain']} -> {a['uncertain']}")
    print(f"  by type before {b['spans_by_type']}")
    print(f"  by type after  {a['spans_by_type']}")
    print(f"  uncertain by reason after {a['uncertain_by_reason']}")
    for rule in metrics["rules_in_order"]:
        c = metrics["rules"][rule]
        brief = {k: v for k, v in c.items() if not isinstance(v, dict)}
        print(f"  {rule}: {brief}")
    k7 = keys["K7_boundary"][:top]
    print("  K7 top pairs: " + "; ".join(f"{p['shorter']}->{p['longer']} ({p['type']}) x{p['count']}" for p in k7))
    print("  K6 top: " + "; ".join(f"{k['text']} {k['counts_by_type']} minority {k['minority_types']}" for k in keys["K6_type_conflict"][:top]))
    print("  K5 top: " + "; ".join(f"{k['text']} pending {k['pending']} / positives {k['positives']} {k['types']}" for k in keys["K5_pending_majority"][:top]))
    print("  K8 top: " + "; ".join(f"{k['text']} {k['types']} x{k['count']}" for k in keys["K8_same_chapter_partial"][:top]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", help="only the build with this target id")
    ap.add_argument("--out", help="scratch run: write everything under this directory (the code may be dirty)")
    args = ap.parse_args(argv)
    config_file = Path(args.config).as_posix()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    builds = [b for b in cfg["builds"] if not args.only or b["target"] == args.only]
    if not builds:
        raise SystemExit(f"no build with target {args.only!r} in {config_file}")
    out_root = Path(args.out).resolve() if args.out else None
    code = git_state()
    if out_root is None and code.get("dirty") is not False:
        raise SystemExit("src/ or configs/ has uncommitted changes; a recorded run is made from committed code (--out for a scratch run)")
    snapshot = load_snapshot(cfg["snapshot"])
    digest = sha256(Path(args.config))
    for b in builds:
        run_build(cfg, config_file, b["source"], b["target"], snapshot, REPO_ROOT, out_root, code, digest)


if __name__ == "__main__":
    main()
