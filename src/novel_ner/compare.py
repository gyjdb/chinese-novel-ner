"""Chapter-level comparison of two name lists (e.g. a model against the LLM extraction).

The LLM extraction is not a gold standard: it misses names and has its own errors.
"Agreement" numbers here say how much two systems overlap, not which one is right.
"""
from collections import Counter, defaultdict

ENTITY_TYPES = ("person", "organization", "location", "dungeon", "item", "trait")


def names_from_mentions(mentions, chapter_ids):
    """{chapter_id: {name: {types}}} from snapshot mention rows."""
    wanted = set(chapter_ids)
    out = {cid: defaultdict(set) for cid in chapter_ids}
    for row in mentions:
        if row["chapter_id"] in wanted:
            out[row["chapter_id"]][row["name"]].add(row["type"])
    return out


def names_from_predictions(predictions, chapter_ids, threshold):
    """{chapter_id: {name: {type}}}; a name seen with several labels keeps its best-scoring one."""
    best = {cid: {} for cid in chapter_ids}
    for p in predictions:
        if p["score"] < threshold:
            continue
        seen = best[p["chapter_id"]].get(p["text"])
        if seen is None or p["score"] > seen[1]:
            best[p["chapter_id"]][p["text"]] = (p["label"], p["score"])
    return {cid: {name: {label} for name, (label, _) in names.items()} for cid, names in best.items()}


def names_from_dictionary(dictionary, chapters, min_len=2):
    """Plain string matching of dictionary names in each chapter (a weak-label baseline).

    Names shorter than min_len are skipped (single-character aliases match everywhere);
    a scoped alias only counts inside its chapter range.
    """
    out = {}
    for ch in chapters:
        text = ch.text
        found = defaultdict(set)
        for row in dictionary:
            name = row["name"]
            scope = row.get("scope")
            if len(name) < min_len or (scope and not scope[0] <= ch.number <= scope[1]):
                continue
            if name in text:
                found[name].add(row["type"])
        out[ch.chapter_id] = found
    return out


def _prf(tp_p, n_pred, tp_r, n_ref):
    p = tp_p / n_pred if n_pred else 0.0
    r = tp_r / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def agreement(pred, ref):
    """Micro-averaged agreement of pred against ref, both {chapter_id: {name: {types}}}.

    exact: same string; typed: same string and a shared type;
    loose: one string contains the other (shows boundary disagreements).
    """
    n_pred = n_ref = exact = typed = loose_p = loose_r = 0
    ref_by_type = Counter()
    hit_by_type = Counter()
    pred_by_type = Counter()
    for cid, ref_names in ref.items():
        pred_names = pred.get(cid, {})
        n_pred += len(pred_names)
        n_ref += len(ref_names)
        for name, types in pred_names.items():
            for t in types:
                pred_by_type[t] += 1
            if name in ref_names:
                exact += 1
                typed += bool(types & ref_names[name])
            if any(name in r or r in name for r in ref_names):
                loose_p += 1
        for name, types in ref_names.items():
            for t in types:
                ref_by_type[t] += 1
                hit_by_type[t] += name in pred_names
            if any(name in p or p in name for p in pred_names):
                loose_r += 1
    return {
        "n_pred": n_pred,
        "n_ref": n_ref,
        "exact": _prf(exact, n_pred, exact, n_ref),
        "typed": _prf(typed, n_pred, typed, n_ref),
        "loose": _prf(loose_p, n_pred, loose_r, n_ref),
        "recall_by_ref_type": {t: round(hit_by_type[t] / ref_by_type[t], 4) if ref_by_type[t] else None
                               for t in ENTITY_TYPES},
        "ref_by_type": {t: ref_by_type[t] for t in ENTITY_TYPES},
        "pred_by_type": {t: pred_by_type[t] for t in ENTITY_TYPES},
    }
