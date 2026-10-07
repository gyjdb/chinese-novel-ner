"""Position-level weak labels from the dictionary and the Opus chapter lists.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.weak configs/weak_v1.json [--run-id <id>]

Training chapters are 1..max_train_chapter minus the 31 development chapters; the test chapters
(781-990) are never labelled or read for anything. Sources, in priority order:
- the chapter's own Opus list (name and type only; a same_as value is never a span);
- dictionary rows that apply in the chapter (scope and reveal chapter, as evaluate.dictionary_predictions);
- dictionary rows blocked there by their scope or reveal chapter;
- (name, type) pairs Opus listed in other training chapters (never a development chapter's list).
Every string a setting's sources propose is matched in the chapter text and one left-to-right pass
keeps the longest match at each position (the guide's flat full-name rule, B2).

Settings:
  S1     dictionary only, positive;
  S1b    S1 with the uncertain regions of S3b that need no Opus list (AutoNER's own setting: a dictionary and
         unknown phrases): dictionary matches positive; blocked dictionary rows, two types, K1 title boundaries,
         K4 tail abbreviations and jieba's proper-noun words in the gaps uncertain;
  S2     chapter list + dictionary, positive (the chapter list's type wins);
  S2raw  S2 without the conversions below;
  S3     all four sources: positive where the chapter list names the string, "uncertain" (no loss,
         AutoNER's unknown spans) where only other sources do (reason = the first of them in priority
         order; the row keeps all its sources) or where the sources give two types;
  S3b    S3 plus words jieba tags as proper nouns in the gaps, uncertain of any type.
Conversions to annotation guide v1.1 (all but S2raw): K1 titles after a person name (B1), K2 the 假
prefix (A6/B3), K3 category words (E3/A2/A5), K4 tail abbreviations of non-person names (E4). There
are no per-name exceptions: a list of whole names would be book knowledge taken from the answers.

data/weak_labels/<run_id>/ gets labels_<setting>.jsonl (a row per chapter, offsets in the chapter text)
and report.md (examples with text); runs/<run_id>/ gets config.json and metrics.json (counts only).
Development labels are built too and scored as a system on the answered chapters, never trained on.

Learning curve: max_train_chapter = dictionary_cutoff = N keeps every training label to
what chapters 1..N give (truncation_rules says how, in words); the record counts the mention rows used and the
positive spans by the prompt version of the chapter list behind them (extract_v1 covers chapters 1-453). Dictionary
rows without a reveal chapter (items, units, registry rows without known) are dated by the first chapter up to 780
whose Opus list names them (the item library and the registry were built from those lists) and dropped when that is
after N; a row no list up to 780 names keeps only the substring test on the training texts, which is not strict (the
same characters can be an ordinary word earlier): the record counts the rows the date drops and the kept rows each
way, and the training spans and regions the undated ones back (metrics: counts; report.md and undated_rows.json in
the label directory: the rows by name).

    .venv\\Scripts\\python -m novel_ner.weak configs/weak_v2.json --verify-run 20261003-weak_v2 [--out <file>]
        rebuilds in memory and compares the label files' sha256 with that run's record; writes nothing unless --out
        is given (then the comparison and the dictionary counts, no text).
    .venv\\Scripts\\python -m novel_ner.weak --compare-rows A.jsonl B.jsonl [--out runs/<id>/<name>.json]
        compares the rows of two label files split by split (numbers and chapter ids only).
"""
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import as_prediction, done_chapters, paragraph_bases, sha256, summarize, type_counts, write_jsonl
from novel_ner.gold import check as check_gold
from novel_ner.score import _context, _touches, gold_spans, strict
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.zeroshot import environment, git_state

LAST_TRAINING_CHAPTER = 780  # the test sample is drawn from the chapters after it
SOURCES = ("chapter_list", "dictionary", "dictionary_blocked", "other_chapters")  # priority order
# positive: in a setting with uncertain regions, the sources whose support makes a match a name
SETTINGS = {
    "S1": {"sources": ("dictionary",), "convert": True, "uncertain": False, "miner": False},
    "S1b": {"sources": ("dictionary", "dictionary_blocked"), "convert": True, "uncertain": True, "miner": True,
            "positive": ("dictionary",)},
    "S2raw": {"sources": ("chapter_list", "dictionary"), "convert": False, "uncertain": False, "miner": False},
    "S2": {"sources": ("chapter_list", "dictionary"), "convert": True, "uncertain": False, "miner": False},
    "S3": {"sources": SOURCES, "convert": True, "uncertain": True, "miner": False, "positive": ("chapter_list",)},
    "S3b": {"sources": SOURCES, "convert": True, "uncertain": True, "miner": True, "positive": ("chapter_list",)},
}
# an uncertain match without the chapter list is named after its first source; later sources may back it too
FIRST_SOURCE_REASON = {"dictionary": "dictionary_applies", "dictionary_blocked": "dictionary_scope_or_reveal",
                       "other_chapters": "other_chapters_only"}
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
PARENTHESES = "（）()"


def split_ids(chapters, dev_ids, test_ids, max_train_chapter):
    """{"train": [...], "dev": [...]} in chapter order: training = chapters up to max_train_chapter minus
    the development chapters; raises if anything touches the test chapters or runs past chapter 780."""
    if max_train_chapter > LAST_TRAINING_CHAPTER:
        raise ValueError(f"max_train_chapter {max_train_chapter} is past chapter {LAST_TRAINING_CHAPTER}")
    missing = set(dev_ids) - set(chapters)
    if missing:
        raise ValueError(f"development chapters not in the snapshot: {sorted(missing)}")
    number = {cid: ch.number for cid, ch in chapters.items()}
    dev = sorted(set(dev_ids), key=number.get)
    train = sorted((c for c in chapters if number[c] <= max_train_chapter and c not in set(dev_ids)), key=number.get)
    if (set(train) | set(dev)) & set(test_ids):
        raise ValueError("a training or development chapter is in the test order")
    if any(number[c] > LAST_TRAINING_CHAPTER for c in train + dev):
        raise ValueError(f"a training or development chapter is past chapter {LAST_TRAINING_CHAPTER}")
    return {"train": train, "dev": dev}


def applies(row, number):
    """A dictionary row counts in a chapter inside its scope and from its reveal chapter on (化名按揭示章节生效)."""
    scope, known = row.get("scope"), row.get("known")
    return (not scope or scope[0] <= number <= scope[1]) and (not known or number >= known)


def listing_date(row, listed_at):
    """The first chapter in listed_at (first_listed()) of the row's name or canonical name, or None."""
    dates = [listed_at[s] for s in (row["name"], row["canonical"]) if s in listed_at]
    return min(dates) if dates else None


def drop_reason(row, min_len, cutoff=None, train_texts=(), listed_at=None):
    """Why clean_dictionary drops a row, or None: the first rule it fails, in this order. The listing date comes
    last, so a row it drops is one every other rule keeps."""
    name = row["name"]
    if any(p in name for p in PARENTHESES):
        return "parenthesis"
    if len(name) < min_len:
        return "shorter_than_min_len"
    if cutoff is None:
        return None
    if listed_at is None:
        raise ValueError("a cutoff needs the first listings (listed_at) to date the rows without a reveal chapter")
    if (row.get("known") or 0) > cutoff:
        return "cutoff_known_after_n"
    if row.get("scope") and row["scope"][0] > cutoff:
        return "cutoff_scope_starts_after_n"
    if not any(name in t for t in train_texts):
        return "cutoff_name_not_in_a_training_text_by_n"
    if not row.get("known"):
        date = listing_date(row, listed_at)
        if date is not None and date > cutoff:
            return "cutoff_first_listed_after_n"
    return None


def clean_dictionary(rows, min_len, cutoff=None, train_texts=(), log=None, listed_at=None):
    """(rows kept, counts of rows dropped by reason). A name with a parenthesis is a registry note
    (…（医院里的投影）) that never occurs in the text; names shorter than min_len would match everywhere.
    With a cutoff N (learning curve) a row stays only if what it says is known by chapter N: it is revealed
    (known) by N, its scope (if any) starts by N, its name occurs in a training text up to N, and a row without
    a reveal chapter (items, units, registry rows without known) is not first listed after N: listed_at
    (first_listed(), the Opus lists up to chapter 780) dates it by its name or canonical name; a row no list
    names has no date and keeps only the text test. The date only ever drops a row the other rules keep.
    log: a list that gets (row, reason) for every dropped row."""
    kept, dropped = [], Counter()
    for row in rows:
        why = drop_reason(row, min_len, cutoff, train_texts, listed_at)
        if why is None:
            kept.append(row)
            continue
        dropped[why] += 1
        if log is not None:
            log.append((row, why))
    return kept, dropped


def is_tail_row(row):
    """A non-person dictionary row whose name is the tail of its own canonical name (琴弦 of 无处不在的琴弦; E4)."""
    return row["type"] != "person" and row["name"] != row["canonical"] and row["canonical"].endswith(row["name"])


def truncation_rules(cfg):
    """What keeps a build's training labels to chapters 1..max_train_chapter, in words (for the record)."""
    n, cutoff = cfg["max_train_chapter"], cfg.get("dictionary_cutoff")
    rules = [
        f"training chapters: 1-{n} minus the development chapters; no mention row of a later chapter is read for a "
        "training label, and the test chapters are never read",
        "a training chapter's positives come from its own Opus list; the other-chapter lexicon, the known person names "
        f"(K1, K2) and the tail abbreviations from same_as (K4) come from training chapters only (never a development "
        f"chapter or a chapter after {n})",
    ]
    if cutoff is None:
        rules.append("dictionary_cutoff null: the whole snapshot dictionary, minus names with a parenthesis or shorter "
                     "than min_len; dictionary tail abbreviations (K4) from every kept row")
    else:
        rules += [
            f"dictionary_cutoff {cutoff}: a row is kept only if its reveal chapter (known) is {cutoff} or earlier, its "
            f"scope (if any) starts by {cutoff}, its name occurs in a training chapter's text up to {cutoff}, and, for a "
            "row without a reveal chapter (items, units, registry rows without known), the first chapter up to 780 whose "
            f"Opus list has its name or canonical name (as name or same_as) is {cutoff} or earlier; a dropped row is "
            "counted under the first rule it fails, in that order (the listing date last, so the rows counted under "
            "cutoff_first_listed_after_n are exactly the rows the date drops that the other rules keep)",
            f"the listing date: the item library and the registry were built from the Opus lists, so a row's first "
            "listing stands for when it entered them (unit names are dated the same way). The date reads the lists of "
            f"every chapter up to 780 (training, development and after {cutoff}; never a chapter after 780) and nothing "
            "else; it never adds a row or a string to any label, it can only drop a row the other rules keep. A "
            f"development chapter's list up to {cutoff} can date a row by {cutoff} where the training chapters' lists "
            "alone would not (the record counts these rows). A late listing also drops a row the book may already use "
            f"by {cutoff} under that name (Opus can list a name late): the rule errs toward dropping",
            f"not strict (stated residual, labels keep it): a row without a reveal chapter that no list up to 780 names "
            "has no date and keeps only the text test, a substring test. A name the book introduces only after "
            f"{cutoff} is kept if the same characters occur by {cutoff} as an ordinary word, an idiom or part of a longer "
            "word (the way 钥匙 occurs inside 钥匙扣 long before the item 钥匙). Such a row never adds a positive "
            "(positives need the chapter's own list) but can make training occurrences uncertain (no loss), including a "
            "chapter-listed name it covers with a longer match, a second type or as a tail. A row dated by its canonical "
            "name's listing keeps the text test for its own name",
            "the record counts the rows the listing date drops (and names them in the label directory), lists every kept "
            f"row without a reveal chapter with the first training chapter up to {cutoff} whose text has its name, its "
            "first listing up to 780 (any split) and its first listing in a training chapter, and counts the training "
            "positive spans and uncertain regions whose matched string is its name and whose sources include its "
            "dictionary type: for all such rows, for the undated residual and for the rows dated only by a list not "
            "trained on",
            f"a kept dictionary row is a K4 tail abbreviation only if its canonical name also occurs in a training "
            f"chapter's text up to {cutoff} (otherwise the tail relation is later knowledge)",
        ]
        if cutoff > n:
            rules.append(f"the cutoff {cutoff} is after max_train_chapter {n}: rows revealed after the training "
                         "chapters are kept, so this build is not truncated")
    rules.append("development rows are labelled with their own chapter's Opus list (also when the chapter is after "
                 "max_train_chapter), the same dictionary and the training-only lexicon; they are scored, never trained on")
    return rules


def mention_rows_by_version(rows, number):
    """[{prompt_version, model, rows, chapters, chapter_numbers}] of mention rows, by prompt version and model."""
    groups = defaultdict(list)
    for m in rows:
        groups[(str(m.get("prompt_version")), str(m.get("model")))].append(m)
    out = []
    for (version, model), ms in sorted(groups.items()):
        numbers = [number[m["chapter_id"]] for m in ms]
        out.append({"prompt_version": version, "model": model, "rows": len(ms),
                    "chapters": len({m["chapter_id"] for m in ms}), "chapter_numbers": [min(numbers), max(numbers)]})
    return out


def by_list_version(rows, list_version):
    """Chapters, positive spans and uncertain regions of a split's labels by the prompt version of the chapter's
    own Opus list ("none" for a chapter without one); a span or region whose sources leave out the chapter
    list counts under "not_from_the_chapter_list"."""
    chapters, spans, unsure = Counter(), Counter(), Counter()
    for r in rows:
        version = list_version.get(r["chapter_id"], "none")
        chapters[version] += 1
        for x in r["spans"]:
            spans[version if any(s == "chapter_list" for s, _ in x["sources"]) else "not_from_the_chapter_list"] += 1
        for x in r["uncertain"]:
            own = x["sources"] and any(s == "chapter_list" for s, _ in x["sources"])
            unsure[version if own else "not_from_the_chapter_list"] += 1
    return {"chapters": dict(sorted(chapters.items())), "positive_spans": dict(sorted(spans.items())),
            "uncertain_regions": dict(sorted(unsure.items()))}


def first_listed(mentions, number, last=LAST_TRAINING_CHAPTER):
    """{string: the first chapter up to `last` whose Opus list has it as a name or a same_as value}. It dates the
    dictionary rows without a reveal chapter under a cutoff (a late date only drops a row) and feeds the record; it
    reads lists of any split up to `last`, never a later one, and no string of it becomes a label."""
    out = {}
    for m in mentions:
        n = number[m["chapter_id"]]
        if n > last:
            continue
        for s in (m["name"], m.get("same_as")):
            if s and (s not in out or n < out[s]):
                out[s] = n
    return out


def undated_rows(dictionary, cutoff, train_texts, listed_at, train_listed_at):
    """For the record: [{row, first_training_text, first_listed, first_listed_in_training, text_test_only,
    dated_only_by_a_list_not_trained_on}] of the kept rows without a reveal chapter (known), which the cutoff dates by
    their first listing up to 780 (any split) or, with no listing (text_test_only), only by the first training chapter
    up to N whose text has the name. train_texts: [(number, text)] of the training chapters up to N; listed_at:
    first_listed() of every list up to 780; train_listed_at: first_listed() of the training chapters' lists."""
    out = []
    for r in dictionary:
        if r.get("known"):
            continue
        first_text = next((n for n, t in train_texts if r["name"] in t), None)
        if first_text is None:
            raise ValueError(f"kept row {r['name']!r} is in no training text up to {cutoff}")
        first, in_training = listing_date(r, listed_at), listing_date(r, train_listed_at)
        if first is not None and first > cutoff:
            raise ValueError(f"kept row {r['name']!r} is first listed at chapter {first}, after {cutoff}")
        out.append({"row": r, "first_training_text": first_text, "first_listed": first,
                    "first_listed_in_training": in_training, "text_test_only": first is None,
                    "dated_only_by_a_list_not_trained_on": first is not None and (in_training is None or in_training > cutoff)})
    return out


DICTIONARY_SOURCES = ("dictionary", "dictionary_blocked")


def dictionary_row_hits(text, labels_row, chosen, rows_by_name, listed_names=()):
    """For the record: [(watched row index, "positive_spans" | "uncertain_regions", (start, end), chapter-listed?)]
    for each span or region of a labels row whose matched string is a watched row's name and whose sources include a
    dictionary pair of that row's type. chapter-listed: the chapter's own list (listed_names) names the matched string
    or a string inside it, so the row's match may hide a would-be positive. chosen: the matches label_chapter resolved
    for the row (K1/K2 only shrink a match, so every span or region with sources lies inside one of them);
    rows_by_name: {name: [(index, row)]}."""
    out = []
    for kind, items in (("positive_spans", labels_row["spans"]), ("uncertain_regions", labels_row["uncertain"])):
        for x in items:
            if not x["sources"]:
                continue  # a phrase-miner region is not a match
            match = next(((s, e) for s, e in chosen if s <= x["start"] and x["end"] <= e), None)
            if match is None:
                raise ValueError(f"[{x['start']},{x['end']}) {x['text']!r} lies in no resolved match")
            matched = text[match[0]:match[1]]
            listed = any(s == "chapter_list" for s, _ in x["sources"]) or any(n in matched for n in listed_names)
            for i, r in rows_by_name.get(matched, ()):
                if any(s in DICTIONARY_SOURCES and t == r["type"] for s, t in x["sources"]):
                    out.append((i, kind, (x["start"], x["end"]), listed))
    return out


def undated_counts(hits, keep=lambda i: True):
    """Counts of distinct training spans and regions among hits [(chapter_id, index, kind, (start, end), listed)]
    whose watched row index passes keep (a region backed by two such rows counts once)."""
    seen = {(c, kind, at): listed for c, i, kind, at, listed in hits if keep(i)}
    return {"positive_spans": sum(1 for _, kind, _ in seen if kind == "positive_spans"),
            "uncertain_regions": sum(1 for _, kind, _ in seen if kind == "uncertain_regions"),
            "uncertain_regions_the_chapter_list_names_it_or_a_part": sum(
                1 for (_, kind, _), listed in seen.items() if kind == "uncertain_regions" and listed),
            "chapters": len({c for c, _, _ in seen})}


def resources(snapshot, split, cfg):
    """What the labels are built from, fixed before any chapter is labelled.

    The other-chapter lexicon, the known person names (K1, K2) and the tail abbreviations (K4) come from
    training chapters only, so no development chapter's list reaches a training label. The lexicon keeps
    names of at least min_len characters, like the dictionary: a book-wide single character (万) would
    match inside ordinary words; single-character names in a chapter's own list are matched there.
    """
    train, wanted = set(split["train"]), set(split["train"]) | set(split["dev"])
    cutoff = cfg.get("dictionary_cutoff")
    number = {c: ch.number for c, ch in snapshot.chapters.items()}
    train_texts = [(number[c], snapshot.chapters[c].text) for c in split["train"] if number[c] <= cutoff] if cutoff is not None else []
    texts = [t for _, t in train_texts] if cutoff is not None else ()
    # under a cutoff, the first listing up to chapter 780 (any split) dates the rows without a reveal chapter
    listed_at = first_listed(snapshot.mentions, number) if cutoff is not None else None
    dropped_rows = []
    dictionary, dropped = clean_dictionary(snapshot.dictionary, cfg["min_len"], cutoff, texts, dropped_rows, listed_at)
    used = [m for m in snapshot.mentions if m["chapter_id"] in wanted]
    if any(snapshot.chapters[m["chapter_id"]].number > LAST_TRAINING_CHAPTER for m in used):
        raise ValueError(f"a mention row past chapter {LAST_TRAINING_CHAPTER} would be used")
    listed, lexicon, versions = defaultdict(set), defaultdict(set), defaultdict(set)
    for m in used:
        listed[m["chapter_id"]].add((m["name"], m["type"]))
        versions[m["chapter_id"]].add(str(m.get("prompt_version")))
        if m["chapter_id"] in train and len(m["name"]) >= cfg["min_len"]:
            lexicon[(m["name"], m["type"])].add(m["chapter_id"])
    training_rows = [m for m in used if m["chapter_id"] in train]
    persons = {m["name"] for m in training_rows if m["type"] == "person"} | {r["name"] for r in dictionary if r["type"] == "person"}
    # under a cutoff a tail relation needs the long name by N too: a canonical name first written later is later knowledge
    tail_rows = [r for r in dictionary if is_tail_row(r)]
    late_tails = [r for r in tail_rows if cutoff is not None and not any(r["canonical"] in t for t in texts)]
    tails = {r["name"] for r in tail_rows if not any(r is x for x in late_tails)}
    tails |= {m["name"] for m in training_rows
              if m["type"] != "person" and m.get("same_as") and m["same_as"] != m["name"] and m["same_as"].endswith(m["name"])}
    list_version = {c: "+".join(sorted(v)) for c, v in versions.items()}
    # the record of the kept rows without a reveal chapter (no label uses it)
    undated = undated_rows(dictionary, cutoff, train_texts, listed_at,
                           first_listed(training_rows, number)) if cutoff is not None else []
    return {
        "dictionary": dictionary, "dictionary_dropped": dropped, "dictionary_dropped_rows": dropped_rows,
        "first_listed_at": listed_at, "undated_rows": undated,
        "listed": listed, "lexicon": lexicon, "persons": persons, "tails": tails,
        "dictionary_tail_rows": len(tail_rows), "dictionary_tail_rows_canonical_after_cutoff": late_tails,
        "titles": sorted(cfg["title_suffixes"], key=len, reverse=True),
        "category_words": set(cfg["category_words"]), "grade": re.compile(cfg["grade_category_pattern"]),
        "mention_rows_used": len(used), "list_version": list_version,
        "train_mention_rows_by_version": mention_rows_by_version(training_rows, number),
        "dev_mention_rows_by_version": mention_rows_by_version([m for m in used if m["chapter_id"] not in train], number),
        "lexicon_pairs_by_list_versions": dict(sorted(Counter(
            "+".join(sorted({list_version[c] for c in where})) for where in lexicon.values()).items())),
    }


def proposals(chapter, res):
    """{string: {(source, type)}} from every source for this chapter."""
    out = defaultdict(set)
    for name, t in res["listed"].get(chapter.chapter_id, ()):
        out[name].add(("chapter_list", t))
    for row in res["dictionary"]:
        out[row["name"]].add(("dictionary" if applies(row, chapter.number) else "dictionary_blocked", row["type"]))
    for (name, t), where in res["lexicon"].items():
        if where - {chapter.chapter_id}:
            out[name].add(("other_chapters", t))
    return out


def active(props, setting):
    """The proposals the setting uses. S1 and S2 drop a string whose only support is dictionary rows of two
    types (plain matching cannot choose; evaluate.dictionary_predictions skips it the same way)."""
    sources = SETTINGS[setting]["sources"]
    out = {}
    for name, pairs in props.items():
        kept = {p for p in pairs if p[0] in sources}
        if not kept:
            continue
        if not SETTINGS[setting]["uncertain"] and not any(s == "chapter_list" for s, _ in kept) \
                and len({t for _, t in kept}) > 1:
            continue
        out[name] = sorted(kept, key=lambda p: (SOURCES.index(p[0]), p[1]))
    return out


def occurrences(text, strings):
    """Every (start, end) where one of strings occurs, by start, longest first."""
    lengths = sorted({len(s) for s in strings}, reverse=True)
    return [(i, i + n) for i in range(len(text)) for n in lengths if i + n <= len(text) and text[i:i + n] in strings]


def resolve(text, strings, found=None):
    """Non-overlapping matches from one left-to-right pass taking the longest string at each position,
    and how many other occurrences were passed over. found: occurrences of a superset of strings."""
    found = occurrences(text, strings) if found is None else [x for x in found if text[x[0]:x[1]] in strings]
    chosen, end = [], 0
    for s, e in found:
        if s >= end:
            chosen.append((s, e))
            end = e
    return chosen, len(found) - len(chosen)


def judge(pairs, setting):
    """(status, types, reason) of a chosen match from its proposals: status "positive", "uncertain" or None."""
    by = defaultdict(set)
    for source, t in pairs:
        by[source].add(t)
    if not SETTINGS[setting]["uncertain"]:
        own = by.get("chapter_list") or by.get("dictionary")
        return ("positive", sorted(own), None) if len(own) == 1 else (None, None, None)
    types = sorted({t for _, t in pairs})
    if len(types) > 1:
        return "uncertain", types, "type_conflict"
    if any(s in by for s in SETTINGS[setting]["positive"]):
        return "positive", types, None
    return "uncertain", types, FIRST_SOURCE_REASON[next(s for s in SOURCES if s in by)]


def convert(match, text, res, setting, tally, examples):
    """The match after K1-K4, or None when it becomes O. tally and examples are updated in place."""
    s = text[match["start"]:match["end"]]
    if s in res["category_words"] or res["grade"].fullmatch(s):
        tally["K3_category_to_O"] += 1
        examples["K3"][s] += 1
        return None
    uncertain_setting = SETTINGS[setting]["uncertain"]
    if match["types"] == ["person"]:
        if s.startswith("假") and s[1:] in res["persons"]:
            match = {**match, "start": match["start"] + 1}
            tally["K2_fake_prefix"] += 1
            examples["K2"][s] += 1
            s = s[1:]
        # the longest title the text ends with; a bare title (大哥) is left as it is, not cut to 大
        title = next((t for t in res["titles"] if s.endswith(t)), None)
        if title and len(s) > len(title):
            x = s[:-len(title)]
            if len(x) == 1 or x in res["persons"]:
                tally["K1_title_stripped"] += 1
                examples["K1_stripped"][f"{s}→{x}"] += 1
                return {**match, "end": match["end"] - len(title)}
            if uncertain_setting:
                tally["K1_title_uncertain"] += 1
                examples["K1_uncertain"][s] += 1
                return {**match, "status": "uncertain", "reason": "title_boundary"}
            tally["K1_title_kept"] += 1
            examples["K1_kept"][s] += 1
        return match
    if match["types"] and "person" not in match["types"] and s in res["tails"]:
        if uncertain_setting:
            tally["K4_tail_uncertain"] += 1
            examples["K4"][s] += 1
            return {**match, "status": "uncertain", "reason": "tail_abbreviation"}
        tally["K4_tail_kept"] += 1
        examples["K4_kept"][s] += 1
    return match


def mine(chapter, flags, min_len):
    """[(start, end)] of words jieba.posseg tags with one of flags (default dictionary, HMM on), in chapter offsets."""
    import jieba
    import jieba.posseg
    jieba.setLogLevel(60)
    out, bases = [], paragraph_bases(chapter)
    for pid, text in chapter.paragraphs:
        base = at = bases[pid]
        for pair in jieba.posseg.cut(text):
            if pair.flag in flags and len(pair.word) >= min_len:
                out.append((at, at + len(pair.word)))
            at += len(pair.word)
        if at != base + len(text):
            raise ValueError(f"{chapter.chapter_id} {pid}: jieba's words do not add up to the paragraph")
    return out


def label_chapter(chapter, res, setting, split="train", props=None, found=None, mined=None):
    """(labels row, tally, examples) for one chapter in one setting."""
    text = chapter.text
    props = proposals(chapter, res) if props is None else props
    use = active(props, setting)
    chosen, passed_over = resolve(text, use, found)
    tally, examples = Counter(passed_over_occurrences=passed_over), defaultdict(Counter)
    spans, uncertain = [], []
    for start, end in chosen:
        pairs = use[text[start:end]]
        status, types, reason = judge(pairs, setting)
        if status is None:
            tally["skipped_type_conflict"] += 1
            continue
        match = {"start": start, "end": end, "status": status, "types": types, "reason": reason, "sources": pairs}
        if SETTINGS[setting]["convert"]:
            match = convert(match, text, res, setting, tally, examples)
        if match is None:
            continue
        piece = text[match["start"]:match["end"]]
        if match["status"] == "positive":
            spans.append({"start": match["start"], "end": match["end"], "type": match["types"][0], "text": piece,
                          "sources": [list(p) for p in match["sources"]]})
        else:
            uncertain.append({"start": match["start"], "end": match["end"], "text": piece, "types": match["types"],
                              "reason": match["reason"], "sources": [list(p) for p in match["sources"]]})
            examples[match["reason"]][piece] += 1
    if SETTINGS[setting]["miner"]:
        for start, end in mined if mined is not None else ():
            if not any(start < e and s < end for s, e in chosen):
                uncertain.append({"start": start, "end": end, "text": text[start:end], "types": None, "reason": "phrase_miner",
                                  "sources": None})
                examples["phrase_miner"][text[start:end]] += 1
    for x in spans + uncertain:
        if text[x["start"]:x["end"]] != x["text"] or not x["text"]:
            raise ValueError(f"{chapter.chapter_id} [{x['start']},{x['end']}) does not hold {x['text']!r}")
    row = {"chapter_id": chapter.chapter_id, "number": chapter.number, "split": split,
           "spans": sorted(spans, key=lambda x: x["start"]), "uncertain": sorted(uncertain, key=lambda x: x["start"])}
    return row, tally, examples


def load_labels(path):
    """{chapter_id: labels row} from a labels_<setting>.jsonl file."""
    return {r["chapter_id"]: r for r in read_jsonl(path)}


def char_allowed(text_len, spans, uncertain):
    """Per character, the frozenset of indices into TAGS the loss may accept.

    A positive span allows only B-type on its first character and I-type on the rest. An uncertain region
    with types allows O plus B-t (first character) or I-t (others) for each listed type; types None allows
    all 13 labels. Every other character allows only O.
    """
    index = {t: i for i, t in enumerate(TAGS)}
    allowed, taken = [frozenset({0})] * text_len, [False] * text_len
    for x, kind in [(x, "span") for x in spans] + [(x, "uncertain") for x in uncertain]:
        if not 0 <= x["start"] < x["end"] <= text_len:
            raise ValueError(f"[{x['start']},{x['end']}) is outside a text of {text_len}")
        for i in range(x["start"], x["end"]):
            if taken[i]:
                raise ValueError(f"character {i} is in two labelled regions")
            taken[i] = True
            prefix = "B" if i == x["start"] else "I"
            if kind == "span":
                allowed[i] = frozenset({index[f"{prefix}-{x['type']}"]})
            elif x["types"] is None:
                allowed[i] = frozenset(range(len(TAGS)))
            else:
                allowed[i] = frozenset({0} | {index[f"{prefix}-{t}"] for t in x["types"]})
    return allowed


def coverage(rows, texts):
    """Counts over a split's labels: positive spans and characters per type, uncertain regions and characters
    per reason, and both kinds by the set of sources behind them ("none" for the phrase miner's regions)."""
    total = sum(len(texts[r["chapter_id"]]) for r in rows)
    spans, chars, by_sources = Counter(), Counter(), Counter()
    regions, unsure, unsure_by_sources = Counter(), Counter(), Counter()
    source_key = lambda pairs: "+".join(sorted({s for s, _ in pairs}, key=SOURCES.index)) if pairs else "none"
    for r in rows:
        for x in r["spans"]:
            spans[x["type"]] += 1
            chars[x["type"]] += x["end"] - x["start"]
            by_sources[source_key(x["sources"])] += 1
        for x in r["uncertain"]:
            regions[x["reason"]] += 1
            unsure[x["reason"]] += x["end"] - x["start"]
            unsure_by_sources[source_key(x["sources"])] += 1
    return {
        "chapters": len(rows), "characters": total,
        "positive_spans": sum(spans.values()), "positive_spans_by_type": {t: spans[t] for t in ENTITY_TYPES},
        "positive_chars_by_type": {t: chars[t] for t in ENTITY_TYPES},
        "positive_char_fraction": round(sum(chars.values()) / total, 5) if total else None,
        "positive_spans_by_sources": dict(by_sources),
        "uncertain_regions_by_reason": dict(regions), "uncertain_chars_by_reason": dict(unsure),
        "uncertain_regions_by_sources": dict(unsure_by_sources),
        "uncertain_char_fraction": round(sum(unsure.values()) / total, 5) if total else None,
    }


def dev_quality(rows, gold, texts):
    """Labels of the answered development chapters scored as a system (positives only, strict), and where
    the uncertain regions sit against the answers."""
    ids = list(gold)
    by_id = {r["chapter_id"]: r for r in rows}
    preds = [{"chapter_id": c, "start": x["start"], "end": x["end"], "text": x["text"], "type": x["type"]}
             for c in ids for x in by_id[c]["spans"]]
    prediction = as_prediction(preds, ids)
    counts = type_counts(gold, prediction)
    errors = strict(gold, prediction)
    inside = touching = regions = regions_without_gold = unsure_chars = 0
    chars = chars_ok = spans_ok = gold_entity_forced_o = gold_o_forced_entity = 0
    index = {t: i for i, t in enumerate(TAGS)}
    for c in ids:
        settled = {s for s in gold[c]["spans"] if not _touches(s, gold[c]["unscored"])}
        unsure = [(x["start"], x["end"]) for x in by_id[c]["uncertain"]]
        inside += sum(1 for s in settled if any(a <= s[0] and s[1] <= b for a, b in unsure))
        touching += sum(1 for s in settled if _touches(s, unsure))
        regions += len(unsure)
        unsure_chars += sum(b - a for a, b in unsure)
        regions_without_gold += sum(1 for u in unsure if not any(_touches(s, [u]) for s in gold[c]["spans"]))
        allowed = char_allowed(len(texts[c]), by_id[c]["spans"], by_id[c]["uncertain"])
        tags = ["O"] * len(texts[c])
        for s, e, t in settled:
            for i in range(s, e):
                tags[i] = f"{'B' if i == s else 'I'}-{t}"
        for i, tag in enumerate(tags):
            if _touches((i, i + 1), gold[c]["unscored"]):
                continue
            chars += 1
            ok = index[tag] in allowed[i]
            chars_ok += ok
            gold_entity_forced_o += tag != "O" and allowed[i] == {0}
            gold_o_forced_entity += tag == "O" and 0 not in allowed[i]
        spans_ok += sum(1 for s, e, t in settled if all(index[tags[i]] in allowed[i] for i in range(s, e)))
    gold_total = sum(1 for c in ids for s in gold[c]["spans"] if not _touches(s, gold[c]["unscored"]))
    return {
        "positives_as_a_system": summarize(list(counts.values()), ENTITY_TYPES),
        "strict_error_kinds": {k: errors[k] for k in ("tp", "fp", "fn", "fp_same_span_wrong_type", "fp_overlapping_gold")},
        "per_chapter_tp_fp_fn": {c: [sum(counts[c][t][i] for t in ENTITY_TYPES) for i in range(3)] for c in ids},
        "uncertain": {"regions": regions, "chars": unsure_chars,
                      "gold_spans_inside_an_uncertain_region": inside, "gold_spans_touching_an_uncertain_region": touching,
                      "regions_touching_no_gold_span": regions_without_gold},
        "gold_vs_allowed_tags": {"gold_spans": gold_total, "gold_spans_all_chars_allowed": spans_ok,
                                 "scored_chars": chars, "chars_gold_tag_allowed": chars_ok,
                                 "char_share_gold_tag_allowed": round(chars_ok / chars, 5) if chars else None,
                                 "gold_entity_chars_where_only_O_allowed": gold_entity_forced_o,
                                 "gold_O_chars_where_O_not_allowed": gold_o_forced_entity},
    }


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def report_lines(run_id, cfg, metrics, rows, examples, gold, texts, res=None, undated=()):
    """report.md: the tables, the truncation and list versions, the kept rows without a chapter number, the dev
    errors in context, conversion and uncertain examples (holds the novel's text and names)."""
    lines = [f"# 弱标签 {run_id}", "", "含小说原文，不入库。【】是范围。设置说明见 configs 里的 description。", "",
             "## 训练章覆盖", "", "| 设置 | 名字处数 | 名字字占比 | 拿不准字占比 | " + " | ".join(ENTITY_TYPES) + " |",
             "|---|---|---|---|" + "---|" * len(ENTITY_TYPES)]
    for s, m in metrics["settings"].items():
        tr = m["train"]
        lines.append(f"| {s} | {tr['positive_spans']} | {tr['positive_char_fraction']:.4f} | {tr['uncertain_char_fraction']:.4f} | "
                     + " | ".join(str(tr["positive_spans_by_type"][t]) for t in ENTITY_TYPES) + " |")
    if res is not None:
        lines += ["", "## 截断规则和名单版本", ""] + [f"- {r}" for r in truncation_rules(cfg)]
        lines += ["", "训练章用到的名单行（按提示词版本和模型）：", ""]
        lines += [f"- {x['prompt_version']} / {x['model']}：{x['rows']} 行，{x['chapters']} 章，第 {x['chapter_numbers'][0]}–"
                  f"{x['chapter_numbers'][1]} 章" for x in metrics["sources"]["train_mention_rows_by_version"]]
        lines += ["", "| 设置 | 训练章按名单版本 | 名字处按本章名单版本 | 拿不准区按本章名单版本 |", "|---|---|---|---|"]
        for s, m in metrics["settings"].items():
            v = m["train"]["by_list_version"]
            lines.append(f"| {s} | {v['chapters']} | {v['positive_spans']} | {v['uncertain_regions']} |")
        lines += ["", f"别的训练章名单里的（名字, 类型）按来源章的名单版本：{metrics['sources']['other_chapter_lexicon_pairs_by_list_versions']}"]
        by_reason = defaultdict(list)
        for row, why in res["dictionary_dropped_rows"]:
            by_reason[why].append(row)
        lines += ["", "词典去掉的条目（按原因）：", ""]
        listed_at = res.get("first_listed_at") or {}
        for why, dropped in sorted(by_reason.items()):
            lines.append(f"- **{why}**（{len(dropped)} 条）：" + "、".join(
                f"{r['name']}（{r['type']}，{r.get('source')}" + (f"，known {r['known']}" if r.get("known") else "")
                + (f"，scope {r['scope']}" if r.get("scope") else "")
                + (f"，全名 {r['canonical']}" if r["canonical"] != r["name"] else "")
                + (f"，名单首列第 {listing_date(r, listed_at)} 章" if why == "cutoff_first_listed_after_n" else "") + "）"
                for r in dropped))
        late = res["dictionary_tail_rows_canonical_after_cutoff"]
        lines.append(f"- **K4 不算尾巴短写（全名不在截断前的训练章正文里）**（{len(late)} 条）："
                     + "、".join(f"{r['name']}←{r['canonical']}" for r in late))
    if metrics["dictionary"].get("rows_without_a_chapter_number"):
        cutoff = cfg["dictionary_cutoff"]
        c = metrics["dictionary"]["rows_without_a_chapter_number"]
        lines += ["", "## 没有揭示章的词典条目：按 Opus 名单首列章定日期", "",
                  f"名单首列在第 {cutoff} 章以后、因此去掉的 {c['dropped_first_listed_after_n']} 条（按来源 "
                  f"{c['dropped_first_listed_after_n_by_source']}，逐条见上面 cutoff_first_listed_after_n）。"
                  f"留下 {c['kept']} 条：名单首列在第 {cutoff} 章以内 {c['dated_by_a_listing_by_n']} 条"
                  f"（其中只靠不参加训练的章——开发章——的名单定到 {cutoff} 章以内的 {c['dated_only_by_a_list_not_trained_on']} 条），"
                  f"780 章以内没有名单列过、只靠正文字串留下的 {c['text_test_only_never_listed_up_to_780']} 条（不严格的残留）。"
                  "定日期读的是 780 章以内所有章（含开发章、含截断以后的章）的名单，只用来去掉条目，不往任何标签里加字串。", ""]
        for s in cfg["settings"]:
            t = metrics["settings"][s]["train"]["rows_without_a_chapter_number"]
            lines.append(f"- {s} 训练章：这些条目撑起的名字处 / 拿不准区（其中本章名单列了这个字串或其中一段的）/ 章数："
                         + "；".join(f"{k} {v['positive_spans']} / {v['uncertain_regions']}"
                                    f"（{v['uncertain_regions_the_chapter_list_names_it_or_a_part']}）/ {v['chapters']}"
                                    for k, v in t.items()))
        lines += ["", "| 名字 | 类型 | 来源 | 全名 | 正文首见章 | 名单首列章 | 训练章名单首列章 | 只靠正文 | "
                  + " | ".join(f"{s} 名字处 / 拿不准区" for s in cfg["settings"]) + " |",
                  "|---|---|---|---|---|---|---|---|" + "---|" * len(cfg["settings"])]
        order = sorted(undated, key=lambda u: (not u["text_test_only"], not u["dated_only_by_a_list_not_trained_on"],
                                               u["first_training_text_chapter"], u["name"]))
        for u in order:
            none = lambda x: x if x is not None else "无"
            lines.append(f"| {u['name']} | {u['type']} | {u['source']} | {u['canonical']} | {u['first_training_text_chapter']} | "
                         f"{none(u['first_listed_chapter'])} | {none(u['first_listed_in_a_training_chapter'])} | "
                         f"{'是' if u['text_test_only'] else ''} | "
                         + " | ".join(f"{u['train'][s]['positive_spans']} / {u['train'][s]['uncertain_regions']}"
                                      for s in cfg["settings"]) + " |")
    lines += ["", f"## 开发章（{len(gold)} 章有答案）上把名字标签当系统打分", "",
              "| 设置 | P | R | F1 | 对 | 多标 | 漏标 | 同范围错类型 | 范围交叉 | 人物以外 F1 | 答案落在拿不准区内 | 拿不准区不碰答案 | 答案标签被允许的字占比 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, m in metrics["settings"].items():
        d = m["dev"]
        mi, e, u, g = d["positives_as_a_system"]["micro"], d["strict_error_kinds"], d["uncertain"], d["gold_vs_allowed_tags"]
        lines.append(f"| {s} | {mi['precision']} | {mi['recall']} | {mi['f1']} | {e['tp']} | {e['fp']} | {e['fn']} | "
                     f"{e['fp_same_span_wrong_type']} | {e['fp_overlapping_gold']} | {d['positives_as_a_system']['non_person_micro']['f1']} | "
                     f"{u['gold_spans_inside_an_uncertain_region']} | {u['regions_touching_no_gold_span']}/{u['regions']} | "
                     f"{g['char_share_gold_tag_allowed']} |")
    for s in cfg["settings"]:
        if s == "S3b":
            continue  # same positives as S3
        lines += ["", f"## {s} 在开发章上的错", ""]
        by_id = {r["chapter_id"]: r for r in rows[s]}
        for c in gold:
            g = {x for x in gold[c]["spans"] if not _touches(x, gold[c]["unscored"])}
            p = {(x["start"], x["end"], x["type"]) for x in by_id[c]["spans"] if not _touches((x["start"], x["end"]), gold[c]["unscored"])}
            lines.append(f"### {c}")
            lines += [f"- 漏标（答案 {t}）：{_context(texts[c], a, b)}" for a, b, t in sorted(g - p)]
            lines += [f"- 多标或错（标 {t}）：{_context(texts[c], a, b)}" for a, b, t in sorted(p - g)]
            if SETTINGS[s]["uncertain"]:
                lines += [f"- 拿不准 {x['reason']} {x['types']}：{_context(texts[c], x['start'], x['end'])}"
                          for x in by_id[c]["uncertain"]]
    for s in cfg["settings"]:
        lines += ["", f"## {s} 训练章里的转换和拿不准（最多 25 个字串，按处数）", ""]
        for kind, counter in sorted(examples[s].items()):
            lines.append(f"- **{kind}**（{sum(counter.values())} 处，{len(counter)} 种）："
                         + "、".join(f"{k} {n}" for k, n in counter.most_common(25)))
    return lines


def compute(cfg):
    """Everything a build makes, in memory (nothing is written): the split, the development answers, the
    resources and every setting's labels with their tallies and examples."""
    snapshot = load_snapshot(cfg["snapshot"])
    dev_ids = chapter_order(json.loads((REPO_ROOT / cfg["dev_config"]).read_text(encoding="utf-8")))
    order = json.loads((REPO_ROOT / cfg["test_order"]).read_text(encoding="utf-8"))
    test_ids = [r["chapter_id"] for r in order["order"]]
    split = split_ids(snapshot.chapters, dev_ids, test_ids, cfg["max_train_chapter"])
    gold_file = REPO_ROOT / cfg["gold_dev"]
    export = json.loads(gold_file.read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the development answers do not pass the gold check: {errors[:3]}")
    scored = done_chapters(export)
    if set(scored) - set(split["dev"]):
        raise SystemExit("answered chapters outside the development set")
    gold = gold_spans(export, scored)
    res = resources(snapshot, split, cfg)

    chapters = [(snapshot.chapters[c], "train") for c in split["train"]] + [(snapshot.chapters[c], "dev") for c in split["dev"]]
    texts = {ch.chapter_id: ch.text for ch, _ in chapters}
    miner = cfg["miner"]
    if miner["tool"] != "jieba.posseg":
        raise SystemExit(f"unknown miner {miner['tool']}")
    rows, tallies, examples = defaultdict(list), defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    blocked_pairs = 0
    # the record of the kept rows without a chapter number: which training spans and regions they back
    rows_by_name, undated_hits = defaultdict(list), {s: [] for s in cfg["settings"]}
    for i, u in enumerate(res["undated_rows"]):
        rows_by_name[u["row"]["name"]].append((i, u["row"]))
    for ch, part in chapters:
        props = proposals(ch, res)
        found = occurrences(ch.text, props)
        mined = mine(ch, set(miner["flags"]), miner["min_len"]) if any(SETTINGS[s]["miner"] for s in cfg["settings"]) else None
        if part == "train":
            blocked_pairs += sum(1 for r in res["dictionary"] if not applies(r, ch.number) and r["name"] in ch.text)
        for s in cfg["settings"]:
            row, tally, ex = label_chapter(ch, res, s, part, props, found, mined)
            rows[s].append(row)
            if part == "train":
                tallies[s].update(tally)
                for kind, counter in ex.items():
                    examples[s][kind].update(counter)
                if rows_by_name:
                    chosen = resolve(ch.text, active(props, s), found)[0]
                    listed_names = {name for name, _ in res["listed"].get(ch.chapter_id, ())}
                    undated_hits[s] += [(ch.chapter_id, *h) for h in dictionary_row_hits(ch.text, row, chosen, rows_by_name,
                                                                                          listed_names)]
    return {"snapshot": snapshot, "split": split, "test_ids": test_ids, "gold_file": gold_file, "scored": scored,
            "gold": gold, "res": res, "texts": texts, "rows": rows, "tallies": tallies, "examples": examples,
            "blocked_pairs": blocked_pairs, "undated_hits": undated_hits}


UNDATED_NOTE = ("rows without a reveal chapter (known) under the cutoff: first listing = the first chapter up to 780 "
                "whose Opus list has the name or the canonical name (as name or same_as), any split; a row first "
                "listed after the cutoff is dropped (dropped_first_listed_after_n: rows every other rule keeps); a kept "
                "row is dated by a listing by the cutoff (dated_only_by_a_list_not_trained_on: no training chapter's "
                "list by the cutoff names it, a development chapter's does) or, never listed up to 780, kept only by "
                "its name occurring as a string in a training text up to the cutoff (text_test_only: the non-strict "
                "residual). Training counts are distinct spans and regions whose matched string is such a row's name "
                "and whose sources include its dictionary type (uncertain_regions_the_chapter_list_names_it_or_a_part: "
                "the chapter's own list names the matched string or a string inside it, so the region may hide a "
                "would-be positive)")


def undated_record(made, cfg):
    """(dictionary counts, {setting: training counts}, {kept: per-row list, dropped: per-row list}) of the rows
    without a reveal chapter, or (None, {}, {}) without a cutoff. The per-row lists name the rows (book content:
    label directory only)."""
    res, settings = made["res"], cfg["settings"]
    undated = res["undated_rows"]
    if cfg.get("dictionary_cutoff") is None:
        return None, {}, {}
    residual = {i for i, u in enumerate(undated) if u["text_test_only"]}
    other_list = {i for i, u in enumerate(undated) if u["dated_only_by_a_list_not_trained_on"]}
    late = [r for r, why in res["dictionary_dropped_rows"] if why == "cutoff_first_listed_after_n"]
    counts = {"dropped_first_listed_after_n": len(late),
              "dropped_first_listed_after_n_by_source": dict(sorted(Counter(r.get("source") for r in late).items())),
              "kept": len(undated), "dated_by_a_listing_by_n": len(undated) - len(residual),
              "dated_only_by_a_list_not_trained_on": len(other_list),
              "text_test_only_never_listed_up_to_780": len(residual), "note": UNDATED_NOTE}
    hits = made["undated_hits"]
    train = {s: {"all": undated_counts(hits[s]),
                 "text_test_only_never_listed_up_to_780": undated_counts(hits[s], residual.__contains__),
                 "dated_only_by_a_list_not_trained_on": undated_counts(hits[s], other_list.__contains__)}
             for s in settings}
    kept = [{"name": u["row"]["name"], "type": u["row"]["type"], "source": u["row"].get("source"),
             "canonical": u["row"]["canonical"], "first_training_text_chapter": u["first_training_text"],
             "first_listed_chapter": u["first_listed"], "first_listed_in_a_training_chapter": u["first_listed_in_training"],
             "text_test_only": u["text_test_only"],
             "dated_only_by_a_list_not_trained_on": u["dated_only_by_a_list_not_trained_on"],
             "train": {s: undated_counts(hits[s], lambda j, i=i: j == i) for s in settings}}
            for i, u in enumerate(undated)]
    dropped = [{"name": r["name"], "type": r["type"], "source": r.get("source"), "canonical": r["canonical"],
                "first_listed_chapter": listing_date(r, res["first_listed_at"])} for r in late]
    return counts, train, {"kept": kept, "dropped_first_listed_after_n": dropped}


def label_bytes(rows):
    """The bytes write_jsonl puts on disk for rows (text mode: each line ends in os.linesep)."""
    return "".join(json.dumps(r, ensure_ascii=False) + os.linesep for r in rows).encode("utf-8")


def verify(cfg, run_id):
    """Rebuild cfg's labels in memory and compare them with runs/<run_id>/config.json: {setting: (sha256 now,
    sha256 recorded)} and the training chapters' sha256 now and recorded. Nothing is written."""
    record = json.loads((REPO_ROOT / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    made = compute(cfg)
    labels = {s: (hashlib.sha256(label_bytes(made["rows"][s])).hexdigest(), (record["labels"].get(s) or {}).get("sha256"))
              for s in cfg["settings"]}
    split = (ids_sha256(made["split"]["train"]), record["split"]["train_ids_sha256"])
    summary, train, _ = undated_record(made, cfg)
    return {"labels": labels, "train_ids_sha256": split,
            "identical": all(a == b for a, b in labels.values()) and split[0] == split[1],
            "dictionary": {"rows": len(made["snapshot"].dictionary), "kept": len(made["res"]["dictionary"]),
                           "dropped": dict(made["res"]["dictionary_dropped"]), "rows_without_a_chapter_number": summary},
            "train_rows_without_a_chapter_number": train}


def _lines_by_split(path):
    """{split: [(chapter_id, raw line without its line end)]} of a labels file, in file order."""
    out = defaultdict(list)
    for line in Path(path).read_bytes().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["split"]].append((row["chapter_id"], line))
    return out


def _regions(row):
    """{key: sources} of a row's positive spans (start, end, type) and uncertain regions (start, end, types, reason)."""
    spans = {(x["start"], x["end"], x["type"]): x["sources"] for x in row["spans"]}
    unsure = {(x["start"], x["end"], tuple(x["types"]) if x["types"] else None, x["reason"]): x["sources"]
              for x in row["uncertain"]}
    return {"positive_spans": spans, "uncertain_regions": unsure}


DIFFERENCE_KINDS = tuple(f"{kind}_{what}" for kind in ("positive_spans", "uncertain_regions")
                         for what in ("only_in_a", "only_in_b", "same_but_sources_differ"))


def compare_rows(path_a, path_b):
    """Compare two label files split by split: per split the number of rows, the sha256 of the raw rows in file
    order (each line followed by \\n) and whether they are byte-identical; for the chapters whose rows differ, the
    positive spans and uncertain regions (by position, type and reason; no text) found in only one file, those found
    in both whose sources differ, and the rows that differ in none of these."""
    a, b = _lines_by_split(path_a), _lines_by_split(path_b)
    digest = lambda lines: hashlib.sha256(b"".join(line + b"\n" for _, line in lines)).hexdigest()
    out = {"a": {"file": Path(path_a).as_posix(), "sha256": sha256(path_a)},
           "b": {"file": Path(path_b).as_posix(), "sha256": sha256(path_b)},
           "rule": "rows_sha256 = sha256 of the split's raw lines in file order, each followed by \\n", "splits": {}}
    for split in sorted(set(a) | set(b)):
        la, lb = a.get(split, []), b.get(split, [])
        by_b = dict(lb)
        differ, diff = [], Counter()
        for cid, line in la:
            if by_b.get(cid) == line:
                continue
            differ.append(cid)
            if cid not in by_b:
                continue
            ra, rb = _regions(json.loads(line)), _regions(json.loads(by_b[cid]))
            here = Counter()
            for kind in ra:
                xa, xb = ra[kind], rb[kind]
                here[f"{kind}_only_in_a"] += len(xa.keys() - xb.keys())
                here[f"{kind}_only_in_b"] += len(xb.keys() - xa.keys())
                here[f"{kind}_same_but_sources_differ"] += sum(1 for k in xa.keys() & xb.keys() if xa[k] != xb[k])
            diff.update(here)
            diff["rows_differing_in_nothing_counted_here"] += not any(here.values())
        ids_a, ids_b = [c for c, _ in la], [c for c, _ in lb]
        out["splits"][split] = {
            "rows_a": len(la), "rows_b": len(lb), "rows_sha256_a": digest(la), "rows_sha256_b": digest(lb),
            "byte_identical": [x for _, x in la] == [x for _, x in lb], "same_chapters_same_order": ids_a == ids_b,
            "chapters_only_in_a": sorted(set(ids_a) - set(ids_b)), "chapters_only_in_b": sorted(set(ids_b) - set(ids_a)),
            "chapters_differing": differ,
            "differences": {k: diff[k] for k in DIFFERENCE_KINDS + ("rows_differing_in_nothing_counted_here",)},
        }
    return out


def build(cfg, run_id, config_file):
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    made = compute(cfg)
    snapshot, split, res, rows, texts, gold = (made[k] for k in ("snapshot", "split", "res", "rows", "texts", "gold"))
    out_dir = REPO_ROOT / "data" / "weak_labels" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    label_files = {}
    for s in cfg["settings"]:
        path = out_dir / f"labels_{s}.jsonl"
        write_jsonl(path, rows[s])
        label_files[s] = {"file": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)}

    undated_summary, undated_train, undated_list = undated_record(made, cfg)
    if undated_summary is not None:
        (out_dir / "undated_rows.json").write_text(json.dumps(
            {"run_id": run_id, "dictionary_cutoff": cfg["dictionary_cutoff"], "note": UNDATED_NOTE,
             "rows": undated_list["kept"], "dropped_first_listed_after_n": undated_list["dropped_first_listed_after_n"]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    metrics = {"run_id": run_id, "settings": {}}
    for s in cfg["settings"]:
        train_rows = [r for r in rows[s] if r["split"] == "train"]
        metrics["settings"][s] = {"train": {**coverage(train_rows, texts),
                                            "conversions_and_resolution": dict(sorted(made["tallies"][s].items())),
                                            "by_list_version": by_list_version(train_rows, res["list_version"])},
                                  "dev": dev_quality(rows[s], gold, texts)}
        if undated_summary is not None:
            metrics["settings"][s]["train"]["rows_without_a_chapter_number"] = undated_train[s]
    cut = [r for r, why in res["dictionary_dropped_rows"] if why.startswith("cutoff_")]
    metrics["dictionary"] = {"rows": len(snapshot.dictionary), "kept": len(res["dictionary"]),
                             "dropped": dict(res["dictionary_dropped"]),
                             "kept_by_source": dict(sorted(Counter(r.get("source") for r in res["dictionary"]).items())),
                             "dropped_by_cutoff_by_source": dict(sorted(Counter(r.get("source") for r in cut).items())),
                             "dropped_by_cutoff_name_in_a_dev_chapter": sum(
                                 1 for r in cut if any(r["name"] in texts[c] for c in split["dev"])),
                             "rows_with_scope_or_reveal": sum(1 for r in res["dictionary"] if r.get("scope") or r.get("known")),
                             "tail_rows_kept": res["dictionary_tail_rows"],
                             "tail_rows_not_used_canonical_not_in_a_training_text_by_n": len(
                                 res["dictionary_tail_rows_canonical_after_cutoff"]),
                             "blocked_row_chapter_pairs_with_the_name_in_a_training_text": made["blocked_pairs"]}
    if undated_summary is not None:
        metrics["dictionary"]["rows_without_a_chapter_number"] = undated_summary
    metrics["sources"] = {"mention_rows_used": res["mention_rows_used"],
                          "train_mention_rows": sum(x["rows"] for x in res["train_mention_rows_by_version"]),
                          "train_mention_rows_by_version": res["train_mention_rows_by_version"],
                          "dev_mention_rows_by_version": res["dev_mention_rows_by_version"],
                          "other_chapter_lexicon_pairs": len(res["lexicon"]),
                          "other_chapter_lexicon_pairs_by_list_versions": res["lexicon_pairs_by_list_versions"],
                          "known_person_names": len(res["persons"]),
                          "tail_abbreviation_strings": len(res["tails"])}
    metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    report = out_dir / "report.md"
    report.write_text("\n".join(report_lines(run_id, cfg, metrics, rows, made["examples"], gold, texts, res,
                                             undated_list.get("kept", ()))) + "\n", encoding="utf-8")
    import jieba
    numbers = [snapshot.chapters[c].number for c in split["train"]]
    record = {
        "run_id": run_id, "config_file": config_file, "config": cfg,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
        "code": git_state(), "environment": {**environment(), "jieba": jieba.__version__},
        "split": {"max_train_chapter": cfg["max_train_chapter"], "train_chapters": len(split["train"]),
                  "train_numbers": [min(numbers), max(numbers)], "train_ids_sha256": ids_sha256(split["train"]),
                  "train_ids_sha256_rule": "sha256 of the sorted chapter ids joined by \\n",
                  "dev_chapters": split["dev"], "test_order_chapters": len(made["test_ids"]),
                  "test_order_overlap": 0},
        "truncation": {"max_train_chapter": cfg["max_train_chapter"], "dictionary_cutoff": cfg.get("dictionary_cutoff"),
                       "rules": truncation_rules(cfg)},
        "gold_dev": {"file": cfg["gold_dev"], "sha256": sha256(made["gold_file"]), "chapters": made["scored"]},
        "labels": label_files, "started_at": started,
    }
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run-id", help="default: <date>-<config name>")
    ap.add_argument("--verify-run", metavar="RUN_ID",
                    help="rebuild the config's labels in memory and compare their sha256 with runs/RUN_ID; writes "
                         "nothing unless --out is given")
    ap.add_argument("--compare-rows", nargs=2, metavar=("A", "B"), help="compare two label files split by split")
    ap.add_argument("--out", help="with --compare-rows or --verify-run: also write the result (numbers, sha256 and "
                                  "chapter ids, no text) to this file")
    args = ap.parse_args()
    if args.compare_rows:
        result = compare_rows(*args.compare_rows)
        if args.out:
            result = {**result, "code": git_state(), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    if not args.config:
        ap.error("a config is needed unless --compare-rows is given")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.verify_run:
        result = verify(cfg, args.verify_run)
        for s, (now, then) in result["labels"].items():
            print(f"{s}: {'same' if now == then else 'DIFFERENT'} {now} (recorded {then})")
        same_split = result["train_ids_sha256"][0] == result["train_ids_sha256"][1]
        print(f"training chapters: {'same' if same_split else 'DIFFERENT'}")
        print(f"dictionary now: kept {result['dictionary']['kept']} of {result['dictionary']['rows']}, "
              f"dropped {result['dictionary']['dropped']}")
        print("identical" if result["identical"] else "NOT identical")
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"verified_run": args.verify_run, "config_file": Path(args.config).as_posix(),
                 "truncation_rules_now": truncation_rules(cfg), **result,
                 "rule": "labels: {setting: [sha256 rebuilt now, sha256 recorded in the run's config.json]}",
                 "code": git_state(), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit(0 if result["identical"] else 1)
    run_id = args.run_id or f"{time.strftime('%Y%m%d')}-{Path(args.config).stem}"
    if (REPO_ROOT / "runs" / run_id / "metrics.json").exists():
        raise SystemExit(f"runs/{run_id} already has results; use a new run id")
    metrics, report = build(cfg, run_id, Path(args.config).as_posix())
    print("training mention rows: " + "; ".join(
        f"{x['prompt_version']} / {x['model']}: {x['rows']} rows in {x['chapters']} chapters {x['chapter_numbers']}"
        for x in metrics["sources"]["train_mention_rows_by_version"]))
    print(f"dictionary: kept {metrics['dictionary']['kept']} of {metrics['dictionary']['rows']}, "
          f"dropped {metrics['dictionary']['dropped']}")
    for s, m in metrics["settings"].items():
        tr, d = m["train"], m["dev"]
        mi = d["positives_as_a_system"]["micro"]
        print(f"{s}: train {tr['positive_spans']} spans {tr['by_list_version']['positive_spans']}, "
              f"positive chars {tr['positive_char_fraction']:.4f}, uncertain chars {tr['uncertain_char_fraction']:.4f}; "
              f"dev P={mi['precision']} R={mi['recall']} F1={mi['f1']} "
              f"(tp {d['strict_error_kinds']['tp']}, fp {d['strict_error_kinds']['fp']}, fn {d['strict_error_kinds']['fn']}), "
              f"gold inside uncertain {d['uncertain']['gold_spans_inside_an_uncertain_region']}")
    print(f"wrote runs/{run_id}/config.json, metrics.json and {report}")


if __name__ == "__main__":
    main()
