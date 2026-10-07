"""Character-level BIO tagger trained on weak labels.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.tagger download configs/train_v1.json
    .venv\\Scripts\\python -m novel_ner.tagger train configs/train_v1.json --setting S2 --seed 1 [--run-id <id>]
    .venv\\Scripts\\python -m novel_ner.tagger sweep configs/train_v1.json --settings S1 S2 S3 --seeds 1 2 3
    .venv\\Scripts\\python -m novel_ner.tagger summarize <run id> ... --out runs/<id>
    .venv\\Scripts\\python -m novel_ner.tagger predict --model-run runs/<id> --chapters <file> --run-id <id> [--chunking offset250|overlap250]
    .venv\\Scripts\\python -m novel_ner.tagger vote --model-run runs/<id> --predictions runs/<id> --run-id <id> [--material-run runs/<id>]

A Chinese BERT-style encoder with one linear layer gives every token one of 13 labels: O, and B-/I- for each
of the six types. With the HFL vocabulary a token is one character (letters and digits form word pieces).

Training labels come from a weak-label build (weak.py writes data/weak_labels/<build>/labels_<setting>.jsonl);
the build's record runs/<build>/config.json must exist and match the label file's sha256, snapshot and training
chapters. Only rows marked split "train" are used, and the run stops if one of them is after chapter 780 or a
dev chapter, or if a dev chapter is after 780. Every character gets the set of labels it may take (weak.char_allowed): one label inside a sure name
(B-t on its first character, I-t after) and O elsewhere; O plus the listed types' B-/I- in an uncertain region;
all 13 when the region's type is unknown. The loss is appendix D.3's partial-label NLL, -log of the probability the model puts
on the allowed set: a one-label set gives ordinary cross-entropy, an all-13 set carries no information and is
left out of the average. A token takes the set of its first character; a token with a sure or uncertain
boundary strictly inside it gets no label (counted).

Text: paragraph-aligned chunks of at most max_chars characters (evaluate.chapter_chunks). Six punctuation
marks the HFL vocabulary lacks are first replaced one-for-one (quote_map), so offsets do not move. Nothing is
truncated: a chunk with more than max_tokens tokens is cut into overlapping windows (stride tokens shared),
and each token takes its prediction from the window where it sits farthest from an edge (counted).
Chunking variants (arm C1, inference only; `predict --chunking`, default "paragraph500" = exactly the
above): "offset250" fills the first chunk only up to OFFSET_FIRST_CHARS characters by the same paragraph rule
(offset_chunks), so every later cut moves; "overlap250" labels both chunkings and gives every token of the
paragraph500 chunking the probabilities from the chunking where it sits farther from its chunk's edge, in tokens
(merge_chunkings; ties: paragraph500), looked up by the token's [start, end) in the offset chunking (a token without
an exact match keeps its own; both counted). Decoding is unchanged. A predict run with a variant must have "c1" in its
id; the chunking is recorded in its config.json and metrics.json. Training and dev scoring always use the default.

Decoding: argmax per token; an I- that does not continue a span of its type starts a new span (counted); a
span never runs across a paragraph break (split there, counted). A span's score is the lowest probability
among its tokens' chosen labels.

Dev: every eval_every epoch the model labels the answered dev chapters, scored strictly (evaluate.type_counts
and summarize, score.strict for the error kinds). The best state by micro F1 is kept (ties: the earlier step),
training stops after `patience` evaluations without a gain, and the best model goes to
runs/<id>/checkpoints/best (not committed). runs/<id>/ also gets config.json and metrics.json (numbers only,
committed) and predictions_dev.jsonl (not committed).

Predictions for the test range (chapters 781-990) may only go to a run whose id contains "test", so the read guard
on runs/*test*/predictions* covers them; chapters after 990 (pre-annotation, not test chapters) need "preann" in the
id instead. `predict --snapshot <id>` reads the chapters from a later export of the book (the model's own snapshot by
default); every chapter in both snapshots must have the same text, and both ids are recorded. A run id is one folder
name under runs/ ("runs/<id>" is read as "<id>"); a nested or relative id is refused, since its outputs would escape
that guard and .gitignore.
test2 seal (the material-scope rules): test2_chapters() reads configs/test2_seal.json and returns the first n chapter ids of
the order it names, whatever its status; sealed_test2_chapters() returns them while the status is "sealed" and the
empty set once it is "open". predict refuses a sealed chapter whatever the run id and snapshot (claude_run and
codex_run refuse packages holding them). Once the seal is open (stage 3), a test2 chapter needs "test2" in the run id
(so the read guard covers it); it does not need "preann", which the other chapters after 990 still need. Both files of a
predict run record how many test2 chapters it holds (test2_chapters). Training rows also refuse the dev2 decision
chapters (configs/dev2_781_990.json) by name, besides chapters after 780 and dev chapters.
`init_from_run` (optional config key, "{seed}" is filled with the run's seed): the weights start from that finished
run's checkpoints/best instead of the fresh backbone head (the tokenizer still comes from the backbone); the run must
use the same backbone id, and its id and config.json sha256 are recorded. `grad_accumulation` (default 1) splits each
batch of batch_size windows into that many micro-batches; the micro losses are weighted by their share of the
batch's informative tokens, so the summed loss and gradient equal the whole batch's, and one optimizer and scheduler
step is taken per batch_size windows (steps_per_epoch and the schedule are unchanged). With 1 the run is the same as
without the key. metrics.json records "sufficiency" (sufficiency()): whether the run trained long enough by the rule in
the research plan (the checkpoint rule itself is unchanged: best dev micro F1, ties earlier).
`deterministic` turns on torch's deterministic algorithms in strict mode: an op without a deterministic version
stops the run instead of warning, and the sdpa attention backward takes its deterministic path, so a seed
repeats exactly. `download` fetches the backbone (pytorch_model.bin and the tokenizer
files, no conversion to safetensors); training only reads the local copy.
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import (as_prediction, chapter_chunks, done_chapters, paragraph_bases, sha256, summarize, type_counts,
                                write_jsonl)
from novel_ner.gold import check as check_gold
from novel_ner.score import gold_spans, strict
from novel_ner.snapshot import PARAGRAPH_SEP, REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.weak import LAST_TRAINING_CHAPTER, TAGS, char_allowed, ids_sha256
from novel_ner.zeroshot import environment, git_state

TAG_INDEX = {t: i for i, t in enumerate(TAGS)}
BACKBONE_FILES = ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json"]
MAX_TRAIN_CHAPTER = LAST_TRAINING_CHAPTER
EXAM_LAST_CHAPTER = 990  # the test range is 781-990 (configs/test_781_990.json); later chapters are pre-annotation only
SEAL_FILE = REPO_ROOT / "configs" / "test2_seal.json"  # the second test's seal (tests point this at a temporary file)
DEV2_FILE = REPO_ROOT / "configs" / "dev2_781_990.json"  # the decision chapters; never training rows
TEST_FILE = REPO_ROOT / "configs" / "annotate_test.json"  # the formal test: the first n chapters of the 781-990 order
DEV_FILE = REPO_ROOT / "configs" / "annotate_dev.json"  # the 31 dev chapters: never training rows, never vote material
DEFAULT_CHUNKING = "paragraph500"  # evaluate.chapter_chunks with the config's max_chars (500 in every config)
CHUNKINGS = (DEFAULT_CHUNKING, "offset250", "overlap250")  # arm C1; see the module docstring
OFFSET_FIRST_CHARS = 250  # offset250: the first chunk's character limit; the rest take max_chars
SUFFICIENCY_RULE = ("sufficient when the best checkpoint is not among the last two evaluations and the last three "
                    "evaluations' dev micro F1 differ by at most 0.005, or when training stopped by patience; a run whose "
                    "chosen checkpoint has non-person F1 0 (or undefined) is insufficient either way "
                    "The checkpoint rule is unchanged: best dev micro F1, ties earlier.")


def read_seal(seal=None):
    """(the test2 seal's content, the base its order path is relative to). seal: the seal file's path (default
    SEAL_FILE) or its content as a dict; the status must be "sealed" or "open"."""
    seal = SEAL_FILE if seal is None else seal
    base = REPO_ROOT
    if not isinstance(seal, dict):
        base = Path(seal).resolve().parent.parent  # the seal lives in <repo>/configs/; its order path is repo-relative
        seal = json.loads(Path(seal).read_text(encoding="utf-8"))
    if seal.get("status") not in ("sealed", "open"):
        raise SystemExit(f"test2 seal status {seal.get('status')!r} is neither 'sealed' nor 'open'")
    return seal, base


def test2_chapters(seal=None):
    """The chapter ids of the second test whatever the seal's status: the first n chapter ids of the order file
    configs/test2_seal.json names (seal: its path or its content as a dict, as read_seal)."""
    content, base = read_seal(seal)
    order = Path(content["order"])
    order = order if order.is_absolute() else base / order
    ids = [row["chapter_id"] for row in json.loads(order.read_text(encoding="utf-8"))["order"][:content["n"]]]
    if len(ids) != content["n"]:
        raise SystemExit(f"{order} holds fewer than the seal's n = {content['n']} chapters")
    return frozenset(ids)


def sealed_test2_chapters(seal=None):
    """The chapter ids of the sealed second test: while configs/test2_seal.json says "sealed", test2_chapters(); the
    empty set once it says "open". seal: as read_seal."""
    content, _ = read_seal(seal)
    return test2_chapters(seal) if content["status"] == "sealed" else frozenset()


def dev2_chapters(path=None):
    """The dev2 decision chapters (configs/dev2_781_990.json)."""
    return frozenset(json.loads(Path(path or DEV2_FILE).read_text(encoding="utf-8"))["chapters"])


def exam_chapters(path=None):
    """The formal test chapters: configs/annotate_test.json's first n of the 781-990 order (annotate.chapter_order)."""
    return frozenset(chapter_order(json.loads(Path(path or TEST_FILE).read_text(encoding="utf-8"))))


def dev_chapters(path=None):
    """The dev chapters by name: configs/annotate_dev.json's list (annotate.chapter_order), whatever a training
    config's dev_set says."""
    return frozenset(chapter_order(json.loads(Path(path or DEV_FILE).read_text(encoding="utf-8"))))


def plain_run_id(run_id):
    """The run id as one folder name under runs/ ("<id>" or "runs/<id>"); anything nested or relative is refused,
    since it would put text outside .gitignore and the runs/*test*/predictions* read guard."""
    parts = Path(run_id).parts if run_id else ()
    if len(parts) == 2 and parts[0] == "runs":
        parts = parts[1:]
    if len(parts) != 1 or parts[0] in (".", "..") or Path(parts[0]).name != parts[0]:
        raise SystemExit(f"run id {run_id!r} must be one folder name under runs/ (<id> or runs/<id>)")
    return parts[0]


def map_quotes(text, quote_map):
    """Replace characters one for one (curly quotes etc. the vocabulary lacks), so every offset stays put."""
    if any(len(k) != 1 or len(v) != 1 for k, v in quote_map.items()):
        raise ValueError("quote_map must map single characters to single characters")
    out = "".join(quote_map.get(c, c) for c in text)
    assert len(out) == len(text)
    return out


def allowed_matrix(length, spans, uncertain):
    """[length, 13] bool: weak.char_allowed's label sets for a chapter (sure span: B-t then I-t; uncertain
    region: O plus the listed types' B-/I-, or all 13 when types is null; elsewhere O; overlaps raise)."""
    out = np.zeros((length, len(TAGS)), dtype=bool)
    out[:, TAG_INDEX["O"]] = True
    only_o = frozenset({TAG_INDEX["O"]})
    for i, labels in enumerate(char_allowed(length, spans, uncertain)):
        if labels != only_o:
            out[i] = False
            out[i, sorted(labels)] = True
    return out


def offset_chunks(chapter, max_chars, first_max_chars=OFFSET_FIRST_CHARS):
    """[(offset in the chapter text, chunk text)] like evaluate.chapter_chunks, except that the first chunk is
    filled only up to first_max_chars by the same rule (whole paragraphs; a longer paragraph alone), so every later
    cut moves by about that much. The chunks joined by PARAGRAPH_SEP give back the chapter text: every character
    is in exactly one chunk, except the paragraph break at each cut, as with chapter_chunks."""
    bases = paragraph_bases(chapter)
    out, cur, size, limit, start = [], [], 0, first_max_chars, 0
    for pid, text in chapter.paragraphs:
        extra = len(text) + (len(PARAGRAPH_SEP) if cur else 0)
        if cur and size + extra > limit:
            out.append((start, PARAGRAPH_SEP.join(cur)))
            cur, size, limit = [], 0, max_chars
            extra = len(text)
        if not cur:
            start = bases[pid]
        cur.append(text)
        size += extra
    if cur:
        out.append((start, PARAGRAPH_SEP.join(cur)))
    if PARAGRAPH_SEP.join(t for _, t in out) != chapter.text or any(chapter.text[s:s + len(t)] != t for s, t in out):
        raise ValueError(f"{chapter.chapter_id}: the offset chunks do not give back the chapter text")
    return out


def chapter_chunking(chapter, max_chars, chunking=DEFAULT_CHUNKING):
    """The [(offset, text)] chunks of one chunking: "paragraph500" is evaluate.chapter_chunks unchanged, "offset250"
    is offset_chunks. "overlap250" is both and is handled by predict, not here."""
    if chunking == DEFAULT_CHUNKING:
        return chapter_chunks(chapter, max_chars)
    if chunking == "offset250":
        return offset_chunks(chapter, max_chars)
    raise ValueError(f"chunking must be {DEFAULT_CHUNKING!r} or 'offset250' here, not {chunking!r}")


def encode(tokenizer, chapter, cfg, stats=None, chunking=DEFAULT_CHUNKING):
    """Model inputs for one chapter, one per window: chapter_id, chunk (index), base (the chunk's offset in the
    chapter), first (index in the chunk of the window's first token), input_ids ([CLS] tokens [SEP]) and
    offsets (each token's [start, end) in the chunk). A chunk gets more than one window only when it has more
    than max_tokens tokens; stats counts that. chunking: "paragraph500" (default, chapter_chunks) or "offset250"."""
    room = cfg["max_tokens"] - 2
    step = room - cfg["stride"]
    if step <= 0:
        raise ValueError("stride must be smaller than max_tokens - 2")
    stats = Counter() if stats is None else stats
    out = []
    for k, (base, text) in enumerate(chapter_chunking(chapter, cfg["max_chars"], chunking)):
        enc = tokenizer(map_quotes(text, cfg["quote_map"]), add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        if any(not 0 <= s <= e <= len(text) for s, e in offsets):
            raise ValueError(f"{chapter.chapter_id}: a token offset falls outside its chunk")
        starts = [0]
        while starts[-1] + room < len(ids):
            starts.append(starts[-1] + step)
        stats["chunks"] += 1
        stats["windows"] += len(starts)
        stats["chunks_over_max_tokens"] += len(starts) > 1
        stats["max_tokens_in_a_chunk"] = max(stats["max_tokens_in_a_chunk"], len(ids) + 2)
        stats["unk_tokens"] += ids.count(tokenizer.unk_token_id)
        for a in starts:
            window = {"chapter_id": chapter.chapter_id, "chunk": k, "base": base, "first": a,
                      "input_ids": [tokenizer.cls_token_id] + ids[a:a + room] + [tokenizer.sep_token_id],
                      "offsets": offsets[a:a + room]}
            assert len(window["input_ids"]) <= cfg["max_tokens"]
            out.append(window)
    return out


def token_allowed(offsets, base, allowed, edges):
    """[T, 13] bool for a window's tokens: each token takes its first character's set; a token with a span edge
    strictly inside it (edges: bool per text position) and a token without characters get none. Returns the
    sets and how many tokens were dropped for an edge inside."""
    o = np.asarray(offsets, dtype=np.int64).reshape(-1, 2) + base
    out = np.zeros((len(o), len(TAGS)), dtype=bool)
    real = o[:, 1] > o[:, 0]
    out[real] = allowed[o[real, 0]]
    ignored = 0
    for i in np.flatnonzero(o[:, 1] - o[:, 0] > 1):
        if edges[o[i, 0] + 1:o[i, 1]].any():
            out[i] = False
            ignored += 1
    return out, ignored


def training_windows(tokenizer, pairs, cfg):
    """Windows with every token's allowed label set ("allowed", special tokens empty) for (chapter, weak-label
    row) pairs, and counts of the token kinds."""
    out, stats = [], Counter()
    for chapter, row in pairs:
        text = chapter.text
        allowed = allowed_matrix(len(text), row["spans"], row["uncertain"])
        edges = np.zeros(len(text) + 1, dtype=bool)
        for x in row["spans"] + row["uncertain"]:
            edges[x["start"]] = edges[x["end"]] = True
        covered = np.zeros(len(text), dtype=bool)
        for w in encode(tokenizer, chapter, cfg, stats):
            sets, ignored = token_allowed(w["offsets"], w["base"], allowed, edges)
            none = np.zeros((1, len(TAGS)), dtype=bool)
            w["allowed"] = np.concatenate([none, sets, none])
            n = sets.sum(1)
            stats["tokens"] += len(sets)
            stats["tokens_one_label"] += int((n == 1).sum())
            stats["tokens_partial"] += int(((n > 1) & (n < len(TAGS))).sum())
            stats["tokens_all_labels_masked"] += int((n == len(TAGS)).sum())
            stats["tokens_ignored_edge_inside"] += ignored
            stats["tokens_without_characters"] += sum(e <= s for s, e in w["offsets"])
            for s, e in w["offsets"]:
                covered[w["base"] + s:w["base"] + e] = True
            out.append(w)
        stats["characters_without_a_token"] += int((~covered).sum())
    return out, dict(stats)


def collate(windows, pad_id):
    """Padded input_ids, attention_mask and allowed sets (padding gets none) for a batch of windows."""
    n = max(len(w["input_ids"]) for w in windows)
    ids = torch.full((len(windows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), n), dtype=torch.long)
    allowed = torch.zeros((len(windows), n, len(TAGS)), dtype=torch.bool)
    for i, w in enumerate(windows):
        k = len(w["input_ids"])
        ids[i, :k] = torch.tensor(w["input_ids"])
        mask[i, :k] = 1
        if "allowed" in w:
            allowed[i, :k] = torch.from_numpy(w["allowed"])
    return ids, mask, allowed


def partial_label_nll(logits, allowed):
    """Mean over informative tokens of -log sum_{y in A_i} p(y | token i) (appendix D.3).

    allowed: bool [..., 13]. Tokens whose set is empty (special, padding, ignored) or all 13 labels (masked)
    are left out of the average; a one-label set gives ordinary cross-entropy."""
    keep = allowed.any(-1) & ~allowed.all(-1)
    if not keep.any():
        return logits.sum() * 0.0
    logp = logits[keep].float().log_softmax(-1)  # rows selected first: an all -inf row would give NaN gradients
    return -torch.logsumexp(logp.masked_fill(~allowed[keep], float("-inf")), -1).mean()


def autocast(cfg, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(cfg.get("bf16")) and device.type == "cuda")


@torch.no_grad()
def token_probs(model, windows, pad_id, cfg, device, batch_size):
    """Label probabilities [tokens, 13] for each window's tokens (special tokens dropped)."""
    model.eval()
    out = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        ids, mask, _ = collate(batch, pad_id)
        with autocast(cfg, device):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        probs = logits.float().softmax(-1).cpu().numpy()
        out += [p[1:1 + len(w["offsets"])] for w, p in zip(batch, probs)]
    return out


def merge_windows(windows, probs):
    """{(chapter_id, chunk): (offsets in the chapter [n, 2], probabilities [n, 13])}: every token's probabilities
    come from the window where it sits farthest from an edge (ties: the earlier window)."""
    sizes = Counter()
    for w in windows:
        key = (w["chapter_id"], w["chunk"])
        sizes[key] = max(sizes[key], w["first"] + len(w["offsets"]))
    merged = {key: (np.full(n, -1), np.zeros((n, 2), dtype=np.int64), np.zeros((n, len(TAGS)), dtype=np.float32))
              for key, n in sizes.items()}
    for w, p in zip(windows, probs):
        far, offsets, best = merged[(w["chapter_id"], w["chunk"])]
        m = len(w["offsets"])
        at = np.arange(m)
        dist = np.minimum(at, m - 1 - at)
        better = np.flatnonzero(dist > far[w["first"]:w["first"] + m])
        far[w["first"] + better] = dist[better]
        offsets[w["first"] + better] = np.asarray(w["offsets"], dtype=np.int64).reshape(m, 2)[better] + w["base"]
        best[w["first"] + better] = p[better]
    return {key: (o, q) for key, (_, o, q) in merged.items()}


def edge_distance(n):
    """Each of n tokens' distance (in tokens) from the nearer end of its chunk: 0, 1, 2, ..., 2, 1, 0."""
    at = np.arange(n)
    return np.minimum(at, n - 1 - at)


def token_keys(offsets):
    """One int64 per token from its [start, end) in the chapter, unique among the real tokens of one chunking."""
    o = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    return o[:, 0] * (1 << 32) + o[:, 1]


def merge_chunkings(primary, other, stats):
    """merge_windows's chunk dicts of two chunkings folded into one of primary's shape: a token of a primary chunk
    takes its probabilities from the other chunking's token with the same [start, end) when that token sits
    farther from its own chunk's edge (edge_distance; ties: primary). A real token without an exact match in the
    other chunking keeps its own; stats counts both cases."""
    by_chapter = defaultdict(list)
    for (cid, _), (offsets, q) in other.items():
        by_chapter[cid].append((offsets, q))
    lookup = {}
    for cid, parts in by_chapter.items():
        offsets = np.concatenate([o for o, _ in parts])
        real = offsets[:, 1] > offsets[:, 0]
        keys = token_keys(offsets)[real]
        order = np.argsort(keys)
        lookup[cid] = (keys[order], np.concatenate([edge_distance(len(o)) for o, _ in parts])[real][order],
                       np.concatenate([q for _, q in parts])[real][order])
    out = {}
    stats["tokens_taken_from_offset_chunking"] += 0
    stats["tokens_without_a_match"] += 0
    for (cid, k), (offsets, q) in primary.items():
        q = q.copy()
        real = offsets[:, 1] > offsets[:, 0]
        found = np.zeros(len(offsets), dtype=bool)
        if cid in lookup and len(lookup[cid][0]):
            keys_b, dist_b, q_b = lookup[cid]
            keys = token_keys(offsets)
            pos = np.minimum(np.searchsorted(keys_b, keys), len(keys_b) - 1)
            found = real & (keys_b[pos] == keys)
            take = found & (dist_b[pos] > edge_distance(len(offsets)))
            q[take] = q_b[pos[take]]
            stats["tokens_taken_from_offset_chunking"] += int(take.sum())
        stats["tokens_without_a_match"] += int((real & ~found).sum())
        out[(cid, k)] = (offsets, q)
    return out


def decode_bio(tags, probs, offsets, text):
    """Spans [(start, end, type, score)] from per-token labels (tags), the chosen labels' probabilities and the
    tokens' [start, end) in text. An I- that does not continue a span of its type starts one; a span is split
    where a paragraph break lies between two of its tokens. score = the lowest probability in the span."""
    spans, current = [], None
    repairs = splits = 0
    for tag, p, (s, e) in zip(tags, probs, offsets):
        if tag == "O":
            current = None
            continue
        kind, entity = tag.split("-", 1)
        if kind == "I" and current is not None and current[2] == entity:
            if PARAGRAPH_SEP not in text[current[1]:s]:
                current[1], current[3] = e, min(current[3], p)
                continue
            splits += 1
        elif kind == "I":
            repairs += 1
        current = [s, e, entity, p]
        spans.append(current)
    return [tuple(x) for x in spans], {"decode_i_without_b_repaired": repairs, "decode_split_at_paragraph_break": splits}


def predict(model, tokenizer, chapters, cfg, device, batch_size=None, chunking=DEFAULT_CHUNKING):
    """Prediction rows {chapter_id, start, end, text, type, score} in chapter offsets, and counts (with the
    chunking's name and its chunks per chunking). chunking (CHUNKINGS): with the default the path is exactly the
    one window set; "overlap250" labels the paragraph500 and offset250 chunkings and merges them (merge_chunkings)."""
    began = time.perf_counter()
    if chunking not in CHUNKINGS:
        raise ValueError(f"chunking must be one of {CHUNKINGS}, not {chunking!r}")
    stats = Counter()
    sets = (DEFAULT_CHUNKING, "offset250") if chunking == "overlap250" else (chunking,)
    windows, per_set = [], {}
    for name in sets:
        before = stats["chunks"]
        windows += [w for ch in chapters for w in encode(tokenizer, ch, cfg, stats, name)]
        per_set[name] = (len(windows), stats["chunks"] - before)  # (windows so far, this chunking's chunks)
    probs = token_probs(model, windows, tokenizer.pad_token_id, cfg, device, batch_size or cfg["batch_size"])
    if chunking == "overlap250":
        k = per_set[DEFAULT_CHUNKING][0]
        merged = merge_chunkings(merge_windows(windows[:k], probs[:k]), merge_windows(windows[k:], probs[k:]), stats)
    else:
        merged = merge_windows(windows, probs)
    parts = defaultdict(list)
    for (cid, k), value in sorted(merged.items()):
        parts[cid].append(value)
    rows = []
    for ch in chapters:
        if not parts[ch.chapter_id]:
            continue
        offsets = np.concatenate([o for o, _ in parts[ch.chapter_id]])
        q = np.concatenate([q for _, q in parts[ch.chapter_id]])
        real = offsets[:, 1] > offsets[:, 0]
        offsets, q = offsets[real], q[real]
        best = q.argmax(1)
        spans, counts = decode_bio([TAGS[i] for i in best], q[np.arange(len(q)), best].tolist(), offsets.tolist(), ch.text)
        stats.update(counts)
        for s, e, t, score in spans:
            text = ch.text[s:e]
            if not text or PARAGRAPH_SEP in text:
                raise ValueError(f"{ch.chapter_id} [{s},{e}) is empty or crosses a paragraph break")
            rows.append({"chapter_id": ch.chapter_id, "start": s, "end": e, "text": text, "type": t, "score": float(score)})
    stats["seconds"] = round(time.perf_counter() - began, 2)
    return rows, {**stats, "chunking": chunking, "chunks_by_chunking": {name: n for name, (_, n) in per_set.items()}}


def dev_scores(gold, rows, ids):
    """Strict scores of prediction rows against dev answers: micro, non-person micro, per type, macro (over the
    types with support), the error kinds and per-chapter [tp, fp, fn]."""
    pred = as_prediction(rows, ids)
    counts = type_counts(gold, pred)
    m = strict(gold, pred)
    per_chapter = {cid: [sum(c[t][i] for t in (*ENTITY_TYPES, "_invalid")) for i in range(3)] for cid, c in counts.items()}
    assert (m["tp"], m["fp"], m["fn"]) == tuple(sum(v[i] for v in per_chapter.values()) for i in range(3))
    return {**summarize(list(counts.values())), "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "fp_same_span_wrong_type": m["fp_same_span_wrong_type"], "fp_overlapping_gold": m["fp_overlapping_gold"],
            "per_chapter": per_chapter}


def informative_tokens(window):
    """How many of a window's tokens carry loss: a non-empty allowed set that is not all 13 labels."""
    a = window["allowed"]
    return int((a.any(1) & ~a.all(1)).sum())


def micro_batches(batch, accumulation):
    """[(windows, weight)]: the batch cut into `accumulation` micro-batches, each weighted by its share of the
    batch's informative tokens, so the weighted sum of the micro losses is the whole batch's mean loss. With
    accumulation 1 the batch comes back whole with weight exactly 1.0."""
    micro = math.ceil(len(batch) / accumulation)
    parts = [batch[m:m + micro] for m in range(0, len(batch), micro)]
    counts = [sum(informative_tokens(w) for w in part) for part in parts]
    total = sum(counts)
    if total == 0:  # nothing informative: every micro loss is zero anyway
        return [(part, 1.0 / len(parts)) for part in parts]
    return [(part, n / total) for part, n in zip(parts, counts)]


def fit(model, tokenizer, windows, dev, cfg, seed, device, log=print):
    """Train on windows from training_windows, scoring dev = {"chapters", "gold", "ids"} every eval_every epoch;
    returns the record, with the best state loaded back into model.

    The caller seeds before building the model (the new layer's initial weights); the batch order is shuffled
    with seed. AdamW (no weight decay on biases and LayerNorm), linear warmup then linear decay to 0 over the
    planned steps, gradient clipping, bf16 autocast on CUDA. grad_accumulation (default 1) cuts every batch of
    batch_size windows into micro-batches (micro_batches); one optimizer and scheduler step per batch either way."""
    from transformers import get_linear_schedule_with_warmup

    if not windows:
        raise ValueError("no training windows")
    model.to(device)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
                                   {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=cfg["lr"])
    size = cfg["batch_size"]
    accumulation = int(cfg.get("grad_accumulation", 1))
    if accumulation < 1 or size % accumulation:
        raise ValueError(f"grad_accumulation {accumulation} must be a positive divisor of batch_size {size}")
    per_epoch = math.ceil(len(windows) / size)
    total = per_epoch * cfg["max_epochs"]
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    warmup = int(cfg["warmup"] * total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total)
    every = max(1, round(per_epoch * cfg["eval_every"]))
    rng = random.Random(seed)
    curve, best, state, losses = [], None, None, []
    step, waited, stopped, began = 0, 0, False, time.perf_counter()
    while step < total and not stopped:
        order = list(range(len(windows)))
        rng.shuffle(order)
        for b in range(0, len(order), size):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for part, weight in micro_batches([windows[i] for i in order[b:b + size]], accumulation):
                ids, mask, allowed = collate(part, tokenizer.pad_token_id)
                with autocast(cfg, device):
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = partial_label_nll(logits, allowed.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss is {loss.item()} at step {step + 1}")
                (loss if weight == 1.0 else loss * weight).backward()  # weight 1.0: the same graph as without accumulation
                value += loss.item() if weight == 1.0 else loss.item() * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(value)
            if step % every == 0 or step == total:
                rows, counts = predict(model, tokenizer, dev["chapters"], cfg, device)
                scores = dev_scores(dev["gold"], rows, dev["ids"])
                f1 = scores["micro"]["f1"] or 0.0
                point = {"step": step, "epoch": round(step / per_epoch, 3), "lr": scheduler.get_last_lr()[0],
                         "train_loss": round(sum(losses) / len(losses), 5), "micro": scores["micro"],
                         "non_person_f1": scores["non_person_micro"]["f1"], "macro_f1": scores["macro_f1"],
                         "outputs": len(rows), "seconds": round(time.perf_counter() - began, 1)}
                curve.append(point)
                losses = []
                log(f"step {step}/{total} epoch {point['epoch']:.2f} loss {point['train_loss']:.4f} "
                    f"dev P={scores['micro']['precision']} R={scores['micro']['recall']} F1={f1:.4f} ({point['seconds']} s)")
                if best is None or f1 > best["f1"]:
                    best = {"step": step, "epoch": point["epoch"], "f1": f1, "scores": scores, "rows": rows, "predict_stats": counts}
                    state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                    waited = 0
                else:
                    waited += 1
                    stopped = waited >= cfg["patience"]
            if stopped or step >= total:
                break
    model.load_state_dict(state)
    return {"curve": curve, "best": best, "steps_per_epoch": per_epoch, "steps_planned": total, "warmup_steps": warmup,
            "eval_every_steps": every, "steps_run": step, "epochs_run": round(step / per_epoch, 3),
            "stopped_early": stopped, "grad_accumulation": accumulation, "micro_batch_size": math.ceil(size / accumulation),
            "train_seconds": round(time.perf_counter() - began, 1)}


def sufficiency(result):
    """Whether a fit trained long enough, by SUFFICIENCY_RULE, from fit's record (curve, best, stopped_early)."""
    curve = result["curve"]
    f1s = [p["micro"]["f1"] or 0.0 for p in curve]
    best_index = next(i for i, p in enumerate(curve) if p["step"] == result["best"]["step"])
    evals = len(curve)
    in_last_two = best_index >= evals - 2
    last_three = round(max(f1s[-3:]) - min(f1s[-3:]), 6) if evals >= 3 else None
    by_patience = bool(result["stopped_early"])
    non_person = result["best"]["scores"]["non_person_micro"]["f1"]
    non_person_zero = not non_person  # 0 or undefined (None: nothing predicted outside person)
    converged = not in_last_two and last_three is not None and last_three <= 0.005
    return {"best_eval_index": best_index, "evals": evals, "best_in_last_two_evals": in_last_two,
            "last_three_f1_range": last_three, "stopped_by_patience": by_patience,
            "non_person_f1_at_best": non_person, "non_person_f1_at_best_is_zero": non_person_zero,
            "sufficient": (converged or by_patience) and not non_person_zero, "rule": SUFFICIENCY_RULE}


def seed_everything(seed, deterministic=True):
    """Seed python, numpy and torch (CPU and CUDA); returns the determinism settings in force."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read when cuBLAS starts
    torch.use_deterministic_algorithms(deterministic)  # strict: warn_only would leave the sdpa backward non-deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return {"seed": seed, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def backbone_path(cfg):
    """The backbone's local directory: backbone.path if the config gives one (tests, smoke runs), else the
    Hugging Face cache copy of id@revision. Never downloads."""
    b = cfg["backbone"]
    if b.get("path"):
        return REPO_ROOT / b["path"]
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(b["id"], revision=b["revision"], allow_patterns=BACKBONE_FILES, local_files_only=True))
    except LocalEntryNotFoundError:
        path = None
    if path is None or not (path / "pytorch_model.bin").exists() or not (path / "vocab.txt").exists():
        raise SystemExit(f"{b['id']}@{b['revision']} is not downloaded; run `python -m novel_ner.tagger download <config>` first")
    return path


def load_backbone(path, weights=None):
    """(fast tokenizer, token-classification model with the 13 labels) from a local directory; the weights come
    from `weights` (a finished run's checkpoint) when given, else from path with a fresh head."""
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if not tokenizer.is_fast:
        raise SystemExit("the tagger needs a fast tokenizer (character offsets)")
    model = AutoModelForTokenClassification.from_pretrained(weights or path, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)),
                                                            label2id=TAG_INDEX)
    return tokenizer, model


def init_run(cfg, seed):
    """The finished run cfg["init_from_run"] names ("{seed}" filled with seed), checked: metrics.json and
    checkpoints/best exist, the same backbone id, the same label list. Returns what the record keeps."""
    pattern = cfg["init_from_run"]
    rid = plain_run_id(pattern.format(seed=seed))
    d = REPO_ROOT / "runs" / rid
    if not (d / "metrics.json").exists() or not (d / "checkpoints" / "best").is_dir():
        raise SystemExit(f"init_from_run {rid}: not a finished training run (needs metrics.json and checkpoints/best)")
    rec = json.loads((d / "config.json").read_text(encoding="utf-8"))
    its_backbone = (rec.get("config") or {}).get("backbone", {}).get("id")
    if its_backbone != cfg["backbone"]["id"]:
        raise SystemExit(f"init_from_run {rid} was trained on backbone {its_backbone!r}, not {cfg['backbone']['id']!r}")
    if rec.get("labels") != TAGS:
        raise SystemExit(f"init_from_run {rid} uses another label list")
    return {"pattern": pattern, "run_id": rid, "checkpoint": f"runs/{rid}/checkpoints/best",
            "config_sha256": sha256(d / "config.json"), "setting": rec.get("setting"), "seed": rec.get("seed"),
            "backbone_id": its_backbone}


def training_rows(rows, snapshot, dev_ids, dev2_ids=None):
    """The weak-label rows marked "train", checked: in the snapshot, chapter 1-780, not a dev chapter, not a dev2
    decision chapter (configs/dev2_781_990.json, named explicitly although every one is after 780), every span
    and region where its text says."""
    dev2_ids = dev2_chapters() if dev2_ids is None else dev2_ids
    out, seen = [], set()
    for r in rows:
        if r.get("split") != "train":
            continue
        cid = r["chapter_id"]
        chapter = snapshot.chapters.get(cid)
        if chapter is None or chapter.number != r["number"]:
            raise SystemExit(f"{cid}: not in snapshot {snapshot.id} as chapter {r['number']}")
        if cid in dev2_ids:
            raise SystemExit(f"{cid}: a dev2 decision chapter ({DEV2_FILE.name}) can never be a training chapter")
        if chapter.number > MAX_TRAIN_CHAPTER or cid in dev_ids or cid in seen:
            raise SystemExit(f"{cid}: a training chapter must be <= {MAX_TRAIN_CHAPTER}, not a dev chapter, and listed once")
        seen.add(cid)
        for x in r["spans"] + r["uncertain"]:
            types = [x["type"]] if "type" in x else (x.get("types") or [])
            if any(t not in ENTITY_TYPES for t in types) or chapter.text[x["start"]:x["end"]] != x["text"]:
                raise SystemExit(f"{cid} [{x['start']},{x['end']}): text or type does not match the snapshot")
        out.append(r)
    if not out:
        raise SystemExit("no rows marked split 'train' in the weak labels")
    return out


def train(cfg, setting, seed, run_id, config_file=None, snapshot=None, log=print):
    """One training run: weak labels of `setting`, one seed; writes runs/<run_id>/."""
    began = time.perf_counter()
    run_id = plain_run_id(run_id)
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    if cfg.get("schedule", "linear") != "linear":
        raise SystemExit("only the linear schedule is implemented")
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    labels_file = REPO_ROOT / "data" / "weak_labels" / cfg["weak_build"] / f"labels_{setting}.jsonl"
    if not labels_file.exists():
        raise SystemExit(f"no weak labels at {labels_file}")
    build_record = REPO_ROOT / "runs" / cfg["weak_build"] / "config.json"
    if not build_record.exists():
        raise SystemExit(f"no record of weak build {cfg['weak_build']} at {build_record}")
    built = json.loads(build_record.read_text(encoding="utf-8"))
    built_snapshot = built.get("snapshot")
    built_snapshot = built_snapshot.get("id") if isinstance(built_snapshot, dict) else built_snapshot
    if built_snapshot != snapshot.id:
        raise SystemExit(f"weak build {cfg['weak_build']} is from snapshot {built_snapshot}, not {snapshot.id}")
    if (built.get("labels") or {}).get(setting, {}).get("sha256") != sha256(labels_file):
        raise SystemExit(f"{labels_file.name} differs from the file weak build {cfg['weak_build']} recorded")
    dev_all = set(chapter_order(json.loads((REPO_ROOT / cfg["dev_set"]).read_text(encoding="utf-8"))))
    if any(c not in snapshot.chapters or snapshot.chapters[c].number > MAX_TRAIN_CHAPTER for c in dev_all):
        raise SystemExit(f"every dev chapter must be in snapshot {snapshot.id} and <= {MAX_TRAIN_CHAPTER}")
    rows = read_jsonl(labels_file)
    train_rows = training_rows(rows, snapshot, dev_all)
    train_ids = sorted(r["chapter_id"] for r in train_rows)
    if ids_sha256(train_ids) != (built.get("split") or {}).get("train_ids_sha256"):
        raise SystemExit(f"the training chapters differ from weak build {cfg['weak_build']}'s split")
    export = json.loads((REPO_ROOT / cfg["gold_dev"]).read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the dev answers do not pass the gold check: {errors[:3]}")
    dev_ids = done_chapters(export)
    if not dev_ids or not set(dev_ids) <= dev_all:
        raise SystemExit("the dev answers must hold finished dev chapters")
    gold = gold_spans(export, dev_ids)
    path = backbone_path(cfg)
    init = init_run(cfg, seed) if cfg.get("init_from_run") else None
    determinism = seed_everything(seed, cfg.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_backbone(path, REPO_ROOT / init["checkpoint"] if init else None)
    record = {
        "run_id": run_id, "config_file": config_file, "config": cfg, "setting": setting, "seed": seed,
        "init_from": init,
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
        "weak_labels": {"build": cfg["weak_build"], "file": labels_file.relative_to(REPO_ROOT).as_posix(),
                        "sha256": sha256(labels_file), "rows_by_split": dict(Counter(r.get("split") for r in rows)),
                        "train_chapters": len(train_ids), "train_ids_sha256": ids_sha256(train_ids),
                        "train_ids_sha256_rule": "sha256 of the sorted chapter ids joined by \\n (weak.ids_sha256)",
                        "train_chapter_numbers": [min(r["number"] for r in train_rows), max(r["number"] for r in train_rows)],
                        "build_record": build_record.relative_to(REPO_ROOT).as_posix()},
        "gold_dev": {"file": cfg["gold_dev"], "sha256": sha256(REPO_ROOT / cfg["gold_dev"]), "chapters": dev_ids},
        # the config's repo-relative path (tests, smoke runs) or None; never the cache path under the user profile
        "backbone": {"id": cfg["backbone"]["id"], "revision": cfg["backbone"].get("revision"), "path": cfg["backbone"].get("path"),
                     "attn_implementation": getattr(model.config, "_attn_implementation", None)},
        "labels": TAGS, "device": str(device), "determinism": determinism,
        "code": git_state(), "environment": {**environment(), "torch_cuda": torch.version.cuda},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    windows, token_stats = training_windows(tokenizer, [(snapshot.chapters[r["chapter_id"]], r) for r in train_rows], cfg)
    log(f"{run_id}: {len(train_ids)} training chapters, {len(windows)} windows, {token_stats['tokens']} tokens")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dev = {"chapters": [snapshot.chapters[c] for c in dev_ids], "gold": gold, "ids": dev_ids}
    result = fit(model, tokenizer, windows, dev, cfg, seed, device, log)
    best = result["best"]
    again, _ = predict(model, tokenizer, dev["chapters"], cfg, device)
    keys = lambda rs: sorted((r["chapter_id"], r["start"], r["end"], r["type"]) for r in rs)
    best_dir = out_dir / "checkpoints" / "best"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    write_jsonl(out_dir / "predictions_dev.jsonl", best["rows"])
    metrics = {
        "run_id": run_id, "setting": setting, "seed": seed,
        "train": {"chapters": len(train_ids), **token_stats},
        "schedule": {k: result[k] for k in ("steps_per_epoch", "steps_planned", "warmup_steps", "eval_every_steps",
                                            "steps_run", "epochs_run", "stopped_early", "grad_accumulation",
                                            "micro_batch_size", "train_seconds")},
        "dev_curve": result["curve"],
        "best": {"step": best["step"], "epoch": best["epoch"], "rule": "highest dev micro F1; ties: the earlier step"},
        "sufficiency": sufficiency(result),
        "dev": {"chapters": len(dev_ids), "scored_gold_spans": best["scores"]["tp"] + best["scores"]["fn"],
                "at_best": best["scores"], "predict_stats_at_best": best["predict_stats"],
                "saved_state_gives_the_same_predictions": keys(again) == keys(best["rows"])},
        "predictions_dev_sha256": sha256(out_dir / "predictions_dev.jsonl"),
        "wall_seconds": round(time.perf_counter() - began, 1),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    m = best["scores"]["micro"]
    log(f"{run_id}: best at step {best['step']} (epoch {best['epoch']}): dev P={m['precision']} R={m['recall']} F1={m['f1']}; wrote {out_dir}")
    return metrics


def default_run_id(setting, seed, suffix=""):
    return f"{time.strftime('%Y%m%d')}-tag-{setting}-s{seed}" + (f"-{suffix}" if suffix else "")


def summarize_runs(run_ids, out):
    """Mean, min and max over seeds of each setting's dev scores at the best step; numbers only."""
    out_dir = REPO_ROOT / "runs" / plain_run_id(out)
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    runs = []
    for rid in run_ids:
        d = REPO_ROOT / "runs" / plain_run_id(rid)
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        runs.append((c, m))
    stat = {"micro_f1": lambda s: s["micro"]["f1"], "micro_precision": lambda s: s["micro"]["precision"],
            "micro_recall": lambda s: s["micro"]["recall"], "non_person_f1": lambda s: s["non_person_micro"]["f1"],
            "macro_f1": lambda s: s["macro_f1"]}
    by_setting = defaultdict(list)
    for c, m in runs:
        by_setting[c["setting"]].append((c, m))
    settings = {}
    for setting, group in by_setting.items():
        values = {k: [f(m["dev"]["at_best"]) for _, m in group] for k, f in stat.items()}
        values["best_epoch"] = [m["best"]["epoch"] for _, m in group]
        settings[setting] = {"runs": [m["run_id"] for _, m in group], "seeds": [c["seed"] for c, _ in group],
                             **{k: {"mean": round(sum(v) / len(v), 4) if None not in v else None,
                                    "min": min(v) if None not in v else None, "max": max(v) if None not in v else None}
                                for k, v in values.items()}}
    same = lambda key: len({json.dumps(c[key], sort_keys=True) for c, _ in runs}) == 1
    metrics = {"run_id": out_dir.name, "runs": len(runs), "settings": settings,
               "same_config_across_runs": same("config"), "same_dev_answers": same("gold_dev"),
               "code_commits": sorted({c["code"]["commit"] for c, _ in runs}), "any_dirty_code": any(c["code"]["dirty"] for c, _ in runs),
               "note": "dev = the answered dev chapters; spread over seeds only, no interval (see the integration notes)",
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"run_id": out_dir.name, "runs": [m["run_id"] for _, m in runs],
                                                     "code": git_state(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    for setting, s in settings.items():
        print(f"{setting}: micro F1 mean {s['micro_f1']['mean']} [{s['micro_f1']['min']}, {s['micro_f1']['max']}] over seeds {s['seeds']}")
    return metrics


def read_chapter_ids(path):
    """Chapter ids from a JSON list, a JSON object with "chapters", or plain text (one per line)."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        data = json.loads(text)
        return list(data["chapters"] if isinstance(data, dict) else data)
    return text.split()


def late_chapters(run_id, chapter_ids, snapshot, test2=frozenset()):
    """(the chapters after 780, those of them up to 990, those in test2): a run id must contain "test" when there are
    chapters in the test range 781-990, "test2" when there are chapters of the second test (test2 = its chapter ids,
    test2_chapters(); while the seal is closed refuse_sealed has already stopped the run, so this rule bites once it
    is open) and "preann" when there are chapters after 990 outside test2 (refused otherwise)."""
    late = [c for c in chapter_ids if snapshot.chapters[c].number > MAX_TRAIN_CHAPTER]
    exam = [c for c in late if snapshot.chapters[c].number <= EXAM_LAST_CHAPTER]
    second = [c for c in chapter_ids if c in test2]
    if exam and "test" not in run_id:
        raise SystemExit(f"{len(exam)} chapters are in the test range {MAX_TRAIN_CHAPTER + 1}-{EXAM_LAST_CHAPTER}: "
                         "the run id must contain 'test'")
    if second and "test2" not in run_id:
        raise SystemExit(f"{len(second)} chapters are in test2 ({SEAL_FILE.name}, seal open): the run id must contain 'test2'")
    beyond = [c for c in late if snapshot.chapters[c].number > EXAM_LAST_CHAPTER and c not in test2]
    if beyond and "preann" not in run_id:
        raise SystemExit(f"{len(beyond)} chapters are after {EXAM_LAST_CHAPTER} and not in test2: the run id must contain 'preann'")
    return late, exam, second


def refuse_sealed(chapter_ids):
    """Stop when any of the chapters is in the sealed test2, whatever the run id and snapshot."""
    sealed = [c for c in chapter_ids if c in sealed_test2_chapters()]
    if sealed:
        raise SystemExit(f"{len(sealed)} chapters are in the sealed test2 ({SEAL_FILE.name}): refused whatever the run id "
                         "and snapshot, until the seal is opened in stage 3")


def predict_run(model_run, chapter_ids, run_id, chapters_file=None, snapshot=None, chunking=DEFAULT_CHUNKING):
    """Label chapters with a trained run's best model; writes runs/<run_id>/predictions.jsonl (evaluate.py's row
    shape) and numbers-only config.json and metrics.json. Chapters 781-990 need a run id containing "test", test2
    chapters (once the seal is open; refused while it is closed) one containing "test2", other chapters after 990 one
    containing "preann" (late_chapters). snapshot: the chapters' snapshot when it is not the model's own (a later
    export); every chapter the two share must have the same text. chunking (CHUNKINGS): a variant other than the
    default needs "c1" in the run id and is recorded in both files."""
    run_id = plain_run_id(run_id)
    if chunking not in CHUNKINGS:
        raise SystemExit(f"chunking must be one of {CHUNKINGS}, not {chunking!r}")
    if chunking != DEFAULT_CHUNKING and "c1" not in run_id:
        raise SystemExit(f"chunking {chunking!r} is the inference variant C1: the run id must contain 'c1'")
    refuse_sealed(chapter_ids)
    model_dir = REPO_ROOT / "runs" / plain_run_id(model_run)
    trained = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = trained["config"]
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    lineage = None
    if snapshot.id != cfg["snapshot"]:
        own = load_snapshot(cfg["snapshot"])
        shared = [c for c in own.chapters if c in snapshot.chapters]
        changed = [c for c in shared if own.chapters[c].text != snapshot.chapters[c].text]
        if changed or len(shared) != len(own.chapters):
            raise SystemExit(f"snapshot {snapshot.id} is not a later export of {own.id}: "
                             f"{len(own.chapters) - len(shared)} chapters missing, {len(changed)} with other text")
        lineage = {"model_snapshot": own.id, "shared_chapters": len(shared), "shared_text_identical": True}
    missing = [c for c in chapter_ids if c not in snapshot.chapters]
    if missing or len(set(chapter_ids)) != len(chapter_ids):
        raise SystemExit(f"chapters missing from snapshot {snapshot.id} or listed twice: {missing[:5]}")
    seal, _ = read_seal()
    late, exam, second = late_chapters(run_id, chapter_ids, snapshot, test2_chapters())
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    best_dir = model_dir / "checkpoints" / "best"
    if not best_dir.exists():
        raise SystemExit(f"{best_dir} does not exist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_backbone(best_dir)
    model.to(device)
    record = {"run_id": run_id, "model_run": trained["run_id"], "model_run_config_sha256": sha256(model_dir / "config.json"),
              "setting": trained.get("setting"), "seed": trained.get("seed"),
              "chapters_file": Path(chapters_file).as_posix() if chapters_file else None, "chapters": chapter_ids,
              "chunking": chunking, "test2_chapters": len(second), "test2_seal_status": seal["status"],
              "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
              "snapshot_lineage": lineage, "device": str(device), "code": git_state(), "environment": {**environment(), "torch_cuda": torch.version.cuda},
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    rows, stats = predict(model, tokenizer, [snapshot.chapters[c] for c in chapter_ids], cfg, device, chunking=chunking)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    metrics = {"run_id": run_id, "chapters": len(chapter_ids), "chapters_after_780": len(late),
               "chapters_after_990": len(late) - len(exam), "test2_chapters": len(second), "chunking": chunking, "outputs": len(rows),
               "outputs_by_type": {t: sum(r["type"] == t for r in rows) for t in ENTITY_TYPES}, "predict_stats": stats,
               "predictions_sha256": sha256(out_dir / "predictions.jsonl"), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} spans in {len(chapter_ids)} chapters; wrote {out_dir}")
    return metrics


def download(cfg):
    """Fetch the backbone's weights and tokenizer files into the Hugging Face cache (no safetensors conversion)."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    from huggingface_hub import snapshot_download

    b = cfg["backbone"]
    path = Path(snapshot_download(b["id"], revision=b["revision"], allow_patterns=BACKBONE_FILES))
    total = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f"{b['id']}@{b['revision']}: {path} ({total} bytes)")
    return path, total


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    d = sub.add_parser("download")
    d.add_argument("config")
    t = sub.add_parser("train")
    t.add_argument("config")
    t.add_argument("--setting", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run-id", help="<id> or runs/<id>; default: <date>-tag-<setting>-s<seed>")
    s = sub.add_parser("sweep")
    s.add_argument("config")
    s.add_argument("--settings", nargs="+", required=True)
    s.add_argument("--seeds", nargs="+", type=int, required=True)
    s.add_argument("--suffix", default="", help="appended to every default run id after a hyphen")
    m = sub.add_parser("summarize")
    m.add_argument("runs", nargs="+")
    m.add_argument("--out", required=True, help="<id> or runs/<id>")
    p = sub.add_parser("predict")
    p.add_argument("--model-run", required=True, help="<id> or runs/<id> of a training run")
    p.add_argument("--chapters", required=True, help="file with chapter ids")
    p.add_argument("--run-id", required=True,
                   help="<id> or runs/<id>; must contain 'test' for chapters 781-990, 'test2' for test2 chapters (once the seal "
                        "is open) and 'preann' for other chapters after 990")
    p.add_argument("--snapshot", help="snapshot id of a later export to read the chapters from (default: the model's)")
    p.add_argument("--chunking", choices=CHUNKINGS, default=DEFAULT_CHUNKING,
                   help="C1 inference variant (default paragraph500 = today's chunks); a variant needs 'c1' in the run id")
    v = sub.add_parser("vote", help="C2: revise the types of a predict run by majority over the model's training-chapter predictions")
    v.add_argument("--model-run", required=True, help="<id> or runs/<id> of the training run that made the predictions")
    v.add_argument("--predictions", required=True, help="<id> or runs/<id> of a predict run on the target chapters")
    v.add_argument("--run-id", required=True, help="<id> or runs/<id>; must contain 'c2' (and 'test'/'preann' as predict)")
    v.add_argument("--material-run", help="<id> or runs/<id> of a predict run on exactly the training chapters; "
                                          "default: predict them now into runs/<run-id>-material")
    args = ap.parse_args()

    if args.action == "summarize":
        summarize_runs(args.runs, args.out)
        return
    if args.action == "predict":
        predict_run(args.model_run, read_chapter_ids(args.chapters), args.run_id, args.chapters,
                    load_snapshot(args.snapshot) if args.snapshot else None, args.chunking)
        return
    if args.action == "vote":
        from novel_ner.vote import vote_run

        vote_run(args.model_run, args.predictions, args.run_id, args.material_run)
        return
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config_file = Path(args.config).as_posix()
    if args.action == "download":
        download(cfg)
    elif args.action == "train":
        train(cfg, args.setting, args.seed, args.run_id or default_run_id(args.setting, args.seed), config_file)
    else:
        jobs = [(st, sd, plain_run_id(default_run_id(st, sd, args.suffix))) for st in args.settings for sd in args.seeds]
        done = [rid for _, _, rid in jobs if (REPO_ROOT / "runs" / rid / "metrics.json").exists()]
        if done:
            raise SystemExit(f"these runs already have results: {done}; use --suffix")
        snapshot = load_snapshot(cfg["snapshot"])
        for setting, seed, run_id in jobs:
            train(cfg, setting, seed, run_id, config_file, snapshot)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
