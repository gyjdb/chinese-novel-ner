import hashlib
import json
from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from novel_ner import tagger
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import chapter_chunks, sha256
from novel_ner.gold import FORMAT
from novel_ner.snapshot import PARAGRAPH_SEP, Chapter, Snapshot
from novel_ner.tagger import (TAGS, allowed_matrix, decode_bio, encode, fit, map_quotes, merge_chunkings, merge_windows,
                              offset_chunks, partial_label_nll, predict, training_windows)
from novel_ner.weak import ids_sha256

QUOTES = {"“": "「", "”": "」", "‘": "『", "’": "』", "…": "⋯", "—": "─"}
CFG = {"max_chars": 500, "max_tokens": 512, "stride": 128, "quote_map": QUOTES, "bf16": False, "batch_size": 2,
       "lr": 1e-4, "weight_decay": 0.01, "warmup": 0.1, "grad_clip": 1.0, "max_epochs": 3, "eval_every": 1.0, "patience": 3}
TRAIN = Chapter("ch0001", 1, [("p1", "徐获拿出回音铃。"), ("p2", "唐医生到了汀城。")], "", "h1")
DEV = Chapter("ch0002", 2, [("p1", "唐医生见到徐获。"), ("p2", "NPC拿着回音铃。")], "", "h2")
LATE = Chapter("ch0781", 781, [("p1", "徐获回到汀城。")], "", "h3")


def tiny_tokenizer(path):
    chars = sorted(set("".join(c.text for c in (TRAIN, DEV, LATE)) + "徐获见到唐医生拿出回音铃汀城「」『』⋯─") - set("\nNPC"))
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, "npc"]) + "\n", encoding="utf-8")
    return BertTokenizerFast.from_pretrained(path)


def tiny_model(tokenizer, seed=0, dropout=0.1):
    torch.manual_seed(seed)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, num_labels=len(TAGS),
                        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
    return BertForTokenClassification(config)


def span(start, end, text, type_):
    return {"start": start, "end": end, "text": text, "type": type_, "sources": ["test"]}


def region(start, end, text, types):
    return {"start": start, "end": end, "text": text, "types": types, "reason": "test"}


def names(allowed, i):
    return {TAGS[k] for k in np.flatnonzero(allowed[i])}


def test_quote_map_replaces_one_for_one_and_removes_unknown_tokens(tmp_path):
    text = "他说：“回音铃……”——‘徐获’"
    assert map_quotes(text, QUOTES) == "他说：「回音铃⋯⋯」──『徐获』"
    assert len(map_quotes(text, QUOTES)) == len(text)
    with pytest.raises(ValueError):
        map_quotes(text, {"…": "..."})
    tok = tiny_tokenizer(tmp_path)
    assert tok.unk_token_id in tok("“回音铃”", add_special_tokens=False)["input_ids"]
    assert tok.unk_token_id not in tok(map_quotes("“回音铃”", QUOTES), add_special_tokens=False)["input_ids"]


def test_allowed_sets_for_sure_uncertain_and_unknown_regions():
    a = allowed_matrix(8, [span(0, 2, "徐获", "person")],
                     [region(3, 5, "回音", ["item", "location"]), region(6, 7, "城", None)])
    assert names(a, 0) == {"B-person"} and names(a, 1) == {"I-person"}
    assert names(a, 2) == {"O"} and names(a, 5) == {"O"} and names(a, 7) == {"O"}
    assert names(a, 3) == {"O", "B-item", "B-location"} and names(a, 4) == {"O", "I-item", "I-location"}
    assert names(a, 6) == set(TAGS)
    with pytest.raises(ValueError):
        allowed_matrix(8, [span(0, 2, "徐获", "person"), span(1, 3, "获拿", "item")], [])


def test_tokens_take_the_first_characters_set_and_skip_an_edge_inside_a_word_piece(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    ch = Chapter("ch0003", 3, [("p1", "徐获见到NPC。"), ("p2", "回音铃")], "", "h")  # "NPC" is one word piece
    sure = [span(0, 2, "徐获", "person"), span(9, 12, "回音铃", "item")]
    windows, stats = training_windows(tok, [(ch, {"spans": sure, "uncertain": [region(4, 6, "NP", ["organization"])]})], CFG)
    assert len(windows) == 1
    w = windows[0]
    assert w["offsets"] == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (7, 8), (9, 10), (10, 11), (11, 12)]
    labels = [names(w["allowed"], i) for i in range(len(w["allowed"]))]
    assert labels == [set(), {"B-person"}, {"I-person"}, {"O"}, {"O"}, set(), {"O"}, {"B-item"}, {"I-item"}, {"I-item"}, set()]
    assert (stats["tokens"], stats["tokens_one_label"], stats["tokens_ignored_edge_inside"]) == (9, 8, 1)
    assert stats["characters_without_a_token"] == 1  # the paragraph break
    # the region covering the whole word piece: no edge inside, all 13 labels allowed (masked)
    _, stats = training_windows(tok, [(ch, {"spans": sure, "uncertain": [region(4, 7, "NPC", None)]})], CFG)
    assert (stats["tokens_all_labels_masked"], stats["tokens_ignored_edge_inside"]) == (1, 0)


def test_long_chunk_gets_overlapping_windows_and_each_token_the_window_farthest_from_an_edge(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    ch = Chapter("ch0004", 4, [("p1", "徐获见到唐医生拿出回音铃汀城")], "", "h")  # 14 tokens
    stats = Counter()
    windows = encode(tok, ch, {**CFG, "max_tokens": 8, "stride": 2}, stats)
    assert [w["first"] for w in windows] == [0, 4, 8]
    assert all(len(w["input_ids"]) <= 8 for w in windows)
    assert (stats["chunks_over_max_tokens"], stats["windows"], stats["max_tokens_in_a_chunk"]) == (1, 3, 16)
    probs = [np.full((len(w["offsets"]), len(TAGS)), k, dtype=np.float32) for k, w in enumerate(windows)]
    offsets, q = merge_windows(windows, probs)[("ch0004", 0)]
    assert q[:, 0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # ties go to the earlier window
    assert offsets.tolist() == [[i, i + 1] for i in range(14)]


def paragraphs(n, length=75):
    """n paragraphs of `length` characters from the tiny vocabulary (the unit is 15 characters)."""
    unit = "徐获见到唐医生拿出回音铃汀城。"
    return [(f"p{i}", unit * (length // len(unit))) for i in range(1, n + 1)]


def test_offset_chunks_cover_every_character_once_and_move_every_cut():
    ch = Chapter("ch0005", 5, paragraphs(10), "", "h")  # 10 x 75 characters + 9 breaks = 759
    plain = chapter_chunks(ch, 500)
    offset = offset_chunks(ch, 500)
    assert [b for b, _ in plain] == [0, 456]  # p1-p6 (455) | p7-p10
    assert [b for b, _ in offset] == [0, 228, 684]  # p1-p3 (227, the first chunk stops at 250) | p4-p9 (455) | p10
    for chunks in (plain, offset):
        covered = np.zeros(len(ch.text), dtype=int)
        for base, text in chunks:
            assert ch.text[base:base + len(text)] == text
            covered[base:base + len(text)] += 1
        assert PARAGRAPH_SEP.join(t for _, t in chunks) == ch.text
        assert covered.max() == 1 and (covered == 0).sum() == len(chunks) - 1  # only the break at each cut is in no chunk
        assert all(ch.text[i] == PARAGRAPH_SEP for i in np.flatnonzero(covered == 0))
    assert not {b for b, _ in plain[1:]} & {b for b, _ in offset[1:]}  # every cut moved
    assert len(offset[0][1]) <= 250 < len(plain[0][1])
    # a first paragraph longer than 250 stands alone, as a paragraph longer than max_chars does in chapter_chunks
    long = Chapter("ch0006", 6, [("p1", "汀" * 300), ("p2", "城" * 10)], "", "h")
    assert offset_chunks(long, 500) == [(0, "汀" * 300), (301, "城" * 10)] and len(chapter_chunks(long, 500)) == 1
    short = Chapter("ch0007", 7, [("p1", "汀城")], "", "h")
    assert offset_chunks(short, 500) == chapter_chunks(short, 500) == [(0, "汀城")]
    assert tagger.chapter_chunking(ch, 500) == plain and tagger.chapter_chunking(ch, 500, "offset250") == offset
    with pytest.raises(ValueError, match="chunking"):
        tagger.chapter_chunking(ch, 500, "overlap250")


def test_merge_chunkings_takes_the_token_farther_from_a_chunk_edge_and_counts_unmatched_tokens():
    def chunk(spans, value):
        o = np.asarray(spans, dtype=np.int64)
        return o, np.full((len(o), len(TAGS)), value, dtype=np.float32)

    # text "abcd\nefgh": the primary chunking cuts at the break, the other has one chunk with "ef" as one word piece
    primary = {("c", 0): chunk([(0, 1), (1, 2), (2, 3), (3, 4)], 0.0), ("c", 1): chunk([(5, 6), (6, 7), (7, 8), (8, 9)], 0.0)}
    other = {("c", 0): chunk([(0, 1), (1, 2), (2, 3), (3, 4), (5, 7), (7, 8), (8, 9)], 1.0)}
    stats = Counter()
    merged = merge_chunkings(primary, other, stats)
    assert merged[("c", 0)][1][:, 0].tolist() == [0, 0, 1, 1]  # distances 0,1,1,0 against 0,1,2,3: ties stay primary
    assert merged[("c", 1)][1][:, 0].tolist() == [0, 0, 0, 0]  # (5,6), (6,7) have no match; (7,8) 1 vs 1 and (8,9) 0 vs 0 tie
    assert merged[("c", 0)][0].tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]] and set(merged) == set(primary)
    assert dict(stats) == {"tokens_taken_from_offset_chunking": 2, "tokens_without_a_match": 2}
    assert primary[("c", 0)][1].sum() == 0  # the primary's arrays are not written to
    # a token without characters keeps its own and is not counted; a chapter the other chunking lacks keeps everything
    stats = Counter()
    assert merge_chunkings({("d", 0): chunk([(0, 1), (1, 1)], 0.0)}, other, stats)[("d", 0)][1][:, 0].tolist() == [0, 0]
    assert dict(stats) == {"tokens_taken_from_offset_chunking": 0, "tokens_without_a_match": 1}
    assert tagger.edge_distance(5).tolist() == [0, 1, 2, 1, 0] and tagger.edge_distance(0).tolist() == []


def test_predict_chunking_variants_and_the_default_path_is_unchanged(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    model = tiny_model(tok)
    ch = Chapter("ch0005", 5, paragraphs(10), "", "h")  # 759 characters: paragraph500 cuts once, offset250 twice
    cfg = {**CFG, "max_tokens": 64, "stride": 16}  # the tiny model has 64 positions: every chunk gets several windows
    key = lambda rs: [(r["chapter_id"], r["start"], r["end"], r["text"], r["type"], r["score"]) for r in rs]
    plain, plain_stats = predict(model, tok, [ch, DEV], cfg, torch.device("cpu"))
    again, again_stats = predict(model, tok, [ch, DEV], cfg, torch.device("cpu"), chunking="paragraph500")
    assert key(plain) == key(again)  # the default and the option at its default: the same rows, scores included
    drop = lambda s: {k: v for k, v in s.items() if k != "seconds"}
    assert drop(plain_stats) == drop(again_stats)
    assert plain_stats["chunking"] == "paragraph500" and plain_stats["chunks_by_chunking"] == {"paragraph500": 3}
    assert plain_stats["chunks"] == 3 and "tokens_taken_from_offset_chunking" not in plain_stats
    offset, offset_stats = predict(model, tok, [ch, DEV], cfg, torch.device("cpu"), chunking="offset250")
    assert offset_stats["chunking"] == "offset250" and offset_stats["chunks_by_chunking"] == {"offset250": 4}
    assert offset_stats["chunks"] == 4 and "tokens_without_a_match" not in offset_stats
    both, both_stats = predict(model, tok, [ch, DEV], cfg, torch.device("cpu"), chunking="overlap250")
    assert both_stats["chunking"] == "overlap250" and both_stats["chunks_by_chunking"] == {"paragraph500": 3, "offset250": 4}
    assert both_stats["chunks"] == 7 and both_stats["windows"] == plain_stats["windows"] + offset_stats["windows"]
    assert both_stats["tokens_without_a_match"] == 0  # one character, one token: every token has its match
    assert 0 < both_stats["tokens_taken_from_offset_chunking"] < 750  # the tokens near the paragraph500 cut, not all
    for rows in (offset, both):
        for r in rows:
            text = {"ch0005": ch, "ch0002": DEV}[r["chapter_id"]].text
            assert text[r["start"]:r["end"]] == r["text"] and PARAGRAPH_SEP not in r["text"]
    with pytest.raises(ValueError, match="chunking"):
        predict(model, tok, [ch], cfg, torch.device("cpu"), chunking="offset")


def test_partial_label_loss_is_cross_entropy_for_one_label_and_ignores_masked_tokens():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, len(TAGS), requires_grad=True)
    labels = torch.randint(0, len(TAGS), (2, 5))
    one_hot = F.one_hot(labels, len(TAGS)).bool()
    assert torch.allclose(partial_label_nll(logits, one_hot), F.cross_entropy(logits.reshape(-1, len(TAGS)), labels.reshape(-1)))
    allowed = one_hot.clone()
    allowed[0, 1] = True  # all 13 labels: masked, no information
    allowed[1, 3] = False  # no label: a special or ignored token
    keep = torch.ones(2, 5, dtype=torch.bool)
    keep[0, 1] = keep[1, 3] = False
    loss = partial_label_nll(logits, allowed)
    assert torch.allclose(loss, F.cross_entropy(logits[keep], labels[keep]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[0, 1].any() and not logits.grad[1, 3].any()
    two = torch.zeros(1, 1, len(TAGS), dtype=torch.bool)
    two[0, 0, [0, 1]] = True
    p = logits[0, 0].softmax(-1)
    assert torch.allclose(partial_label_nll(logits[:1, :1], two), -torch.log(p[0] + p[1]))
    assert partial_label_nll(logits, torch.ones_like(one_hot)).item() == 0.0


def test_decode_repairs_i_without_b_and_splits_at_a_paragraph_break():
    text = "徐获拿回音铃\n汀城到了"
    tags = ["I-person", "I-person", "O", "B-item", "I-item", "I-person", "I-person", "I-person", "B-location", "B-location"]
    probs = [0.9, 0.8, 0.99, 0.7, 0.6, 0.5, 0.95, 0.85, 0.4, 0.3]
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 11)]
    spans, counts = decode_bio(tags, probs, offsets, text)
    assert spans == [(0, 2, "person", 0.8), (3, 5, "item", 0.6), (5, 6, "person", 0.5), (7, 9, "person", 0.85),
                     (9, 10, "location", 0.4), (10, 11, "location", 0.3)]
    assert counts == {"decode_i_without_b_repaired": 2, "decode_split_at_paragraph_break": 1}


def test_train_and_predict_end_to_end_on_tiny_chapters(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    row = {"spans": [span(0, 2, "徐获", "person"), span(4, 7, "回音铃", "item"), span(9, 12, "唐医生", "person"),
                     span(14, 16, "汀城", "location")], "uncertain": []}
    windows, _ = training_windows(tok, [(TRAIN, row)], CFG)
    model = tiny_model(tok)
    gold = {"ch0002": {"spans": {(0, 3, "person"), (5, 7, "person"), (14, 17, "item")}, "unscored": []}}
    cfg = {**CFG, "batch_size": 1, "max_steps": 3}
    result = fit(model, tok, windows, {"chapters": [DEV], "gold": gold, "ids": ["ch0002"]}, cfg, 0, torch.device("cpu"),
                 log=lambda *_: None)
    assert (result["steps_run"], len(result["curve"])) == (3, 3)
    rows, stats = predict(model, tok, [DEV], cfg, torch.device("cpu"))
    assert rows and stats["windows"] == 1
    for r in rows:
        assert DEV.text[r["start"]:r["end"]] == r["text"] and "\n" not in r["text"]
        assert r["type"] in ENTITY_TYPES and 0 < r["score"] <= 1
    key = lambda rs: sorted((r["start"], r["end"], r["type"]) for r in rs)
    assert key(rows) == key(result["best"]["rows"])  # the best state is the one loaded back


def test_grad_accumulation_one_is_the_same_run_and_two_gives_the_same_losses_up_to_float_tolerance(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    train2 = Chapter("ch0003", 3, [("p1", "唐医生拿出回音铃。")], "", "h")
    pairs = [(TRAIN, {"spans": [span(0, 2, "徐获", "person"), span(4, 7, "回音铃", "item")], "uncertain": []}),
             (train2, {"spans": [span(0, 3, "唐医生", "person")], "uncertain": [region(5, 8, "回音铃", ["item"])]})]
    windows, _ = training_windows(tok, pairs, CFG)
    assert len(windows) == 2 and tagger.informative_tokens(windows[0]) != tagger.informative_tokens(windows[1])
    gold = {"ch0002": {"spans": {(0, 3, "person"), (5, 7, "person"), (14, 17, "item")}, "unscored": []}}
    dev = {"chapters": [DEV], "gold": gold, "ids": ["ch0002"]}
    key = lambda rs: sorted((r["start"], r["end"], r["type"], round(r["score"], 6)) for r in rs)

    def run(extra, dropout):
        model = tiny_model(tok, dropout=dropout)
        cfg = {**CFG, "batch_size": 2, "max_epochs": 4, "max_steps": 4, **extra}
        result = fit(model, tok, windows, dev, cfg, 0, torch.device("cpu"), log=lambda *_: None)
        rows, _ = predict(model, tok, [DEV], cfg, torch.device("cpu"))
        return result, key(rows), [float(p.detach().sum()) for p in model.parameters()]

    plain, plain_rows, plain_params = run({}, 0.1)
    one, one_rows, one_params = run({"grad_accumulation": 1}, 0.1)
    assert [p["train_loss"] for p in plain["curve"]] == [p["train_loss"] for p in one["curve"]]
    assert plain_rows == one_rows and plain_params == one_params  # bit-identical with the key absent and with 1
    assert (one["grad_accumulation"], one["micro_batch_size"]) == (1, 2)
    # dropout off: two micro-batches of one window must give the whole batch's loss and gradient up to float error
    whole, _, whole_params = run({"grad_accumulation": 1}, 0.0)
    halves, _, halves_params = run({"grad_accumulation": 2}, 0.0)
    assert (halves["grad_accumulation"], halves["micro_batch_size"]) == (2, 1)
    assert len(whole["curve"]) == len(halves["curve"]) == 4
    for a, b in zip(whole["curve"], halves["curve"]):
        assert abs(a["train_loss"] - b["train_loss"]) < 1e-4
    for a, b in zip(whole_params, halves_params):
        assert abs(a - b) < 1e-3
    # the weights follow the informative tokens; with nothing informative they are even
    parts = tagger.micro_batches(windows, 2)
    assert [len(p) for p, _ in parts] == [1, 1] and abs(sum(w for _, w in parts) - 1.0) < 1e-9
    assert parts[0][1] == tagger.informative_tokens(windows[0]) / (tagger.informative_tokens(windows[0]) + tagger.informative_tokens(windows[1]))
    assert tagger.micro_batches(windows, 1) == [(windows, 1.0)]
    blank = [{**w, "allowed": np.ones_like(w["allowed"])} for w in windows]
    assert [w for _, w in tagger.micro_batches(blank, 2)] == [0.5, 0.5]
    with pytest.raises(ValueError, match="grad_accumulation"):
        fit(tiny_model(tok), tok, windows, dev, {**CFG, "batch_size": 2, "grad_accumulation": 3}, 0, torch.device("cpu"), log=lambda *_: None)


def point(step, f1, non_person=0.5):
    return {"step": step, "micro": {"f1": f1}, "non_person_f1": non_person}


def result(f1s, best_index, stopped, non_person=0.5):
    curve = [point(i + 1, f) for i, f in enumerate(f1s)]
    best = {"step": best_index + 1, "scores": {"non_person_micro": {"f1": non_person}}}
    return {"curve": curve, "best": best, "stopped_early": stopped}


def test_sufficiency_rule_from_next_route():
    s = tagger.sufficiency(result([0.5, 0.8, 0.9, 0.9, 0.896], 2, False))
    assert s["sufficient"] and not s["best_in_last_two_evals"] and s["last_three_f1_range"] == 0.004 and s["evals"] == 5
    assert s["best_eval_index"] == 2 and not s["stopped_by_patience"] and s["non_person_f1_at_best"] == 0.5
    assert not tagger.sufficiency(result([0.5, 0.8, 0.9, 0.9, 0.89], 2, False))["sufficient"]  # the last three move by 0.01
    assert not tagger.sufficiency(result([0.5, 0.8, 0.9, 0.9, 0.91], 4, False))["sufficient"]  # best is the last evaluation
    assert not tagger.sufficiency(result([0.5, 0.8, 0.9, 0.91, 0.9], 3, False))["sufficient"]  # best is the second to last
    assert tagger.sufficiency(result([0.9, 0.8, 0.7, 0.6], 0, True))["sufficient"]  # stopped by patience
    assert not tagger.sufficiency(result([0.9, 0.8, 0.7, 0.6], 0, True, non_person=0.0))["sufficient"]  # non-person F1 0
    assert tagger.sufficiency(result([0.9, 0.8, 0.7, 0.6], 0, True, non_person=None))["non_person_f1_at_best_is_zero"]
    short = tagger.sufficiency(result([0.5, 0.9], 1, False))
    assert short["last_three_f1_range"] is None and not short["sufficient"]
    assert "0.005" in short["rule"] and "patience" in short["rule"]


def test_sealed_test2_chapters_reads_the_seal_file(tmp_path):
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"order": [{"chapter_id": "ch1262"}, {"chapter_id": "ch1345"}, {"chapter_id": "ch1207"}]}), encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed", "order": str(order), "n": 2}), encoding="utf-8")
    assert tagger.sealed_test2_chapters(seal) == tagger.test2_chapters(seal) == {"ch1262", "ch1345"}
    assert tagger.sealed_test2_chapters({"status": "sealed", "order": str(order), "n": 3}) == {"ch1262", "ch1345", "ch1207"}
    # open: the chapters are still test2's (the run-id rule needs them), only the refusal is lifted
    assert tagger.sealed_test2_chapters({"status": "open", "order": str(order), "n": 2}) == frozenset()
    assert tagger.test2_chapters({"status": "open", "order": str(order), "n": 2}) == {"ch1262", "ch1345"}
    assert tagger.read_seal(seal)[0]["status"] == "sealed" and tagger.read_seal({"status": "open"})[0]["status"] == "open"
    for f in (tagger.sealed_test2_chapters, tagger.test2_chapters, tagger.read_seal):
        with pytest.raises(SystemExit, match="neither"):
            f({"status": "unsealed", "order": str(order), "n": 2})
    for f in (tagger.sealed_test2_chapters, tagger.test2_chapters):
        with pytest.raises(SystemExit, match="fewer"):
            f({"status": "sealed", "order": str(order), "n": 4})
    # the repository's own seal: 40 chapters of the second test, none of them before chapter 996, whatever its status
    real = tagger.test2_chapters()
    assert len(real) == 40 and all(int(c[2:]) >= 996 for c in real)
    assert tagger.sealed_test2_chapters() == (real if tagger.read_seal()[0]["status"] == "sealed" else frozenset())
    # and its formal test list (the vote's material guard): 60 chapters, all in 781-990, none of them dev2
    exam = tagger.exam_chapters()
    assert len(exam) == 60 and all(780 < int(c[2:]) <= 990 for c in exam) and not exam & tagger.dev2_chapters()
    # and its dev list: 31 chapters, all at or before 780, so disjoint from dev2, the test and test2
    dev = tagger.dev_chapters()
    assert len(dev) == 31 and all(int(c[2:]) <= 780 for c in dev) and not dev & (exam | tagger.dev2_chapters() | real)


def training_setup(tmp_path, monkeypatch):
    """A tiny backbone, snapshot, weak build and dev answers under tmp_path; returns (cfg, snapshot)."""
    tok = tiny_tokenizer(tmp_path / "vocab")
    tiny_model(tok).save_pretrained(tmp_path / "backbone")
    tok.save_pretrained(tmp_path / "backbone")
    train2 = Chapter("ch0003", 3, [("p1", "汀城到了。")], "", "h4")
    snapshot = Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in (TRAIN, DEV, train2, LATE)}, [], [])
    rows = [{"chapter_id": "ch0001", "number": 1, "split": "train", "spans": [span(0, 2, "徐获", "person")],
             "uncertain": [region(4, 7, "回音铃", ["item"])]},
            {"chapter_id": "ch0002", "number": 2, "split": "dev", "spans": [], "uncertain": []},
            {"chapter_id": "ch0003", "number": 3, "split": "train", "spans": [], "uncertain": []}]
    write_build(tmp_path, rows)
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "gold" / "dev.json").write_text(json.dumps(gold_export(DEV, [
        {"start": 5, "end": 7, "text": "徐获", "type": "person", "pending": False}]), ensure_ascii=False), encoding="utf-8")
    cfg = {**CFG, "snapshot": "snap-test", "weak_build": "b1", "dev_set": "dev_set.json", "gold_dev": "data/gold/dev.json",
           "backbone": {"id": "tiny", "path": "backbone"}, "max_steps": 2, "deterministic": False}
    monkeypatch.setattr(tagger, "REPO_ROOT", tmp_path)
    (tmp_path / "dev_set.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tagger, "chapter_order", lambda dev_cfg: ["ch0002"])
    monkeypatch.setattr(tagger, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(tagger, "environment", lambda: {"python": "test"})
    return cfg, snapshot, rows


def test_init_from_run_loads_a_finished_run_of_the_same_backbone_and_sufficiency_is_recorded(tmp_path, monkeypatch):
    cfg, snapshot, _ = training_setup(tmp_path, monkeypatch)
    metrics = tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    s = metrics["sufficiency"]
    assert set(s) == {"best_eval_index", "evals", "best_in_last_two_evals", "last_three_f1_range", "stopped_by_patience",
                      "non_person_f1_at_best", "non_person_f1_at_best_is_zero", "sufficient", "rule"}
    assert s["evals"] == 2 and s["best_in_last_two_evals"] and not s["sufficient"]  # two evaluations cannot satisfy the rule
    assert metrics["schedule"]["grad_accumulation"] == 1 and metrics["schedule"]["micro_batch_size"] == 2
    assert json.loads((tmp_path / "runs" / "r1" / "config.json").read_text(encoding="utf-8"))["init_from"] is None
    # "{seed}" names the run of the same seed; the record keeps that run's id and config.json sha256
    cfg2 = {**cfg, "init_from_run": "runs/r{seed}"}
    tagger.train(cfg2, "S2", 1, "r2", snapshot=snapshot, log=lambda *_: None)
    rec = json.loads((tmp_path / "runs" / "r2" / "config.json").read_text(encoding="utf-8"))
    assert rec["init_from"]["run_id"] == "r1" and rec["init_from"]["pattern"] == "runs/r{seed}"
    assert rec["init_from"]["config_sha256"] == sha256(tmp_path / "runs" / "r1" / "config.json")
    assert rec["init_from"]["checkpoint"] == "runs/r1/checkpoints/best" and rec["init_from"]["backbone_id"] == "tiny"
    assert tmp_path.name not in json.dumps(rec)
    with pytest.raises(SystemExit, match="not a finished"):  # no such run
        tagger.train({**cfg, "init_from_run": "runs/r9"}, "S2", 1, "r3", snapshot=snapshot, log=lambda *_: None)
    with pytest.raises(SystemExit, match="backbone"):
        tagger.train({**cfg2, "backbone": {"id": "other-tiny", "path": "backbone"}}, "S2", 1, "r3", snapshot=snapshot, log=lambda *_: None)
    assert not (tmp_path / "runs" / "r3").exists()


def test_predict_refuses_sealed_test2_chapters_whatever_the_run_id(tmp_path, monkeypatch):
    cfg, snapshot, _ = training_setup(tmp_path, monkeypatch)
    tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"order": [{"chapter_id": "ch0781"}, {"chapter_id": "ch0003"}]}), encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed", "order": str(order), "n": 1}), encoding="utf-8")
    monkeypatch.setattr(tagger, "SEAL_FILE", seal)
    for run_id in ("r2-test", "r2-preann", "r2-final-test2"):
        with pytest.raises(SystemExit, match="sealed test2"):
            tagger.predict_run("runs/r1", ["ch0781"], run_id, snapshot=snapshot)
        assert not (tmp_path / "runs" / run_id).exists()
    tagger.predict_run("runs/r1", ["ch0003"], "r2-x", snapshot=snapshot)  # not sealed (n = 1)
    assert json.loads((tmp_path / "runs" / "r2-x" / "metrics.json").read_text(encoding="utf-8"))["test2_chapters"] == 0
    seal.write_text(json.dumps({"status": "open", "order": str(order), "n": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="test2"):  # open: a test2 chapter needs "test2" in the run id; "test" alone is not enough
        tagger.predict_run("runs/r1", ["ch0781"], "r3-test", snapshot=snapshot)
    assert not (tmp_path / "runs" / "r3-test").exists()
    tagger.predict_run("runs/r1", ["ch0781"], "r3-test2", snapshot=snapshot)  # "test2" also satisfies the test range's "test"
    assert (tmp_path / "runs" / "r3-test2" / "predictions.jsonl").exists()


def test_predict_needs_test2_in_the_run_id_once_the_seal_is_open(tmp_path, monkeypatch):
    """Stage 3: the seal is opened, test2 (chapters after 990) is predicted into runs whose id names test2; the other
    chapters after 990 still need "preann"; while sealed nothing changes."""
    cfg, snapshot, _ = training_setup(tmp_path, monkeypatch)
    tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    second = Chapter("ch0996", 996, [("p1", "徐获到了汀城。")], "", "h5")
    other = Chapter("ch0997", 997, [("p1", "唐医生到了。")], "", "h6")
    later = Snapshot("snap-later", None, {"source_commit": "d"}, {**snapshot.chapters, "ch0996": second, "ch0997": other}, [], [])
    monkeypatch.setattr(tagger, "load_snapshot", lambda sid: {"snap-test": snapshot, "snap-later": later}[sid])
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"order": [{"chapter_id": "ch0996"}, {"chapter_id": "ch0997"}]}), encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed", "order": str(order), "n": 1}), encoding="utf-8")
    monkeypatch.setattr(tagger, "SEAL_FILE", seal)
    assert tagger.test2_chapters() == tagger.sealed_test2_chapters() == {"ch0996"}
    with pytest.raises(SystemExit, match="sealed test2"):  # sealed: refused whatever the run id, even one naming test2
        tagger.predict_run("runs/r1", ["ch0996"], "r2-predict-test2", snapshot=later)
    assert not (tmp_path / "runs" / "r2-predict-test2").exists()
    seal.write_text(json.dumps({"status": "open", "order": str(order), "n": 1}), encoding="utf-8")
    assert tagger.test2_chapters() == {"ch0996"} and tagger.sealed_test2_chapters() == frozenset()
    for bad in ("r2-preann", "r2-test", "r2-preann-test"):  # open: a test2 chapter needs "test2"; "preann" or "test" is not enough
        with pytest.raises(SystemExit, match="must contain 'test2'"):
            tagger.predict_run("runs/r1", ["ch0996"], bad, snapshot=later)
        assert not (tmp_path / "runs" / bad).exists()
    for ids in (["ch0997"], ["ch0996", "ch0997"]):  # a chapter after 990 outside test2 still needs "preann"
        with pytest.raises(SystemExit, match="must contain 'preann'"):
            tagger.predict_run("runs/r1", ids, "r2-test2", snapshot=later)
    assert not (tmp_path / "runs" / "r2-test2").exists()
    m = tagger.predict_run("runs/r1", ["ch0996"], "r2-predict-test2", snapshot=later)
    assert (m["chapters"], m["chapters_after_780"], m["chapters_after_990"], m["test2_chapters"]) == (1, 1, 1, 1)
    rec = json.loads((tmp_path / "runs" / "r2-predict-test2" / "config.json").read_text(encoding="utf-8"))
    assert (rec["test2_chapters"], rec["test2_seal_status"]) == (1, "open")
    assert rec["snapshot"]["id"] == "snap-later" and rec["snapshot_lineage"]["model_snapshot"] == "snap-test"
    m = tagger.predict_run("runs/r1", ["ch0996", "ch0997"], "r3-test2-preann", snapshot=later)
    assert (m["chapters_after_990"], m["test2_chapters"]) == (2, 1)
    m = tagger.predict_run("runs/r1", ["ch0997"], "r4-preann", snapshot=later)  # no test2 chapter: as before the seal existed
    assert m["test2_chapters"] == 0
    assert json.loads((tmp_path / "runs" / "r4-preann" / "config.json").read_text(encoding="utf-8"))["test2_seal_status"] == "open"


def test_predict_run_records_the_chunking_and_a_variant_needs_c1_in_the_run_id(tmp_path, monkeypatch):
    cfg, snapshot, _ = training_setup(tmp_path, monkeypatch)
    tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    runs = tmp_path / "runs"
    tagger.predict_run("runs/r1", ["ch0003"], "r2-a", snapshot=snapshot)
    tagger.predict_run("runs/r1", ["ch0003"], "r2-b", snapshot=snapshot, chunking="paragraph500")
    assert sha256(runs / "r2-a" / "predictions.jsonl") == sha256(runs / "r2-b" / "predictions.jsonl")
    for rid in ("r2-a", "r2-b"):
        assert json.loads((runs / rid / "config.json").read_text(encoding="utf-8"))["chunking"] == "paragraph500"
        m = json.loads((runs / rid / "metrics.json").read_text(encoding="utf-8"))
        assert m["chunking"] == "paragraph500" and m["predict_stats"]["chunks_by_chunking"] == {"paragraph500": 1}
    for chunking in ("offset250", "overlap250"):
        with pytest.raises(SystemExit, match="c1"):
            tagger.predict_run("runs/r1", ["ch0003"], f"r3-{chunking}", snapshot=snapshot, chunking=chunking)
        assert not (runs / f"r3-{chunking}").exists()
    with pytest.raises(SystemExit, match="chunking must be"):
        tagger.predict_run("runs/r1", ["ch0003"], "r3-c1", snapshot=snapshot, chunking="offset")
    m = tagger.predict_run("runs/r1", ["ch0003"], "r3-c1-overlap", snapshot=snapshot, chunking="overlap250")
    assert m["chunking"] == "overlap250" and m["predict_stats"]["chunks_by_chunking"] == {"paragraph500": 1, "offset250": 1}
    assert m["predict_stats"]["tokens_without_a_match"] == 0 and m["predict_stats"]["tokens_taken_from_offset_chunking"] == 0
    assert json.loads((runs / "r3-c1-overlap" / "config.json").read_text(encoding="utf-8"))["chunking"] == "overlap250"
    m = tagger.predict_run("runs/r1", ["ch0003"], "r3-c1-offset", snapshot=snapshot, chunking="offset250")
    assert m["chunking"] == "offset250" and m["predict_stats"]["chunks_by_chunking"] == {"offset250": 1}


def test_training_rows_refuse_dev2_chapters_by_name(tmp_path, monkeypatch):
    real_file = tagger.DEV2_FILE  # the repository's own list, before the test points the module at a temporary one
    cfg, snapshot, rows = training_setup(tmp_path, monkeypatch)
    dev2 = tmp_path / "dev2.json"
    dev2.write_text(json.dumps({"chapters": ["ch0003"]}), encoding="utf-8")
    monkeypatch.setattr(tagger, "DEV2_FILE", dev2)
    assert tagger.dev2_chapters() == {"ch0003"}
    with pytest.raises(SystemExit, match="dev2 decision chapter"):
        tagger.training_rows(rows, snapshot, {"ch0002"})
    with pytest.raises(SystemExit, match="dev2 decision chapter"):
        tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    assert not (tmp_path / "runs" / "r1").exists()
    assert len(tagger.training_rows(rows, snapshot, {"ch0002"}, dev2_ids=frozenset())) == 2
    # the repository's own dev2 list: 25 chapters, all after 780
    real = tagger.dev2_chapters(real_file)
    assert len(real) == 25 and all(int(c[2:]) > 780 for c in real)


def gold_export(chapter, annotations):
    return {"format": FORMAT, "set": "dev", "snapshot_id": "snap-test", "guide_version": "1.1", "precedents": [],
            "chapters": {chapter.chapter_id: {
                "content_hash": chapter.content_hash, "text_sha256": hashlib.sha256(chapter.text.encode("utf-8")).hexdigest(),
                "status": "done", "provenance": "manual", "checks": {"a": True}, "annotations": annotations}}}


def write_build(root, rows, setting="S2"):
    """Labels of weak build b1 and its record runs/b1/config.json, matching them as weak.build would."""
    labels = root / "data" / "weak_labels" / "b1" / f"labels_{setting}.jsonl"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    record = root / "runs" / "b1" / "config.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"snapshot": {"id": "snap-test"}, "labels": {setting: {"sha256": sha256(labels)}},
                                  "split": {"train_ids_sha256": ids_sha256(r["chapter_id"] for r in rows if r["split"] == "train")}}),
                      encoding="utf-8")
    return labels, record


def test_train_writes_numbers_only_records_and_predict_keeps_late_chapters_in_test_runs(tmp_path, monkeypatch):
    tok = tiny_tokenizer(tmp_path / "vocab")
    tiny_model(tok).save_pretrained(tmp_path / "backbone")
    tok.save_pretrained(tmp_path / "backbone")
    train2 = Chapter("ch0003", 3, [("p1", "汀城到了。")], "", "h4")
    snapshot = Snapshot("snap-test", None, {"source_commit": "c"}, {c.chapter_id: c for c in (TRAIN, DEV, train2, LATE)}, [], [])
    rows = [{"chapter_id": "ch0001", "number": 1, "split": "train", "spans": [span(0, 2, "徐获", "person")],
             "uncertain": [region(4, 7, "回音铃", ["item"])]},
            {"chapter_id": "ch0002", "number": 2, "split": "dev", "spans": [], "uncertain": []},
            {"chapter_id": "ch0003", "number": 3, "split": "train", "spans": [], "uncertain": []}]
    labels, record = write_build(tmp_path, rows)
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "gold" / "dev.json").write_text(json.dumps(gold_export(DEV, [
        {"start": 5, "end": 7, "text": "徐获", "type": "person", "pending": False}]), ensure_ascii=False), encoding="utf-8")
    cfg = {**CFG, "snapshot": "snap-test", "weak_build": "b1", "dev_set": "dev_set.json", "gold_dev": "data/gold/dev.json",
           "backbone": {"id": "tiny", "path": "backbone"}, "max_steps": 2, "deterministic": False}
    monkeypatch.setattr(tagger, "REPO_ROOT", tmp_path)
    (tmp_path / "dev_set.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tagger, "chapter_order", lambda dev_cfg: ["ch0002"])
    monkeypatch.setattr(tagger, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(tagger, "environment", lambda: {"python": "test"})
    metrics = tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot, log=lambda *_: None)
    run = tmp_path / "runs" / "r1"
    for name in ("config.json", "metrics.json"):
        assert not set((run / name).read_text(encoding="utf-8")) & set("徐获回音铃唐医生汀城")  # no novel text
    assert tmp_path.name not in (run / "config.json").read_text(encoding="utf-8")  # no absolute local path
    record_out = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert record_out["backbone"]["path"] == "backbone" and "local_path" not in record_out["backbone"]
    assert record_out["weak_labels"]["train_ids_sha256"] == ids_sha256(["ch0003", "ch0001"])  # the weak build's rule
    assert (run / "predictions_dev.jsonl").exists() and (run / "checkpoints" / "best" / "config.json").exists()
    assert metrics["train"]["chapters"] == 2 and metrics["dev"]["saved_state_gives_the_same_predictions"]
    assert metrics["train"]["tokens_partial"] == 3
    with pytest.raises(SystemExit):
        tagger.train(cfg, "S2", 1, "r1", snapshot=snapshot)
    assert tagger.summarize_runs(["runs/r1"], "runs/r1-summary")["settings"]["S2"]["seeds"] == [1]
    with pytest.raises(SystemExit, match="test"):
        tagger.predict_run("runs/r1", ["ch0781"], "r2-late", snapshot=snapshot)
    tagger.predict_run("runs/r1", ["ch0781"], "r2-test", snapshot=snapshot)
    assert (tmp_path / "runs" / "r2-test" / "predictions.jsonl").exists()
    # "runs/<id>" is read as "<id>"; a nested or relative id is refused before anything is written
    tagger.predict_run("r1", ["ch0781"], "runs/r5-test", snapshot=snapshot)
    assert (tmp_path / "runs" / "r5-test" / "predictions.jsonl").exists() and not (tmp_path / "runs" / "runs").exists()
    for bad in ("a/test", "../x-test", "runs/a/x-test"):
        with pytest.raises(SystemExit, match="one folder name"):
            tagger.predict_run("runs/r1", ["ch0781"], bad, snapshot=snapshot)
    # a later export: chapters after 990 need "preann" (not "test"); the shared chapters must have the same text
    beyond = Chapter("ch0991", 991, [("p1", "徐获到了汀城。")], "", "h5")
    later = Snapshot("snap-later", None, {"source_commit": "d"}, {**snapshot.chapters, "ch0991": beyond}, [], [])
    monkeypatch.setattr(tagger, "load_snapshot", lambda sid: {"snap-test": snapshot, "snap-later": later}[sid])
    for bad_id, word in (("r6-test", "preann"), ("r6-late", "preann")):
        with pytest.raises(SystemExit, match=word):
            tagger.predict_run("runs/r1", ["ch0991"], bad_id, snapshot=later)
    with pytest.raises(SystemExit, match="test"):
        tagger.predict_run("runs/r1", ["ch0781", "ch0991"], "r6-preann", snapshot=later)
    metrics_beyond = tagger.predict_run("runs/r1", ["ch0991"], "r6-preann", snapshot=later)
    assert (metrics_beyond["chapters_after_780"], metrics_beyond["chapters_after_990"]) == (1, 1)
    rec = json.loads((tmp_path / "runs" / "r6-preann" / "config.json").read_text(encoding="utf-8"))
    assert rec["snapshot"]["id"] == "snap-later" and rec["snapshot_lineage"] == {
        "model_snapshot": "snap-test", "shared_chapters": 4, "shared_text_identical": True}
    assert json.loads((tmp_path / "runs" / "r2-test" / "config.json").read_text(encoding="utf-8"))["snapshot_lineage"] is None
    changed = Snapshot("snap-later", None, {}, {**later.chapters, "ch0781": Chapter("ch0781", 781, [("p1", "改过的字。")])}, [], [])
    with pytest.raises(SystemExit, match="not a later export"):
        tagger.predict_run("runs/r1", ["ch0991"], "r7-preann", snapshot=changed)
    assert not (tmp_path / "runs" / "r7-preann").exists()
    with pytest.raises(SystemExit, match="one folder name"):
        tagger.train(cfg, "S2", 1, "sub/x", snapshot=snapshot)
    assert not any((tmp_path / p).exists() for p in ("runs/a", "runs/sub", "x-test"))

    # the weak build's record must exist and match the labels file and its training chapters
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the file weak build"):
        tagger.train(cfg, "S2", 1, "r3", snapshot=snapshot)
    labels, record = write_build(tmp_path, rows)
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")),
                                  "split": {"train_ids_sha256": ids_sha256(["ch0001"])}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="training chapters differ"):
        tagger.train(cfg, "S2", 1, "r3", snapshot=snapshot)
    record.unlink()
    with pytest.raises(SystemExit, match="no record of weak build"):
        tagger.train(cfg, "S2", 1, "r3", snapshot=snapshot)
    # a training row after chapter 780 stops the run, and so does a dev chapter marked "train"
    write_build(tmp_path, [{"chapter_id": "ch0781", "number": 781, "split": "train", "spans": [], "uncertain": []}])
    with pytest.raises(SystemExit, match="780"):
        tagger.train(cfg, "S2", 1, "r3", snapshot=snapshot)
    write_build(tmp_path, [{"chapter_id": "ch0002", "number": 2, "split": "train", "spans": [], "uncertain": []}])
    with pytest.raises(SystemExit, match="not a dev chapter"):
        tagger.train(cfg, "S2", 1, "r3", snapshot=snapshot)
    assert not (tmp_path / "runs" / "r3").exists()
    # a dev set past chapter 780 (with answers for it) is refused before any answer file is read
    write_build(tmp_path, rows)
    (tmp_path / "data" / "gold" / "late.json").write_text(json.dumps(gold_export(LATE, [
        {"start": 0, "end": 2, "text": "徐获", "type": "person", "pending": False}]), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tagger, "chapter_order", lambda dev_cfg: ["ch0781"])
    with pytest.raises(SystemExit, match="every dev chapter"):
        tagger.train({**cfg, "gold_dev": "data/gold/late.json"}, "S2", 1, "r4", snapshot=snapshot)
    assert not (tmp_path / "runs" / "r4").exists()


def test_run_ids_are_one_folder_name_under_runs():
    assert tagger.plain_run_id("r1") == tagger.plain_run_id("runs/r1") == tagger.plain_run_id("r1/") == "r1"
    for bad in ("", ".", "..", "runs/..", "a/test", "../x-test", "runs/a/x-test", "/abs-test", "C:/x-test"):
        with pytest.raises(SystemExit, match="one folder name"):
            tagger.plain_run_id(bad)
    with pytest.raises(SystemExit):
        tagger.plain_run_id(tagger.default_run_id("S2", 1, "a/b"))  # a sweep --suffix with a separator


def test_deterministic_mode_is_strict(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # restored after the test
    try:
        settings = tagger.seed_everything(1, True)
        assert settings["deterministic_algorithms"] and not settings["deterministic_warn_only"]
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        tagger.seed_everything(0, False)
    assert not torch.are_deterministic_algorithms_enabled()


def test_download_fetches_only_the_listed_files_and_training_never_downloads(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []

    def fake_download(repo_id, revision=None, allow_patterns=None, local_files_only=False):
        calls.append((repo_id, revision, tuple(allow_patterns), local_files_only, tagger.os.environ.get("DISABLE_SAFETENSORS_CONVERSION")))
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 10)
        (tmp_path / "vocab.txt").write_bytes(b"[PAD]\n")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    cfg = {"backbone": {"id": "org/model", "revision": "abc123"}}
    assert tagger.download(cfg) == (tmp_path, 16)
    assert calls == [("org/model", "abc123", tuple(tagger.BACKBONE_FILES), False, "1")]
    with pytest.raises(SystemExit, match="not downloaded"):
        tagger.backbone_path(cfg)
    assert calls[-1][3] is True
