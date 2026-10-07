import pytest

from novel_ner.evaluate import dictionary_predictions, write_jsonl
from novel_ner.snapshot import Chapter, Snapshot
from novel_ner.weak import (TAGS, active, by_list_version, char_allowed, clean_dictionary, compare_rows, coverage,
                            dictionary_row_hits, label_bytes, label_chapter, mine, occurrences, proposals, resolve, resources,
                            split_ids, truncation_rules, undated_counts, undated_record)

CFG = {"min_len": 2, "title_suffixes": ["医生", "先生", "哥", "哥哥", "大哥", "夫人", "船长"],
       "category_words": ["车票", "药剂", "道具"], "grade_category_pattern": "^[A-Za-zＡ-Ｚ]级(车票|药剂|道具|副本|玩家)$",
       "dictionary_cutoff": None}


def chapter(cid, number, text):
    return Chapter(cid, number, [("p0001", text)], "", "h")


def mention(cid, name, type_, same_as=None):
    return {"chapter_id": cid, "name": name, "type": type_, "same_as": same_as}


def entry(name, type_, canonical=None, **kw):
    return {"canonical": canonical or name, "name": name, "type": type_, **kw}


def setup(chapters, mentions=(), dictionary=(), dev=(), max_train=780, cutoff=None):
    snap = Snapshot("s", None, {}, {c.chapter_id: c for c in chapters}, list(mentions), list(dictionary))
    return resources(snap, split_ids(snap.chapters, list(dev), [], max_train), {**CFG, "dictionary_cutoff": cutoff})


def labels(res, ch, setting, **kw):
    row, tally, _ = label_chapter(ch, res, setting, **kw)
    spans = [(x["text"], x["start"], x["type"]) for x in row["spans"]]
    unsure = [(x["text"], x["types"], x["reason"]) for x in row["uncertain"]]
    return spans, unsure, tally, row


def test_longest_match_wins_whatever_its_source_and_sources_keep_priority_order():
    ch1 = chapter("ch0001", 1, "汀城第一研究所就在汀城。")
    ch2 = chapter("ch0002", 2, "汀城第一研究所在汀城。")
    res = setup([ch1, ch2], [mention("ch0001", "汀城", "location"), mention("ch0001", "汀城第一研究所", "organization"),
                             mention("ch0002", "汀城", "location")],
                [entry("汀城第一研究所", "organization")])
    spans, _, _, row = labels(res, ch1, "S2")
    assert spans == [("汀城第一研究所", 0, "organization"), ("汀城", 9, "location")]
    assert row["spans"][0]["sources"] == [["chapter_list", "organization"], ["dictionary", "organization"]]
    # ch2 lists only the short name: S2 labels it inside the long one; in S3 the long name another chapter
    # listed (and the dictionary has) is matched first and is uncertain, not split
    assert labels(res, ch2, "S2")[0] == [("汀城第一研究所", 0, "organization"), ("汀城", 8, "location")]
    res_no_dict = setup([ch1, ch2], [mention("ch0001", "汀城第一研究所", "organization"), mention("ch0002", "汀城", "location")])
    assert labels(res_no_dict, ch2, "S2")[0] == [("汀城", 0, "location"), ("汀城", 8, "location")]
    spans, unsure, _, _ = labels(res_no_dict, ch2, "S3")
    assert spans == [("汀城", 8, "location")]
    assert unsure == [("汀城第一研究所", ["organization"], "other_chapters_only")]


def test_same_as_values_never_become_spans():
    ch = chapter("ch0001", 1, "路人甲就是徐获。")
    res = setup([ch], [mention("ch0001", "路人甲", "person", same_as="徐获")])
    spans, unsure, _, _ = labels(res, ch, "S3")
    assert spans == [("路人甲", 0, "person")] and unsure == []


def test_scope_and_reveal_chapter_block_in_s2_and_make_s3_uncertain():
    text = "何普到了E27。"
    chapters = [chapter("ch0015", 15, text), chapter("ch0040", 40, text), chapter("ch0060", 60, text)]
    res = setup(chapters, [], [entry("何普", "person", known=50, scope=None),
                               entry("E27", "location", canonical="E27虫洞点", known=None, scope=[10, 20]),
                               entry("E27", "location", canonical="E27虫洞点", known=55, scope=None)])
    early, mid, late = chapters
    assert labels(res, mid, "S2")[0] == []
    assert labels(res, mid, "S3")[1] == [("何普", ["person"], "dictionary_scope_or_reveal"),
                                         ("E27", ["location"], "dictionary_scope_or_reveal")]
    assert labels(res, early, "S2")[0] == [("E27", 4, "location")]
    assert labels(res, late, "S2")[0] == [("何普", 0, "person"), ("E27", 4, "location")]
    # one applying row and one blocked row of the same name: the applying one decides the reason
    assert labels(res, late, "S3")[1] == [("何普", ["person"], "dictionary_applies"), ("E27", ["location"], "dictionary_applies")]


def test_uncertain_reason_is_the_first_source_and_the_row_keeps_every_source():
    ch1 = chapter("ch0001", 1, "童话城和灰塔。")
    text = "童话城和灰塔，徐获来了。"
    ch2 = chapter("ch0002", 2, text)
    res = setup([ch1, ch2], [mention("ch0001", "童话城", "location"), mention("ch0001", "灰塔", "location")],
                [entry("童话城", "location")])
    row, _, _ = label_chapter(ch2, res, "S3b", mined=[(7, 9)])
    # 童话城 is named after the dictionary, the first source, though another chapter lists it too
    assert [(x["text"], x["reason"], x["sources"]) for x in row["uncertain"]] == [
        ("童话城", "dictionary_applies", [["dictionary", "location"], ["other_chapters", "location"]]),
        ("灰塔", "other_chapters_only", [["other_chapters", "location"]]),
        ("徐获", "phrase_miner", None)]
    counts = coverage([row], {"ch0002": text})
    assert counts["uncertain_regions_by_reason"] == {"dictionary_applies": 1, "other_chapters_only": 1, "phrase_miner": 1}
    assert counts["uncertain_regions_by_sources"] == {"dictionary+other_chapters": 1, "other_chapters": 1, "none": 1}


def test_titles_are_cut_to_a_single_character_or_a_known_name_and_kept_or_uncertain_otherwise():
    ch1 = chapter("ch0001", 1, "马特来了。")
    text = "唐医生、马特船长、蝴蝶夫人和大哥。徐哥哥来了。"
    ch2 = chapter("ch0002", 2, text)
    res = setup([ch1, ch2], [mention("ch0001", "马特", "person")]
                + [mention("ch0002", n, "person") for n in ("唐医生", "马特船长", "蝴蝶夫人", "大哥", "徐哥哥")])
    spans, unsure, tally, _ = labels(res, ch2, "S2")
    assert spans == [("唐", 0, "person"), ("马特", 4, "person"), ("蝴蝶夫人", 9, "person"), ("大哥", 14, "person"),
                     ("徐", 17, "person")]
    assert tally["K1_title_stripped"] == 3 and tally["K1_title_kept"] == 1
    spans, unsure, _, _ = labels(res, ch2, "S3")
    assert [s[0] for s in spans] == ["唐", "马特", "大哥", "徐"]
    assert unsure == [("蝴蝶夫人", ["person"], "title_boundary")]
    assert [s[0] for s in labels(res, ch2, "S2raw")[0]] == ["唐医生", "马特船长", "蝴蝶夫人", "大哥", "徐哥哥"]


def test_fake_prefix_is_cut_only_before_a_known_person_name():
    text = "假何普笑了，何普没笑，假面来了。"
    ch = chapter("ch0001", 1, text)
    res = setup([ch], [mention("ch0001", "假面", "person")],
                [entry("假何普", "person"), entry("何普", "person")])
    spans, _, tally, _ = labels(res, ch, "S2")
    assert spans == [("何普", 1, "person"), ("何普", 6, "person"), ("假面", text.index("假面"), "person")]
    assert tally["K2_fake_prefix"] == 1


def test_category_words_and_grade_plus_category_become_o():
    text = "他用车票换了E级车票和回程票。"
    ch = chapter("ch0001", 1, text)
    res = setup([ch], [mention("ch0001", n, "item") for n in ("车票", "E级车票", "回程票")])
    for setting in ("S2", "S3"):
        spans, unsure, tally, _ = labels(res, ch, setting)
        assert spans == [("回程票", text.index("回程票"), "item")] and unsure == []
        assert tally["K3_category_to_O"] == 2
    assert len(labels(res, ch, "S2raw")[0]) == 3


def test_tail_abbreviations_of_non_person_names_are_kept_in_s2_and_uncertain_in_s3():
    ch1 = chapter("ch0001", 1, "去童话城。")
    text = "琴弦断了，无处不在的琴弦还在，童话城很远，胜男笑了。"
    ch2 = chapter("ch0002", 2, text)
    res = setup([ch1, ch2], [mention("ch0001", "童话城", "location", same_as="粉红帽的童话城"),
                             mention("ch0002", "童话城", "location")],
                [entry("琴弦", "item", canonical="无处不在的琴弦"), entry("无处不在的琴弦", "item"),
                 entry("胜男", "person", canonical="李胜男")])
    spans, _, tally, _ = labels(res, ch2, "S2")
    assert [s[0] for s in spans] == ["琴弦", "无处不在的琴弦", "童话城", "胜男"]
    assert tally["K4_tail_kept"] == 2
    spans, unsure, _, _ = labels(res, ch2, "S3")
    assert spans == []  # 童话城 is listed here, but as another chapter's tail abbreviation it turns uncertain
    assert unsure == [("琴弦", ["item"], "tail_abbreviation"), ("无处不在的琴弦", ["item"], "dictionary_applies"),
                      ("童话城", ["location"], "tail_abbreviation"),
                      ("胜男", ["person"], "dictionary_applies")]  # a person's tail is part of a name: no K4


def test_two_types_s2_takes_the_chapter_list_s3_is_uncertain_s1_skips_like_the_dictionary_system():
    ch = chapter("ch0001", 1, "第十七医院在白塔旁。")
    dictionary = [entry("第十七医院", "dungeon"), entry("白塔", "location"), entry("白塔", "organization")]
    res = setup([ch], [mention("ch0001", "第十七医院", "organization")], dictionary)
    assert labels(res, ch, "S2")[0] == [("第十七医院", 0, "organization")]
    assert labels(res, ch, "S3")[1] == [("第十七医院", ["dungeon", "organization"], "type_conflict"),
                                        ("白塔", ["location", "organization"], "type_conflict")]
    s1 = labels(res, ch, "S1")[0]
    preds, conflicts = dictionary_predictions(dictionary, [ch])
    assert s1 == [(p["text"], p["start"], p["type"]) for p in preds] == [("第十七医院", 0, "dungeon")]
    assert conflicts == ["白塔"]


def test_other_chapter_lexicon_and_known_names_come_from_training_chapters_only():
    ch1 = chapter("ch0001", 1, "尼塞先生来了，尼塞走了，E27很远。")
    dev = chapter("ch0002", 2, "尼塞去了E27。")
    ch3 = chapter("ch0003", 3, "E27到了。")
    res = setup([ch1, dev, ch3], [mention("ch0001", "尼塞先生", "person"), mention("ch0002", "尼塞", "person"),
                                  mention("ch0003", "E27", "location")], dev=["ch0002"])
    assert "尼塞" not in res["persons"] and ("尼塞", "person") not in res["lexicon"]
    spans, unsure, _, _ = labels(res, ch1, "S3")
    assert spans == []  # the dev chapter's 尼塞 does not reach a training label, nor makes 尼塞 a known name
    assert unsure == [("尼塞先生", ["person"], "title_boundary"), ("E27", ["location"], "other_chapters_only")]
    assert labels(res, ch3, "S3")[0] == [("E27", 0, "location")]  # its own listing is not "another chapter"
    spans, unsure, _, row = labels(res, dev, "S3", split="dev")
    assert row["split"] == "dev" and spans == [("尼塞", 0, "person")]
    assert unsure == [("E27", ["location"], "other_chapters_only")]


def test_dictionary_cleaning_and_learning_curve_cutoff():
    rows = [entry("周凝（医院里的投影）", "person"), entry("锁", "item"), entry("何普", "person", known=8),
            entry("周凝", "person", known=20), entry("回音铃", "item"), entry("灰塔", "location"),
            entry("徐知", "person", canonical="徐获", known=9, scope=[12, 15])]
    kept, dropped = clean_dictionary(rows, 2)
    assert [r["name"] for r in kept] == ["何普", "周凝", "回音铃", "灰塔", "徐知"]
    assert dropped == {"parenthesis": 1, "shorter_than_min_len": 1}
    log = []
    kept, dropped = clean_dictionary(rows, 2, cutoff=10, train_texts=["何普拿着回音铃，徐知也在。"], log=log, listed_at={})
    assert [r["name"] for r in kept] == ["何普", "回音铃"]
    assert dropped == {"parenthesis": 1, "shorter_than_min_len": 1, "cutoff_known_after_n": 1,
                       "cutoff_name_not_in_a_training_text_by_n": 1, "cutoff_scope_starts_after_n": 1}
    # 周凝 is revealed after N; 灰塔 is never written by N; 徐知 is revealed by N (9) but its scope starts after N
    assert [(r["name"], why) for r, why in log if why.startswith("cutoff_")] == [
        ("周凝", "cutoff_known_after_n"), ("灰塔", "cutoff_name_not_in_a_training_text_by_n"),
        ("徐知", "cutoff_scope_starts_after_n")]
    # a scope that starts by N is kept though it ends after N: its end only matters in later chapters
    assert clean_dictionary(rows[-1:], 2, cutoff=12, train_texts=["徐知"], listed_at={})[0] == rows[-1:]


def test_a_tier_takes_nothing_after_n_into_training_labels():
    # tier N = 10: ch0001 and ch0005 train; ch0008 (<= N) and ch0020 (> N) are dev; ch0030 is after N
    text1 = "马特先生在灰塔，E27很远，白塔也在。"
    ch1, ch5 = chapter("ch0001", 1, text1), chapter("ch0005", 5, "灰塔。")
    dev8, dev20 = chapter("ch0008", 8, "何普。"), chapter("ch0020", 20, "白塔和E27，何普来了。")
    ch30 = chapter("ch0030", 30, "E27的马特和何普。")
    chapters = [ch1, ch5, dev8, dev20, ch30]
    mentions = [mention("ch0001", "马特先生", "person"), mention("ch0005", "灰塔", "location"),
                mention("ch0020", "白塔", "location"), mention("ch0030", "E27", "location"), mention("ch0030", "马特", "person")]
    dictionary = [entry("E27", "location", known=25), entry("何普", "person")]
    tier = setup(chapters, mentions, dictionary, dev=["ch0008", "ch0020"], max_train=10, cutoff=10)
    # E27 is revealed after N; 何普 is written only in a dev chapter and after N, never in a training text by N
    assert tier["dictionary"] == [] and tier["dictionary_dropped"] == {
        "cutoff_known_after_n": 1, "cutoff_name_not_in_a_training_text_by_n": 1}
    assert set(tier["lexicon"]) == {("马特先生", "person"), ("灰塔", "location")}  # nothing from ch0030 or ch0020
    assert "马特" not in tier["persons"] and tier["mention_rows_used"] == 3
    assert [x["rows"] for x in tier["train_mention_rows_by_version"]] == [2]
    spans, unsure, _, _ = labels(tier, ch1, "S3")
    assert spans == [] and unsure == [("马特先生", ["person"], "title_boundary"), ("灰塔", ["location"], "other_chapters_only")]
    # a dev chapter after N keeps its own list (never trained on); the dropped dictionary row is gone there too
    spans, unsure, _, row = labels(tier, dev20, "S3", split="dev")
    assert row["split"] == "dev" and spans == [("白塔", 0, "location")] and unsure == []

    full = setup(chapters, mentions, dictionary, dev=["ch0008", "ch0020"])  # no tier: ch0030 trains, whole dictionary
    spans, unsure, _, _ = labels(full, ch1, "S3")
    assert spans == [("马特", 0, "person")]  # 马特 is a known person name once ch0030 trains
    assert [(x[0], x[2]) for x in unsure] == [("灰塔", "other_chapters_only"), ("E27", "dictionary_scope_or_reveal")]
    spans, unsure, _, _ = labels(full, dev20, "S3", split="dev")
    assert spans == [("白塔", 0, "location")]
    assert [(x[0], x[2]) for x in unsure] == [("E27", "dictionary_scope_or_reveal"), ("何普", "dictionary_applies")]


def test_a_dictionary_tail_needs_its_canonical_name_by_n():
    ch1, ch30 = chapter("ch0001", 1, "琴弦断了。"), chapter("ch0030", 30, "无处不在的琴弦。")
    mentions, dictionary = [mention("ch0001", "琴弦", "item")], [entry("琴弦", "item", canonical="无处不在的琴弦")]
    tier = setup([ch1, ch30], mentions, dictionary, max_train=10, cutoff=10)
    assert [r["name"] for r in tier["dictionary"]] == ["琴弦"]  # the row stays: its own name is written by N
    assert tier["tails"] == set() and [r["name"] for r in tier["dictionary_tail_rows_canonical_after_cutoff"]] == ["琴弦"]
    assert labels(tier, ch1, "S3")[:2] == ([("琴弦", 0, "item")], [])  # listed here and not a tail: positive
    # no cutoff (the whole dictionary, also with few training chapters): the row is a tail as before
    for res in (setup([ch1, ch30], mentions, dictionary), setup([ch1, ch30], mentions, dictionary, max_train=10)):
        assert res["tails"] == {"琴弦"} and res["dictionary_tail_rows_canonical_after_cutoff"] == []
        assert labels(res, ch1, "S3")[1] == [("琴弦", ["item"], "tail_abbreviation")]
    # the long name written in a training chapter by N makes it a tail inside the tier too
    ch2 = chapter("ch0002", 2, "无处不在的琴弦。")
    assert setup([ch1, ch2, ch30], mentions, dictionary, max_train=10, cutoff=10)["tails"] == {"琴弦"}


def test_mention_rows_and_spans_are_counted_by_the_prompt_version_of_the_chapter_list():
    ch1, ch454 = chapter("ch0001", 1, "汀城到了，徐获来了。"), chapter("ch0454", 454, "汀城。")
    mentions = [{**mention("ch0001", "汀城", "location"), "prompt_version": "extract_v1", "model": "m"},
                {**mention("ch0454", "汀城", "location"), "prompt_version": "extract_v1.1", "model": "m"}]
    dictionary = [entry("徐获", "person")]
    res = setup([ch1, ch454], mentions, dictionary)
    assert res["list_version"] == {"ch0001": "extract_v1", "ch0454": "extract_v1.1"}
    assert res["train_mention_rows_by_version"] == [
        {"prompt_version": "extract_v1", "model": "m", "rows": 1, "chapters": 1, "chapter_numbers": [1, 1]},
        {"prompt_version": "extract_v1.1", "model": "m", "rows": 1, "chapters": 1, "chapter_numbers": [454, 454]}]
    assert res["lexicon_pairs_by_list_versions"] == {"extract_v1+extract_v1.1": 1}
    rows = [labels(res, ch, "S2")[3] for ch in (ch1, ch454)]
    assert by_list_version(rows, res["list_version"]) == {
        "chapters": {"extract_v1": 1, "extract_v1.1": 1},
        "positive_spans": {"extract_v1": 1, "extract_v1.1": 1, "not_from_the_chapter_list": 1}, "uncertain_regions": {}}
    tier = setup([ch1, ch454], mentions, dictionary, max_train=453, cutoff=453)
    assert [x["prompt_version"] for x in tier["train_mention_rows_by_version"]] == ["extract_v1"]


def test_truncation_rules_name_the_cutoff_or_say_there_is_none():
    tier = truncation_rules({"max_train_chapter": 50, "dictionary_cutoff": 50})
    assert any("dictionary_cutoff 50" in r for r in tier) and any("canonical" in r for r in tier)
    # rows without a reveal chapter are dated by their first Opus listing up to 780, a date that only drops rows
    assert any("first chapter up to 780 whose Opus list" in r and "cutoff_first_listed_after_n" in r for r in tier)
    assert any("never adds a row" in r and "only drop" in r and "development chapter's list" in r
               and "never a chapter after 780" in r for r in tier)
    # the rows no list names keep the text test: stated as not strict, and how it is recorded
    assert any(r.startswith("not strict") and "substring" in r and "no list up to 780" in r for r in tier)
    assert any(r.startswith("the record") and "undated residual" in r for r in tier)
    full = truncation_rules({"max_train_chapter": 780, "dictionary_cutoff": None})
    assert any("dictionary_cutoff null" in r for r in full) and not any("not strict" in r for r in full)
    assert not any("Opus list has" in r for r in full)
    assert any("not truncated" in r for r in truncation_rules({"max_train_chapter": 50, "dictionary_cutoff": 100}))


def test_rows_without_a_reveal_chapter_are_dated_by_their_first_opus_listing():
    # tier N = 10: ch0001 and ch0002 train, ch0008 is dev, ch0030 is after N, ch0781 is never read
    ch1 = chapter("ch0001", 1, "钥匙扣挂着回音铃，徐获看着铜铃和灰塔。")
    ch2 = chapter("ch0002", 2, "钥匙、灵魂银针、何普和铜铃都在。")
    dev = chapter("ch0008", 8, "金针。")
    ch30 = chapter("ch0030", 30, "她拿到钥匙，大铜铃响了，徐获和何普来了。")
    ch781 = chapter("ch0781", 781, "灰塔。")
    chapters = [ch1, ch2, dev, ch30, ch781]
    mentions = [mention("ch0001", "回音铃", "item"), mention("ch0001", "徐获", "person"),
                mention("ch0002", "灵魂银针", "item"), mention("ch0008", "金针", "item"),
                mention("ch0030", "钥匙", "item"), mention("ch0030", "大铜铃", "item", same_as="铜铃"),
                mention("ch0030", "何普", "person"), mention("ch0781", "灰塔", "location")]
    dictionary = [entry("钥匙", "item", source="items"), entry("铜铃", "item", source="items"),
                  entry("回音铃", "item", source="items"), entry("银针", "item", canonical="灵魂银针", source="items"),
                  entry("灰塔", "location", source="registry"), entry("金针", "item", source="items"),
                  entry("何普", "person", source="registry", known=5), entry("徐获", "person", source="registry", known=20)]
    tier = setup(chapters, mentions, dictionary, dev=["ch0008"], max_train=10, cutoff=10)
    # 钥匙 passes the text test (钥匙扣) but is first listed at 30; 铜铃 too, as a same_as value at 30: both dropped.
    # 回音铃 is listed by N; 银针 is dated by its canonical name's listing; 灰塔 is listed only after 780 (never read),
    # so it has no date and keeps the text test; 何普 has a reveal chapter, so its late listing does not matter
    assert [r["name"] for r in tier["dictionary"]] == ["回音铃", "银针", "灰塔", "何普"]
    assert [(r["name"], why) for r, why in tier["dictionary_dropped_rows"]] == [
        ("钥匙", "cutoff_first_listed_after_n"), ("铜铃", "cutoff_first_listed_after_n"),
        ("金针", "cutoff_name_not_in_a_training_text_by_n"), ("徐获", "cutoff_known_after_n")]
    assert "灰塔" not in tier["first_listed_at"] and tier["first_listed_at"]["铜铃"] == 30
    got = {u["row"]["name"]: (u["first_training_text"], u["first_listed"], u["first_listed_in_training"],
                              u["text_test_only"], u["dated_only_by_a_list_not_trained_on"]) for u in tier["undated_rows"]}
    assert got == {"回音铃": (1, 1, 1, False, False), "银针": (2, 2, 2, False, False), "灰塔": (1, None, None, True, False)}
    # the dropped 钥匙 no longer makes ch0002's 钥匙 uncertain; without a cutoff it does
    assert not any(x[0] == "钥匙" for x in labels(tier, ch2, "S3")[1])
    full = setup(chapters, mentions, dictionary, dev=["ch0008"])
    assert ("钥匙", ["item"], "dictionary_applies") in labels(full, ch2, "S3")[1]
    # null cutoff: nothing dated, nothing dropped but the usual rules, nothing recorded
    assert full["dictionary"] == dictionary and full["first_listed_at"] is None and full["undated_rows"] == []
    assert clean_dictionary(dictionary, 2) == clean_dictionary(dictionary, 2, listed_at={"钥匙": 999})
    with pytest.raises(ValueError, match="listed_at"):
        clean_dictionary(dictionary, 2, cutoff=10, train_texts=[ch1.text])
    # the date never adds a row: whatever the listings say, the rows kept are among those the other rules keep
    texts = [ch1.text, ch2.text]
    without_dates = clean_dictionary(dictionary, 2, 10, texts, listed_at={})[0]
    assert [r["name"] for r in without_dates] == ["钥匙", "铜铃", "回音铃", "银针", "灰塔", "何普"]
    for listed_at in (tier["first_listed_at"], {r["name"]: 1 for r in dictionary}, {r["name"]: 99 for r in dictionary}):
        kept = clean_dictionary(dictionary, 2, 10, texts, listed_at=listed_at)[0]
        assert all(any(r is k for k in without_dates) for r in kept)
    assert clean_dictionary(dictionary, 2, 10, texts, listed_at={r["name"]: 1 for r in dictionary})[0] == without_dates
    assert [r["name"] for r in clean_dictionary(dictionary, 2, 10, texts,
                                                listed_at={r["name"]: 99 for r in dictionary})[0]] == ["何普"]


def test_undated_rows_are_recorded_with_their_dates_and_the_regions_they_back():
    # tier N = 10: 钥匙 occurs by N only inside 钥匙扣 (an ordinary word); the book makes it an item at ch0030
    ch1 = chapter("ch0001", 1, "钥匙扣挂着回音铃，灰塔很远。")
    ch2 = chapter("ch0002", 2, "假何普和钥匙。回音铃响了。")
    dev = chapter("ch0008", 8, "钥匙。")
    ch30 = chapter("ch0030", 30, "她拿到钥匙。")
    ch781 = chapter("ch0781", 781, "灰塔。")
    mentions = [mention("ch0001", "回音铃", "item"), mention("ch0002", "回音铃", "item"),
                mention("ch0008", "钥匙", "item"), mention("ch0030", "钥匙", "item"),
                mention("ch0781", "灰塔", "location"), mention("ch0002", "何普", "person")]
    dictionary = [entry("钥匙", "item", source="items"), entry("回音铃", "item", source="items"),
                  entry("灰塔", "location", source="registry"), entry("假何普", "person", source="registry"),
                  entry("徐获", "person", source="registry", known=1)]
    chapters = [ch1, ch2, dev, ch30, ch781]
    cfg = {"dictionary_cutoff": 10, "settings": ["S3"]}
    tier = setup(chapters, mentions, dictionary, dev=["ch0008"], max_train=10, cutoff=10)
    got = {u["row"]["name"]: (u["first_training_text"], u["first_listed"], u["first_listed_in_training"],
                              u["text_test_only"], u["dated_only_by_a_list_not_trained_on"]) for u in tier["undated_rows"]}
    # the dev chapter's list (ch0008) dates 钥匙 by N, which the training lists alone would not: kept and counted;
    # a list after chapter 780 is never read (灰塔: no date); a row with a reveal chapter is not in the record
    assert got == {"钥匙": (1, 8, None, False, True), "回音铃": (1, 1, 1, False, False),
                   "灰塔": (1, None, None, True, False), "假何普": (2, None, None, True, False)}
    # attribution by the matched string: the K2-converted region 何普 (matched as 假何普, which hides the listed 何普)
    # belongs to the 假何普 row; 钥匙 and the listed 回音铃 to theirs
    spans, unsure, _, row = labels(tier, ch2, "S3")
    assert ("钥匙", ["item"], "dictionary_applies") in unsure and ("回音铃", 7, "item") in spans
    props = proposals(ch2, tier)
    chosen = resolve(ch2.text, active(props, "S3"), occurrences(ch2.text, props))[0]
    watched = {u["row"]["name"]: [(i, u["row"])] for i, u in enumerate(tier["undated_rows"])}
    listed = {name for name, _ in tier["listed"]["ch0002"]}
    hits = dictionary_row_hits(ch2.text, row, chosen, watched, listed)
    names = [tier["undated_rows"][i]["row"]["name"] for i, *_ in hits]
    # 假何普's region is chapter-listed through the listed 何普 inside it (the match hides a would-be positive)
    assert sorted(zip(names, [h[1] for h in hits], [h[2] for h in hits], [h[3] for h in hits])) == [
        ("假何普", "uncertain_regions", (1, 3), True), ("回音铃", "positive_spans", (7, 10), True),
        ("钥匙", "uncertain_regions", (4, 6), False)]
    assert [h[3] for h in dictionary_row_hits(ch2.text, row, chosen, watched)] == [True, False, False]  # sources only
    made = {"res": tier, "undated_hits": {"S3": [("ch0002", *h) for h in hits]}}
    counts, train, rows = undated_record(made, cfg)
    assert {k: v for k, v in counts.items() if k != "note"} == {
        "dropped_first_listed_after_n": 0, "dropped_first_listed_after_n_by_source": {}, "kept": 4,
        "dated_by_a_listing_by_n": 2, "dated_only_by_a_list_not_trained_on": 1, "text_test_only_never_listed_up_to_780": 2}
    assert train["S3"]["text_test_only_never_listed_up_to_780"] == {
        "positive_spans": 0, "uncertain_regions": 1, "uncertain_regions_the_chapter_list_names_it_or_a_part": 1, "chapters": 1}
    assert train["S3"]["dated_only_by_a_list_not_trained_on"] == {
        "positive_spans": 0, "uncertain_regions": 1, "uncertain_regions_the_chapter_list_names_it_or_a_part": 0, "chapters": 1}
    assert train["S3"]["all"]["positive_spans"] == 1 and train["S3"]["all"]["uncertain_regions"] == 2
    assert [(r["name"], r["train"]["S3"]["uncertain_regions"]) for r in rows["kept"]] == [
        ("钥匙", 1), ("回音铃", 0), ("灰塔", 0), ("假何普", 1)]
    # without the dev chapter's list 钥匙 is first listed at 30: dropped, and named in the record with its date
    no_dev = setup(chapters, [m for m in mentions if m["chapter_id"] != "ch0008"], dictionary, dev=["ch0008"],
                   max_train=10, cutoff=10)
    assert [u["row"]["name"] for u in no_dev["undated_rows"]] == ["回音铃", "灰塔", "假何普"]
    counts, _, rows = undated_record({"res": no_dev, "undated_hits": {"S3": []}}, cfg)
    assert counts["dropped_first_listed_after_n"] == 1 and counts["dropped_first_listed_after_n_by_source"] == {"items": 1}
    assert rows["dropped_first_listed_after_n"] == [
        {"name": "钥匙", "type": "item", "source": "items", "canonical": "钥匙", "first_listed_chapter": 30}]
    assert not any(x[0] == "钥匙" for x in labels(no_dev, ch2, "S3")[1])
    # no cutoff: no dates, nothing recorded
    full = setup(chapters, mentions, dictionary, dev=["ch0008"])
    assert full["undated_rows"] == []
    assert undated_record({"res": full, "undated_hits": {}}, {**cfg, "dictionary_cutoff": None}) == (None, {}, {})


def test_label_bytes_match_the_file_and_rows_are_compared_split_by_split(tmp_path):
    span = {"start": 0, "end": 2, "type": "person", "text": "徐获", "sources": [["chapter_list", "person"]]}
    train = {"chapter_id": "ch0001", "number": 1, "split": "train", "uncertain": [], "spans": [span]}
    dev = {"chapter_id": "ch0002", "number": 2, "split": "dev", "spans": [], "uncertain": []}
    dev_b = {**dev, "spans": [{"start": 0, "end": 2, "type": "location", "text": "汀城", "sources": [["dictionary", "location"]]}]}
    dev3 = {"chapter_id": "ch0003", "number": 3, "split": "dev", "uncertain": [], "spans": [span]}
    dev3_b = {**dev3, "spans": [{**span, "sources": [["chapter_list", "person"], ["dictionary", "person"]]}]}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [train, dev, dev3])
    write_jsonl(b, [train, dev_b, dev3_b])
    assert a.read_bytes() == label_bytes([train, dev, dev3])
    out = compare_rows(a, b)["splits"]
    assert out["train"]["byte_identical"] and out["train"]["rows_sha256_a"] == out["train"]["rows_sha256_b"]
    assert out["train"]["chapters_differing"] == [] and not any(out["train"]["differences"].values())
    assert not out["dev"]["byte_identical"] and out["dev"]["chapters_differing"] == ["ch0002", "ch0003"]
    assert {k: v for k, v in out["dev"]["differences"].items() if v} == {
        "positive_spans_only_in_b": 1, "positive_spans_same_but_sources_differ": 1}


def test_split_leaves_out_dev_chapters_and_refuses_test_chapters_or_late_chapters():
    chapters = {c.chapter_id: c for c in [chapter(f"ch{n:04d}", n, "x") for n in (1, 2, 3, 4, 781)]}
    split = split_ids(chapters, ["ch0002"], ["ch0781"], 780)
    assert split == {"train": ["ch0001", "ch0003", "ch0004"], "dev": ["ch0002"]}
    assert split_ids(chapters, ["ch0002"], ["ch0781"], 3)["train"] == ["ch0001", "ch0003"]
    with pytest.raises(ValueError):
        split_ids(chapters, ["ch0002"], ["ch0003"], 780)
    with pytest.raises(ValueError):
        split_ids(chapters, ["ch0781"], [], 780)
    with pytest.raises(ValueError):
        split_ids(chapters, [], [], 781)


def test_miner_fills_only_gaps_in_s3b():
    text = "徐获去了汀城。"
    ch = chapter("ch0001", 1, text)
    res = setup([ch], [mention("ch0001", "徐获", "person")])
    mined = mine(ch, {"nr", "ns"}, 2)
    assert (4, 6) in mined
    spans, unsure, _, _ = labels(res, ch, "S3b", mined=[(0, 2), (1, 3), (4, 6)])
    assert spans == [("徐获", 0, "person")] and unsure == [("汀城", None, "phrase_miner")]
    assert labels(res, ch, "S3", mined=[(4, 6)])[1] == []


def test_s1b_keeps_dictionary_names_positive_and_uses_no_opus_list():
    # ch1 lists 汀城 and 马特船长; the other chapter lists 何普. None of it may reach S1b; the dictionary may.
    text = "马特船长从汀城来，何普在E27，琴弦响了。"
    ch1 = chapter("ch0001", 1, text)
    ch2 = chapter("ch0002", 2, "何普。")
    res = setup([ch1, ch2], [mention("ch0001", "汀城", "location"), mention("ch0001", "马特船长", "person"),
                             mention("ch0002", "何普", "person")],
                [entry("马特船长", "person"), entry("E27", "location", known=5),
                 entry("琴弦", "item", canonical="无处不在的琴弦"), entry("无处不在的琴弦", "item")])
    spans, unsure, _, _ = labels(res, ch1, "S1b", mined=[(5, 7), (9, 11)])
    assert spans == []
    assert unsure == [("马特船长", ["person"], "title_boundary"), ("汀城", None, "phrase_miner"),
                      ("何普", None, "phrase_miner"), ("E27", ["location"], "dictionary_scope_or_reveal"),
                      ("琴弦", ["item"], "tail_abbreviation")]
    # the same match is positive in S1b when nothing makes it doubtful, as in S1
    ch3 = chapter("ch0003", 3, "无处不在的琴弦。")
    res3 = setup([ch3], [], [entry("无处不在的琴弦", "item")])
    assert labels(res3, ch3, "S1b", mined=[])[0] == labels(res3, ch3, "S1")[0] == [("无处不在的琴弦", 0, "item")]


def test_char_allowed_gives_explicit_sets_in_tag_order():
    assert len(TAGS) == 13 and TAGS[:3] == ["O", "B-person", "I-person"]
    spans = [{"start": 0, "end": 2, "type": "person"}]
    uncertain = [{"start": 3, "end": 5, "types": ["item", "location"]}, {"start": 6, "end": 7, "types": None}]
    allowed = [{TAGS[i] for i in s} for s in char_allowed(8, spans, uncertain)]
    assert allowed == [{"B-person"}, {"I-person"}, {"O"}, {"O", "B-item", "B-location"}, {"O", "I-item", "I-location"},
                       {"O"}, set(TAGS), {"O"}]
    with pytest.raises(ValueError):
        char_allowed(8, spans, [{"start": 1, "end": 3, "types": None}])
