from types import SimpleNamespace

from novel_ner.novelty import novelty, weak_names


def test_weak_names_cut_mentions_at_the_cutoff():
    snap = SimpleNamespace(
        dictionary=[{"name": "回音铃", "type": "item"}, {"name": "白鸦", "type": "person", "known": 30}],
        mentions=[{"chapter_id": "ch0010", "name": "林小雨"}, {"chapter_id": "ch0090", "name": "灰塔"}],
    )
    dictionary, mentions, sources = weak_names(snap, 50)
    assert dictionary == {"回音铃", "白鸦"} and mentions == {"林小雨"}
    assert sources["mention_rows_after_cutoff_ignored"] == 1
    assert sources["dictionary_known_max"] == 30 and sources["dictionary_rows_known_after_cutoff"] == 0


def test_novelty_counts_exact_text_any_type_and_skips_unscored():
    text = "林小雨拿出回音铃，林小雨看着林。灰塔"
    gold = {"ch0100": {
        "spans": {(0, 3, "person"), (5, 8, "item"), (9, 12, "person"), (14, 15, "person"), (16, 18, "location")},
        "unscored": [(16, 18)],  # the location is left out, as in scoring
    }}
    out = novelty(gold, {"ch0100": text}, {"林小雨", "回音铃"})
    assert out["spans"] == 4 and out["new_spans"] == 1  # only the bare surname is new
    assert out["new_share"] == 0.25
    assert out["distinct_texts"] == 3 and out["new_distinct_texts"] == 1
    assert out["by_type"]["person"] == {"spans": 3, "new_spans": 1, "distinct_names": 2, "new_distinct_names": 1}
    assert out["by_type"]["location"]["spans"] == 0
