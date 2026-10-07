from novel_ner.llm_task import chapters_markdown, convert, locate
from novel_ner.sample_test import stratified_order
from novel_ner.score import disagreements, strict
from novel_ner.snapshot import Chapter, Snapshot

PARAS = [("p0001", "林小雨看着林小雨的影子。"), ("p0002", "她去了灰塔城。")]  # text: "林小雨看着林小雨的影子。\n她去了灰塔城。"


def snap():
    return Snapshot("snap-test", None, {}, {"ch9999": Chapter("ch9999", 9999, PARAS, "测试", "h")}, [], [])


def test_locate_counts_every_appearance_including_overlaps():
    assert locate("林小雨看着林小雨", "林小雨", 2) == 5
    assert locate("明明明明", "明明", 3) == 2
    assert locate("林小雨", "林小雨", 2) is None


def test_convert_turns_occurrences_into_chapter_offsets_and_keeps_errors():
    answer = {"chapters": [{"chapter_id": "ch9999", "mentions": [
        {"paragraph_id": "p0001", "text": "林小雨", "occurrence": 2, "type": "person"},
        {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": "location"},
        {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": "location"},  # exact repeat
        {"paragraph_id": "p0002", "text": "白塔城", "occurrence": 1, "type": "location"},  # not in text
        {"paragraph_id": "p0001", "text": "影子", "occurrence": 1, "type": "thing"},  # bad type
        {"paragraph_id": "p0001", "text": "林小雨", "occurrence": 1, "type": None, "pending": True,
         "stuck": ["type", "mood"], "candidates": "person / 不标", "doubt": "?"},
    ]}]}
    out = convert(answer, snap(), ["ch9999"])["ch9999"]
    spans = [(s["start"], s["end"], s["type"], s["pending"]) for s in out["spans"]]
    assert spans == [(0, 3, None, True), (5, 8, "person", False), (16, 19, "location", False)]
    assert out["spans"][0]["stuck"] == ["type"]
    assert len(out["errors"]) == 2


def test_chapters_markdown_has_ids_for_the_model():
    md = chapters_markdown(snap(), ["ch9999"])
    assert md.startswith("# ch9999 第9999章 测试") and "[p0002] 她去了灰塔城。" in md


def test_strict_scoring_with_unscored_regions_and_invalid_predictions():
    gold = {"c": {"spans": {(0, 3, "person"), (5, 8, "person"), (16, 19, "location")}, "unscored": [(20, 22)]}}
    pred = {"c": {"spans": {(0, 3, "person"), (5, 8, "item"), (20, 21, "item")}, "pending": [(16, 19)], "errors": 1}}
    m = strict(gold, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)  # (20,21) touches an unscored region; 1 invalid counts as fp
    assert m["fp_same_span_wrong_type"] == 1 and m["fn_where_draft_was_pending"] == 1
    assert m["per_type"]["person"]["support"] == 2


def test_disagreements_list_what_a_person_must_decide():
    a = {"c": {"spans": {(0, 3, "person"), (5, 8, "person")}, "pending": []}}
    b = {"c": {"spans": {(0, 3, "person")}, "pending": [(16, 19)]}}
    d = disagreements(a, b)["c"]
    assert d["only_a"] == [(5, 8, "person")] and d["only_b"] == [] and d["to_review"] == 2


def test_stratified_order_keeps_every_prefix_proportional():
    strata = {"big": [f"b{i}" for i in range(80)], "small": [f"s{i}" for i in range(20)]}
    order = stratified_order(strata, seed=1)
    assert len(order) == 100 and len({c for c, _ in order}) == 100
    first = [s for _, s in order[:30]]
    assert abs(first.count("small") - 6) <= 1
    assert stratified_order(strata, seed=1) == order


def test_agreement_between_passes_without_gold(tmp_path, monkeypatch):
    import json
    from novel_ner import score
    monkeypatch.setattr(score, "REPO_ROOT", tmp_path)
    task = tmp_path / "t"
    task.mkdir()
    (task / "task.json").write_text(json.dumps({"name": "t", "chapters": ["c"]}), encoding="utf-8")
    span = lambda s, e, t, p=False: {"start": s, "end": e, "type": t, "pending": p}
    (task / "draft_A.json").write_text(json.dumps({"chapters": {"c": {"spans": [span(0, 3, "person"), span(5, 8, "item")], "errors": []}}}), encoding="utf-8")
    (task / "draft_B.json").write_text(json.dumps({"chapters": {"c": {"spans": [span(0, 3, "person"), span(9, 11, None, True)], "errors": []}}}), encoding="utf-8")
    result = score.agreement([task], "r")
    row = result["tasks"][0]
    assert (row["shared"], row["only_a"], row["only_b"], row["pending_marks"]) == (1, 1, 0, 1)
    assert row["to_review_per_chapter"] == {"c": 2} and result["to_review_total"] == 2
    assert (tmp_path / "runs" / "r" / "agreement.json").exists()


def test_convert_flags_a_chapter_the_answer_left_out():
    out = convert({"chapters": []}, snap(), ["ch9999"])
    assert out["ch9999"]["errors"] == ["chapter missing from the answer"]



def test_convert_flags_a_chapter_without_a_mentions_list():
    out = convert({"chapters": [{"chapter_id": "ch9999", "entities": []}]}, snap(), ["ch9999"])
    assert out["ch9999"]["errors"] == ["no 'mentions' list for this chapter in the answer"]



def test_convert_refuses_two_readings_of_one_place_in_either_order():
    m1 = {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": "location"}
    m2 = {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": None, "pending": True}
    for mentions in ([m1, m2], [m2, m1]):
        out = convert({"chapters": [{"chapter_id": "ch9999", "mentions": mentions}]}, snap(), ["ch9999"])["ch9999"]
        assert any("different readings" in e for e in out["errors"])



def test_convert_refuses_two_different_pending_listings_of_one_place():
    m1 = {"paragraph_id": "p0002", "text": "灰塔城", "occurrence": 1, "type": None, "pending": True, "doubt": "甲"}
    m2 = dict(m1, doubt="乙")
    out = convert({"chapters": [{"chapter_id": "ch9999", "mentions": [m1, m2]}]}, snap(), ["ch9999"])["ch9999"]
    assert any("different readings" in e for e in out["errors"])
    out = convert({"chapters": [{"chapter_id": "ch9999", "mentions": [m1, dict(m1)]}]}, snap(), ["ch9999"])["ch9999"]
    assert out["errors"] == [] and len(out["spans"]) == 1  # an exact repeat is just dropped
