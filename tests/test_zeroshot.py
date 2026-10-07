from novel_ner.compare import agreement, names_from_predictions
from novel_ner.gliner_zh import SPACE_MARK, CharSplitter, JiebaSplitter, NaturalSpaceMarks
from novel_ner.snapshot import Chapter
from novel_ner.zeroshot import chunk_chapter, to_paragraph


def test_chunks_keep_paragraph_offsets():
    ch = Chapter("ch9999", 9999, [("p1", "甲乙丙"), ("p2", "丁戊"), ("p3", "己庚辛壬")])
    chunks = chunk_chapter(ch, max_chars=6)
    assert [text for text, _ in chunks] == ["甲乙丙\n丁戊", "己庚辛壬"]
    text, spans = chunks[0]
    for pid, offset, length in spans:
        assert text[offset:offset + length] == dict(ch.paragraphs)[pid]
    assert to_paragraph({"start": 4, "end": 6}, spans) == ("p2", 0, 2)
    assert to_paragraph({"start": 2, "end": 5}, spans) is None  # crosses the break


def test_char_splitter_keeps_ascii_runs():
    words = [w for w, _, _ in CharSplitter()("徐获去E27-014区。")]
    assert words == ["徐", "获", "去", "E27", "-", "014", "区", "。"]


def test_natural_marks_only_after_whitespace():
    words = list(NaturalSpaceMarks(CharSplitter())("甲乙\n丙"))
    assert words == [(SPACE_MARK + "甲", 0, 1), ("乙", 1, 2), (SPACE_MARK + "丙", 3, 4)]


def test_jieba_offsets_slice_the_text():
    text = "徐获拉开门，\n一个女人站在门外。"
    for word, start, end in JiebaSplitter()(text):
        assert text[start:end] == word and not word.isspace()


def test_agreement_counts():
    ref = {"c1": {"徐获": {"person"}, "特事局": {"organization"}}}
    preds = [
        {"chapter_id": "c1", "text": "徐获", "label": "person", "score": 0.9},
        {"chapter_id": "c1", "text": "徐获", "label": "item", "score": 0.4},
        {"chapter_id": "c1", "text": "特事", "label": "organization", "score": 0.8},
        {"chapter_id": "c1", "text": "门", "label": "location", "score": 0.2},
    ]
    pred = names_from_predictions(preds, ["c1"], threshold=0.3)
    assert pred == {"c1": {"徐获": {"person"}, "特事": {"organization"}}}
    m = agreement(pred, ref)
    assert (m["n_pred"], m["n_ref"]) == (2, 2)
    assert m["exact"]["precision"] == 0.5 and m["exact"]["recall"] == 0.5
    assert m["loose"]["recall"] == 1.0
    assert m["recall_by_ref_type"]["organization"] == 0.0
