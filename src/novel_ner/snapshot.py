"""Read-only access to a dataset snapshot exported by the novel project.

A snapshot lives in data/snapshots/<id>/ and is never edited here; if the data
needs to change, the novel project exports a new snapshot with a new id.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = REPO_ROOT / "data" / "snapshots"
PARAGRAPH_SEP = "\n"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Chapter:
    chapter_id: str
    number: int
    paragraphs: list  # [(paragraph_id, text)]
    title: str = ""
    content_hash: str = ""  # the novel project's hash of this chapter's text

    @property
    def text(self):
        return PARAGRAPH_SEP.join(t for _, t in self.paragraphs)


@dataclass
class Snapshot:
    id: str
    path: Path
    meta: dict
    chapters: dict  # chapter_id -> Chapter
    mentions: list  # LLM-extracted names, one row per (chapter, name)
    dictionary: list  # alias registry, item library, dungeon names


def load_snapshot(snapshot_id, root=SNAPSHOT_ROOT, verify=True):
    path = Path(root) / snapshot_id
    meta = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    if meta["id"] != snapshot_id:
        raise ValueError(f"snapshot.json says id={meta['id']}, expected {snapshot_id}")
    if verify:
        for name, info in meta["files"].items():
            # The exporter hashes the text with "\n" but writes it in text mode, so on Windows
            # the file has "\r\n"; hash what it meant to write.
            data = (path / name).read_bytes().replace(b"\r\n", b"\n")
            digest = hashlib.sha256(data).hexdigest()
            if digest != info["sha256"]:
                raise ValueError(f"{name}: sha256 does not match snapshot.json; re-export the snapshot")
    chapters = {}
    for row in read_jsonl(path / "chapters.jsonl"):
        chapters[row["chapter_id"]] = Chapter(row["chapter_id"], row["chapter_number"],
                                              [tuple(p) for p in row["paragraphs"]],
                                              row.get("title", ""), row.get("content_hash", ""))
    return Snapshot(
        id=snapshot_id,
        path=path,
        meta=meta,
        chapters=chapters,
        mentions=read_jsonl(path / "weak" / "mentions.jsonl"),
        dictionary=read_jsonl(path / "weak" / "dictionary.jsonl"),
    )
