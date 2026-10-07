"""Task packages for GPT annotation drafts, and importing what GPT returns.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.llm_task build  --name calib --chapters ch0057 ch0046
    .venv\\Scripts\\python -m novel_ner.llm_task import data/llm_tasks/calib/pass_A.json --run A --model "GPT-..."

A package (data/llm_tasks/<name>/, holds novel text, never committed) has the fixed drafting
instructions, the annotation guide, the chapters with paragraph ids, and a README for the user.
GPT answers with mentions located by (paragraph_id, text, occurrence); `import` turns them into
character offsets. A mention that cannot be located is kept as a system error, never dropped
silently (guide, appendix 1).
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from novel_ner.annotate import test_chapters
from novel_ner.compare import ENTITY_TYPES
from novel_ner.snapshot import REPO_ROOT, load_snapshot

INSTRUCTIONS = {"v1": REPO_ROOT / "docs" / "llm_annotation" / "instructions_v1.md"}
GUIDES = {"1.1": REPO_ROOT / "docs" / "annotation_guide" / "v1.1.md"}
STUCK = ("qualification", "boundary", "type")
TASK_ROOT = REPO_ROOT / "data" / "llm_tasks"

README = """# GPT 标注任务包：{name}

章节：{chapters}。起草说明 {instructions}，规则 v{guide}，快照 {snapshot}。

## 操作步骤

1. 先把这个文件夹复制到项目外面（比如桌面）。如果用 Codex，让它只在复制出来的文件夹里工作，免得它读到项目里你自己的答案文件。
2. 开一个**全新的对话**，不带任何之前的聊天记录、你的标注或疑难清单。
3. 把三个文件交给它：`instructions.md`（起草说明）、`guide.md`（规则）、`chapters.md`（原文），说一句：
   “请按 instructions.md 标注 chapters.md 里的全部章节，只输出 JSON。”
4. 把它输出的 JSON 存成 `pass_A.json`，放回项目里的这个文件夹（{folder}）。
5. 再开一个**全新的对话**，重复第 3 步，存成 `pass_B.json`。两遍之间不要让它看到上一遍的结果。
6. 告诉 Claude 用的模型名（比如 GPT 的版本、Codex 的版本）和日期。

这些文件含小说原文：只在本机和这两次 GPT 对话里用，不要发到别处。
"""


def chapters_markdown(snapshot, chapter_ids):
    lines = []
    for cid in chapter_ids:
        ch = snapshot.chapters[cid]
        lines.append(f"# {cid} 第{ch.number}章 {ch.title}")
        lines.append("")
        lines += [f"[{pid}] {text}" for pid, text in ch.paragraphs]
        lines.append("")
    return "\n".join(lines)


def build(name, chapter_ids, snapshot_id, instructions="v1", guide="1.1"):
    snapshot = load_snapshot(snapshot_id)
    missing = [c for c in chapter_ids if c not in snapshot.chapters]
    if missing:
        raise ValueError(f"not in the snapshot: {missing}")
    folder = TASK_ROOT / name
    if folder.exists():
        raise SystemExit(f"{folder} exists; pick a new name (packages are never overwritten)")
    folder.mkdir(parents=True)
    shutil.copyfile(INSTRUCTIONS[instructions], folder / "instructions.md")
    shutil.copyfile(GUIDES[guide], folder / "guide.md")
    (folder / "chapters.md").write_text(chapters_markdown(snapshot, chapter_ids), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    task = {"name": name, "snapshot_id": snapshot_id, "chapters": chapter_ids, "instructions_version": instructions,
            "guide_version": guide, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit": commit}
    (folder / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "README.md").write_text(README.format(
        name=name, chapters="、".join(f"第{snapshot.chapters[c].number}章" for c in chapter_ids),
        instructions=instructions, guide=guide, snapshot=snapshot_id, folder=folder), encoding="utf-8")
    return folder


def locate(paragraph, text, occurrence):
    """Start of the occurrence-th appearance of text in paragraph (overlaps count), or None."""
    start = -1
    for _ in range(occurrence):
        start = paragraph.find(text, start + 1)
        if start < 0:
            return None
    return start


def convert(answer, snapshot, chapter_ids):
    """GPT's answer -> {chapter_id: {"spans": [...], "errors": [...]}} with chapter-level offsets."""
    out = {cid: {"spans": [], "errors": []} for cid in chapter_ids}
    for chapter in answer.get("chapters", []):
        cid = chapter.get("chapter_id")
        if cid not in out:
            out.setdefault("_unknown", {"spans": [], "errors": []})["errors"].append(f"unexpected chapter {cid!r}")
            continue
        ch = snapshot.chapters[cid]
        starts, offset = {}, 0
        for pid, text in ch.paragraphs:
            starts[pid] = (offset, text)
            offset += len(text) + 1
        seen = {}
        if not isinstance(chapter.get("mentions"), list):
            out[cid]["errors"].append("no 'mentions' list for this chapter in the answer")
            continue
        for m in chapter["mentions"]:
            where = f"{m.get('paragraph_id')} {m.get('text')!r} #{m.get('occurrence')}"
            pending = bool(m.get("pending"))
            if m.get("paragraph_id") not in starts:
                out[cid]["errors"].append(f"{where}: no such paragraph")
                continue
            base, paragraph = starts[m["paragraph_id"]]
            text, occurrence = m.get("text") or "", m.get("occurrence")
            if not text or not isinstance(occurrence, int) or occurrence < 1:
                out[cid]["errors"].append(f"{where}: missing text or occurrence")
                continue
            at = locate(paragraph, text, occurrence)
            if at is None:
                out[cid]["errors"].append(f"{where}: not found in the paragraph")
                continue
            if not pending and m.get("type") not in ENTITY_TYPES:
                out[cid]["errors"].append(f"{where}: type {m.get('type')!r} is not one of the six")
                continue
            key = (base + at, base + at + len(text))
            reading = (None if pending else m["type"], pending, json.dumps(m.get("stuck")), m.get("candidates"),
                       m.get("doubt"), m.get("note"))
            if key in seen:
                if seen[key] != reading:  # the same place listed twice with different content: GPT contradicted itself
                    out[cid]["errors"].append(f"{where}: listed twice at the same place with different readings")
                continue  # an identical repeat (guide appendix 1: deduplicate)
            seen[key] = reading
            out[cid]["spans"].append({
                "start": key[0], "end": key[1], "text": text, "paragraph_id": m["paragraph_id"],
                "type": None if pending else m["type"], "pending": pending,
                "stuck": [s for s in m.get("stuck", []) if s in STUCK] if pending else [],
                "candidates": m.get("candidates", "") if pending else "",
                "doubt": m.get("doubt", "") if pending else "",
                "note": m.get("note", ""),
            })
        spans = sorted(out[cid]["spans"], key=lambda s: (s["start"], s["end"]))
        for a, b in zip(spans, spans[1:]):
            if b["start"] < a["end"]:
                out[cid]["errors"].append(f"overlap: {a['text']!r} and {b['text']!r} at {b['start']}")
        out[cid]["spans"] = spans
    answered = {c.get("chapter_id") for c in answer.get("chapters", [])}
    for cid in chapter_ids:
        if cid not in answered:
            out[cid]["errors"].append("chapter missing from the answer")
    return out


def import_answer(path, run, model):
    path = Path(path)
    folder = path.parent
    task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    snapshot = load_snapshot(task["snapshot_id"], verify=False)
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):  # tolerate a fenced answer pasted as-is
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    answer = json.loads(raw)
    draft = {
        "task": task["name"], "run": run, "model": model, "source_file": path.name,
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "snapshot_id": task["snapshot_id"], "instructions_version": answer.get("instructions_version"),
        "guide_version": answer.get("guide_version"), "chapters": convert(answer, snapshot, task["chapters"]),
    }
    target = folder / f"draft_{run}.json"
    target.write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    for cid, c in draft["chapters"].items():
        pending = sum(s["pending"] for s in c["spans"])
        print(f"{cid}: {len(c['spans']) - pending} spans, {pending} pending, {len(c['errors'])} errors")
        for e in c["errors"][:10]:
            print("   ", e)
    print(f"wrote {target}")
    return draft


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--chapters", nargs="+", required=True)
    b.add_argument("--snapshot", default="to0990-79dc77f8d354")
    b.add_argument("--instructions", default="v1")
    b.add_argument("--guide", default="1.1")
    t = sub.add_parser("build-test", help="one package per batch of the test order")
    t.add_argument("--config", default="configs/test_781_990.json")
    i = sub.add_parser("import")
    i.add_argument("file")
    i.add_argument("--run", required=True)
    i.add_argument("--model", required=True)
    args = ap.parse_args()
    if args.action == "build":
        print(f"wrote {build(args.name, args.chapters, args.snapshot, args.instructions, args.guide)}")
    elif args.action == "build-test":
        cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))
        order_file = Path(args.config).with_suffix(".order.json").as_posix()
        ids = test_chapters(order_file, cfg["n"])
        size = cfg["batch_size"]
        for b in range(0, len(ids), size):
            name = f"test_b{b // size + 1:02d}"
            folder = build(name, ids[b:b + size], cfg["snapshot"], cfg["frozen"]["instructions_version"],
                           cfg["frozen"]["guide_version"])
            print(f"wrote {folder}")
    else:
        import_answer(args.file, args.run, args.model)


if __name__ == "__main__":
    main()
