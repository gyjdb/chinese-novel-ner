"""Check a prefilled file (test, dev, dev2 or test2) against the GPT passes and the name knowledge sheet it was built from.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.verify_prefill data/gold/test/incoming/prefilled_test_b12.json [--before <older prefilled>]
    .venv\\Scripts\\python -m novel_ner.verify_prefill data/gold/dev2/incoming/prefilled_dev2_b05.json --set dev2

--set names the set the file must belong to: a file that says another set, or lies under another set's folder
data/gold/<other>/, is refused (gold.own_set_path); without --set the file's own "set" is taken as it is.

Invariants (project annotation protocol):
- a confident span is either given identically by both passes, or is exactly a typed sheet row's name with its type;
- every span either pass gave is still covered by an output mark, or by a skipped record (a 不标 decision);
- no two output marks overlap; every pending mark has candidates, a doubt and a stuck step;
- with --before: every change between the two files lies on an occurrence of a sheet name.
Prints the first problems found; exit code 1 if any.
"""
import argparse
import json
from pathlib import Path

from novel_ner.adjudicate import load_knowledge
from novel_ner.gold import SETS, own_set_path
from novel_ner.snapshot import REPO_ROOT, load_snapshot


def verify(prefilled, knowledge, before=None):
    problems = []
    snapshot = load_snapshot(prefilled["snapshot_id"], verify=False)
    typed = {(r["name"], r["type"]) for r in knowledge if r["type"] != "none"}
    for task in prefilled.get("llm_drafts", {}):
        drafts = {run: json.loads((REPO_ROOT / "data" / "llm_tasks" / task / f"draft_{run}.json").read_text(encoding="utf-8"))
                  for run in ("A", "B")}
        for cid in drafts["A"]["chapters"]:
            chap = prefilled["chapters"][cid]
            text = snapshot.chapters[cid].text
            anns = sorted(chap["annotations"], key=lambda a: (a["start"], a["end"]))
            sure = [{(s["start"], s["end"], s["type"]) for s in drafts[r]["chapters"][cid]["spans"] if not s["pending"]}
                    for r in ("A", "B")]
            for a, b in zip(anns, anns[1:]):
                if b["start"] < a["end"]:
                    problems.append(f"{cid}: overlap {a['text']} / {b['text']}")
            for a in anns:
                if text[a["start"]:a["end"]] != a["text"]:
                    problems.append(f"{cid}: text mismatch at {a['start']}")
                if a["pending"]:
                    if not (a["candidates"] and a["doubt"] and a["stuck"]):
                        problems.append(f"{cid}: pending {a['text']} lacks candidates/doubt/stuck")
                    continue
                key = (a["start"], a["end"], a["type"])
                if not (key in sure[0] and key in sure[1]) and (a["text"], a["type"]) not in typed:
                    problems.append(f"{cid}: confident {a['text']} {a['type']} came from neither both passes nor the sheet")
            covered = anns + chap.get("skipped", [])
            for r in ("A", "B"):
                for s in drafts[r]["chapters"][cid]["spans"]:
                    if not any(c["start"] < s["end"] and s["start"] < c["end"] for c in covered):
                        problems.append(f"{cid}: pass {r} span {s['text']} vanished")
            if before is not None and cid in before["chapters"]:
                names = [r["name"] for r in knowledge]
                spots = [(i, i + len(n)) for n in names for i in range(len(text)) if text.startswith(n, i)]
                old = {(a["start"], a["end"], a["type"], a["pending"]) for a in before["chapters"][cid]["annotations"]}
                new = {(a["start"], a["end"], a["type"], a["pending"]) for a in anns}
                for s, e, *_ in old ^ new:
                    if not any(x < e and s < y for x, y in spots):
                        problems.append(f"{cid}: change at [{s},{e}) {text[s:e]} is not on a sheet name")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefilled")
    ap.add_argument("--before")
    ap.add_argument("--knowledge", default=str(REPO_ROOT / "data" / "gold" / "name_knowledge.tsv"))
    ap.add_argument("--set", dest="set_name", choices=SETS, default=None,
                    help="the set the file belongs to (refused if the file says otherwise or lies under another set's folder)")
    args = ap.parse_args()
    prefilled = json.loads(Path(args.prefilled).read_text(encoding="utf-8"))
    before = json.loads(Path(args.before).read_text(encoding="utf-8")) if args.before else None
    if args.set_name is not None:
        for path, data in ((args.prefilled, prefilled), (args.before, before)):
            if path:
                own_set_path(path, args.set_name, REPO_ROOT)
                if data.get("set") != args.set_name:
                    raise SystemExit(f"{path} is a {data.get('set')} export, not {args.set_name}")
    problems = verify(prefilled, load_knowledge(args.knowledge), before)
    for p in problems[:40]:
        print(p)
    print("OK: all invariants hold" if not problems else f"{len(problems)} problem(s)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
