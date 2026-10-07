"""Teacher labels for the small tagger (stage 1, "position-level distillation"): Opus labels a fixed
sample of training chapters place by place under the exam's own drafting instructions, and its answers become a
weak-label build the tagger trains on next to the weak S3 labels of the very same chapters.

    set PYTHONPATH=src
    .venv\\Scripts\\python -m novel_ner.teacher sample configs/teacher_train100.json
    .venv\\Scripts\\python -m novel_ner.teacher build-packs configs/teacher_train100.json [--packs pilot|rest|all]
    .venv\\Scripts\\python -m novel_ner.teacher run configs/teacher_train100.json --packs pilot --run-id <id>
    .venv\\Scripts\\python -m novel_ner.teacher check configs/teacher_train100.json --packs pilot --run-id <id>
    .venv\\Scripts\\python -m novel_ner.teacher labels configs/teacher_train100.json --packs all --build-id 20261004-teacher100

Stage 2 (configs/teacher_train300.json, an extension config with base_config / base_chapters_file / base_labels_build):

    .venv\\Scripts\\python -m novel_ner.teacher extend configs/teacher_train300.json
    .venv\\Scripts\\python -m novel_ner.teacher build-packs configs/teacher_train300.json --packs new
    .venv\\Scripts\\python -m novel_ner.teacher run configs/teacher_train300.json --packs new --run-id <id>
    .venv\\Scripts\\python -m novel_ner.teacher check configs/teacher_train300.json --packs new|all300 --run-id <id>
    .venv\\Scripts\\python -m novel_ner.teacher labels configs/teacher_train300.json --packs all300 --build-id 20261004-teacher300

extend: the pool is chapters 1..max_train_chapter of the snapshot minus the dev chapters minus the base chapters file's
training chapters (the count the config's pool_rule states is checked). Two signals per pool chapter: signal_a = the
number of positive non-person spans in the weak S3 row of the chapter (the file the base config names; its sha256 is
checked against the weak build's record); signal_b = the uncertainty of the three student models the config's
signal_b.rule names (runs/<id>): each predicts every pool chapter in an ordinary predict run <model run>-predict-pool<N>
(reused when it exists with the same chapters and its recorded predictions sha256), and u = (union spans not predicted
by all three + spans all three predicted whose lowest score is below low_score) / (union spans), 0 when the union has
fewer than min_union_spans spans. Selection as the config's rule: list A = the n_new/2 highest signal_a (ties: lower
chapter number first), list B = the n_new/2 highest signal_b among the rest (same tie rule); packs of batch_size from
first_index: A sorted by signal_a descending, then B by signal_b descending. Every selected chapter is checked again
against the dev set, the base chapters, the test, dev2 and test2 chapters and max_train_chapter. Writes
configs/<config stem>.chapters.json (ids, numbers, signals, cuts, strata, overlap facts, the sha256 of the weak file and
of the predict runs' predictions, the code commit; the base packs are copied in so every command sees all packs), never
replaced unless --force. The extension file accepts --packs new (the new packs) and all300 (base + new training packs);
check and labels verify every base pack's pass against the sha256 the base labels build recorded (the old passes are
reused byte for byte, never re-run); check reports the counts and the weak agreement for the new packs and the base
packs separately besides the whole selection. No answer file, dev2, test or test2 is read by extend.

sample: the training pool is chapters 1-780 of the training snapshot minus the dev chapters, sorted by number and
cut into `strata` equal-count ranges; novel_ner.sample_test.stratified_order spreads the strata evenly through one
order (seed fixed), and the first n chapters go into packs of batch_size (teach_b01, teach_b02, ...). A dev pilot
pack (teach_dev_b01) takes dev_pilot.n of the dev config's chapters with its own seed, for measuring how far the
teacher agrees with the dev answers (the answers were drafted by GPT and adjudicated by the user, so that is
agreement with the exam setter's conventions, not absolute quality). Writes configs/<name>.chapters.json
(chapter ids and numbers only, committed) and refuses to change an existing one unless --force.
build-packs: one llm_task package per pack (the frozen instructions and guide, the chapters from the training
snapshot), after checking every chapter: a training pack holds chapters <= 780 that are not dev, test, dev2 or
test2 chapters; the dev pack holds dev chapters only. Existing packages are kept and reported, never overwritten.
run: claude_run.run_many on the chosen packs (one isolated Claude Code call per pack, pass_<label>.json, every
call logged under runs/<run id>/claude_calls.json). Packs that already have a pass are skipped by run_pass. The
seal of test2 is checked there too. For an extension chapters file only --packs new may run: the base packs' passes
are reused byte for byte and never re-run, so a selection holding a base pack is refused before any call.
check: for every chosen pack with a pass: JSON validity, evaluate.llm_predictions bookkeeping (mentions that
could not be located, bad types, chapters missing), mentions, pending count and share, overlapping places,
counts per type, and agreement with the weak S3 labels of the same chapters; for the dev pack, strict scores
against the dev answers (dev_pilot_scores, the only function here that opens a file under data/gold). Writes
runs/<run id>/metrics.json and config.json, counts only.
labels: data/weak_labels/<build id>/labels_T.jsonl (teacher rows) and labels_W.jsonl (the weak S3 rows of the
same chapters, copied from the weak build after checking that file's sha256 against its record), both split
"train", for the chosen training packs that have a pass. Never a dev chapter, never the dev pilot pack, never a
chapter after 780. The config's labels_build_id is reserved for --packs all; a pilot build (--packs pilot) must be
given its own --build-id, so the pilot can never take the full build's name. Conversion (CONVERSION_RULES): a located mention with one of the six types is a positive span
with sources ["teacher_<label>"]; a pending place is an uncertain region of unknown type (all 13 labels allowed,
so it carries no loss); positive spans that overlap each other or a pending place are merged into one uncertain
region of unknown type (reason teacher_overlap); exact repeats are merged; mentions that cannot be located or
have no valid type are dropped and counted; a chapter without a pass or without a mentions list is left out
and counted. runs/<build id>/config.json carries what tagger.train checks (snapshot, labels {T, W} with sha256,
split with train_ids_sha256) plus the rules, the packs, the sha256 of every pass file and the claude_calls run
ids; metrics.json holds the counts and the teacher-versus-weak agreement.

Material scope (the material-scope rules): the teacher sees the drafting instructions, the guide and the chapter
text, exactly what GPT saw when drafting the exam; never the name knowledge table, the answer files, the
adjudication records or the review lists. The labels builder opens nothing under data/gold; tests/test_teacher.py
runs it in a repository layout without that directory and checks the source of every function but
dev_pilot_scores for a gold reference. Everything is read from the training snapshot (to0990); chapters after
990 are never touched here.
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from novel_ner.annotate import chapter_order
from novel_ner.compare import ENTITY_TYPES
from novel_ner.evaluate import as_prediction, done_chapters, llm_predictions, sha256, summarize, type_counts, write_jsonl
from novel_ner.llm_task import TASK_ROOT, build as build_package
from novel_ner.sample_test import stratified_order
from novel_ner.score import _touches, strict
from novel_ner.snapshot import REPO_ROOT, load_snapshot, read_jsonl
from novel_ner.weak import LAST_TRAINING_CHAPTER, char_allowed, ids_sha256
from novel_ner.zeroshot import environment, git_state

SELECTIONS = ("pilot", "rest", "all", "new", "all300")  # pilot/rest/all: a sampled chapters file; new/all300: an extension file
INHERITED_KEYS = ("weak_build", "weak_setting")  # an extension config takes these from its base config when it lacks them
PREDICT_RUN_KEYS = ("run_id", "model_run", "predictions_sha256", "outputs", "reused")
CONVERSION_RULES = [
    "positive span: a mention evaluate.llm_predictions can place (paragraph id, text, occurrence) with a type among "
    "the six, sources [teacher_<label>]; an exact repeat (same place, same type) is merged and counted",
    "pending place (pending true, located): an uncertain region with types null and reason teacher_pending, so all 13 "
    "labels are allowed there and the loss skips it (the existing partial-label mechanism; a pending that cannot be "
    "located leaves no region and is counted)",
    "overlap: positive spans that overlap each other or a pending place are merged, with the pending places they "
    "touch, into one uncertain region over their union with types null and reason teacher_overlap (counted, with "
    "how many positives and pending places went in); pending places that overlap only each other merge into one "
    "teacher_pending region",
    "invalid mention (not an object, no text, bad occurrence, unknown paragraph, text not found, type missing or not "
    "one of the six): dropped and counted; it leaves no span and no region",
    "a chapter whose pass is missing, unreadable, or answered without a mentions list is left out of the build and "
    "counted; a mention listed under a chapter not in the pack is counted and ignored",
    "rows carry split train only; a dev chapter, the dev pilot pack or a chapter after 780 stops the build",
]


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_config(cfg, config_file):
    """The config with the keys every command reads filled in: an extension config (base_config) inherits
    INHERITED_KEYS from its base config when it lacks them, and chapters_file defaults to configs/<config
    stem>.chapters.json. The filled-in keys are listed under "inherited" so a record shows what came from where."""
    cfg = dict(cfg)
    inherited = {}
    if "base_config" in cfg:
        base = json.loads((REPO_ROOT / cfg["base_config"]).read_text(encoding="utf-8"))
        for key in INHERITED_KEYS:
            if key not in cfg:
                cfg[key] = inherited[key] = base[key]
    if "chapters_file" not in cfg:
        cfg["chapters_file"] = inherited["chapters_file"] = f"configs/{Path(config_file).stem}.chapters.json"
    if inherited:
        cfg["inherited"] = inherited
    return cfg


def chapters_file(cfg):
    return REPO_ROOT / cfg["chapters_file"]


def read_chapters(cfg):
    path = chapters_file(cfg)
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m novel_ner.teacher sample <config>` (or `extend` for an "
                         "extension config) first")
    return json.loads(path.read_text(encoding="utf-8"))


def is_extension(data):
    """Whether a chapters file was written by extend (base packs copied in, new packs added) rather than sample."""
    return "new_packs" in data


def full_selection(data):
    """The selection the config's labels_build_id is reserved for: all300 for an extension file, all otherwise."""
    return "all300" if is_extension(data) else "all"


TEACHER_BUILD_KIND = "teacher labels build (novel_ner.teacher labels)"  # build_labels' record "kind"; evaluate and posthoc key on it


def build_train_ids(build, chapters, root=None):
    """A teacher labels build's training chapter ids recomputed from its record, for the checks of evaluate.span_system
    and posthoc.training_chapters (weak.split_ids does not apply: the teacher chapters are a sample of the training
    chapters): the chapters of the training packs the record lists under teacher.packs (the packs that had a pass),
    read from the chapters file the record hashed (chapters_file.file; its sha256 must still match), each in the
    snapshot, at most config.max_train_chapter, no dev chapter of split.dev_chapters, never the dev pack, none twice;
    sorted by chapter number. The caller compares weak.ids_sha256 of the result with split.train_ids_sha256 (a chapter
    the build dropped for lacking a mentions list would make them differ). Raises KeyError or ValueError on a record
    it cannot recompute."""
    root = root or REPO_ROOT
    ref = build["chapters_file"]
    path = root / ref["file"]
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"{ref['file']} is not the chapters file the build hashed")
    data = json.loads(path.read_text(encoding="utf-8"))
    packs = list(build["teacher"]["packs"])
    if data["dev_pack"] in packs or len(set(packs)) != len(packs):
        raise ValueError("the dev pack is among the training packs, or a pack is listed twice")
    ids = [c for name in packs for c in data["packs"][name]]
    dev, limit = set(build["split"]["dev_chapters"]), build["config"]["max_train_chapter"]
    if len(set(ids)) != len(ids) or any(c not in chapters or c in dev or chapters[c].number > limit for c in ids):
        raise ValueError("a pack chapter is listed twice, missing from the snapshot, a dev chapter or after max_train_chapter")
    return sorted(ids, key=lambda c: chapters[c].number)


def dev_chapter_ids(cfg):
    """The dev chapters (configs/annotate_dev.json order), as a list."""
    return list(chapter_order(json.loads((REPO_ROOT / cfg["dev_config"]).read_text(encoding="utf-8"))))


def training_pool(snapshot, dev_ids, max_train_chapter):
    """Chapter ids 1..max_train_chapter minus the dev chapters, sorted by chapter number."""
    if max_train_chapter > LAST_TRAINING_CHAPTER:
        raise SystemExit(f"max_train_chapter {max_train_chapter} is past chapter {LAST_TRAINING_CHAPTER}")
    pool = [c for c in snapshot.chapters.values() if c.number <= max_train_chapter and c.chapter_id not in set(dev_ids)]
    return [c.chapter_id for c in sorted(pool, key=lambda c: c.number)]


def equal_count_strata(pool, k):
    """{stratum: [chapter ids]}: k consecutive ranges of the sorted pool whose counts differ by at most one."""
    if not 1 <= k <= len(pool):
        raise SystemExit(f"cannot cut {len(pool)} chapters into {k} strata")
    out = {}
    for i in range(k):
        out[f"s{i + 1:02d}"] = pool[i * len(pool) // k:(i + 1) * len(pool) // k]
    return out


def pack_names(cfg, count):
    return [f"{cfg['packs']['prefix']}{i + 1:02d}" for i in range(count)]


def sample(cfg, config_file, snapshot=None, force=False):
    """Draw the training chapters and the dev pilot, group them into packs and write the chapters file."""
    if "base_config" in cfg:
        raise SystemExit("an extension config (base_config) is drawn by `extend`, not `sample`")
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    dev_ids = dev_chapter_ids(cfg)
    missing = [c for c in dev_ids if c not in snapshot.chapters]
    if missing:
        raise SystemExit(f"dev chapters not in snapshot {snapshot.id}: {missing[:5]}")
    pool = training_pool(snapshot, dev_ids, cfg["max_train_chapter"])
    strata = equal_count_strata(pool, cfg["strata"])
    order = stratified_order(strata, cfg["seed"])
    n, size = cfg["n"], cfg["batch_size"]
    if n > len(order) or n % size:
        raise SystemExit(f"n {n} must be at most {len(order)} and a multiple of batch_size {size}")
    chosen = order[:n]
    names = pack_names(cfg, n // size)
    packs = {name: [cid for cid, _ in chosen[i * size:(i + 1) * size]] for i, name in enumerate(names)}
    pilot_cfg = cfg["dev_pilot"]
    dev_pilot = random.Random(pilot_cfg["seed"]).sample(sorted(dev_ids), pilot_cfg["n"])
    dev_pilot.sort(key=lambda c: snapshot.chapters[c].number)
    dev_pack = f"{cfg['packs']['dev_prefix']}01"
    packs[dev_pack] = dev_pilot
    pilot = names[:cfg["packs"]["pilot_training_packs"]] + [dev_pack]
    number = lambda c: snapshot.chapters[c].number
    sampled = Counter(stratum for _, stratum in chosen)
    data = {
        "config": Path(config_file).name, "snapshot": snapshot.id, "seed": cfg["seed"], "code_commit": git_state()["commit"],
        "pool": {"chapters": len(pool), "numbers": [number(pool[0]), number(pool[-1])],
                 "max_train_chapter": cfg["max_train_chapter"], "dev_chapters_excluded": len(dev_ids),
                 "rule": "chapters 1..max_train_chapter of the snapshot minus the dev config's chapters, sorted by number, "
                         "cut into equal-count ranges; novel_ner.sample_test.stratified_order spreads the strata evenly"},
        "strata": {s: {"chapters": len(ids), "numbers": [number(ids[0]), number(ids[-1])], "sampled": sampled.get(s, 0)}
                   for s, ids in strata.items()},
        "n": n, "batch_size": size,
        "order": [{"chapter_id": cid, "number": number(cid), "stratum": stratum} for cid, stratum in chosen],
        "training_packs": names, "dev_pack": dev_pack, "packs": packs, "pilot": pilot,
        "dev_pilot": {"n": pilot_cfg["n"], "seed": pilot_cfg["seed"], "chapters": dev_pilot,
                      "numbers": [number(c) for c in dev_pilot],
                      "source": f"{cfg['dev_config']} ({len(dev_ids)} dev chapters; that they have answers is checked by "
                                "`check`, which is the only command that opens the answers)"},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = chapters_file(cfg)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("packs") == packs and old.get("order") == data["order"]:
            print(f"{out} already holds this sample (unchanged)")
            return old
        if not force:
            raise SystemExit(f"{out} exists with a different sample; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {n} training chapters in {len(names)} packs, dev pilot {dev_pilot}")
    return data


def select_packs(data, which, with_dev=True):
    """Pack names for a selection. A sampled chapters file: pilot (the first training packs and the dev pack), rest
    (the other training packs) or all; with_dev False drops the dev pack. An extension file (extend): new (the new
    packs) or all300 (the base packs then the new packs; it has no dev pack)."""
    if is_extension(data):
        if which == "new":
            return list(data["new_packs"])
        if which == "all300":
            return list(data["training_packs"])
        raise SystemExit(f"--packs must be new or all300 for an extension chapters file, not {which!r}")
    train, dev = list(data["training_packs"]), data["dev_pack"]
    if which == "pilot":
        names = list(data["pilot"])
    elif which == "rest":
        names = [n for n in train if n not in data["pilot"]]
    elif which == "all":
        names = train + [dev]
    else:
        raise SystemExit(f"--packs must be pilot, rest or all for a sampled chapters file, not {which!r}")
    return [n for n in names if with_dev or n != dev]


def pool_size_from_rule(rule):
    """The chapter count a pool_rule states in parentheses, e.g. "(649 chapters)"."""
    m = re.search(r"\((\d+) chapters\)", rule)
    if not m:
        raise SystemExit("pool_rule must state the pool size as '(<n> chapters)'")
    return int(m.group(1))


def model_runs_from_rule(rule, count=3):
    """The run ids a signal rule names as runs/<id>, in order, exactly `count` distinct ones."""
    runs = re.findall(r"runs/([A-Za-z0-9][A-Za-z0-9._-]*)", rule)
    if len(runs) != count or len(set(runs)) != count:
        raise SystemExit(f"signal_b.rule must name exactly {count} distinct model runs as runs/<id>; found {runs}")
    return runs


def predict_suffix_from_rule(rule, pool_size):
    """The predict run suffix the rule states (<model run>-predict-pool<N>); N must be the pool size."""
    m = re.search(r"<model run>(-predict-pool(\d+))", rule)
    if not m:
        raise SystemExit("signal_b.rule must state the predict run ids as <model run>-predict-pool<N>")
    if int(m.group(2)) != pool_size:
        raise SystemExit(f"signal_b.rule names predict runs for a pool of {m.group(2)} chapters, the pool has {pool_size}")
    return m.group(1)


def non_person_spans(row):
    """signal_a of a weak labels row: its positive spans whose type is not person."""
    return sum(1 for s in row["spans"] if s["type"] != "person")


def predict_pool(model_run, chapter_ids, run_id):
    """One ordinary predict run of a trained model on training chapters (tagger.predict_run; imported here so the
    module stays light); returns its metrics."""
    from novel_ner.tagger import predict_run

    return predict_run(model_run, chapter_ids, run_id)


def read_predict_run(run_id):
    """(record, metrics, rows) of a finished predict run, after checking the predictions file against the recorded
    sha256; None when the run has no metrics yet."""
    run_dir = REPO_ROOT / "runs" / run_id
    if not (run_dir / "metrics.json").exists():
        return None
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    path = run_dir / "predictions.jsonl"
    if not path.exists() or sha256(path) != metrics.get("predictions_sha256"):
        raise SystemExit(f"{run_id}: predictions.jsonl is missing or differs from the sha256 its metrics recorded")
    return record, metrics, read_jsonl(path)


def pool_predictions(model_runs, pool_ids, suffix, snapshot_id):
    """The three students' predictions on the pool: {model run: {run_id, model_run, predictions_sha256, outputs, reused,
    rows}}. A run <model run><suffix> that exists is reused when its record lists exactly the pool chapters, names the
    model run, was made on snapshot_id and used the default chunking; otherwise it is made now with predict_pool."""
    out = {}
    for model_run in model_runs:
        run_id = f"{model_run}{suffix}"
        found = read_predict_run(run_id)
        reused = found is not None
        if not reused:
            predict_pool(model_run, list(pool_ids), run_id)
            found = read_predict_run(run_id)
            if found is None:
                raise SystemExit(f"{run_id}: the predict run left no metrics")
        record, metrics, rows = found
        if list(record.get("chapters") or []) != list(pool_ids):
            raise SystemExit(f"{run_id}: its chapters are not the pool (reuse refused; move the run away or rename it)")
        if record.get("model_run") != model_run:
            raise SystemExit(f"{run_id}: made by {record.get('model_run')}, not {model_run}")
        run_snapshot = record.get("snapshot")
        run_snapshot = run_snapshot.get("id") if isinstance(run_snapshot, dict) else run_snapshot
        if run_snapshot != snapshot_id:
            raise SystemExit(f"{run_id}: predicted on snapshot {run_snapshot}, not {snapshot_id} (reuse refused)")
        if record.get("chunking", "paragraph500") != "paragraph500":
            raise SystemExit(f"{run_id}: not the default chunking")
        out[model_run] = {"run_id": run_id, "model_run": model_run, "predictions_sha256": metrics["predictions_sha256"],
                          "outputs": len(rows), "reused": reused, "rows": rows}
    return out


def uncertainty(rows_by_run, pool_ids, low_score, min_union_spans):
    """signal_b per pool chapter from the seeds' prediction rows ({run: [rows]}): {cid: {"union_spans", "uncertain_spans",
    "signal_b"}}; a span is one (chapter, start, end, type) of the union; uncertain = not predicted by every run, or
    predicted by all with the lowest score below low_score; signal_b = uncertain / union, 0 under min_union_spans
    (the stored value is rounded to 4 decimals; the ranking uses the exact quotient, which the two counts give)."""
    runs = len(rows_by_run)
    scores = defaultdict(list)
    for rows in rows_by_run.values():
        seen = set()
        for r in rows:
            key = (r["chapter_id"], r["start"], r["end"], r["type"])
            if key in seen:
                raise SystemExit(f"{r['chapter_id']}: a predict run lists the same span twice")
            seen.add(key)
            scores[key].append(r["score"])
    union, uncertain = Counter(), Counter()
    for key, s in scores.items():
        union[key[0]] += 1
        if len(s) < runs or min(s) < low_score:
            uncertain[key[0]] += 1
    out = {}
    for cid in pool_ids:
        n, k = union.get(cid, 0), uncertain.get(cid, 0)
        exact = k / n if n >= min_union_spans and n else 0.0
        out[cid] = {"union_spans": n, "uncertain_spans": k, "signal_b": round(exact, 4), "_exact": exact}
    return out


def rank_lists(pool_ids, signal_a, signal_b, n_each, number):
    """(list A, list B): A = the n_each pool chapters with the highest signal_a, B = the n_each with the highest
    signal_b among the rest; ties within a list: the lower chapter number first; each list in its ranking order."""
    if 2 * n_each > len(pool_ids):
        raise SystemExit(f"cannot take two lists of {n_each} from a pool of {len(pool_ids)} chapters")
    a = sorted(pool_ids, key=lambda c: (-signal_a[c], number(c)))[:n_each]
    rest = [c for c in pool_ids if c not in set(a)]
    b = sorted(rest, key=lambda c: (-signal_b[c], number(c)))[:n_each]
    return a, b


def verify_new_chapters(ids, snapshot, dev_ids, base_ids, excluded, max_train_chapter):
    """Stop unless every new chapter is a training chapter of the snapshot that is not a dev, base, test, dev2 or
    test2 chapter, none twice."""
    if len(set(ids)) != len(ids):
        raise SystemExit("a chapter was selected twice")
    for cid in ids:
        ch = snapshot.chapters.get(cid)
        if ch is None:
            raise SystemExit(f"{cid} is not in snapshot {snapshot.id}")
        if ch.number > max_train_chapter:
            raise SystemExit(f"{cid} is after chapter {max_train_chapter}: refused")
        if cid in dev_ids:
            raise SystemExit(f"{cid} is a dev chapter: refused")
        if cid in base_ids:
            raise SystemExit(f"{cid} is one of the base chapters: refused")
        for key, members in excluded.items():
            if cid in members:
                raise SystemExit(f"{cid} is a {key} chapter: refused")


def strata_ranges(base_data):
    """{stratum: (lo, hi)} chapter-number ranges of the base chapters file."""
    return {s: tuple(v["numbers"]) for s, v in base_data["strata"].items()}


def stratum_of(number, ranges):
    for s, (lo, hi) in ranges.items():
        if lo <= number <= hi:
            return s
    return None


def extend(cfg, config_file, snapshot=None, force=False):
    """Rank the pool by the config's two signals, select the new chapters by its rule, form the new packs and write the
    extension chapters file (see the module docstring). Reads the weak labels file, the three students' predict runs
    (made here when missing) and the base chapters file; no answer file, dev2, test or test2 chapter."""
    if "base_config" not in cfg:
        raise SystemExit("extend needs an extension config (base_config, base_chapters_file, base_labels_build)")
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    number = lambda c: snapshot.chapters[c].number
    base_path = REPO_ROOT / cfg["base_chapters_file"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("snapshot") != snapshot.id or is_extension(base):
        raise SystemExit(f"{cfg['base_chapters_file']} is not a sampled chapters file of snapshot {snapshot.id}")
    base_packs = list(base["training_packs"])
    base_ids = [c for name in base_packs for c in base["packs"][name]]
    if len(set(base_ids)) != len(base_ids) or any(c not in snapshot.chapters for c in base_ids):
        raise SystemExit("the base chapters file lists a chapter twice or one not in the snapshot")
    dev_ids = dev_chapter_ids(cfg)
    if set(dev_ids) & set(base_ids):
        raise SystemExit("a base training chapter is a dev chapter")
    pool = [c for c in training_pool(snapshot, dev_ids, cfg["max_train_chapter"]) if c not in set(base_ids)]
    expected = pool_size_from_rule(cfg["pool_rule"])
    if len(pool) != expected:
        raise SystemExit(f"the pool has {len(pool)} chapters, the config's pool_rule says {expected}")
    sel = cfg["selection"]
    # signal_a: the weak labels file (sha256 checked against the weak build's record by weak_rows)
    wrows, weak_source = weak_rows(cfg, pool)
    if weak_source["file"] not in sel["signal_a"]["rule"]:
        raise SystemExit(f"signal_a.rule does not name the weak labels file {weak_source['file']}")
    signal_a = {c: non_person_spans(wrows[c]) for c in pool}
    # signal_b: the students' uncertainty on the pool
    model_runs = model_runs_from_rule(sel["signal_b"]["rule"])
    suffix = predict_suffix_from_rule(sel["signal_b"]["rule"], len(pool))
    preds = pool_predictions(model_runs, pool, suffix, snapshot.id)
    unc = uncertainty({m: p["rows"] for m, p in preds.items()}, pool, sel["signal_b"]["low_score"], sel["signal_b"]["min_union_spans"])
    signal_b = {c: unc[c]["_exact"] for c in pool}
    # selection
    n_new, packs_cfg = cfg["n_new"], sel["packs"]
    size, first = packs_cfg["batch_size"], packs_cfg["first_index"]
    if n_new % 2 or (n_new // 2) % size:
        raise SystemExit(f"n_new {n_new} must be even and each half a multiple of batch_size {size}")
    list_a, list_b = rank_lists(pool, signal_a, signal_b, n_new // 2, number)
    excluded = excluded_chapters(cfg)
    verify_new_chapters(list_a + list_b, snapshot, set(dev_ids), set(base_ids), excluded, cfg["max_train_chapter"])
    names = [f"{packs_cfg['prefix']}{first + i:02d}" for i in range(n_new // size)]
    if set(names) & set(base_packs):
        raise SystemExit("a new pack name is already a base pack name")
    chosen = list_a + list_b
    new_packs = {name: chosen[i * size:(i + 1) * size] for i, name in enumerate(names)}
    ranges = strata_ranges(base)
    in_list = {**{c: "A" for c in list_a}, **{c: "B" for c in list_b}}
    pack_of = {c: name for name, ids in new_packs.items() for c in ids}
    row = lambda c: {"chapter_id": c, "number": number(c), "list": in_list.get(c), "signal_a": signal_a[c],
                     "signal_b": unc[c]["signal_b"], "union_spans": unc[c]["union_spans"], "uncertain_spans": unc[c]["uncertain_spans"]}
    new_rows = [{**row(c), "pack": pack_of[c], "stratum": stratum_of(number(c), ranges)} for c in chosen]
    top_b_overall = sorted(pool, key=lambda c: (-signal_b[c], number(c)))[:n_new // 2]
    u_zero = lambda ids: sum(1 for c in ids if unc[c]["union_spans"] < sel["signal_b"]["min_union_spans"])
    cut_a, cut_b = signal_a[list_a[-1]], unc[list_b[-1]]["signal_b"]
    a_values = sorted(signal_a.values())
    strata = {s: {"numbers": list(r), "pool": sum(1 for c in pool if lo <= number(c) <= hi),
                  "base_sampled": base["strata"][s]["sampled"],
                  "new_a": sum(1 for c in list_a if lo <= number(c) <= hi), "new_b": sum(1 for c in list_b if lo <= number(c) <= hi)}
              for s, r in ranges.items() for lo, hi in [r]}
    for v in strata.values():
        v["new"] = v["new_a"] + v["new_b"]
    code = git_state()
    data = {
        "config": Path(config_file).name, "kind": "extension (novel_ner.teacher extend)", "snapshot": snapshot.id,
        "code_commit": code["commit"], "code_dirty": code["dirty"],
        "base": {"config": cfg["base_config"], "chapters_file": cfg["base_chapters_file"], "sha256": sha256(base_path),
                 "labels_build": cfg["base_labels_build"], "packs": base_packs, "chapters": len(base_ids)},
        "pool": {"chapters": len(pool), "numbers": [number(pool[0]), number(pool[-1])], "max_train_chapter": cfg["max_train_chapter"],
                 "dev_chapters_excluded": len(dev_ids), "base_chapters_excluded": len(base_ids), "rule": cfg["pool_rule"]},
        "signals": {
            "signal_a": {**sel["signal_a"], "source": weak_source,
                         "pool_spread": {"median": a_values[len(a_values) // 2], "p90": a_values[min(len(a_values) - 1, int(0.9 * len(a_values)))],
                                         "max": a_values[-1]}},
            "signal_b": {**sel["signal_b"], "model_runs": model_runs, "predict_run_suffix": suffix,
                         "predict_runs": {m: {k: p[k] for k in PREDICT_RUN_KEYS} for m, p in preds.items()},
                         "pool_with_u_0": u_zero(pool),
                         "value_rule": "signal_b = uncertain_spans / union_spans (0 when union_spans < min_union_spans); ranked on the "
                                       "exact quotient, stored rounded to 4 decimals"}},
        "selection_rule": sel["rule"],
        "n_new": n_new, "batch_size": size,
        "cut": {"signal_a_100th": cut_a, "signal_b_100th": cut_b,
                "note": f"the signal value of the last chapter of each list ({n_new // 2} per list)"},
        "lists": {"A": list_a, "B": list_b},
        "new": new_rows,
        "pool_signals": [row(c) for c in pool],
        "strata": strata,
        "overlap": {"list_a_in_top_by_signal_b": sum(1 for c in list_a if c in set(top_b_overall)),
                    "list_b_with_signal_a_at_cut_a": sum(1 for c in list_b if signal_a[c] == cut_a),
                    "list_b_with_signal_a_above_cut_a": sum(1 for c in list_b if signal_a[c] > cut_a),
                    "u_0": {"pool": u_zero(pool), "list_a": u_zero(list_a), "list_b": u_zero(list_b)},
                    "note": "list_a_in_top_by_signal_b: how many of A would also be in the top by signal_b over the whole pool; "
                            "list_b_with_signal_a_at_cut_a: B chapters that missed A only by the tie rule"},
        "base_packs": base_packs, "new_packs": names, "training_packs": base_packs + names, "dev_pack": None,
        "packs": {**{name: list(base["packs"][name]) for name in base_packs}, **new_packs},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = chapters_file(cfg)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("packs") == data["packs"] and old.get("new") == data["new"]:
            print(f"{out} already holds this extension (unchanged)")
            return old
        if not force:
            raise SystemExit(f"{out} exists with a different extension; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    first10 = lambda ids: [number(c) for c in ids[:10]]
    print(f"wrote {out}: pool {len(pool)}, {n_new} new chapters in {len(names)} packs {names[0]}..{names[-1]}")
    print(f"cut values: signal_a {cut_a} (pool median {data['signals']['signal_a']['pool_spread']['median']}, "
          f"p90 {data['signals']['signal_a']['pool_spread']['p90']}, max {a_values[-1]}); signal_b {cut_b}")
    print(f"strata (new A/B): " + ", ".join(f"{s} {v['new_a']}/{v['new_b']}" for s, v in strata.items()))
    print(f"u = 0: pool {data['overlap']['u_0']['pool']}, list A {data['overlap']['u_0']['list_a']}, list B {data['overlap']['u_0']['list_b']}")
    print(f"list A first 10 chapter numbers {first10(list_a)}; list B first 10 {first10(list_b)}")
    print(f"predict runs: " + ", ".join(f"{p['run_id']} ({'reused' if p['reused'] else 'made now'}, {p['outputs']} spans)" for p in preds.values()))
    return data


def base_pass_hashes(cfg, data, names, label):
    """Check that every base pack among `names` still has the pass file the base labels build recorded (same sha256;
    the old passes are reused byte for byte) and that the base chapters file is the one that build hashed; returns the
    reuse record. Only for an extension chapters file."""
    build = cfg["base_labels_build"]
    record_file = REPO_ROOT / "runs" / build / "config.json"
    if not record_file.exists():
        raise SystemExit(f"base labels build {build}: {record_file} is missing")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if record.get("kind") != TEACHER_BUILD_KIND or (record.get("teacher") or {}).get("label") != label:
        raise SystemExit(f"base labels build {build} is not a teacher labels build with label {label}")
    ref = record.get("chapters_file") or {}
    if ref.get("file") != data["base"]["chapters_file"] or sha256(REPO_ROOT / ref["file"]) != ref.get("sha256"):
        raise SystemExit(f"the base chapters file is not the one base labels build {build} hashed")
    expected = record["teacher"]["pass_files"]
    for name in names:
        if name not in data["base_packs"]:
            raise SystemExit(f"{name} is not a base pack")
        path = TASK_ROOT / name / f"pass_{label}.json"
        if name not in expected:
            raise SystemExit(f"{name}: base labels build {build} recorded no pass for it")
        if not path.exists() or sha256(path) != expected[name]["sha256"]:
            raise SystemExit(f"{name}: pass_{label}.json is missing or differs from the file base labels build {build} recorded; "
                             "the old passes are reused byte for byte, never re-run")
    return {"build": build, "record": record_file.relative_to(REPO_ROOT).as_posix(), "packs": list(names), "sha256_match": True,
            "rule": "every base pack's pass file has the sha256 the base labels build recorded; the base chapters file is the one it hashed"}


def excluded_chapters(cfg):
    """{"test": ids, "dev2": ids, "test2": ids}: the chapters no teacher pack may hold (the first n of each test
    order, the dev2 list), read from the configs the teacher config names."""
    out = {}
    for key, rel in cfg["exclude"].items():
        c = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        if "chapters" in c:
            out[key] = frozenset(c["chapters"])
        else:
            order = json.loads((REPO_ROOT / rel).with_suffix(".order.json").read_text(encoding="utf-8"))["order"]
            out[key] = frozenset(row["chapter_id"] for row in order[:c["n"]])
    return out


def check_pack(name, ids, snapshot, dev_ids, excluded, is_dev_pack, max_train_chapter):
    """Stop unless a training pack holds training chapters only (<= max_train_chapter, not dev, test, dev2 or
    test2) and a dev pack holds dev chapters only."""
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{name}: a chapter is listed twice")
    for cid in ids:
        for key, members in excluded.items():
            if cid in members:
                raise SystemExit(f"{name}: {cid} is a {key} chapter")
        ch = snapshot.chapters.get(cid)
        if ch is None:
            raise SystemExit(f"{name}: {cid} is not in snapshot {snapshot.id}")
        if ch.number > max_train_chapter:
            raise SystemExit(f"{name}: {cid} is after chapter {max_train_chapter}")
        if is_dev_pack and cid not in dev_ids:
            raise SystemExit(f"{name}: {cid} is not a dev chapter; the dev pack holds dev chapters only")
        if not is_dev_pack and cid in dev_ids:
            raise SystemExit(f"{name}: {cid} is a dev chapter; only the dev pack may hold one")


def build_packs(cfg, which="all", snapshot=None):
    """One task package per chosen pack; packages that exist are skipped and listed."""
    data = read_chapters(cfg)
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    if data["snapshot"] != snapshot.id:
        raise SystemExit(f"the chapters file was drawn from snapshot {data['snapshot']}, not {snapshot.id}")
    dev_ids = set(dev_chapter_ids(cfg))
    excluded = excluded_chapters(cfg)
    names = select_packs(data, which)
    for name in names:
        check_pack(name, data["packs"][name], snapshot, dev_ids, excluded, name == data["dev_pack"], cfg["max_train_chapter"])
    everywhere = [c for name in data["training_packs"] for c in data["packs"][name]]
    if len(set(everywhere)) != len(everywhere):
        raise SystemExit("a chapter is in two training packs of the chapters file")
    built, kept = [], []
    for name in names:
        if (TASK_ROOT / name).exists():
            kept.append(name)
            continue
        build_package(name, data["packs"][name], snapshot.id, cfg["frozen"]["instructions_version"], cfg["frozen"]["guide_version"])
        built.append(name)
    print(f"built {len(built)} packages {built}; kept {len(kept)} existing {kept}")
    return built, kept


def run_packs(cfg, which, run_id, label=None, model=None, effort=None, workers=None):
    """Opus on the chosen packs through claude_run.run_many; nothing is called here when the packages are missing."""
    from novel_ner.claude_run import run_many

    data = read_chapters(cfg)
    names = select_packs(data, which)
    if is_extension(data) and set(names) & set(data["base_packs"]):
        raise SystemExit("the base packs' passes are reused byte for byte and never re-run: for an extension chapters file "
                         "`run` takes --packs new only")
    tasks = [TASK_ROOT / n for n in names]
    missing = [t.name for t in tasks if not (t / "task.json").exists()]
    if missing:
        raise SystemExit(f"packages missing: {missing}; run build-packs first")
    t = cfg["teacher"]
    return run_many(tasks, label or t["label"], model or t["model"], effort or t["effort"], run_id, workers or t["workers"])


def read_pass(task_dir, label):
    """(answer dict or None, state): the stored pass of a package, state 'missing', 'invalid_json' or 'ok'."""
    path = Path(task_dir) / f"pass_{label}.json"
    if not path.exists():
        return None, "missing"
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None, "invalid_json"
    return (answer, "ok") if isinstance(answer, dict) else (None, "invalid_json")


def pass_places(answer, snapshot, chapter_ids):
    """Per chapter, what the teacher put where: {cid: {"has_list", "positives": [(s, e, type)], "pending": [(s, e)],
    "invalid": n, "abstained": n}}, with evaluate.llm_predictions' bookkeeping."""
    preds, book = llm_predictions(answer, snapshot, chapter_ids)
    with_list = {c.get("chapter_id") for c in (answer.get("chapters") or []) if isinstance(c, dict) and isinstance(c.get("mentions"), list)}
    out = {cid: {"has_list": cid in with_list, "positives": [], "pending": [], "invalid": book["invalid_by_chapter"].get(cid, 0),
                 "abstained": book["abstained_by_chapter"].get(cid, 0)} for cid in chapter_ids}
    for p in preds:
        if snapshot.chapters[p["chapter_id"]].text[p["start"]:p["end"]] != p["text"]:
            raise SystemExit(f"{p['chapter_id']} [{p['start']},{p['end']}): a located mention does not match the chapter text")
        out[p["chapter_id"]]["positives"].append((p["start"], p["end"], p["type"]))
    for cid, s, e in book["pending_places"]:
        out[cid]["pending"].append((s, e))
    return out, book


def cluster_places(positives, pending):
    """Groups of places that overlap (transitively): items (start, end, kind, type) sorted, kind positive/pending."""
    items = sorted([(s, e, "positive", t) for s, e, t in positives] + [(s, e, "pending", None) for s, e in pending])
    groups, current, end = [], [], -1
    for item in items:
        if current and item[0] < end:
            current.append(item)
            end = max(end, item[1])
        else:
            if current:
                groups.append(current)
            current, end = [item], item[1]
    if current:
        groups.append(current)
    return groups


def convert_chapter(chapter, positives, pending, source):
    """A teacher labels row for one chapter from its located positives [(s, e, type)] and pending places [(s, e)],
    following CONVERSION_RULES; returns (row, counts)."""
    counts = Counter()
    pos, pend = set(positives), set(pending)
    counts["duplicate_positives_merged"] = len(positives) - len(pos)
    counts["duplicate_pending_merged"] = len(pending) - len(pend)
    spans, uncertain = [], []
    for group in cluster_places(pos, pend):
        s, e = group[0][0], max(x[1] for x in group)
        kinds = Counter(x[2] for x in group)
        text = chapter.text[s:e]
        if len(group) == 1 and kinds["positive"]:
            spans.append({"start": s, "end": e, "type": group[0][3], "text": text, "sources": [source]})
        elif kinds["positive"]:
            uncertain.append({"start": s, "end": e, "text": text, "types": None, "reason": "teacher_overlap", "sources": [source]})
            counts["overlap_regions"] += 1
            counts["overlap_members_positive"] += kinds["positive"]
            counts["overlap_members_pending"] += kinds["pending"]
        else:
            uncertain.append({"start": s, "end": e, "text": text, "types": None, "reason": "teacher_pending", "sources": [source]})
            counts["pending_regions"] += 1
            counts["pending_places_merged_into_one_region"] += len(group) - 1
    char_allowed(len(chapter.text), spans, uncertain)  # raises if anything still overlaps
    row = {"chapter_id": chapter.chapter_id, "number": chapter.number, "split": "train", "spans": spans, "uncertain": uncertain}
    counts["spans"] = len(spans)
    for t in ENTITY_TYPES:
        counts[f"spans_{t}"] = sum(x["type"] == t for x in spans)
    return row, dict(counts)


def label_sets(row):
    """{"spans": {(s, e, type)}, "uncertain": [(s, e)]} of a labels row (teacher or weak)."""
    return {"spans": {(x["start"], x["end"], x["type"]) for x in row["spans"]},
            "uncertain": [(x["start"], x["end"]) for x in row["uncertain"]]}


def agreement_with_weak(teacher, weak):
    """Strict span+type agreement of the teacher's positives (after conversion: overlaps already merged into
    uncertain regions) with the weak positives of the same chapters (teacher = prediction, weak = reference; both
    {cid: label_sets}), read two ways: every teacher span counted, and teacher spans inside weak uncertain regions
    left out as the loss would leave them; plus how many teacher spans touch a weak uncertain region.
    fp_same_span_wrong_type counts the same place given another type by the two sources."""
    if not teacher:
        return None
    pred = {c: {"spans": teacher[c]["spans"]} for c in teacher}
    plain = strict({c: {"spans": weak[c]["spans"]} for c in teacher}, pred)
    masked = strict({c: {"spans": weak[c]["spans"], "unscored": weak[c]["uncertain"]} for c in teacher}, pred)
    brief = lambda m: {k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "fp_same_span_wrong_type")}
    return {"chapters": len(teacher), "teacher_spans": sum(len(teacher[c]["spans"]) for c in teacher),
            "weak_spans": sum(len(weak[c]["spans"]) for c in teacher),
            "weak_uncertain_regions": sum(len(weak[c]["uncertain"]) for c in teacher),
            "teacher_spans_in_weak_uncertain": sum(1 for c in teacher for s in teacher[c]["spans"] if _touches(s, weak[c]["uncertain"])),
            "teacher_vs_weak_positives": {**brief(plain), "per_type": plain["per_type"]},
            "teacher_vs_weak_positives_outside_weak_uncertain": brief(masked),
            "note": "the weak labels are not answers; this is agreement between two label sources on the same chapters"}


def weak_rows(cfg, chapter_ids, dev_rows_allowed=False):
    """The weak build's rows (setting cfg['weak_setting']) for chapter_ids, after checking the label file's sha256
    against the build's record and the record's snapshot; every chapter must have a row."""
    build = cfg["weak_build"]
    path = REPO_ROOT / "data" / "weak_labels" / build / f"labels_{cfg['weak_setting']}.jsonl"
    record_file = REPO_ROOT / "runs" / build / "config.json"
    if not path.exists() or not record_file.exists():
        raise SystemExit(f"weak build {build}: {path} or {record_file} is missing")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    digest = sha256(path)
    if (record.get("labels") or {}).get(cfg["weak_setting"], {}).get("sha256") != digest:
        raise SystemExit(f"{path.name} differs from the file weak build {build} recorded")
    built_snapshot = record.get("snapshot")
    built_snapshot = built_snapshot.get("id") if isinstance(built_snapshot, dict) else built_snapshot
    if built_snapshot != cfg["snapshot"]:
        raise SystemExit(f"weak build {build} is from snapshot {built_snapshot}, not {cfg['snapshot']}")
    rows = {r["chapter_id"]: r for r in read_jsonl(path)}
    missing = [c for c in chapter_ids if c not in rows]
    if missing:
        raise SystemExit(f"weak build {build} has no row for {missing[:5]}")
    if not dev_rows_allowed and any(rows[c].get("split") != "train" for c in chapter_ids):
        raise SystemExit(f"weak build {build}: a chosen chapter is not a training row")
    source = {"build": build, "setting": cfg["weak_setting"], "file": path.relative_to(REPO_ROOT).as_posix(), "sha256": digest,
              "record": record_file.relative_to(REPO_ROOT).as_posix(), "sha256_matches_record": True}
    return {c: rows[c] for c in chapter_ids}, source


def pass_provenance(task_dir, label):
    """The pass file's sha256 and the claude_calls run ids of the attempts that produced it (from the attempt
    records next to the package; numbers and ids only)."""
    task_dir = Path(task_dir)
    runs, attempts = set(), 0
    for f in sorted(task_dir.glob(f"claude_{label}_attempt*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8")).get("record") or {}
        except ValueError:
            continue
        attempts += 1
        if rec.get("run_id"):
            runs.add(rec["run_id"])
    return {"file": (task_dir / f"pass_{label}.json").relative_to(REPO_ROOT).as_posix() if task_dir.is_relative_to(REPO_ROOT)
            else f"pass_{label}.json", "sha256": sha256(task_dir / f"pass_{label}.json"), "attempt_files": attempts,
            "claude_calls_run_ids": sorted(runs)}


def read_task(name, data, cfg):
    """The package's task.json, checked against the chapters file and the frozen versions."""
    task_dir = TASK_ROOT / name
    if not (task_dir / "task.json").exists():
        raise SystemExit(f"{name}: no package at {task_dir}; run build-packs first")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if task["chapters"] != data["packs"][name]:
        raise SystemExit(f"{name}: the package's chapters differ from the chapters file")
    if task["snapshot_id"] != cfg["snapshot"]:
        raise SystemExit(f"{name}: package from snapshot {task['snapshot_id']}, not {cfg['snapshot']}")
    frozen = cfg["frozen"]
    if (task.get("instructions_version"), task.get("guide_version")) != (frozen["instructions_version"], frozen["guide_version"]):
        raise SystemExit(f"{name}: the package's instructions or guide version is not the frozen one")
    return task_dir, task


def dev_pilot_scores(gold_file, snapshot, chapter_ids, places, book):
    """Strict scores of the teacher on the dev pilot chapters against the dev answers (score.strict with
    score.gold_spans; the answers' pending marks and unscored ranges are left out as evaluate does). Scored on the
    teacher's located positives as evaluate scores a compared system (every located, typed mention counts; the
    conversion's overlap merging is not applied here), with its unlocated mentions as wrong predictions and its
    pending places as abstentions. The only function in this module that opens a file under data/gold; called
    for the dev pack alone."""
    from novel_ner.gold import check as check_gold
    from novel_ner.score import gold_spans

    path = REPO_ROOT / gold_file
    export = json.loads(path.read_text(encoding="utf-8"))
    errors, _, _ = check_gold(export, snapshot)
    if errors:
        raise SystemExit(f"the dev answers do not pass the gold check: {errors[:3]}")
    done = set(done_chapters(export))
    missing = [c for c in chapter_ids if c not in done]
    if missing:
        raise SystemExit(f"dev pilot chapters without finished answers: {missing}")
    gold = gold_spans(export, chapter_ids)
    preds = [{"chapter_id": c, "start": s, "end": e, "type": t} for c in chapter_ids for s, e, t in places[c]["positives"]]
    pred = as_prediction(preds, chapter_ids)
    for c in chapter_ids:
        pred[c]["errors"] = places[c]["invalid"]
        pred[c]["pending"] = list(places[c]["pending"])
    m = strict(gold, pred)
    summary = summarize(list(type_counts(gold, pred).values()))
    return {"gold_file": gold_file, "gold_sha256": sha256(path), "chapters": chapter_ids,
            "gold_provenance": {c: export["chapters"][c].get("provenance") for c in chapter_ids},
            "gold_spans": sum(len(g["spans"]) for g in gold.values()),
            "gold_unscored_regions": sum(len(g["unscored"]) for g in gold.values()),
            "strict": {k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "invalid_predictions",
                                         "fp_same_span_wrong_type", "fp_overlapping_gold", "fn_where_draft_was_pending")},
            "micro": summary["micro"], "non_person_micro": summary["non_person_micro"], "macro_f1": summary["macro_f1"],
            "per_type": summary["per_type"],
            "note": "agreement with the dev answers (GPT-drafted, user-adjudicated): the exam setter's conventions, not absolute quality"}


TOTAL_KEYS = ("chapters", "mentions", "positives", "pending", "pending_located", "pending_unlocated", "invalid",
              "overlap_regions", "duplicate_positives_merged", "chapters_without_a_mentions_list", "spans_after_conversion")


def pack_totals(packs):
    """The counts of several packs' check entries added up, with the pending share and the per-type positives."""
    total = Counter()
    for p in packs.values():
        for k in TOTAL_KEYS:
            total[k] += p[k]
    return {**total, "packs": len(packs), "pending_share_of_mentions": round(total["pending"] / total["mentions"], 4) if total["mentions"] else None,
            "positives_by_type": {t: sum(p["positives_by_type"][t] for p in packs.values()) for t in ENTITY_TYPES},
            "positive_mentions_by_type": {t: sum(p["positive_mentions_by_type"][t] for p in packs.values()) for t in ENTITY_TYPES},
            "problems": dict(sum((Counter(p["problems"]) for p in packs.values()), Counter()))}


def check(cfg, config_file, which, run_id, label=None, snapshot=None):
    """Counts for every chosen pack that has a pass, agreement with the weak labels, dev pilot scores for the
    dev pack; for an extension chapters file also the new packs and the base packs apart (by_group), after checking
    the base packs' passes against the base labels build; runs/<run_id>/ gets metrics.json and config.json (counts
    only)."""
    label = label or cfg["teacher"]["label"]
    out_dir = REPO_ROOT / "runs" / run_id
    if (out_dir / "metrics.json").exists():
        raise SystemExit(f"{out_dir} already has results; use a new run id")
    data = read_chapters(cfg)
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    names = select_packs(data, which)
    packs, missing, provenance = {}, [], {}
    teacher_train, teacher_dev = {}, {}
    dev_places, dev_book, dev_ids = {}, {"invalid_by_chapter": {}, "abstained_by_chapter": {}, "pending_places": []}, []
    for name in names:
        task_dir, task = read_task(name, data, cfg)
        answer, state = read_pass(task_dir, label)
        if answer is None:
            missing.append({"pack": name, "state": state})
            continue
        provenance[name] = pass_provenance(task_dir, label)
        chapters = task["chapters"]
        places, book = pass_places(answer, snapshot, chapters)
        mentions = sum(len(c["mentions"]) for c in answer.get("chapters") or [] if isinstance(c, dict) and isinstance(c.get("mentions"), list))
        counts = Counter()
        for cid in chapters:
            if not places[cid]["has_list"]:
                counts["chapters_without_a_mentions_list"] += 1
                continue
            row, c = convert_chapter(snapshot.chapters[cid], places[cid]["positives"], places[cid]["pending"], f"teacher_{label}")
            counts.update(c)
            (teacher_dev if name == data["dev_pack"] else teacher_train)[cid] = label_sets(row)
        positives = sum(len(p["positives"]) for p in places.values())
        pending_located = sum(len(p["pending"]) for p in places.values())
        abstained = sum(p["abstained"] for p in places.values())
        packs[name] = {
            "chapters": len(chapters), "json_valid": True, "mentions": mentions, "positives": positives,
            "positives_by_type": {t: counts.get(f"spans_{t}", 0) for t in ENTITY_TYPES},  # after overlap merging
            "positive_mentions_by_type": {t: sum(1 for p in places.values() for x in p["positives"] if x[2] == t) for t in ENTITY_TYPES},
            "pending": abstained, "pending_located": pending_located, "pending_unlocated": abstained - pending_located,
            "pending_share_of_mentions": round(abstained / mentions, 4) if mentions else None,
            "invalid": sum(p["invalid"] for p in places.values()), "problems": book["problems"],
            "chapters_missing": book["chapters_missing"],
            "overlap_regions": counts.get("overlap_regions", 0),
            "overlap_members": {"positive": counts.get("overlap_members_positive", 0), "pending": counts.get("overlap_members_pending", 0)},
            "duplicate_positives_merged": counts.get("duplicate_positives_merged", 0),
            "duplicate_pending_merged": counts.get("duplicate_pending_merged", 0),
            "chapters_without_a_mentions_list": counts.get("chapters_without_a_mentions_list", 0),
            "spans_after_conversion": counts.get("spans", 0), "pending_regions": counts.get("pending_regions", 0),
        }
        if name == data["dev_pack"]:
            dev_ids, dev_places, dev_book = chapters, places, book
    totals = pack_totals(packs)
    weak_source, weak, agreement = None, {}, {"training_packs": None, "dev_pack": None}
    if teacher_train or teacher_dev:
        wrows, weak_source = weak_rows(cfg, sorted(teacher_train) + sorted(teacher_dev), dev_rows_allowed=True)
        weak = {c: label_sets(r) for c, r in wrows.items()}
        agreement = {"training_packs": agreement_with_weak(teacher_train, weak), "dev_pack": agreement_with_weak(teacher_dev, weak)}
    by_group, base_passes = None, None
    if is_extension(data):  # the new packs and the base packs reported apart, besides the whole selection
        base_passes = base_pass_hashes(cfg, data, [n for n in names if n in data["base_packs"]], label)
        by_group = {}
        for group in ("new_packs", "base_packs"):
            sub = {n: packs[n] for n in data[group] if n in packs}
            rows = {c: teacher_train[c] for n in sub for c in data["packs"][n] if c in teacher_train}
            by_group[group] = {"packs": list(sub), "totals": pack_totals(sub) if sub else None,
                               "agreement_with_weak": agreement_with_weak(rows, weak) if rows else None}
    dev_pilot, dev_pilot_skipped = None, None
    if not dev_ids:
        dev_pilot_skipped = "the dev pack is not in the selection or has no pass"
    elif not all(dev_places[c]["has_list"] for c in dev_ids):
        dev_pilot_skipped = "a dev pilot chapter was answered without a mentions list"
    else:
        dev_pilot = dev_pilot_scores(cfg["gold_dev"], snapshot, dev_ids, dev_places, dev_book)
    metrics = {"run_id": run_id, "label": label, "selection": which, "packs": packs, "packs_without_pass": missing,
               "totals": totals, "agreement_with_weak": agreement, "by_group": by_group,
               "dev_pilot": dev_pilot, "dev_pilot_skipped": dev_pilot_skipped,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record = {"run_id": run_id, "config_file": Path(config_file).as_posix() if config_file else None, "config": cfg,
              "chapters_file": {"file": cfg["chapters_file"], "sha256": sha256(chapters_file(cfg))},
              "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
              "label": label, "selection": which, "packs": list(packs), "pass_files": provenance,
              "claude_calls_run_ids": sorted({r for p in provenance.values() for r in p["claude_calls_run_ids"]}),
              "weak_source": weak_source, "base_passes": base_passes, "code": git_state(),
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, p in packs.items():
        print(f"{name}: {p['mentions']} mentions, {p['positives']} positives, {p['pending']} pending, {p['invalid']} invalid, "
              f"{p['overlap_regions']} overlap regions, problems {p['problems']}")
    if missing:
        print(f"no pass yet: {[m['pack'] for m in missing]}")
    if agreement["training_packs"]:
        a = agreement["training_packs"]["teacher_vs_weak_positives"]
        print(f"teacher vs weak positives (training packs): P={a['precision']} R={a['recall']} F1={a['f1']}")
    for group, g in (by_group or {}).items():
        t = g["totals"]
        if t:
            a = g["agreement_with_weak"]["teacher_vs_weak_positives"] if g["agreement_with_weak"] else None
            print(f"{group}: {t['packs']} packs, {t['chapters']} chapters, positives by type {t['positives_by_type']}, pending {t['pending']} "
                  f"({t['pending_share_of_mentions']} of mentions), invalid {t['invalid']}, overlap regions {t['overlap_regions']}"
                  + (f"; vs weak P={a['precision']} R={a['recall']} F1={a['f1']}" if a else ""))
    if dev_pilot:
        print(f"dev pilot vs answers: P={dev_pilot['strict']['precision']} R={dev_pilot['strict']['recall']} F1={dev_pilot['strict']['f1']}")
    print(f"wrote {out_dir}")
    return metrics


def build_labels(cfg, config_file, which, build_id, label=None, snapshot=None):
    """labels_T.jsonl and labels_W.jsonl for the chosen training packs that have a pass, with the record
    tagger.train checks. Opens no answer file (tests check the source of this function for that)."""
    label = label or cfg["teacher"]["label"]
    data = read_chapters(cfg)
    full = full_selection(data)
    if which != full and build_id == cfg.get("labels_build_id"):
        raise SystemExit(f"build id {build_id} is the config's labels_build_id, reserved for --packs {full} (the full build); "
                         "give a partial build its own --build-id")
    out_dir = REPO_ROOT / "data" / "weak_labels" / build_id
    run_dir = REPO_ROOT / "runs" / build_id
    if out_dir.exists() or (run_dir / "config.json").exists():
        raise SystemExit(f"{out_dir} or {run_dir} exists; use a new build id (builds are never overwritten)")
    snapshot = snapshot or load_snapshot(cfg["snapshot"])
    dev_ids = set(dev_chapter_ids(cfg))
    names = select_packs(data, which, with_dev=False)
    if not names:
        raise SystemExit("no training packs in the selection")
    reused = base_pass_hashes(cfg, data, [n for n in names if n in data["base_packs"]], label) if is_extension(data) else None
    rows, counts, provenance, missing = [], Counter(), {}, []
    for name in names:
        if name == data["dev_pack"]:
            raise SystemExit("the dev pilot pack never enters a labels build")
        task_dir, task = read_task(name, data, cfg)
        answer, state = read_pass(task_dir, label)
        if answer is None:
            missing.append({"pack": name, "state": state})
            counts["chapters_dropped_pass_" + state] += len(task["chapters"])
            continue
        provenance[name] = pass_provenance(task_dir, label)
        places, book = pass_places(answer, snapshot, task["chapters"])
        counts["mentions_under_a_chapter_not_in_the_pack"] += book["problems"].get("mentions_under_a_chapter_not_in_this_task", 0)
        for cid in task["chapters"]:
            chapter = snapshot.chapters[cid]
            if chapter.number > cfg["max_train_chapter"] or cid in dev_ids:
                raise SystemExit(f"{name}: {cid} is a dev chapter or after chapter {cfg['max_train_chapter']}; refused")
            if not places[cid]["has_list"]:
                counts["chapters_dropped_no_mentions_list"] += 1
                continue
            row, c = convert_chapter(chapter, places[cid]["positives"], places[cid]["pending"], f"teacher_{label}")
            counts.update(c)
            counts["invalid_mentions_dropped"] += places[cid]["invalid"]
            counts["pending_unlocated_dropped"] += places[cid]["abstained"] - len(places[cid]["pending"])
            rows.append(row)
    if not rows:
        raise SystemExit("no chapter with a teacher pass in the selection")
    ids = [r["chapter_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a chapter appears in two packs")
    wrows, weak_source = weak_rows(cfg, ids)
    out_dir.mkdir(parents=True)
    files = {}
    for setting, setting_rows in (("T", rows), ("W", [wrows[c] for c in ids])):
        path = out_dir / f"labels_{setting}.jsonl"
        write_jsonl(path, setting_rows)
        files[setting] = {"file": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(path)}
    numbers = [r["number"] for r in rows]
    uncertain_by_reason = Counter(u["reason"] for r in rows for u in r["uncertain"])
    weak_uncertain = Counter(u["reason"] for c in ids for u in wrows[c]["uncertain"])
    metrics = {
        "run_id": build_id, "chapters": len(rows), "packs": len(provenance), "packs_without_pass": missing,
        "settings": {
            "T": {"chapters": len(rows), "spans": sum(len(r["spans"]) for r in rows),
                  "spans_by_type": {t: counts.get(f"spans_{t}", 0) for t in ENTITY_TYPES},
                  "uncertain": sum(uncertain_by_reason.values()), "uncertain_by_reason": dict(uncertain_by_reason),
                  "overlap_regions": counts.get("overlap_regions", 0),
                  "overlap_members": {"positive": counts.get("overlap_members_positive", 0), "pending": counts.get("overlap_members_pending", 0)},
                  "duplicate_positives_merged": counts.get("duplicate_positives_merged", 0),
                  "duplicate_pending_merged": counts.get("duplicate_pending_merged", 0),
                  "pending_places_merged_into_one_region": counts.get("pending_places_merged_into_one_region", 0),
                  "invalid_mentions_dropped": counts.get("invalid_mentions_dropped", 0),
                  "pending_unlocated_dropped": counts.get("pending_unlocated_dropped", 0),
                  "mentions_under_a_chapter_not_in_the_pack": counts.get("mentions_under_a_chapter_not_in_the_pack", 0),
                  "chapters_dropped": {k[len("chapters_dropped_"):]: v for k, v in counts.items() if k.startswith("chapters_dropped_")}},
            "W": {"chapters": len(ids), "spans": sum(len(wrows[c]["spans"]) for c in ids),
                  "spans_by_type": {t: sum(x["type"] == t for c in ids for x in wrows[c]["spans"]) for t in ENTITY_TYPES},
                  "uncertain": sum(weak_uncertain.values()), "uncertain_by_reason": dict(weak_uncertain)}},
        "agreement_T_vs_W": agreement_with_weak({r["chapter_id"]: label_sets(r) for r in rows}, {c: label_sets(wrows[c]) for c in ids}),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    record = {
        "run_id": build_id, "config_file": Path(config_file).as_posix() if config_file else None, "config": cfg,
        "kind": "teacher labels build (novel_ner.teacher labels)",
        "snapshot": {"id": snapshot.id, "source_commit": snapshot.meta.get("source_commit")},
        "code": git_state(), "environment": environment(),
        "split": {"max_train_chapter": cfg["max_train_chapter"], "train_chapters": len(ids), "train_numbers": [min(numbers), max(numbers)],
                  "train_ids_sha256": ids_sha256(ids), "train_ids_sha256_rule": "sha256 of the sorted chapter ids joined by \\n",
                  "dev_chapters": sorted(dev_ids, key=lambda c: snapshot.chapters[c].number), "selection": which,
                  "rule": "the chosen training packs' chapters that have a teacher pass; identical for T and W"},
        "labels": files,
        "teacher": {"label": label, "source": f"teacher_{label}", "model": cfg["teacher"]["model"], "effort": cfg["teacher"]["effort"],
                    "instructions_version": cfg["frozen"]["instructions_version"], "guide_version": cfg["frozen"]["guide_version"],
                    "packs": list(provenance), "pass_files": provenance,
                    "claude_calls_run_ids": sorted({r for p in provenance.values() for r in p["claude_calls_run_ids"]}),
                    "reused_passes": reused},
        "weak_source": weak_source,
        "chapters_file": {"file": cfg["chapters_file"], "sha256": sha256(chapters_file(cfg))},
        "conversion_rules": CONVERSION_RULES,
        "material_scope": "instructions + guide + chapter text only (what GPT saw when drafting the exam); no name knowledge "
                          "table, answers, adjudication records or review lists; this builder opens no answer file "
                          "(the material-scope rules; tests/test_teacher.py checks it)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    t = metrics["settings"]["T"]
    print(f"{build_id}: {len(rows)} chapters; T {t['spans']} spans, {t['uncertain']} uncertain {t['uncertain_by_reason']}; "
          f"W {metrics['settings']['W']['spans']} spans; wrote {out_dir} and {run_dir}")
    if missing:
        print(f"left out (no pass): {[m['pack'] for m in missing]}; chapters dropped {t['chapters_dropped']}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("sample")
    s.add_argument("config")
    s.add_argument("--force", action="store_true", help="replace an existing chapters file that differs")
    e = sub.add_parser("extend", help="rank the pool by the config's signals and add the new packs (extension config)")
    e.add_argument("config")
    e.add_argument("--force", action="store_true", help="replace an existing chapters file that differs")
    b = sub.add_parser("build-packs")
    b.add_argument("config")
    b.add_argument("--packs", choices=SELECTIONS, default="all")
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--packs", choices=SELECTIONS, required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--label")
    r.add_argument("--model")
    r.add_argument("--effort")
    r.add_argument("--workers", type=int)
    c = sub.add_parser("check")
    c.add_argument("config")
    c.add_argument("--packs", choices=SELECTIONS, default="all")
    c.add_argument("--run-id", required=True)
    c.add_argument("--label")
    l = sub.add_parser("labels")
    l.add_argument("config")
    l.add_argument("--packs", choices=SELECTIONS, default="all")
    l.add_argument("--build-id", help="default: the config's labels_build_id")
    l.add_argument("--label")
    args = ap.parse_args()
    config_file = Path(args.config).as_posix()
    cfg = resolve_config(read_config(args.config), config_file)
    if args.action == "sample":
        sample(cfg, config_file, force=args.force)
    elif args.action == "extend":
        extend(cfg, config_file, force=args.force)
    elif args.action == "build-packs":
        build_packs(cfg, args.packs)
    elif args.action == "run":
        run_packs(cfg, args.packs, args.run_id, args.label, args.model, args.effort, args.workers)
    elif args.action == "check":
        check(cfg, config_file, args.packs, args.run_id, args.label)
    else:
        build_labels(cfg, config_file, args.packs, args.build_id or cfg["labels_build_id"], args.label)


if __name__ == "__main__":
    main()
