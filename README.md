# Weak supervision and LLM teacher labels for Chinese web-novel NER

Named-entity recognition on a long Chinese web novel, where many names are invented by the author. The study asks how far a small extractive model gets when the only labels available are an incomplete dictionary and per-chapter name lists produced earlier by an LLM (names and types, no positions), how much a small set of span-level LLM teacher labels adds, and how the result compares with prompting LLMs directly.

The novel text, the answer keys and all model outputs are **not** included: the novel is copyrighted. This repository holds the code, the configurations, the annotation guideline and the final numbers.

The project was developed in a private repository; this is a snapshot of it published after the final evaluation, so the history here starts at that release.

## Task

- Six entity types: person, location, organization, item, dungeon, trait.
- Flat, position-level annotation; scoring is strict (chapter, start, end and type must all match), micro precision / recall / F1.
- Model: character-level BIO tagging (13 labels) on `hfl/chinese-roberta-wwm-ext` (~100M parameters), 500-character windows.

## Method

1. **Weak labels** (`src/novel_ner/weak.py`, `configs/weak_v2.json`). Dictionary entries and the chapter's LLM name list are matched back to the text and converted mechanically to the annotation guideline (titles after a person name are cut, category words are dropped, abbreviations of non-person names become uncertain). Uncertain positions get a partial-label loss: the model is not penalised as long as it puts its probability on the allowed labels (the "unknown" idea of AutoNER). Settings S1 (dictionary only) to S3 (dictionary + lists + uncertain regions) form an ablation; S3 is the weak-label system.
2. **Teacher labels** (`src/novel_ner/teacher.py`, `configs/teacher_train100.json`). An LLM annotates 100 training chapters position by position under the same guideline used for the answer keys; its "pending" marks become no-loss regions.
3. **Two-stage training** (`configs/train_s1_base_a3.json`). The S3 checkpoint (749 weakly labelled chapters) is fine-tuned on the 100 teacher-labelled chapters. This is the final model, A3.
4. **Comparisons** (`configs/recipes_v1.json`, `configs/consistency_v*.json`, `configs/train_s*.json`). Six rounds on a separate 25-chapter dev set: label source, number of teacher chapters, backbone (base / large, MacBERT, LERT), inference-time context, label-consistency rules, fine-tuning recipe. Each round's comparisons and decision rules were committed before scoring.

## Evaluation

- **Answer keys (silver standard).** An LLM labels every chapter twice in fresh sessions under a frozen guideline (`docs/annotation_guide/v1.1.md`, prompts in `docs/llm_annotation/`); spans both passes agree on are kept, disagreements are resolved with human book knowledge and a reviewer model.
- **Splits.** Training chapters 1–780 (749 training, 31 dev for checkpoint selection); a 25-chapter decision set from 781–990; a **sealed** 40-chapter test set from 996–1350, labelled before any system ran, locked in code, and scored once after the method was frozen (`configs/stage3_final.json`, `configs/test2_seal.json`).
- **Uncertainty.** Three seeds per trained system; paired bootstrap (10,000 rounds) over chapters stratified by chapter range, and a second interval that also resamples the seed (`src/novel_ner/posthoc.py`).

## Results (sealed test set, 40 chapters, 1,790 scored mentions)

| System | F1 | Precision | Recall | Non-person F1 | New-name recall |
|---|---|---|---|---|---|
| **A3**: weak labels (749 ch.) → teacher labels (100 ch.) | **0.908** | 0.923 | 0.892 | 0.807 | 0.851 |
| S3: weak labels only (749 ch.) | 0.897 | 0.908 | 0.886 | 0.770 | 0.821 |
| A2: teacher labels only (100 ch.) | 0.888 | 0.893 | 0.883 | 0.755 | 0.842 |
| Claude Opus, prompted with the full guideline | 0.942 | 0.978 | 0.909 | 0.880 | 0.884 |
| Claude Sonnet, prompted with the full guideline | 0.940 | 0.975 | 0.908 | 0.892 | 0.887 |
| GLiNER (multilingual), zero-shot | 0.606 | 0.594 | 0.620 | 0.304 | 0.455 |
| Dictionary match | 0.517 | 0.959 | 0.354 | 0.230 | 0 |

Trained systems are means over 3 seeds; LLMs are single runs. "New names" are answer strings found neither in the dictionary nor in the LLM name lists of chapters 1–780 (55% of the test mentions).

Paired differences in F1 (95% intervals: chapters / chapters and seeds):

- A3 − A2: +0.020 [0.0089, 0.0323] / [0.0075, 0.0338]
- A3 − S3: +0.010 [−0.0028, 0.0243] / [−0.0066, 0.0285]; new-name recall +0.030 [0.0128, 0.0517] / [0.0110, 0.0532]
- Opus − A3: +0.035 [0.0241, 0.0472] / [0.0234, 0.0483]; Sonnet − A3: +0.033 [0.0209, 0.0447] / [0.0204, 0.0460]

All numbers, including precision and non-person differences, are in `results/test2_final.json`.

## Limitations

- One novel by one author; dungeon (5 mentions) and trait (15) scores are not interpretable.
- The answer keys are a silver standard drafted under the same guideline the LLM systems receive in their prompt, and the LLMs may abstain; the gap to the LLMs mixes model capacity, context, input conditions and label quality, which were not separated.
- Three seeds; differences smaller than the seed spread are not attributed to methods.

## Layout

```
src/novel_ner/      code (weak labels, tagger, teacher labels, evaluation, bootstrap analysis, annotation tooling)
tests/              unit tests (no novel text)
configs/            training, weak-label, teacher-label, split and scoring configurations
docs/               annotation guideline v1.1 (Chinese) and the LLM annotation prompts
results/            final test-set numbers
```

## Running

Python 3.13 with PyTorch 2.11 (CUDA) and the packages in `requirements.txt`.

```
set PYTHONPATH=src
python -m pytest -q tests
```

Training and scoring expect a data snapshot under `data/snapshots/<id>/` (`chapters.jsonl` with per-chapter paragraphs, `weak/dictionary.jsonl`, `weak/mentions.jsonl`, `snapshot.json`), which is not distributed. With such a snapshot:

```
python -m novel_ner.weak configs/weak_v2.json
python -m novel_ner.tagger train configs/train_v2.json --setting S3 --seed 1
python -m novel_ner.tagger train configs/train_s1_base_a3.json --setting T --seed 1
python -m novel_ner.evaluate configs/eval_test2_final.json --run-id <new id>
```

## License

The code is released under the MIT License (see `LICENSE`). The license does not cover the novel, which is neither included nor licensed here.
