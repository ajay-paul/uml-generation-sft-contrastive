# Corpus preparation

This directory preserves the downloaded source corpus and its provisional preparation pipeline. The main experiment and ten-item results are documented in the [repository README](../README.md).

## Contents

- `sources/`: downloaded payloads, original notices, metadata and source locks.
- `src/fetch_sources.py`: restore source payloads using recorded URLs/checksums.
- `src/build_corpus.py`: extract text, project UML, screen pairs, assign grouped folds and augment training rows.
- `src/verify_corpus.py`: check source integrity, graph structure, negatives and partition separation.
- `src/prepare_training.py`: create a fresh experiment workspace using the sibling `uml_source_run_fold0` code.
- `data/`: 271 provisional retained pairs, 378 quarantined pairs and five prepared folds.
- `annotations/`: review sheets and explicitly labelled draft examples.
- `docs/`: source research and dataset documentation.
- `QA_REPORT.json`: original preparation report; see the root cleanup report for checks after formatting.

The 2,850 PURE records and 1,680 published user-story lines are an unlabelled exploratory pool, not supervised examples. None contributed to the reported training or testing. Retained English–UML references are provisional, not independently certified gold. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source terms and attribution.

## Checks

From this directory:

```bash
python src/verify_corpus.py
python -m unittest discover -s tests -v
```

## Rebuild

Rebuilding overwrites derived corpus files. Use a copy if preserving the historical preparation:

```bash
python src/build_corpus.py
python src/verify_corpus.py
```

The included source payloads suffice for the normal build. To restore missing payloads:

```bash
python src/fetch_sources.py
```

The optional `--original-documents` flag also downloads PURE's larger original-document archive, which is not needed for this experiment. Optional approved records in `annotations/adjudicated.jsonl` are consumed by the builder. Draft review material does not count as approval.

## Prepare a new run

```bash
python src/prepare_training.py --fold 0 --out ../../new_experiment --accept-provisional
```

The output must not already exist. This copies the canonical experiment scripts and the corpus module, writes a combined training/validation/test file and records its hash. It does not train a model. The supplied sibling experiment already contains the exact fold 0 data used in the reported run; rebuilding is not necessary to inspect the saved results.
