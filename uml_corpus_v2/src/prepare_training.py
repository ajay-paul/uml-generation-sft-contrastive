"""Prepare an isolated v2 training workspace; does not alter the old pilot."""

import argparse
import json
import shutil
from pathlib import Path

from corpus import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument(
        "--accept-provisional",
        action="store_true",
        help="Explicitly acknowledge that this is not independently reviewed gold",
    )
    a = p.parse_args()
    if not a.accept_provisional:
        p.error(
            "These are provisional source-derived annotations. Review them first, or explicitly pass --accept-provisional for an exploratory run."
        )
    out = Path(a.out)
    if out.exists():
        p.error("Use a new workspace; never overwrite a previous experiment")
    template = ROOT.parent / "uml_source_run_fold0"
    out.mkdir(parents=True)
    for name in ("src", "scripts"):
        shutil.copytree(
            template / name, out / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("requirements.txt", "models.lock.json"):
        shutil.copyfile(template / name, out / name)
    shutil.copyfile(ROOT / "src/corpus.py", out / "src/corpus.py")
    rows = []
    for split, filename in [
        ("train", "train.augmented.jsonl"),
        ("validation", "validation.original.jsonl"),
        ("test", "test.original.jsonl"),
    ]:
        for r in read(ROOT / "data" / f"fold{a.fold}" / filename):
            r["split"] = split
            r["domain"] = "source_model:" + r["document_id"]
            rows.append(r)
    write(out / "data/augmented.jsonl", rows)
    manifest = {
        "schema": "uml-projection-v2",
        "active_fold": a.fold,
        "split_unit": "source-family-and-near-duplicate-component",
        "annotation_status": "provisional",
        "test_augmented": False,
        "source_corpus": str(ROOT),
        "dataset_sha256": sha((out / "data/augmented.jsonl").read_bytes()),
    }
    (out / "data/corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {out.resolve()} with {len(rows)} rows. No model training has run.")


if __name__ == "__main__":
    main()
