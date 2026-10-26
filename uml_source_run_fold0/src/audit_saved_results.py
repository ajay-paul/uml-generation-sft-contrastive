"""Recalculate the reported scores without loading model weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from common import ROOT, SYSTEM, digest, read_jsonl
from metrics import CATEGORIES, score, summarize


def audit(out):
    selection = json.loads((out / "selection.json").read_text())
    assert selection["dataset_sha256"] == digest(ROOT / "data/augmented.jsonl")
    assert selection["selection_sha256"] == digest(out / "selected_test.jsonl")
    assert selection["system_prompt_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest()
    selected = read_jsonl(out / "selected_test.jsonl")
    assert [row["id"] for row in selected] == selection["ids"]
    assert len(selected) == len({row["parent_id"] for row in selected}) == 10
    comparison = {
        (row["model"], row["condition"]): row
        for row in csv.DictReader((out / "comparison.csv").open())
    }
    count = 0
    for model in ("llama", "qwen"):
        for condition in ("baseline", "sft", "contrastive"):
            folder = out / model / condition
            predictions = read_jsonl(folder / "predictions.jsonl")
            assert [row["id"] for row in predictions] == selection["ids"]
            for index, (prediction, reference) in enumerate(zip(predictions, selected), 1):
                item = folder / f"{index:02d}_{reference['id']}"
                assert (item / "raw_output.txt").read_text().strip() == prediction["output"].strip()
                assert json.loads((item / "reference.json").read_text()) == reference["target"]
                recalculated = score(prediction["output"], reference["target"])
                assert recalculated == prediction["scores"], (model, condition, reference["id"])
                assert recalculated == json.loads((item / "scores.json").read_text())
                count += 1
            totals = summarize(predictions)
            saved = comparison[model, condition]
            for category in CATEGORIES:
                for aggregation in ("micro", "macro"):
                    value = (
                        totals[category]["micro"]["f1"]
                        if aggregation == "micro"
                        else totals[category]["macro_f1"]
                    )
                    assert abs(value - float(saved[f"{category}_{aggregation}_f1"])) < 1e-12
            for key in ("json_valid_rate", "uml_subset_valid_rate", "exact_graph_rate"):
                assert abs(totals[key] - float(saved[key])) < 1e-12
    print(
        f"PASS: {count} saved completions, reference files, metadata hashes and comparison scores."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_test10_run1")
    audit(parser.parse_args().out)
