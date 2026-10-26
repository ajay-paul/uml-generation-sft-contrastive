"""Check all corpus outputs without loading an LLM or installing ML libraries."""

import json

from corpus import ROOT, norm, read, sha, signature, validate


def main():
    lock = json.loads((ROOT / "sources/files.lock.json").read_text())
    for name, expected in lock.items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("Source checksum mismatch: " + name)
    originals = read(ROOT / "data/paired_originals.provisional.jsonl")
    for r in originals:
        validate(r["target"])
        seen = {signature(r["target"])}
        assert len(r["negatives"]) == 4
        for negative in r["negatives"]:
            validate(negative["target"])
            key = signature(negative["target"])
            assert key not in seen
            seen.add(key)
    for fold in range(5):
        parts = {
            s: read(ROOT / "data" / f"fold{fold}" / (s + ".original.jsonl"))
            for s in ["train", "validation", "test"]
        }
        for key in ["id", "parent_id", "leakage_group"]:
            sets = {s: {r[key] for r in rs} for s, rs in parts.items()}
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                assert sets[a].isdisjoint(sets[b]), (fold, key, a, b)
        texts = {s: {norm(r["requirement"]) for r in rs} for s, rs in parts.items()}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            assert texts[a].isdisjoint(texts[b])
        augmented = read(ROOT / "data" / f"fold{fold}" / "train.augmented.jsonl")
        ids = set()
        for r in augmented:
            assert r["id"] not in ids
            ids.add(r["id"])
            assert r["split"] == "train" and r["parent_id"] in {
                x["parent_id"] for x in parts["train"]
            }
            validate(r["target"])
            seen = {signature(r["target"])}
            for n in r["negatives"]:
                validate(n["target"])
                key = signature(n["target"])
                assert key not in seen
                seen.add(key)
        sft = read(ROOT / "data" / f"fold{fold}" / "train_sft.jsonl")
        cl = read(ROOT / "data" / f"fold{fold}" / "train_contrastive.jsonl")
        assert [(x["id"], x["requirement"], x["target"]) for x in sft] == [
            (x["id"], x["requirement"], x["target"]) for x in cl
        ]
    print(
        f"PASS: source hashes; {len(originals)} provisional original pairs; all five grouped folds; train-only augmentation; valid unique negatives; identical SFT/CL positives."
    )
    print("These tests establish structural/integrity properties, .")


if __name__ == "__main__":
    main()
