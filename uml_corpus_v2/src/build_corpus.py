"""Build provisional supervised pairs and a separately unlabelled requirements pool."""

import collections
import csv
import difflib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from corpus import (
    ROOT,
    augment,
    negative_candidates,
    norm,
    project_ecore,
    read,
    screen,
    sha,
    signature,
    validate,
    write,
)


def ingest_requirements():
    rows = []
    inventory = []
    with zipfile.ZipFile(ROOT / "sources/pure/requirements-xml.zip") as z:
        for name in sorted(z.namelist()):
            if not name.startswith("XMLZIPFile/") or not name.endswith(".xml"):
                continue
            data = z.read(name)
            doc = "PURE-" + re.sub("[^A-Za-z0-9]+", "-", Path(name).stem).strip("-")
            root = ET.fromstring(data)
            counter = collections.Counter()

            def walk(e, ancestors):
                tag = e.tag.split("}")[-1]
                position = counter[tag]
                counter[tag] += 1
                eligible = (
                    tag == "req"
                    or tag == "text_body"
                    and "req" not in ancestors
                    and "text_body" not in ancestors
                )
                if eligible:
                    text = " ".join(" ".join(e.itertext()).split())
                    if len(text) >= 25:
                        rows.append(
                            {
                                "id": f"{doc}-{tag}-{position + 1:04d}",
                                "source_collection": "PURE",
                                "document_id": doc,
                                "requirement": text,
                                "source_url": "https://zenodo.org/records/7118517",
                                "source_file": name,
                                "source_sha256": sha(data),
                                "locator": {
                                    "element_tag": tag,
                                    "document_order_index": position + 1,
                                    "requirement_id": e.get("id"),
                                },
                                "record_kind": "explicit_req_element"
                                if tag == "req"
                                else "unclassified_prose",
                                "annotation_status": "unannotated",
                                "target": None,
                            }
                        )
                for child in e:
                    walk(child, ancestors + [tag])

            before = len(rows)
            walk(root, [])
            inventory.append(
                {
                    "collection": "PURE",
                    "document_id": doc,
                    "source_file": name,
                    "sha256": sha(data),
                    "extracted_records": len(rows) - before,
                }
            )
    for path in sorted((ROOT / "sources/stories").glob("*.txt")):
        data = path.read_bytes()
        doc = "US-" + path.stem
        before = len(rows)
        try:
            decoded = data.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            decoded = data.decode("cp1252")
            encoding = "cp1252"
        for i, line in enumerate(decoded.splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            rows.append(
                {
                    "id": f"{doc}-L{i:04d}",
                    "source_collection": "Dalpiaz-user-stories",
                    "document_id": doc,
                    "requirement": text,
                    "source_url": "https://zenodo.org/records/13880060",
                    "source_file": path.name,
                    "source_sha256": sha(data),
                    "source_encoding": encoding,
                    "locator": {"line": i},
                    "record_kind": "user_story",
                    "annotation_status": "unannotated",
                    "target": None,
                }
            )
        inventory.append(
            {
                "collection": "Dalpiaz-user-stories",
                "document_id": doc,
                "source_file": path.name,
                "sha256": sha(data),
                "extracted_records": len(rows) - before,
            }
        )
    for r in rows:
        r["text_sha256"] = sha(r["requirement"])
        r["normalized_text_cluster"] = sha(norm(r["requirement"]))
        r["annotation_priority_score"] = sum(
            bool(re.search(p, r["requirement"], re.I))
            for p in [
                r"\b(each|every)\b",
                r"\b(contains|consists|associated|belongs|owns)\b",
                r"\b(attribute|field|record|identifier|name)\b",
                r"\b(one|many|zero|multiple|several)\b",
            ]
        )
    write(ROOT / "data/requirements_pool.jsonl", rows)
    write(ROOT / "data/document_inventory.jsonl", inventory)
    grouped = collections.defaultdict(list)
    for r in rows:
        if 50 <= len(r["requirement"]) <= 1600:
            grouped[r["document_id"]].append(r)
    for rs in grouped.values():
        rs.sort(key=lambda r: (-r["annotation_priority_score"], r["id"]))
    queue = []
    for i in range(8):
        for doc in sorted(grouped):
            if i < len(grouped[doc]):
                queue.append(grouped[doc][i])
    write(ROOT / "data/annotation_queue.jsonl", queue)
    return rows, inventory, queue


def family(model):
    if "eirene" in model.lower():
        return "PURE-EIRENE"
    for prefix in [
        "MSOfficeWord",
        "MSOfficeExcel",
        "MSVisio",
        "UEML",
        "UMLDI",
        "QVT",
        "GWPN",
        "PNML",
        "DoDAF",
        "ODP",
        "OCL",
        "MiningMart",
    ]:
        if model.startswith(prefix):
            return prefix
    return model


def apply_adjudications(records, excluded):
    path = ROOT / "annotations/adjudicated.jsonl"
    if not path.exists():
        return records, excluded
    original = {r["id"]: r for r in records + excluded}
    pool = {r["id"]: r for r in read(ROOT / "data/requirements_pool.jsonl")}
    seen = set()
    for decision in read(path):
        key = decision["id"]
        if key in seen:
            raise ValueError("Repeated adjudication: " + key)
        seen.add(key)
        if (
            not decision.get("reviewer_1")
            or not decision.get("reviewer_2")
            or decision["reviewer_1"] == decision["reviewer_2"]
        ):
            raise ValueError("Two distinct recorded reviewers are required")
        if decision["decision"] not in ["accept", "reject"]:
            raise ValueError("Decision must be accept or reject")
        if key in original:
            r = dict(original[key])
        elif key in pool:
            r = dict(pool[key])
            r.update(
                parent_id=key,
                variant="original",
                source_family=family(r["document_id"]),
                review_status="provisional",
                reference_graph=None,
                projection_omissions=[],
                masked_unspecified_fields=[],
            )
        else:
            raise ValueError("Unknown source record: " + key)
        records = [x for x in records if x["id"] != key]
        excluded = [x for x in excluded if x["id"] != key]
        r["review_record"] = decision
        if decision["decision"] == "reject":
            r["quality_flags"] = ["rejected_by_recorded_review"]
            excluded.append(r)
            continue
        text = decision.get("requirement", r["requirement"])
        if text not in r["requirement"]:
            raise ValueError("Reviewed requirement must retain the source text or an exact excerpt")
        evidence = decision.get("evidence_quotes", [])
        if not evidence or any(not q or q not in text for q in evidence):
            raise ValueError("Evidence quotes must occur in the requirement")
        target = decision["target"]
        validate(target)
        negatives = negative_candidates(target)
        # Bind approvals to the exact graph/negative data, preventing stale reviews after edits.
        if decision.get("target_sha256") != signature(target):
            raise ValueError("Reviewed target hash mismatch")
        if decision.get("negative_sha256") != [signature(n["target"]) for n in negatives]:
            raise ValueError("Reviewed negative hashes mismatch")
        r.update(
            requirement=text,
            target=target,
            negatives=negatives,
            text_sha256=sha(text),
            quality_flags=[],
            review_status="two_reviewer_adjudication_record_supplied",
            independent_review=True,
        )
        records.append(r)
    return records, excluded


def build_pairs():
    source = ROOT / "sources/uml_specs"
    fragments = {r["unique_id"]: r for r in csv.DictReader((source / "fragments.csv").open())}
    labels = list(csv.DictReader((source / "labels.csv").open()))
    records = []
    excluded = []
    for label in labels:
        f = fragments[label["fragment_id"]]
        stem = f"{f['model']}_{f['kind']}{f['number']}"
        row = {
            "id": "UML-" + label["id"],
            "parent_id": "UML-F" + label["fragment_id"],
            "variant": "original",
            "source_collection": "Yang-2022-UML-English",
            "document_id": f["model"],
            "source_family": family(f["model"]),
            "requirement": label["label"].strip(),
            "source_url": "https://github.com/songyang-dev/uml-classes-and-specs/releases/tag/v1.0.0",
            "source_file": f"zoo/{stem}.ecore",
            "label_id": label["id"],
            "fragment_id": label["fragment_id"],
            "source_validation": {
                "in_english": label["in_english"],
                "validated": label["validated"],
            },
            "review_status": "provisional_automated_screen_only",
            "independent_review": False,
        }
        issues = []
        if label["in_english"] != "t" or label["validated"] != "t":
            issues.append("source_not_english_validated")
        try:
            path = source / "zoo" / (stem + ".ecore")
            graph, omissions = project_ecore(path)
            target, flags, masked = screen(row["requirement"], graph)
            issues += flags
            row.update(
                reference_graph=graph,
                target=target,
                projection_omissions=omissions,
                masked_unspecified_fields=masked,
                source_sha256=sha(path.read_bytes()),
                negatives=negative_candidates(target),
            )
            if len(row["negatives"]) != 4:
                issues.append("insufficient_distinct_negatives")
        except (ValueError, KeyError, ET.ParseError) as ex:
            issues.append("projection_error:" + str(ex))
            row["target"] = None
        row["quality_flags"] = issues
        row["text_sha256"] = sha(row["requirement"])
        (excluded if issues else records).append(row)
    records, excluded = apply_adjudications(records, excluded)
    families = {r["source_family"]: r["source_family"] for r in records}

    def find(x):
        while families[x] != x:
            families[x] = families[families[x]]
            x = families[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            families[max(a, b)] = min(a, b)

    matches = []
    for i, a in enumerate(records):
        na = norm(a["requirement"])
        for b in records[:i]:
            if a["source_family"] == b["source_family"]:
                continue
            nb = norm(b["requirement"])
            same = na == nb
            near = (
                min(len(na), len(nb)) >= 40
                and difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio() >= 0.92
            )
            if same or near:
                union(a["source_family"], b["source_family"])
                matches.append({"a": a["id"], "b": b["id"], "kind": "exact" if same else "near"})
    groups = collections.defaultdict(list)
    for r in records:
        groups[find(r["source_family"])].append(r)
    sizes = [0] * 5
    assignment = {}
    for group, rs in sorted(groups.items(), key=lambda x: (-len(x[1]), sha("42:" + x[0]))):
        fold = min(range(5), key=lambda i: (sizes[i], i))
        assignment[group] = fold
        sizes[fold] += len(rs)
    dedup = []
    seen = {}
    for r in records:
        r["leakage_group"] = find(r["source_family"])
        r["group_fold"] = assignment[r["leakage_group"]]
        key = (norm(r["requirement"]), signature(r["target"]))
        if key in seen:
            r["quality_flags"] = ["duplicate_pair_of:" + seen[key]]
            excluded.append(r)
        else:
            seen[key] = r["id"]
            dedup.append(r)
    write(ROOT / "data/paired_originals.provisional.jsonl", dedup)
    write(ROOT / "data/quarantine.jsonl", excluded)
    write(ROOT / "data/near_duplicate_links.jsonl", matches)
    return dedup, excluded, labels


def export(records, fold):
    out = ROOT / "data" / f"fold{fold}"
    splits = collections.defaultdict(list)
    for r in records:
        split = (
            "test"
            if r["group_fold"] == fold
            else "validation"
            if r["group_fold"] == (fold + 1) % 5
            else "train"
        )
        splits[split].append({**r, "split": split, "domain": "source_model:" + r["document_id"]})
    for split, rows in splits.items():
        write(out / (split + ".original.jsonl"), rows)
    train = [v for r in splits["train"] for v in augment(r)]
    write(out / "train.augmented.jsonl", train)
    for name in ["train_sft", "train_contrastive"]:
        write(
            out / (name + ".jsonl"),
            [
                {
                    "id": r["id"],
                    "parent_id": r["parent_id"],
                    "leakage_group": r["leakage_group"],
                    "requirement": r["requirement"],
                    "target": r["target"],
                    **({"negatives": r["negatives"]} if name.endswith("contrastive") else {}),
                }
                for r in train
            ],
        )
    return {
        "fold": fold,
        "originals": {k: len(v) for k, v in splits.items()},
        "groups": {k: len({r["leakage_group"] for r in v}) for k, v in splits.items()},
        "train_augmented": len(train),
        "train_negative_pairs": sum(len(r["negatives"]) for r in train),
        "augmentation_types": dict(collections.Counter(r["variant"] for r in train)),
    }


def main():
    rows, inventory, queue = ingest_requirements()
    pairs, excluded, labels = build_pairs()
    stats = {
        "status": "PROVISIONAL: NOT INDEPENDENTLY REVIEWED GOLD",
        "raw_requirements_or_prose": len(rows),
        "raw_by_collection": dict(collections.Counter(r["source_collection"] for r in rows)),
        "documents_by_collection": dict(collections.Counter(r["collection"] for r in inventory)),
        "annotation_queue": len(queue),
        "source_labels": len(labels),
        "provisional_original_pairs": len(pairs),
        "quarantined_pairs": len(excluded),
        "source_models_in_pairs": len({r["document_id"] for r in pairs}),
        "leakage_groups": len({r["leakage_group"] for r in pairs}),
        "quality_flags": dict(
            collections.Counter(f.split(":")[0] for r in excluded for f in r["quality_flags"])
        ),
        "relation_kinds": dict(
            collections.Counter(x["kind"] for r in pairs for x in r["target"]["relations"])
        ),
        "negative_mutations": dict(
            collections.Counter(n["mutation"] for r in pairs for n in r["negatives"])
        ),
        "negative_difficulty": dict(
            collections.Counter(n["difficulty"] for r in pairs for n in r["negatives"])
        ),
        "folds": [export(pairs, i) for i in range(5)],
    }
    (ROOT / "data/statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
