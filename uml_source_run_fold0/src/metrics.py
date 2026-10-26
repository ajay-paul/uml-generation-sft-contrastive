"""Strict operational metrics. No automatic repair or fuzzy graph matching."""

import json

from common import facts, validate

CATEGORIES = [
    "classes",
    "attributes",
    "relations",
    "full_relations",
    "graph",
    "known_multiplicities",
]


def parse(text):
    def no_duplicates(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("Duplicate JSON key")
            d[k] = v
        return d

    try:
        g = json.loads(text, object_pairs_hook=no_duplicates)
    except (ValueError, TypeError):
        return None, False, False
    try:
        validate(g)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None, True, False
    return g, True, True


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"precision": p, "recall": r, "f1": f}


def score(text, gold):
    pred, syntax, valid = parse(text)
    out = {"json_valid": int(syntax), "uml_subset_valid": int(valid)}
    for k in CATEGORIES:
        g = facts(gold, k)
        p = facts(pred, k) if valid else set()
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        out[k] = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
        if not valid:
            out[k]["f1"] = 0.0
    out["exact_graph"] = int(valid and facts(pred, "graph") == facts(gold, "graph"))
    return out


def summarize(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    out = {"n": n, "independent_parents": len({r["parent_id"] for r in rows})}
    for k in ["json_valid", "uml_subset_valid", "exact_graph"]:
        out[k + "_rate"] = sum(r["scores"][k] for r in rows) / n
    for k in CATEGORIES:
        counts = {c: sum(r["scores"][k][c] for r in rows) for c in ["tp", "fp", "fn"]}
        out[k] = {
            "micro": {**counts, **prf(**counts)},
            "macro_f1": sum(r["scores"][k]["f1"] for r in rows) / n,
        }
    return out


def robustness(rows):
    families = {}
    for r in rows:
        families.setdefault(r["parent_id"], {})[r["variant"]] = r
    consistent = []
    correct = []
    cf = []
    for family in families.values():
        if not all(
            k in family for k in ["original", "paraphrase_1", "paraphrase_2", "counterfactual"]
        ):
            continue
        views = [family[k] for k in ["original", "paraphrase_1", "paraphrase_2"]]
        preds = [parse(r["output"])[0] for r in views]
        consistent.append(
            int(
                all(g is not None for g in preds)
                and all(facts(preds[0], "graph") == facts(g, "graph") for g in preds[1:])
            )
        )
        correct.append(int(all(r["scores"]["exact_graph"] for r in views)))
        cf.append(
            int(
                family["original"]["scores"]["exact_graph"]
                and family["counterfactual"]["scores"]["exact_graph"]
            )
        )

    def mean(x):
        return sum(x) / len(x) if x else None

    return {
        "parent_count": len(cf),
        "paraphrase_consistency": mean(consistent),
        "all_paraphrases_correct": mean(correct),
        "original_and_counterfactual_both_exact": mean(cf),
    }
