"""Audit and visualize one paired held-out fold/seed; never pool seeds or repair outputs."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from common import ROOT, SYSTEM, digest, read_jsonl, select_rows
from metrics import robustness, score, summarize

MODELS = ("llama", "qwen")
CONDITIONS = ("baseline", "sft", "contrastive")
LABELS = {"baseline": "Base checkpoint", "sft": "SFT", "contrastive": "SFT + contrastive"}
COLORS = {"baseline": "#64748b", "sft": "#2676b8", "contrastive": "#d57924"}
CATEGORIES = (
    "classes",
    "attributes",
    "relations",
    "full_relations",
    "graph",
    "known_multiplicities",
)


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    i = int(p)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (p - i)


def micro_f1(rows):
    counts = [sum(r["scores"]["full_relations"][k] for r in rows) for k in ("tp", "fp", "fn")]
    tp, fp, fn = counts
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def paired_gain(base, candidate, resamples=10000):
    b = {r["id"]: r for r in base}
    c = {r["id"]: r for r in candidate}
    if len(b) != len(base) or len(c) != len(candidate) or not b or b.keys() != c.keys():
        raise ValueError("Need identical unique held-out examples")
    groups = {}
    for key in sorted(b):
        if b[key]["leakage_group"] != c[key]["leakage_group"]:
            raise ValueError("Group mismatch")
        groups.setdefault(b[key]["leakage_group"], []).append(key)
    names = sorted(groups)
    rng = random.Random(2026)
    gains = []
    for _ in range(resamples):
        sample = [key for group in rng.choices(names, k=len(names)) for key in groups[group]]
        gains.append(100 * (micro_f1([c[i] for i in sample]) - micro_f1([b[i] for i in sample])))
    return {
        "parents": len(b),
        "independent_source_groups": len(names),
        "delta_pp": 100 * (micro_f1(candidate) - micro_f1(base)),
        "ci_low_pp": percentile(gains, 0.025),
        "ci_high_pp": percentile(gains, 0.975),
        "resamples": resamples,
        "bootstrap_seed": 2026,
    }


def audit_run(folder, model, condition, args):
    meta = json.loads((folder / "evaluation.json").read_text())
    for key, expected in {
        "model": model,
        "protocol": args.protocol,
        "fold": args.fold,
        "seed": args.seed,
        "split": "test",
        "quantization": "nf4",
    }.items():
        if meta.get(key) != expected:
            raise ValueError(f"{folder}: incompatible {key}")
    if not meta.get("decoding", "").startswith("explicit-greedy-v2"):
        raise ValueError(f"{folder}: rerun with corrected explicit greedy decoding")
    if meta["dataset_sha256"] != digest(ROOT / "data/augmented.jsonl"):
        raise ValueError(f"{folder}: dataset hash mismatch")
    if meta["system_prompt_sha256"] != hashlib.sha256(SYSTEM.encode()).hexdigest():
        raise ValueError(f"{folder}: system prompt hash mismatch")
    train = {}
    if condition == "baseline":
        if meta.get("adapter"):
            raise ValueError("Baseline must have no adapter")
    else:
        if not meta.get("adapter"):
            raise ValueError(f"{folder}: missing adapter provenance")
        train = json.loads((Path(meta["adapter"]).parent / "run.json").read_text())
        for key, expected in {
            "model": model,
            "method": condition,
            "fold": args.fold,
            "protocol": args.protocol,
            "seed": args.seed,
            "status": "complete",
            "revision": meta["revision"],
            "dataset_sha256": meta["dataset_sha256"],
            "quantization": meta["quantization"],
            "system_prompt_sha256": meta["system_prompt_sha256"],
        }.items():
            if train.get(key) != expected:
                raise ValueError(f"{folder}: adapter metadata mismatch: {key}")
    gold = {r["id"]: r for r in select_rows("test", args.fold, args.protocol, meta["views"])}
    predictions = read_jsonl(folder / "predictions.jsonl")
    if len(predictions) != len(gold) or {r["id"] for r in predictions} != set(gold):
        raise ValueError(f"{folder}: incomplete or duplicate test predictions")
    for r in predictions:
        target = gold[r["id"]]
        if any(r[k] != target[k] for k in ("parent_id", "variant")):
            raise ValueError(f"{folder}: incorrect parent/variant metadata")
        # Recompute strictly from raw output. Never trust manually edited score fields.
        r["scores"] = score(r["output"], target["target"])
        r["leakage_group"] = target["leakage_group"]
        if not math.isfinite(r["latency_seconds"]) or r["latency_seconds"] < 0:
            raise ValueError(f"{folder}: invalid latency")
    originals = sorted(
        (r for r in predictions if r["variant"] == "original"), key=lambda r: r["parent_id"]
    )
    summary = summarize(originals)
    row = {
        "model": model,
        "condition": condition,
        "parents": summary["independent_parents"],
        "independent_source_groups": len({r["leakage_group"] for r in originals}),
    }
    for category in CATEGORIES:
        for stat in ("tp", "fp", "fn", "precision", "recall", "f1"):
            row[f"{category}_micro_{stat}"] = summary[category]["micro"][stat]
        row[f"{category}_macro_f1"] = summary[category]["macro_f1"]
    for metric in ("json_valid_rate", "uml_subset_valid_rate", "exact_graph_rate"):
        row[metric] = summary[metric]
    latencies = [r["latency_seconds"] for r in originals]
    row["latency_original_p50_seconds"] = percentile(latencies, 0.5)
    row["latency_original_p95_seconds"] = percentile(latencies, 0.95)
    seconds = sum(latencies)
    row["generated_tokens_per_second_originals"] = (
        sum(r["generated_tokens"] for r in originals) / seconds if seconds else None
    )
    row["token_cap_rate_originals"] = sum(r["hit_token_cap"] for r in originals) / len(originals)
    saved_summary = json.loads((folder / "summary.json").read_text())
    row["generation_peak_allocated_gib_all_views"] = saved_summary["generation_peak_allocated_gib"]
    row["training_seconds"] = train.get("wall_seconds")
    row["training_peak_allocated_gib"] = train.get("peak_allocated_gib")
    row["best_epoch"] = train.get("best_epoch")
    robust = robustness(predictions)
    for name in (
        "paraphrase_consistency",
        "all_paraphrases_correct",
        "original_and_counterfactual_both_exact",
    ):
        row[name] = robust[name]
    ranked = [r for r in originals if "ranking" in r]
    row["ranking_recall_at_1_originals"] = (
        sum(r["ranking"]["rank"] == 1 for r in ranked) / len(ranked)
        if len(ranked) == len(originals)
        else None
    )
    row["ranking_mrr_originals"] = (
        sum(r["ranking"]["reciprocal_rank"] for r in ranked) / len(ranked)
        if len(ranked) == len(originals)
        else None
    )
    row["source"] = str(folder)
    return {
        "metrics": row,
        "originals": originals,
        "metadata": meta,
        "prediction_sha256": digest(folder / "predictions.jsonl"),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def charts(runs, comparisons, out, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    handles = [Patch(color=COLORS[c], label=LABELS[c]) for c in CONDITIONS]

    def save(fig, name, footer, legend=True):
        fig.suptitle(title, fontsize=13)
        if legend:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=3,
                frameon=False,
            )
        fig.text(0.5, 0.02, footer, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.07, 1, 0.86))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, key, caption, percent=True):
        maximum = 0
        for i, model in enumerate(MODELS):
            for j, condition in enumerate(CONDITIONS):
                value = runs[model, condition]["metrics"][key]
                x = i + (j - 1) * 0.25
                if value is None:
                    ax.text(x, 0, "N/A", ha="center", va="bottom", fontsize=8)
                    continue
                value *= 100 if percent else 1
                maximum = max(maximum, value)
                ax.bar(x, value, width=0.23, color=COLORS[condition])
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xticks(range(2), ["Llama 3.1 8B", "Qwen3 8B"])
        ax.set_title(caption)
        ax.set_ylim(0, 112 if percent else max(1, maximum * 1.22))
        if percent:
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars(axes[0], "known_multiplicities_micro_f1", "Known multiplicity endpoint micro-F1")
    bars(axes[1], "known_multiplicities_micro_recall", "Known multiplicity endpoint recall")
    save(
        fig,
        "08_known_multiplicities",
        "Only non-null multiplicity assertions count as facts; unknown fields earn no true positives.",
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, caption) in zip(
        axes.flat,
        [
            ("classes_micro_f1", "Class micro-F1"),
            ("attributes_micro_f1", "Attribute micro-F1"),
            ("relations_micro_f1", "Typed-relation micro-F1"),
            ("full_relations_micro_f1", "Full-relation micro-F1 (primary)"),
            ("graph_micro_f1", "Graph micro-F1"),
            ("exact_graph_rate", "Exact graph match"),
        ],
    ):
        bars(ax, key, caption)
    save(fig, "01_quality", "Original held-out requirements only; strict raw outputs; no repairs.")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, key, caption in zip(
        axes,
        ["json_valid_rate", "uml_subset_valid_rate", "token_cap_rate_originals"],
        ["JSON validity", "UML subset validity", "Token-cap rate (lower is better)"],
    ):
        bars(ax, key, caption)
    save(
        fig,
        "02_validity",
        "UML subset validation includes schema, endpoints, multiplicities and duplicate checks.",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, caption in zip(
        axes.flat,
        [
            "latency_original_p50_seconds",
            "latency_original_p95_seconds",
            "generated_tokens_per_second_originals",
            "generation_peak_allocated_gib_all_views",
        ],
        [
            "Median latency (seconds)",
            "95th-percentile latency (seconds)",
            "Generated tokens / second",
            "Peak allocated GPU memory (GiB; all views)",
        ],
    ):
        bars(ax, key, caption, percent=False)
    save(
        fig,
        "03_efficiency",
        "Latency/throughput: originals, after warm-up, excluding loading and ranking; single-run descriptive measurements.",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = []
    for i, row in enumerate(comparisons):
        ax.hlines(
            i, row["ci_low_pp"], row["ci_high_pp"], color=COLORS[row["candidate"]], linewidth=3
        )
        ax.plot(row["delta_pp"], i, "o", color=COLORS[row["candidate"]])
        labels.append(
            f"{row['model'].title()}: {LABELS[row['candidate']]} vs {LABELS[row['baseline']]}"
        )
    ax.axvline(0, linestyle="--", color="#777777")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Full-relation micro-F1 gain (percentage points); source-group paired 95% percentile interval"
    )
    ax.grid(axis="x", alpha=0.2)
    save(
        fig,
        "04_paired_gains",
        "Source-group cluster bootstrap within one fold/seed. Small pilot intervals are descriptive, not proof of population superiority.",
        False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, key, caption in zip(
        axes,
        [
            "paraphrase_consistency",
            "all_paraphrases_correct",
            "original_and_counterfactual_both_exact",
        ],
        ["Paraphrase consistency", "All paraphrases exact", "Original + counterfactual exact"],
    ):
        bars(ax, key, caption)
    save(
        fig,
        "05_robustness",
        "N/A for original-only v2 test; parent-level families; variants are not independent observations. Consistent outputs can still be incorrect.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars(axes[0], "ranking_recall_at_1_originals", "Candidate ranking Recall@1")
    bars(axes[1], "ranking_mrr_originals", "Candidate ranking MRR × 100")
    save(
        fig,
        "06_ranking",
        "Originals: gold vs four hard negatives using mean completion log-probability. Diagnostic; not generation accuracy.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars(axes[0], "training_seconds", "Adaptation wall time (seconds)", percent=False)
    bars(
        axes[1],
        "training_peak_allocated_gib",
        "Adaptation peak allocated GPU memory (GiB)",
        percent=False,
    )
    save(
        fig,
        "07_training_cost",
        "Recorded adaptation costs only; base-checkpoint pretraining cost is not measured (N/A).",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results_greedy_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--protocol", choices=["within", "domain"], default="within")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument(
        "--no-plots", action="store_true", help="Tables and audit only; no Matplotlib needed"
    )
    args = p.parse_args()
    if args.resamples < 100:
        p.error("Use at least 100 bootstrap resamples")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error("Choose a fresh report directory")
    runs = {}
    for model in MODELS:
        for condition in CONDITIONS:
            folder = (
                Path(args.root)
                / args.protocol
                / f"fold{args.fold}"
                / model
                / f"seed{args.seed}"
                / condition
            )
            runs[model, condition] = audit_run(folder, model, condition, args)
    for key in ("dataset_sha256", "system_prompt_sha256", "quantization", "decoding", "views"):
        if len({r["metadata"][key] for r in runs.values()}) != 1:
            raise ValueError(f"Incompatible conditions: {key}")
    for model in MODELS:
        for key in ("model_id", "revision"):
            if len({runs[model, c]["metadata"][key] for c in CONDITIONS}) != 1:
                raise ValueError(f"{model}: conditions use different {key}")
    comparisons = []
    for model in MODELS:
        for base, candidate in [
            ("baseline", "sft"),
            ("baseline", "contrastive"),
            ("sft", "contrastive"),
        ]:
            comparisons.append(
                {
                    "model": model,
                    "baseline": base,
                    "candidate": candidate,
                    **paired_gain(
                        runs[model, base]["originals"],
                        runs[model, candidate]["originals"],
                        args.resamples,
                    ),
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", [r["metrics"] for r in runs.values()])
    write_csv(out / "paired_comparisons.csv", comparisons)
    per_parent = []
    for (model, condition), run in runs.items():
        for r in run["originals"]:
            per_parent.append(
                {
                    "model": model,
                    "condition": condition,
                    "parent_id": r["parent_id"],
                    "full_relation_f1": r["scores"]["full_relations"]["f1"],
                    "exact_graph": r["scores"]["exact_graph"],
                    "json_valid": r["scores"]["json_valid"],
                    "uml_subset_valid": r["scores"]["uml_subset_valid"],
                    "latency_seconds": r["latency_seconds"],
                }
            )
    write_csv(out / "per_parent.csv", per_parent)
    (out / "audit.json").write_text(
        json.dumps(
            {
                "settings": vars(args),
                "runs": [
                    {
                        "model": m,
                        "condition": c,
                        "metadata": r["metadata"],
                        "prediction_sha256": r["prediction_sha256"],
                    }
                    for (m, c), r in runs.items()
                ],
                "metrics_module_sha256": digest(ROOT / "src/metrics.py"),
                "common_module_sha256": digest(ROOT / "src/common.py"),
                "warning": "Single-fold/seed pilot. Augmentations are not independent samples. No causal attribution from base-vs-contrastive alone.",
            },
            indent=2,
        )
        + "\n"
    )
    n = next(iter(runs.values()))["metrics"]["parents"]
    if not args.no_plots:
        charts(
            runs,
            comparisons,
            out,
            f"UML generation | {args.protocol}, fold {args.fold}, seed {args.seed} | {n} original fragments (grouped by source)",
        )
    print(f"Saved audited tables and requested charts to {out}")


if __name__ == "__main__":
    main()
