"""Plot saved ten-requirement evaluations; never invokes a language model."""

from pathlib import Path

CATEGORIES = (
    ("classes", "Class F1"),
    ("attributes", "Attribute F1"),
    ("relations", "Typed-relation F1"),
    ("full_relations", "Full-relation F1"),
    ("graph", "Graph F1"),
    ("known_multiplicities", "Known-multiplicity F1"),
)
CONDITIONS = ("baseline", "sft", "contrastive")
LABELS = ("Base checkpoint", "SFT", "SFT + contrastive")
COLORS = ("#64748b", "#2676b8", "#d57924")


def create_charts(out, comparison, runs, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out = Path(out) / "charts"
    out.mkdir(exist_ok=True)
    lookup = {(r["model"], r["condition"]): r for r in comparison}
    handles = [Patch(color=color, label=label) for color, label in zip(COLORS, LABELS)]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n = len(selected)

    def save(fig, name, title, footer, legend=True):
        fig.suptitle(f"{title} | {n} shared original test requirements", fontsize=14)
        if legend:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=3,
                frameon=False,
            )
        fig.text(0.5, 0.015, footer, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.085, 1, 0.88))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, key, title):
        for i, model in enumerate(("llama", "qwen")):
            for j, condition in enumerate(CONDITIONS):
                value = lookup[model, condition][key] * 100
                x = i + (j - 1) * 0.25
                ax.bar(x, value, width=0.23, color=COLORS[j])
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        ax.set_xticks([0, 1], ["Llama 3.1 8B", "Qwen3 8B"])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel("Percent")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [(k + "_micro_f1", t.replace("F1", "micro-F1")) for k, t in CATEGORIES[:5]]
    panels.append(("exact_graph_rate", "Exact graph match"))
    for ax, (key, title) in zip(axes.flat, panels):
        bars(ax, key, title)
    save(
        fig,
        "01_quality_micro",
        "Pooled fact quality",
        "Micro-F1 pools TP/FP/FN across tests. Exact match is the percentage of completely matching graphs.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, title) in zip(axes.flat, CATEGORIES):
        bars(ax, key + "_macro_f1", title.replace("F1", "mean F1"))
    save(
        fig,
        "02_mean_scores",
        "Mean scores across requirements (macro-F1)",
        "Each requirement has equal weight. Valid empty/empty categories score 100%; invalid outputs score 0 per item.\n"
        "A high mean for a sparse category can reflect many requirements with no facts of that category.",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars(ax, "graph_macro_f1", "Mean Graph F1 across all selected requirements")
    save(
        fig,
        "03_mean_graph",
        "Mean overall graph score",
        "Arithmetic mean of the per-requirement Graph F1 scores; not an average of the six metric categories.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, key, title in zip(
        axes,
        ("json_valid_rate", "uml_subset_valid_rate", "exact_graph_rate"),
        ("JSON validity", "UML subset validity", "Exact graph match"),
    ):
        bars(ax, key, title)
    save(
        fig,
        "04_validity",
        "Validity and complete correctness",
        "Structural validity does not establish semantic correctness. Exact match uses the provisional reference.",
    )

    configs = [(m, c) for m in ("llama", "qwen") for c in CONDITIONS]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, category, title in zip(
        axes,
        ("graph", "full_relations"),
        ("Graph F1 by requirement", "Full-relation F1 by requirement"),
    ):
        matrix = [[r["scores"][category]["f1"] * 100 for r in runs[key]] for key in configs]
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n), selected, rotation=35, ha="right")
        ax.set_yticks(
            range(6), [f"{m.capitalize()} / {LABELS[CONDITIONS.index(c)]}" for m, c in configs]
        )
        ax.set_title(title)
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(
                    x,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if value > 55 else "black",
                    fontsize=9,
                )
    save(
        fig,
        "05_per_requirement",
        "Individual test scores (%)",
        "All six conditions use the same IDs. Full-relation F1 is 100% for a valid empty prediction when the reference has no relations.",
        legend=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars(axes[0], "known_multiplicities_micro_f1", "Known-multiplicity micro-F1")
    bars(axes[1], "known_multiplicities_macro_f1", "Known-multiplicity mean F1")
    save(
        fig,
        "06_known_multiplicities",
        "Known endpoint multiplicities",
        "Only non-null bounds count as facts. Empty/empty categories score 100% under the existing evaluator.",
    )
    print(f"Saved six charts in PNG, PDF and SVG: {out}")
