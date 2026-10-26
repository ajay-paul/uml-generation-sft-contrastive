"""Ten held-out requirements; raw generation, strict scoring, exact-match diagrams."""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

from common import ROOT, SYSTEM, digest, norm, select_rows, write_jsonl
from metrics import parse, score, summarize


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plantuml(graph):
    # Internal aliases avoid special characters in class identifiers.
    aliases = {norm(c["name"]): f"C{i}" for i, c in enumerate(graph["classes"])}

    def safe(value):
        return (
            str(value).replace("\\", "\\\\").replace('"', "'").replace("\n", " ").replace("\r", " ")
        )

    lines = ["@startuml", "hide empty methods", "skinparam classAttributeIconSize 0"]
    for c in graph["classes"]:
        lines.append(f'class "{safe(c["name"])}" as {aliases[norm(c["name"])]} {{')
        for a in c["attributes"]:
            suffix = "" if a["type"] is None else " : " + safe(a["type"])
            lines.append("  " + safe(a["name"]) + suffix)
        lines.append("}")
    for r in graph["relations"]:
        source, target = aliases[norm(r["source"])], aliases[norm(r["target"])]
        if r["kind"] == "generalization":
            lines.append(f"{source} --|> {target}")
        else:
            sm = "" if r["source_multiplicity"] is None else f' "{safe(r["source_multiplicity"])}"'
            tm = "" if r["target_multiplicity"] is None else f' "{safe(r["target_multiplicity"])}"'
            arrow = "*--" if r["kind"] == "composition" else "--"
            lines.append(f"{source}{sm} {arrow}{tm} {target}")
    return "\n".join(lines + ["@enduml", ""])


def prepare(a):
    if a.out.exists():
        raise FileExistsError("Choose a new output directory; existing results are preserved.")
    rows = sorted(select_rows("test", a.fold, "within", "original"), key=lambda r: r["id"])
    # Choose distinct parents AND distinct requirement texts before seeing any predictions.
    unique, parents, texts = [], set(), set()
    for r in rows:
        text = " ".join(r["requirement"].split()).casefold()
        if r["parent_id"] not in parents and text not in texts:
            unique.append(r)
            parents.add(r["parent_id"])
            texts.add(text)
    if len(unique) < 10:
        raise ValueError("Fewer than 10 distinct original test requirements.")
    chosen = random.Random(a.seed).sample(unique, 10)
    a.out.mkdir(parents=True)
    write_jsonl(a.out / "selected_test.jsonl", chosen)
    save(
        a.out / "selection.json",
        {
            "fold": a.fold,
            "seed": a.seed,
            "dataset_sha256": digest(ROOT / "data/augmented.jsonl"),
            "selection_sha256": digest(a.out / "selected_test.jsonl"),
            "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "ids": [r["id"] for r in chosen],
            "selection": "seeded random sample; original test rows only",
            "diagram_rule": "valid and exact normalized graph match to provisional reference",
        },
    )
    (a.out / "requirements.txt").write_text(
        "\n\n".join(f"{i:02d}. {r['id']}\n{r['requirement']}" for i, r in enumerate(chosen, 1))
        + "\n",
        encoding="utf-8",
    )
    print(f"Selected 10 shared test requirements: {a.out / 'requirements.txt'}")


def run(a):
    import torch
    from modeling import MODELS, load, prompt_ids, revision, tokenizer_for
    from transformers import GenerationConfig, set_seed

    selection = json.loads((a.out / "selection.json").read_text())
    assert (selection["fold"], selection["seed"]) == (a.fold, a.seed), "Selection settings changed"
    assert selection["dataset_sha256"] == digest(ROOT / "data/augmented.jsonl"), "Dataset changed"
    assert selection["selection_sha256"] == digest(a.out / "selected_test.jsonl"), (
        "Selection changed"
    )
    assert selection["system_prompt_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest(), (
        "Prompt changed"
    )
    rows = [json.loads(s) for s in (a.out / "selected_test.jsonl").read_text().splitlines()]
    folder = a.out / a.model / a.condition
    if folder.exists():
        raise FileExistsError(f"Results already exist: {folder}")
    adapter = None
    if a.condition != "baseline":
        adapter = ROOT / f"runs/within/fold{a.fold}/{a.model}/seed{a.seed}/{a.condition}/best"
        meta = json.loads((adapter.parent / "run.json").read_text())
        expected = {
            "model": a.model,
            "method": a.condition,
            "fold": a.fold,
            "seed": a.seed,
            "protocol": "within",
            "status": "complete",
            "dataset_sha256": selection["dataset_sha256"],
            "system_prompt_sha256": selection["system_prompt_sha256"],
            "quantization": "nf4",
        }
        for k, v in expected.items():
            if meta.get(k) != v:
                raise ValueError(f"Adapter metadata mismatch: {k}")
    set_seed(a.seed)
    tok = tokenizer_for(a.model)
    model = load(a.model, str(adapter) if adapter else None, quantization="nf4")
    model.eval()
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=1024,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    folder.mkdir(parents=True)
    save(
        folder / "evaluation.json",
        {
            "model": MODELS[a.model],
            "revision": revision(a.model),
            "adapter": str(adapter) if adapter else None,
            "condition": a.condition,
            "quantization": "nf4",
            "decoding": "greedy; 1024 new tokens; Qwen thinking disabled",
            **selection,
        },
    )
    records = []
    with (
        torch.inference_mode(),
        (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream,
    ):
        for i, r in enumerate(rows, 1):
            ids = torch.tensor(
                [prompt_ids(tok, a.model, r["requirement"])],
                device=model.get_input_embeddings().weight.device,
            )
            if ids.shape[1] + 1024 > 2048:
                raise ValueError(f"{r['id']}: exceeds context budget; no truncation")
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=cfg,
                use_model_defaults=False,
                do_sample=False,
                num_beams=1,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            tokens = output[0, ids.shape[1] :]
            raw = tok.decode(tokens, skip_special_tokens=True).strip()
            scores = score(raw, r["target"])
            item = folder / f"{i:02d}_{re.sub(r'[^a-zA-Z0-9_-]', '_', r['id'])}"
            item.mkdir()
            (item / "requirement.txt").write_text(r["requirement"] + "\n", encoding="utf-8")
            (item / "raw_output.txt").write_text(raw + "\n", encoding="utf-8")
            save(item / "reference.json", r["target"])
            save(item / "scores.json", scores)
            record = {
                "id": r["id"],
                "parent_id": r["parent_id"],
                "variant": r["variant"],
                "output": raw,
                "scores": scores,
                "generation_seconds": elapsed,
                "diagram_status": "not created: no exact reference match",
            }
            if scores["exact_graph"]:
                graph = parse(raw)[0]
                save(item / "graph.json", graph)
                puml = item / "diagram.puml"
                puml.write_text(plantuml(graph), encoding="utf-8")
                record["diagram_status"] = "PlantUML saved; install plantuml for SVG rendering"
                if shutil.which("plantuml"):
                    try:
                        result = subprocess.run(
                            ["plantuml", "-tsvg", str(puml.resolve())],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        (item / "render.log").write_text(
                            result.stdout + result.stderr, encoding="utf-8"
                        )
                        record["diagram_status"] = (
                            "SVG rendered"
                            if result.returncode == 0 and puml.with_suffix(".svg").exists()
                            else "PlantUML saved; SVG render failed (see render.log)"
                        )
                    except (subprocess.TimeoutExpired, OSError) as exc:
                        record["diagram_status"] = f"PlantUML saved; renderer failed: {exc}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
            print(
                f"{a.model} {a.condition} {r['id']}: valid={scores['uml_subset_valid']} exact={scores['exact_graph']} full_relation_F1={scores['full_relations']['f1']:.3f}",
                flush=True,
            )
    save(folder / "summary.json", summarize(records))


def finish(a):
    selected = json.loads((a.out / "selection.json").read_text())["ids"]
    comparison, details, runs = [], [], {}
    for model in ("llama", "qwen"):
        for condition in ("baseline", "sft", "contrastive"):
            path = a.out / model / condition / "predictions.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            if [r["id"] for r in rows] != selected:
                raise ValueError(f"Incomplete or mismatched results: {path}")
            runs[model, condition] = rows
            totals = summarize(rows)
            row = {
                "model": model,
                "condition": condition,
                "n": totals["n"],
                "json_valid_rate": totals["json_valid_rate"],
                "uml_subset_valid_rate": totals["uml_subset_valid_rate"],
                "exact_graph_rate": totals["exact_graph_rate"],
            }
            for category in (
                "classes",
                "attributes",
                "relations",
                "full_relations",
                "graph",
                "known_multiplicities",
            ):
                row[category + "_micro_f1"] = totals[category]["micro"]["f1"]
                row[category + "_macro_f1"] = totals[category]["macro_f1"]
            comparison.append(row)
            for r in rows:
                details.append(
                    {
                        "model": model,
                        "condition": condition,
                        "id": r["id"],
                        "valid": r["scores"]["uml_subset_valid"],
                        "exact_match": r["scores"]["exact_graph"],
                        "full_relation_f1": r["scores"]["full_relations"]["f1"],
                        "diagram_status": r["diagram_status"],
                    }
                )
    for filename, rows in [("comparison.csv", comparison), ("per_requirement.csv", details)]:
        with (a.out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Saved comparison.csv and per_requirement.csv. Scores are fractions, not percentages.")
    from plot_test10 import create_charts

    create_charts(a.out, comparison, runs, selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run", "summarize"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["llama", "qwen"])
    p.add_argument("--condition", choices=["baseline", "sft", "contrastive"])
    a = p.parse_args()
    if a.action == "run" and (not a.model or not a.condition):
        p.error("run requires --model and --condition")
    {"prepare": prepare, "run": run, "summarize": finish}[a.action](a)
