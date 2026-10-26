"""Generate held-out graphs, measure task quality, latency and optional ranking."""

import argparse
import json
import time
from pathlib import Path

import torch
from common import ROOT, digest, select_rows, write_jsonl
from metrics import robustness, score, summarize
from modeling import (
    MODELS,
    encode_candidates,
    load,
    prompt_ids,
    revision,
    sequence_scores,
    tokenizer_for,
)
from transformers import GenerationConfig, set_seed


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--protocol", choices=["within", "domain"], default="within")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--views", choices=["original", "all"], default="all")
    p.add_argument("--rank", action="store_true")
    p.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh results directory")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    if a.adapter:
        m = json.loads((Path(a.adapter).parent / "run.json").read_text())
        assert m["status"] == "complete" and m["fold"] == a.fold and m["protocol"] == a.protocol, (
            "Adapter split mismatch"
        )
        assert m["dataset_sha256"] == digest(ROOT / "data/augmented.jsonl"), (
            "Dataset changed since training"
        )
    rows = select_rows(a.split, a.fold, a.protocol, a.views)
    tok = tokenizer_for(a.model)
    model = load(a.model, a.adapter, quantization=a.quantization)
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

    def generate(req):
        ids = torch.tensor([prompt_ids(tok, a.model, req)], device="cuda")
        if ids.shape[1] + cfg.max_new_tokens > 2048:
            raise ValueError("Prompt plus output exceeds 2048 experiment budget")
        return ids, model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=cfg,
            use_model_defaults=False,
            do_sample=False,
            num_beams=1,
        )

    results = []
    with torch.inference_mode():
        warm = torch.tensor(
            [prompt_ids(tok, a.model, "Track an Asset with assetId (String).")], device="cuda"
        )
        model.generate(
            input_ids=warm,
            attention_mask=torch.ones_like(warm),
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for r in rows:
            torch.cuda.synchronize()
            start = time.perf_counter()
            ids, output = generate(r["requirement"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            completion = output[0, ids.shape[1] :]
            raw = tok.decode(completion, skip_special_tokens=True).strip()
            endids = eos if isinstance(eos, list) else [eos]
            record = {
                "id": r["id"],
                "parent_id": r["parent_id"],
                "leakage_group": r["leakage_group"],
                "variant": r["variant"],
                "domain": r["domain"],
                "output": raw,
                "latency_seconds": elapsed,
                "generated_tokens": len(completion),
                "hit_token_cap": int(len(completion) >= 1024 and int(completion[-1]) not in endids),
                "scores": score(raw, r["target"]),
            }
            results.append(record)
            print(r["id"], record["scores"]["full_relations"]["f1"], flush=True)
        generation_peak = torch.cuda.max_memory_allocated() / 2**30
        if a.rank:
            for r, record in zip(rows, results):
                scores = sequence_scores(
                    model,
                    encode_candidates(
                        tok, a.model, r, [r["target"]] + [n["target"] for n in r["negatives"]]
                    ),
                ).tolist()
                # Conservative tie handling: all tied negatives precede the gold.
                rank = 1 + sum(x >= scores[0] for x in scores[1:])
                record["ranking"] = {
                    "candidate_count": len(scores),
                    "rank": rank,
                    "reciprocal_rank": 1 / rank,
                    "gold_minus_best_negative": scores[0] - max(scores[1:]),
                    "scores": scores,
                    "negative_types": [n["mutation"] for n in r["negatives"]],
                }
    write_jsonl(out / "predictions.jsonl", results)
    summary = {
        "all_views": summarize(results),
        "by_variant": {
            v: summarize([r for r in results if r["variant"] == v])
            for v in sorted({r["variant"] for r in results})
        },
        "robustness": robustness(results),
        "latency_p50_seconds": percentile([r["latency_seconds"] for r in results], 0.5),
        "latency_p95_seconds": percentile([r["latency_seconds"] for r in results], 0.95),
        "generation_peak_allocated_gib": generation_peak,
        "token_cap_rate": sum(r["hit_token_cap"] for r in results) / len(results),
    }
    if a.rank:
        summary["ranking"] = {
            "candidate_count": 5,
            "recall_at_1": sum(r["ranking"]["rank"] == 1 for r in results) / len(results),
            "mrr": sum(r["ranking"]["reciprocal_rank"] for r in results) / len(results),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "evaluation.json").write_text(
        json.dumps(
            {
                **vars(a),
                "model_id": MODELS[a.model],
                "revision": revision(a.model),
                "dataset_sha256": digest(ROOT / "data/augmented.jsonl"),
                "system_prompt_sha256": __import__("hashlib")
                .sha256(__import__("common").SYSTEM.encode())
                .hexdigest(),
                "decoding": "explicit-greedy-v2, one beam, max_new_tokens=1024; Qwen thinking disabled",
                "generation_peak_excludes_ranking": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
