"""QLoRA SFT or SFT plus conditional sequence-likelihood contrastive loss."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from common import ROOT, digest, select_rows, serialize
from modeling import (
    MODELS,
    encode_candidates,
    load,
    objective,
    revision,
    sequence_scores,
    tokenizer_for,
)
from transformers import get_linear_schedule_with_warmup, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=["sft", "contrastive"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--protocol", choices=["within", "domain"], default="within")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--negatives", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--views", choices=["all", "original"], default="all")
    p.add_argument("--negative-mode", choices=["hard", "random"], default="hard")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-updates", type=int, default=0)
    a = p.parse_args()
    if a.epochs < 1 or a.accum < 1 or a.lr <= 0:
        raise ValueError("Invalid hyperparameters")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing runs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    rows = select_rows("train", a.fold, a.protocol, a.views)
    val = select_rows("val", a.fold, a.protocol, "original")
    tok = tokenizer_for(a.model)
    for r in rows + val:
        encode_candidates(
            tok, a.model, r, [r["target"]] + [n["target"] for n in r["negatives"]], a.max_length
        )
    model = load(a.model, training=True)
    model.print_trainable_parameters()
    model.train()
    from bitsandbytes.optim import PagedAdamW8bit

    optimizer = PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01
    )
    nsteps = math.ceil(len(rows) / a.accum) * a.epochs
    if a.max_updates:
        nsteps = min(nsteps, a.max_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(nsteps * 0.1)), nsteps)
    metadata = {
        **vars(a),
        "model_id": MODELS[a.model],
        "revision": revision(a.model),
        "status": "running",
        "dataset_sha256": digest(ROOT / "data/augmented.jsonl"),
        "system_prompt_sha256": __import__("hashlib")
        .sha256(__import__("common").SYSTEM.encode())
        .hexdigest(),
        "train_ids": [r["id"] for r in rows],
        "validation_ids": [r["id"] for r in val],
        "quantization": "nf4",
        "rank": 16,
        "alpha": 32,
        "dropout": 0,
        "planned_updates": nsteps,
        "selection": "lowest mean per-example gold completion NLL on canonical validation set; earliest tie",
    }
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    updates = 0
    best = float("inf")
    best_epoch = None
    train_tokens = 0
    for epoch in range(a.epochs):
        order = list(range(len(rows)))
        random.Random(a.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        logs = []
        for pos, ix in enumerate(order):
            r = rows[ix]
            candidates = [r["target"]]
            if a.method == "contrastive":
                if a.negative_mode == "hard":
                    ns = r["negatives"]
                    candidates += [
                        ns[(ix + epoch + j) % len(ns)]["target"] for j in range(a.negatives)
                    ]
                else:
                    # Only train-set targets, distinct parent and structure; never val/test.
                    pool = {
                        serialize(x["target"]): x["target"]
                        for x in rows
                        if x["parent_id"] != r["parent_id"]
                        and serialize(x["target"]) != serialize(r["target"])
                    }
                    candidates += random.Random(a.seed + epoch * len(rows) + ix).sample(
                        list(pool.values()), a.negatives
                    )
            encoded = encode_candidates(tok, a.model, r, candidates, a.max_length)
            train_tokens += int(encoded[1].sum())
            scores = sequence_scores(model, encoded)
            loss, nll, cl = objective(
                scores, a.weight if a.method == "contrastive" else 0, a.temperature
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; run aborted")
            group_start = (pos // a.accum) * a.accum
            group_size = min(a.accum, len(order) - group_start)
            (loss / group_size).backward()
            logs.append([loss.item(), nll.item(), cl.item()])
            if (pos + 1) % a.accum == 0 or pos + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                print(
                    json.dumps({"epoch": epoch + 1, "update": updates, "loss": logs[-1][0]}),
                    flush=True,
                )
                if a.max_updates and updates >= a.max_updates:
                    break
        model.eval()
        with torch.no_grad():
            losses = [
                -sequence_scores(
                    model, encode_candidates(tok, a.model, r, [r["target"]], a.max_length)
                )[0].item()
                for r in val
            ]
        v = sum(losses) / len(losses)
        if v < best:
            best = v
            best_epoch = epoch + 1
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        log = {
            "epoch": epoch + 1,
            "updates": updates,
            "train_loss": sum(x[0] for x in logs) / len(logs),
            "train_gold_nll": sum(x[1] for x in logs) / len(logs),
            "train_contrastive_loss": sum(x[2] for x in logs) / len(logs),
            "val_gold_nll": v,
            "elapsed_seconds": time.perf_counter() - start,
        }
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
        model.train()
        if a.max_updates and updates >= a.max_updates:
            break
    torch.cuda.synchronize()
    metadata.update(
        status="complete",
        best_epoch=best_epoch,
        best_val_nll=best,
        updates=updates,
        train_forward_tokens=train_tokens,
        wall_seconds=time.perf_counter() - start,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
