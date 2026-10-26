"""Run on the VM before experiments; save exact model revisions and environment."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import torch
from common import ROOT
from modeling import MODELS, encode_candidates, prompt_ids, tokenizer_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock-models", action="store_true")
    p.add_argument("--model", choices=MODELS)
    a = p.parse_args()
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30,
        "packages": {
            x: importlib.metadata.version(x)
            for x in [
                "transformers",
                "peft",
                "accelerate",
                "bitsandbytes",
                "huggingface-hub",
                "tokenizers",
            ]
        },
    }
    if torch.cuda.is_available():
        report.update(
            gpu=torch.cuda.get_device_name(0),
            bf16=torch.cuda.is_bf16_supported(),
            vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        )
        assert report["bf16"], "BF16 required for this configuration"
    (ROOT / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    (ROOT / "environment.lock.txt").write_text(
        subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)
    )
    if a.lock_models:
        from huggingface_hub import HfApi

        path = ROOT / "models.lock.json"
        if path.exists():
            print("Keeping existing model revision lock.")
        else:
            lock = {
                k: {"model_id": v, "revision": HfApi().model_info(v).sha} for k, v in MODELS.items()
            }
            path.write_text(json.dumps(lock, indent=2) + "\n")
    if a.model:
        tok = tokenizer_for(a.model)
        lengths = []
        completions = []
        budgets = []
        from common import read_jsonl

        for r in read_jsonl(ROOT / "data/augmented.jsonl"):
            for g in [r["target"]] + [n["target"] for n in r["negatives"]]:
                enc = encode_candidates(tok, a.model, r, [g])
                lengths.append(enc[0].shape[1])
                completions.append(enc[2].shape[1])
            budgets.append(len(prompt_ids(tok, a.model, r["requirement"])) + 1024)
        assert max(completions) <= 1024, "Generation cap is smaller than a gold completion"
        assert max(budgets) <= 2048, "Prompt plus generation budget exceeds configured context"
        print(
            json.dumps(
                {
                    "model": a.model,
                    "candidate_count": len(lengths),
                    "max_training_tokens": max(lengths),
                    "max_completion_tokens": max(completions),
                    "max_prompt_plus_generation": max(budgets),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
