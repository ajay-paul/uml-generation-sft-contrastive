import argparse
import json
import time
from pathlib import Path

import torch
from common import validate
from modeling import MODELS, load, prompt_ids, tokenizer_for
from transformers import GenerationConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    if args.adapter:
        adapter = Path(args.adapter)
        if not (adapter / "adapter_config.json").is_file():
            parser.error(f"Adapter not found: {adapter}")
        metadata = json.loads((adapter.parent / "run.json").read_text())
        if metadata.get("status") != "complete":
            parser.error("This adapter's training run is not complete.")

    print(f"Loading {args.model}...")
    print(f"Adapter: {args.adapter or 'None — base model'}")

    tokenizer = tokenizer_for(args.model)
    model = load(args.model, adapter=args.adapter)
    model.eval()

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id

    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=1024,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    print("\nReady.")
    print("Enter a requirement across one or more lines.")
    print("Submit with a blank line. Type /exit to quit.")
    print("Each submission starts a fresh request.\n")

    while True:
        lines = []
        try:
            while True:
                line = input("Requirement> " if not lines else "... ")
                if line.strip() == "/exit":
                    return
                if not line.strip():
                    if lines:
                        break
                    continue
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nClosed.")
            return

        requirement = "\n".join(lines)
        tokens = prompt_ids(tokenizer, args.model, requirement)

        if len(tokens) + config.max_new_tokens > 2048:
            print("Input exceeds the experiment's token budget.")
            print("Please shorten the requirement.\n")
            continue

        device = model.get_input_embeddings().weight.device
        inputs = torch.tensor([tokens], device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=config,
                use_model_defaults=False,
                do_sample=False,
                num_beams=1,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        completion = output[0, inputs.shape[1] :]
        raw = tokenizer.decode(completion, skip_special_tokens=True).strip()

        print("\nModel output:")
        try:
            graph = json.loads(raw)
        except json.JSONDecodeError:
            print(raw)
            print("\nValidation: output is not valid JSON.")
        else:
            print(json.dumps(graph, indent=2, ensure_ascii=False))
            try:
                validate(graph)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                print(f"\nStructural validation failed: {exc}")
            else:
                print("\nStructural validation passed.")
                print("Check semantic correctness against your requirement.")

        end_ids = eos if isinstance(eos, list) else [eos]
        if len(completion) >= config.max_new_tokens and int(completion[-1]) not in end_ids:
            print("Output reached the token limit and may be incomplete.")

        print(f"Generation time: {elapsed:.1f} seconds\n")


if __name__ == "__main__":
    main()
