import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}


def main():
    parser = argparse.ArgumentParser(description="Terminal chat without a system prompt.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling. Default: deterministic greedy generation.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")

    model_id = MODELS[args.model]

    # For an adapter, load its recorded original checkpoint.
    if args.adapter:
        from peft import PeftConfig

        adapter_config = PeftConfig.from_pretrained(args.adapter)
        model_id = adapter_config.base_model_name_or_path

    print(f"Loading: {model_id}")
    print(f"Adapter: {args.adapter or 'None'}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    messages = []

    print("\nReady — no system message.")
    print("Enter one message per line.")
    print("/clear: reset history | /exit: quit\n")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_text.lower() == "/exit":
            break

        if user_text.lower() == "/clear":
            messages.clear()
            print("Conversation cleared.\n")
            continue

        if not user_text:
            continue

        pending_messages = messages + [{"role": "user", "content": user_text}]

        template_options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }

        if args.model == "qwen":
            template_options["enable_thinking"] = False

        input_ids = tokenizer.apply_chat_template(
            pending_messages,
            **template_options,
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit is not None and input_ids.shape[1] + args.max_new_tokens > context_limit:
            print("Conversation is too long. Use /clear and try again.\n")
            continue

        # Build a clean configuration rather than inheriting sampling defaults.
        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

        if args.sample:
            generation_config.temperature = 0.7
            generation_config.top_p = 0.9

        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                use_model_defaults=False,
            )

        answer_ids = output[0, input_ids.shape[1] :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        print(f"\nAssistant> {answer}\n")

        messages = pending_messages + [{"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
