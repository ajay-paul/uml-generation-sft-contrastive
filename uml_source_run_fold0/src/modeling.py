"""Shared model loading, completion masking and sequence likelihood objective."""

import json

import torch
import torch.nn.functional as F
from common import ROOT, SYSTEM, serialize
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen3-8B"}


def revision(alias):
    path = ROOT / "models.lock.json"
    if not path.exists():
        raise RuntimeError("Run python src/preflight.py --lock-models first.")
    entry = json.loads(path.read_text())[alias]
    if entry["model_id"] != MODELS[alias]:
        raise ValueError("Model ID does not match lock")
    return entry["revision"]


def tokenizer_for(alias):
    tok = AutoTokenizer.from_pretrained(MODELS[alias], revision=revision(alias))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def prompt_ids(tok, alias, requirement):
    kwargs = {"enable_thinking": False} if alias == "qwen" else {}
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": requirement}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )


def encode_candidates(tok, alias, row, targets, max_length=2048):
    """Encode the same inference prefix, then append only JSON + end-of-turn.

    This avoids applying a different assistant template during Qwen training.
    No truncation is allowed. Padding is masked by POSITION, not token ID.
    """
    prefix = prompt_ids(tok, alias, row["requirement"])
    end = tok.convert_tokens_to_ids("<|eot_id|>") if alias == "llama" else tok.eos_token_id
    if end is None:
        raise ValueError("Missing assistant end-of-turn token")
    seq = [prefix + tok.encode(serialize(g), add_special_tokens=False) + [end] for g in targets]
    if max(map(len, seq)) > max_length:
        raise ValueError(
            f"{row['id']}: {max(map(len, seq))} tokens exceeds {max_length}; increase consistently, do not truncate"
        )
    length = max(map(len, seq))
    plen = len(prefix)
    ids = torch.full((len(seq), length), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full((len(seq), length - plen), -100, dtype=torch.long)
    for i, s in enumerate(seq):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
        labels[i, : len(s) - plen] = torch.tensor(s[plen:])
    return ids, mask, labels, plen


def sequence_scores(model, encoded):
    ids, mask, labels, plen = encoded
    device = model.get_input_embeddings().weight.device
    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    keep = ids.shape[1] - plen + 1
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep
        ).logits[:, :-1, :]
    scores = []
    # Compute CE in chunks to avoid a single full float32 vocabulary tensor.
    for i in range(len(ids)):
        total = logits[i, 0, 0] * 0
        for start in range(0, labels.shape[1], 64):
            target = labels[i, start : start + 64]
            total = total + F.cross_entropy(
                logits[i, start : start + 64].float(), target, ignore_index=-100, reduction="sum"
            )
        scores.append(-total / (labels[i] != -100).sum())
    return torch.stack(scores)


def objective(scores, weight=0.1, temperature=0.1):
    if temperature <= 0 or weight < 0:
        raise ValueError("Invalid loss settings")
    nll = -scores[0]
    contrast = F.cross_entropy(
        (scores / temperature)[None, :], torch.zeros(1, dtype=torch.long, device=scores.device)
    )
    return nll + weight * contrast, nll, contrast


def load(alias, adapter=None, training=False, quantization="nf4"):
    if not torch.cuda.is_available():
        raise RuntimeError("Training/inference requires the NVIDIA VM; no CUDA GPU detected")
    kwargs = {
        "revision": revision(alias),
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(MODELS[alias], **kwargs)
    if training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    elif adapter:
        meta = json.loads((__import__("pathlib").Path(adapter).parent / "run.json").read_text())
        if meta["model"] != alias or meta["revision"] != revision(alias):
            raise ValueError("Adapter model/revision mismatch")
        model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = not training
    return model
