"""Evaluate a (LoRA-tuned) model on JSON fact extraction."""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def clean(text: str) -> str:
    text = THINK_RE.sub("", text)
    m = FENCE_RE.match(text)
    return (m.group(1) if m else text).strip()


def parse(text: str) -> dict | None:
    try:
        obj = json.loads(clean(text))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def norm(v) -> str:
    return "null" if v is None else str(v).strip().lower()


def pairs(obj: dict | None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if obj is None:
        return out

    def walk(value, key: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(value, list):
            for v in value:
                walk(v, f"{key}[]")
        else:
            out.add((key, norm(value)))

    walk(obj, "")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    tokenizer = AutoTokenizer.from_pretrained(a.adapter or a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    rows = list(load_dataset("json", data_files=a.data)["train"])
    prompts = [
        tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        for r in rows
    ]

    raws: list[str] = []
    gen_time = 0.0
    for i in range(0, len(prompts), a.batch):
        enc = tokenizer(prompts[i : i + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        gen_time += time.perf_counter() - t0
        raws += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    valid = exact = tp = n_pred = n_gold = 0
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/predictions.jsonl", "w") as f:
        for r, raw in zip(rows, raws):
            gold = json.loads(r["completion"][0]["content"])
            pred = parse(raw)
            valid += pred is not None
            exact += pred == gold
            gp, pp = pairs(gold), pairs(pred)
            tp += len(gp & pp)
            n_pred += len(pp)
            n_gold += len(gp)
            f.write(json.dumps({"gold": gold, "raw": raw, "pred": pred}, ensure_ascii=False) + "\n")

    n = len(rows)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    metrics = {
        "rows": n,
        "model": a.model,
        "adapter": a.adapter,
        "json_valid": valid / n,
        "exact_match": exact / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "latency_ms_per_row": gen_time * 1000 / n,
    }
    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
