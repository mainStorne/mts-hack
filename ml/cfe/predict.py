import argparse
import json
import os
import time

import case_metrics
from codec import close_states, decode_target, messages, snap_to_catalog

case_metrics.load_catalog(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "catalog.json"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--catalog", default="data/raw/catalog.json")
    ap.add_argument("--no-close", action="store_true")
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    dtype = torch.float32 if a.fp32 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to("cuda")
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(a.adapter or a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    calls = read_jsonl(a.calls)
    catalog = json.load(open(a.catalog, encoding="utf-8"))
    prompts = [
        tok.apply_chat_template(messages(c), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        for c in calls
    ]
    lens = [len(tok(p).input_ids) for p in prompts]
    order = sorted(range(len(calls)), key=lambda i: lens[i])
    raws = [""] * len(calls)
    gpu_total = 0.0
    torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(order), a.batch):
        idx = order[s:s + a.batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        gpu_total += time.perf_counter() - t0
        for i, g in zip(idx, gen):
            raws[i] = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"{min(s + a.batch, len(order))}/{len(order)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    invalid = dropped_total = 0
    preds = []
    for c, raw in zip(calls, raws):
        facts, dropped = decode_target(raw, c)
        if dropped < 0:
            invalid += 1
        else:
            dropped_total += dropped
        if not a.no_close:
            facts = close_states(facts)
        if not a.no_snap:
            facts = snap_to_catalog(facts, catalog)
        preds.append({**{k: v for k, v in c.items() if k != "facts"}, "facts": facts})
    with open(f"{a.out}/pred.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(f"{a.out}/raw.jsonl", "w", encoding="utf-8") as f:
        for c, r in zip(calls, raws):
            f.write(json.dumps({"call_id": c["call_id"], "raw": r}, ensure_ascii=False) + "\n")
    res = {
        "params": sum(p.numel() for p in model.parameters()),
        "gpu_seconds_total": gpu_total,
        "gpu_sec_per_call": gpu_total / len(calls),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "batch": a.batch,
        "n_calls": len(calls),
        "invalid_calls": invalid,
        "dropped_facts": dropped_total,
        "gpu": torch.cuda.get_device_name(),
    }
    json.dump(res, open(f"{a.out}/resources.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    if all("facts" in c for c in calls):
        m = case_metrics.evaluate({c["call_id"]: c for c in calls}, {p["call_id"]: p for p in preds})
        json.dump(m, open(f"{a.out}/metrics.json", "w"), indent=1)
        print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
