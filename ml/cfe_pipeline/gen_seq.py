#!/usr/bin/env python3
"""Per-turn генерация значения action: ruT5-base (full FT, fp32) или Gemma 3 1B (LoRA, fp16). Меряет время обучения и генерации.

  python gen_seq.py --arch t5|gemma --model NAME --train train.jsonl --eval a.jsonl b.jsonl --out runs/NAME
"""
import argparse, json, os, random, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=["t5", "gemma"]); ap.add_argument("--model"); ap.add_argument("--train")
ap.add_argument("--eval", nargs="+"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--beams", type=int, default=1)
ap.add_argument("--load", help="каталог сохранённой модели/адаптера: пропустить обучение")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0); dev = "cuda"
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
tr = load(a.train)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, get_linear_schedule_with_warmup
tok = AutoTokenizer.from_pretrained(a.model)
if a.arch == "t5":
    model = AutoModelForSeq2SeqLM.from_pretrained(a.load or a.model, dtype=torch.float32).to(dev); amp = False
else:
    from peft import LoraConfig, get_peft_model
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").to(dev)
    if a.load:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.load)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    amp = True
params = [p for p in model.parameters() if p.requires_grad]
print("trainable", sum(p.numel() for p in params), flush=True)


def prompt(x):
    if a.arch == "t5": return x
    return tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)


def batch_train(b):
    if a.arch == "t5":
        enc = tok([x["input"] for x in b], max_length=384, truncation=True, padding=True, return_tensors="pt")
        lab = tok(text_target=[x["target"] for x in b], max_length=64, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        return {**{k: v.to(dev) for k, v in enc.items()}, "labels": lab.to(dev)}
    ids, labs = [], []
    for x in b:
        p = tok(prompt(x["input"]), add_special_tokens=False, truncation=True, max_length=640).input_ids
        t = tok(x["target"] + "<end_of_turn>", add_special_tokens=False).input_ids[:64]
        ids.append(p + t); labs.append([-100] * len(p) + t)
    m = max(map(len, ids))
    pad = lambda s, v: s + [v] * (m - len(s))
    return {"input_ids": torch.tensor([pad(s, tok.pad_token_id) for s in ids], device=dev),
            "attention_mask": torch.tensor([pad([1] * len(s), 0) for s in ids], device=dev),
            "labels": torch.tensor([pad(s, -100) for s in labs], device=dev)}


opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler(enabled=amp)
t0 = time.time(); model.train()
for ep in range(0 if a.load else a.epochs):
    random.shuffle(tr)
    for i in range(0, len(tr), a.bs):
        b = batch_train(tr[i:i + a.bs])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(**b).loss
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sch.step()
        if (i // a.bs) % 100 == 0: print(f"ep{ep} {i}/{len(tr)} loss {loss.item():.4f} t={time.time() - t0:.0f}s", flush=True)
train_s = time.time() - t0
if not a.load:
    model.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
model.eval(); stats = {"train_s": round(train_s, 1), "train_n": len(tr), "epochs": a.epochs, "gen": {}}
for path in a.eval:
    ev = load(path); name = os.path.basename(path).split(".")[0]
    order = sorted(range(len(ev)), key=lambda i: len(ev[i]["input"])); out = [None] * len(ev); t1 = time.time()
    for i in range(0, len(order), 32):
        idx = order[i:i + 32]
        enc = tok([prompt(ev[j]["input"]) for j in idx], max_length=640, truncation=True, padding=True, return_tensors="pt", add_special_tokens=(a.arch == "t5")).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            g = model.generate(**enc, max_new_tokens=48, num_beams=a.beams, do_sample=False, pad_token_id=tok.pad_token_id)
        if a.arch == "gemma": g = g[:, enc.input_ids.shape[1]:]
        for j, s in zip(idx, tok.batch_decode(g, skip_special_tokens=True)): out[j] = s.strip()
    sec = time.time() - t1; stats["gen"][name] = {"n": len(ev), "sec": round(sec, 1), "ms_per_turn": round(1000 * sec / len(ev), 1)}
    with open(f"{a.out}/pred_{name}.jsonl", "w", encoding="utf-8") as f:
        for x, o in zip(ev, out): f.write(json.dumps({"id": x["id"], "pred": o, "target": x["target"]}, ensure_ascii=False) + "\n")
    print(name, stats["gen"][name], flush=True)
json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1); print("done", stats, flush=True)
