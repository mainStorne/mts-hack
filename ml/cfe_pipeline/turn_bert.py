#!/usr/bin/env python3
"""Fine-tune mdeberta-v3-base: бинарный классификатор «в реплике есть action» (с контекстом prev/next).

  python turn_bert.py --train data/turn_clf/train_clean.jsonl --val data/turn_clf/val_clean.jsonl --out runs/turn-bert
Последние 10% звонков train — holdout для порогов (не обучаются). Сохраняет pred_hold.npy / pred_val.npy.
"""
import argparse, json, os, random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--out")
ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--maxlen", type=int, default=192)
ap.add_argument("--pool", default="cls"); ap.add_argument("--prefix", default="")
ap.add_argument("--neg_keep", type=float, default=0.4, help="доля негативов в эпохе (ускорение)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0)
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
tr, va = load(a.train), load(a.val)
cids = list(dict.fromkeys(r["cid"] for r in tr)); hold_c = set(cids[int(len(cids) * 0.9):])
hold = [r for r in tr if r["cid"] in hold_c]; tr = [r for r in tr if r["cid"] not in hold_c]
print(len(tr), len(hold), len(va), flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
dev = "cuda"


class M(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(a.model, dtype=torch.float32); s.head = torch.nn.Linear(s.enc.config.hidden_size, 1)
    def forward(s, **kw):
        h = s.enc(**kw).last_hidden_state
        if a.pool == "mean":
            mk = kw["attention_mask"].unsqueeze(-1).to(h.dtype); v = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
        else:
            v = h[:, 0]
        return s.head(v).squeeze(-1)


def enc(rows):
    first = [f"{a.prefix}{r['speaker']}: {r['text']}" for r in rows]
    second = [f"до: {r['prev'][:300]} || после: {r['next'][:200]}" for r in rows]
    return tok(first, second, truncation="longest_first", max_length=a.maxlen, padding=True, return_tensors="pt")


def predict(m, rows):
    m.eval(); out = []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), 64):
            b = [rows[j] for j in order[i:i + 64]]
            out += torch.sigmoid(m(**{k: v.to(dev) for k, v in enc(b).items()}).float()).cpu().tolist()
    p = np.zeros(len(rows)); p[order] = out; return p


def report(name, y, p):
    o = np.argsort(-p); ys = y[o]; tp = np.cumsum(ys); pr = tp / np.arange(1, len(ys) + 1); rc = tp / ys.sum()
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9); k = int(np.argmax(f1)); ap_ = float((pr * ys).sum() / ys.sum())
    print(f"[{name}] AP={ap_:.4f} bestF1={f1[k]:.4f} thr={p[o][k]:.3f} P={pr[k]:.3f} R={rc[k]:.3f}", flush=True)


m = M().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
pos = [r for r in tr if r["y"]]; neg = [r for r in tr if not r["y"]]
steps = a.epochs * ((len(pos) + int(len(neg) * a.neg_keep)) // a.bs + 1)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler()
yh = np.array([r["y"] for r in hold]); yv = np.array([r["y"] for r in va])
for ep in range(a.epochs):
    m.train(); ep_rows = pos + random.sample(neg, int(len(neg) * a.neg_keep)); random.shuffle(ep_rows)
    for i in range(0, len(ep_rows), a.bs):
        b = ep_rows[i:i + a.bs]
        y = torch.tensor([r["y"] for r in b], dtype=torch.float32, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = m(**{k: v.to(dev) for k, v in enc(b).items()})
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        if (i // a.bs) % 200 == 0: print(f"ep{ep} {i}/{len(ep_rows)} loss {loss.item():.4f}", flush=True)
    ph, pv = predict(m, hold), predict(m, va)
    report(f"ep{ep} holdout", yh, ph); report(f"ep{ep} val", yv, pv)
    np.save(f"{a.out}/pred_hold_ep{ep}.npy", ph); np.save(f"{a.out}/pred_val_ep{ep}.npy", pv)
print("done", flush=True)
