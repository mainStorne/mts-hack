#!/usr/bin/env python3
"""Гибрид GLiNER (кандидаты) + LLM (каноничное value или НЕТ): make / infer / tune / measure."""
import argparse
import json
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import struct_predict as P  # noqa: E402

T, S = P.T, P.S
import case_metrics as M  # noqa: E402

M.load_catalog(os.path.join(T.ROOT, "data/raw/catalog.json"))
import rules  # noqa: E402

vl = T.value_labels(T.load_catalog())

MIN_CONF = 0.3
RAW = {"train": "runs/llm-raw/raw_train.jsonl",
       "dev": "runs/struct-lora-v21/raw_dev.jsonl",
       "test": "runs/struct-lora-v21/raw_test.jsonl"}
CALLS = {k: P.SPLITS[k] for k in ("train", "dev", "test")}
SYSTEM = ("Ты нормализуешь факты из звонка в техподдержку. Дана реплика (помечена >>>), атрибут и фрагмент из неё. "
          "Верни одной строкой каноничное значение атрибута. Предпочитай значение из списка допустимых; "
          "если подходящего нет — короткую формулировку по тексту. Если фрагмент не является фактом этого атрибута "
          "(предложение оператора без согласия клиента, догадка, отвергнутый вариант, вопрос), верни НЕТ.")
NO = "НЕТ"
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def candidates(call, rec):
    out, seen = [], set()
    for t, raw in enumerate(rec["raws"]):
        text = call["turns"][t]["text"]
        o = rec["offs"][t]
        for item, _, _ in S._value_items(raw):
            v = S._f(item) if isinstance(item, dict) else None
            if not v or v[1] < MIN_CONF or v[2] is None or v[3] is None:
                continue
            st, en = v[2], v[3]
            attr = (item.get("attribute") or {}).get("label")
            if attr not in S.ATTRS or not (o <= st and en <= o + len(text)):
                continue
            key = (t, st, en, attr)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(call_id=call["call_id"], t=t, st=st, en=en, attr=attr, conf=v[1],
                            span=text[st - o:en - o]))
    return out


def prompt(call, c):
    ctx = "\n".join(T.render_context(call, c["t"]).split("\n")[1:])
    user = (f"Атрибут: {c['attr']}\nДопустимые значения: {'; '.join(vl[c['attr']])}\n"
            f"Фрагмент: «{c['span']}»\nКонтекст:\n{ctx}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def label(call, c, off):
    s, e = c["st"] - off, c["en"] - off
    G = [g for g in call["facts"] if g["source"]["turn_idx"] == c["t"] and g["attribute"] == c["attr"]]

    def ov(g):
        if not g["source"].get("span"):
            return 0
        gs, ge = g["source"]["span"]
        return min(e, ge) - max(s, gs)
    hit = [g for g in G if ov(g) > 0]
    if hit:
        return max(hit, key=ov)["value"]
    if len(G) == 1:
        return G[0]["value"]
    if G:
        return max(G, key=lambda g: M._value_sim(c["span"], g["value"]))["value"]
    return NO


def load_split(k):
    calls = T.load_calls(CALLS[k])
    recs = {r["call_id"]: r for r in T.load_calls(RAW[k])}
    return calls, recs


def cmd_make(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    os.makedirs("data/llm_norm", exist_ok=True)
    mx = 0
    for k in ("train", "dev"):
        calls, recs = load_split(k)
        rows = []
        for c in calls:
            rec = recs[c["call_id"]]
            for cd in candidates(c, rec):
                lab = label(c, cd, rec["offs"][cd["t"]])
                rows.append({"prompt": prompt(c, cd), "completion": [{"role": "assistant", "content": lab}],
                             "meta": json.dumps(cd, ensure_ascii=False)})
        before = sum(r["completion"][0]["content"] == NO for r in rows) / max(1, len(rows))
        if k == "train" and before > 0.5:
            rng = random.Random(0)
            pos = [r for r in rows if r["completion"][0]["content"] != NO]
            neg = [r for r in rows if r["completion"][0]["content"] == NO]
            rng.shuffle(neg)
            rows = pos + neg[:len(pos)]
            rng.shuffle(rows)
        after = sum(r["completion"][0]["content"] == NO for r in rows) / max(1, len(rows))
        for r in rows:
            txt = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)
            mx = max(mx, len(tok(txt)["input_ids"]))
        with open(f"data/llm_norm/{k}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{k}: rows={len(rows)} no_before={before:.3f} no_after={after:.3f}", flush=True)
    print("max_prompt_tokens", mx)


def post(txt):
    txt = THINK.sub("", txt)
    lines = [l for l in txt.split("\n") if l.strip()]
    s = lines[0].strip().strip("«»\"'. ") if lines else ""
    if not s or s.upper().startswith(NO):
        return NO
    return s


def generate(model_name, adapter, prompts, batch=32):
    """-> (outs, ms_per_cand, llm_params, peak_gb, model, tok)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    txts = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            for p in prompts]
    order = sorted(range(len(txts)), key=lambda i: len(txts[i]))
    outs = [None] * len(txts)
    torch.cuda.reset_peak_memory_stats()
    t_gen = 0.0
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        enc = tok([txts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t_gen += time.perf_counter() - t0
        for j, o in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[j] = post(o)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return outs, t_gen * 1000 / max(1, len(txts)), params, peak


def ok_value(lab, out, attr):
    if out == NO:
        return False
    pv = S.snap(attr, out, vl, 0.5)
    vsim = M._value_sim(lab, pv)
    cg, cp = M._canon(lab, attr), M._canon(pv, attr)
    if cg == cp and (cg != lab or cp != pv):
        vsim = max(vsim, 1.0)
    need = M.VALUE_SIM if min(len(M._tokens(lab)), len(M._tokens(pv))) <= 2 else M.VALUE_SIM_LONG
    return vsim >= need


def cmd_infer(a):
    calls, recs = load_split(a.split)
    cs, ps, labs = [], [], []
    for c in calls:
        rec = recs[c["call_id"]]
        for cd in candidates(c, rec):
            cs.append(cd)
            ps.append(prompt(c, cd))
            if a.split == "dev":
                labs.append(label(c, cd, rec["offs"][cd["t"]]))
    outs, ms, params, peak = generate(a.model, a.adapter, ps, a.batch)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/llm_{a.split}.jsonl", "w", encoding="utf-8") as fh:
        for cd, o in zip(cs, outs):
            fh.write(json.dumps({k: cd[k] for k in ("call_id", "t", "st", "en", "attr")} | {"out": o},
                                ensure_ascii=False) + "\n")
    info = {"model": a.model, "adapter": a.adapter, "n_cand": len(cs), "ms_per_cand": ms,
            "llm_params": params, "peak_vram_gb": peak,
            "empty_or_no_frac": sum(o == NO for o in outs) / max(1, len(outs))}
    if labs:
        info["verdict_acc"] = sum((l == NO) == (o == NO) for l, o in zip(labs, outs)) / len(labs)
        pos = [(l, o, cd["attr"]) for l, o, cd in zip(labs, outs, cs) if l != NO]
        info["value_ok"] = sum(ok_value(l, o, at) for l, o, at in pos) / max(1, len(pos))
        info["label_no_frac"] = sum(l == NO for l in labs) / len(labs)
    json.dump(info, open(f"{a.out}/infer_{a.split}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def load_llm(d, k):
    out = {}
    for l in open(f"{d}/llm_{k}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        out.setdefault(r["call_id"], {})[(r["t"], r["st"], r["en"], r["attr"])] = r["out"]
    return out


_CTX = {}


def _ev(args):
    tau, snap_th = args
    c = _CTX
    return args, P.evaluate(c["calls"], P.predict(c["calls"], c["raws"], vl, c["dm"], tau, snap_th, None, llm=c["llm"]))


def cmd_tune(a):
    dm = rules.load_date_model()
    v21 = json.load(open("runs/struct-lora-v21/params.json"))
    calls = {k: T.load_calls(CALLS[k]) for k in ("dev", "test")}
    raws = {k: {r["call_id"]: r for r in T.load_calls(RAW[k])} for k in ("dev", "test")}
    llm = {k: load_llm(a.dir, k) for k in ("dev", "test")}
    _CTX.update(calls=calls["dev"], raws=raws["dev"], dm=dm, llm=llm["dev"])
    tau = {k: max(MIN_CONF, v) for k, v in v21["tau_attr"].items()}
    snap_th = 0.5
    ths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with multiprocessing.get_context("fork").Pool(max(1, (os.cpu_count() or 3) - 2)) as pool:
        for _ in range(2):
            for at in S.ATTRS:
                res = pool.map(_ev, [({**tau, at: th}, snap_th) for th in ths])
                tau[at] = max(zip(ths, res), key=lambda x: x[1][1]["f1"])[0]
    md = P.evaluate(calls["dev"], P.predict(calls["dev"], raws["dev"], vl, dm, tau, snap_th, None, llm=llm["dev"]))
    preds = P.predict(calls["test"], raws["test"], vl, dm, tau, snap_th, None, llm=llm["test"])
    mt = P.evaluate(calls["test"], preds)
    json.dump({"tau_attr": tau, "snap_th": snap_th, "dev_f1": md["f1"], "test_f1": mt["f1"]},
              open(f"{a.dir}/params.json", "w"), indent=1)
    with open(f"{a.dir}/pred_test.jsonl", "w", encoding="utf-8") as fh:
        for p in preds.values():
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    b = {k: P.evaluate(calls[k], P.predict(calls[k], raws[k], vl, dm, v21["tau_attr"], v21["snap_th"], None))
         for k in ("dev", "test")}
    lines = ["| split | " + " | ".join(P.COLS) + " |", "|" + "---|" * (len(P.COLS) + 1),
             P.row("hybrid dev", md), P.row("hybrid test", mt), P.row("v21 dev", b["dev"]), P.row("v21 test", b["test"]),
             "", f"tau_attr: {json.dumps(tau)}"]
    open(f"{a.dir}/summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_measure(a):
    import torch
    calls = T.load_calls(P.SPLITS["test"])[:a.n]
    torch.cuda.reset_peak_memory_stats()
    ext = P.load_model("runs/struct-lora")
    gl_params = sum(p.numel() for p in ext.parameters())
    tmp = "/tmp/measure_raw.jsonl"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    P.raw_split(ext, calls, tmp)
    torch.cuda.synchronize()
    t_gl = time.perf_counter() - t0
    gl_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    recs = {r["call_id"]: r for r in T.load_calls(tmp)}
    ps = [prompt(c, cd) for c in calls for cd in candidates(c, recs[c["call_id"]])]
    outs, ms, params, peak = generate(a.model, a.adapter, ps, 8)
    res = {"gliner_params": gl_params, "llm_params": params, "total_params": gl_params + params,
           "within_1.5e9": gl_params + params <= 1.5e9, "gliner_peak_gb": gl_peak,
           "combined_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "n_cand": len(ps),
           "ms_per_call": (t_gl * 1000 + ms * len(ps)) / len(calls)}
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    res["nvsmi_process_mb"] = next((int(l.split(",")[1]) for l in q.splitlines()
                                    if l.split(",")[0].strip() == str(os.getpid())), None)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    i = sub.add_parser("infer")
    i.add_argument("--model", required=True)
    i.add_argument("--adapter")
    i.add_argument("--split", choices=["dev", "test"], required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--batch", type=int, default=32)
    t = sub.add_parser("tune")
    t.add_argument("--dir", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter")
    m.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    {"make": cmd_make, "infer": cmd_infer, "tune": cmd_tune, "measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()
