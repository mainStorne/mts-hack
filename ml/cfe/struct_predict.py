#!/usr/bin/env python3
"""Инференс GLiNER2.5 (zero-shot или с LoRA-адаптером), подбор порогов на dev, отчёт по case_metrics."""
import argparse
import itertools
import json
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case_metrics  # noqa: E402
case_metrics.load_catalog(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "catalog.json"))
import clf_task as T  # noqa: E402
import rules  # noqa: E402
import denoise as DN  # noqa: E402
import struct_schema as S  # noqa: E402

MODEL = os.environ.get("CFE_GLINER_BASE", "fastino/gliner2.5-multi-v1")  # в образах — путь снимка (офлайн)
D = os.path.join(T.ROOT, "data")
SPLITS = {
    "dev": os.path.join(D, "struct", "dev_calls.jsonl"),
    "train": os.path.join(D, "struct", "train_calls.jsonl"),
    "test": os.path.join(D, "raw", "val.jsonl"),
    "test_noisy": os.path.join(D, "val_noisy.jsonl"),
}
BASE = T.baseline()
GATE_KEYS = ["fact_precision", "fact_recall", "temporal_recall", "state_accuracy", "grounding_turn"]
COLS = ["fact_precision", "fact_recall", "f1", "temporal_recall", "state_accuracy", "grounding_turn",
        "grounding_span", "n_pred_facts"]


def load_model(adapter, device="cuda"):
    import gliner_patch
    gliner_patch.apply()
    from gliner2 import AutoExtractor
    ext = AutoExtractor.from_pretrained(MODEL)
    ext = ext.to(device)
    if adapter:
        p = os.path.join(adapter, "best")
        if not os.path.isdir(p):
            p = os.path.join(adapter, "final")
        from gliner2.training.lora import load_lora_adapter
        load_lora_adapter(ext, p)
    return ext


def raw_split(ext, calls, path, denoise=False, kind="entities"):
    schema = S.build_schema(ext, kind)
    with open(path, "w", encoding="utf-8") as fh:
        for call in calls:
            cc = DN.denoise_call(call)[0] if denoise else call
            ctxs = [T.render_context(cc, t) for t in range(len(cc["turns"]))]
            offs = [S.turn_offset(c, cc, t) for t, c in enumerate(ctxs)]
            raws = ext.batch_extract(ctxs, schema, batch_size=16, threshold=0.1,
                                     include_confidence=True, include_spans=True)
            fh.write(json.dumps({"call_id": call["call_id"], "offs": offs, "raws": raws}, ensure_ascii=False) + "\n")


def f1(m):
    p, r = m["fact_precision"], m["fact_recall"]
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def predict(calls, raws, vl, dm, tau, snap_th, date_tau, denoise=False, llm=None, keep_conf=False):
    if denoise and llm is not None:
        raise ValueError("llm with denoise unsupported")
    out = {}
    for c in calls:
        r = raws[c["call_id"]]
        p = {k: v for k, v in c.items() if k != "facts"}
        lc = llm.get(c["call_id"], {}) if llm is not None else None
        if denoise:
            cc, maps = DN.denoise_call(c)
            facts = S.call_facts(cc, r["raws"], r["offs"], vl, dm, tau, snap_th, date_tau)
            p["facts"] = DN.restore_sources(facts, c, maps)
        else:
            p["facts"] = S.call_facts(c, r["raws"], r["offs"], vl, dm, tau, snap_th, date_tau, lc, keep_conf=keep_conf)
        out[c["call_id"]] = p
    return out


def evaluate(calls, preds):
    m = case_metrics.evaluate({c["call_id"]: c for c in calls}, preds)
    m["f1"] = f1(m)
    return m


def row(label, m):
    cells = [("" if m.get(k) is None else str(m[k])) for k in COLS]
    return f"| {label} | " + " | ".join(cells) + " |"


def passes(m):
    return all((m.get(k) or 0) >= BASE[k] - 0.02 for k in GATE_KEYS)


def score(m):
    """F1 минус суммарный недобор до планки gate по метрикам из GATE_KEYS."""
    return m["f1"] - sum(max(0.0, BASE[k] - 0.02 - (m.get(k) or 0)) for k in GATE_KEYS)



_CTX = {}


def _dev_eval(args):
    tau, snap_th, dtau = args
    c = _CTX
    return args, evaluate(c["calls"], predict(c["calls"], c["raws"], c["vl"], c["dm"], tau, snap_th, dtau, c["denoise"]))


def _tune(a, calls, raws, vl, dm):
    _CTX.update(calls=calls["dev"], raws=raws["dev"], vl=vl, dm=dm, denoise=a.denoise)
    lines = ["## dev: перебор", "",
             "| tau/snap/date_tau | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    best = None
    grid = list(itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.85], [None, 0.3, 0.5, 0.7]))
    with multiprocessing.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for (tau, snap_th, dtau), m in pool.map(_dev_eval, grid):
            lines.append(row(f"{tau}/{snap_th}/{dtau}", m))
            if best is None or score(m) > best[0]:
                best = (score(m), (tau, snap_th, dtau))
        chosen = best
        tau, snap_th, dtau = chosen[1]
        tau_attr = {a_: tau for a_ in S.ATTRS}
        lines += ["", "## dev: tau по атрибутам", "", "| attr/tau | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
        for at in S.ATTRS:
            top = None
            ths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for th, (_, m) in zip(ths, pool.map(_dev_eval, [({**tau_attr, at: th}, snap_th, dtau) for th in ths])):
                lines.append(row(f"{at}/{th}", m))
                if top is None or score(m) > top[0]:
                    top = (score(m), th)
            tau_attr[at] = top[1]
    return tau, tau_attr, snap_th, dtau, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-raw", action="store_true", help="не гонять модель, взять raw_*.jsonl")
    ap.add_argument("--denoise", action="store_true", help="нормализатор ASR-шума (для модели, обученной на struct_dn)")
    ap.add_argument("--schema", choices=["entities", "fact"], default="entities")
    ap.add_argument("--splits", default="dev,test,test_noisy")
    ap.add_argument("--infer-only", action="store_true", help="только raw_*.jsonl, без подбора порогов")
    ap.add_argument("--params", help="params.json с tau_attr/snap_th/date_tau: оценка без подбора порогов")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keys = [k.strip() for k in a.splits.split(",") if k.strip()]
    if "dev" not in keys:
        raise SystemExit("--splits must include dev")
    if a.infer_only and a.reuse_raw:
        raise SystemExit("--infer-only and --reuse-raw are exclusive")
    splits = {k: SPLITS[k] for k in keys}
    calls = {k: T.load_calls(p) for k, p in splits.items()}
    if not a.reuse_raw:
        ext = load_model(a.adapter)
        for k in splits:
            raw_split(ext, calls[k], os.path.join(a.out, f"raw_{k}.jsonl"), a.denoise, a.schema)
        if a.infer_only:
            return
    raws = {k: {r["call_id"]: r for r in T.load_calls(os.path.join(a.out, f"raw_{k}.jsonl"))} for k in splits}
    vl = T.value_labels(T.load_catalog())
    dm = rules.load_date_model()
    if a.params:
        p = json.load(open(a.params))
        tau, tau_attr, snap_th, dtau = p.get("tau"), p["tau_attr"], p["snap_th"], p["date_tau"]
        lines = ["## dev: фиксированные пороги", "", f"params from: {a.params}"]
    else:
        tau, tau_attr, snap_th, dtau, lines = _tune(a, calls, raws, vl, dm)
    m = evaluate(calls["dev"], predict(calls["dev"], raws["dev"], vl, dm, tau_attr, snap_th, dtau, a.denoise))
    params = {"tau": tau, "tau_attr": tau_attr, "snap_th": snap_th, "date_tau": dtau,
              "dev_f1": m["f1"], "dev_gate_ok": passes(m), "fixed_from": a.params}
    json.dump(params, open(os.path.join(a.out, "params.json"), "w"), indent=1)

    lines += ["", f"params: {json.dumps(params)}", f"tau_attr: {json.dumps(tau_attr)}", "", "## итоги",
              "| split | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    res = {}
    lines.append(row("dev", m))
    for k in ("test", "test_noisy"):
        if k not in calls:
            continue
        preds = predict(calls[k], raws[k], vl, dm, tau_attr, snap_th, dtau, a.denoise)
        res[k] = evaluate(calls[k], preds)
        lines.append(row(k, res[k]))
        with open(os.path.join(a.out, f"pred_{k}.jsonl"), "w", encoding="utf-8") as fh:
            for p in preds.values():
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    if "test" in calls:
        m0 = evaluate(calls["test"], predict(calls["test"], raws["test"], vl, dm, tau_attr, snap_th, None, a.denoise))
        lines.append(row("test (date_tau=None)", m0))
    bm = BASE
    lines.append(row("baseline", bm))
    gate = ({k: (res["test"].get(k) or 0) >= BASE[k] - 0.02 for k in GATE_KEYS} if "test" in res else None)
    lines += ["", f"gate: {json.dumps(gate)}"]
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
