#!/usr/bin/env python3
"""Кривая «ресурсы × качество»: python3 cfe/curve.py --root runs/bench --out runs/bench/curve.md

Читает runs/bench/<точка>/<срез>/{metrics,resources}.json (+ runs/bench/<точка>/error.txt), пишет curve.md и curve.csv.
"""
import argparse
import csv
import json
import os
import sys

SLICES = ["val", "p20", "p40", "p60"]  # p60 — дополнительный, только для базовых точек
GATE = ["fact_precision", "fact_recall", "temporal_recall", "state_accuracy", "grounding_turn"]
REF = "ens-hf-b16"
ROOT = ["runs/bench"]
ORDER = ["gliner", "gemma-b16", "ens-hf-b1", "ens-hf-b16", "ens-hf-b16-ctx2048", "ens-hf-b16-ctx1536", "ens-hf-b16-new512",
         "ens-vllm-b16", "ens-vllm-b64", "ens-cpu-t8", "ens-cpu-full"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, d=4):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def errors_section(pts, data):
    """Разбор ошибок по типам фактов (attribute) для опорной, выбранной и лучшей по F1 точек на val и p40 (gold — data/raw/val.jsonl)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import case_metrics as cm
    import report as R
    gold_path = "data/raw/val.jsonl"
    if not os.path.exists(gold_path):
        return []
    cm.load_catalog(gold_path)
    gold = R.read_jsonl(gold_path)
    show = []
    for p in pts:
        if p not in show and any(os.path.exists(f"{ROOT[0]}/{p}/{x}/pred.jsonl") for x in SLICES):
            show.append(p)
    out = ["## Разбор ошибок по типам фактов", "",
           "FN = пропущено (n_gold − matched), FP = лишние (n_pred − matched); temporal — доля верных дат среди сопоставленных с восстановимой датой; "
           "turn — доля верных реплик-источников среди сопоставленных.", ""]
    for p in show:
        for s in ("val", "p20", "p40", "p60"):
            pp = f"{ROOT[0]}/{p}/{s}/pred.jsonl"
            if not os.path.exists(pp):
                continue
            pred = R.read_jsonl(pp)
            ids = {x["call_id"] for x in pred}
            att, pol = R.breakdown([g for g in gold if g["call_id"] in ids], pred)  # прогоны с --limit: gold только для звонков выборки
            out += [f"### {p} / {s}", "", "| attribute | n_gold | n_pred | matched | FN | FP | precision | recall | temporal | turn |", "|---|---|---|---|---|---|---|---|---|---|"]
            for a_, v in att.items():
                out.append(f"| {a_} | {v['n_gold']} | {v['n_pred']} | {v['matched']} | {v['n_gold'] - v['matched']} | {v['n_pred'] - v['matched']} | "
                           f"{fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['temporal'])} | {fmt(v['turn'])} |")
            out += ["", "polarity: " + ", ".join(f"{k}: n_gold {v['n_gold']}, recall {fmt(v['recall'])}" for k, v in pol.items()), ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/bench")
    ap.add_argument("--out", default="runs/bench/curve.md")
    a = ap.parse_args()
    ROOT[0] = a.root
    pts = [d for d in ORDER if os.path.isdir(os.path.join(a.root, d))]
    pts += sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and d not in pts)
    data = {p: {s: (load(f"{a.root}/{p}/{s}/metrics.json"), load(f"{a.root}/{p}/{s}/resources.json")) for s in SLICES} for p in pts}

    def m(p, s, k):
        return (data[p][s][0] or {}).get(k)

    def r(p, s, k):
        return (data[p][s][1] or {}).get(k)

    # Парето по val: нет точки с не большим gpu_sec и не меньшим F1 (строго лучшей хотя бы по одному)
    gp = [p for p in pts if m(p, "val", "f1") is not None and r(p, "val", "gpu_sec_per_call") is not None]
    pareto = set()
    for p in gp:
        dom = any(q != p and r(q, "val", "gpu_sec_per_call") <= r(p, "val", "gpu_sec_per_call") and m(q, "val", "f1") >= m(p, "val", "f1")
                  and (r(q, "val", "gpu_sec_per_call") < r(p, "val", "gpu_sec_per_call") or m(q, "val", "f1") > m(p, "val", "f1")) for q in gp)
        if not dom:
            pareto.add(p)

    base = load("data/raw/baseline_metrics.json") or {}
    ens = load("runs/ens-v3/params.json") or {}
    hw = next(((r(p, "val", "hardware") for p in pts if r(p, "val", "hardware"))), {})
    L = ["# Кривая «ресурсы × качество»", "",
         "**Методика (cfe/bench.py).** Монопольный GPU; прогрев на двух звонках исключён; GPU-секунды = время между "
         "`cuda.synchronize()` вокруг инференса GLiNER и генерации Gemma / число звонков (для vLLM — стеночное время "
         "параллельных запросов пачки); загрузка моделей не входит. `peak_vram_smi_gb` — максимум nvidia-smi за прогон "
         "(для vLLM включает резерв `gpu-memory-utilization`). Срезы: val; p20 / p40 — val с `asr_noise --p 0.2/0.4 --seed 7` "
         "(noise_level 0,45 / 0,65 против 0,25); p60 — `--p 0.6` (0,85), только базовые точки; на p20/p40 `grounding_span` не показывается (шум не пересчитывает gold-спаны).", "",
         f"Железо: {json.dumps(hw, ensure_ascii=False)}", "",
         f"Опорная точка качества: `{REF}` (ансамбль v3 с настройками по умолчанию).", ""]
    if base:
        L.append("Планка организаторов (baseline_metrics.json): " + ", ".join(f"{k} {fmt(base.get(k))}" for k in GATE))
        L.append("")
    if ens.get("oof"):
        L.append("OOF ансамбля (runs/ens-v3): " + ", ".join(f"{k} {fmt(v)}" for k, v in ens["oof"].items() if isinstance(v, (int, float)))
                 + ". Замеры ансамбля на val — in-sample для реранкера, OOF-строка — несмещённая оценка.")
        L.append("")
    cols = ["params", "weights_disk_gb", "gpu_sec_per_call", "wall_sec_per_call", "cpu_sec_per_call", "peak_vram_gb", "peak_rss_gb"]
    mc = ["fact_precision", "fact_recall", "f1", "temporal_recall", "state_accuracy", "grounding_turn"]
    rows = []
    for s in SLICES:
        L += [f"## Срез {s}", ""]
        head = ["точка"] + cols + ["peak_vram_smi_gb", "P", "R", "F1", "temporal", "state", "grounding_turn"] + (["grounding_span"] if s == "val" else []) + ["schema_violations", "pareto"]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for p in pts:
            met, res = data[p][s]
            if res is None and s == "val":
                err = ""
                ep = f"{a.root}/{p}/error.txt"
                if os.path.exists(ep):
                    err = open(ep).read().strip().replace("\n", " ")[:200]
                L.append(f"| {p} | failed: {err} |" + " |" * (len(head) - 2))
                continue
            if res is None:
                continue
            cells = [p, fmt(res.get("params")), fmt(res.get("weights_disk_gb"), 2), fmt(res.get("gpu_sec_per_call"), 3),
                     fmt(res.get("wall_sec_per_call"), 3), fmt(res.get("cpu_sec_per_call"), 3), fmt(res.get("peak_vram_gb"), 2),
                     fmt(res.get("peak_rss_gb"), 2), fmt(res.get("peak_vram_smi_gb"), 2)]
            cells += [fmt((met or {}).get(k)) for k in mc]
            if s == "val":
                cells.append(fmt((met or {}).get("grounding_span")))
            cells += [str(res.get("schema_violations")), "yes" if p in pareto and s == "val" else ""]
            L.append("| " + " | ".join(cells) + " |")
            rows.append([s, p] + cells[1:])
        L.append("")
    L += ["## Падение от шума (F1 и temporal / state)", "", "| точка | F1 val | F1 p20 | F1 p40 | F1 p60 | Δ p40 | Δ p60 | state val | state p40 |", "|---|---|---|---|---|---|---|---|---|"]
    for p in pts:
        f = [m(p, s, "f1") for s in SLICES]
        if f[0] is None:
            continue
        d = None if f[2] is None else f[2] - f[0]
        d6 = None if f[3] is None else f[3] - f[0]
        L.append(f"| {p} | {fmt(f[0])} | {fmt(f[1])} | {fmt(f[2])} | {fmt(f[3])} | {fmt(d)} | {fmt(d6)} | {fmt(m(p,'val','state_accuracy'))} | {fmt(m(p,'p40','state_accuracy'))} |")
    L.append("")
    L += ["## Латентность и память (время обработки пачки: при batch 1 — задержка одного звонка)", "",
          "| точка | срез | n | batch | p50, с | p95, с | max, с | wall/звонок, с | CPU-с/звонок | peak RSS, ГБ | peak VRAM smi, ГБ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pts:
        for s in SLICES:
            res = data[p][s][1]
            if res:
                L.append(f"| {p} | {s} | {res.get('n_calls')} | {res.get('batch')} | {fmt(res.get('batch_latency_p50_s'),1)} | {fmt(res.get('batch_latency_p95_s'),1)} | "
                         f"{fmt(res.get('batch_latency_max_s'),1)} | {fmt(res.get('wall_sec_per_call'),1)} | {fmt(res.get('cpu_sec_per_call'),1)} | "
                         f"{fmt(res.get('peak_rss_gb'),2)} | {fmt(res.get('peak_vram_smi_gb'),2)} |")
    L.append("")
    L += errors_section(pts, data)
    L.append("Умолчания образов не менялись по результатам кривой: vLLM, batch 16, промпт 3072, генерация 1024.")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slice", "point"] + cols + ["peak_vram_smi_gb"] + mc + ["grounding_span", "schema_violations", "pareto"])
        for row in rows:
            w.writerow(row)
    print(a.out)


if __name__ == "__main__":
    main()
