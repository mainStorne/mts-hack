import argparse
import json
from datetime import datetime

import case_metrics as cm

ATTRS = ["service_state", "tariff", "ticket_status", "action", "agreement"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def breakdown(gold, pred):
    pm = {p["call_id"]: p for p in pred}
    st = {a: dict(n_gold=0, n_pred=0, matched=0, temporal_ok=0, n_temporal=0, turn_ok=0) for a in ATTRS}
    pol = {1: [0, 0], -1: [0, 0]}  # gold, matched
    for g in gold:
        gf = g["facts"]
        pf = pm.get(g["call_id"], {}).get("facts", [])
        for f in gf:
            st[f["attribute"]]["n_gold"] += 1
            pol[f["polarity"]][0] += 1
        for f in pf:
            st[f["attribute"]]["n_pred"] += 1
        day = datetime.fromisoformat(g["started_at"]).date()
        rec = {d.isoformat() for t in g.get("turns", []) for d in cm.recover_dates(t["text"], day)}
        for gf_, pf_, _ in cm.match_facts(gf, pf):
            s = st[gf_["attribute"]]
            s["matched"] += 1
            pol[gf_["polarity"]][1] += 1
            gd = {cm._day(gf_.get("valid_from")), cm._day(gf_.get("valid_to"))} - {None}
            if gf_["attribute"] in cm.TEMPORAL_ATTRS and gd & rec:
                s["n_temporal"] += 1
                if cm._day(gf_.get("valid_from")) == cm._day(pf_.get("valid_from")) and \
                        cm._day(gf_.get("valid_to")) == cm._day(pf_.get("valid_to")):
                    s["temporal_ok"] += 1
            if (pf_.get("source") or {}).get("turn_idx") == gf_["source"]["turn_idx"]:
                s["turn_ok"] += 1
    r = lambda a, b: round(a / b, 4) if b else None
    out = {}
    for a, s in st.items():
        out[a] = {**s, "precision": r(s["matched"], s["n_pred"]), "recall": r(s["matched"], s["n_gold"]),
                  "temporal": r(s["temporal_ok"], s["n_temporal"]), "turn": r(s["turn_ok"], s["matched"])}
    return out, {str(k): {"n_gold": v[0], "recall": r(v[1], v[0])} for k, v in pol.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cm.load_catalog(a.gold)
    gold = read_jsonl(a.gold)
    base = json.load(open("data/raw/baseline_metrics.json"))
    f = lambda x: "—" if x is None else f"{x:.4f}"
    rows = ["| run | params (B) | gpu_sec_per_call | peak_vram_gb | fact_P | fact_R | temporal_recall | state_accuracy | grounding_turn | invalid_calls |",
            "|---|---|---|---|---|---|---|---|---|---|",
            f"| glm-5.3 baseline | — | — | — | {f(base['fact_precision'])} | {f(base['fact_recall'])} | {f(base['temporal_recall'])} | {f(base['state_accuracy'])} | {f(base['grounding_turn'])} | — |"]
    crit, detail = [], []
    for run in a.runs:
        m = json.load(open(f"{run}/metrics.json"))
        rs = json.load(open(f"{run}/resources.json"))
        rows.append(f"| {run} | {rs['params'] / 1e9:.3f} | {rs['gpu_sec_per_call']:.2f} | {rs['peak_vram_gb']:.2f} | "
                    f"{f(m['fact_precision'])} | {f(m['fact_recall'])} | {f(m['temporal_recall'])} | "
                    f"{f(m['state_accuracy'])} | {f(m['grounding_turn'])} | {rs['invalid_calls']} |")
        dt = (m["temporal_recall"] or 0) - base["temporal_recall"]
        ds = (m["state_accuracy"] or 0) - base["state_accuracy"]
        crit.append(f"| {run} | {dt:+.4f} {'OK' if dt >= -0.02 else 'FAIL'} | {ds:+.4f} {'OK' if ds >= -0.02 else 'FAIL'} |")
        attr, pol = breakdown(gold, read_jsonl(f"{run}/pred.jsonl"))
        json.dump({"by_attribute": attr, "by_polarity": pol}, open(f"{run}/by_attribute.json", "w"), indent=1, ensure_ascii=False)
        detail += [f"\n### {run}\n", "| attribute | n_gold | n_pred | matched | precision | recall | temporal | turn |", "|---|---|---|---|---|---|---|---|"]
        for k, s in attr.items():
            detail.append(f"| {k} | {s['n_gold']} | {s['n_pred']} | {s['matched']} | {f(s['precision'])} | {f(s['recall'])} | {f(s['temporal'])} | {f(s['turn'])} |")
        detail.append("\npolarity recall: " + ", ".join(f"{k}: {f(v['recall'])} (n={v['n_gold']})" for k, v in pol.items()))
    md = ["# Summary\n", *rows, "\n## Criterion (Δ vs glm-5.3, threshold −0.02)\n",
          "| run | Δtemporal | Δstate |", "|---|---|---|", *crit, "\n## By attribute", *detail]
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
