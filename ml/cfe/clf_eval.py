"""Общий отчёт GLiNER по answers.jsonl. Только stdlib."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case_metrics  # noqa: E402
case_metrics.load_catalog(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "catalog.json"))
import textsim  # noqa: E402
import clf_task as T  # noqa: E402
import codec  # noqa: E402
import rules  # noqa: E402

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VCONFS = [0.0, 0.5, 0.7, 0.9]
ROOT = T.ROOT


def load_answers(path):
    return {r["call_id"]: r for r in map(json.loads, open(path, encoding="utf-8")) if r}


def thresholds(tau):
    """Одинаковый порог детекции для всех атрибутов, без порога уверенности значения."""
    return {"det": {a: tau for a in T.ATTRS}, "value_conf": {a: 0.0 for a in T.ATTRS}}


def predict(call, ans, th, catalog, objs, dm):
    slots = []
    for key, p in ans["det"].items():
        t, a = key.split("|")
        if p is None or p < th["det"][a]:
            continue
        ty = ans["typed"].get(key)
        if not ty or not ty.get("value") or ty.get("value_conf", 1.0) < th["value_conf"][a]:
            continue
        slots.append({
            "t": int(t), "attr": a, "value": ty["value"],
            "polarity": rules.polarity(a, ty["value"]),
            "offset": rules.resolve_date(dm, call, int(t), ty["value"], a),
            "object": (ty.get("object") or objs[0]) if a == "service_state" else None,
        })
    slots.sort(key=lambda s: (s["t"], s["attr"]))
    return {"call_id": call["call_id"], "facts": T.assemble(call, slots, ans.get("name"), catalog)}


def _auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sr = sum(ranks[k] for k, (_, l) in enumerate(allv) if l == 1)
    return round((sr - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def oracle(calls, catalog, vl, dm=None):
    preds = {}
    for c in calls:
        slots, name = [], ""
        for f in c["facts"]:
            if f["object"]["type"] == "client" and not name:
                name = f["object"]["name"]
        for (t, a), f in T.gold_slots(c).items():
            if f["value"] not in vl[a]:
                continue
            if dm is not None:
                k = rules.resolve_date(dm, c, t, f["value"], a)
                pol = rules.polarity(a, f["value"])
            else:
                k = codec.day_offset(c["started_at"], f["valid_from"])
                pol = f["polarity"]
            slots.append({"t": t, "attr": a, "value": f["value"], "polarity": pol,
                          "offset": k, "object": f["object"]["name"] if a == "service_state" else None})
        preds[c["call_id"]] = {"facts": T.assemble(c, slots, name, catalog)}
    return preds



def diagnostics(calls, answers, vl, dm):
    pos, neg = [], []
    v_ok = v_n = d_ok = d_n = p_ok = p_n = n_ok = n_n = 0
    for c in calls:
        ans = answers.get(c["call_id"])
        if not ans:
            continue
        gs = T.gold_slots(c)
        for t in range(len(c["turns"])):
            for a in T.ATTRS:
                p = ans["det"].get(f"{t}|{a}")
                if p is None:
                    continue
                (pos if (t, a) in gs else neg).append(p)
        for (t, a), f in gs.items():
            ty = ans["typed"].get(f"{t}|{a}")
            if not ty:
                continue
            v_n += 1
            v_ok += textsim._value_sim(f["value"], ty.get("value") or "") >= 0.85
            d_n += 1
            d_ok += rules.resolve_date(dm, c, t, ty["value"], a) == codec.day_offset(c["started_at"], f["valid_from"])
            p_n += 1
            p_ok += rules.polarity(a, ty["value"]) == f["polarity"]
        gold_names = [f["object"]["name"] for f in c["facts"] if f["object"]["type"] == "client"]
        if gold_names:
            n_n += 1
            g = gold_names[0].split()[0].lower()
            pn = (ans.get("name") or "").lower()
            n_ok += bool(pn) and textsim._sim(g, pn) >= 0.8
    r = lambda a, b: round(a / b, 4) if b else None
    return {"detection_auc": _auc(pos, neg), "value_acc": r(v_ok, v_n), "n_typed_gold": v_n,
            "date_acc": r(d_ok, d_n), "polarity_acc": r(p_ok, p_n), "name_acc": r(n_ok, n_n)}


def _f1(m):
    p, r = m["fact_precision"], m["fact_recall"]
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def tune_thresholds(sub, answers, catalog, objs, dm, start):
    """Покоординатный подъём F1 на dev: для каждого атрибута по T.ATTRS — сначала τ детекции из TAUS, затем порог value_conf из VCONFS; старт — thresholds(start)."""
    gold = {c["call_id"]: c for c in sub}

    def f1(th):
        return _f1(case_metrics.evaluate(gold, {c["call_id"]: predict(c, answers[c["call_id"]], th, catalog, objs, dm) for c in sub}))
    th = thresholds(start)
    best = f1(th)
    for a in T.ATTRS:
        for kind, grid in (("det", TAUS), ("value_conf", VCONFS)):
            for v in grid:
                cand = {**th, kind: {**th[kind], a: v}}
                s = f1(cand)
                if s > best:
                    best, th = s, cand
    return th


def report(calls, answers, out, method, catalog=None, th=None, tune=None):
    B = T.baseline()
    catalog = catalog or T.load_catalog()
    dm = rules.load_date_model()
    vl = T.value_labels(catalog)
    objs = T.object_labels(catalog)
    gold = {c["call_id"]: c for c in calls}
    answers = {k: v for k, v in answers.items() if k in gold}
    sub = [c for c in calls if c["call_id"] in answers]
    subgold = {c["call_id"]: c for c in sub}
    rows = {}
    for tau in TAUS:
        preds = {}
        with open(os.path.join(out, f"pred_tau{tau}.jsonl"), "w", encoding="utf-8") as fh:
            for c in sub:
                p = predict(c, answers[c["call_id"]], thresholds(tau), catalog, objs, dm)
                preds[c["call_id"]] = p
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        m = case_metrics.evaluate(subgold, preds)
        m["f1"] = _f1(m)
        rows[f"{method} tau={tau}"] = m
    best = max(TAUS, key=lambda t: rows[f"{method} tau={t}"]["f1"])
    m = case_metrics.evaluate(subgold, oracle(sub, catalog, vl))
    m["f1"] = _f1(m)
    rows["oracle"] = m
    m = case_metrics.evaluate(subgold, oracle(sub, catalog, vl, dm))
    m["f1"] = _f1(m)
    rows["oracle+dates"] = m
    for name, path in (("baseline", "data/raw/baseline_metrics.json"), ("lora", "runs/cfe-0.8b/val/metrics.json")):
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            b = json.load(open(p))
            b["f1"] = _f1(b)
            rows[name] = b
    if th is not None:
        preds = {c["call_id"]: predict(c, answers[c["call_id"]], th, catalog, objs, dm) for c in sub}
        m = case_metrics.evaluate(subgold, preds)
        m["f1"] = _f1(m)
        rows[f"{method} per-attr"] = m
    bm = rows[f"{method} tau={best}"]
    gate = {"best_tau": best, "f1_gt_baseline": bm["f1"] > B["f1"],
            "temporal_ok": (bm["temporal_recall"] or 0) >= B["temporal_recall"] - 0.02,
            "state_ok": (bm["state_accuracy"] or 0) >= B["state_accuracy"] - 0.02}
    gate["passed"] = all(gate[k] for k in ("f1_gt_baseline", "temporal_ok", "state_ok"))
    summary = {"method": method, "n_calls": len(sub), "rows": rows, "gate": gate,
               "best": {"tau": best, **rows[f"{method} tau={best}"]},
               "diagnostics": diagnostics(sub, answers, vl, dm)}
    if th is not None:
        pm = rows[f"{method} per-attr"]
        summary["per_attr"] = {"thresholds": th, **pm}
        summary["gate_per_attr"] = {"f1_gt_baseline": pm["f1"] > B["f1"],
                                    "temporal_ok": (pm["temporal_recall"] or 0) >= B["temporal_recall"] - 0.02,
                                    "state_ok": (pm["state_accuracy"] or 0) >= B["state_accuracy"] - 0.02}
        summary["gate_per_attr"]["passed"] = all(summary["gate_per_attr"].values())
    if tune:
        json.dump(tune_thresholds(sub, answers, catalog, objs, dm, best), open(tune, "w", encoding="utf-8"), indent=1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    cols = ["fact_precision", "fact_recall", "f1", "temporal_recall", "state_accuracy", "n_pred_facts"]
    md = [f"# {method}: {len(sub)} звонков", "", "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, v in rows.items():
        md.append(f"| {k} | " + " | ".join(str(v.get(c)) for c in cols) + " |")
    md += ["", f"gate: {json.dumps(gate, ensure_ascii=False)}", "", "diagnostics:"]
    md += [f"- {k}: {v}" for k, v in summary["diagnostics"].items()]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--taus", help="JSON с порогами по атрибутам (data/det_taus.json)")
    ap.add_argument("--tune", help="куда записать пороги, подобранные на этих звонках (dev)")
    a = ap.parse_args()
    th = json.load(open(a.taus, encoding="utf-8")) if a.taus else None
    report(T.load_calls(a.calls), load_answers(a.answers), a.out, a.method, th=th, tune=a.tune)
