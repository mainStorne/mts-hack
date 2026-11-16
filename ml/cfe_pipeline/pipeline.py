#!/usr/bin/env python3
"""CompactFactExtraction: CPU-пайплайн «очистка ASR → кандидаты по каталогу → классификатор → сборка фактов».

  python3 pipeline.py train  data/train_clean.jsonl -o model.pkl
  python3 pipeline.py predict model.pkl data/val_clean.jsonl -o pred.jsonl
"""
import argparse, json, pickle, re, sys, collections
from datetime import datetime

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from candidates import CandidateGen
from features import mention_features, FAIL, REJECT as REJECT_RX
from labels import gold_matches
import dates as DT
import datesel as DS
import datelogic as DLG

HERE = __file__.rsplit("/", 1)[0] or "."
ABSTAIN_PAST_NORMAL = True
ATTRS = ["service_state", "action", "agreement", "tariff", "ticket_status"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def action_value(c, d):
    """«X — не помогло», если рядом звучит неудача"""
    turns = d["turns"]; i = c["turn"]
    txt = " ".join(turns[j]["text"].lower() for j in range(i, min(len(turns), i + 3)))
    return c["value"] + " — не помогло" if FAIL.search(txt) else c["value"]


def final_value(c, d):
    return action_value(c, d) if c["attr"] == "action" else c["value"]


def polarity(value):
    return -1 if value == "больше не используется" else 1


def mentions(d, gen):
    cands, objs, tickets, toks = gen.call(d)
    feats = [mention_features(c, d, toks) for c in cands]
    for c in cands:
        c["fvalue"] = final_value(c, d)
    return cands, feats, objs, tickets


def label_mentions(d, cands):
    y = []
    for c in cands:
        m = gold_matches(d["facts"], c["attr"], c["fvalue"], polarity(c["value"]))
        y.append(1 if m else 0)
    return y


def date_examples(d, cands):
    """обучающие примеры для ранжировщиков дат: (feats, label) для valid_from и valid_to"""
    XF, YF, XT, YT = [], [], [], []
    for c in cands:
        if c["attr"] not in DS.TEMP_ATTRS:
            continue
        ms = gold_matches(d["facts"], c["attr"], c["fvalue"], polarity(c["value"]))
        ms = [d["facts"][gi] for gi, _ in ms if abs((d["facts"][gi]["source"]["turn_idx"] or 0) - c["turn"]) <= 1]
        if not ms:
            continue
        g = min(ms, key=lambda g: abs(g["source"]["turn_idx"] - c["turn"]))
        vf, vt = g["valid_from"][:10], (g["valid_to"] or "")[:10] or None
        fo = DS.date_options(d, c)
        if any(day == vf for day, _ in fo):
            for day, f in fo:
                XF.append(f); YF.append(int(day == vf))
        to = DS.to_options(d, c)
        if any(day == vt for day, _ in to):
            for day, f in to:
                XT.append(f); YT.append(int(day == vt))
    return XF, YF, XT, YT


def fit_rankers(XF, YF, XT, YT, XA=None, YA=None, C=1.0):
    out = {}
    for name, X, Y in (("date_from", XF, YF), ("date_to", XT, YT), ("anchor", XA, YA)):
        if not X:
            continue
        vec = DictVectorizer(); clf = LogisticRegression(C=C, max_iter=3000).fit(vec.fit_transform(X), Y)
        out[name] = {"vec": vec, "clf": clf}
    return out


def choose(ranker, opts):
    if not opts:
        return None
    pr = ranker["clf"].predict_proba(ranker["vec"].transform([f for _, f in opts]))[:, 1]
    return opts[int(np.argmax(pr))][0]


FIXATE = re.compile(r"(фиксир|оформл|оформил|записыва|записал|зафиксир|итак|итог|резюмир|подытож|договорились|подтвержда)")
CONF = re.compile(r"\b(да|подходит|давайте|хорошо|согласн|ладно|конечно|устраивает|подтверждаю|оформляйте|договорились|угу|ага)\b")


def anchor_options(d, mention_turns, attr):
    """кандидаты якорной реплики факта: реплики упоминаний и 1–2 реплики после них"""
    mt = sorted(set(mention_turns)); first, last = mt[0], mt[-1]
    opts = sorted({t + k for t in mt for k in (0, 1, 2) if t + k < len(d["turns"])})
    out = []
    for t in opts:
        tr = d["turns"][t]; low = tr["text"].lower(); sp = tr["speaker"] or "?"
        prev_m = max([m for m in mt if m <= t], default=first)
        f = {f"a={attr}": 1, f"a_sp={attr}:{sp}": 1, f"a_ment={attr}:{t in mt}": 1,
             f"a_off={attr}:{min(6, t - first)}": 1, f"a_offprev={attr}:{min(3, t - prev_m)}:{t in mt}": 1,
             f"a_firstlast={attr}:{'first' if t == first else ('last' if t == last else 'mid')}": 1,
             f"a_nment={attr}:{min(3, len(mt))}": 1, f"a_q={attr}:{'?' in low}": 1,
             f"a_short={attr}:{min(4, len(low) // 40)}": 1}
        for tag, rx in (("fix", FIXATE), ("conf", CONF), ("rej", REJECT_RX)):
            if rx.search(low):
                f[f"a_{tag}={attr}:{sp}:{t in mt}"] = 1
        if t > 0 and "?" in d["turns"][t - 1]["text"]:
            f[f"a_prevq={attr}:{sp}"] = 1
        out.append((t, f))
    return out


def anchor_examples(d, cands):
    X, Y = [], []
    groups = collections.defaultdict(list)
    for c in cands:
        groups[(c["attr"], c["fvalue"])].append(c)
    for (attr, val), cs in groups.items():
        ms = gold_matches(d["facts"], attr, val, polarity(cs[0]["value"]))
        if not ms:
            continue
        first = min(c["turn"] for c in cs)
        g = min((d["facts"][gi] for gi, _ in ms), key=lambda g: abs(g["source"]["turn_idx"] - first))
        opts = anchor_options(d, [c["turn"] for c in cs], attr)
        if any(t == g["source"]["turn_idx"] for t, _ in opts):
            for t, f in opts:
                X.append(f); Y.append(int(t == g["source"]["turn_idx"]))
    return X, Y


def train(args):
    gen = CandidateGen(f"{HERE}/data/catalog.json")
    X, Y = [], []
    for n, d in enumerate(load(args.data)):
        cands, feats, _, _ = mentions(d, gen)
        X += feats; Y += label_mentions(d, cands)
        if (n + 1) % 500 == 0: print(n + 1, len(X), file=sys.stderr)
    vec = DictVectorizer()
    Xm = vec.fit_transform(X)
    clf = LogisticRegression(C=args.C, max_iter=3000, class_weight=None)
    clf.fit(Xm, Y)
    print(f"обучено: {len(Y)} упоминаний, доля позитивов {np.mean(Y):.3f}", file=sys.stderr)
    pickle.dump({"vec": vec, "clf": clf, "thr": {a: args.thr for a in ATTRS}}, open(args.o, "wb"))


# ---------------------------------------------------------------- сборка фактов
def nearest(items, turn):
    """ближайшее упоминание: в той же реплике → последнее до → первое после"""
    same = [x for x in items if x[0] == turn]
    if same: return same[0]
    before = [x for x in items if x[0] < turn]
    if before: return before[-1]
    return items[0] if items else None


def canon_tickets(tickets):
    """обрезанный ASR номер («525») → полный номер того же звонка («5252»)"""
    nums = {n for _, n in tickets}
    full = {n: max((m for m in nums if m.startswith(n)), key=len) for n in nums}
    return [(t, full[n]) for t, n in tickets]


def assemble(d, cands, probs, objs, tickets, thr, rankers=None):
    call_dt = datetime.fromisoformat(d["started_at"]); base = call_dt.date()
    tickets = canon_tickets(tickets)
    groups = collections.defaultdict(list)
    for c, p in zip(cands, probs):
        groups[(c["attr"], c["fvalue"])].append((p, c))
    facts = []
    for (attr, value), ms in groups.items():
        p, _ = max(ms, key=lambda x: x[0])
        if p < thr[attr]:
            continue
        # якорь — первое уверенное упоминание; при наличии обученного выбора — лучшая реплика из окрестности
        c = min((c for q, c in ms if q >= 0.5 * p), key=lambda c: c["turn"])
        ti = c["turn"]
        if rankers and "anchor" in rankers:
            ti = choose(rankers["anchor"], anchor_options(d, [m["turn"] for q, m in ms if q >= 0.5 * p], attr))
        t = d["turns"][ti]
        if attr == "ticket_status":
            tk = nearest(tickets, ti)
            obj = {"type": "ticket", "name": f"заявка {tk[1]}" if tk else "заявка"}
        elif attr == "service_state":
            ob = nearest(objs, ti)
            obj = {"type": ob[1], "name": ob[2]} if ob else {"type": "service", "name": ""}
        else:
            obj = {"type": "client", "name": "клиент"}
        vf, vt = d["started_at"], None   # даты проставляет datelogic ниже
        facts.append({"object": obj, "attribute": attr, "value": value, "polarity": polarity(value),
                      "speaker": t["speaker"], "valid_from": vf, "valid_to": vt,
                      "source": {"turn_idx": ti, "span": [c["start"], c["end"]], "quote": t["text"][c["start"]:c["end"]]},
                      "_p": round(float(p), 4), "_pos": c["start"], "_c": c,
                      "_mentions": sorted({(m["turn"], m["start"]) for q, m in ms if q >= 0.5 * p})})
    # «работает нормально» при наличии поломки в том же звонке — это прошлое состояние, начало которого
    # в разговоре не называется (в эталоне — произвольная дата): такой факт не выдаём, полно заполнить его нельзя
    if ABSTAIN_PAST_NORMAL and any(f["attribute"] == "service_state" and "нормально" not in f["value"] for f in facts):
        facts = [f for f in facts if not (f["attribute"] == "service_state" and f["value"] == "работает нормально")]
    agr = (lambda f: choose(rankers["date_from"], DS.date_options(d, f["_c"]))) if rankers else None
    DLG.set_dates(d, facts, key_fn=lambda f: f["object"]["name"], agreement_from=agr)
    for f in facts:
        f.pop("_pos", None); f.pop("_c", None); f.pop("_mentions", None)
    return facts


def predict_calls(model, calls):
    global ABSTAIN_PAST_NORMAL
    ABSTAIN_PAST_NORMAL = model.get("abstain_past_normal", ABSTAIN_PAST_NORMAL)
    gen = CandidateGen(f"{HERE}/data/catalog.json")
    out = []
    for d in calls:
        cands, feats, objs, tickets = mentions(d, gen)
        probs = model["clf"].predict_proba(model["vec"].transform(feats))[:, 1] if feats else []
        facts = assemble(d, cands, probs, objs, tickets, model["thr"], model.get("rankers"))
        out.append({"call_id": d["call_id"], "started_at": d["started_at"], "facts": facts})
    return out


def predict(args):
    model = pickle.load(open(args.model, "rb"))
    if args.thr is not None:
        model["thr"] = {a: args.thr for a in ATTRS}
    with open(args.o, "w", encoding="utf-8") as fo:
        for r in predict_calls(model, load(args.data)):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("train"); a.add_argument("data"); a.add_argument("-o", default="model.pkl")
    a.add_argument("--C", type=float, default=0.5); a.add_argument("--thr", type=float, default=0.5)
    b = sp.add_parser("predict"); b.add_argument("model"); b.add_argument("data"); b.add_argument("-o", default="pred.jsonl")
    b.add_argument("--thr", type=float)
    args = ap.parse_args()
    {"train": train, "predict": predict}[args.cmd](args)
