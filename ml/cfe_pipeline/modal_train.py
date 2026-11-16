"""Обучение CPU-пайплайна в Modal: признаки параллельно по чанкам (.map), затем LogisticRegression.

  modal run modal_train.py --data data/train_clean.jsonl --out model.pkl [--c 0.5] [--chunks 40]
"""
import json, pickle, pathlib
import modal

HERE = pathlib.Path(__file__).parent
SRC = ["datesel.py", "datelogic.py", "textnorm.py", "candidates.py", "features.py", "labels.py", "pipeline.py", "dates.py", "case_metrics.py"]
ATTRS = ["service_state", "action", "agreement", "tariff", "ticket_status"]
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("pymorphy3==2.0.6", "pymorphy3-dicts-ru", "scikit-learn==1.9.1", "numpy")
         .add_local_file(HERE / "data/catalog.json", "/root/data/catalog.json"))
for f in SRC:
    image = image.add_local_file(HERE / f, f"/root/{f}")

app = modal.App("cfe-train", image=image)


@app.function(cpu=2, memory=2048, timeout=1800)
def featurize(lines: list[str]):
    import sys; sys.path.insert(0, "/root")
    import pipeline as P
    from candidates import CandidateGen
    gen = CandidateGen("/root/data/catalog.json")
    X, Y, D = [], [], [[], [], [], [], [], []]
    for l in lines:
        d = json.loads(l)
        cands, feats, _, _ = P.mentions(d, gen)
        X += feats; Y += P.label_mentions(d, cands)
        for acc, part in zip(D, P.date_examples(d, cands) + P.anchor_examples(d, cands)):
            acc += part
    return X, Y, D


@app.function(cpu=8, memory=16384, timeout=3600)
def fit(X, Y, D, C: float):
    import sys; sys.path.insert(0, "/root")
    import pipeline as P
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = DictVectorizer(); Xm = vec.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=3000).fit(Xm, Y)
    return pickle.dumps({"vec": vec, "clf": clf, "rankers": P.fit_rankers(*D)})


@app.local_entrypoint()
def main(data: str = "data/train_clean.jsonl", out: str = "model.pkl", c: float = 0.5, chunks: int = 40):
    lines = open(data, encoding="utf-8").read().splitlines()
    k = (len(lines) + chunks - 1) // chunks
    parts = [lines[i:i + k] for i in range(0, len(lines), k)]
    X, Y, D = [], [], [[], [], [], [], [], []]
    for x, y, dd in featurize.map(parts):
        X += x; Y += y
        for acc, part in zip(D, dd):
            acc += part
    print(f"признаки: {len(Y)} упоминаний, позитивов {sum(Y) / len(Y):.3f}")
    print(f"примеры дат: from {len(D[1])} (поз. {sum(D[1])}), to {len(D[3])} (поз. {sum(D[3])}); якорей {len(D[5])} (поз. {sum(D[5])})")
    m = pickle.loads(fit.remote(X, Y, D, c))
    m["thr"] = {a: 0.5 for a in ATTRS}
    pickle.dump(m, open(out, "wb"))
    print("модель сохранена:", out)
