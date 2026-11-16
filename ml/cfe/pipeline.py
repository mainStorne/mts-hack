"""Инференс CFE: GLiNER2.5 (детекция, значение, объект, имя) + rules.py (дата, полярность)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clf_eval  # noqa: E402
import clf_task as T  # noqa: E402
import rules  # noqa: E402
from gliner_data import NAME_TYPE  # noqa: E402


def _tasks(**tasks):
    return {k: {"labels": labels, "multi_label": multi} for k, (labels, multi) in tasks.items()}


class Pipeline:
    def __init__(self, model_path, det_min=0.3):
        import torch
        from gliner2 import AutoExtractor
        self.det_min = det_min
        self.catalog = T.load_catalog()
        self.vl = T.value_labels(self.catalog)
        self.objs = T.object_labels(self.catalog)
        if not os.path.exists(rules.DATE_MODEL_PATH):
            raise FileNotFoundError("data/date_model.json: run python3 cfe/rules.py")
        self.dm = rules.load_date_model()
        self.ext = AutoExtractor.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.ext.to("cuda")

    def answers(self, call):
        """Сырые ответы модели: вероятности детекции по (реплика|атрибут), типизация, имя клиента."""
        det_labels = [T.ATTR_RU[x] for x in T.ATTRS] + [T.NO_FACT]
        det, typed = {}, {}
        # classify_text, а не Classifier: схема Classifier запрещает скобки в метках, а обучение шло с ними
        for t in range(len(call["turns"])):
            ctx = T.render_context(call, t)
            res = self.ext.classify_text(ctx, _tasks(attrs=(det_labels, True)), threshold=0.0, include_confidence=True)
            p = {r["label"]: r["confidence"] for r in res["attrs"]}
            for at in T.ATTRS:
                det[f"{t}|{at}"] = p.get(T.ATTR_RU[at], 0.0)
            for at in T.ATTRS:
                if det[f"{t}|{at}"] < self.det_min:
                    continue
                tasks = {f"value_{at}": (self.vl[at], False)}
                if at == "service_state":
                    tasks["object"] = (self.objs, False)
                r = self.ext.classify_text(ctx + "\nТип факта: " + T.ATTR_RU[at], _tasks(**tasks), include_confidence=True)
                v = r[f"value_{at}"]
                typed[f"{t}|{at}"] = {"value": v["label"], "value_conf": v["confidence"],
                                      "object": r["object"]["label"] if at == "service_state" else None}
        e = self.ext.extract_entities(T.transcript_head(call), [NAME_TYPE], include_confidence=True)
        ents = e["entities"].get(NAME_TYPE, [])
        name = max(ents, key=lambda x: x["confidence"])["text"] if ents else None
        return {"call_id": call["call_id"], "det": det, "name": name, "typed": typed}

    def extract(self, call, th):
        """Факты звонка в схеме кейса; th — словарь порогов clf_eval.thresholds."""
        return clf_eval.predict(call, self.answers(call), th, self.catalog, self.objs, self.dm)["facts"]
