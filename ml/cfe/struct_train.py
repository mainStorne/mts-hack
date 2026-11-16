#!/usr/bin/env python3
"""LoRA-дообучение GLiNER2.5: value + span attributes (attribute/type/polarity) + person, date, либо структура fact."""
import argparse
import json
import os

from gliner2 import AutoExtractor
from gliner2.processor import SamplingConfig
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "fastino/gliner2.5-multi-v1"

def has_fact(o):
    return bool(o["entities"].get("value")) or any(
        s.get("fact", {}).get("value") not in ("", None) for s in o.get("json_structures", []))


def has_date(o):
    return bool(o["entities"].get("date")) or any(
        s.get("fact", {}).get("date") not in ("", None) for s in o.get("json_structures", []))


def self_check(model, dev_path):
    recs = [json.loads(l) for l in open(dev_path, encoding="utf-8")]
    for r in recs:
        r["output"].setdefault("entities", {})
    pos = next(r for r in recs if has_fact(r["output"]))
    neg = next(r for r in recs if not has_fact(r["output"]))
    dat = next(r for r in recs if has_date(r["output"]))
    for name, rec in (("pos", pos), ("neg", neg), ("date", dat)):
        try:
            model.processor.transform_record(rec["input"], rec["output"], build_targets=True)
        except Exception as exc:
            raise SystemExit(f"self_check failed ({name}): {exc}: {json.dumps(rec, ensure_ascii=False)[:2000]}")
    print("self_check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--data", default="struct")
    ap.add_argument("--out", default="runs/struct-lora")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    model = AutoExtractor.from_pretrained(MODEL)
    model.processor.sampling_config = SamplingConfig(synthetic_entity_label_prob=0.0)
    train = os.path.join(ROOT, "data", a.data, "train.jsonl")
    dev = os.path.join(ROOT, "data", a.data, "dev.jsonl")
    self_check(model, dev)
    kw = dict(output_dir=a.out, num_epochs=a.epochs, eval_strategy="epoch", save_best=True,
              max_train_samples=a.max_train)
    if a.smoke:
        kw = dict(output_dir="runs/struct-smoke", num_epochs=1, eval_strategy="no", max_train_samples=64)
    cfg = TrainingConfig(
        batch_size=a.batch_size, gradient_accumulation_steps=a.accum,
        encoder_lr=1e-4, task_lr=5e-4, warmup_ratio=0.05, scheduler_type="cosine", fp16=True, seed=a.seed,
        logging_steps=50, report_to_wandb=False, validate_data=True,
        use_lora=True, lora_r=16, lora_alpha=32.0, lora_dropout=0.0, save_adapter_only=True,
        lora_target_modules=["encoder", "all_task_heads"], **kw)
    tr = ExtractorTrainer(model, cfg)
    n = sum(p.numel() for n_, p in tr.model.named_parameters() if p.requires_grad and "boundary_head" in n_)
    print(f"lora boundary_head params: {n}")
    if n == 0:
        raise SystemExit("LoRA does not cover boundary_head")
    tr.train(train_data=train, eval_data=None if a.smoke else dev)
    print("TRAIN DONE")


if __name__ == "__main__":
    main()
