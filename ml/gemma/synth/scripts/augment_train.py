#!/usr/bin/env python3
"""Контролируемая аугментация train (подход B): обёртка над synth_lib.augment.

Использует ТОЛЬКО train (val/test как источники не используются).
Для каждого исходного звонка выбирается стек из 1–2 безопасных преобразований;
если применить безопасно не удаётся — звонок пропускается.

Выход:
  data/augmented_train.jsonl       — аугментированные звонки
  data/augment_metadata.jsonl      — provenance (источник, преобразования, сид)
  reports/validation_augmented.json — отчёт validate_synthetic (--no-train-sim:
      аугменты намеренно похожи на train, проверка на копии train неприменима)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from synth_lib import augment as A  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import validate_synthetic as V  # noqa: E402

import re  # noqa: E402  (нужен для _norm_err)

# стеки преобразований: (название, [функции])
STACKS = [
    ("date_shift",        [A.date_shift]),
    ("renumber",          [A.renumber_tickets]),
    ("tariff_rename",     [A.tariff_rename]),
    ("filler_insert",     [A.filler_insert]),
    ("paraphrase",        [A.paraphrase_unreferenced]),
    ("asr_noise",         [A.asr_noise]),
    ("date_shift+asr",    [A.date_shift, A.asr_noise]),
    ("renumber+asr",      [A.renumber_tickets, A.asr_noise]),
    ("filler+paraphrase", [A.filler_insert, A.paraphrase_unreferenced]),
    ("date_shift+filler", [A.date_shift, A.filler_insert]),
    ("tariff+filler",     [A.tariff_rename, A.filler_insert]),
    ("paraphrase+asr",    [A.paraphrase_unreferenced, A.asr_noise]),
]
STACK_WEIGHTS = [3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2]


def _norm_err(msg: str) -> str:
    """Нормализация текста ошибки: убираем cid, числа, значения в кавычках.
    Позволяет отличать «новые» ошибки аугментации от унаследованных из gold
    (у raw train часть разметки изначально битая: цепочки, span/quote)."""
    msg = msg.split(":", 1)[1].strip()          # префикс "cid: "
    msg = re.sub(r"\d+", "<N>", msg)
    msg = re.sub(r"'[^']*'", "<V>", msg)
    return msg


def _apply_stack(call: dict, names: list[str], rng: random.Random,
                 global_seed: int, i: int) -> tuple[dict | None, list[str]]:
    applied, cur = [], call
    for k, fn in enumerate(names):
        if fn is A.asr_noise:
            p = rng.choice([0.05, 0.08, 0.12])
            nxt = A.asr_noise(cur, p, f"{global_seed}-{i}-{k}")
        elif fn is A.date_shift:
            nxt = A.date_shift(cur, rng.choice([-3, -2, -1, 1, 2, 3]))
        elif fn in (A.renumber_tickets, A.tariff_rename, A.filler_insert,
                    A.paraphrase_unreferenced):
            nxt = fn(cur, rng)
        else:
            nxt = None
        if nxt is None:
            return None, applied
        cur = nxt
        applied.append(fn.__name__)
    return cur, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT.parent / "data/raw/train.jsonl"))
    ap.add_argument("--n-out", type=int, default=360)
    ap.add_argument("--out", default=str(ROOT / "data/augmented_train.jsonl"))
    ap.add_argument("--meta", default=str(ROOT / "data/augment_metadata.jsonl"))
    ap.add_argument("--seed", type=int, default=20260201)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    train = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    # только train; звонки с object.type "other" пропускаем (валидатор
    # допускает service/device/ticket/client — синтетика "other" не использует)
    pool = [c for c in train
            if all(f["object"]["type"] != "other" for f in c.get("facts", []))]

    # ошибки, уже присутствующие в исходной gold-разметке (у raw train часть
    # цепочек/span изначально битая) — аугментация их не должна ухудшать
    src_errs = {}
    for i, c in enumerate(pool):
        errs, _ = V.check_call(c, None)
        src_errs[i] = {_norm_err(e) for e in errs}

    order = list(range(len(pool)))
    rng.shuffle(order)
    out, meta, seen = [], [], set()
    tries = 0
    n_new_err = n_src_err = 0
    while len(out) < a.n_out and tries < a.n_out * 30:
        tries += 1
        i = order[tries % len(order)]
        if (i, tries // len(order)) in seen:
            continue
        src = pool[i]
        names = rng.choices([s[0] for s in STACKS], weights=STACK_WEIGHTS, k=1)[0]
        fns = dict(STACKS)[names]
        aug, applied = _apply_stack(src, fns, rng, a.seed, tries)
        if aug is None:
            continue
        # аугментация не должна добавлять НОВЫЕ ошибки к разметке
        errs_aug, _ = V.check_call(aug, None)
        new = {_norm_err(e) for e in errs_aug} - src_errs[i]
        if new:
            n_new_err += 1
            continue
        if src_errs[i]:
            n_src_err += 1
        key = (i, tuple(sorted(applied)))
        if key in seen:
            continue
        seen.add(key)
        # call_id совпадает с источником (corrupt_call его не меняет) —
        # добавляем суффикс, чтобы в смешанных вариантах не было дубликатов
        aug["call_id"] = f'{aug["call_id"]}#aug{len(out) + 1}'
        out.append(aug)
        meta.append({
            "call_id": aug["call_id"],
            "origin": "augmented_from_train",
            "source_call_id": src["call_id"],
            "transforms": applied,
            "source_had_errors": bool(src_errs[i]),
            "seed": a.seed,
        })
        if len(out) % 100 == 0:
            print(f"  собрано {len(out)}/{a.n_out}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for c in out:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(a.meta, "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"аугментировано: {len(out)} -> {a.out} (исходов задействовано: "
          f"{len(set(m['source_call_id'] for m in meta))}/{len(pool)})")
    print(f"отклонено из-за новых ошибок разметки: {n_new_err}; "
          f"оставлено с унаследованными ошибками gold: {n_src_err}")


if __name__ == "__main__":
    main()
