"""LoRA SFT Gemma 3 1B для кейса МТС CompactFactExtraction.

Основа — проверенный на этом сервере sft/train.py команды (fp16 + sdpa, т.к.
V100 — Volta sm_70 без нативного bf16 и без FlashAttention-2).

Что добавлено сверх командного скрипта:
  * completion_only_loss=True — loss считается ТОЛЬКО на целевом JSON
    (проверяется автоматически, см. verify_loss_mask);
  * проверка, что LoRA-параметры действительно требуют градиента;
  * smoke-режим для безопасной проверки конфигурации перед полным запуском;
  * контроль свободной видеопамяти перед стартом (GPU общий с командой);
  * фиксация seed, сохранение конфигурации запуска, поддержка resume.

Запуск:
  .venv/bin/python train.py --smoke                       # быстрая проверка
  .venv/bin/python train.py                               # полное обучение
  .venv/bin/python train.py --resume                      # продолжить с последнего чекпоинта
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
FREE_VRAM_WARN_GB = 12.0   # ниже этого — предупредить (GPU общий с командой)


# --------------------------------------------------------------------------- #
def gpu_report() -> tuple[float, float]:
    """(свободно ГБ, всего ГБ) или (-1, -1) если CUDA недоступна."""
    if not torch.cuda.is_available():
        return -1.0, -1.0
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def verify_lora_trainable(model) -> tuple[int, int]:
    """(число обучаемых параметров, всего). Падает, если LoRA не применился."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise SystemExit("КРИТИЧНО: ни один параметр не требует градиента — LoRA не применился")
    return trainable, total


def verify_loss_mask(trainer, tokenizer) -> dict:
    """Проверяет, что loss считается только на целевом JSON, а не на промпте."""
    import re
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    masked = int((labels == -100).sum())
    active = int((labels != -100).sum())
    if active == 0:
        raise SystemExit("КРИТИЧНО: все метки замаскированы — модель ничему не научится")
    if masked == 0:
        raise SystemExit("КРИТИЧНО: нет замаскированных токенов — судя по всему, "
                         "loss считается и на промпте (completion_only_loss не сработал)")

    # декодируем обучаемую часть: это должен быть целевой JSON-массив
    flat = labels[labels != -100].flatten().tolist()
    text = re.sub(r"\s+", "", tokenizer.decode(flat, skip_special_tokens=True))
    ok_json = (text.startswith("[{") and text.endswith("]") and '"a"' in text) \
        or (text != "" and set(text.replace("[]", "")) == set())  # батч только из звонков без фактов: «[][]…»
    info = {
        "masked_prompt_tokens": masked,
        "trained_completion_tokens": active,
        "completion_ratio": round(active / (active + masked), 4),
        "completion_is_target_json": ok_json,
    }
    print(f"  маска loss: промпт замаскирован ({masked} токенов), "
          f"обучается только ответ ({active} токенов, {info['completion_ratio']:.1%})")
    if not ok_json:
        print(f"  КРИТИЧНО: обучаемая часть не похожа на целевой JSON: {text[:150]!r}")
        raise SystemExit("Маска loss настроена неверно — см. completion_only_loss в config.yaml")
    return info


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--train-file", type=Path, default=None)
    ap.add_argument("--eval-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="продолжить с последнего чекпоинта в output_dir")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke-тест: 20 шагов, batch 2, сохранение в adapters/_smoke")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="разрешить обучение на CPU (медленно, но безопасно для "
                         "проверки всего пути без занятия GPU)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    m, t, i = cfg["model"], cfg["train"], cfg["infer"]

    model_name = a.model or m["name"]
    train_file = a.train_file or ROOT / cfg["data"]["out_dir"] / "sft_train.jsonl"
    eval_file = a.eval_file or ROOT / cfg["data"]["out_dir"] / "sft_eval.jsonl"
    out_dir = (a.out or ROOT / t["output_dir"]).resolve()
    if a.smoke:
        out_dir = ROOT / "adapters" / "_smoke"

    epochs = 0.05 if a.smoke else (a.epochs if a.epochs is not None else t["epochs"])
    lr = a.lr if a.lr is not None else t["lr"]
    batch = 2 if a.smoke else (a.batch if a.batch is not None else t["batch_size"])
    accum = 1 if a.smoke else (a.grad_accum if a.grad_accum is not None else t["grad_accum"])
    max_len = a.max_length if a.max_length is not None else t["max_length"]
    max_steps = 20 if a.smoke else a.max_steps
    seed = a.seed if a.seed is not None else t["seed"]
    lora_r = a.lora_r if a.lora_r is not None else cfg["lora"]["r"]
    # SFTConfig (trl 1.18) не принимает warmup_ratio — считаем warmup_steps явно
    warmup_ratio = t["warmup_ratio"]

    use_cpu = a.allow_cpu or not torch.cuda.is_available()
    if use_cpu:
        print("Режим CPU: GPU не занимаем (проверка всего пути обучения).")
    else:
        free, total = gpu_report()
        print(f"GPU: {torch.cuda.get_device_name(0)}; свободно {free:.1f} / {total:.1f} ГБ")
        if free >= 0 and free < FREE_VRAM_WARN_GB:
            print(f"ОТМЕНА: свободно меньше {FREE_VRAM_WARN_GB} ГБ — GPU занят другим "
                  f"участником. Используйте --allow-cpu для проверки пути или "
                  f"согласуйте ресурсы с командой.")
            raise SystemExit(2)

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---- токенайзер (нужен до выбора примеров для smoke) -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- данные ------------------------------------------------------------------
    train_ds = load_dataset("json", data_files=str(train_file))["train"]
    eval_ds = load_dataset("json", data_files=str(eval_file))["train"] if eval_file.exists() else None
    if a.smoke:
        # Берём САМЫЕ КОРОТКИЕ примеры: при маленьком max_length длинные
        # обрезаются целиком, и TRL молча отбрасывает их как fully masked.
        def tok_len(ex):
            text = tokenizer.apply_chat_template(ex["prompt"], tokenize=False,
                                                 add_generation_prompt=True)
            return (len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    + len(tokenizer(ex["completion"][0]["content"],
                                    add_special_tokens=False)["input_ids"]))
        order = sorted(range(len(train_ds)), key=lambda i: tok_len(train_ds[i]))
        keep = [i for i in order if tok_len(train_ds[i]) <= max_len][:64] or order[:64]
        train_ds = train_ds.select(keep)
        if eval_ds is not None:
            eval_ds = eval_ds.select(range(min(8, len(eval_ds))))
        print(f"smoke: отобраны {len(train_ds)} самых коротких примеров (<= {max_len} токенов)")
    print(f"train: {len(train_ds)} примеров; eval: {len(eval_ds) if eval_ds is not None else 0}")

    # SFTConfig (trl 1.18) не принимает warmup_ratio — пересчитываем в шаги
    steps_per_epoch = max(1, len(train_ds) // (batch * accum))
    total_steps = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    print(f"шагов на эпоху: {steps_per_epoch}; всего шагов: {total_steps}; "
          f"warmup: {warmup_steps}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[m["torch_dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=m["attn_implementation"])

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    sft = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        max_length=max_len,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=t["gradient_checkpointing"],
        packing=t["packing"],
        completion_only_loss=t["completion_only_loss"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=t["logging_steps"],
        save_strategy="steps" if a.smoke else "epoch",
        save_steps=10 if a.smoke else 500,
        save_total_limit=2,
        eval_strategy="epoch" if (eval_ds is not None and not a.smoke) else "no",
        seed=seed,
        data_seed=seed,
        report_to="none",
        use_cpu=use_cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # ---- проверки перед обучением ---------------------------------------------------
    trainable, total_p = verify_lora_trainable(trainer.model)
    print(f"LoRA применён: обучаемых параметров {trainable:,} из {total_p:,} "
          f"({trainable/total_p:.2%})")
    mask_info = verify_loss_mask(trainer, tokenizer)

    run_cfg = {
        "model": model_name, "seed": seed, "epochs": epochs, "lr": lr,
        "batch_size": batch, "grad_accum": accum, "max_length": max_len,
        "max_steps": max_steps, "lora_r": lora_r, "lora_alpha": cfg["lora"]["alpha"],
        "target_modules": cfg["lora"]["target_modules"],
        "warmup_ratio": warmup_ratio, "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "completion_only_loss": t["completion_only_loss"],
        "torch_dtype": m["torch_dtype"], "attn": m["attn_implementation"],
        "trainable_params": trainable, "total_params": total_p,
        "loss_mask": mask_info, "train_file": str(train_file),
        "n_train": len(train_ds), "smoke": a.smoke,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- обучение ---------------------------------------------------------------------
    resume = None
    if a.resume:
        cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not cks:
            raise SystemExit(f"Чекпоинты не найдены в {out_dir}")
        resume = str(cks[-1])
        print(f"Продолжаем с {resume}")

    print("\nСтарт обучения...", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    summary = {
        **run_cfg,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.training_loss,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter": str(out_dir / "adapter"),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ГОТОВО ===")
    print(json.dumps({k: summary[k] for k in
                      ("train_loss", "train_runtime_s", "peak_vram_gb", "adapter")}, indent=2))


if __name__ == "__main__":
    main()
