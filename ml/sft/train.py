"""LoRA SFT on a JSONL prompt/completion dataset (TRL conversational format)."""

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--train", required=True)
    p.add_argument("--eval")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--group-by-length", action="store_true")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    train_ds = load_dataset("json", data_files=a.train)["train"]
    eval_ds = load_dataset("json", data_files=a.eval)["train"] if a.eval else None

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, attn_implementation="sdpa")

    lora = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_len,
        fp16=True,
        bf16=False,
        gradient_checkpointing=not a.no_grad_ckpt,
        train_sampling_strategy="group_by_length" if a.group_by_length else "random",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(f"{a.out}/adapter")
    tokenizer.save_pretrained(f"{a.out}/adapter")

    summary = {
        "model": a.model,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "train_runtime_s": result.metrics["train_runtime"],
        "final_train_loss": result.training_loss,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    with open(f"{a.out}/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
