"""Forward+backward throughput of the LoRA decoder on synthetic batches.

Used to choose the fla / batch / gradient-checkpointing config for cfe training.
"""
import argparse
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
p.add_argument("--no-fla", action="store_true")
p.add_argument("--shapes", default="2x1536,4x1536,8x1536,4x6144,8x6144")
a = p.parse_args()

if a.no_fla:
    sys.modules["fla"] = None  # makes `import fla` fail -> transformers uses torch fallback

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

fla = not a.no_fla
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda()
model = get_peft_model(
    model,
    LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
)
decoder = model.base_model.model.model

model.eval()
x = torch.randint(0, 150000, (1, 1536), generator=torch.Generator().manual_seed(0)).cuda()
with torch.no_grad():
    print(f"checksum {decoder(input_ids=x).last_hidden_state.float().abs().mean().item():.6f}", flush=True)

for ckpt in (True, False):
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    for shape in a.shapes.split(","):
        b, l = map(int, shape.split("x"))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = None
        try:
            x = torch.randint(0, 150000, (b, l), device="cuda")
            times = []
            for i in range(5):
                torch.cuda.synchronize()
                t = time.perf_counter()
                decoder(input_ids=x).last_hidden_state.float().pow(2).mean().backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter() - t)
            avg = sum(times) / len(times)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"fla={fla} ckpt={ckpt} {shape} tok_s={b * l / avg:.0f} peak_gb={peak:.2f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"fla={fla} ckpt={ckpt} {shape} OOM", flush=True)
        x = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
