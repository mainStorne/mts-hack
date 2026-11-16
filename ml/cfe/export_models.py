#!/usr/bin/env python3
"""Экспорт артефактов для образов: слитая Gemma, снимок базы GLiNER, адаптер GLiNER, пороги, MANIFEST.json.

python cfe/export_models.py --gemma-base P --gemma-adapter P --gliner-adapter runs/struct-lora \
    --gliner-params runs/struct-lora-v21/params.json --ensemble runs/ens-v3/params.json --out models
"""
import argparse
import hashlib
import json
import os
import shutil

GLINER_BASE = "fastino/gliner2.5-multi-v1"
GLINER_REV = "7b005274e82da37dcf1f5fd9d6a2ef413c6a004c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for n in ("gemma-base", "gemma-adapter", "gliner-adapter", "gliner-params", "ensemble", "out"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = a.out
    os.makedirs(out, exist_ok=True)

    gdir = os.path.join(out, "gemma-merged")
    model = AutoModelForCausalLM.from_pretrained(a.gemma_base, torch_dtype=torch.float16)
    model = peft.PeftModel.from_pretrained(model, a.gemma_adapter).merge_and_unload()
    model.save_pretrained(gdir, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.gemma_adapter).save_pretrained(gdir)
    gemma_params = sum(p.numel() for p in model.parameters())

    from huggingface_hub import snapshot_download
    snapshot_download(GLINER_BASE, revision=GLINER_REV, cache_dir=os.path.join(out, "hf", "hub"))

    best = os.path.join(a.gliner_adapter, "best")
    if not os.path.isdir(best):
        raise SystemExit(f"{best} не найден")
    dst = os.path.join(out, "gliner-lora", "best")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(best, dst)
    for f in os.listdir(dst):  # образы запускаются под произвольным uid: файлы должны читаться всеми
        os.chmod(os.path.join(dst, f), 0o644)
    shutil.copy(a.gliner_params, os.path.join(out, "gliner-params.json"))
    shutil.copy(a.ensemble, os.path.join(out, "ensemble-params.json"))

    sys_path = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(0, sys_path)
    import struct_predict as P
    ext = P.load_model(os.path.join(out, "gliner-lora"), "cpu")
    gliner_params = sum(p.numel() for p in ext.parameters())

    files = {}
    for r, _, fs in os.walk(out):
        if os.sep + "hf" + os.sep + "hub" in r + os.sep and "blobs" in r:
            continue  # дубли снимка: те же файлы доступны через snapshots/
        for f in fs:
            p = os.path.join(r, f)
            if f == "MANIFEST.json" or os.path.islink(p):
                continue
            files[os.path.relpath(p, out)] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    import gliner2
    man = {"gemma_params": gemma_params, "gliner_params": gliner_params,
           "sources": {"gemma_base": a.gemma_base, "gemma_adapter": a.gemma_adapter, "gliner_adapter": a.gliner_adapter,
                       "gliner_params": a.gliner_params, "ensemble": a.ensemble,
                       "gliner_base": f"{GLINER_BASE}@{GLINER_REV}"},
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                        "gliner2": getattr(gliner2, "__version__", "?")},
           "files": files}
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(f"gemma_params {gemma_params} gliner_params {gliner_params} files {len(files)}")


if __name__ == "__main__":
    main()
