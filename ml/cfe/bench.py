#!/usr/bin/env python3
"""Единый замер ресурсов и качества. Конфигурация моделей — переменные окружения API (см. cfe/runtime.py).

python cfe/bench.py --calls PATH --out DIR [--mode ensemble|gliner|gemma] [--batch N] [--limit N] [--allow-shared]

Методика: монопольный GPU (иначе код 2); прогрев на первых двух звонках исключён; звонки пачками по --batch в порядке файла
через Runtime.process (тот же код, что в API); GPU-секунды — время между cuda.synchronize() вокруг инференса GLiNER и
генерации Gemma, делённое на число звонков; для бэкенда vLLM — стеночное время параллельных запросов пачки;
загрузка моделей в замер не входит. peak_vram_smi_gb — максимум nvidia-smi memory.used за прогон (единственный показатель
памяти для vLLM, включает память всего GPU, в том числе резерв vLLM).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_schema as V  # noqa: E402


def smi(q):
    try:
        return subprocess.run(["nvidia-smi", f"--query-{q[0]}={q[1]}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def foreign_pids():
    own = {os.getpid(), os.getppid()}
    try:
        own |= {int(x) for x in subprocess.run(["pgrep", "-f", "vllm"], capture_output=True, text=True).stdout.split()}
    except Exception:  # noqa: BLE001
        pass
    return [p for p in smi(("compute-apps", "pid")).split() if p.isdigit() and int(p) not in own]


def dir_gb(path):
    if not path or not os.path.exists(path):
        return 0.0
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            fp = os.path.join(r, f)
            if not os.path.islink(fp):
                tot += os.path.getsize(fp)
    return tot / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ensemble", "gliner", "gemma"])
    ap.add_argument("--batch", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--allow-shared", action="store_true")
    a = ap.parse_args()
    import torch
    from runtime import Runtime
    smi_base = 0.0
    cuda = (os.environ.get("CFE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"
    if cuda and not a.allow_shared:
        fp = foreign_pids()
        if fp:
            print("GPU занят: " + " ".join(fp), file=sys.stderr)
            sys.exit(2)
    calls = [json.loads(l) for l in open(a.calls, encoding="utf-8") if l.strip()]
    if a.limit:
        calls = calls[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    v = smi(("gpu", "memory.used")).split("\n")[0]
    smi_base = float(v) if v.replace(".", "").isdigit() else 0.0
    if os.environ.get("BENCH_SMI_BASE_MB"):  # entrypoint GPU-образа замерил память до запуска vLLM
        smi_base = float(os.environ["BENCH_SMI_BASE_MB"])  # память GPU до загрузки (чужие процессы/резидентные сервисы)
    rt = Runtime(a.mode)
    batch = a.batch or rt.gemma_batch

    def prep(c):  # как в API: реплики индексируются позицией, исходные idx возвращаются в source.turn_idx
        orig = [t.get("idx", i) if t.get("idx") is not None else i for i, t in enumerate(c["turns"])]
        return {"call_id": c["call_id"], "started_at": c["started_at"],
                "turns": [{"idx": i, "speaker": t["speaker"], "text": t["text"]} for i, t in enumerate(c["turns"])]}, orig

    rt.process([prep(c)[0] for c in calls[:2]])  # прогрев
    peak = [0.0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            v = smi(("gpu", "memory.used")).split("\n")[0]
            if v.replace(".", "").isdigit():
                peak[0] = max(peak[0], float(v))
            stop.wait(0.5)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        threading.Thread(target=poll, daemon=True).start()
    rt.reset_stats()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    preds, lat = [], []
    for i in range(0, len(calls), batch):
        chunk = [prep(c) for c in calls[i:i + batch]]
        tb = time.perf_counter()
        res = rt.process([c for c, _ in chunk])
        lat.append(time.perf_counter() - tb)
        for (c, orig), facts in zip(chunk, res):
            for f in facts:
                f["source"]["turn_idx"] = orig[f["source"]["turn_idx"]]
            preds.append({"call_id": c["call_id"], "facts": facts})
        print(f"{len(preds)}/{len(calls)}", file=sys.stderr, flush=True)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    stop.set()
    pred_path = os.path.join(a.out, "pred.jsonl")
    with open(pred_path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    metrics = None
    if "facts" in calls[0]:
        gold = os.path.join(a.out, "gold.tmp.jsonl")
        with open(gold, "w", encoding="utf-8") as fh:
            for c in calls:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        r = subprocess.run([sys.executable, os.path.join(HERE, "case_metrics.py"), gold, pred_path], capture_output=True, text=True)
        os.remove(gold)
        metrics = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        if "fact_precision" in metrics and "fact_recall" in metrics:
            p, rr = metrics["fact_precision"], metrics["fact_recall"]
            metrics["f1"] = round(2 * p * rr / (p + rr), 4) if p + rr else 0.0
        json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    sch = V.stats(calls, preds)
    n, s = len(calls), rt.stats
    env = os.environ.get
    weights = [env("CFE_MODEL") if rt.mode != "gemma" else None, env("CFE_GEMMA_MODEL") if rt.mode != "gliner" else None]
    # база GLiNER: каталог снимка (CFE_GLINER_BASE) либо запись в HF-кэше; весь кэш (общий на сервере) не считаем
    gb = env("CFE_GLINER_BASE", "")
    hub = os.path.join(env("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--fastino--gliner2.5-multi-v1")
    base_dir = os.path.dirname(os.path.dirname(gb)) if os.sep + "snapshots" + os.sep in gb else hub
    if rt.mode == "gemma":
        base_dir = None
    cpu_name = ""
    try:
        cpu_name = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except OSError:
        pass
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        h.update(open(f, "rb").read())
    res = {
        "n_calls": n, "mode": rt.mode, "device": rt.device, "gemma_backend": rt.backend, "batch": batch,
        "max_prompt_tokens": rt.max_prompt, "max_new_tokens": rt.max_new, "params": rt.param_count(),
        "weights_disk_gb": round(sum(dir_gb(w) for w in weights if w) + dir_gb(base_dir), 3),
        "wall_sec_per_call": wall / n,
        "batch_latency_p50_s": sorted(lat)[len(lat) // 2], "batch_latency_p95_s": sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.95))],
        "batch_latency_max_s": max(lat),
        "gpu_sec_per_call": (s["gliner_s"] + s["gemma_s"]) / n if cuda else None,
        "gpu_sec_gliner_per_call": s["gliner_s"] / n if cuda else None,
        "gpu_sec_gemma_per_call": s["gemma_s"] / n if cuda else None,
        "cpu_sec_per_call": cpu / n, "gen_tokens_per_call": s["gen_tokens"] / n,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0,
        "peak_vram_smi_gb": (peak[0] - smi_base) / 1024 if cuda else None,  # максимум nvidia-smi за прогон минус память до загрузки моделей
        "peak_vram_smi_abs_gb": peak[0] / 1024 if cuda else None, "smi_base_gb": smi_base / 1024 if cuda else None,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "invalid_calls": sch["invalid_calls"], "schema_violations": sch["violations"],
        "hardware": {"gpu": smi(("gpu", "name")).split("\n")[0] if cuda else None, "cpu": cpu_name or platform.processor(),
                     "torch_threads": torch.get_num_threads()},
        "code_sha256": h.hexdigest(),
    }
    json.dump(res, open(os.path.join(a.out, "resources.json"), "w"), indent=1)
    if sch["list"]:
        open(os.path.join(a.out, "schema_violations.txt"), "w").write("\n".join(sch["list"]) + "\n")
    print(json.dumps({"metrics": metrics, **{k: res[k] for k in ("gpu_sec_per_call", "wall_sec_per_call", "peak_vram_smi_gb", "schema_violations")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
