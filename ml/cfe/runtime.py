"""Единый рантайм CFE: используется API (cfe/api.py) и замером (cfe/bench.py), чтобы исполнялся один и тот же код.

Окружение: CFE_MODEL, CFE_PARAMS, CFE_TAU, CFE_SNAP, CFE_DENOISE, CFE_SCHEMA, CFE_ENSEMBLE, CFE_GEMMA_MODEL,
CFE_GEMMA_ADAPTER (необязательно: пусто -> CFE_GEMMA_MODEL уже слитая модель), CFE_GEMMA_BATCH (16),
CFE_DEVICE (cuda|cpu), CFE_GEMMA_BACKEND (hf|vllm), CFE_VLLM_URL, CFE_GEMMA_MAX_PROMPT (3072),
CFE_GEMMA_MAX_NEW (1024), CFE_THREADS.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clf_task as T  # noqa: E402
import rules  # noqa: E402
import denoise as D  # noqa: E402
import struct_predict as P  # noqa: E402
import struct_schema as S  # noqa: E402


def fix_intervals(facts):
    """valid_to раньше valid_from нарушает схему (§2): закрываем звено днём valid_from (однодневное звено допустимо). Метрики кейса не меняются."""
    for f in facts:
        vt, vf = f.get("valid_to"), f.get("valid_from")
        if vt and vf and vt < vf:
            f["valid_to"] = vf
    return facts


class Runtime:
    def __init__(self, mode=None):
        import torch
        self.torch = torch
        env = os.environ.get
        self.model_path = env("CFE_MODEL", os.path.join(T.ROOT, "runs", "struct-lora"))
        self.denoise = env("CFE_DENOISE", "0") == "1"
        kind = env("CFE_SCHEMA", "entities")
        self.ens = json.load(open(env("CFE_ENSEMBLE"))) if env("CFE_ENSEMBLE") else None
        if self.ens and self.denoise:
            raise RuntimeError("CFE_ENSEMBLE несовместим с CFE_DENOISE")
        if env("CFE_PARAMS"):
            p = json.load(open(env("CFE_PARAMS")))
            self.tau, self.snap = p["tau_attr"], p["snap_th"]
        else:
            self.tau, self.snap = float(env("CFE_TAU", "0.8")), float(env("CFE_SNAP", "0.5"))
        self.device = env("CFE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = env("CFE_GEMMA_BACKEND", "hf")
        self.gemma_batch = int(env("CFE_GEMMA_BATCH", "16"))
        self.max_prompt = int(env("CFE_GEMMA_MAX_PROMPT", "3072"))
        self.max_new = int(env("CFE_GEMMA_MAX_NEW", "1024"))
        if env("CFE_THREADS"):
            torch.set_num_threads(int(env("CFE_THREADS")))
        self.mode = mode or ("ensemble" if self.ens else "gliner")
        if self.mode in ("ensemble", "gemma") and not env("CFE_GEMMA_MODEL"):
            raise RuntimeError("CFE_GEMMA_MODEL не задан")
        if self.mode == "ensemble" and not self.ens:
            raise RuntimeError("режим ensemble требует CFE_ENSEMBLE")
        self.ext = self.schema = self.gemma = self.ensemble = None
        if self.mode in ("ensemble", "gliner"):
            self.ext = P.load_model(self.model_path, self.device)
            self.schema = S.build_schema(self.ext, kind)
        self.vl = T.value_labels(T.load_catalog())
        self.dm = rules.load_date_model()
        if self.mode in ("ensemble", "gemma"):
            from gemma_gen import GemmaExtractor, VllmGemma
            if self.backend == "vllm":
                self.gemma = VllmGemma(env("CFE_GEMMA_MODEL"), env("CFE_VLLM_URL", "http://127.0.0.1:8001"),
                                       self.max_prompt, self.max_new)
            else:
                self.gemma = GemmaExtractor(env("CFE_GEMMA_MODEL"), env("CFE_GEMMA_ADAPTER") or None,
                                            self.max_prompt, self.max_new, self.device)
        if self.mode == "ensemble":
            import ensemble
            self.ensemble = ensemble
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"n_calls": 0, "gliner_s": 0.0, "gemma_s": 0.0, "post_s": 0.0, "gen_tokens": 0}
        if self.gemma is not None:
            self.gemma.gen_tokens = 0

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def process(self, calls, taus=None):
        taus = taus or [None] * len(calls)
        t_all = time.perf_counter()
        gl = gm = 0.0
        ccs, recs, maps_l = [], [], []
        for call in calls:
            if self.ext is None:
                ccs.append(call)
                continue
            if self.denoise:
                cc, maps = D.denoise_call(call)
            else:
                cc, maps = call, None
            ctxs = [T.render_context(cc, t) for t in range(len(cc["turns"]))]
            offs = [S.turn_offset(c, cc, t) for t, c in enumerate(ctxs)]
            self._sync()
            t0 = time.perf_counter()
            raws = self.ext.batch_extract(ctxs, self.schema, batch_size=16, threshold=0.1,
                                          include_confidence=True, include_spans=True)
            self._sync()
            gl += time.perf_counter() - t0
            ccs.append(cc)
            recs.append({"offs": offs, "raws": raws})
            maps_l.append(maps)
        out = None
        if self.gemma is not None:
            self._sync()
            t0 = time.perf_counter()
            gf = self.gemma.extract_batch(ccs)
            self._sync()
            gm = time.perf_counter() - t0
            if self.mode == "gemma":
                out = gf
            else:
                out = [self.ensemble.ensemble_call(cc, rec, g, self.ens) for cc, rec, g in zip(ccs, recs, gf)]
        else:
            out = []
            for call, cc, rec, maps, tau in zip(calls, ccs, recs, maps_l, taus):
                facts = S.call_facts(cc, rec["raws"], rec["offs"], self.vl, self.dm,
                                     self.tau if tau is None else tau, self.snap, None)
                if self.denoise:
                    D.restore_sources(facts, call, maps)
                out.append(facts)
        out = [fix_intervals(f) for f in out]
        total = time.perf_counter() - t_all
        s = self.stats
        s["n_calls"] += len(calls)
        s["gliner_s"] += gl
        s["gemma_s"] += gm
        s["post_s"] += total - gl - gm
        s["gen_tokens"] = self.gemma.gen_tokens if self.gemma is not None else 0
        return out

    def param_count(self):
        n = 0
        if self.ext is not None:
            n += sum(p.numel() for p in self.ext.parameters())
        if self.gemma is not None:
            if self.backend == "vllm":
                mf = os.path.join(os.path.dirname(os.environ["CFE_GEMMA_MODEL"].rstrip("/")), "MANIFEST.json")
                n += json.load(open(mf))["gemma_params"]
            else:
                n += sum(p.numel() for p in self.gemma.model.parameters())
        return n
