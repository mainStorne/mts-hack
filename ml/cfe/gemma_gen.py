"""Gemma 3 1B + LoRA как генеративный экстрактор фактов внутри процесса API (промпт и декодер — из ml/gemma)."""
import importlib.util
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GEMMA_DIR = os.path.join(os.path.dirname(HERE), "gemma")


def _load_gemma_modules():
    """gemma/*.py импортируют свой `codec`, одноимённый с cfe/codec.py: на время импорта подменяем его в sys.modules."""
    saved = {k: sys.modules.pop(k) for k in ("codec",) if k in sys.modules}
    added = GEMMA_DIR not in sys.path
    sys.path.insert(0, GEMMA_DIR)
    try:
        mods = {}
        for name in ("codec", "prepare_dataset", "predict"):
            spec = importlib.util.spec_from_file_location("gemma_" + name, os.path.join(GEMMA_DIR, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            if name == "codec":
                sys.modules["codec"] = mod
            spec.loader.exec_module(mod)
            mods[name] = mod
        return mods
    finally:
        sys.modules.pop("codec", None)
        sys.modules.update(saved)
        if added:
            sys.path.remove(GEMMA_DIR)


class _GemmaBase:
    """Общая часть бэкендов: промпты и разбор ответов."""

    def _init_common(self, tokenizer_path, max_prompt_tokens, max_new_tokens):
        from transformers import AutoTokenizer
        self.g = _load_gemma_modules()
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.max_prompt, self.max_new = max_prompt_tokens, max_new_tokens
        self.overhead = self.g["prepare_dataset"].template_overhead(self.tok)
        catalog = json.load(open(os.path.join(os.path.dirname(GEMMA_DIR), "data", "raw", "catalog.json"), encoding="utf-8"))
        self.devices = set(catalog["devices"])
        self.gen_tokens = 0

    def _prompts(self, calls):
        codec, pd = self.g["codec"], self.g["prepare_dataset"]
        prompts = []
        for call in calls:
            text = pd.truncate_to_tokens(codec.render_user(call), self.tok, self.max_prompt, self.overhead)
            prompts.append(self.tok.apply_chat_template(
                [{"role": "system", "content": codec.SYSTEM}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True))
        return prompts

    def _facts(self, calls, raws):
        pr = self.g["predict"]
        out = []
        for call, raw in zip(calls, raws):
            facts = []
            for o in pr.parse_facts_array(raw) or []:
                f = pr.to_fact(o, call, self.devices)
                if f is not None:
                    facts.append(f)
            out.append(facts)
        return out


class GemmaExtractor(_GemmaBase):
    def __init__(self, model_path, adapter=None, max_prompt_tokens=3072, max_new_tokens=1024, device="cuda"):
        from transformers import AutoModelForCausalLM
        self._init_common(adapter or model_path, max_prompt_tokens, max_new_tokens)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval().to(device)

    def extract_batch(self, calls):
        """calls: [{call_id, started_at, turns:[{idx, speaker, text}]}] -> список фактов (официальная схема) на каждый звонок."""
        if not calls:
            return []
        prompts = self._prompts(calls)
        kw = dict(max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        if len(prompts) == 1:
            kw["prompt_lookup_num_tokens"] = 8  # assisted-генерация: только батч 1
        enc = self.tok(prompts, return_tensors="pt", padding=len(prompts) > 1, add_special_tokens=False).to(self.model.device)
        try:
            with torch.inference_mode():
                gen = self.model.generate(**enc, **kw)
        except torch.cuda.OutOfMemoryError:
            if len(calls) == 1:
                raise
            del enc
            torch.cuda.empty_cache()
            h = len(calls) // 2
            return self.extract_batch(calls[:h]) + self.extract_batch(calls[h:])
        new = gen[:, enc["input_ids"].shape[1]:]
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        self.gen_tokens += int(sum(1 for t in new.flatten().tolist() if t not in eos))
        raws = self.tok.batch_decode(new, skip_special_tokens=True)
        return self._facts(calls, raws)


class VllmGemma(_GemmaBase):
    """Gemma на внешнем сервере vLLM (OpenAI-совместимый /v1/completions); в процессе только токенайзер."""

    def __init__(self, model_path, url="http://127.0.0.1:8001", max_prompt_tokens=3072, max_new_tokens=1024):
        self._init_common(model_path, max_prompt_tokens, max_new_tokens)
        self.url = url.rstrip("/")

    def _one(self, prompt):
        body = json.dumps({"model": "gemma", "prompt": prompt, "max_tokens": self.max_new, "temperature": 0,
                           "add_special_tokens": False, "skip_special_tokens": True}).encode()
        req = urllib.request.Request(self.url + "/v1/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        return d["choices"][0]["text"], d["usage"]["completion_tokens"]

    def extract_batch(self, calls):
        if not calls:
            return []
        prompts = self._prompts(calls)
        with ThreadPoolExecutor(max_workers=len(calls)) as ex:
            res = list(ex.map(self._one, prompts))
        self.gen_tokens += sum(n for _, n in res)
        return self._facts(calls, [t for t, _ in res])
