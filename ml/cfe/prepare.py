import argparse
import copy
import json
import os
import random
import zlib

from codec import encode_target, messages

LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def _noise_word(w: str, rng: random.Random) -> str | None:
    op = rng.choice(["del_char", "swap", "sub", "del_word"])
    if op == "del_word":
        return None
    if len(w) < 3:
        return w
    i = rng.randrange(len(w) - 1)
    if op == "del_char":
        return w[:i] + w[i + 1:]
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + rng.choice(LETTERS) + w[i + 1:]


def asr_noise(text: str, rng: random.Random) -> str:
    words = text.split()
    out = []
    for w in words:
        if rng.random() < 0.15:
            w = _noise_word(w, rng)
            if w is None:
                continue
        out.append(w)
    if rng.random() < 0.1 and out:
        out = out[: max(1, int(len(out) * 0.7))]
        out.append("—")
    return " ".join(out)


def noisy_call(call: dict) -> dict:
    c = copy.deepcopy(call)
    rng = random.Random(zlib.crc32(call["call_id"].encode()))
    for t in c["turns"]:
        t["text"] = asr_noise(t["text"], rng)
    return c


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    train = read_jsonl(f"{a.raw}/train.jsonl")
    val = read_jsonl(f"{a.raw}/val.jsonl")

    rows, aug, skipped = [], 0, 0

    def add(call, is_aug):
        nonlocal aug, skipped
        prompt = messages(call)
        comp = [{"role": "assistant", "content": encode_target(call)}]
        ids = tok.apply_chat_template(prompt + comp, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) > a.max_tokens:
            skipped += 1
            return
        rows.append({"prompt": prompt, "completion": comp})
        aug += is_aug

    for i, c in enumerate(train):
        add(c, 0)
        if i % 3 == 0:
            add(noisy_call(c), 1)
    random.Random(0).shuffle(rows)

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/cfe_train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.out}/val_adv.jsonl", "w", encoding="utf-8") as f:
        for c in val:
            f.write(json.dumps(noisy_call(c), ensure_ascii=False) + "\n")
    print(f"train_rows {len(rows)}")
    print(f"augmented_rows {aug}")
    print(f"skipped_too_long {skipped}")
    print(f"val_adv_rows {len(val)}")


if __name__ == "__main__":
    main()
