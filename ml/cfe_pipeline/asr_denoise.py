#!/usr/bin/env python3
"""Автоматическая очистка ASR-шума в русскоязычных транскриптах (CompactFactExtraction).

Три команды:

  # 1) один раз: собрать модель из корпуса (+ каталог и ручные поправки после проверки)
  python3 asr_denoise.py build --corpus train.jsonl val.jsonl --catalog catalog.json \\
          --overrides review/overrides.json -o asr_model.pkl

  # 2) почистить вход: jsonl в формате звонков ИЛИ обычный текст (строка = реплика)
  python3 asr_denoise.py fix asr_model.pkl test.jsonl -o test_clean.jsonl \\
          --layer test_noise_layer.jsonl --report test_report.tsv
  echo "сийчас заяфку пасмотрю" | python3 asr_denoise.py fix asr_model.pkl - -o -

  # 3) наложить шум обратно (точное восстановление исходника)
  python3 asr_denoise.py restore test_clean.jsonl test_noise_layer.jsonl -o test_noisy.jsonl

Что делает fix:
  * исправляет искажённые слова (аканье/иканье, оглушение, замены/пропуски букв, сильные искажения,
    «ошибки реальным словом», слияния слов, «-ця» → «-тся»), сохраняя регистр;
  * не трогает бренды в CamelCase, латиницу, цифры, дизфлюенции («эээ», «ну-у»), доменные слова и их
    падежные формы, слова из списка ручных отказов;
  * сжимает двойные пробелы (след выпавшего слова);
  * для jsonl пересчитывает facts[].source.span/quote на очищенный текст;
  * пишет слой шума: для каждой реплики список правок [clean_start, clean_end, noisy, clean, rule] —
    по нему restore восстанавливает исходный текст байт в байт.
Пунктуация, регистр начала фразы, выпавшие слова и оборванные концы реплик НЕ восстанавливаются.

Зависимость: pip install pymorphy3 pymorphy3-dicts-ru
"""
import argparse, array, collections, gzip, json, math, pickle, re, sys, zlib

TOK = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")
CYR = re.compile(r"^[а-яё-]+$")
CAMEL = re.compile(r"[а-яё][А-ЯЁ]")
DISFL = re.compile(r"^(?:[аеиоуыэюяё]+|м+|хм+|угу|ага|ну+|ну-у+|а-а+|э-э+|о-о+|у-у+|ой-ой|ох|ах|эх|фух|тьфу|кхм|гм)$|(.)\1\1")
VOW = set("аоеиэяёыюу")
PHP = set()
for g in ["оа", "еиэ", "яи", "ея", "её", "юу", "ыи", "бп", "вф", "гк", "дт", "жш", "зс", "цс", "щш", "чщ"]:
    PHP |= {(a, b) for a in g for b in g if a != b}
KEYMAP = str.maketrans({"о": "а", "и": "е", "я": "е", "э": "е", "ё": "е", "ы": "е", "ю": "у", "б": "п", "в": "ф",
                        "г": "к", "д": "т", "ж": "ш", "з": "с", "ц": "с", "щ": "ш"})
ALTS = {}
for a, b in ["оа", "еи", "яи", "ея", "бп", "вф", "гк", "дт", "жш", "зс"]:
    ALTS.setdefault(a, set()).add(b); ALTS.setdefault(b, set()).add(a)
SLANG = {"инет", "телек", "комп", "капец", "глючит", "вайфай", "симка", "симку", "симки", "тырнет", "мобилка", "планшетик"}
NOREAL = {"лена", "случалось", "проходил", "гавно", "дима", "заглючил", "подвисла"}
NOISE_DOMAIN = {"сплиттир", "бизлимит", "связьсирвис"}             # шумовые варианты, найденные вручную
STEMS = {"сплиттир": "сплиттер", "бизлимит": "безлимит", "связьсирвис": "связьсервис"}
STEM_RE = re.compile(r"(?i)(?<![А-Яа-яЁё])(" + "|".join(STEMS) + r")([а-яё]*)")
RULES = {"oov": "нет в словаре и в лексике корпуса → ближайшее частое слово корпуса с учётом соседних слов",
         "real": "словарное, но редкое слово; близкое частое слово многократно чаще стоит в этом контексте",
         "dict": "нет в корпусе; фонетический вариант (1–2 замены) есть в словаре pymorphy3",
         "split": "слияние двух слов → разделено пробелом",
         "call": "сильное искажение → похожее слово из этого же звонка или частое слово, подтверждённое контекстом",
         "space": "двойной пробел — след выпавшего слова/знака; сжат до одного",
         "manual": "ручная поправка (по итогам проверки корпуса)",
         "stem": "известная шумовая основа, исправлена с сохранением окончания"}


def morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


def key(w): return w.translate(KEYMAP)


def deletes(k, dmax):
    out = {k}; front = {k}
    for _ in range(dmax):
        front = {x[:i] + x[i + 1:] for x in front for i in range(len(x))}
        out |= front
    return out


def sub_cost(a, b):
    if a == b: return 0.0
    if (a, b) in PHP: return 0.3
    if a in VOW and b in VOW: return 0.5
    return 1.0


def wdist(s, t):
    """взвешенное расстояние «шумное → чистое»: фонетические замены дёшевы, пропуск буквы дешевле лишней"""
    n, mm = len(s), len(t)
    if abs(n - mm) > 3: return 99
    D = [[0.0] * (mm + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0] = D[i - 1][0] + 1.0
    for j in range(1, mm + 1): D[0][j] = D[0][j - 1] + (0.5 if j in (1, mm) else 0.7)
    for i in range(1, n + 1):
        for j in range(1, mm + 1):
            v = min(D[i - 1][j] + 1.0, D[i][j - 1] + (0.5 if j in (1, mm) else 0.7),
                    D[i - 1][j - 1] + sub_cost(s[i - 1], t[j - 1]))
            if i > 1 and j > 1 and s[i - 1] == t[j - 2] and s[i - 2] == t[j - 1]: v = min(v, D[i - 2][j - 2] + 0.8)
            D[i][j] = v
    return D[n][mm]


def thr(n): return 0.6 if n <= 3 else 1.0 if n <= 5 else 1.6 if n <= 8 else 2.2


def tsya(w): return [w[:-2] + "тся", w[:-2] + "ться"] if len(w) > 4 and w[-2:] in ("ця", "ца") else []


def match_case(src, dst):
    if src.isupper() and len(src) > 1: return dst.upper()
    if src[:1].isupper(): return dst[:1].upper() + dst[1:]
    return dst


def read_calls(path):
    """jsonl звонков или простой текст (строка = реплика одного «звонка»)"""
    f = sys.stdin if path == "-" else open(path, encoding="utf-8")
    lines = [l.rstrip("\n") for l in f]
    if lines and lines[0].lstrip().startswith("{") and '"turns"' in lines[0]:
        return [json.loads(l) for l in lines if l.strip()], "jsonl"
    return [{"call_id": "text", "turns": [{"idx": i, "speaker": None, "text": l} for i, l in enumerate(lines)]}], "text"


# ------------------------------------------------------------------ build
def build(args):
    m = morph()
    freq = collections.Counter(); big = collections.Counter()
    for p in args.corpus:
        for l in open(p, encoding="utf-8"):
            for t in json.loads(l)["turns"]:
                ws = [w.lower() for w in TOK.findall(t["text"])]
                freq.update(ws); big.update(zip(ws, ws[1:]))
    known = {w: (m.word_is_known(w) or ("-" in w and all(m.word_is_known(x) for x in w.split("-") if x)))
             for w in freq if CYR.match(w)}
    vocab = {w: c for w, c in freq.items() if known.get(w)}
    catwords = set()
    if args.catalog:
        for v in json.load(open(args.catalog, encoding="utf-8")).values():
            for s in v: catwords |= {w.lower() for w in TOK.findall(s)}
    idx = _index(vocab)
    domain = set()
    for w in sorted((w for w in freq if CYR.match(w) and not known.get(w)), key=lambda w: -freq[w]):
        f = freq[w]
        if f < 15 and w not in catwords: continue
        sc = _scored(w, idx) + [(0.3, c) for c in tsya(w) if c in vocab]
        if not [c for dd, c in sc if vocab[c] >= 5 * f and dd <= 1.2]: domain.add(w)
    domain |= {w for w in SLANG | catwords if w in freq and CYR.match(w)}
    domain -= NOISE_DOMAIN
    overrides = json.load(open(args.overrides, encoding="utf-8")) if args.overrides else {}
    # слова с частотой 1 вне словаря — это сами шумовые формы; для скрипта их частота и так считается равной 1
    keep = set(vocab) | domain
    once = array.array("I", sorted({zlib.crc32(w.encode()) for w, c in freq.items() if c == 1 and w not in keep}))
    model = {"freq": {w: c for w, c in freq.items() if c >= 2 or w in keep}, "once_crc": once.tobytes(),
             "big": {k: v for k, v in big.items() if v >= 2}, "domain": sorted(domain),
             "known": sorted(w for w, k in known.items() if k), "overrides": overrides}
    with gzip.open(args.o, "wb", compresslevel=9) as fo:
        pickle.dump(model, fo, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"модель: {len(vocab) + len(domain)} слов словаря, {len(domain)} доменных, "
          f"{len(model['big'])} биграмм, {len(overrides)} ручных поправок → {args.o}", file=sys.stderr)


def _index(V):
    ix = collections.defaultdict(set)
    for w in V:
        k = key(w)
        for x in deletes(k, 2 if len(k) >= 5 else 1): ix[x].add(w)
    return ix


def _scored(w, ix):
    k = key(w); cs = set()
    for x in deletes(k, 2 if len(k) >= 5 else 1): cs |= ix.get(x, set())
    cs.discard(w)
    return [(d, c) for c in sorted(cs) for d in [wdist(w, c)] if d <= thr(len(w))]   # sorted → детерминированно


# ------------------------------------------------------------------ fix
class Denoiser:
    def __init__(self, model_path, adapt_calls=None):
        with open(model_path, "rb") as fi: gz = fi.read(2) == b"\x1f\x8b"
        M = pickle.load(gzip.open(model_path, "rb") if gz else open(model_path, "rb"))   # старые несжатые модели тоже читаются
        self.m = morph()
        self.freq = collections.Counter(M["freq"]); self.big = collections.Counter(M["big"])
        if adapt_calls:      # частоты самого входа: помогают с именами и терминами, словарь не меняют
            once = array.array("I"); once.frombytes(M.get("once_crc", b"")); once = set(once)
            for d in adapt_calls:   # слова корпуса с частотой 1 хранятся хэшами — восстанавливаем их «1»
                for t in d["turns"]:
                    for w in TOK.findall(t["text"]):
                        lw = w.lower()
                        if lw not in self.freq and zlib.crc32(lw.encode()) in once: self.freq[lw] = 1
            for d in adapt_calls:
                for t in d["turns"]:
                    ws = [w.lower() for w in TOK.findall(t["text"])]
                    self.freq.update(ws); self.big.update(zip(ws, ws[1:]))
        self.known = set(M["known"]); self.domain = set(M["domain"]); self.ov = M["overrides"]
        self.vocab = {w: self.freq[w] for w in self.known | self.domain}
        self.idx = _index(self.vocab)
        self.buck = collections.defaultdict(list)
        for w, c in self.vocab.items():
            if c >= 30 and len(w) >= 4: self.buck[w[0]].append(w)
        self.cache = {}; self.dv = {}; self.lex = {}

    def is_known(self, w):
        return w in self.known or self.m.word_is_known(w)

    def domain_form(self, n, c):
        """n — другая падежная форма доменного (неизвестного словарю) слова c → это не шум"""
        if n == c or self.m.word_is_known(c) or c.startswith(n) or c.startswith("связь"): return False
        if "-" in c and len(set(c.split("-"))) == 1: return False
        if c not in self.lex:
            forms = set()
            for p in self.m.parse(c)[:3]:
                if p.normal_form[:3] == c[:3]:
                    try: forms |= {x.word for x in p.lexeme}
                    except Exception: pass
            self.lex[c] = forms
        return n in self.lex[c]

    def is_dom_infl(self, lw):
        if lw in self.vocab: return False
        for suf in ("ом", "ому", "у", "а", "е", "ы", "ам", "ами", "ах", "ой", "ою", "ей", "и", "ов", "ю", "я", "ем"):
            if lw.endswith(suf) and lw[:-len(suf)] in self.domain and len(lw) - len(suf) >= 4: return True
        return False

    def dict_variants(self, lw):
        one = {lw[:i] + b + lw[i + 1:] for i, ch in enumerate(lw) for b in ALTS.get(ch, ())}
        out = {v for v in one if self.m.word_is_known(v)}
        if not out and len(lw) >= 6:
            out = {v[:i] + b + v[i + 1:] for v in one for i, ch in enumerate(v) for b in ALTS.get(ch, ())}
            out = {x for x in out if x != lw and self.m.word_is_known(x)}
        return out

    def token(self, w, prev, nxt, callvocab):
        lw = w.lower(); V = self.vocab; big = self.big; freq = self.freq
        if not CYR.match(lw) or DISFL.match(lw) or CAMEL.search(w) or self.is_dom_infl(lw): return None
        if lw in V:      # «ошибка реальным словом»
            f = freq[lw]
            if f > 40 or len(lw) < 4 or lw in SLANG or lw in self.domain or lw in NOREAL: return None
            lem = {p.normal_form for p in self.m.parse(lw)}; best = None
            if lw not in self.cache: self.cache[lw] = _scored(lw, self.idx)
            for dd, c in self.cache[lw]:
                if dd > 0.6 or V[c] < 100 * max(f, 1) or len(c) < len(lw): continue
                if lem & {p.normal_form for p in self.m.parse(c)}: continue
                cc = big[(prev, c)] + big[(c, nxt)]; cw = big[(prev, lw)] + big[(lw, nxt)]
                if cc >= 3 and cc >= 5 * (cw + 1) and (best is None or (cc, V[c]) > best[0]): best = ((cc, V[c]), c)
            return (best[1], "real") if best else None
        if lw not in self.cache: self.cache[lw] = _scored(lw, self.idx) + [(0.3, c) for c in tsya(lw) if c in V]
        best = None
        for dd, c in self.cache[lw]:
            if V[c] < 3 * max(freq[lw], 1) and c not in callvocab: continue
            ctx = big[(prev, c)] + big[(c, nxt)]
            s = -3 * dd + 0.35 * math.log(V[c] + 1) + 0.5 * math.log(ctx + 1) + (c in callvocab) + 0.4 * (c[-1:] == lw[-1:])
            if best is None or s > best[0]: best = (s, c)
        if best: return best[1], "oov"
        if lw not in self.dv: self.dv[lw] = self.dict_variants(lw) if len(lw) >= 8 else set()
        if self.dv[lw]:
            c = max(sorted(self.dv[lw]), key=lambda c: (-wdist(lw, c), big[(prev, c)] + big[(c, nxt)], c[-1:] == lw[-1:]))
            return c, "dict"
        bs = None
        for i in range(2, len(lw) - 1):
            l, r = lw[:i], lw[i:]
            if l in V and r in V and V[l] >= 30 and V[r] >= 30 and big[(l, r)] >= 20 and (bs is None or big[(l, r)] > bs[0]):
                bs = (big[(l, r)], l + " " + r)
        if bs: return bs[1], "split"
        if len(lw) < 5: return None
        bc = None
        for c in sorted(set(self.buck.get(lw[0], ())) | {c for c in callvocab if c[:1] == lw[0]}):
            if abs(len(c) - len(lw)) > 1: continue
            dd = wdist(lw, c)
            if dd > 0.38 * len(lw): continue
            ctx = big[(prev, c)] + big[(c, nxt)]; inc = c in callvocab
            if not inc and ctx < 2: continue
            s = -dd + 0.6 * math.log(ctx + 1) + 1.5 * inc + 0.15 * math.log(V.get(c, 1) + 1) + 0.5 * (c[-1:] == lw[-1:]) - 0.3 * abs(len(c) - len(lw))
            if bc is None or s > bc[0]: bc = (s, c, inc, ctx)
        return (bc[1], "call") if bc and (bc[2] or bc[3] >= 5) else None

    def call(self, d):
        """→ список правок по репликам: (noisy_start, noisy_end, noisy, clean, rule)"""
        cv = {w.lower() for t in d["turns"] for w in TOK.findall(t["text"]) if w.lower() in self.vocab}
        for f in d.get("facts") or []:
            for s in ((f.get("object") or {}).get("name") or "", f.get("value") or ""):
                cv |= {w.lower() for w in TOK.findall(s) if CYR.match(w.lower()) and (w.lower() in self.vocab or self.is_known(w.lower()))}
        res = []
        for t in d["turns"]:
            text = t["text"]; toks = list(TOK.finditer(text)); eds = []
            for i, mt in enumerate(toks):
                w = mt.group(); prev = toks[i - 1].group().lower() if i else "<s>"
                nxt = toks[i + 1].group().lower() if i + 1 < len(toks) else "</s>"
                r = self.token(w, prev, nxt, cv)
                if r is None and "-" in w and w.lower() not in self.vocab:
                    parts = [(p, self.token(p, prev, nxt, cv) if p else None) for p in w.split("-")]
                    if any(rr and rr[1] != "split" for _, rr in parts):
                        r = ("-".join(match_case(p, rr[0]) if rr and rr[1] != "split" else p for p, rr in parts), "oov")
                if not r: continue
                new, rule = match_case(w, r[0]), r[1]
                pk = f"{w.lower()}|{r[0].lower()}"
                if pk in self.ov:                                  # ручная поправка
                    if self.ov[pk] is None: continue
                    new, rule = match_case(w, self.ov[pk]), "manual"
                elif self.domain_form(w.lower(), r[0].lower()): continue
                if new != w: eds.append((mt.start(), mt.end(), w, new, rule))
            for mt in STEM_RE.finditer(text):                    # шумовые основы
                new = match_case(mt.group(), STEMS[mt.group(1).lower()] + mt.group(2).lower())
                if CAMEL.search(mt.group()): new = new.replace("Связьсервис", "СвязьСервис")
                eds = [e for e in eds if not (e[0] < mt.end() and mt.start() < e[1])]
                if new != mt.group(): eds.append((mt.start(), mt.end(), mt.group(), new, "stem"))
            for mt in re.finditer(r"(?<=\S)  +(?=\S)", text): eds.append((mt.start(), mt.end(), mt.group(), " ", "space"))
            res.append(sorted(eds))
        return res


def apply_edits(text, eds):
    parts, pos, cl, layer = [], 0, 0, []
    for ns, ne, nz, cn, rule in eds:
        parts.append(text[pos:ns]); cl += ns - pos
        layer.append([cl, cl + len(cn), nz, cn, rule]); parts.append(cn); cl += len(cn); pos = ne
    parts.append(text[pos:])
    return "".join(parts), layer


def map_off(p, eds, side):
    delta = 0
    for ns, ne, nz, cn, _ in eds:
        if ne <= p: delta += len(cn) - (ne - ns); continue
        if ns < p < ne: return ns + delta if side == "start" else ns + delta + len(cn)
        break
    return p + delta


def fix(args):
    calls, kind = read_calls(args.input)
    dn = Denoiser(args.model, calls if args.adapt else None)
    out = sys.stdout if args.o == "-" else open(args.o, "w", encoding="utf-8")
    lay = open(args.layer, "w", encoding="utf-8") if args.layer else None
    rep = open(args.report, "w", encoding="utf-8") if args.report else None
    if rep: rep.write("call_id\tturn_idx\tтекст с шумом\tочищенный текст\tчто исправлено\tпочему\n")
    total = collections.Counter()
    for n, d in enumerate(calls):
        eds_all = dn.call(d); layer = []
        for t, eds in zip(d["turns"], eds_all):
            noisy = t["text"]; clean, L = apply_edits(noisy, eds); t["text"] = clean
            layer.append({"turn_idx": t["idx"], "edits": L}); total.update(e[4] for e in eds)
            if rep and eds:
                rep.write("\t".join([str(d.get("call_id")), str(t["idx"]), noisy, clean,
                                     "; ".join(f"{e[2]} → {e[3]}" for e in eds),
                                     "; ".join(f"«{e[2]}»: {RULES[e[4]]}" for e in eds)]) + "\n")
        for f in d.get("facts") or []:                    # спаны фактов → в координаты очищенного текста
            src = f.get("source") or {}; sp = src.get("span"); ti = src.get("turn_idx")
            if sp and ti is not None and 0 <= ti < len(eds_all):
                a, b = map_off(sp[0], eds_all[ti], "start"), map_off(sp[1], eds_all[ti], "end")
                src["span"] = [a, b]; src["quote"] = d["turns"][ti]["text"][a:b]
        if kind == "jsonl": out.write(json.dumps(d, ensure_ascii=False) + "\n")
        else: out.write("\n".join(t["text"] for t in d["turns"]) + "\n")
        if lay: lay.write(json.dumps({"call_id": d.get("call_id"), "turns": layer}, ensure_ascii=False) + "\n")
        if (n + 1) % 200 == 0: print(f"{n + 1}/{len(calls)}", file=sys.stderr)
    print(f"готово: {len(calls)} звонков, правок {sum(total.values())}: {dict(total)}", file=sys.stderr)


# ------------------------------------------------------------------ restore
def restore(args):
    layers = {}
    for l in open(args.layer, encoding="utf-8"):
        d = json.loads(l); layers[d["call_id"]] = {t["turn_idx"]: t["edits"] for t in d["turns"]}
    calls, kind = read_calls(args.clean)
    if kind == "text": calls[0]["call_id"] = next(iter(layers))
    out = sys.stdout if args.o == "-" else open(args.o, "w", encoding="utf-8")
    for d in calls:
        L = layers.get(d["call_id"], {})
        for t in d["turns"]:
            s = t["text"]
            for cs, ce, nz, *_ in sorted(L.get(t["idx"], []), reverse=True): s = s[:cs] + nz + s[ce:]
            t["text"] = s
        out.write(json.dumps(d, ensure_ascii=False) + "\n" if kind == "jsonl" else "\n".join(t["text"] for t in d["turns"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--corpus", nargs="+", required=True); b.add_argument("--catalog")
    b.add_argument("--overrides"); b.add_argument("-o", default="asr_model.pkl")
    f = sp.add_parser("fix"); f.add_argument("model"); f.add_argument("input"); f.add_argument("-o", default="-")
    f.add_argument("--layer", help="куда записать слой шума (для restore)"); f.add_argument("--report", help="TSV: что/почему исправлено")
    f.add_argument("--no-adapt", dest="adapt", action="store_false", help="не учитывать частоты самого входа")
    r = sp.add_parser("restore"); r.add_argument("clean"); r.add_argument("layer"); r.add_argument("-o", default="-")
    a = ap.parse_args()
    {"build": build, "fix": fix, "restore": restore}[a.cmd](a)
