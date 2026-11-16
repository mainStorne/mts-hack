"""Признаки упоминаний-кандидатов для классификаторов (sklearn, разреженные dict-признаки)."""
import re

NEG = {"не", "нет", "никакой", "ни", "никогда", "без"}
CONFIRM = re.compile(r"\b(да|подходит|давайте|хорошо|согласн|ладно|конечно|устраивает|подтверждаю|оформляйте|договорились|верно|угу|отлично)\b")
REJECT = re.compile(r"\b(нет|не надо|не нужно|не хочу|откаж|отказ|потом|не готов|не буду|подумаю|не стоит|обойдусь|не интересно)\b")
OFFER = re.compile(r"(предлож|можем подключить|могу подключить|хотите|акци|со скидкой|бонус|не желаете|выгодн|рекоменд|посоветую|могу предложить)")
FAIL = re.compile(r"(не помог|без толку|толку ноль|толку нет|бесполезн|ничего не измени|результата нет|не помогает|не сработал|всё так же|то же самое|не вышло|ноль внимания|не получилось)")
DATE_RE = re.compile(r"(вчера|позавчера|сегодня|завтра|послезавтра|назад|через|понедельник|вторник|сред[уыа]|четверг|пятниц|суббот|воскресен|недел|месяц|на днях)")


def mention_features(c, d, toks):
    turns = d["turns"]; i = c["turn"]; t = turns[i]
    low = t["text"].lower()
    sp = t["speaker"] or "?"
    a = c["attr"]
    f = {"bias": 1, f"attr={a}": 1, f"val={c['value']}": 1, f"attr_sp={a}_{sp}": 1,
         f"score={round(c['score'], 2)}": 1, "score": c["score"],
         "relpos": i / max(1, len(turns) - 1), "q": int("?" in low), "first3": int(i < 3)}
    lem = [x[0] for x in toks[i]]
    if c["tok0"] is not None:
        a0, a1 = c["tok0"], c["tok1"]
        for w in lem[max(0, a0 - 4):a0]:
            f["L_" + w] = 1
        for w in lem[a1 + 1:a1 + 5]:
            f["R_" + w] = 1
        if any(w in NEG for w in lem[max(0, a0 - 3):a0 + 1]):
            f["neg_before"] = 1
        f[f"neg_before_{a}"] = f.get("neg_before", 0)
    for w in set(lem):
        f["T_" + w] = 1
    for tag, rx in (("offer", OFFER), ("confirm", CONFIRM), ("reject", REJECT), ("fail", FAIL), ("date", DATE_RE)):
        if rx.search(low):
            f[f"{tag}_{a}_{sp}"] = 1
    # соседние реплики
    for off, name in ((1, "N"), (-1, "P"), (2, "N2")):
        j = i + off
        if 0 <= j < len(turns):
            tl = turns[j]["text"].lower(); sj = turns[j]["speaker"] or "?"
            for w in [x[0] for x in toks[j]][:8]:
                f[f"{name}_{w}"] = 1
            for tag, rx in (("confirm", CONFIRM), ("reject", REJECT), ("fail", FAIL), ("offer", OFFER)):
                if rx.search(tl[:120]):
                    f[f"{name}{tag}_{a}_{sj}"] = 1
    return f
