"""Семейства сценариев: структурированный сценарий -> диалог + эталонные факты.

Каждая семья — функция (b: Builder, m: dict), которая строит реплики и
регистрирует факты строго синхронно: факт появляется только тогда, когда в
диалоге есть его носитель-цитата. Предложения оператора без подтверждения
клиента, отвергнутые гипотезы и вопросы фактами НЕ становятся.

Ключевые конвенции gold (проверены по train.jsonl):
  * action/agreement/tariff -> объект client; service_state -> service/device;
    ticket_status -> ticket;
  * polarity -1 — только явное «больше не используется» от клиента;
  * single-valued цепочки (service_state/tariff/ticket_status) закрываются:
    valid_to предыдущего значения = valid_from следующего;
  * agreement с будущей датой, озвученной в цитате: valid_from = дата события.
"""
from __future__ import annotations

from . import catalog as cat
from .catalog import hour_phrase
from .dates import NUM_WORDS

# ------------------------------------------------------------------ пулы ----
GREET_O = [
    "Здравствуйте, компания «СвязьСервис», меня зовут Ольга. Чем могу помочь?",
    "Добрый день, техническая поддержка, слушаю вас.",
    "Здравствуйте! Поддержка, что случилось?",
    "Алло, служба поддержки, здравствуйте.",
    "МТС, добрый день. Чем помогаем?",
    "Здравствуйте, оператор на линии.",
]
GREET_C = [
    "Здравствуйте, это {name}. У меня {svc} не работает.",
    "Здравствуйте! Звоню впервые, зовут {name}. Не работает {svc}.",
    "Добрый день, {name} беспокоит. Опять не работает {svc}.",
    "Здравствуйте. Меня зовут {name}, звоню по поводу {svc_g}.",
    "Добрый день! {name} я. Совсем никак не фурычит {svc}.",
]
GREET_C_NOGREET = [
    "Слушайте, у меня уже неделю {svc} глючит, я не знаю, что делать.",
    "Так, у меня проблема: {svc} опять барахлит.",
    "Ой, ну наконец-то дозвонилась. Уже неделю не работает {svc}.",
    "Здравствуйте. У меня {svc} — вообще ничего не открывается.",
    "Добрый день. Простите, что отнимаю время: {svc} сломался.",
]
FILL_O = [
    "Так, минуточку, посмотрю вашу карточку…",
    "Понимаю вас. Сейчас гляну, что у вас на линии.",
    "Так-так, вижу обращение. Одну секунду.",
    "Понял. Посмотрю историю по вашему аккаунту.",
    "Да, я вас слушаю внимательно.",
    "Так, карточка открыта. Продолжайте.",
]
FILL_C = [
    "Да уж, сил уже нет с этим.",
    "Вот именно, представляете, как это бесит.",
    "Ну да, и что теперь делать, ума не приложу.",
    "Вот-вот, я об этом и говорю.",
    "Извините, что повторяюсь, просто уже замучилась.",
]
OP_ACK = ["Понял вас.", "Ясно.", "Хорошо, зафиксировал.", "Понимаю.", "Так, вижу."]
CLOSE_O = [
    "Хорошо, тогда всё оформил. Спасибо за звонок, всего доброго!",
    "Зафиксировал обращение. Если что — звоните, поможем.",
    "Ок, вопрос принят. Всего вам доброго!",
    "Спасибо за обращение, доброго дня!",
    "Всё записал, ожидайте, как договорились. До свидания!",
]
CLOSE_C = ["Спасибо, всего доброго.", "Хорошо, спасибо большое!", "Ладно, жду. Спасибо!",
           "Понятно, спасибо. До свидания.", "Угу, спасибо."]


def _c(rng, pool, **kw):
    s = rng.choice(pool)
    return s.format(**kw) if kw else s


# ------------------------------------------------------------ служебные блоки
GREET_C_PLAIN = [
    "Здравствуйте, это {name}.",
    "Добрый день, {name} беспокоит.",
    "Здравствуйте! {name} я.",
    "Добрый день. Меня зовут {name}.",
]
# тема звонка без симптома (симптом будет озвучен позже, с фактом)
GREET_C_TOPIC = [
    "Здравствуйте. Меня зовут {name}, звоню по поводу {svc_g}.",
    "Добрый день, {name} беспокоит. Звоню по поводу {svc_g}.",
    "Здравствуйте, это {name}. Дело как раз касается {svc_g}.",
]
# симптом без указания даты (состояние продолжается на момент звонка)
GREET_C_UNDATED = [
    "Здравствуйте, это {name}. У меня {svc} не работает.",
    "Добрый день, {name} беспокоит. Опять не работает {svc}.",
    "Добрый день! {name} я. Совсем никак не фурычит {svc}.",
    "Здравствуйте. Меня зовут {name}. С {svc_g} что-то не так.",
]


def opening(b, m, svc=None, no_svc=False, greeting: str = "default"):
    """greeting: default — общий пул (симптом/тема), topic — тема без симптома,
    undated — симптом без даты (состояние продолжается, vf = день звонка)."""
    r = b.rng
    m["svc_main"] = svc if svc is not None else b.service()
    prof = b.profile(m["svc_main"])
    m["symptom"] = r.choice(prof["symptoms"])
    pool = {"default": GREET_C, "topic": GREET_C_TOPIC, "undated": GREET_C_UNDATED}[greeting]
    if r.random() < 0.78:
        b.turn("operator", _c(r, GREET_O))
        if no_svc:
            b.turn("client", _c(r, GREET_C_PLAIN, name=b.client_obj.name))
        else:
            b.turn("client", _c(r, pool, name=b.client_obj.name, svc=m["svc_main"].name,
                                svc_g=cat.gen(m["svc_main"].name)))
    else:
        if no_svc:
            b.turn("client", _c(r, GREET_C_PLAIN, name=b.client_obj.name))
        else:
            b.turn("client", _c(r, pool, svc=m["svc_main"].name, name=b.client_obj.name,
                                svc_g=cat.gen(m["svc_main"].name)))
    b.turn("operator", _c(r, OP_ACK) + " Что там у вас?")
    m["colloq"] = r.choice(prof["colloq"])
    return m


def filler_exchange(b, force=False):
    r = b.rng
    if force or r.random() < 0.35:
        client_ack(b)
        b.turn("operator", _c(r, FILL_O))
        if force or r.random() < 0.5:
            b.turn("client", _c(r, FILL_C))


def client_ack(b):
    """Короткая реплика клиента, если последним говорил оператор."""
    if b.turns and b.last_speaker() == "operator":
        b.turn("client", b.rng.choice(["Так, понятно.", "Ага.", "Угу, хорошо.",
                                       "Хорошо.", "Да-да.", "Ну ладно."]))


def closing(b):
    r = b.rng
    client_ack(b)
    b.turn("operator", _c(r, CLOSE_O))
    if r.random() < 0.75:
        b.turn("client", _c(r, CLOSE_C))


def rejected_guess(b, m, wrong_obj, reason_text):
    """Гипотеза оператора, которую клиент отвергает. Факта нет."""
    r = b.rng
    b.turn("operator", f"А может, дело в {cat.prep(wrong_obj.name)}? Такое бывает, когда {reason_text}.")
    b.turn("client", _c(r, ["Да нет, {obj} в порядке, дело не в нём.",
                            "Нет, {obj} тут точно ни при чём, я проверяла.",
                            "Не-не, с {obj_g} всё нормально, я в нём уверена.",
                            "Да нет что вы, {obj} работает нормально."],
                      obj=wrong_obj.name, obj_g=cat.gen(wrong_obj.name)))


def dangling_offer(b, m, offer_text):
    """Предложение оператора, на которое клиент не отвечает по существу."""
    r = b.rng
    b.turn("operator", offer_text)
    b.turn("client", _c(r, ["Не знаю даже, подумаю ещё.",
                            "Ну не уверена насчёт этого.",
                            "Я пока не готова это обсуждать, давайте про {svc}.",
                            "Оставлю как есть, спасибо."], svc=cat.gen(m["colloq"])))


# ============================================================== семейства ====
def fam_svc_recent_break(b, m):
    """Услуга сломалась сегодня или 1–5 дней назад; действия, заявка, перезвонок."""
    r = b.rng
    opening(b, m)
    off = r.choice([0, -1, -1, -2, -3, -4, -5])
    ph, hour = b.dg.exact(off)
    svc, col = m["svc_main"], m["colloq"]
    h = b.carrier("client", [("f", "короче, у меня "),
                             ("q", f"{m['symptom']} {ph}".strip(), "s0"),
                             ("f", ".")])
    b.fact(h, svc, "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    b.turn("operator", f"{_c(r, OP_ACK)} Давно так? Что-то уже пробовали?")
    if r.random() < 0.75:
        act = r.choice(cat.ACTIONS_DONE)
        failed = act not in {"пополнение счёта", "проверка баланса", "проверка оплаты"}
        off2 = r.choice([-2, -1, 0])
        ph2, _ = b.dg.exact(off2, tense="past")
        q2 = f"{act}{cat.ACTIONS_FAILED_SUFFIX if failed else ''}"
        h2 = b.carrier("client", [("f", "Я уже всякое пробовала: "), ("q", q2, "a0"),
                                  ("f", f". {ph2[0].upper() + ph2[1:]} делала, всё равно так же.")])
        b.fact(h2, b.client, "action", q2, vf=off2, quote_key="a0")
        b.turn("operator", "Ясно, значит, аппаратная часть под вопросом.")
    client_ack(b)
    filler_exchange(b)
    client_ack(b)
    tkt = b.ticket()
    ph_wd, off_wd = b.dg.future_wd(r.randrange(7))
    h3 = b.carrier("operator", [
        ("f", "Оформляю "),
        ("q", f"заявку {tkt.name.split()[-1]}", "t0"),
        ("f", ", инженеры проверят линию. И "),
        ("q", f"оператор перезвонит вам {ph_wd}", "g0"),
        ("f", "."),
    ])
    b.fact(h3, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    b.fact(h3, b.client, "agreement", f"перезвонок специалиста, оператор перезвонит {ph_wd}",
           vf=off_wd, quote_key="g0")
    if r.random() < 0.6:
        b.turn("client", "Да, хорошо, буду ждать.")
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_svc_broken_fixed_in_call(b, m):
    """Проблема с прошлой недели; во время звонка восстановилось -> цепочка."""
    r = b.rng
    opening(b, m)
    svc, col = m["svc_main"], m["colloq"]
    k = r.randrange(7)
    ph, off = b.dg.past_wd(k, form=r.choice(["acc", "acc", "gen_from"]))
    q1 = f"{m['symptom']} {ph}".strip()
    h = b.carrier("client", [("f", "У меня "), ("q", q1, "s0"),
                             ("f", ". Сижу уже намучилась.")])
    b.fact(h, svc, "service_state", m["symptom"], vf=off, quote_key="s0")
    b.turn("operator", f"{_c(r, OP_ACK)} Давайте перезагрузим оборудование: выключите из "
                       f"розетки, подождите полминуты, включите снова.")
    b.turn("client", "Так, секунд сорок прошло… включаю.")
    b.turn("operator", "Ну что, как теперь?")
    if r.random() < 0.8:
        b.turn("client", "О! Заработало, всё грузит как раньше.")
        h2 = b.carrier("operator", [("f", "Отлично, значит, сбой был на линии, "),
                                    ("q", "сейчас всё работает нормально", "s1"),
                                    ("f", ". Наблюдайте ещё день, если повторится — звоните.")])
        b.fact(h2, svc, "service_state", "работает нормально", vf=0, quote_key="s1")
        b.close_last(svc, off, 0)
    else:
        b.turn("client", "Нет, всё равно не пашет.")
        b.turn("operator", "Понял, тогда оформим заявку на проверку.")
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_svc_month_unused(b, m):
    """Клиент уже не пользуется одной услугой (polarity -1), повод — другая."""
    r = b.rng
    opening(b, m, greeting="topic")
    svc_main = m["svc_main"]
    svc_old = b.service(r.choice([n for n in cat.SERVICE_PROFILES if n != svc_main.name]))
    off = b.dg.month_back()
    ph_old, _ = b.dg.exact(off)
    wrong_obj = b.device("роутер")
    off_m = r.choice([-1, -2, -3])
    ph_m, hour_m = b.dg.exact(off_m)
    h0 = b.carrier("client", [("f", "Да вот, смотрите: "),
                              ("q", f"{m['symptom']} {ph_m}", "m0"),
                              ("f", ", сил никаких.")])
    b.fact(h0, svc_main, "service_state", m["symptom"], vf=off_m, hour=hour_m,
           quote_key="m0")
    rejected_guess(b, m, wrong_obj, "устаревшая прошивка сбивает канал")
    b.turn("operator", _c(r, OP_ACK))
    h = b.carrier("client", [
        ("f", "И вообще, "),
        ("q", f"я от {cat.gen(svc_old.name)} отказалась ещё {ph_old}", "s0"),
        ("f", ", он мне не нужен, я им "),
        ("q", "больше не пользуюсь", "s1"),
        ("f", "."),
    ])
    b.fact(h, svc_old, "service_state", "больше не используется", vf=off,
           polarity=-1, quote_key="s0")
    b.turn("operator", "Понял, тогда эту услугу не трогаем. Разберёмся с основной.")
    filler_exchange(b)
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_svc_misattribution(b, m):
    """Клиент винит одну услугу, оператор выясняет — проблема точно в другой."""
    r = b.rng
    opening(b, m)
    svc_a, svc_b = m["svc_main"], b.service(
        r.choice([n for n in cat.SERVICE_PROFILES if n != m["svc_main"].name]))
    col = m["colloq"]
    off = r.choice([0, -1, -2, -3])
    ph, hour = b.dg.exact(off)
    h = b.carrier("client", [("f", "Точно вам говорю: "),
                             ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ", я уже всё перепробовала.")])
    b.fact(h, svc_a, "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    b.turn("operator", f"Хорошо. А {svc_b.name} у вас как, работает?")
    b.turn("client", f"Вот {svc_b.name} как раз нормально, это точно не она.")
    filler_exchange(b)
    client_ack(b)
    tkt = b.ticket()
    h2 = b.carrier("operator", [
        ("f", f"Понял: {svc_b.name} — в порядке. Разбираемся. Оформляю "),
        ("q", f"заявку {tkt.name.split()[-1]}", "t0"),
        ("f", " на проверку, это займёт день-два.")])
    b.fact(h2, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_svc_two_services(b, m):
    """Две услуги: одна сломалась, вторая работает нормально."""
    r = b.rng
    opening(b, m)
    svc_b = b.service(r.choice([n for n in cat.SERVICE_PROFILES if n != m["svc_main"].name]))
    off = r.choice([-1, -2, -3, -5])
    ph, hour = b.dg.exact(off)
    col = m["colloq"]
    h = b.carrier("client", [("f", "Значит так: "), ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ", а со второй беда — не знаю, что и думать.")])
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    b.turn("operator", f"Скажите, {svc_b.name} не подводит? Бывает, заодно глючат.")
    b.turn("client", f"Нет, {svc_b.name} работает нормально, я проверяла.")
    h2 = b.carrier("operator", [("f", f"Понял: {svc_b.name} — "),
                                ("q", "работает нормально", "s1"),
                                ("f", ". Значит, разбираемся с основной.")])
    b.fact(h2, svc_b, "service_state", "работает нормально", vf=off, quote_key="s1")
    filler_exchange(b)
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_svc_state_sequence(b, m):
    """Цепочка состояний: месяц назад сломалось -> неделя норм -> снова сломалось."""
    r = b.rng
    opening(b, m)
    svc, col = m["svc_main"], m["colloq"]
    off1 = b.dg.month_back()
    ph1, _ = b.dg.exact(off1)
    off3 = r.choice([-1, -2])
    ph3, hour = b.dg.exact(off3)
    sym = m["symptom"]
    h = b.carrier("client", [
        ("f", "Слушайте, история долгая. "),
        ("q", f"{ph1} всё сломалось", "s0"),
        ("f", ", две недели назад вроде починилось, было "),
        ("q", "работает нормально", "s1"),
        ("f", f", а {ph3} — опять "),
        ("q", sym, "s2"),
        ("f", "."),
    ])
    b.fact(h, svc, "service_state", sym, vf=off1, quote_key="s0")
    b.fact(h, svc, "service_state", "работает нормально", vf=-14, quote_key="s1")
    b.fact(h, svc, "service_state", sym, vf=off3, hour=hour, quote_key="s2")
    b.close_last(svc, off1, -14)
    b.close_last(svc, -14, off3)
    b.turn("operator", "Понял, значит, проблема плавающая. Заявку зафиксируем.")
    filler_exchange(b)
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_tariff_recap(b, m):
    """Текущий тариф + смена месяц назад; клиент путается, оператор уточняет."""
    r = b.rng
    tariffs = r.sample(cat.load_catalog()["tariffs"], 3)
    new_t, old_t, other_t = tariffs
    opening(b, m, no_svc=True)
    b.turn("client", f"Скажите, а какой у меня сейчас тариф? Я запуталась совсем, кажется, {other_t}?")
    off = b.dg.month_back()
    ph, _ = b.dg.exact(off)
    h = b.carrier("operator", [("f", "Нет, у вас сейчас "),
                               ("q", f"тариф {new_t}", "t0"),
                               ("f", ". Раньше стоял другой, вы переходили.")])
    b.fact(h, b.client, "tariff", new_t, vf=off, quote_key="t0")
    b.turn("client", f"И какой же был раньше? Я помню только, что {old_t} точно нет.")
    h2 = b.carrier("operator", [("f", "Смотрите: "),
                                ("q", f"тариф {old_t}", "t1"),
                                (f" был у вас до {ph}, потом сменился.")])
    b.fact(h2, b.client, "tariff", old_t, vf=off - 30, vt=off, quote_key="t1")
    b.turn("client", f"А-а, точно, {old_t} был. А чем новый лучше?")
    dangling_offer(b, m, f"Могу рассказать про {new_t} подробнее: там больше гигабайтов.")
    closing(b)
    m["scenario_type"] = "tariff"
    return m


def fam_tariff_recent_change(b, m):
    """Смена тарифа 3–6 дней назад."""
    r = b.rng
    tariffs = r.sample(cat.load_catalog()["tariffs"], 2)
    new_t, old_t = tariffs
    opening(b, m, no_svc=True)
    off = r.choice([-3, -4, -5, -6])
    ph, _ = b.dg.exact(off)
    h = b.carrier("client", [("f", "Кстати, я "),
                             ("q", f"{ph} сменила тариф на {new_t}", "t0"),
                             ("f", ", в приложении. Списания какие-то странные пошли.")])
    b.fact(h, b.client, "tariff", new_t, vf=off, quote_key="t0")
    h2 = b.carrier("operator", [("f", "Вижу: до смены у вас стоял "),
                                ("q", f"тариф {old_t}", "t1"),
                                ("f", ", сейчас уже новый.")])
    b.fact(h2, b.client, "tariff", old_t, vf=off - 30, vt=off, quote_key="t1")
    b.append_free(h2, " Списания пересчитаются автоматически, не переживайте.")
    b.turn("client", "Уф, то-то же, а то я уже испугалась.")
    closing(b)
    m["scenario_type"] = "tariff"
    return m


def fam_tariff_planned_change(b, m):
    """Запланированный переход: завтра / в будущий день недели."""
    r = b.rng
    tariffs = r.sample(cat.load_catalog()["tariffs"], 2)
    new_t, old_t = tariffs
    opening(b, m, no_svc=True)
    from .dates import WD_GEN
    k = r.randrange(7)
    ph, off = b.dg.future_wd(k)
    h = b.carrier("client", [("f", "Мне в приложении показывают, что "),
                             ("q", f"переход на тариф {new_t} {ph}", "t0"),
                             ("f", ". Так и должно быть? Я в прошлом месяце заявку оставляла.")])
    b.fact(h, b.client, "tariff", new_t, vf=off, quote_key="t0")
    h2 = b.carrier("operator", [("f", f"Да, всё верно: до {WD_GEN[k]} действует "),
                                ("q", f"тариф {old_t}", "t1"),
                                ("f", ", дальше автоматически переключится.")])
    b.fact(h2, b.client, "tariff", old_t, vf=-30, vt=off, quote_key="t1")
    b.turn("client", "Уф, хорошо, значит, я ничего не сломала.")
    closing(b)
    m["scenario_type"] = "tariff"
    return m


def fam_tariff_offer_refused(b, m):
    """Оператор предлагает смену тарифа, клиент отказывается."""
    r = b.rng
    tariffs = r.sample(cat.load_catalog()["tariffs"], 2)
    new_t, cur_t = tariffs
    opening(b, m, no_svc=True)
    b.turn("client", "Да вот, собственно, пришла квитанция, и хочу разобраться.")
    h = b.carrier("operator", [("f", "Кстати, у вас сейчас "),
                               ("q", f"тариф {cur_t}", "t0"),
                               ("f", f". Могу предложить перейти на {new_t}, там выгоднее гигабайты.")])
    b.fact(h, b.client, "tariff", cur_t, vf=0, quote_key="t0")
    b.turn("client", r.choice(["Нет, спасибо, меня всё устраивает.",
                               "Да нет, оставлю как есть.",
                               "Не нужно, я привыкла к своему.",
                               "Нет-нет, переходить не буду."]))
    h2 = b.carrier("operator", [("f", "Хорошо, без изменений. "),
                                ("q", f"клиент отказался от перехода на тариф {new_t}", "g0"),
                                ("f", ", тариф не трогаем.")])
    b.fact(h2, b.client, "agreement",
           f"клиент отказался от перехода на тариф {new_t}", vf=0, quote_key="g0")
    closing(b)
    m["scenario_type"] = "tariff"
    return m


def fam_tariff_offer_dangling(b, m):
    """Предложение тарифа повисло; «да» клиента относится к другому вопросу."""
    r = b.rng
    tariffs = r.sample(cat.load_catalog()["tariffs"], 2)
    new_t, _ = tariffs
    opening(b, m, no_svc=True)
    b.turn("client", "Да вот, слушайте, не знаю, что и делать.")
    b.turn("operator", f"Могу предложить переход на {new_t}, будет дешевле при том же интернете.")
    b.turn("client", "Ну, не знаю даже…")
    b.turn("operator", f"Подумайте. Кстати, а {m['svc_main'].name} у вас сейчас как, работает?")
    b.turn("client", "Да нет, с ним беда, я по этому поводу и звоню.")
    b.turn("operator", "Понял, понял.")
    off = r.choice([-1, -2])
    ph, hour = b.dg.exact(off)
    col = m["colloq"]
    h = b.carrier("client", [("f", "Так вот, у меня "), ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ". Вот про тариф потом отдельным звонком.")])
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    b.turn("operator", "Хорошо, тариф не трогаем, разберёмся с интернетом.")
    closing(b)
    m["scenario_type"] = "tariff"
    return m


def fam_tkt_created_now(b, m):
    """Заявка создаётся в этом звонке; статус «принята»; подтверждение по смс."""
    r = b.rng
    opening(b, m, no_svc=True)
    off = r.choice([-1, -2])
    ph, hour = b.dg.exact(off)
    col = m["colloq"]
    h = b.carrier("client", [("f", "У меня "), ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ", уже всё перепробовала.")])
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    tkt = b.ticket()
    h2 = b.carrier("operator", [("f", "Сейчас оформлю обращение… вот, готово: "),
                                ("q", f"заявка {tkt.name.split()[-1]}", "t0"),
                                ("f", " принята, по ней "),
                                ("q", "придёт подтверждение по смс", "g0"),
                                ("f", ".")])
    b.fact(h2, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    b.fact(h2, b.client, "agreement", "подтверждение заявки по смс", vf=0, quote_key="g0")
    if r.random() < 0.6:
        b.turn("client", "Хорошо, жду смску.")
    closing(b)
    m["scenario_type"] = "ticket_status"
    return m


def fam_tkt_history_chain(b, m):
    """Заявка недельной давности: принята -> в работе -> выполнена/перезвонок."""
    r = b.rng
    opening(b, m, no_svc=True)
    tkt = b.ticket()
    num = tkt.name.split()[-1]
    off0 = r.choice([-6, -7, -8, -9])
    ph0, _ = b.dg.exact(off0)
    off_mid = max(off0 + 2, r.choice([-4, -3]))
    ph_mid, _ = b.dg.exact(off_mid)
    b.turn("client", f"Здравствуйте, это {b.client_obj.name.split()[0]}. Что там с заявкой {num}?")
    h2 = b.carrier("operator", [
        ("f", "Секунду, смотрю историю… Вижу: "),
        ("q", f"заявка {num} {ph0} была принята", "t0"),
        ("f", ", "),
        ("q", f"{ph_mid} перешла в работу", "t1"),
        ("f", "."),
    ])
    b.fact(h2, tkt, "ticket_status", "принята", vf=off0, vt=off_mid, quote_key="t0")
    b.fact(h2, tkt, "ticket_status", "в работе", vf=off_mid, quote_key="t1")
    if r.random() < 0.55:
        b.append_free(h2, " И сегодня, кстати, ")
        b.add_quote(h2, "её уже закрыли — выполнена", "t2")
        b.append_free(h2, ".")
        b.fact(h2, tkt, "ticket_status", "выполнена", vf=0, quote_key="t2")
        b.close_last(tkt, off_mid, 0)
        b.turn("client", "Отлично, спасибо!")
    else:
        b.append_free(h2, " Инженеры обещают результат ")
        b.add_quote(h2, "в течение двух дней", "g0")
        b.append_free(h2, ", я вам и перезвоню.")
        b.fact(h2, b.client, "agreement", "перезвонок специалиста в течение двух дней",
               vf=2, quote_key="g0")
        b.turn("client", "Ну и когда ждать-то? Уже сил нет.")
    closing(b)
    m["scenario_type"] = "ticket_status"
    return m


def fam_tkt_visit_scheduled(b, m):
    """Выезд специалиста: статус «назначен выезд» + согласование времени."""
    r = b.rng
    opening(b, m, no_svc=True)
    tkt = b.ticket()
    num = tkt.name.split()[-1]
    b.turn("client", f"Мне говорили, по заявке {num} кто-то должен приехать. Так?")
    k = r.randrange(7)
    ph_v, off_v = b.dg.future_wd(k)
    hour = r.choice([9, 13, 14, 18])
    ph_h = hour_phrase(hour) or "утром"
    h = b.carrier("operator", [("f", "Так, смотрю… "),
                               ("q", f"по заявке {num} назначен выезд специалиста", "t0"),
                               ("f", f". Можем поставить {ph_v} {ph_h}. Подходит?")])
    b.fact(h, tkt, "ticket_status", "назначен выезд специалиста", vf=-2, quote_key="t0")
    b.turn("client", f"Да, подходит, {ph_v} {ph_h} я буду дома.")
    h3 = b.carrier("operator", [("f", "Записал: "),
                                ("q", f"специалист приедет {ph_v} {ph_h}", "g0"),
                                ("f", ". Перед выездом позвонит за полчаса.")])
    b.fact(h3, b.client, "agreement", f"выезд специалиста, специалист приедет {ph_v} {ph_h}",
           vf=off_v, hour=hour, quote_key="g0")
    closing(b)
    m["scenario_type"] = "agreement"
    return m


def fam_tkt_wrong_number(b, m):
    """Оператор путает номер заявки и исправляется; факт только по верному."""
    r = b.rng
    opening(b, m, no_svc=True)
    tkt = b.ticket()
    wrong = b.ticket()
    num, wnum = tkt.name.split()[-1], wrong.name.split()[-1]
    b.turn("client", f"Здравствуйте! Звоню по заявке {num}, что там по ней?")
    h = b.carrier("operator", [
        ("f", f"Так, заявка {wnum}… нет, прошу прощения, {wnum} — это не ваша. Вот она: "),
        ("q", f"заявка {num} в работе", "t0"),
        ("f", ", инженеры смотрят канал.")])
    off = r.choice([-3, -4])
    ph, _ = b.dg.exact(off)
    b.append_free(h, f" Принята она была {ph}.")
    b.fact(h, tkt, "ticket_status", "принята", vf=off - 1, vt=off, quote=False)
    b.fact(h, tkt, "ticket_status", "в работе", vf=off, quote_key="t0")
    b.turn("client", f"Ага, то есть по {num} всё в процессе?")
    b.turn("operator", f"Именно, по {num} всё в процессе, ждать осталось недолго.")
    closing(b)
    m["scenario_type"] = "ticket_status"
    return m


def fam_tkt_two_tickets(b, m):
    """Две заявки: старая выполнена месяц назад, новая в работе."""
    r = b.rng
    opening(b, m, no_svc=True)
    t1, t2 = b.ticket(), b.ticket()
    n1, n2 = t1.name.split()[-1], t2.name.split()[-1]
    off1 = b.dg.month_back()
    ph1, _ = b.dg.exact(off1)
    b.turn("client", f"У меня было обращение {n1} и потом {n2}. Как с ними?")
    h = b.carrier("operator", [
        ("f", "Смотрю обе: "),
        ("q", f"{n1} {ph1} была выполнена", "t0"),
        ("f", ", а "),
        ("q", f"{n2} сейчас в работе", "t1"),
        ("f", "."),
    ])
    b.fact(h, t1, "ticket_status", "выполнена", vf=off1, vt=off1, quote_key="t0")
    b.fact(h, t2, "ticket_status", "в работе", vf=r.choice([-3, -2, -1]), quote_key="t1")
    filler_exchange(b)
    closing(b)
    m["scenario_type"] = "ticket_status"
    return m


def fam_act_reboot_helped(b, m):
    """Перезагрузка сегодня помогла: action + цепочка состояний устройства."""
    r = b.rng
    opening(b, m, greeting="topic")
    dev = b.device()
    col = m["colloq"]
    h = b.carrier("client", [
        ("f", f"Знаете, я {r.choice(['сегодня утром', 'сегодня', 'сегодня после обеда'])} "),
        ("q", "перезагрузка роутера", "a0"),
        ("f", f" сделала, и "),
        ("q", "теперь всё работает нормально", "s0"),
        ("f", f". А до этого {col} вообще не грузил."),
    ])
    off_break = r.choice([-2, -3])
    ph_b, _ = b.dg.exact(off_break)
    b.fact(h, dev, "action", "перезагрузка роутера", vf=0, quote_key="a0")
    b.fact(h, dev, "service_state", "работает нормально", vf=0, quote_key="s0")
    b.turn("operator", "А когда начались проблемы-то?")
    h2 = b.carrier("client", [("f", "Да "), ("q", f"{ph_b} всё висло", "s1"), ("f", ".")])
    b.fact(h2, dev, "service_state", "не подключается к сети", vf=off_break, quote_key="s1")
    b.close_last(dev, off_break, 0)
    b.turn("operator", "Хорошо, что помогло. Если повторится — пишите в чат.")
    closing(b)
    m["scenario_type"] = "action"
    return m


def fam_act_tried_many(b, m):
    """Клиент перечисляет безуспешные действия."""
    r = b.rng
    opening(b, m)
    col = m["colloq"]
    off = r.choice([-2, -3, -4])
    ph, hour = b.dg.exact(off)
    acts = r.sample(cat.ACTIONS_DONE, 2)
    q0 = f"{acts[0]}{cat.ACTIONS_FAILED_SUFFIX}"
    q1 = f"{acts[1]}{cat.ACTIONS_FAILED_SUFFIX}"
    h = b.carrier("client", [
        ("f", f"У меня прям беда: {m['symptom']} {ph}. Я и "),
        ("q", q0, "a0"),
        ("f", ", и "),
        ("q", q1, "a1"),
        ("f", " — бесполезно."),
    ])
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour,
           quote=False)
    b.fact(h, b.client, "action", q0, vf=off, quote_key="a0")
    b.fact(h, b.client, "action", q1, vf=off, quote_key="a1")
    b.turn("operator", "Понял, значит, на вашей стороне всё испробовано. Заявку ставлю.")
    filler_exchange(b)
    closing(b)
    m["scenario_type"] = "action"
    return m


def fam_act_refusal_and_topup(b, m):
    """Клиент пополнил счёт; услуга не заработала; от дополнительной отказался."""
    r = b.rng
    opening(b, m, greeting="undated")
    off = r.choice([-1, -2])
    ph, _ = b.dg.exact(off)
    h = b.carrier("client", [("f", "Слушайте, я "), ("q", f"{ph} пополнила счёт", "a0"),
                             ("f", ", и деньги ушли, а "),
                             ("q", m["symptom"], "s0"),
                             ("f", " — как было, так и осталось.")])
    b.fact(h, b.client, "action", "пополнение счёта", vf=off, quote_key="a0")
    # продолжающееся состояние услуги: недатировано -> vf = день звонка (как в gold)
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=0, quote_key="s0")
    b.turn("operator", "Так, баланс вижу, оплата прошла. Разберёмся с услугой. "
                       "И ещё: можем подключить видеонаблюдение со скидкой, интересно?")
    b.turn("client", f"Да нет, спасибо, не нужно. С {cat.gen(m['colloq'])} бы разобраться.")
    h2 = b.carrier("operator", [("f", "Хорошо, "),
                                ("q", "клиент отказался от подключения видеонаблюдения", "g0"),
                                ("f", ". Занимаемся вашей проблемой.")])
    b.fact(h2, b.client, "agreement", "клиент отказался от подключения видеонаблюдения",
           vf=0, quote_key="g0")
    closing(b)
    m["scenario_type"] = "action"
    return m


def fam_agr_conditional(b, m):
    """Условное согласие: только вечером / только после шести."""
    r = b.rng
    opening(b, m, no_svc=True)
    k = r.randrange(7)
    ph_v, off_v = b.dg.future_wd(k)
    hour = r.choice([18, 19, 9])
    ph_h = hour_phrase(hour) or "после шести"
    b.turn("client", "Да вот, хочу записаться на выезд, а то сама не справлюсь.")
    b.turn("operator", f"Конечно, оформим. По вашей проблеме нужен выезд мастера. {ph_v} удобно?")
    b.turn("client", f"С утра никак, я на работе. Только {ph_h}.")
    b.turn("operator", f"Хорошо, ставлю {ph_v} {ph_h}.")
    h = b.carrier("client", [("f", r.choice(["Договорились.", "Да, давайте так.",
                                             "Ок, так и сделаем."]) + " "),
                             ("q", f"только {ph_h}", "g0"),
                             ("f", ".")])
    b.fact(h, b.client, "agreement", f"выезд специалиста, специалист приедет {ph_v} {ph_h}",
           vf=off_v, hour=hour, quote_key="g0")
    closing(b)
    m["scenario_type"] = "agreement"
    return m


def fam_agr_mind_changed(b, m):
    """Согласился — передумал; оператор согласился отложить."""
    r = b.rng
    opening(b, m)
    b.turn("client", "Да вот, думаю, может роутер новый взять?")
    b.turn("operator", "Могу оформить аренду нового роутера на месяц, бесплатная примерка.")
    b.turn("client", "Да, давайте, хорошо.")
    b.turn("operator", "Оформляю…")
    b.turn("client", "Так, стоп. Нет, я передумала, у меня и старый нормально держит.")
    h = b.carrier("operator", [("f", "Хорошо, не проблема. Фиксирую: "),
                               ("q", "клиент передумал, от аренды роутера отказалась", "g0"),
                               ("f", ".")])
    b.fact(h, b.client, "agreement", "клиент отказался от аренды нового роутера",
           vf=0, quote_key="g0")
    off = r.choice([-1, -2])
    ph, hour = b.dg.exact(off)
    h2 = b.carrier("client", [("f", "А вот главное: "),
                              ("q", f"{m['symptom']} {ph}", "s0"),
                              ("f", " — вот это почините, пожалуйста.")])
    b.fact(h2, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    closing(b)
    m["scenario_type"] = "agreement"
    return m


def fam_agr_unanswered(b, m):
    """«Да» клиента относится к другому вопросу; предложение повисло."""
    r = b.rng
    opening(b, m)
    off = r.choice([-1, -2, -3])
    ph, hour = b.dg.exact(off)
    col = m["colloq"]
    h = b.carrier("client", [("f", "У меня прям беда: "), ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ", вот не знаю уже что делать.")])
    b.fact(h, m["svc_main"], "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    b.turn("operator", "Ясно. Давайте проверим: загорается ли индикатор питания на роутере?")
    b.turn("client", "Да, горит, зелёный.")
    b.turn("operator", "Отлично. И последний вопрос: не хотите пока подключить безлимит "
                       "на соцсети, раз сидите без дела?")
    b.turn("client", "Да, я как раз весь день дома, сайт не открывается вообще.")
    tkt = b.ticket()
    h2 = b.carrier("operator", [("f", "Понял, про безлимит переспрошу в другой раз. Оформляю "),
                                ("q", f"заявку {tkt.name.split()[-1]}", "t0"),
                                ("f", ".")])
    b.fact(h2, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    closing(b)
    m["scenario_type"] = "agreement"
    return m


def fam_billing_double(b, m):
    """Двойное списание: state факт + возврат средств через N дней."""
    r = b.rng
    opening(b, m)
    off = r.choice([-3, -4, -5])
    ph, _ = b.dg.exact(off)
    h = b.carrier("client", [("f", f"Смотрите, {ph} пришло списание, "),
                             ("q", "двойное списание", "s0"),
                             ("f", f" за {m['colloq']}. Деньги сняли два раза!")])
    b.fact(h, m["svc_main"], "service_state", "двойное списание", vf=off, quote_key="s0")
    b.turn("operator", "Так, вижу в биллинге дубль. Сделаю возврат.")
    b.turn("client", "Ух, слава богу. А то я уже думала, совсем всё пропало.")
    n = r.choice([2, 3, 4, 5])
    ph_n = f"через {NUM_WORDS[n]} " + ("дня" if n in (2, 3, 4) else "дней")
    h2 = b.carrier("operator", [("f", "Оформляю возврат средств, "),
                                ("q", f"деньги поступят {ph_n}", "g0"),
                                ("f", ".")])
    b.fact(h2, b.client, "agreement", f"возврат средств, деньги поступят {ph_n}",
           vf=n, quote_key="g0")
    b.turn("client", "Хорошо, буду ждать.")
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_phone_line(b, m):
    """Домашний телефон: треск/обрывы, заявка на проверку линии."""
    r = b.rng
    svc = b.service("домашний телефон")
    opening(b, m, svc=svc)
    m["colloq"] = r.choice(b.profile(svc)["colloq"])
    off = r.choice([-1, -2, -4])
    ph, hour = b.dg.exact(off)
    sym = r.choice(b.profile(svc)["symptoms"])
    h = b.carrier("client", [("f", "Слушайте, "), ("q", f"{sym} {ph}", "s0"),
                             ("f", ", поговорить невозможно.")])
    b.fact(h, svc, "service_state", sym, vf=off, hour=hour, quote_key="s0")
    b.turn("operator", "Проверим линию. У вас трубка радиотелефон или обычная?")
    b.turn("client", "Обычная, проводная.")
    tkt = b.ticket()
    h2 = b.carrier("operator", [
        ("f", "Понял, значит, дело в линии. Оформляю "),
        ("q", f"заявку {tkt.name.split()[-1]}", "t0"),
        ("f", " на проверку линии на станции.")])
    b.fact(h2, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    closing(b)
    m["scenario_type"] = "service_state"
    return m


def fam_tv_settop(b, m):
    """Приставка: перенастройка удалённая завтра/через N дней."""
    r = b.rng
    svc = b.service("интерактивное телевидение")
    opening(b, m, svc=svc)
    m["colloq"] = r.choice(b.profile(svc)["colloq"])
    off = r.choice([-1, -2])
    ph, hour = b.dg.exact(off)
    sym = r.choice(b.profile(svc)["symptoms"])
    h = b.carrier("client", [("f", "У меня "), ("q", f"{sym} {ph}", "s0"),
                             ("f", ". Утром включаю — а она висит.")])
    b.fact(h, svc, "service_state", sym, vf=off, hour=hour, quote_key="s0")
    b.turn("operator", "Сделаем перенастройку удалённо, без мастера.")
    b.turn("client", "И когда?")
    n = r.choice([1, 2, 3])
    ph_n = "завтра" if n == 1 else f"через {NUM_WORDS[n]} " + ("дня" if n in (2, 3) else "дней")
    h2 = b.carrier("operator", [("q", f"перенастройка приставки удалённо, срок — {ph_n}", "g0"),
                                ("f", ". Приставку не трогайте до вечера.")])
    b.fact(h2, b.client, "agreement", f"перенастройка приставки удалённо, срок — {ph_n}",
           vf=n, quote_key="g0")
    b.turn("client", "Ок, не трону.")
    closing(b)
    m["scenario_type"] = "agreement"
    return m


def fam_followup_recap(b, m):
    """Повторный звонок: пересказ истории старой заявки."""
    r = b.rng
    opening(b, m, no_svc=True)
    tkt = b.ticket()
    num = tkt.name.split()[-1]
    off0 = r.choice([-8, -9, -10])
    ph0, _ = b.dg.exact(off0)
    b.turn("client", f"Здравствуйте, я уже звонила насчёт заявки {num}. Новостей нет?")
    h = b.carrier("operator", [
        ("f", "Смотрю историю: "),
        ("q", f"заявка {num} {ph0} была принята", "t0"),
        ("f", ", до вчерашнего дня "),
        ("q", "числилась в работе", "t1"),
        ("f", "."),
    ])
    b.fact(h, tkt, "ticket_status", "принята", vf=off0, vt=-1, quote_key="t0")
    b.fact(h, tkt, "ticket_status", "в работе", vf=-1, quote_key="t1")
    b.turn("client", "И что, инженеры что-то делают?")
    b.turn("operator", "Да, инженеры уже на линии, обещают к вечеру результат.")
    closing(b)
    m["scenario_type"] = "ticket_status"
    return m


def fam_mixed_full(b, m):
    """Большой звонок: услуга + тариф + заявка + выезд. 5-7 фактов."""
    r = b.rng
    opening(b, m)
    svc, col = m["svc_main"], m["colloq"]
    off = r.choice([-3, -4, -5, -7])
    ph, hour = b.dg.exact(off)
    h = b.carrier("client", [("f", "У меня беда: "), ("q", f"{m['symptom']} {ph}", "s0"),
                             ("f", ", я уже не знаю, к кому идти.")])
    b.fact(h, svc, "service_state", m["symptom"], vf=off, hour=hour, quote_key="s0")
    act = r.choice(["перезагрузка роутера", "проверка кабеля", "перезагрузка модема"])
    b.turn("operator", _c(r, OP_ACK) + " И что вы уже делали?")
    h2 = b.carrier("client", [("f", "Была "), ("q", f"{act} — не помогло", "a0"),
                              ("f", ", всё по инструкции делала.")])
    b.fact(h2, b.client, "action", f"{act} — не помогло", vf=off, quote_key="a0")
    filler_exchange(b, force=True)
    t_cur = r.choice(cat.load_catalog()["tariffs"])
    h3 = b.carrier("operator", [("f", "Так, по линии вижу сбой. У вас подключён "),
                                ("q", f"тариф {t_cur}", "t_cur"),
                                ("f", ", активен?")])
    b.fact(h3, b.client, "tariff", t_cur, vf=0, quote_key="t_cur")
    b.turn("client", "Да, тариф этот, я ничего не меняла.")
    tkt = b.ticket()
    k = r.randrange(7)
    ph_v, off_v = b.dg.future_wd(k)
    h4 = b.carrier("operator", [
        ("f", "Оформляю "), ("q", f"заявку {tkt.name.split()[-1]}", "t0"),
        ("f", " на диагностику, и "), ("q", f"специалист приедет {ph_v}", "g0"),
        ("f", ", это бесплатно по вашему тарифу."),
    ])
    b.fact(h4, tkt, "ticket_status", "принята", vf=0, quote_key="t0")
    b.fact(h4, b.client, "agreement", f"выезд специалиста, специалист приедет {ph_v}",
           vf=off_v, quote_key="g0")
    b.turn("client", "Хорошо, буду дома.")
    closing(b)
    m["scenario_type"] = "mixed"
    return m


FAMILIES = [
    fam_svc_recent_break, fam_svc_broken_fixed_in_call, fam_svc_month_unused,
    fam_svc_misattribution, fam_svc_two_services, fam_svc_state_sequence,
    fam_tariff_recap, fam_tariff_recent_change, fam_tariff_planned_change,
    fam_tariff_offer_refused, fam_tariff_offer_dangling,
    fam_tkt_created_now, fam_tkt_history_chain, fam_tkt_visit_scheduled,
    fam_tkt_wrong_number, fam_tkt_two_tickets,
    fam_act_reboot_helped, fam_act_tried_many, fam_act_refusal_and_topup,
    fam_agr_conditional, fam_agr_mind_changed, fam_agr_unanswered,
    fam_billing_double, fam_phone_line, fam_tv_settop, fam_followup_recap,
    fam_mixed_full,
]

WEIGHTS = [10, 8, 6, 7, 6, 6,
           7, 7, 6, 7, 6,
           8, 8, 7, 6, 6,
           8, 7, 7,
           7, 6, 7,
           7, 7, 7, 6,
           10]
