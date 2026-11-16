# one-zero-eight · MTS MWS CompactFactExtraction

Команда **one-zero-eight** на Innopolis Global Hack (13–16 ноября 2026), кейс MTS MWS. Мы сделали компактную модель извлечения фактов из звонков в поддержку: GLiNER + Gemma 3 1B с логистическим реранкером (1,29 млрд параметров; F1 0,642 на val против 0,356 у GLM-5.3), Docker-образы для CPU и GPU с HTTP API, воспроизводимое обучение и отчёт «ресурсы × качество». Рядом лежит шаблон агента поддержки (backend и frontend).

<table>
  <tr>
    <td align="center" width="25%"><img src="assets/team/dima.png" width="120" alt="Дмитрий"><br><b>Дмитрий</b><br>ML-трек: обучение LoRA-экстракторов, ансамбль, замеры, образы и воспроизводимость</td>
    <td align="center" width="25%"><img src="assets/team/nikita.png" width="120" alt="Никита"><br><b>Никита</b><br>питч, исследование отчётов, обучение v2/v3, экономика решения</td>
    <td align="center" width="25%"><img src="assets/team/nastya.png" width="120" alt="Настя"><br><b>Настя</b><br>разбор кейса и данных, временные интервалы, кодек Gemma</td>
    <td align="center" width="25%"><img src="assets/team/max.png" width="120" alt="Максим"><br><b>Максим</b><br>оценка качества, экономическое обоснование, мультиязычность</td>
  </tr>
</table>

- Решение: [`ml/README.md`](ml/README.md), отчёт [`ml/docs/RESOURCES_QUALITY.md`](ml/docs/RESOURCES_QUALITY.md), развёртывание [`ml/DEPLOY.md`](ml/DEPLOY.md), веса [`ml/weights`](ml/weights/README.md).
- Презентация: [`index.html`](index.html) (офлайн-версия [`mws-master-slides.html`](mws-master-slides.html), PDF [`exports/deck-backup.pdf`](exports/deck-backup.pdf)).

---

# Support Agent: шаблон для хакатона

Агент поддержки оператора связи: специалисты (triage, billing, tariffs, tech, human) передают друг другу диалог (handoff), состояние диалога хранится в Postgres.

- [`backend/`](backend/README.md): FastAPI + LangChain/LangGraph + Postgres + Phoenix.
- [`frontend/`](frontend/README.md): React 19 + Vite + TanStack + Tailwind/DaisyUI (стек сайта InNoHassle, только интерфейс чата с агентом).

```bash
cd backend && cp .env.example .env && docker compose up -d db phoenix && uv run support-api
cd frontend && pnpm install && pnpm dev        # VITE_API_URL=http://localhost:8000
```
