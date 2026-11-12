# Support Agent — frontend

A chat with the customer support agent of a telecom operator (demo brand "Support Agent").
Single page app: React 19, Vite, TanStack Router/Query, Tailwind 4 + DaisyUI 5, Lingui (ru by default, en).

## Run

```sh
cp .env.example .env.local   # VITE_API_URL, default http://localhost:8000
pnpm install
pnpm dev                     # http://localhost:3000
```

The backend must allow CORS from the frontend origin.

## Scripts

| Script                | What it does                       |
| --------------------- | ---------------------------------- |
| `pnpm dev`            | dev server                         |
| `pnpm build`          | production build into `dist`       |
| `pnpm typecheck`      | `tsc --build --noEmit`             |
| `pnpm lint`           | ESLint (with the Lingui plugin)    |
| `pnpm test`           | Vitest (jsdom)                     |
| `pnpm prettify`       | Prettier                           |
| `pnpm lingui:extract` | update `src/locales/*.po` catalogs |

Catalogs are imported straight from `.po` by the Lingui Vite plugin, so there is no separate compile step.
After `pnpm lingui:extract`, translate the new empty messages in `src/locales/ru.po`.

## Structure

- `src/api/chat` — types, config (`VITE_API_URL`), client id (`X-Client-Id`, kept in localStorage), plain `fetch` client, query options.
- `src/components/chat` — `chat-session.ts` (the conversation store: SSE client, handoffs, approvals, sessionStorage), the page and its parts.
- `src/components/layout`, `toast`, `icons` — header, theme/locale switches, toasts, logo.
- `src/app` — entry, providers, file-based routes (`routes/index.tsx` is `/`, optional `?chat=<id>`).
- `src/locales` — Lingui setup and `en.po` / `ru.po`.
