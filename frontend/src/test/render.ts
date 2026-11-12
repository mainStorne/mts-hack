import { setupI18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, createElement, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";

// Mounts a component with Lingui (English, no catalog) and React Query, for tests without a router
export function mount(element: ReactElement) {
  const container = document.createElement("div");
  document.body.append(container);
  const root: Root = createRoot(container);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const i18n = setupI18n({ locale: "en", messages: { en: {} } });
  act(() =>
    root.render(
      createElement(
        QueryClientProvider,
        { client },
        createElement(I18nProvider, { i18n }, element),
      ),
    ),
  );
  return {
    container,
    client,
    unmount() {
      act(() => root.unmount());
      client.clear();
      container.remove();
    },
  };
}

// Types into a controlled textarea the way a user does
export function typeInto(textarea: HTMLTextAreaElement, value: string) {
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )!.set!;
    setter.call(textarea, value);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

// A response stream of server-sent events
export function sseStream(
  events: [string, unknown][],
  options: { separator?: string } = {},
) {
  const { separator = "\n" } = options;
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const [name, data] of events)
        controller.enqueue(
          new TextEncoder().encode(
            `event: ${name}${separator}data: ${JSON.stringify(data)}${separator}${separator}`,
          ),
        );
      controller.close();
    },
  });
}
