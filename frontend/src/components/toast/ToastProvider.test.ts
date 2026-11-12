import { messages as enMessages } from "@/locales/en.po";
import { messages as ruMessages } from "@/locales/ru.po";
import { setupI18n, type I18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { act, createElement, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ToastContainer } from "./ToastContainer.tsx";
import { ToastProvider, useToast } from "./ToastContext.tsx";
import type { ToastContextType } from "./types.ts";

// Descriptors without the macro: they are only for this test, not for the catalogs
const title = { id: "test.title", message: "Success" };
const message = { id: "test.message", message: "Done" };

function ToastHarness({
  onReady,
}: {
  onReady: (toast: ToastContextType) => void;
}) {
  const toast = useToast();
  useEffect(() => onReady(toast), [onReady, toast]);
  return createElement(ToastContainer);
}

describe("reactive toast localization", () => {
  let container: HTMLDivElement;
  let root: Root;
  let i18n: I18n;
  let toast: ToastContextType;

  beforeEach(async () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    i18n = setupI18n({
      locale: "en",
      messages: {
        en: { ...enMessages },
        ru: {
          ...ruMessages,
          "test.title": "Успех",
          "test.message": "Готово",
        },
      },
    });
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root.render(
        createElement(
          I18nProvider,
          { i18n },
          createElement(
            ToastProvider,
            null,
            createElement(ToastHarness, {
              onReady: (value) => {
                toast = value;
              },
            }),
          ),
        ),
      );
    });
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  });

  it.each(["showSuccess", "showError"] as const)(
    "%s keeps the descriptors until render and retranslates the same toast on a locale switch",
    async (method) => {
      await act(async () => toast[method](title, message));
      const notification = document.querySelector(".alert")!;
      expect(toast.toasts[0].title).toBe(title);
      expect(toast.toasts[0].message).toBe(message);
      expect(notification.textContent).toBe("SuccessDone");

      await act(async () => i18n.activate("ru"));

      expect(document.querySelector(".alert")).toBe(notification);
      expect(notification.textContent).toBe("УспехГотово");
    },
  );

  it("preserves raw strings without looking them up in catalogs", async () => {
    await act(async () => {
      toast.showToast({
        type: "error",
        title: "Error",
        message: "API details",
        duration: 0,
      });
    });
    const notification = document.querySelector(".alert")!;

    await act(async () => i18n.activate("ru"));

    expect(notification.textContent).toBe("ErrorAPI details");
  });

  it("retranslates the close button", async () => {
    await act(async () => {
      toast.showToast({ type: "success", title, duration: 0 });
    });
    const close = document.querySelector<HTMLButtonElement>(".alert button")!;
    expect(close.type).toBe("button");
    expect(close.title).toBe("Close notification");

    await act(async () => i18n.activate("ru"));

    expect(close.title).toBe("Закрыть уведомление");
  });

  it("closes a toast after its duration, once the leaving animation is over", async () => {
    vi.useFakeTimers();
    await act(async () => {
      toast.showToast({ type: "success", title: "Saved", duration: 1000 });
    });
    expect(document.querySelector(".alert")).not.toBeNull();
    await act(async () => vi.advanceTimersByTime(1000 + 400));
    expect(document.querySelector(".alert")).toBeNull();
    vi.useRealTimers();
  });
});
