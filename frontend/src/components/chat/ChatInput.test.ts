import { mount, typeInto } from "@/test/render";
import { act, createElement, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "./ChatInput";

describe("ChatInput", () => {
  const send = vi.fn();
  const stop = vi.fn();
  let view: ReturnType<typeof mount>;

  function Composer({ isPending = false }: { isPending?: boolean }) {
    const [text, setText] = useState("");
    return createElement(ChatInput, {
      text,
      setText,
      onSend: send,
      onStop: stop,
      isPending,
    });
  }
  const textarea = () => view.container.querySelector("textarea")!;
  const button = (title: string) =>
    view.container.querySelector<HTMLButtonElement>(
      `button[title="${title}"]`,
    )!;
  function press(key: string, init: KeyboardEventInit = {}) {
    const event = new KeyboardEvent("keydown", {
      key,
      bubbles: true,
      cancelable: true,
      ...init,
    });
    act(() => {
      textarea().dispatchEvent(event);
    });
    return event;
  }

  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    send.mockReset();
    stop.mockReset();
  });
  afterEach(() => {
    view.unmount();
    vi.unstubAllGlobals();
  });

  it("sends the trimmed question on Enter and keeps Shift+Enter for a new line", () => {
    view = mount(createElement(Composer));
    typeInto(textarea(), "  Hello  ");

    const newLine = press("Enter", { shiftKey: true });
    expect(newLine.defaultPrevented).toBe(false);
    expect(send).not.toHaveBeenCalled();

    const enter = press("Enter");
    expect(enter.defaultPrevented).toBe(true);
    expect(send).toHaveBeenCalledExactlyOnceWith("Hello");
  });

  it("does not send an empty question or while the answer is written", () => {
    view = mount(createElement(Composer));
    expect(button("Send").disabled).toBe(true);
    typeInto(textarea(), "   ");
    press("Enter");
    expect(send).not.toHaveBeenCalled();
    typeInto(textarea(), "Hello");
    expect(button("Send").disabled).toBe(false);
    act(() => button("Send").click());
    expect(send).toHaveBeenCalledWith("Hello");
  });

  it("offers to stop while the answer is written", () => {
    view = mount(createElement(Composer, { isPending: true }));
    expect(view.container.querySelector('button[title="Send"]')).toBeNull();
    typeInto(textarea(), "Hello");
    press("Enter");
    expect(send).not.toHaveBeenCalled();
    act(() => button("Stop").click());
    expect(stop).toHaveBeenCalledOnce();
  });
});
