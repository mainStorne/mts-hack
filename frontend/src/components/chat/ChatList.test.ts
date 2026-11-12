import { mount } from "@/test/render";
import { act, createElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatList } from "./ChatList";

const api = vi.hoisted(() => ({
  listChats: vi.fn(),
  getChat: vi.fn(),
  renameChat: vi.fn(),
  deleteChat: vi.fn(),
  openChat: vi.fn(),
  forgetChat: vi.fn(),
  showError: vi.fn(),
}));
vi.mock("@/api/chat/api", () => ({
  listChats: api.listChats,
  getChat: api.getChat,
  renameChat: api.renameChat,
  deleteChat: api.deleteChat,
}));
vi.mock("./chat-session", () => ({
  useChatSession: () => ({ chatId: "chat-1" }),
  openChat: api.openChat,
  forgetChat: api.forgetChat,
}));
vi.mock("@/components/toast", () => ({
  useToast: () => ({ showError: api.showError }),
}));

const chats = [
  { chat_id: "chat-1", title: "Data left", created_at: "", updated_at: "" },
  { chat_id: "chat-2", title: "Tariff", created_at: "", updated_at: "" },
];
const settle = () =>
  act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });

describe("chat list", () => {
  let view: ReturnType<typeof mount>;
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    Object.values(api).forEach((mock) => mock.mockReset());
  });
  afterEach(() => {
    view.unmount();
    vi.unstubAllGlobals();
  });
  const text = () => view.container.textContent;

  it("recovers from a failed load with the detail of the server and a retry", async () => {
    const { ApiError } = await import("@/api/helpers/errors");
    api.listChats
      .mockRejectedValueOnce(new ApiError(503, "Assistant is unavailable"))
      .mockResolvedValueOnce(chats);
    view = mount(createElement(ChatList));
    await settle();
    expect(text()).toContain("Assistant is unavailable");

    act(() =>
      view.container.querySelector<HTMLButtonElement>("button")!.click(),
    );
    await settle();
    expect(text()).toContain("Data left");
    expect(text()).toContain("Tariff");
    expect(text()).not.toContain("unavailable");
    expect(api.listChats).toHaveBeenCalledTimes(2);
  });

  it("says when there are no chats", async () => {
    api.listChats.mockResolvedValueOnce([]);
    view = mount(createElement(ChatList));
    await settle();
    expect(text()).toContain("No chats yet");
  });

  it("opens a chat on a click and tells the page it was selected", async () => {
    api.listChats.mockResolvedValueOnce(chats);
    const onSelect = vi.fn();
    view = mount(createElement(ChatList, { onSelect }));
    await settle();
    const second = [...view.container.querySelectorAll("button")].find(
      (button) => button.textContent === "Tariff",
    )!;
    act(() => second.click());
    expect(api.openChat).toHaveBeenCalledWith("chat-2");
    expect(onSelect).toHaveBeenCalledOnce();
  });

  it("deletes a confirmed chat and forgets it", async () => {
    api.listChats.mockResolvedValueOnce(chats);
    api.deleteChat.mockResolvedValueOnce(undefined);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    view = mount(createElement(ChatList));
    await settle();
    const bin = view.container.querySelector<HTMLButtonElement>(
      'button[title="Delete"]',
    )!;
    act(() => bin.click());
    await settle();
    expect(confirm).toHaveBeenCalledOnce();
    expect(api.deleteChat).toHaveBeenCalledWith("chat-1");
    expect(api.forgetChat).toHaveBeenCalledWith("chat-1");
    expect(text()).not.toContain("Data left");
    expect(text()).toContain("Tariff");
    confirm.mockRestore();
  });

  it("does not delete a chat the user did not confirm", async () => {
    api.listChats.mockResolvedValueOnce(chats);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    view = mount(createElement(ChatList));
    await settle();
    act(() =>
      view.container
        .querySelector<HTMLButtonElement>('button[title="Delete"]')!
        .click(),
    );
    expect(api.deleteChat).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("renames a chat when the name is saved", async () => {
    api.listChats.mockResolvedValueOnce(chats);
    api.renameChat.mockResolvedValueOnce({ ...chats[0], title: "Internet" });
    view = mount(createElement(ChatList));
    await settle();
    act(() =>
      view.container
        .querySelector<HTMLButtonElement>('button[title="Rename"]')!
        .click(),
    );
    const input = view.container.querySelector<HTMLInputElement>("input")!;
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!;
      setter.call(input, "  Internet ");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    act(() => {
      input.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
      input.blur();
    });
    await settle();
    expect(api.renameChat).toHaveBeenCalledWith("chat-1", "Internet");
    expect(text()).toContain("Internet");
  });
});
