import type { Answer } from "@/api/chat/types";
import { mount, typeInto } from "@/test/render";
import { act, createElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssistantMessage } from "./ChatMessages";

const api = vi.hoisted(() => ({
  sendFeedback: vi.fn(),
  rememberFeedback: vi.fn(),
  decideApproval: vi.fn(),
  showSuccess: vi.fn(),
  showError: vi.fn(),
}));
vi.mock("@/api/chat/api", () => ({ sendFeedback: api.sendFeedback }));
vi.mock("./chat-session", () => ({
  rememberFeedback: api.rememberFeedback,
  decideApproval: api.decideApproval,
}));
vi.mock("@/components/toast", () => ({
  useToast: () => ({
    showSuccess: api.showSuccess,
    showError: api.showError,
  }),
}));

const answer: Answer = {
  answer_id: "answer-1",
  chat_id: "chat-1",
  query: "Why was I charged?",
  answer: "It is the monthly fee.",
  state: {
    active_agent: "billing",
    verified: true,
    customer_name: "Ivan",
    ticket_id: null,
    escalated: false,
  },
  handoffs: [],
  approval: null,
  feedback: null,
  feedback_comment: null,
};
const settle = () => act(async () => {});

describe("answer feedback", () => {
  let view: ReturnType<typeof mount>;
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    Object.values(api).forEach((mock) => mock.mockReset());
    api.sendFeedback.mockResolvedValue(undefined);
  });
  afterEach(() => {
    view.unmount();
    vi.unstubAllGlobals();
  });
  function render(data: Answer = answer) {
    view = mount(
      createElement(AssistantMessage, {
        answer: data,
        isLast: true,
        isPending: false,
      }),
    );
  }
  const votes = () =>
    [...view.container.querySelectorAll<HTMLButtonElement>("button")].filter(
      (button) => ["👍", "👎"].includes(button.textContent ?? ""),
    );

  it("sends a vote, then the comment with the same vote", async () => {
    render();
    expect(votes()).toHaveLength(2);
    act(() => votes()[0].click());
    await settle();
    expect(api.sendFeedback).toHaveBeenLastCalledWith(
      "chat-1",
      "answer-1",
      "like",
    );
    expect(api.rememberFeedback).toHaveBeenLastCalledWith(
      "answer-1",
      "like",
      null,
    );

    typeInto(
      view.container.querySelector<HTMLTextAreaElement>("textarea")!,
      "  Clear and useful  ",
    );
    const send = [...view.container.querySelectorAll("button")].find(
      (button) => button.textContent === "Send",
    )!;
    act(() => send.click());
    await settle();
    expect(api.sendFeedback).toHaveBeenLastCalledWith(
      "chat-1",
      "answer-1",
      "like",
      "Clear and useful",
    );
    expect(api.rememberFeedback).toHaveBeenLastCalledWith(
      "answer-1",
      "like",
      "Clear and useful",
    );
    expect(view.container.querySelector("textarea")).toBeNull();
  });

  it("takes a vote back with a click on the chosen one", async () => {
    render({ ...answer, feedback: "dislike" });
    expect(votes()[1].getAttribute("aria-pressed")).toBe("true");
    act(() => votes()[1].click());
    await settle();
    expect(api.sendFeedback).toHaveBeenLastCalledWith(
      "chat-1",
      "answer-1",
      null,
    );
    expect(votes()[1].getAttribute("aria-pressed")).toBe("false");
    expect(view.container.querySelector("textarea")).toBeNull();
  });

  it("shows the vote and the comment given before without the comment form", () => {
    render({
      ...answer,
      feedback: "dislike",
      feedback_comment: "The summary missed a key detail.",
    });
    expect(votes()[1].getAttribute("aria-pressed")).toBe("true");
    expect(view.container.querySelector("textarea")).toBeNull();
  });

  it("tells when a vote could not be saved", async () => {
    api.sendFeedback.mockRejectedValueOnce(new Error("Offline"));
    render();
    act(() => votes()[0].click());
    await settle();
    expect(api.showError).toHaveBeenCalledOnce();
    expect(votes()[0].getAttribute("aria-pressed")).toBe("false");
  });

  it("asks for a decision instead of a vote while an approval waits", () => {
    render({
      ...answer,
      approval: {
        id: "approval-1",
        actions: [{ name: "switch_tariff", description: "Switch to Smart" }],
      },
    });
    expect(votes()).toHaveLength(0);
    expect(view.container.textContent).toContain("Switch to Smart");
    const buttons = [...view.container.querySelectorAll("button")];
    act(() => buttons.find((b) => b.textContent === "Approve")!.click());
    act(() => buttons.find((b) => b.textContent === "Reject")!.click());
    expect(api.decideApproval.mock.calls).toEqual([["approve"], ["reject"]]);
  });
});
