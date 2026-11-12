import type { Answer } from "@/api/chat/types";
import { mount } from "@/test/render";
import { createElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AssistantMessage,
  ErrorMessage,
  StreamingMessage,
} from "./ChatMessages";

vi.mock("./chat-session", () => ({
  rememberFeedback: vi.fn(),
  decideApproval: vi.fn(),
}));
vi.mock("@/api/chat/api", () => ({ sendFeedback: vi.fn() }));
vi.mock("@/components/toast", () => ({
  useToast: () => ({ showSuccess: vi.fn(), showError: vi.fn() }),
}));

const answer: Answer = {
  answer_id: "answer-1",
  chat_id: "chat-1",
  query: "Why was I charged?",
  answer: "You have **5 GB** left.\n\n| a | b |\n|---|---|\n| 1 | 2 |",
  state: {
    active_agent: "billing",
    verified: true,
    customer_name: null,
    ticket_id: null,
    escalated: false,
  },
  handoffs: [
    {
      from_agent: "triage",
      to_agent: "billing",
      reason: "A question on a bill",
    },
    { from_agent: "billing", to_agent: "human", reason: "Needs a person" },
  ],
  approval: null,
  feedback: null,
  feedback_comment: null,
};

describe("messages", () => {
  let view: ReturnType<typeof mount>;
  beforeEach(() => vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true));
  afterEach(() => {
    view.unmount();
    vi.unstubAllGlobals();
  });

  it("puts the handoff markers of an answer before it and renders its Markdown", () => {
    view = mount(
      createElement(AssistantMessage, {
        answer,
        isLast: true,
        isPending: false,
      }),
    );
    const text = view.container.textContent ?? "";
    expect(text).toContain("Billing agent");
    expect(text).toContain("Human operator");
    expect(text.indexOf("Billing agent")).toBeLessThan(text.indexOf("5 GB"));
    expect(view.container.querySelector("strong")?.textContent).toBe("5 GB");
    expect(view.container.querySelector("table")).not.toBeNull();
    expect(
      view.container.querySelector('[title="A question on a bill"]'),
    ).not.toBeNull();
  });

  it("shows the markers and the status of an answer that is being written", () => {
    view = mount(
      createElement(StreamingMessage, {
        answerText: "Looking at your **bill**",
        status: "Checking the bill",
        handoffs: [{ from: "triage", to: "billing", label: "Billing" }],
      }),
    );
    const text = view.container.textContent ?? "";
    expect(text).toContain("Billing");
    expect(text).toContain("Checking the bill");
    expect(view.container.querySelector("strong")?.textContent).toBe("bill");
  });

  it("says that it is sending the request when nothing came yet", () => {
    view = mount(
      createElement(StreamingMessage, {
        answerText: "",
        status: "",
        handoffs: [],
      }),
    );
    expect(view.container.textContent).toContain("Sending request");
  });

  it("shows the error with the hint about the draft", () => {
    view = mount(
      createElement(ErrorMessage, {
        error: new Error("Assistant is unavailable"),
        hasDraft: true,
        hasApproval: false,
      }),
    );
    expect(view.container.textContent).toContain("Assistant is unavailable");
    expect(view.container.textContent).toContain("Your draft is below");
  });
});
