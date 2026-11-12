import type { Answer, SessionState } from "@/api/chat/types";
import { sseStream } from "@/test/render";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  openChatStream: vi.fn(),
  openDecisionStream: vi.fn(),
  cancelChat: vi.fn(),
  history: vi.fn(),
  invalidate: vi.fn(),
}));
vi.mock("@/api/chat/api", () => ({
  openChatStream: api.openChatStream,
  openDecisionStream: api.openDecisionStream,
  cancelChat: api.cancelChat,
  listChats: vi.fn(),
  getChat: vi.fn(),
}));
vi.mock("@/app/query-client", () => ({
  queryClient: { fetchQuery: api.history, invalidateQueries: api.invalidate },
}));

const state: SessionState = {
  active_agent: "triage",
  verified: false,
  customer_name: null,
  ticket_id: null,
  escalated: false,
};
const answer: Answer = {
  answer_id: "answer-1",
  chat_id: "chat-1",
  query: "How much data is left?",
  answer: "You have **5 GB** left.",
  state,
  handoffs: [],
  approval: null,
  feedback: null,
  feedback_comment: null,
};

describe("chat session", () => {
  let session: typeof import("./chat-session");
  let snapshot: ReturnType<typeof session.useChatSession>;
  let root: Root;
  let container: HTMLDivElement;

  beforeEach(async () => {
    vi.resetModules();
    sessionStorage.clear();
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    Object.values(api).forEach((mock) => mock.mockReset());
    api.cancelChat.mockResolvedValue(undefined);
    session = await import("./chat-session");
    function Probe() {
      snapshot = session.useChatSession();
      return null;
    }
    container = document.createElement("div");
    root = createRoot(container);
    act(() => root.render(createElement(Probe)));
  });
  afterEach(() => {
    act(() => root.unmount());
    vi.unstubAllGlobals();
  });

  it("streams the chat, the status and the whole answer so far into the session", async () => {
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => (finish = resolve));
    const encoder = new TextEncoder();
    api.openChatStream.mockResolvedValueOnce(
      new ReadableStream<Uint8Array>({
        async start(controller) {
          const send = (name: string, data: unknown) =>
            controller.enqueue(
              encoder.encode(
                `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`,
              ),
            );
          send("chat", { chat_id: "chat-1" });
          send("status", { text: "Checking your balance" });
          send("answer", { text: "You have" });
          await gate;
          send("answer", { text: "You have **5 GB**" });
          send("answer", { text: "" });
          send("answer", { text: "You have **5 GB** left." });
          send("done", answer);
          controller.close();
        },
      }),
    );

    let asking!: Promise<void>;
    act(() => {
      asking = session.askQuestion(answer.query);
    });
    await vi.waitFor(() => expect(snapshot.answerText).toBe("You have"));
    expect(snapshot.chatId).toBe("chat-1");
    expect(snapshot.status).toBe("Checking your balance");
    expect(snapshot.isPending).toBe(true);
    expect(api.openChatStream).toHaveBeenCalledWith(
      answer.query,
      null,
      expect.any(AbortSignal),
    );

    await act(async () => {
      finish();
      await asking;
    });
    expect(snapshot.isPending).toBe(false);
    expect(snapshot.error).toBeNull();
    expect(snapshot.messages).toEqual([
      { role: "user", content: answer.query },
      { role: "assistant", answer },
    ]);
    expect(api.invalidate).toHaveBeenCalled();
  });

  it("reads events that end their lines with CRLF and the last one without a blank line", async () => {
    api.openChatStream.mockResolvedValueOnce(
      sseStream([["chat", { chat_id: "chat-1" }]], { separator: "\r\n" }),
    );
    api.openChatStream.mockResolvedValueOnce(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              `event: done\r\ndata: ${JSON.stringify(answer)}`,
            ),
          );
          controller.close();
        },
      }),
    );
    await act(async () => session.askQuestion("First"));
    expect(snapshot.error?.message).toBe(
      "The connection ended before the response was complete.",
    );
    expect(snapshot.chatId).toBe("chat-1");
    await act(async () => session.askQuestion("Second"));
    expect(snapshot.error).toBeNull();
    expect(snapshot.messages.at(-1)).toEqual({ role: "assistant", answer });
  });

  it("shows a handoff as soon as the agent event comes, and the handoffs of the answer after it", async () => {
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => (finish = resolve));
    const encoder = new TextEncoder();
    const handedOff: Answer = {
      ...answer,
      state: { ...state, active_agent: "billing", verified: true },
      handoffs: [
        {
          from_agent: "triage",
          to_agent: "billing",
          reason: "A bill question",
        },
      ],
    };
    api.openChatStream.mockResolvedValueOnce(
      new ReadableStream<Uint8Array>({
        async start(controller) {
          const send = (name: string, data: unknown) =>
            controller.enqueue(
              encoder.encode(
                `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`,
              ),
            );
          // The first agent is not a handoff
          send("agent", { name: "triage", label: "Triage" });
          send("agent", { name: "billing", label: "Billing agent" });
          await gate;
          send("done", handedOff);
          controller.close();
        },
      }),
    );

    let asking!: Promise<void>;
    act(() => {
      asking = session.askQuestion("Why was I charged?");
    });
    await vi.waitFor(() => expect(snapshot.liveHandoffs).toHaveLength(1));
    expect(snapshot.liveHandoffs).toEqual([
      { from: "triage", to: "billing", label: "Billing agent" },
    ]);
    expect(snapshot.state.active_agent).toBe("billing");
    expect(snapshot.agentLabel).toBe("Billing agent");

    await act(async () => {
      finish();
      await asking;
    });
    expect(snapshot.liveHandoffs).toEqual([]);
    expect(snapshot.agentLabel).toBeNull();
    expect(snapshot.state).toEqual(handedOff.state);
    const last = snapshot.messages.at(-1);
    expect(last?.role === "assistant" && last.answer.handoffs).toEqual(
      handedOff.handoffs,
    );
  });

  it("marks the conversation as escalated when the human takes it over", async () => {
    api.openChatStream.mockResolvedValueOnce(
      sseStream([["agent", { name: "human", label: "Operator" }]]),
    );
    await act(async () => session.askQuestion("I want a human"));
    // The stream ended without an answer
    expect(snapshot.state.active_agent).toBe("human");
    expect(snapshot.state.escalated).toBe(true);
  });

  it("gives the question back when the stream reports an error", async () => {
    api.openChatStream.mockResolvedValueOnce(
      sseStream([
        ["chat", { chat_id: "chat-1" }],
        ["error", { detail: "Assistant unavailable" }],
      ]),
    );
    await act(async () => session.askQuestion("Hello"));
    expect(snapshot.error?.message).toBe("Assistant unavailable");
    expect(snapshot.draftQuery).toBe("Hello");
    expect(snapshot.isPending).toBe(false);
    // The failed question is not in the chat when the next one is asked
    api.openChatStream.mockResolvedValueOnce(sseStream([["done", answer]]));
    await act(async () => session.askQuestion("Hello again"));
    expect(snapshot.messages).toEqual([
      { role: "user", content: "Hello again" },
      { role: "assistant", answer },
    ]);
    expect(snapshot.error).toBeNull();
  });

  it("surfaces a failed request", async () => {
    api.openChatStream.mockRejectedValueOnce(
      new Error("Assistant unavailable"),
    );
    await act(async () => session.askQuestion("Hello"));
    expect(snapshot.error?.message).toBe("Assistant unavailable");
    expect(snapshot.messages).toEqual([{ role: "user", content: "Hello" }]);
  });

  it("takes a cancelled question back out of the chat", async () => {
    api.openChatStream.mockResolvedValueOnce(
      sseStream([["cancelled", { detail: "Cancelled" }]]),
    );
    await act(async () => session.askQuestion("Never mind"));
    expect(snapshot.messages).toEqual([]);
    expect(snapshot.draftQuery).toBe("Never mind");
    expect(snapshot.isPending).toBe(false);
    expect(snapshot.error).toBeNull();
  });

  it("stops a running question on the server and gives it back", async () => {
    const encoder = new TextEncoder();
    api.openChatStream.mockResolvedValueOnce(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            encoder.encode(`event: chat\ndata: {"chat_id":"chat-1"}\n\n`),
          );
        },
      }),
    );
    let asking!: Promise<void>;
    act(() => {
      asking = session.askQuestion("Long question");
    });
    await vi.waitFor(() => expect(snapshot.chatId).toBe("chat-1"));
    let given = "";
    act(() => {
      given = session.cancelQuestion();
    });
    await act(async () => asking);
    expect(given).toBe("Long question");
    expect(api.cancelChat).toHaveBeenCalledWith("chat-1");
    expect(api.openChatStream.mock.calls[0][2].aborted).toBe(true);
    expect(snapshot.isPending).toBe(false);
    expect(snapshot.messages).toEqual([]);
    expect(snapshot.draftQuery).toBe("Long question");
  });

  it("keeps the conversation in sessionStorage but not what is being written", async () => {
    api.openChatStream.mockResolvedValueOnce(sseStream([["done", answer]]));
    await act(async () => session.askQuestion(answer.query));
    act(() => session.setDraftQuery("Next"));
    const saved = JSON.parse(sessionStorage.getItem("support-agent-session")!);
    expect(saved.chatId).toBe("chat-1");
    expect(saved.draftQuery).toBe("Next");
    expect(saved.messages).toHaveLength(2);
    expect(saved).not.toHaveProperty("answerText");

    vi.resetModules();
    const restored = await import("./chat-session");
    let reloaded!: ReturnType<typeof restored.useChatSession>;
    const reloadedRoot = createRoot(document.createElement("div"));
    act(() =>
      reloadedRoot.render(
        createElement(() => {
          reloaded = restored.useChatSession();
          return null;
        }),
      ),
    );
    expect(reloaded.chatId).toBe("chat-1");
    expect(reloaded.messages).toHaveLength(2);
    expect(reloaded.draftQuery).toBe("Next");
    act(() => reloadedRoot.unmount());
  });

  it("starts a new conversation without the agent state of the old one", async () => {
    api.openChatStream.mockResolvedValueOnce(
      sseStream([
        [
          "done",
          {
            ...answer,
            state: { ...state, active_agent: "tech", verified: true },
          },
        ],
      ]),
    );
    await act(async () => session.askQuestion(answer.query));
    expect(snapshot.state.active_agent).toBe("tech");
    const composerId = snapshot.composerId;
    act(() => session.startNewChat());
    expect(snapshot.messages).toEqual([]);
    expect(snapshot.chatId).toBeNull();
    expect(snapshot.state).toEqual(state);
    expect(snapshot.composerId).toBe(composerId + 1);
  });

  it("sends the decision and replaces the approval with the answer that waited for it", async () => {
    const waiting: Answer = {
      ...answer,
      answer: "Switch to the Smart tariff?",
      approval: {
        id: "approval-1",
        actions: [{ name: "switch_tariff", description: "Switch to Smart" }],
      },
    };
    api.history.mockResolvedValueOnce({
      ...waiting,
      state,
      answers: [],
      waiting,
    });
    await act(async () => session.openChat("chat-1"));
    expect(snapshot.messages).toEqual([
      { role: "user", content: waiting.query },
      { role: "assistant", answer: waiting },
    ]);

    const done: Answer = {
      ...answer,
      answer: "Done: the tariff is switched.",
      state: { ...state, active_agent: "tariffs" },
    };
    api.openDecisionStream.mockResolvedValueOnce(
      sseStream([
        ["status", { text: "Switching" }],
        ["done", done],
      ]),
    );
    await act(async () => session.decideApproval("approve"));
    expect(api.openDecisionStream).toHaveBeenCalledWith(
      "chat-1",
      "approve",
      expect.any(AbortSignal),
    );
    expect(snapshot.messages).toEqual([
      { role: "user", content: waiting.query },
      { role: "assistant", answer: done },
    ]);
    expect(snapshot.state.active_agent).toBe("tariffs");

    // Nothing waits for a decision any more
    await act(async () => session.decideApproval("reject"));
    expect(api.openDecisionStream).toHaveBeenCalledTimes(1);
  });

  it("opens a chat with its state, retries a failed opening and ignores duplicates", async () => {
    api.history.mockRejectedValueOnce(new Error("History unavailable"));
    await act(async () => session.openChat("chat-1"));
    expect(snapshot.openingError).toEqual(new Error("History unavailable"));
    act(() => session.setDraftQuery("Keep my draft"));
    const composerId = snapshot.composerId;

    api.history.mockResolvedValueOnce({
      chat_id: "chat-1",
      title: "Data",
      created_at: "",
      updated_at: "",
      state: { ...state, active_agent: "billing", verified: true },
      answers: [answer],
      waiting: null,
    });
    await act(async () => session.openChat("chat-1"));
    expect(snapshot.openingError).toBeNull();
    expect(snapshot.state.active_agent).toBe("billing");
    expect(snapshot.messages).toEqual([
      { role: "user", content: answer.query },
      { role: "assistant", answer },
    ]);
    expect(snapshot.draftQuery).toBe("Keep my draft");
    expect(snapshot.composerId).toBe(composerId);

    await act(async () => session.openChat("chat-1"));
    expect(api.history).toHaveBeenCalledTimes(2);
  });

  it("remembers the vote on an answer", async () => {
    api.openChatStream.mockResolvedValueOnce(sseStream([["done", answer]]));
    await act(async () => session.askQuestion(answer.query));
    act(() => session.rememberFeedback("answer-1", "dislike", "Wrong"));
    const last = snapshot.messages.at(-1);
    expect(last?.role === "assistant" && last.answer).toMatchObject({
      feedback: "dislike",
      feedback_comment: "Wrong",
    });
  });

  it("forgets a deleted chat only when it is the open one", async () => {
    api.openChatStream.mockResolvedValueOnce(sseStream([["done", answer]]));
    await act(async () => session.askQuestion(answer.query));
    act(() => session.forgetChat("another"));
    expect(snapshot.chatId).toBe("chat-1");
    act(() => session.forgetChat("chat-1"));
    expect(snapshot.chatId).toBeNull();
    expect(snapshot.messages).toEqual([]);
  });
});

describe("parseEvent", () => {
  it("reads the name and the data of an event and skips comments and broken ones", async () => {
    const { parseEvent } = await import("./chat-session");
    expect(parseEvent(`event: status\ndata: {"text":"Hi"}`)).toEqual({
      name: "status",
      data: { text: "Hi" },
    });
    expect(parseEvent(`: ping`)).toBeUndefined();
    expect(parseEvent(`event: status\ndata: {broken`)).toBeUndefined();
    expect(parseEvent(`event: answer\ndata: {"text":\ndata: "multi"}`)).toEqual(
      { name: "answer", data: { text: "multi" } },
    );
  });
});
