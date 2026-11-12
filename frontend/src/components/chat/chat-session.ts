import { cancelChat, openChatStream, openDecisionStream } from "@/api/chat/api";
import { chatQueryOptions, chatsQueryOptions } from "@/api/chat/queries";
import {
  INITIAL_STATE,
  type AgentName,
  type Answer,
  type Decision,
  type Feedback,
  type SessionState,
  type StreamEvent,
} from "@/api/chat/types";
import { queryClient } from "@/app/query-client";
import { i18n } from "@lingui/core";
import { msg } from "@lingui/core/macro";
import { useSyncExternalStore } from "react";

export type ChatMessage =
  { role: "user"; content: string } | { role: "assistant"; answer: Answer };

// The agent that took the conversation over while the answer is being written
export type LiveHandoff = { from: AgentName; to: AgentName; label: string };

type ChatSession = {
  messages: ChatMessage[];
  chatId: string | null;
  // What the agent knows about the conversation: who answers, whether the customer is verified, ...
  state: SessionState;
  // The name of the active agent as the server wrote it, while the answer is being written
  agentLabel: string | null;
  draftQuery: string;
  composerId: number;
  // The answer being written and what the agent is doing meanwhile
  answerText: string;
  status: string;
  liveHandoffs: LiveHandoff[];
  isPending: boolean;
  error: Error | null;
  // A chat being opened from the list of the user's chats, and why it could not be
  isOpening: boolean;
  openingError: unknown;
};

// The conversation lives outside the page: the questions and the answers are kept in this tab (sessionStorage), so a
// reload keeps them too; an answer that was being written is lost with the page. The server keeps every chat of the
// user as well: one of them opens again from the list of chats (`openChat`)
const STORAGE_KEY = "support-agent-session";

type SavedSession = Pick<
  ChatSession,
  "messages" | "chatId" | "state" | "draftQuery"
>;

function loadSaved(): Partial<SavedSession> {
  try {
    const saved: unknown = JSON.parse(
      sessionStorage.getItem(STORAGE_KEY) ?? "null",
    );
    if (typeof saved !== "object" || saved === null) return {};
    const { messages, chatId, state, draftQuery } = saved as SavedSession;
    return {
      messages: Array.isArray(messages) ? messages : [],
      chatId: typeof chatId === "string" ? chatId : null,
      state: state ? { ...INITIAL_STATE, ...state } : INITIAL_STATE,
      draftQuery: typeof draftQuery === "string" ? draftQuery : "",
    };
  } catch {
    return {};
  }
}

let session: ChatSession = {
  messages: [],
  chatId: null,
  state: INITIAL_STATE,
  draftQuery: "",
  ...loadSaved(),
  agentLabel: null,
  composerId: 0,
  answerText: "",
  status: "",
  liveHandoffs: [],
  isPending: false,
  error: null,
  isOpening: false,
  openingError: null,
};
let abortController: AbortController | null = null;
const listeners = new Set<() => void>();

function update(patch: Partial<ChatSession>) {
  session = { ...session, ...patch };
  persist();
  listeners.forEach((listener) => listener());
}

function persist() {
  const saved: SavedSession = {
    messages: session.messages,
    chatId: session.chatId,
    state: session.state,
    draftQuery: session.draftQuery,
  };
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
  } catch {
    // Storage is full or blocked: the conversation just does not survive a reload
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useChatSession() {
  return useSyncExternalStore(subscribe, () => session);
}

// For code that must see the newest state, not the one of the last render
export function getChatId() {
  return session.chatId;
}

export function setDraftQuery(draftQuery: string) {
  update({ draftQuery });
}

function invalidateChats() {
  void queryClient.invalidateQueries({
    queryKey: chatsQueryOptions().queryKey,
    exact: true,
  });
}

export function parseEvent(block: string): StreamEvent | undefined {
  let name: string | undefined;
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:"))
      data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!name || data.length === 0) return undefined;
  try {
    return { name, data: JSON.parse(data.join("\n")) } as StreamEvent;
  } catch {
    return undefined;
  }
}

function applyEvent(event: StreamEvent) {
  if (event.name === "chat") {
    update({ chatId: event.data.chat_id });
    // The chat already exists even if generation fails or is cancelled
    invalidateChats();
  } else if (event.name === "status") {
    update({ status: event.data.text });
  } else if (event.name === "answer") {
    // The whole answer so far replaces the previous one; an empty text resets it
    update({ answerText: event.data.text });
  } else if (event.name === "agent") {
    const { name, label } = event.data;
    const from = session.state.active_agent;
    update({
      state: {
        ...session.state,
        active_agent: name,
        escalated: session.state.escalated || name === "human",
      },
      agentLabel: label,
      // The agent that is told again is not a handoff
      liveHandoffs:
        from === name
          ? session.liveHandoffs
          : [...session.liveHandoffs, { from, to: name, label }],
    });
  }
}

// The final answer, or undefined when the question was cancelled
async function readAnswer(
  stream: ReadableStream<BufferSource> | null,
  signal: AbortSignal,
): Promise<Answer | undefined> {
  if (!stream) {
    throw new Error(i18n._(msg`The server returned no answer stream.`));
  }
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  // A cancelled question closes the stream even when no event is coming
  const stop = () => void reader.cancel().catch(() => undefined);
  signal.addEventListener("abort", stop, { once: true });
  try {
    return await readEvents(reader, signal);
  } finally {
    signal.removeEventListener("abort", stop);
  }
}

// The final answer, or undefined when the question was cancelled
async function readEvents(
  reader: ReadableStreamDefaultReader<string>,
  signal: AbortSignal,
): Promise<Answer | undefined> {
  let buffer = "";

  // The final answer, "cancelled" when the server stopped the question, nothing for the other events
  const handle = (block: string): Answer | "cancelled" | undefined => {
    const event = parseEvent(block);
    if (!event) return undefined;
    if (event.name === "cancelled") return "cancelled";
    if (event.name === "error") throw new Error(event.data.detail);
    if (event.name === "done") return event.data;
    applyEvent(event);
    return undefined;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (signal.aborted) return undefined;
    if (done) break;
    // Servers may end the lines with CRLF
    buffer = (buffer + value).replace(/\r\n/g, "\n");

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const result = handle(block);
      if (result === "cancelled") return undefined;
      if (result) return result;
    }
  }
  // The last event may come without the blank line after it
  const result = buffer.trim() ? handle(buffer) : undefined;
  if (result === "cancelled") return undefined;
  if (result) return result;
  throw new Error(
    i18n._(msg`The connection ended before the response was complete.`),
  );
}

function beginRequest() {
  abortController?.abort();
  const controller = new AbortController();
  abortController = controller;
  return controller;
}

function finishAnswer(answer: Answer) {
  update({
    chatId: answer.chat_id,
    state: answer.state,
    agentLabel: null,
    liveHandoffs: [],
  });
  invalidateChats();
}

export async function askQuestion(query: string) {
  const controller = beginRequest();

  // A question left without an answer (it failed) is not in the chat: the agent does not remember it either
  const answered =
    session.messages.at(-1)?.role === "user"
      ? session.messages.slice(0, -1)
      : session.messages;
  update({
    messages: [...answered, { role: "user", content: query }],
    draftQuery: "",
    answerText: "",
    status: "",
    liveHandoffs: [],
    error: null,
    isPending: true,
  });

  try {
    const stream = await openChatStream(
      query,
      session.chatId,
      controller.signal,
    );
    const answer = await readAnswer(stream, controller.signal);
    if (!controller.signal.aborted && !answer) {
      cancelQuestion();
      return;
    }
    if (answer && !controller.signal.aborted) {
      update({
        messages: [...session.messages, { role: "assistant", answer }],
      });
      finishAnswer(answer);
    }
  } catch (e) {
    if (!controller.signal.aborted) {
      update({ error: e instanceof Error ? e : new Error(String(e)) });
    }
  } finally {
    if (abortController === controller) {
      abortController = null;
      update({
        isPending: false,
        draftQuery: session.error
          ? session.draftQuery || query
          : session.draftQuery,
      });
    }
  }
}

// The user's decision on what the agent is about to do for them (the `approval` of the last answer): it is done only
// when approved. The answer that waited for the decision takes the place of the approval
export async function decideApproval(decision: Decision) {
  const last = session.messages.at(-1);
  const chatId = session.chatId;
  if (!chatId || last?.role !== "assistant" || !last.answer.approval) return;
  const controller = beginRequest();
  update({
    answerText: "",
    status: "",
    liveHandoffs: [],
    error: null,
    isPending: true,
  });

  try {
    const stream = await openDecisionStream(
      chatId,
      decision,
      controller.signal,
    );
    const answer = await readAnswer(stream, controller.signal);
    if (answer && !controller.signal.aborted && session.chatId === chatId) {
      update({
        messages: [
          ...session.messages.slice(0, -1),
          { role: "assistant", answer },
        ],
      });
      finishAnswer(answer);
    }
  } catch (e) {
    if (!controller.signal.aborted) {
      update({ error: e instanceof Error ? e : new Error(String(e)) });
    }
  } finally {
    if (abortController === controller) {
      abortController = null;
      update({ isPending: false });
    }
  }
}

// Stops the question on the server too. The question is not in the chat then (the agent does not remember it
// either): it is given back, to be asked again or changed
export function cancelQuestion(): string {
  const last = session.messages.at(-1);
  const question =
    session.isPending && last?.role === "user" ? last.content : "";
  const chatId = session.chatId;
  if (session.isPending && chatId) {
    // Nothing to do if the server is already done: the request only makes sure it stops
    cancelChat(chatId).catch(() => undefined);
  }
  abortController?.abort();
  abortController = null;
  update({
    isPending: false,
    liveHandoffs: [],
    agentLabel: null,
    messages: question ? session.messages.slice(0, -1) : session.messages,
    draftQuery: session.draftQuery || question,
  });
  return question;
}

// A new conversation: the agent does not see the earlier questions any more
export function startNewChat() {
  cancelQuestion();
  update({
    messages: [],
    chatId: null,
    state: INITIAL_STATE,
    agentLabel: null,
    composerId: session.composerId + 1,
    draftQuery: "",
    answerText: "",
    status: "",
    liveHandoffs: [],
    error: null,
    isOpening: false,
    openingError: null,
  });
}

// A chat the user deleted: when it is the open one, a new conversation starts in its place
export function forgetChat(chatId: string) {
  if (session.chatId === chatId) startNewChat();
}

// One of the user's chats with its questions and answers, as the server keeps them, to go on with it
export async function openChat(chatId: string) {
  const sameChat = chatId === session.chatId;
  if (
    sameChat &&
    (session.isOpening || (!session.isPending && session.openingError === null))
  ) {
    return;
  }
  if (!sameChat || session.isPending) cancelQuestion();
  update({
    messages: [],
    chatId,
    agentLabel: null,
    composerId: sameChat ? session.composerId : session.composerId + 1,
    draftQuery: sameChat ? session.draftQuery : "",
    answerText: "",
    status: "",
    liveHandoffs: [],
    error: null,
    isOpening: true,
    openingError: null,
  });

  try {
    const history = await queryClient.fetchQuery(chatQueryOptions(chatId));
    if (session.chatId !== chatId) return; // Another chat was opened meanwhile
    update({
      isOpening: false,
      state: history.state,
      // The question that waits for the user's decision comes last, with what it waits for
      messages: [
        ...history.answers,
        ...(history.waiting ? [history.waiting] : []),
      ].flatMap((answer): ChatMessage[] => [
        { role: "user", content: answer.query },
        { role: "assistant", answer },
      ]),
    });
  } catch (e) {
    if (session.chatId !== chatId) return;
    update({ isOpening: false, openingError: e });
  }
}

// The user's vote on an answer: the answer shows it when the page opens again
export function rememberFeedback(
  answerId: string,
  feedback: Feedback | null,
  comment: string | null,
) {
  update({
    messages: session.messages.map((message) =>
      message.role === "assistant" && message.answer.answer_id === answerId
        ? {
            ...message,
            answer: {
              ...message.answer,
              feedback,
              feedback_comment: comment,
            },
          }
        : message,
    ),
  });
}
