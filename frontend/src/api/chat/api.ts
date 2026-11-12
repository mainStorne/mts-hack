import { ApiError } from "@/api/helpers/errors";
import { API_URL } from "./config";
import { getClientId } from "./client-id";
import type { Chat, ChatHistory, Decision, Feedback } from "./types";

type RequestOptions = {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  stream?: boolean;
};

async function request(path: string, options: RequestOptions = {}) {
  const { method = "GET", body, signal, stream = false } = options;
  const headers: Record<string, string> = {
    "X-Client-Id": getClientId(),
    Accept: stream ? "text/event-stream" : "application/json",
  };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await ApiError.fromResponse(response);
  return response;
}

async function json<T>(path: string, options?: RequestOptions): Promise<T> {
  const response = await request(path, options);
  return (await response.json()) as T;
}

async function openStream(path: string, body: unknown, signal: AbortSignal) {
  const response = await request(path, {
    method: "POST",
    body,
    signal,
    stream: true,
  });
  return response.body;
}

// The answer to a question as a stream of events
export function openChatStream(
  query: string,
  chatId: string | null,
  signal: AbortSignal,
) {
  return openStream("/chats/stream", { query, chat_id: chatId }, signal);
}

// The answer to the question that waited for the user's decision
export function openDecisionStream(
  chatId: string,
  decision: Decision,
  signal: AbortSignal,
) {
  return openStream(
    `/chats/${encodeURIComponent(chatId)}/decision/stream`,
    { decision },
    signal,
  );
}

export async function cancelChat(chatId: string): Promise<void> {
  await request(`/chats/${encodeURIComponent(chatId)}/cancel`, {
    method: "POST",
  });
}

export const listChats = () => json<Chat[]>("/chats");

export const getChat = (chatId: string) =>
  json<ChatHistory>(`/chats/${encodeURIComponent(chatId)}`);

export const renameChat = (chatId: string, title: string) =>
  json<Chat>(`/chats/${encodeURIComponent(chatId)}`, {
    method: "PATCH",
    body: { title },
  });

export async function deleteChat(chatId: string): Promise<void> {
  await request(`/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
}

export async function sendFeedback(
  chatId: string,
  answerId: string,
  feedback: Feedback | null,
  comment?: string | null,
): Promise<void> {
  await request(
    `/chats/${encodeURIComponent(chatId)}/answers/${encodeURIComponent(answerId)}/feedback`,
    { method: "POST", body: { feedback, comment: comment ?? null } },
  );
}
