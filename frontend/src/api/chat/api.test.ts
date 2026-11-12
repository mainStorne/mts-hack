import { ApiError } from "@/api/helpers/errors";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cancelChat,
  deleteChat,
  listChats,
  openChatStream,
  openDecisionStream,
  renameChat,
  sendFeedback,
} from "./api";
import { getClientId } from "./client-id";

describe("backend client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  function lastCall() {
    const [url, init] = fetchMock.mock.calls.at(-1)!;
    return { url: url as string, init: init as RequestInit };
  }

  it("keeps one client id in localStorage and sends it with every request", async () => {
    const id = getClientId();
    expect(id).toMatch(/^[0-9a-f-]{36}$/);
    expect(localStorage.getItem("client-id")).toBe(id);
    expect(getClientId()).toBe(id);

    fetchMock.mockResolvedValueOnce(Response.json([]));
    await listChats();
    const { url, init } = lastCall();
    expect(url).toBe("http://localhost:8000/chats");
    expect(init.method).toBe("GET");
    expect((init.headers as Record<string, string>)["X-Client-Id"]).toBe(id);
  });

  it("opens the answer stream with the question and the chat", async () => {
    const body = new ReadableStream();
    fetchMock.mockResolvedValueOnce(new Response(body));
    const signal = new AbortController().signal;
    await openChatStream("Hello", "chat-1", signal);
    const { url, init } = lastCall();
    expect(url).toBe("http://localhost:8000/chats/stream");
    expect(init.method).toBe("POST");
    expect(init.signal).toBe(signal);
    expect(JSON.parse(init.body as string)).toEqual({
      query: "Hello",
      chat_id: "chat-1",
    });
    const headers = init.headers as Record<string, string>;
    expect(headers.Accept).toBe("text/event-stream");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("opens the decision stream, cancels, renames, deletes and votes", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    await openDecisionStream("a/b", "reject", new AbortController().signal);
    expect(lastCall().url).toBe(
      "http://localhost:8000/chats/a%2Fb/decision/stream",
    );
    expect(JSON.parse(lastCall().init.body as string)).toEqual({
      decision: "reject",
    });

    await cancelChat("chat-1");
    expect(lastCall().url).toBe("http://localhost:8000/chats/chat-1/cancel");
    expect(lastCall().init.method).toBe("POST");

    await deleteChat("chat-1");
    expect(lastCall().init.method).toBe("DELETE");

    fetchMock.mockResolvedValueOnce(
      Response.json({ chat_id: "chat-1", title: "New" }),
    );
    await renameChat("chat-1", "New");
    expect(lastCall().init.method).toBe("PATCH");
    expect(JSON.parse(lastCall().init.body as string)).toEqual({
      title: "New",
    });

    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    await sendFeedback("chat-1", "answer-1", "like", "Nice");
    expect(lastCall().url).toBe(
      "http://localhost:8000/chats/chat-1/answers/answer-1/feedback",
    );
    expect(JSON.parse(lastCall().init.body as string)).toEqual({
      feedback: "like",
      comment: "Nice",
    });
  });

  it("raises the detail of an error response, also of a 503", async () => {
    fetchMock.mockResolvedValueOnce(
      Response.json({ detail: "Assistant is unavailable" }, { status: 503 }),
    );
    const error = await listChats().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 503,
      detail: "Assistant is unavailable",
      message: "Assistant is unavailable",
    });

    fetchMock.mockResolvedValueOnce(new Response("oops", { status: 500 }));
    await expect(listChats()).rejects.toMatchObject({
      status: 500,
      detail: null,
    });
  });
});
