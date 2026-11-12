import { chatsQueryOptions } from "@/api/chat/queries";
import { formatApiErrorMessage } from "@/api/helpers/errors";
import { Logo } from "@/components/icons/Logo";
import { Header } from "@/components/layout/Header";
import { Trans, useLingui } from "@lingui/react/macro";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ChatInput } from "./ChatInput";
import { ChatList } from "./ChatList";
import {
  AssistantMessage,
  ErrorMessage,
  StreamingMessage,
  UserMessage,
} from "./ChatMessages";
import { NewChatButton } from "./NewChatButton";
import {
  askQuestion,
  cancelQuestion,
  getChatId,
  openChat,
  setDraftQuery,
  useChatSession,
} from "./chat-session";

const DRAWER_ID = "chat-drawer";

// How close to the end the chat must be scrolled to follow a new answer as it is written
const FOLLOW_DISTANCE_PX = 80;

// A chat as in messengers: the earlier messages above, the newest at the bottom over the message box. The chats of the
// user are on the left (in a drawer on a phone)
export function ChatPage({
  urlChat,
  onChatChange,
}: {
  // The chat the address asks for (`?chat=`)
  urlChat?: string;
  // The open chat changed: the address follows it
  onChatChange: (chatId: string | null) => void;
}) {
  const { t } = useLingui();

  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isFollowing = useRef(true);
  const hasSyncedUrl = useRef(false);

  // The conversation stays when the user goes to another page and comes back
  const {
    messages,
    chatId,
    state,
    agentLabel,
    answerText,
    status,
    liveHandoffs,
    isPending,
    error,
    isOpening,
    openingError,
    draftQuery,
    composerId,
  } = useChatSession();
  const { data: chats } = useQuery(chatsQueryOptions());

  // A chat from the address is opened unless it is the open one
  useEffect(() => {
    if (urlChat && urlChat !== getChatId()) void openChat(urlChat);
  }, [urlChat]);

  // The address follows the open chat. The first time it is left as it is if it names a chat: that one is being opened
  useEffect(() => {
    const isFirst = !hasSyncedUrl.current;
    hasSyncedUrl.current = true;
    if (isFirst && urlChat) return;
    if (chatId !== (urlChat ?? null)) onChatChange(chatId);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reacts to the open chat only
  }, [chatId]);

  // Follow the chat to its end while the user has not scrolled up to read
  useEffect(() => {
    const scroll = scrollRef.current;
    if (!scroll || !isFollowing.current) return;
    scroll.scrollTop = scroll.scrollHeight;
  }, [messages, answerText, status, liveHandoffs, error, isOpening]);

  const handleScroll = () => {
    const scroll = scrollRef.current;
    if (!scroll) return;
    isFollowing.current =
      scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <
      FOLLOW_DISTANCE_PX;
  };

  const handleSend = (query: string) => {
    isFollowing.current = true;
    void askQuestion(query);
  };

  const isEmpty =
    messages.length === 0 && !isPending && !isOpening && !openingError;
  const lastMessage = messages.at(-1);
  const title =
    chats?.find((chat) => chat.chat_id === chatId)?.title ?? t`Support Agent`;

  return (
    <div className="drawer lg:drawer-open h-full">
      <input
        id={DRAWER_ID}
        type="checkbox"
        className="drawer-toggle"
        checked={isDrawerOpen}
        onChange={(e) => setIsDrawerOpen(e.target.checked)}
      />

      <div className="drawer-content flex h-full min-h-0 min-w-0 flex-col">
        <Header
          title={title}
          state={state}
          agentLabel={agentLabel}
          menuId={DRAWER_ID}
        />

        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex min-h-0 grow flex-col overflow-y-auto"
        >
          <div className="mx-auto flex w-full max-w-4xl grow flex-col gap-4 p-4">
            {isEmpty && (
              <EmptyState
                showExamples={!draftQuery.trim()}
                onPick={(query) => {
                  setDraftQuery(query);
                  inputRef.current?.focus();
                }}
              />
            )}

            {isOpening && (
              <>
                <div className="skeleton h-10 w-1/2 self-end" />
                <div className="skeleton h-24 w-3/4" />
              </>
            )}
            {openingError !== null && (
              <div className="border-base-300 bg-base-200 text-base-content rounded-field flex flex-col gap-1 self-start border! px-4 py-2">
                <Trans>Could not open the chat</Trans>
                <span className="text-base-content/60 text-sm">
                  {formatApiErrorMessage(openingError)}
                </span>
                {chatId && !isOpening && (
                  <button
                    type="button"
                    className="btn btn-sm self-start"
                    onClick={() => void openChat(chatId)}
                  >
                    <Trans>Retry</Trans>
                  </button>
                )}
              </div>
            )}

            {messages.map((message, i) =>
              message.role === "user" ? (
                <UserMessage key={`${chatId}:${i}`} text={message.content} />
              ) : (
                <AssistantMessage
                  key={`${chatId}:${i}`}
                  answer={message.answer}
                  isLast={i === messages.length - 1}
                  isPending={isPending}
                />
              ),
            )}

            {isPending && (
              <StreamingMessage
                answerText={answerText}
                status={status}
                handoffs={liveHandoffs}
              />
            )}
            {!isPending && error && (
              <ErrorMessage
                error={error}
                hasDraft={lastMessage?.role === "user" && !!draftQuery.trim()}
                hasApproval={
                  lastMessage?.role === "assistant" &&
                  !!lastMessage.answer.approval
                }
              />
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-4xl px-4 pt-2 pb-4">
          <ChatInput
            key={composerId}
            inputRef={inputRef}
            text={draftQuery}
            setText={setDraftQuery}
            onSend={handleSend}
            onStop={cancelQuestion}
            isPending={isPending}
          />
        </div>
      </div>

      <div className="drawer-side z-40 h-full">
        <label
          htmlFor={DRAWER_ID}
          aria-label={t`Close menu`}
          className="drawer-overlay"
        />
        <aside className="bg-base-100 border-base-300 flex h-full w-72 flex-col gap-3 border-r p-3">
          <div className="flex items-center gap-2 px-1">
            <Logo className="size-8" />
            <span className="text-lg font-medium">
              <Trans>Support Agent</Trans>
            </span>
          </div>
          <NewChatButton onClick={() => setIsDrawerOpen(false)} />
          <div className="min-h-0 grow overflow-y-auto">
            <ChatList onSelect={() => setIsDrawerOpen(false)} />
          </div>
        </aside>
      </div>
    </div>
  );
}

function EmptyState({
  showExamples,
  onPick,
}: {
  showExamples: boolean;
  onPick: (query: string) => void;
}) {
  const { t } = useLingui();
  return (
    <div className="flex grow flex-col items-center justify-center gap-2 text-center">
      <Logo className="size-16" />
      <h2 className="text-xl font-medium">
        <Trans>How can I help you?</Trans>
      </h2>
      <span className="text-base-content/70 max-w-xl">
        <Trans>
          I can help with your balance and bills, tariffs and services, and
          connection problems.
        </Trans>
      </span>
      {showExamples && (
        <>
          <div className="mt-2 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
            {[
              t`How much data do I have left this month?`,
              t`Why was I charged for a service I did not order?`,
              t`I want to switch to a cheaper tariff.`,
              t`My mobile internet stopped working. What should I do?`,
            ].map((query) => (
              <button
                key={query}
                type="button"
                className="btn btn-outline h-auto justify-start py-3 text-left font-normal whitespace-normal"
                onClick={() => onPick(query)}
              >
                {query}
              </button>
            ))}
          </div>
          <p className="text-base-content/60 mt-2 max-w-2xl text-sm">
            <Trans>
              The agent asks for your approval before changing your tariff or
              services.
            </Trans>
          </p>
        </>
      )}
    </div>
  );
}
