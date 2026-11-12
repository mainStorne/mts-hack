import { Trans } from "@lingui/react/macro";
import { startNewChat, useChatSession } from "./chat-session";

// Starts a new conversation with the agent; there is nothing to start while the chat is empty
export function NewChatButton({ onClick }: { onClick?: () => void }) {
  const { messages, isPending } = useChatSession();

  return (
    <button
      type="button"
      className="btn btn-primary btn-sm w-full"
      disabled={messages.length === 0 && !isPending}
      onClick={() => {
        startNewChat();
        onClick?.();
      }}
    >
      <span className="icon-[material-symbols--add-comment-outline-rounded]" />
      <Trans>New chat</Trans>
    </button>
  );
}
