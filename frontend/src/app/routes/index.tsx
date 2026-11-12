import { ChatPage } from "@/components/chat/ChatPage";
import { createFileRoute } from "@tanstack/react-router";

// `?chat=<id>` opens one of the user's chats
export const Route = createFileRoute("/")({
  validateSearch: (search: Record<string, unknown>): { chat?: string } => ({
    chat:
      typeof search.chat === "string" && search.chat ? search.chat : undefined,
  }),
  component: Index,
});

function Index() {
  const { chat } = Route.useSearch();
  const navigate = Route.useNavigate();
  return (
    <ChatPage
      urlChat={chat}
      onChatChange={(chatId) =>
        void navigate({ search: { chat: chatId ?? undefined }, replace: true })
      }
    />
  );
}
