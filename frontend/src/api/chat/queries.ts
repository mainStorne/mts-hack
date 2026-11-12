import { queryOptions } from "@tanstack/react-query";
import { getChat, listChats } from "./api";

export const chatsQueryOptions = () =>
  queryOptions({ queryKey: ["chats"], queryFn: listChats });

export const chatQueryOptions = (chatId: string) =>
  queryOptions({
    queryKey: ["chats", chatId],
    queryFn: () => getChat(chatId),
  });
