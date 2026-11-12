import type { SessionState } from "@/api/chat/types";
import { AgentBadge } from "@/components/chat/AgentBadge";
import { SwitchLocaleButton } from "@/components/layout/SwitchLocaleButton";
import SwitchThemeButton from "@/components/layout/SwitchThemeButton";
import { useLingui } from "@lingui/react/macro";

export function Header({
  title,
  state,
  agentLabel,
  menuId,
}: {
  title: string;
  state: SessionState;
  agentLabel: string | null;
  // The checkbox of the drawer with the chat list that the menu button toggles
  menuId: string;
}) {
  const { t } = useLingui();
  return (
    <header className="border-base-300 flex items-center gap-2 border-b px-2 py-2 sm:px-4">
      <label
        htmlFor={menuId}
        title={t`Chats`}
        className="btn btn-ghost btn-square btn-sm lg:hidden"
      >
        <span className="icon-[material-symbols--menu-rounded] text-2xl" />
      </label>
      <div className="flex min-w-0 grow flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
        <h1 className="truncate text-lg font-medium">{title}</h1>
        <AgentBadge state={state} agentLabel={agentLabel} />
      </div>
      <div className="flex shrink-0 items-center">
        <SwitchLocaleButton />
        <SwitchThemeButton />
      </div>
    </header>
  );
}
