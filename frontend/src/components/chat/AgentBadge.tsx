import type { AgentName, SessionState } from "@/api/chat/types";
import { cn } from "@/lib/ui/cn";
import { Trans, useLingui } from "@lingui/react/macro";
import { agentIcons, agentTitles } from "./agents";

// Who answers now (with the name the server gave the agent, if it did), and what is known about the customer
export function AgentBadge({
  state,
  agentLabel,
}: {
  state: SessionState;
  agentLabel?: string | null;
}) {
  const { i18n } = useLingui();
  const agent: AgentName = state.active_agent;
  const { verified, customer_name, escalated, ticket_id } = state;
  const name = customer_name ?? undefined;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span
        className={cn(
          "badge badge-soft gap-1",
          agent === "human" ? "badge-accent" : "badge-primary",
        )}
      >
        <span className={cn(agentIcons[agent], "text-base")} />
        {agentLabel || i18n._(agentTitles[agent])}
      </span>
      {verified && (
        <span className="badge badge-soft badge-secondary gap-1">
          <span className="icon-[material-symbols--verified-user-outline-rounded] text-base" />
          {name ? <Trans>Verified: {name}</Trans> : <Trans>Verified</Trans>}
        </span>
      )}
      {escalated && (
        <span className="badge badge-soft badge-accent gap-1">
          <span className="icon-[material-symbols--headset-mic-outline-rounded] text-base" />
          {ticket_id ? (
            <Trans>Escalated to a human, ticket {ticket_id}</Trans>
          ) : (
            <Trans>Escalated to a human</Trans>
          )}
        </span>
      )}
    </div>
  );
}
