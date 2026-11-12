import type { AgentName } from "@/api/chat/types";
import { useLingui } from "@lingui/react/macro";
import { agentIcons, agentTitles } from "./agents";

// "→ Billing agent": the conversation went to another agent. `label` is the name the server gave to the agent
export function HandoffMarker({
  to,
  label,
  reason,
}: {
  to: AgentName;
  label?: string;
  reason?: string;
}) {
  const { i18n } = useLingui();
  return (
    <div
      className="text-base-content/60 flex items-center gap-3 text-sm"
      title={reason || undefined}
    >
      <span className="bg-base-300 h-px grow" />
      <span className="flex min-w-0 items-center gap-1.5">
        <span className="icon-[material-symbols--arrow-right-alt-rounded] text-lg" />
        <span className={`${agentIcons[to]} shrink-0 text-base`} />
        <span className="font-medium">{label || i18n._(agentTitles[to])}</span>
        {reason && (
          <span className="hidden truncate sm:inline">· {reason}</span>
        )}
      </span>
      <span className="bg-base-300 h-px grow" />
    </div>
  );
}
