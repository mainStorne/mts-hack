import type { AgentName } from "@/api/chat/types";
import type { MessageDescriptor } from "@lingui/core";
import { msg } from "@lingui/core/macro";

// The agents of the support team, named for the user
export const agentTitles: Record<AgentName, MessageDescriptor> = {
  triage: msg`Triage agent`,
  billing: msg`Billing agent`,
  tariffs: msg`Tariffs agent`,
  tech: msg`Technical support agent`,
  human: msg`Human operator`,
};

export const agentIcons: Record<AgentName, string> = {
  triage: "icon-[material-symbols--alt-route-rounded]",
  billing: "icon-[material-symbols--receipt-long-outline-rounded]",
  tariffs: "icon-[material-symbols--sim-card-outline-rounded]",
  tech: "icon-[material-symbols--support-agent-rounded]",
  human: "icon-[material-symbols--person-outline-rounded]",
};
