import type { Approval as ApprovalData } from "@/api/chat/types";
import { Trans } from "@lingui/react/macro";
import { decideApproval } from "./chat-session";

// What the agent is about to do for the customer (switch a tariff, change a service, ...). Nothing is done until the
// user approves it here; a newer question in the chat drops it undone
export function Approval({
  approval,
  isLast,
  isPending,
}: {
  approval: ApprovalData;
  isLast: boolean;
  isPending: boolean;
}) {
  return (
    <div className="border-base-300 bg-base-100 rounded-field flex flex-col gap-2 border! p-3">
      <span className="font-medium">
        <Trans>The agent needs your approval to do this:</Trans>
      </span>
      <ul className="flex flex-col gap-1">
        {approval.actions.map((action, i) => (
          <li key={i} className="ml-5 list-disc">
            {action.description}
          </li>
        ))}
      </ul>
      {!isLast ? (
        <span className="text-base-content/60 text-sm">
          <Trans>Not done: a newer question was asked.</Trans>
        </span>
      ) : isPending ? (
        <span className="loading loading-spinner loading-sm" />
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => decideApproval("approve")}
          >
            <Trans>Approve</Trans>
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => decideApproval("reject")}
          >
            <Trans>Reject</Trans>
          </button>
        </div>
      )}
    </div>
  );
}
