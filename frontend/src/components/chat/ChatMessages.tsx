import type { Answer } from "@/api/chat/types";
import { formatApiErrorMessage } from "@/api/helpers/errors";
import { cn } from "@/lib/ui/cn";
import { Trans, useLingui } from "@lingui/react/macro";
import AnimatedDots from "./AnimatedDots";
import { AnswerBody } from "./AnswerBody";
import { Approval } from "./Approval";
import { Feedback } from "./Feedback";
import { HandoffMarker } from "./HandoffMarker";
import type { LiveHandoff } from "./chat-session";

const assistantBubble =
  "border-base-300 bg-base-200 text-base-content rounded-field flex max-w-full min-w-0 flex-col gap-2 self-start border! px-4 py-2 wrap-anywhere";

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="bg-primary text-primary-content rounded-field max-w-[80%] min-w-0 self-end rounded-br-none px-4 py-2 wrap-anywhere whitespace-pre-wrap">
      {text}
    </div>
  );
}

// An answer of the agent. The handoffs that happened while it was written come first, as markers
export function AssistantMessage({
  answer,
  isLast,
  isPending,
}: {
  answer: Answer;
  isLast: boolean;
  isPending: boolean;
}) {
  const { approval } = answer;
  return (
    <>
      {answer.handoffs.map((handoff, i) => (
        <HandoffMarker key={i} to={handoff.to_agent} reason={handoff.reason} />
      ))}
      {(answer.answer.trim() || approval) && (
        <div
          tabIndex={0}
          className={cn(assistantBubble, !!approval && "w-full")}
        >
          {answer.answer.trim() && <AnswerBody text={answer.answer} />}
          {approval && (
            <Approval
              approval={approval}
              isLast={isLast}
              isPending={isPending}
            />
          )}
          {!approval && answer.answer_id && (
            <Feedback
              chatId={answer.chat_id}
              answerId={answer.answer_id}
              feedback={answer.feedback}
              feedbackComment={answer.feedback_comment}
            />
          )}
        </div>
      )}
    </>
  );
}

// The answer being written: what is written so far and what the agent is doing meanwhile
export function StreamingMessage({
  answerText,
  status,
  handoffs,
}: {
  answerText: string;
  status: string;
  handoffs: LiveHandoff[];
}) {
  const { t } = useLingui();
  return (
    <>
      {handoffs.map((handoff, i) => (
        <HandoffMarker key={i} to={handoff.to} label={handoff.label} />
      ))}
      <div className={assistantBubble} aria-live="polite">
        {answerText && <AnswerBody text={answerText} />}
        <div className="text-base-content/70 flex text-sm">
          <span>
            {status || (answerText ? t`Receiving answer` : t`Sending request`)}
          </span>
          <AnimatedDots />
        </div>
      </div>
    </>
  );
}

export function ErrorMessage({
  error,
  hasDraft,
  hasApproval,
}: {
  error: Error;
  // The question is back in the box to be edited and sent again
  hasDraft: boolean;
  // The error came while a decision was being made
  hasApproval: boolean;
}) {
  return (
    <div className={cn(assistantBubble, "gap-1")} role="alert">
      <span>
        <Trans>Could not complete the request.</Trans>
      </span>
      <span className="text-base-content/60 text-sm">
        {formatApiErrorMessage(error)}
      </span>
      {hasDraft && (
        <span className="text-sm">
          <Trans>Your draft is below. You can edit it and send it again.</Trans>
        </span>
      )}
      {hasApproval && (
        <span className="text-sm">
          <Trans>
            The result of the action could not be confirmed. Check your account
            before trying again.
          </Trans>
        </span>
      )}
    </div>
  );
}
