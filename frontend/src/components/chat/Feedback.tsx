import { sendFeedback } from "@/api/chat/api";
import type { Feedback as FeedbackValue } from "@/api/chat/types";
import { formatApiErrorDescriptor } from "@/api/helpers/errors";
import { useToast } from "@/components/toast";
import { cn } from "@/lib/ui/cn";
import { msg } from "@lingui/core/macro";
import { Trans, useLingui } from "@lingui/react/macro";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { rememberFeedback } from "./chat-session";

const feedbackEmoji: Record<FeedbackValue, string> = {
  like: "👍",
  dislike: "👎",
};

// The comment is cut to this length
const COMMENT_MAX_LENGTH = 2000;

// 👍/👎 under an answer; a click on the chosen one takes the vote back. After a vote there is a box to say what was liked or
// not; the comment is sent with the same vote. In a chat opened again, the vote and the comment given before are shown.
export function Feedback({
  chatId,
  answerId,
  feedback,
  feedbackComment,
}: {
  chatId: string;
  answerId: string;
  feedback: FeedbackValue | null;
  feedbackComment: string | null;
}) {
  const { t } = useLingui();
  const { showSuccess, showError } = useToast();
  const [chosen, setChosen] = useState<FeedbackValue | null>(feedback);
  const [comment, setComment] = useState("");
  const [isCommentSent, setIsCommentSent] = useState(!!feedbackComment);

  const vote = useMutation({
    mutationFn: (value: FeedbackValue | null) =>
      sendFeedback(chatId, answerId, value),
    onSuccess: (_data, value) => {
      setChosen(value);
      setIsCommentSent(false);
      setComment("");
      rememberFeedback(answerId, value, null);
      if (value) {
        showSuccess(
          value === "like"
            ? msg`Thank you!`
            : msg`Thank you, we will look into it`,
        );
      }
    },
    onError: (error) => {
      showError(
        msg`Could not save the rating`,
        formatApiErrorDescriptor(error),
      );
    },
  });

  const sendComment = useMutation({
    mutationFn: (value: { feedback: FeedbackValue; comment: string }) =>
      sendFeedback(chatId, answerId, value.feedback, value.comment),
    onSuccess: (_data, value) => {
      setIsCommentSent(true);
      rememberFeedback(answerId, value.feedback, value.comment);
      showSuccess(msg`Thank you for the comment!`);
    },
    onError: (error) => {
      showError(
        msg`Could not send the comment`,
        formatApiErrorDescriptor(error),
      );
    },
  });

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        {(["like", "dislike"] as const).map((value) => (
          <button
            key={value}
            type="button"
            title={value === "like" ? t`Helpful` : t`Not helpful`}
            aria-pressed={chosen === value}
            className={cn(
              "btn btn-ghost btn-sm btn-square text-base",
              chosen === value && "bg-base-300",
              chosen && chosen !== value && "opacity-40",
            )}
            disabled={vote.isPending}
            onClick={() => vote.mutate(chosen === value ? null : value)}
          >
            {vote.isPending && vote.variables === value ? (
              <span className="loading loading-spinner loading-sm" />
            ) : (
              feedbackEmoji[value]
            )}
          </button>
        ))}
      </div>
      {chosen && !isCommentSent && (
        <>
          <textarea
            className="textarea w-full resize-y"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            maxLength={COMMENT_MAX_LENGTH}
            placeholder={
              chosen === "like" ? t`What did you like?` : t`What was wrong?`
            }
            disabled={sendComment.isPending}
          />
          <button
            type="button"
            className="btn btn-primary btn-sm self-start"
            disabled={!comment.trim() || sendComment.isPending}
            onClick={() =>
              sendComment.mutate({ feedback: chosen, comment: comment.trim() })
            }
          >
            {sendComment.isPending && (
              <span className="loading loading-spinner loading-sm" />
            )}
            <Trans>Send</Trans>
          </button>
        </>
      )}
    </div>
  );
}
