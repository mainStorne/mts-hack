// Types of the support agent backend (hand-written: the contract is small and stable)

export type AgentName = "triage" | "billing" | "tariffs" | "tech" | "human";

export type SessionState = {
  active_agent: AgentName;
  verified: boolean;
  customer_name: string | null;
  ticket_id: string | null;
  escalated: boolean;
};

export type Handoff = {
  from_agent: AgentName;
  to_agent: AgentName;
  reason: string;
};

export type ApprovalAction = { name: string; description: string };

export type Approval = { id: string; actions: ApprovalAction[] };

export type Feedback = "like" | "dislike";

export type Answer = {
  answer_id: string;
  chat_id: string;
  query: string;
  answer: string;
  state: SessionState;
  handoffs: Handoff[];
  approval: Approval | null;
  feedback: Feedback | null;
  feedback_comment: string | null;
};

export type Chat = {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatHistory = Chat & {
  state: SessionState;
  answers: Answer[];
  // The answer that waits for the user's decision on its `approval`
  waiting: Answer | null;
};

export type Decision = "approve" | "reject";

// Events of the answer stream (`text/event-stream`)
export type StreamEvent =
  | { name: "chat"; data: { chat_id: string } }
  | { name: "status"; data: { text: string } }
  | { name: "agent"; data: { name: AgentName; label: string } }
  | { name: "answer"; data: { text: string } }
  | { name: "done"; data: Answer }
  | { name: "error"; data: { detail: string } }
  | { name: "cancelled"; data: { detail: string } };

export const INITIAL_STATE: SessionState = {
  active_agent: "triage",
  verified: false,
  customer_name: null,
  ticket_id: null,
  escalated: false,
};
