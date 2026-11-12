-- The tables of the API (applied by `support-db init` and when the API starts; safe to repeat).
-- The agent's own memory (LangGraph checkpoints) is created by AsyncPostgresSaver.setup().

-- A conversation of a customer with the assistant. `checkpoint`: the agent's thread after the last answer, where the next
-- question goes on from; `waiting`: an answer that waits for the customer's decision on an action (HITL), with the checkpoint
-- the agent paused at.
CREATE TABLE IF NOT EXISTS chats (
    chat_id text PRIMARY KEY,
    client_id text NOT NULL,
    title text NOT NULL,
    title_by text NOT NULL DEFAULT 'question' CHECK (title_by IN ('question', 'model', 'user')),
    checkpoint text,
    waiting_checkpoint text,
    waiting jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chats_client_idx ON chats (client_id, updated_at DESC);

-- The answers as the client got them, with the customer's 👍/👎.
CREATE TABLE IF NOT EXISTS answers (
    id bigserial PRIMARY KEY,
    chat_id text NOT NULL REFERENCES chats (chat_id) ON DELETE CASCADE,
    answer_id text NOT NULL UNIQUE,
    span_id text,  -- the root span of the answer's trace in Phoenix: what a vote annotates
    response jsonb NOT NULL,
    feedback text CHECK (feedback IN ('like', 'dislike')),
    comment text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS answers_chat_idx ON answers (chat_id, id);
