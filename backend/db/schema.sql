CREATE EXTENSION IF NOT EXISTS vector;

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

-- The knowledge base the agents search (db/kb.py, retrieval.py). Seeded from src/support/data/knowledge_base.json for the ids
-- that are not there yet; after that the table is the source of truth. `embedding` is filled by retrieval.sync() when
-- EMBEDDINGS_ROUTE is set. 1024 dimensions: bge-m3. For another model change the number here and clear the column first:
-- `UPDATE kb_articles SET embedding = NULL`.
CREATE TABLE IF NOT EXISTS kb_articles (
    id text PRIMARY KEY,
    title text NOT NULL,
    text text NOT NULL,
    embedding vector(1024),
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- A table made before the dimension was fixed (idempotent when it is 1024 already).
ALTER TABLE kb_articles ALTER COLUMN embedding TYPE vector(1024);
CREATE INDEX IF NOT EXISTS kb_articles_embedding_idx ON kb_articles USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_articles_fts_idx ON kb_articles USING gin (to_tsvector('russian', title || ' ' || text));

-- Articles crawled from a site (kb_import): where from, the page, and what the customer must provide (ids of kb_import/inputs.py).
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'demo';  -- 'demo' | 'support.mts.ru'
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'article';  -- 'article' | 'faq'
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS url text;
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS product text;
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS section text;
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS parent_id text;  -- a faq row: its article's id
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS popular boolean NOT NULL DEFAULT false;
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS required_inputs text[];  -- NULL = not analysed yet
ALTER TABLE kb_articles ADD COLUMN IF NOT EXISTS modified_at text;

-- The mock back office of the operator (crm.py): tariffs, customers with their charges, known outages and the tickets of
-- conversations handed to a human. Seeded from src/support/data/seed.json for the keys that are not there yet; after that the
-- tables are the source of truth. Replace them (and crm.py) with your real systems. PINs are plain text: it is demo data.
CREATE TABLE IF NOT EXISTS tariffs (
    id text PRIMARY KEY,
    name text NOT NULL,
    price integer NOT NULL,
    data_gb integer,  -- NULL: unlimited
    minutes integer NOT NULL,
    sms integer NOT NULL,
    description text NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    phone text PRIMARY KEY,
    pin text NOT NULL,
    name text NOT NULL,
    region text NOT NULL,
    tariff_id text NOT NULL REFERENCES tariffs (id),
    balance numeric(12, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS charges (
    id text PRIMARY KEY,
    phone text NOT NULL REFERENCES customers (phone) ON DELETE CASCADE,
    date date NOT NULL,
    description text NOT NULL,
    amount numeric(12, 2) NOT NULL,
    disputed text  -- the customer's reason, once a dispute is filed
);
CREATE INDEX IF NOT EXISTS charges_phone_idx ON charges (phone, date, id);

CREATE TABLE IF NOT EXISTS outages (
    region text NOT NULL,
    service text NOT NULL,
    status text NOT NULL,
    PRIMARY KEY (region, service)
);

CREATE TABLE IF NOT EXISTS tickets (
    id text PRIMARY KEY,
    phone text,
    summary text NOT NULL,
    priority text NOT NULL DEFAULT 'normal',
    notes text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
