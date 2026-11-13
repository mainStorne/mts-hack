from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment variables or `.env` (names are case-insensitive, e.g. LLM_ROUTE)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    # An OpenAI-compatible server (llama.cpp, vLLM, a hosted API...). The model name is informational for llama.cpp.
    llm_route: HttpUrl = HttpUrl("http://localhost:8080/v1")
    llm_api_key: str = "not-needed"
    llm_model: str = "local"
    llm_max_tokens: int = Field(8192, gt=0)  # thinking tokens count against it
    llm_temperature: float = Field(0.0, ge=0)
    # Qwen-style "thinking" through chat_template_kwargs; servers that do not know it ignore it.
    llm_thinking: bool = True

    # Every user message is screened before the agent (agent/guardrail.py). It fails open when the model is down.
    guardrail_enabled: bool = True
    agent_max_model_calls: int = Field(
        8, gt=0, description="Model calls per question, including tool rounds and handoffs."
    )
    # A long conversation is summarized before the next question once it reaches this many (approximate) tokens.
    agent_summarize_trigger_tokens: int = Field(3000, gt=0)
    agent_summarize_keep_messages: int = Field(6, ge=2)

    # Knowledge-base search (support/retrieval.py): an embeddings server and a reranker server, both OpenAI/llama.cpp-style
    # (POST /embeddings, POST /rerank). Candidates come from the vectors, the reranker orders them and drops weak matches.
    # Unset = the plain word-match search of crm.py.
    embeddings_route: HttpUrl | None = None
    embeddings_model: str = "bge-m3"
    embeddings_query_prefix: str = ""
    reranker_route: HttpUrl | None = None
    reranker_model: str = "bge-reranker-v2-m3"
    # Raw reranker scores (logits) below this are "not found": the agent then says so instead of citing a loosely related article.
    reranker_min_score: float = 0.0
    kb_candidates: int = Field(8, gt=0)
    kb_results: int = Field(3, gt=0)

    # Arize Phoenix (docker-compose.yml), e.g. http://localhost:6006. Unset = no tracing.
    phoenix_endpoint: HttpUrl | None = None
    phoenix_project: str = "support"

    database_url: str | None = None

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_root_path: str = Field("", pattern=r"^(/\S*[^/])?$")
    api_cors_allow_origin_regex: str = ".*"


settings = Settings()
