"""A scripted chat model: the tests decide what the model says, and read what it was given."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call-{name}-{len(args)}", "type": "tool_call"}


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def use(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class ScriptedModel(BaseChatModel):
    """Answers with the next scripted message on each call. `seen` keeps, per call, the system prompt and the names of the tools
    the model was offered."""

    script: list[AIMessage]
    seen: list[dict] = Field(default_factory=list)
    offered: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.offered = [getattr(t, "name", None) or t["function"]["name"] for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = next((m.text for m in messages if m.type == "system"), "")
        self.seen.append({"system": system, "tools": list(self.offered)})
        if not self.script:
            raise AssertionError("the script ran out: the agent made one more model call")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])
