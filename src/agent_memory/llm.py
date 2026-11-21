"""Deterministic offline chat and optional key-backed live chat."""

import os
import re
from typing import Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_memory.store import Memory


class ExtractedMemory(BaseModel):
    text: str = Field(min_length=1)
    kind: Literal["fact", "preference", "event"]
    topic: str = Field(min_length=1)


class ExtractedMemories(BaseModel):
    items: list[ExtractedMemory]


class AgentLLM(Protocol):
    def respond(self, system_prompt: str, user_text: str, recalled: list[Memory]) -> str:
        """Answer the current turn using only relevant remembered context."""

    def extract(self, user_text: str) -> list[ExtractedMemory]:
        """Extract durable memories from the user's statement."""


class FakeLLM:
    """Offline demo model: deterministic output and deliberately narrow extraction."""

    def respond(self, system_prompt: str, user_text: str, recalled: list[Memory]) -> str:
        del system_prompt
        if recalled:
            facts = "; ".join(memory.text for memory in recalled)
            return f"I remember: {facts}. You asked: {user_text}"
        return f"I have no saved context yet. You said: {user_text}"

    def extract(self, user_text: str) -> list[ExtractedMemory]:
        extracted: list[ExtractedMemory] = []
        for sentence in re.split(r"(?<=[.!])\s+", user_text.strip()):
            sentence = sentence.rstrip(".! ")
            match = re.fullmatch(r"I (?:now )?prefer (.+)", sentence, flags=re.I)
            if match:
                extracted.append(
                    ExtractedMemory(text=sentence, kind="preference", topic="preference")
                )
                continue
            match = re.fullmatch(r"I (?:now )?live in (.+)", sentence, flags=re.I)
            if not match:
                match = re.fullmatch(r"I moved to (.+)", sentence, flags=re.I)
            if match:
                extracted.append(ExtractedMemory(text=sentence, kind="fact", topic="home_city"))
                continue
            match = re.fullmatch(r"my ([a-z][a-z ]{1,40}) is (.+)", sentence, flags=re.I)
            if match:
                topic = re.sub(r"\s+", "_", match.group(1).strip().lower())
                extracted.append(ExtractedMemory(text=sentence, kind="fact", topic=topic))
        return extracted


class LiveLLM:
    """LangChain chat model; extraction uses validated structured output."""

    def __init__(self, model: object) -> None:
        self._model = model

    def respond(self, system_prompt: str, user_text: str, recalled: list[Memory]) -> str:
        del recalled
        response = self._model.invoke([SystemMessage(system_prompt), HumanMessage(user_text)])
        if not isinstance(response.content, str):
            raise TypeError("Expected a text response from the chat model")
        return response.content

    def extract(self, user_text: str) -> list[ExtractedMemory]:
        structured = self._model.with_structured_output(ExtractedMemories)
        result = structured.invoke(
            [
                SystemMessage(
                    "Extract only durable facts and preferences explicitly stated by the user. "
                    "Return no memories for questions, hypotheticals, secrets, or instructions. "
                    "Use a stable, concise topic key so later updates replace the same fact."
                ),
                HumanMessage(user_text),
            ]
        )
        if not isinstance(result, ExtractedMemories):
            raise TypeError("Expected validated structured memory extraction")
        return result.items


def create_llm() -> AgentLLM:
    """Use a configured live provider or the explicit offline FakeLLM demo."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if openai_key and anthropic_key:
        raise ValueError("Set only one LLM provider API key")
    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("Set OPENAI_MODEL when using OPENAI_API_KEY")
        return LiveLLM(ChatOpenAI(model=model, api_key=openai_key))
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        model = os.environ.get("ANTHROPIC_MODEL")
        if not model:
            raise ValueError("Set ANTHROPIC_MODEL when using ANTHROPIC_API_KEY")
        return LiveLLM(ChatAnthropic(model=model, api_key=anthropic_key))
    return FakeLLM()
