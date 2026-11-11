"""Thin OpenAI-compatible chat client. Works with llm7.io (default, free tier), OpenAI,
Ollama's /v1 endpoint, or any compatible server. Configure with LLM_BASE_URL, LLM_API_KEY,
LLM_MODEL."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULTS = {
    "LLM_BASE_URL": "https://api.llm7.io/v1",
    "LLM_API_KEY": "unused",
    "LLM_MODEL": "codestral-latest",
}


@dataclass
class LLMClient:
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", DEFAULTS["LLM_BASE_URL"]))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", DEFAULTS["LLM_API_KEY"]))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULTS["LLM_MODEL"]))
    temperature: float = 0.0
    timeout_s: float = 90.0

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None):
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message
