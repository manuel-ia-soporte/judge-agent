"""LLM Client for multi-provider support.

Supports OpenAI, Anthropic, and auto-selection based on environment variables.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    AUTO = "auto"


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    provider: LLMProvider
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


# Pricing per 1M tokens (approximate, as of Jan 2025)
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    return round(cost, 6)


class LLMClient:
    """Multi-provider LLM client."""

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        api_key: Optional[str] = None,
    ) -> None:
        self._provider_str = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._api_key = api_key

        self._resolved_provider: Optional[LLMProvider] = None
        self._resolved_model: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    def _resolve_provider(self) -> tuple[LLMProvider, str, str]:
        """Resolve provider, model, and API key from config or environment."""
        provider_str = self._provider_str.lower()

        if provider_str == "auto":
            # Prefer OpenRouter, fallback to OpenAI, then Anthropic
            if os.getenv("OPENROUTER_API_KEY"):
                provider = LLMProvider.OPENROUTER
                default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
                api_key = os.getenv("OPENROUTER_API_KEY", "")
            elif os.getenv("OPENAI_API_KEY"):
                provider = LLMProvider.OPENAI
                default_model = "gpt-4o-mini"
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = LLMProvider.ANTHROPIC
                default_model = "claude-3-5-sonnet-20241022"
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            else:
                raise ValueError(
                    "No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY."
                )
        elif provider_str == "openai":
            provider = LLMProvider.OPENAI
            default_model = "gpt-4o-mini"
            api_key = self._api_key or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
        elif provider_str == "anthropic":
            provider = LLMProvider.ANTHROPIC
            default_model = "claude-3-5-sonnet-20241022"
            api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
        elif provider_str == "openrouter":
            provider = LLMProvider.OPENROUTER
            default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            api_key = self._api_key or os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
        else:
            raise ValueError(f"Unsupported provider: {provider_str}")

        model = self._model or default_model
        return provider, model, api_key

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=float(self._timeout))
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        if self._resolved_provider is None:
            self._resolved_provider, self._resolved_model, api_key = self._resolve_provider()
        else:
            _, _, api_key = self._resolve_provider()

        provider = self._resolved_provider
        model = self._resolved_model or ""

        client = await self._ensure_client()
        t0 = time.perf_counter()

        if provider == LLMProvider.OPENROUTER:
            resp = await self._call_openrouter(client, api_key, model, prompt, system_prompt, **kwargs)
        elif provider == LLMProvider.OPENAI:
            resp = await self._call_openai(client, api_key, model, prompt, system_prompt, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            resp = await self._call_anthropic(client, api_key, model, prompt, system_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.latency_ms = latency_ms
        resp.cost_usd = _estimate_cost(model, resp.input_tokens, resp.output_tokens)

        return resp

    async def _call_openai(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENAI,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
        )

    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC,
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            raw_response=data,
        )

    async def _call_openrouter(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> LLMResponse:
        """Call OpenRouter API - supports any model available on OpenRouter."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://finance-judge-system.local",
            "X-Title": "Finance Judge System",
        }

        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENROUTER,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw_response=data,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
