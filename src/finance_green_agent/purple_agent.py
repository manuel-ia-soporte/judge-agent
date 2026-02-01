"""Purple Agent - Answers SEC/EDGAR questions using offline tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .agent_core.get_agent import Parameters, get_agent


@dataclass
class PurpleAgentConfig:
    model_name: str = "openai/gpt-4o-mini"
    max_turns: int = 15
    tools: list[str] | None = None
    llm_config: dict[str, Any] | None = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = [
                "google_web_search",
                "edgar_search",
                "parse_cached_html",
                "retrieve_information",
            ]
        if self.llm_config is None:
            self.llm_config = {"temperature": 0.0}


def get_purple_agent_config() -> PurpleAgentConfig:
    """Build config from environment variables."""
    return PurpleAgentConfig(
        model_name=os.environ.get("PURPLE_AGENT_MODEL", "openai/gpt-4o-mini"),
        max_turns=int(os.environ.get("PURPLE_AGENT_MAX_TURNS", "15")),
        tools=os.environ.get("PURPLE_AGENT_TOOLS", "").split(",")
        if os.environ.get("PURPLE_AGENT_TOOLS")
        else None,
        llm_config={
            "temperature": float(os.environ.get("PURPLE_AGENT_TEMPERATURE", "0.0"))
        },
    )


async def run_purple_agent(question: str, session_id: str | None = None) -> dict[str, Any]:
    """
    Run the purple agent to answer a SEC/EDGAR question.
    
    Returns:
        dict with keys: answer, metadata, error (if any)
    """
    config = get_purple_agent_config()
    
    params = Parameters(
        model_name=config.model_name,
        max_turns=config.max_turns,
        tools=config.tools,
        llm_config=config.llm_config,
    )
    
    try:
        agent = get_agent(params)
        answer, metadata = await agent.run(question, session_id=session_id)
        return {
            "answer": answer,
            "metadata": metadata,
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": "",
            "metadata": {},
            "error": str(exc),
        }
