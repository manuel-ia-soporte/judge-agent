# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Server settings
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 8000
    A2A_HOST: str = "0.0.0.0"
    A2A_PORT: int = 8001

    # SEC API settings
    SEC_API_KEY: Optional[str] = None
    SEC_RATE_LIMIT: int = 10

    # Agent settings
    JUDGE_AGENT_ID: str = "judge_agent_001"
    MAX_CONCURRENT_EVALUATIONS: int = 5
    EVALUATION_TIMEOUT: int = 30

    # Rubric settings
    DEFAULT_RUBRICS: list = [
        "factual_accuracy",
        "source_fidelity",
        "regulatory_compliance",
        "financial_reasoning",
        "materiality_relevance"
    ]

    PASS_THRESHOLD: float = 1.5

    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()