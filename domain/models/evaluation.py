# domain/models/evaluation.py
from dataclasses import dataclass
from enum import Enum


class RubricCategory(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_FIDELITY = "source_fidelity"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    FINANCIAL_REASONING = "financial_reasoning"
    MATERIALITY = "materiality"


@dataclass(frozen=True)
class RubricScore:
    score: int
    rationale: str

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("Rubric score must be between 0 and 100")
