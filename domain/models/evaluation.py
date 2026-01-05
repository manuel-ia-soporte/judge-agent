# domain/models/evaluation.py
from dataclasses import dataclass
from enum import Enum


class RubricCategory(Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_FIDELITY = "source_fidelity"


@dataclass(frozen=True)
class RubricScore:
    score: int
    rationale: str
