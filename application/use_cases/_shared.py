# application/use_cases/_shared.py
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EvaluationAssumptions:
    confidence_threshold: float = 0.7
    allow_estimates: bool = True
    require_sources: bool = True


@dataclass(frozen=True)
class EvaluationContext:
    signals: Dict[str, float]
    assumptions: EvaluationAssumptions
