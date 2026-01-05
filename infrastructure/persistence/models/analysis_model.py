# infrastructure/persistence/models/analysis_model.py
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnalysisModel:
    id: str
    cik: str
    created_at: datetime
    result: dict
