# infrastructure/persistence/repositories/analysis_repository_impl.py
from typing import Dict

from domain.models.entities import FinancialAnalysis


class AnalysisRepositoryImpl:
    def save(self, analysis: FinancialAnalysis) -> Dict:
        return {
            "id": str(analysis.id),
            "company_cik": analysis.company_cik,
            "created_at": analysis.created_at.isoformat(),
            "summary": analysis.summary,
            "confidence_score": analysis.confidence_score,
        }
