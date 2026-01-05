# agents/finance_agent/sec_analyzer.py
from typing import Dict, Any, List

from domain.models.entities import SECDocument


class SECAnalyzer:
    @staticmethod
    def summarize(documents: List[SECDocument]) -> Dict[str, Any]:
        return {
            "document_count": len(documents),
            "forms": list({doc.form_type for doc in documents}),
        }
