# domain/services/risk_assessment_service.py
from domain.models.entities import SECDocument


class RiskAssessmentService:
    """
    Domain service responsible for risk extraction, categorization,
    and overall risk scoring.
    """

    def extract_risk_factors(self, documents: list[SECDocument]) -> list[str]:
        risks: list[str] = []

        for doc in documents:
            text = doc.content.lower()

            if "liquidity risk" in text:
                risks.append("liquidity_risk")
            if "regulatory risk" in text:
                risks.append("regulatory_risk")
            if "market risk" in text:
                risks.append("market_risk")

        return risks

    def categorize_risks(self, risks: list[str]) -> dict[str, list[str]]:
        categories: dict[str, list[str]] = {
            "financial": [],
            "regulatory": [],
            "market": [],
        }

        for risk in risks:
            if risk == "liquidity_risk":
                categories["financial"].append(risk)
            elif risk == "regulatory_risk":
                categories["regulatory"].append(risk)
            else:
                categories["market"].append(risk)

        return categories

    def assess_overall_risk(self, categorized_risks: dict[str, list[str]]) -> float:
        risk_count = sum(len(v) for v in categorized_risks.values())
        return min(1.0, risk_count / 10.0)
