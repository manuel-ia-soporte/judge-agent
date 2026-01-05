# application/use_cases/compare_companies_use_case.py
from domain.models.entities import SECDocument
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService


class CompareCompaniesUseCase:
    """
    Application use case that compares two companies across
    financial, operational, strategic, and risk dimensions.
    """

    def __init__(
        self,
        financial_service: FinancialAnalysisService,
        operational_service: OperationalAnalysisService,
        strategic_service: StrategicAnalysisService,
        risk_service: RiskAssessmentService,
    ) -> None:
        self._financial = financial_service
        self._operational = operational_service
        self._strategic = strategic_service
        self._risk = risk_service

    def execute(
        self,
        company_a_docs: list[SECDocument],
        company_b_docs: list[SECDocument],
    ) -> dict[str, dict]:
        return {
            "company_a": self._analyze_company(company_a_docs),
            "company_b": self._analyze_company(company_b_docs),
        }

    def _analyze_company(self, documents: list[SECDocument]) -> dict:
        financial_metrics = self._financial.extract_metrics(documents)
        ratios = self._financial.calculate_ratios(financial_metrics)

        operational_metrics = self._operational.extract_operational_metrics(documents)
        efficiency = self._operational.analyze_operational_efficiency(
            operational_metrics
        )

        strategic = {}
        strategic.update(self._strategic.analyze_strategic_position(documents))
        strategic.update(self._strategic.assess_competitive_advantage(documents))
        strategic.update(self._strategic.analyze_growth_strategies(documents))
        strategic.update(self._strategic.assess_innovation_capability(documents))
        strategic.update(self._strategic.analyze_market_position(documents))

        risks = self._risk.extract_risk_factors(documents)
        categorized = self._risk.categorize_risks(risks)
        risk_score = self._risk.assess_overall_risk(categorized)

        return {
            "financial": {"metrics": financial_metrics, "ratios": ratios},
            "operational": {"efficiency": efficiency},
            "strategic": strategic,
            "risk": {
                "factors": risks,
                "categories": categorized,
                "score": risk_score,
            },
        }
