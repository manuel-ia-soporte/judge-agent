# infrastructure/bootstrap/finance_agent_factory.py

from agents.finance_agent.core.finance_agent import FinanceAgent
from agents.finance_agent.strategies.analysis_strategy import FullAnalysisStrategy
from application.use_cases.analyze_company_use_case import AnalyzeCompanyUseCase


def create_finance_agent(use_case: AnalyzeCompanyUseCase) -> FinanceAgent:
    strategy = FullAnalysisStrategy(use_case)
    return FinanceAgent(strategy=strategy)
