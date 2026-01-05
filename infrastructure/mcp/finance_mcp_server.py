# infrastructure/mcp/finance_mcp_server.py
from agents.finance_agent.core.finance_agent import FinanceAgent
from application.use_cases.analyze_company_use_case import AnalyzeCompanyCommand


class FinanceMCPServer:
    def __init__(self, agent: FinanceAgent) -> None:
        self._agent = agent

    def analyze(self, command: AnalyzeCompanyCommand) -> dict:
        return self._agent.analyze(command)
