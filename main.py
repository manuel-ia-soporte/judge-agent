# main.py
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# === Domain Services (Pure Business Logic) ===
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService
from domain.services.rubrics_service import RubricsService

# === Application Use Cases (Orchestrators) ===
from application.use_cases.analyze_company_use_case import AnalyzeCompanyUseCase
from application.use_cases.assess_risk_use_case import AssessRiskUseCase
from application.use_cases.evaluate_analysis import EvaluateAnalysisUseCase
from application.use_cases.score_rubrics import ScoreRubricsUseCase

# === Application Ports (Interfaces) ===
from application.ports.analysis_ports import (
    FinancialAnalysisPort,
    OperationalAnalysisPort,
    StrategicAnalysisPort,
)
from application.ports.sec_filing_port import SECFilingPort

# === Infrastructure Adapters (Implement Ports) ===
from infrastructure.adapters.analysis_adapters import (
    FinancialAnalysisAdapter,
    OperationalAnalysisAdapter,
    StrategicAnalysisAdapter,
)
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter

# === Agents ===
from agents.finance_agent.core.finance_agent import FinanceAgent
from agents.finance_agent.strategies.analysis_strategy import FullAnalysisStrategy
from agents.judge_agent.judge_agent import JudgeAgent
from agents.registry.agent_registry import AgentRegistry

# === Policies ===
# from application.policies.simple_role_policy import SimpleRolePolicy

# === Routers ===
from api.routers.analysis_router import router as analysis_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DependencyContainer:
    """Composition Root: All dependencies wired here."""

    def __init__(self):
        # --- Infrastructure Adapters (Outer Layer) ---
        sec_adapter: SECFilingPort = SECEdgarAdapter()

        # --- Domain Services (Core) ---
        financial_svc = FinancialAnalysisService()
        operational_svc = OperationalAnalysisService()
        strategic_svc = StrategicAnalysisService()
        risk_svc = RiskAssessmentService()
        rubrics_svc = RubricsService()

        # --- Port Adapters (Bridge Core ↔ App) ---
        financial_port: FinancialAnalysisPort = FinancialAnalysisAdapter(financial_svc)
        operational_port: OperationalAnalysisPort = OperationalAnalysisAdapter(operational_svc)
        strategic_port: StrategicAnalysisPort = StrategicAnalysisAdapter(strategic_svc)

        # --- Use Cases (Application Layer) ---
        risk_use_case = AssessRiskUseCase(operational_analysis=operational_port)
        analyze_use_case = AnalyzeCompanyUseCase(
            financial_analysis=financial_port,
            operational_analysis=operational_port,
            strategic_analysis=strategic_port,
        )
        score_rubrics = ScoreRubricsUseCase(rubrics_service=rubrics_svc)
        evaluate_use_case = EvaluateAnalysisUseCase(score_rubrics_use_case=score_rubrics)

        # --- Agents ---
        strategy = FullAnalysisStrategy(
            risk_use_case=risk_use_case,
            financial_service=financial_svc,
            operational_service=operational_svc,
            strategic_service=strategic_svc,
        )
        self.finance_agent = FinanceAgent(strategy=strategy)
        self.judge_agent = JudgeAgent(evaluate_use_case=evaluate_use_case)

        # --- Registry ---
        self.agent_registry = AgentRegistry()
        self.agent_registry.register("finance", self.finance_agent)
        self.agent_registry.register("judge", self.judge_agent)

    def get_finance_agent(self) -> FinanceAgent:
        return self.finance_agent

    async def start(self) -> None:
        """Initialize any async resources."""
        logger.info("DependencyContainer initialized")

    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("DependencyContainer shutting down")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("🚀 Starting Finance Judge System...")
    container = app.state.container
    await container.start()
    yield
    await container.shutdown()
    logger.info("🛑 Finance Judge System stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Finance Judge System",
        description="Multi-agent financial intelligence platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    container = DependencyContainer()
    app.state.container = container

    def get_finance_agent() -> FinanceAgent:
        return container.get_finance_agent()

    app.include_router(
        analysis_router,
        prefix="/api/v1/analysis",
        tags=["Analysis"],
        dependencies=[Depends(get_finance_agent)],
    )

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": list(container.agent_registry.list_agents().keys()),
        }

    @app.get("/agents")
    async def list_agents():
        agents_info = {}
        for name, agent in container.agent_registry.list_agents().items():
            agents_info[name] = {
                "type": type(agent).__name__,
                "capabilities": list(agent.list_capabilities().keys()) if hasattr(agent, 'list_capabilities') else []
            }
        return {"agents": agents_info}

    @app.post("/api/v1/evaluate")
    async def evaluate_analysis(request: dict):
        """
        Evaluate an analysis using the JudgeAgent.
        Expected request body: {"analysis_content": "...", "agent_id": "..."}
        """
        import uuid
        from contracts.evaluation_contracts import EvaluationRequest, RubricCategory

        eval_request = EvaluationRequest(
            analysis_id=request.get("analysis_id", f"eval_{uuid.uuid4().hex[:8]}"),
            agent_id=request.get("agent_id", "finance_agent"),
            analysis_content=request.get("analysis_content", ""),
            source_documents=request.get("source_documents", []),
            rubrics_to_evaluate=list(RubricCategory),
            context=request.get("context", {}),
        )

        result = await container.judge_agent.evaluate_with_use_case(eval_request)
        return result

    return app


# Composition Root
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)