# main.py
"""
Main application with Dependency Injection.
Follows Hexagonal Architecture: core logic is independent of infrastructure.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# === Domain Services (Core Business Logic) ===
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService
from domain.services.rubrics_service import RubricsService
from domain.services.evaluation_service import EvaluationService

# === Application Use Cases ===
from application.use_cases.analyze_company_use_case import AnalyzeCompanyUseCase
from application.use_cases.compare_companies_use_case import CompareCompaniesUseCase
from application.use_cases.assess_risk_use_case import AssessRiskUseCase
from application.use_cases.evaluate_analysis import EvaluateAnalysisUseCase
from application.use_cases.score_rubrics import ScoreRubricsUseCase

# === Infrastructure Adapters ===
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter
from infrastructure.external.sec_client import SECClient

# === Agents ===
from agents.finance_agent.core.finance_agent import FinanceAgent
from agents.finance_agent.strategies.analysis_strategy import FullAnalysisStrategy
from agents.judge_agent.judge_agent import JudgeAgent
from agents.registry.agent_registry import AgentRegistry

# === Policies & Security ===
from application.policies.simple_role_policy import SimpleRolePolicy

# === Audit & Observability ===
from infrastructure.audit.capability_audit_logger import CapabilityAuditLogger

# === API Routers ===
from api.routers.analysis_router import router as analysis_router


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DependencyContainer:
    """
    Dependency Injection Container (Composition Root).
    Implements Inversion of Control: infrastructure depends on core, not vice versa.
    """

    def __init__(self):
        logger.info("🔧 Initializing Dependency Container...")

        # --- Infrastructure Layer ---
        sec_client = SECClient()
        self.sec_adapter = SECEdgarAdapter(sec_client)
        self.audit_logger = CapabilityAuditLogger()

        # --- Policy (Security Boundary) ---
        self.role_policy = SimpleRolePolicy({
            "analyst": {"finance:analyze", "finance:compare"},
            "judge": {"judge:evaluate"},
            "admin": {"finance:*", "judge:*"}
        })

        # --- Domain Services (Pure Business Logic) ---
        financial_svc = FinancialAnalysisService()
        operational_svc = OperationalAnalysisService()
        strategic_svc = StrategicAnalysisService()
        risk_svc = RiskAssessmentService()
        rubrics_svc = RubricsService()
        evaluation_svc = EvaluationService()

        # --- Application Use Cases (Orchestrators) ---
        self.risk_use_case = AssessRiskUseCase(operational_analysis=operational_svc)

        self.analyze_use_case = AnalyzeCompanyUseCase(
            financial_analysis=financial_svc,
            operational_analysis=operational_svc,
            strategic_analysis=strategic_svc,
        )

        self.compare_use_case = CompareCompaniesUseCase(
            financial_service=financial_svc,
            operational_service=operational_svc,
            strategic_service=strategic_svc,
            risk_service=risk_svc,
        )

        score_rubrics = ScoreRubricsUseCase(rubrics_service=rubrics_svc)
        self.evaluate_use_case = EvaluateAnalysisUseCase(score_rubrics=score_rubrics)

        # --- Agents (Orchestrators with Capabilities) ---
        strategy = FullAnalysisStrategy(risk_use_case=self.risk_use_case)
        self.finance_agent = FinanceAgent(strategy=strategy)
        self.judge_agent = JudgeAgent(evaluate_use_case=self.evaluate_use_case)

        # --- Agent Registry (Service Locator for Inter-Agent Coordination) ---
        self.agent_registry = AgentRegistry()
        self.agent_registry.register("finance", self.finance_agent)
        self.agent_registry.register("judge", self.judge_agent)

        logger.info("✅ Dependency Container ready.")

    async def start(self):
        """Async startup hook."""
        await asyncio.sleep(0)  # placeholder for real async init (e.g., warm caches)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down services...")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app with full wiring."""
    app = FastAPI(
        title="Finance Judge System",
        description="Multi-agent financial intelligence platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS (adjust in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create shared container
    container = DependencyContainer()

    # Dependency provider for routers
    def get_finance_agent():
        return container.finance_agent

    # Mount routers
    app.include_router(
        analysis_router,
        prefix="/api/v1/analysis",
        tags=["Analysis"],
        dependencies=[Depends(get_finance_agent)],
    )

    # Health & discovery endpoints
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": list(container.agent_registry._agents.keys()),
        }

    @app.get("/agents")
    async def list_agents():
        return {"agents": container.agent_registry.list_agents()}

    # Lifecycle events
    @app.on_event("startup")
    async def on_startup():
        logger.info("🚀 Starting Finance Judge System...")
        await container.start()

    @app.on_event("shutdown")
    async def on_shutdown():
        await container.shutdown()
        logger.info("🛑 Finance Judge System stopped.")

    return app


# Entry point
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")