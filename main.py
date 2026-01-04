# main.py
"""
Main application with Dependency Injection
"""
import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import use cases
from application.use_cases.analyze_company_use_case import AnalyzeCompanyUseCase
from application.use_cases.compare_companies_use_case import CompareCompaniesUseCase

# Import adapters
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter
from infrastructure.external.sec_client import SECClient

# Import services
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService

# Import agent
from agents.finance_agent.core.finance_agent import FinanceAgent
from agents.finance_agent.factories.analyzer_factory import AnalyzerFactory

# Import API
from api.routers.analysis_router import router as analysis_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DependencyContainer:
    """Dependency Injection Container"""

    def __init__(self):
        # Infrastructure layer
        self.sec_client = SECClient()
        self.sec_adapter = SECEdgarAdapter(self.sec_client)

        # Domain services
        self.financial_service = FinancialAnalysisService()
        self.risk_service = RiskAssessmentService()

        # Use cases
        self.analyze_use_case = AnalyzeCompanyUseCase(
            sec_repository=self.sec_adapter,
            financial_service=self.financial_service,
            risk_service=self.risk_service
        )

        self.compare_use_case = CompareCompaniesUseCase(
            sec_repository=self.sec_adapter,
            financial_service=self.financial_service,
            risk_service=self.risk_service
        )

        # Agent
        self.analyzer_factory = AnalyzerFactory()

        self.finance_agent = FinanceAgent(
            agent_id="finance_agent_001",
            analyze_use_case=self.analyze_use_case,
            compare_use_case=self.compare_use_case
        )

    async def start(self):
        """Start all services"""
        await self.finance_agent.start()
        logging.info("All services started")

    async def stop(self):
        """Stop all services"""
        await self.finance_agent.stop()
        logging.info("All services stopped")


def create_app() -> FastAPI:
    """Create FastAPI application with dependency injection"""

    app = FastAPI(
        title="Finance Judge System",
        description="Multi-agent financial analysis system with SEC EDGAR integration",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependency container
    container = DependencyContainer()

    # Dependency injection for routes
    async def get_finance_agent() -> FinanceAgent:
        return container.finance_agent

    # Include routers
    app.include_router(
        analysis_router,
        prefix="/api/v1/analysis",
        tags=["analysis"],
        dependencies=[Depends(get_finance_agent)]
    )

    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "finance_judge_system",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }

    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        await container.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        await container.stop()

    return app


# Application entry point
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )