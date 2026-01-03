# infrastructure/mcp/judge_mcp_server.py
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
import uvicorn
from typing import List, Dict, Any
from contracts.evaluation_contracts import EvaluationRequest, EvaluationResult
from application.use_cases.evaluate_analysis import EvaluateAnalysisUseCase
from infrastructure.sec_edgar.sec_client import SECClient
import logging


class JudgeMCPAdapter:
    """Adapter for MCP server interface"""

    def __init__(self, use_case: EvaluateAnalysisUseCase):
        self.use_case = use_case
        self.mcp = FastMCP("FinanceJudgeAgent")
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools"""

        @self.mcp.tool()
        async def evaluate_financial_analysis(
                analysis_content: str,
                agent_id: str,
                source_documents: List[Dict[str, Any]] = None,
                rubrics: List[str] = None
        ) -> Dict[str, Any]:
            """Evaluate financial analysis against SEC rubrics"""
            try:
                request = EvaluationRequest(
                    analysis_id=f"mcp_{agent_id}_{hash(analysis_content)}",
                    agent_id=agent_id,
                    analysis_content=analysis_content,
                    source_documents=source_documents or [],
                    rubrics_to_evaluate=rubrics or [
                        "factual_accuracy",
                        "source_fidelity",
                        "regulatory_compliance",
                        "financial_reasoning",
                        "materiality_relevance"
                    ]
                )

                result = await self.use_case.execute(request)
                return result.dict()

            except Exception as e:
                logging.error(f"Evaluation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.mcp.tool()
        async def evaluate_specific_rubric(
                analysis_content: str,
                rubric: str,
                source_documents: List[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Evaluate specific rubric"""
            # Similar implementation focused on single rubric
            pass

        @self.mcp.tool()
        async def batch_evaluate(
                analyses: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Batch evaluate multiple analyses"""
            results = []
            for analysis in analyses:
                result = await evaluate_financial_analysis(
                    analysis["content"],
                    analysis["agent_id"],
                    analysis.get("sources", [])
                )
                results.append(result)
            return results

    def get_app(self) -> FastAPI:
        """Get FastAPI application"""
        return self.mcp._app


class JudgeMCPServer:
    """MCP server for Judge Agent"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Finance Judge Agent MCP Server")

        # Initialize dependencies
        self.sec_client = SECClient()
        self.use_case = EvaluateAnalysisUseCase(
            mcp_client=None,  # Would be injected
            a2a_client=None,  # Would be injected
            sec_data_provider=self.sec_client
        )

        self.adapter = JudgeMCPAdapter(self.use_case)
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "judge_agent"}

        @self.app.get("/capabilities")
        async def get_capabilities():
            with open(".well-known/agent.json") as f:
                import json
                return json.load(f)

        # Mount MCP routes
        self.app.mount("/mcp", self.adapter.get_app())

    async def start(self):
        """Start the MCP server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()