# main.py
import asyncio
import logging
from typing import List
import uvicorn
from fastapi import FastAPI
from infrastructure.mcp.judge_mcp_server import JudgeMCPServer
from infrastructure.a2a.a2a_server import A2AServer
from agents.judge_agent.judge_agent import JudgeAgent
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinanceJudgeSystem:
    """Main application orchestrator"""

    def __init__(self):
        self.app = FastAPI(title="Finance Judge Agent System")
        self.judge_agent = JudgeAgent()
        self.mcp_server = JudgeMCPServer()
        self.a2a_server = A2AServer()

        self._setup_routes()

    def _setup_routes(self):
        """Setup main application routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "Finance Judge Agent System",
                "version": "1.0.0",
                "endpoints": {
                    "mcp": "/mcp",
                    "a2a": "/a2a",
                    "health": "/health",
                    "agent_status": "/agent/status"
                }
            }

        @self.app.get("/agent/status")
        async def agent_status():
            return self.judge_agent.get_status()

        @self.app.post("/agent/evaluate")
        async def evaluate_analysis(request: dict):
            from contracts.evaluation_contracts import EvaluationRequest
            eval_request = EvaluationRequest(**request)
            result = await self.judge_agent.evaluate(eval_request)
            return result.dict()

    async def start_services(self):
        """Start all services"""
        logger.info("Starting Finance Judge System...")

        # Start MCP server in background
        mcp_task = asyncio.create_task(self.mcp_server.start())

        # Start A2A server
        a2a_config = uvicorn.Config(
            app=self.a2a_server.get_app(),
            host=settings.A2A_HOST,
            port=settings.A2A_PORT,
            log_level="info"
        )
        a2a_server = uvicorn.Server(a2a_config)
        a2a_task = asyncio.create_task(a2a_server.serve())

        logger.info(f"MCP Server: http://{settings.MCP_HOST}:{settings.MCP_PORT}")
        logger.info(f"A2A Server: http://{settings.A2A_HOST}:{settings.A2A_PORT}")
        logger.info("Services started successfully")

        # Wait for servers
        await asyncio.gather(mcp_task, a2a_task)

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Finance Judge System...")
        await self.judge_agent.stop()
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    system = FinanceJudgeSystem()

    try:
        await system.start_services()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())