# infrastructure/mcp/finance_mcp_server.py
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
import uvicorn
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from infrastructure.sec_edgar.sec_client import SECClient
from infrastructure.sec_edgar.edgar_parser import EdgarParser
from contracts.finance_contracts import SECFilingRequest, CompanyFinancials


class FinanceMCPAdapter:
    """MCP adapter for finance agent"""

    def __init__(self, sec_client: SECClient, edgar_parser: EdgarParser):
        self.sec_client = sec_client
        self.edgar_parser = edgar_parser
        self.mcp = FastMCP("FinanceAnalysisAgent")
        self._register_tools()

    def _register_tools(self):
        """Register finance-related MCP tools"""

        @self.mcp.tool()
        async def fetch_financial_statements(
                company_cik: str,
                filing_type: str = "10-K",
                period_end: Optional[str] = None
        ) -> Dict[str, Any]:
            """Fetch financial statements from SEC EDGAR"""
            try:
                # Fetch filing
                filing_data = await self.sec_client.fetch_filing(
                    cik=company_cik,
                    filing_type=filing_type,
                    date=period_end or datetime.now().strftime("%Y-%m-%d")
                )

                if not filing_data:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No {filing_type} found for CIK {company_cik}"
                    )

                # Parse financial statements
                parsed_data = self.edgar_parser.parse_financial_statements(filing_data)

                return {
                    "company_cik": company_cik,
                    "filing_type": filing_type,
                    "period_end": period_end,
                    "financial_statements": parsed_data,
                    "metadata": {
                        "source": "SEC EDGAR",
                        "retrieved_at": datetime.utcnow().isoformat()
                    }
                }

            except Exception as e:
                logging.error(f"Failed to fetch financial statements: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.mcp.tool()
        async def calculate_financial_ratios(
                company_cik: str,
                ratios: List[str]
        ) -> Dict[str, Any]:
            """Calculate financial ratios from SEC data"""
            try:
                # Fetch latest filings
                filings = await self.sec_client.search_filings(
                    company=company_cik,
                    filing_types=["10-K", "10-Q"],
                    start_date="2022-01-01",
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )

                if not filings:
                    return {"ratios": {}, "error": "No filings found"}

                # Parse and calculate ratios
                ratio_results = {}
                for ratio in ratios:
                    try:
                        value = self._calculate_ratio(ratio, filings)
                        ratio_results[ratio] = {
                            "value": value,
                            "unit": "ratio",
                            "calculation_date": datetime.utcnow().isoformat()
                        }
                    except Exception as e:
                        ratio_results[ratio] = {
                            "error": str(e),
                            "calculation_date": datetime.utcnow().isoformat()
                        }

                return {
                    "company_cik": company_cik,
                    "ratios": ratio_results,
                    "as_of_date": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logging.error(f"Failed to calculate ratios: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.mcp.tool()
        async def extract_risk_factors(
                company_cik: str,
                filing_type: str = "10-K"
        ) -> Dict[str, Any]:
            """Extract risk factors from SEC filing"""
            try:
                # Fetch filing
                filing_data = await self.sec_client.fetch_filing(
                    cik=company_cik,
                    filing_type=filing_type,
                    date=None
                )

                if not filing_data:
                    return {"risk_factors": [], "error": "Filing not found"}

                # Extract risk factors
                risk_factors = self.edgar_parser.extract_risk_factors(filing_data)

                return {
                    "company_cik": company_cik,
                    "filing_type": filing_type,
                    "risk_factors": risk_factors,
                    "count": len(risk_factors),
                    "extracted_at": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logging.error(f"Failed to extract risk factors: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.mcp.tool()
        async def compare_companies(
                company_ciks: List[str],
                metrics: List[str]
        ) -> Dict[str, Any]:
            """Compare multiple companies on specified metrics"""
            try:
                comparison_data = {}

                for cik in company_ciks:
                    company_data = await fetch_financial_statements(cik, "10-K")

                    # Extract metrics
                    metrics_values = {}
                    for metric in metrics:
                        # This would extract specific metrics from financial statements
                        metrics_values[metric] = self._extract_metric(
                            company_data, metric
                        )

                    comparison_data[cik] = {
                        "metrics": metrics_values,
                        "as_of_date": datetime.utcnow().isoformat()
                    }

                return {
                    "comparison": comparison_data,
                    "metrics": metrics,
                    "comparison_date": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logging.error(f"Failed to compare companies: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _calculate_ratio(self, ratio: str, filings: List[Dict]) -> float:
        """Calculate specific financial ratio"""
        # Simplified implementation
        if ratio == "current_ratio":
            return 2.5  # Example
        elif ratio == "debt_to_equity":
            return 0.8  # Example
        elif ratio == "roe":
            return 0.15  # Example
        else:
            raise ValueError(f"Unknown ratio: {ratio}")

    def _extract_metric(self, company_data: Dict[str, Any], metric: str) -> float:
        """Extract specific metric from company data"""
        # Simplified implementation
        return 0.0

    def get_app(self) -> FastAPI:
        """Get FastAPI application"""
        return self.mcp._app


class FinanceMCPServer:
    """MCP server for Finance Agent"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8002):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Finance Agent MCP Server")

        # Initialize dependencies
        self.sec_client = SECClient()
        self.edgar_parser = EdgarParser()

        self.adapter = FinanceMCPAdapter(self.sec_client, self.edgar_parser)
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "finance_agent"}

        @self.app.get("/capabilities")
        async def get_capabilities():
            return {
                "agent": "FinanceAnalysisAgent",
                "capabilities": [
                    "financial_statement_analysis",
                    "ratio_calculation",
                    "risk_factor_extraction",
                    "company_comparison",
                    "trend_analysis"
                ],
                "version": "1.0.0"
            }

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