# application/interfaces/mcp_interface.py
from typing import Dict, Any, List, Optional
import httpx
import asyncio
import json
import logging
from pydantic import BaseModel


class MCPTool(BaseModel):
    """MCP Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requires_auth: bool = False


class MCPClient:
    """Client for MCP protocol communication"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = httpx.AsyncClient(timeout=30.0)
        self.logger = logging.getLogger(__name__)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool"""
        url = f"{self.base_url}/tools/{tool_name}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self.session.post(
                url,
                json=params,
                headers=headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"MCP tool error: {response.status_code} - {response.text}")
                raise Exception(f"MCP tool failed: {response.text}")

        except Exception as e:
            self.logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            raise

    async def list_tools(self) -> List[MCPTool]:
        """List available MCP tools"""
        url = f"{self.base_url}/tools"

        try:
            response = await self.session.get(url)
            if response.status_code == 200:
                tools_data = response.json()
                return [MCPTool(**tool) for tool in tools_data.get("tools", [])]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Failed to list MCP tools: {e}")
            return []

    async def batch_call_tools(
            self,
            tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch call multiple tools"""
        results = []

        for tool_call in tool_calls:
            try:
                result = await self.call_tool(
                    tool_call["tool_name"],
                    tool_call["params"]
                )
                results.append({
                    "tool_name": tool_call["tool_name"],
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool_name": tool_call["tool_name"],
                    "success": False,
                    "error": str(e)
                })
                await asyncio.sleep(0.1)  # Rate limiting

        return results

    async def health_check(self) -> bool:
        """Check MCP server health"""
        url = f"{self.base_url}/health"

        try:
            response = await self.session.get(url, timeout=5.0)
            return response.status_code == 200
        except:
            return False

    async def close(self):
        """Close client session"""
        await self.session.aclose()


class MCPInterface:
    """Interface for MCP protocol operations"""

    def __init__(self, judge_mcp_url: str, finance_mcp_url: str):
        self.judge_client = MCPClient(judge_mcp_url)
        self.finance_client = MCPClient(finance_mcp_url)

    async def evaluate_analysis_mcp(
            self,
            analysis_content: str,
            agent_id: str,
            source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate analysis via MCP"""
        return await self.judge_client.call_tool("evaluate_financial_analysis", {
            "analysis_content": analysis_content,
            "agent_id": agent_id,
            "source_documents": source_documents
        })

    async def fetch_financial_data(
            self,
            company_cik: str,
            filing_type: str
    ) -> Dict[str, Any]:
        """Fetch financial data via MCP"""
        return await self.finance_client.call_tool("fetch_financial_statements", {
            "company_cik": company_cik,
            "filing_type": filing_type
        })

    async def validate_metrics(
            self,
            metrics: Dict[str, float],
            source_cik: str
    ) -> Dict[str, Any]:
        """Validate metrics against source via MCP"""
        return await self.judge_client.call_tool("validate_financial_metrics", {
            "metrics": metrics,
            "source_cik": source_cik
        })