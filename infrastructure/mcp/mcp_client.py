# infrastructure/mcp/mcp_client.py
from typing import Dict, Any, List, Optional
import httpx
import asyncio
import logging
from datetime import datetime, UTC
from urllib.parse import urljoin


class MCPClient:
    """Generic MCP client implementation"""

    def __init__(
            self,
            base_url: str,
            agent_name: str,
            version: str = "1.0.0",
            timeout: float = 30.0
    ):
        self.base_url = base_url
        self.agent_name = agent_name
        self.version = version
        self.timeout = timeout
        self.session = httpx.AsyncClient(timeout=timeout)
        self.logger = logging.getLogger(__name__)
        self.request_id = 0

    async def invoke(
            self,
            method: str,
            params: Dict[str, Any],
            tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke MCP method"""
        self.request_id += 1

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{self.agent_name}/{self.version}",
            "X-MCP-Agent": self.agent_name
        }

        url = urljoin(self.base_url, f"/mcp/{tool_name}" if tool_name else "/mcp")

        try:
            self.logger.debug(f"MCP request to {url}: {method}")

            response = await self.session.post(
                url,
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()

                if "error" in result:
                    self.logger.error(f"MCP error: {result['error']}")
                    raise Exception(f"MCP error: {result['error']}")

                return result.get("result", {})

            else:
                error_msg = f"MCP HTTP error: {response.status_code}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

        except httpx.TimeoutException:
            self.logger.error(f"MCP timeout for {method}")
            raise TimeoutError(f"MCP call timed out: {method}")

        except Exception as e:
            self.logger.error(f"MCP call failed: {e}")
            raise

    async def call_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call specific MCP tool"""
        return await self.invoke("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        result = await self.invoke("tools/list", {})
        return result.get("tools", [])

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        result = await self.invoke("resources/list", {})
        return result.get("resources", [])

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read resource"""
        return await self.invoke("resources/read", {"uri": uri})

    async def health_check(self) -> Dict[str, Any]:
        """Check MCP server health"""
        try:
            response = await self.session.get(
                urljoin(self.base_url, "/health"),
                timeout=5.0
            )
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "timestamp": datetime.now(UTC).isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }

    async def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema"""
        try:
            response = await self.session.get(
                urljoin(self.base_url, "/.well-known/mcp.json"),
                timeout=10.0
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Failed to get schema: {e}")
            return {}

    async def batch_call(
            self,
            calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch call multiple tools"""
        results = []

        for call in calls:
            try:
                result = await self.call_tool(call["tool"], call.get("arguments", {}))
                results.append({
                    "tool": call["tool"],
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool": call["tool"],
                    "success": False,
                    "error": str(e)
                })

                # Small delay between calls
                await asyncio.sleep(0.1)

        return results

    async def close(self):
        """Close client"""
        await self.session.aclose()


class MCPClientFactory:
    """Factory for creating MCP clients"""

    @staticmethod
    def create_judge_client(base_url: str) -> MCPClient:
        """Create judge agent MCP client"""
        return MCPClient(
            base_url=base_url,
            agent_name="FinanceJudgeAgent",
            version="1.0.0"
        )

    @staticmethod
    def create_finance_client(base_url: str) -> MCPClient:
        """Create finance agent MCP client"""
        return MCPClient(
            base_url=base_url,
            agent_name="FinanceAnalysisAgent",
            version="1.0.0"
        )

    @staticmethod
    def create_sec_client(base_url: str) -> MCPClient:
        """Create SEC agent MCP client"""
        return MCPClient(
            base_url=base_url,
            agent_name="SECDataAgent",
            version="1.0.0"
        )