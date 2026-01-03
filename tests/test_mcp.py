# tests/test_mcp.py
"""
MCP Protocol Tests
"""
import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from infrastructure.mcp.mcp_client import MCPClient, MCPClientFactory
from infrastructure.mcp.judge_mcp_server import JudgeMCPServer
from infrastructure.mcp.finance_mcp_server import FinanceMCPServer
from contracts.evaluation_contracts import EvaluationRequest, RubricCategory


class TestMCPClient:
    """Test MCP Client functionality"""

    @pytest.fixture
    def mcp_client(self):
        """Create MCP client instance"""
        return MCPClient(
            base_url="http://localhost:8000",
            agent_name="TestAgent",
            version="1.0.0"
        )

    @pytest.mark.asyncio
    async def test_invoke_success(self, mcp_client):
        """Test successful MCP invocation"""
        with patch.object(mcp_client.session, 'post') as mock_post:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "result": {"success": True, "data": "test"},
                "id": 1
            }
            mock_post.return_value = mock_response

            # Invoke method
            result = await mcp_client.invoke("test.method", {"param": "value"})

            assert result["success"] == True
            assert result["data"] == "test"

            # Check request format
            call_args = mock_post.call_args
            assert call_args[0][0].endswith("/mcp")
            payload = call_args[1]['json']
            assert payload["method"] == "test.method"
            assert payload["params"] == {"param": "value"}

    @pytest.mark.asyncio
    async def test_invoke_error(self, mcp_client):
        """Test MCP invocation with error"""
        with patch.object(mcp_client.session, 'post') as mock_post:
            # Mock error response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": 1
            }
            mock_post.return_value = mock_response

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await mcp_client.invoke("nonexistent.method", {})

            assert "Method not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invoke_http_error(self, mcp_client):
        """Test MCP invocation with HTTP error"""
        with patch.object(mcp_client.session, 'post') as mock_post:
            # Mock HTTP error
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                await mcp_client.invoke("test.method", {})

            assert "HTTP error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, mcp_client):
        """Test MCP invocation timeout"""
        with patch.object(mcp_client.session, 'post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()

            with pytest.raises(TimeoutError):
                await mcp_client.invoke("test.method", {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_call_tool(self, mcp_client):
        """Test calling specific tool"""
        with patch.object(mcp_client, 'invoke') as mock_invoke:
            mock_invoke.return_value = {"result": "tool_output"}

            result = await mcp_client.call_tool(
                "evaluate_analysis",
                {"analysis": "test"}
            )

            assert result == {"result": "tool_output"}
            mock_invoke.assert_called_once_with(
                "tools/call",
                {"name": "evaluate_analysis", "arguments": {"analysis": "test"}},
                tool_name=None
            )

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test listing available tools"""
        with patch.object(mcp_client, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "tools": [
                    {"name": "tool1", "description": "Test tool 1"},
                    {"name": "tool2", "description": "Test tool 2"}
                ]
            }

            tools = await mcp_client.list_tools()

            assert len(tools) == 2
            assert tools[0]["name"] == "tool1"
            assert tools[1]["name"] == "tool2"

    @pytest.mark.asyncio
    async def test_batch_call(self, mcp_client):
        """Test batch calling multiple tools"""
        with patch.object(mcp_client, 'call_tool') as mock_call:
            mock_call.side_effect = [
                {"result": "output1"},
                Exception("Tool failed"),
                {"result": "output3"}
            ]

            calls = [
                {"tool": "tool1", "arguments": {}},
                {"tool": "tool2", "arguments": {}},
                {"tool": "tool3", "arguments": {}}
            ]

            results = await mcp_client.batch_call(calls)

            assert len(results) == 3
            assert results[0]["success"] == True
            assert results[0]["result"] == {"result": "output1"}
            assert results[1]["success"] == False
            assert "Tool failed" in results[1]["error"]
            assert results[2]["success"] == True

    @pytest.mark.asyncio
    async def test_health_check(self, mcp_client):
        """Test health check"""
        with patch.object(mcp_client.session, 'get') as mock_get:
            # Mock healthy response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            health = await mcp_client.health_check()

            assert health["healthy"] == True
            assert health["status_code"] == 200

            # Mock unhealthy response
            mock_response.status_code = 500
            health = await mcp_client.health_check()

            assert health["healthy"] == False
            assert health["status_code"] == 500

            # Mock exception
            mock_get.side_effect = Exception("Connection failed")
            health = await mcp_client.health_check()

            assert health["healthy"] == False
            assert "Connection failed" in health["error"]

    @pytest.mark.asyncio
    async def test_get_schema(self, mcp_client):
        """Test getting MCP schema"""
        with patch.object(mcp_client.session, 'get') as mock_get:
            # Mock schema response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "TestAgent",
                "version": "1.0.0",
                "capabilities": ["tools", "resources"]
            }
            mock_get.return_value = mock_response

            schema = await mcp_client.get_schema()

