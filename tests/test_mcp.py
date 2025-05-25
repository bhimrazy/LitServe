# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.mcp import MCPPrompt, MCPResource, MCPSpec, MCPTool
from litserve.utils import wrap_litserve_start


class TestMCPAPI(ls.LitAPI):
    """Simple test API for MCP testing."""

    def setup(self, device):
        self.model = "test-model"

    def decode_request(self, request, context_kwargs=None):
        return request

    def predict(self, input_data, context=None):
        return f"Test response for: {input_data}"

    def encode_response(self, output, context=None):
        return output


class CustomMCPAPI(ls.LitAPI):
    """API with custom MCP methods."""

    def setup(self, device):
        self.model = "custom-model"

    def decode_request(self, request, context_kwargs=None):
        return request

    def predict(self, input_data, context=None):
        return f"Custom response for: {input_data}"

    def encode_response(self, output, context=None):
        return output

    def get_tools(self):
        """Return custom tools."""
        tool_data = {
            "name": "custom_tool",
            "description": "A custom tool for testing",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Query to process"}},
            },
        }
        return [MCPTool.model_validate(tool_data)]

    def call_tool(self, name, arguments):
        """Handle custom tool calls."""
        if name == "custom_tool":
            query = arguments.get("query", "")
            return f"Custom tool processed: {query}"
        raise ValueError(f"Unknown tool: {name}")

    def get_resources(self):
        """Return custom resources."""
        return [
            MCPResource(
                uri="custom://data/sample",
                name="Sample Data",
                description="Sample data resource",
                mime_type="application/json",
            )
        ]

    def read_resource(self, uri):
        """Handle custom resource reads."""
        if uri == "custom://data/sample":
            return {"sample": "data", "type": "custom"}
        raise ValueError(f"Unknown resource: {uri}")

    def get_prompts(self):
        """Return custom prompts."""
        return [
            MCPPrompt(
                name="custom_prompt",
                description="A custom prompt for testing",
                arguments=[{"name": "topic", "description": "Topic to discuss", "required": True}],
            )
        ]

    def get_prompt(self, name, arguments):
        """Handle custom prompt gets."""
        if name == "custom_prompt":
            topic = arguments.get("topic", "general")
            from litserve.specs.mcp import GetPromptResponse, MCPMessage

            return GetPromptResponse(
                description=f"Custom prompt about {topic}",
                messages=[MCPMessage(role="user", content=f"Let's discuss {topic}")],
            ).model_dump()
        raise ValueError(f"Unknown prompt: {name}")


@pytest.fixture
def mcp_initialize_request():
    """Standard MCP initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "clientInfo": {"name": "Test MCP Client", "version": "1.0.0"},
        },
    }


@pytest.fixture
def mcp_tools_list_request():
    """MCP tools/list request."""
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


@pytest.fixture
def mcp_call_tool_request():
    """MCP tools/call request."""
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "predict", "arguments": {"input": "test input"}},
    }


@pytest.mark.asyncio
async def test_mcp_spec_initialization():
    """Test basic MCP spec initialization."""
    spec = MCPSpec()
    assert spec.server_name == "LitServe MCP Server"
    assert spec.server_version == "1.0.0"
    assert spec.capabilities is not None
    assert spec.stream is True


@pytest.mark.asyncio
async def test_mcp_initialize(mcp_initialize_request):
    """Test MCP initialize protocol."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            response = await ac.post("/mcp", json=mcp_initialize_request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert "result" in data

            result = data["result"]
            assert result["protocolVersion"] == "2024-11-05"
            assert "capabilities" in result
            assert "serverInfo" in result
            assert result["serverInfo"]["name"] == "LitServe MCP Server"


@pytest.mark.asyncio
async def test_mcp_tools_list(mcp_initialize_request, mcp_tools_list_request):
    """Test MCP tools/list method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # List tools
            response = await ac.post("/mcp", json=mcp_tools_list_request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 2
            assert "result" in data

            result = data["result"]
            assert "tools" in result
            assert len(result["tools"]) >= 2  # At least predict and health_check

            tool_names = [tool["name"] for tool in result["tools"]]
            assert "predict" in tool_names
            assert "health_check" in tool_names


@pytest.mark.asyncio
async def test_mcp_call_tool(mcp_initialize_request, mcp_call_tool_request):
    """Test MCP tools/call method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # Call tool
            response = await ac.post("/mcp", json=mcp_call_tool_request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 3
            assert "result" in data

            result = data["result"]
            assert "content" in result
            assert len(result["content"]) > 0
            assert result["content"][0]["type"] == "text"
            assert "Test response for" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_resources_list(mcp_initialize_request):
    """Test MCP resources/list method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    request = {"jsonrpc": "2.0", "id": 4, "method": "resources/list", "params": {}}

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # List resources
            response = await ac.post("/mcp", json=request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 4
            assert "result" in data

            result = data["result"]
            assert "resources" in result
            assert len(result["resources"]) >= 2  # At least server info and capabilities


@pytest.mark.asyncio
async def test_mcp_read_resource(mcp_initialize_request):
    """Test MCP resources/read method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    request = {"jsonrpc": "2.0", "id": 5, "method": "resources/read", "params": {"uri": "litserve://server/info"}}

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # Read resource
            response = await ac.post("/mcp", json=request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 5
            assert "result" in data

            result = data["result"]
            assert "contents" in result
            assert len(result["contents"]) > 0


@pytest.mark.asyncio
async def test_mcp_prompts_list(mcp_initialize_request):
    """Test MCP prompts/list method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    request = {"jsonrpc": "2.0", "id": 6, "method": "prompts/list", "params": {}}

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # List prompts
            response = await ac.post("/mcp", json=request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 6
            assert "result" in data

            result = data["result"]
            assert "prompts" in result
            assert len(result["prompts"]) >= 1  # At least analyze_prediction


@pytest.mark.asyncio
async def test_mcp_get_prompt(mcp_initialize_request):
    """Test MCP prompts/get method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "prompts/get",
        "params": {"name": "analyze_prediction", "arguments": {"input": "test input", "output": "test output"}},
    }

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # Get prompt
            response = await ac.post("/mcp", json=request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 7
            assert "result" in data

            result = data["result"]
            assert "messages" in result
            assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_mcp_custom_api(mcp_initialize_request):
    """Test MCP with custom API methods."""
    spec = MCPSpec()
    server = ls.LitServer(CustomMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Initialize first
            await ac.post("/mcp", json=mcp_initialize_request, timeout=10)

            # Test custom tools
            tools_request = {"jsonrpc": "2.0", "id": 8, "method": "tools/list", "params": {}}
            response = await ac.post("/mcp", json=tools_request, timeout=10)
            data = response.json()
            tool_names = [tool["name"] for tool in data["result"]["tools"]]
            assert "custom_tool" in tool_names

            # Test custom tool call
            call_request = {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "custom_tool", "arguments": {"query": "test query"}},
            }
            response = await ac.post("/mcp", json=call_request, timeout=10)
            data = response.json()
            assert "Custom tool processed" in data["result"]["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_rest_endpoints():
    """Test REST endpoints for debugging."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Test tools REST endpoint
            response = await ac.get("/mcp/tools", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert "tools" in data

            # Test resources REST endpoint
            response = await ac.get("/mcp/resources", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert "resources" in data

            # Test prompts REST endpoint
            response = await ac.get("/mcp/prompts", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert "prompts" in data


@pytest.mark.asyncio
async def test_mcp_error_handling():
    """Test MCP error handling."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Test invalid JSON-RPC
            invalid_request = {"invalid": "request"}
            response = await ac.post("/mcp", json=invalid_request, timeout=10)
            assert response.status_code == 400
            data = response.json()
            assert "error" in data

            # Test unknown method
            unknown_method = {"jsonrpc": "2.0", "id": 10, "method": "unknown/method", "params": {}}
            response = await ac.post("/mcp", json=unknown_method, timeout=10)
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert "Method not found" in data["error"]["message"]


@pytest.mark.asyncio
async def test_mcp_cors_handling():
    """Test CORS preflight handling."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # Test OPTIONS request
            response = await ac.options("/mcp", timeout=10)
            assert response.status_code == 200
            assert "Access-Control-Allow-Origin" in response.headers
            assert response.headers["Access-Control-Allow-Origin"] == "*"


@pytest.mark.asyncio
async def test_mcp_ping():
    """Test MCP ping method."""
    spec = MCPSpec()
    server = ls.LitServer(TestMCPAPI(), spec=spec)

    request = {"jsonrpc": "2.0", "id": 11, "method": "ping", "params": {}}

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            response = await ac.post("/mcp", json=request, timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 11
            assert "result" in data
