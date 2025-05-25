"""Additional tests for MCP 2025-03-26 Streamable HTTP features."""

import pytest
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.mcp import MCPSpec, mcp_tool


# Test tool registration with decorators
@mcp_tool(
    name="test_decorator_tool",
    description="A tool registered via decorator",
    input_schema={
        "type": "object",
        "properties": {"value": {"type": "string", "description": "Test value"}},
        "required": ["value"],
    },
)
def test_decorator_tool(value: str) -> str:
    """Test tool function."""
    return f"Processed: {value}"


class TestAPI(ls.LitAPI):
    def setup(self, device):
        pass

    def decode_request(self, request):
        return request

    def predict(self, x):
        return {"result": f"processed_{x}"}

    def encode_response(self, output):
        return output


@pytest.mark.asyncio
async def test_mcp_session_management():
    """Test session management with Mcp-Session-Id header."""
    spec = MCPSpec(enable_sessions=True, stateless_mode=False)
    server = ls.LitServer(TestAPI(), spec=spec)

    # Create ASGI app
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # First request should create a session
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        assert response.status_code == 200

        # Should get a session ID in response
        session_id = response.headers.get("Mcp-Session-Id")
        assert session_id is not None
        assert len(session_id) == 32  # UUID hex length

        # Subsequent requests with session ID should work
        tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        headers = {"Mcp-Session-Id": session_id}
        response = await ac.post("/mcp", json=tools_request, headers=headers)
        assert response.status_code == 200

        # Session termination
        response = await ac.delete("/mcp", headers=headers)
        assert response.status_code == 204


@pytest.mark.asyncio
async def test_mcp_decorator_tools():
    """Test tool registration via decorators."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
            },
        }

        await ac.post("/mcp", json=init_request)

        # List tools - should include decorator-registered tool
        tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await ac.post("/mcp", json=tools_request)
        assert response.status_code == 200

        tools = response.json()["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "test_decorator_tool" in tool_names

        # Call the decorator-registered tool
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "test_decorator_tool", "arguments": {"value": "test_input"}},
        }

        response = await ac.post("/mcp", json=call_request)
        assert response.status_code == 200

        result = response.json()["result"]
        assert result["content"][0]["text"] == "Processed: test_input"


@pytest.mark.asyncio
async def test_mcp_json_rpc_batching():
    """Test JSON-RPC batching support."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Send batch request
        batch_request = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "prompts/list", "params": {}},
        ]

        response = await ac.post("/mcp", json=batch_request)
        assert response.status_code == 200

        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 3

        # Check that all requests were processed
        ids = [result["id"] for result in results]
        assert sorted(ids) == [1, 2, 3]


@pytest.mark.asyncio
async def test_mcp_sse_streaming():
    """Test SSE streaming support."""
    spec = MCPSpec(enable_sessions=True)
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # First create a session
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        session_id = response.headers.get("Mcp-Session-Id")

        # Test SSE connection
        headers = {"Mcp-Session-Id": session_id, "Accept": "text/event-stream"}

        async with ac.stream("GET", "/mcp", headers=headers) as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"] == "text/event-stream"

            # Read first few lines to verify SSE format
            lines = []
            async for line in stream.aiter_lines():
                lines.append(line)
                if len(lines) >= 3:  # Just test a few lines
                    break

            # Should have event and data lines
            assert any(line.startswith("event:") for line in lines)
            assert any(line.startswith("data:") for line in lines)


@pytest.mark.asyncio
async def test_mcp_protocol_version():
    """Test that the correct protocol version is returned."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        assert response.status_code == 200

        result = response.json()["result"]
        assert result["protocolVersion"] == "2025-03-26"


@pytest.mark.asyncio
async def test_mcp_stateless_mode():
    """Test stateless mode operation."""
    spec = MCPSpec(enable_sessions=False, stateless_mode=True)
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # In stateless mode, no session ID should be returned
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        assert response.status_code == 200

        # Should not get a session ID
        session_id = response.headers.get("Mcp-Session-Id")
        assert session_id is None


@pytest.mark.asyncio
async def test_mcp_error_handling_invalid_session():
    """Test error handling for invalid session IDs."""
    spec = MCPSpec(enable_sessions=True)
    server = ls.LitServer(TestAPI(), spec=spec)

    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Try to use an invalid session ID for SSE
        headers = {"Mcp-Session-Id": "invalid-session-id", "Accept": "text/event-stream"}

        response = await ac.get("/mcp", headers=headers)
        assert response.status_code == 404
        assert "Invalid session ID" in response.json()["error"]

        # Try to delete an invalid session
        response = await ac.delete("/mcp", headers={"Mcp-Session-Id": "invalid-session-id"})
        assert response.status_code == 404
        assert "Invalid session ID" in response.json()["error"]


if __name__ == "__main__":
    import asyncio

    # Run a quick test to verify functionality
    async def quick_test():
        print("Running quick test of MCP 2025-03-26 features...")

        # Test basic functionality
        spec = MCPSpec(enable_sessions=True)
        server = ls.LitServer(TestAPI(), spec=spec)
        app = server.app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Test initialization
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}, "resources": {}},
                    "clientInfo": {"name": "Test Client", "version": "1.0.0"},
                },
            }

            response = await ac.post("/mcp", json=init_request)
            if response.status_code == 200:
                print("✅ Initialization successful")
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    print(f"✅ Session management working: {session_id[:8]}...")

                # Test tools list including decorator tools
                tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

                response = await ac.post("/mcp", json=tools_request)
                if response.status_code == 200:
                    tools = response.json()["result"]["tools"]
                    tool_names = [tool["name"] for tool in tools]
                    print(f"✅ Tools available: {tool_names}")
                    if "test_decorator_tool" in tool_names:
                        print("✅ Decorator tool registration working")
            else:
                print(f"❌ Test failed: {response.status_code}")

    asyncio.run(quick_test())
