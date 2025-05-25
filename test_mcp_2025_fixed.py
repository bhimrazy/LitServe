"""Additional tests for MCP 2025-03-26 Streamable HTTP features."""

import pytest
from httpx import AsyncClient

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
        # Initialize session
        init_request = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["protocolVersion"] == "2025-03-26"

        # Check session header
        session_id = response.headers.get("Mcp-Session-Id")
        assert session_id is not None

        # Use session for tools list
        headers = {"Mcp-Session-Id": session_id}
        tools_request = {"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}}

        response = await ac.post("/mcp", json=tools_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "tools" in data["result"]


@pytest.mark.asyncio
async def test_decorator_tool_registration():
    """Test tool registration using decorators."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # List tools
        tools_request = {"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}}

        response = await ac.post("/mcp", json=tools_request)
        assert response.status_code == 200
        data = response.json()
        tools = data["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "test_decorator_tool" in tool_names

        # Call the decorator tool
        call_request = {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {"name": "test_decorator_tool", "arguments": {"value": "test_input"}},
        }

        response = await ac.post("/mcp", json=call_request)
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["content"][0]["text"] == "Processed: test_input"


@pytest.mark.asyncio
async def test_jsonrpc_batching():
    """Test JSON-RPC batching support."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Send batch request
        batch_request = [
            {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            },
            {"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}},
        ]

        response = await ac.post("/mcp", json=batch_request)
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["id"] == "init-1"
        assert results[1]["id"] == "tools-1"


@pytest.mark.asyncio
async def test_sse_streaming():
    """Test Server-Sent Events streaming."""
    spec = MCPSpec(enable_sessions=True)
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Initialize session first
        init_request = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        session_id = response.headers.get("Mcp-Session-Id")

        # Request SSE stream
        headers = {"Mcp-Session-Id": session_id, "Accept": "text/event-stream"}

        async with ac.stream("GET", "/mcp", headers=headers) as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"] == "text/event-stream"

            # Read a few lines to test streaming
            lines = []
            async for line in stream.aiter_lines():
                lines.append(line)
                if len(lines) >= 3:  # Just test a few lines
                    break


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid requests."""
    spec = MCPSpec()
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Invalid JSON-RPC request
        invalid_request = {"not_jsonrpc": "2.0", "id": "invalid-1", "method": "invalid_method"}

        response = await ac.post("/mcp", json=invalid_request)
        assert response.status_code == 200  # Should still return 200 with error in body
        data = response.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_stateless_mode():
    """Test stateless mode operation."""
    spec = MCPSpec(enable_sessions=False, stateless_mode=True)
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Initialize without sessions
        init_request = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await ac.post("/mcp", json=init_request)
        assert response.status_code == 200
        # Should not have session header in stateless mode
        assert "Mcp-Session-Id" not in response.headers


@pytest.mark.asyncio
async def test_session_cleanup():
    """Test session cleanup via DELETE method."""
    spec = MCPSpec(enable_sessions=True)
    server = ls.LitServer(TestAPI(), spec=spec)
    app = server.app

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Initialize session
        init_request = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        response = await ac.post("/mcp", json=init_request)
        session_id = response.headers.get("Mcp-Session-Id")

        # Delete session
        headers = {"Mcp-Session-Id": session_id, "Accept": "text/event-stream"}
        response = await ac.delete("/mcp", headers=headers)
        assert response.status_code == 404  # Session should be deleted


if __name__ == "__main__":
    import asyncio

    async def quick_test():
        """Quick test runner."""
        spec = MCPSpec(enable_sessions=True)
        server = ls.LitServer(TestAPI(), spec=spec)
        app = server.app

        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test session management
            init_request = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }

            response = await ac.post("/mcp", json=init_request)
            print(f"Initialize response: {response.status_code}")
            session_id = response.headers.get("Mcp-Session-Id")

            if session_id:
                headers = {"Mcp-Session-Id": session_id}
                tools_request = {"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}}
                response = await ac.post("/mcp", json=tools_request, headers=headers)
                print(f"Tools list response: {response.status_code}")
            else:
                print("No session ID received")

    asyncio.run(quick_test())
