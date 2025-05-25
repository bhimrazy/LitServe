"""Example demonstrating the MCP 2025-03-26 Streamable HTTP specification with LitServe.

This example shows:
1. Tool registration using decorators
2. Session management
3. SSE streaming support
4. Custom resources and prompts
5. Integration with LitAPI predict functions

"""

import json
from typing import Any, Dict

import litserve as ls
from litserve.specs.mcp import mcp_tool


# Example 1: Using the global mcp_tool decorator
@mcp_tool(
    name="calculate_sum",
    description="Calculate the sum of two numbers",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
)
def calculate_sum(a: float, b: float) -> float:
    """Calculate sum of two numbers."""
    return a + b


@mcp_tool(
    name="get_user_info",
    description="Get information about a user",
    input_schema={
        "type": "object",
        "properties": {"user_id": {"type": "string", "description": "User ID"}},
        "required": ["user_id"],
    },
)
def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get user information."""
    # In a real application, this would query a database
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "created_at": "2025-01-01T00:00:00Z",
    }


class StreamableMCPAPI(ls.LitAPI):
    """Example API demonstrating MCP 2025-03-26 features."""

    def setup(self, device):
        """Setup the API."""
        # Simulate a simple AI model
        self.model_data = {
            "name": "ExampleAI",
            "version": "1.0.0",
            "capabilities": ["text-processing", "data-analysis"],
        }

    def decode_request(self, request: Any) -> Dict[str, Any]:
        """Decode incoming request."""
        if isinstance(request, str):
            try:
                return json.loads(request)
            except json.JSONDecodeError:
                return {"text": request}
        return request

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using the model."""
        text = inputs.get("text", "")

        # Simple text processing
        word_count = len(text.split()) if text else 0
        char_count = len(text)

        return {
            "processed_text": text.upper(),
            "analysis": {
                "word_count": word_count,
                "char_count": char_count,
                "sentiment": "positive" if "good" in text.lower() else "neutral",
            },
            "model_info": self.model_data,
        }

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the response."""
        return output

    # Custom MCP methods
    def get_tools(self):
        """Get custom tools for this API."""
        return [
            ls.MCPTool(
                name="analyze_text",
                description="Analyze text using the AI model",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Text to analyze"}},
                    "required": ["text"],
                },
            ),
            ls.MCPTool(
                name="get_model_info",
                description="Get information about the AI model",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Handle custom tool calls."""
        if name == "analyze_text":
            text = arguments.get("text", "")
            return self.predict({"text": text})
        if name == "get_model_info":
            return self.model_data
        raise ValueError(f"Unknown tool: {name}")

    def get_resources(self):
        """Get custom resources for this API."""
        return [
            ls.MCPResource(
                uri="litserve://api/model-config",
                name="Model Configuration",
                description="Current model configuration and settings",
                mime_type="application/json",
            ),
            ls.MCPResource(
                uri="litserve://api/stats",
                name="API Statistics",
                description="Runtime statistics for the API",
                mime_type="application/json",
            ),
        ]

    def read_resource(self, uri: str) -> Any:
        """Handle custom resource reads."""
        if uri == "litserve://api/model-config":
            return {"model": self.model_data, "config": {"max_tokens": 1000, "temperature": 0.7, "batch_size": 1}}
        if uri == "litserve://api/stats":
            return {"requests_processed": 42, "uptime_seconds": 3600, "memory_usage_mb": 256}
        raise ValueError(f"Unknown resource: {uri}")

    def get_prompts(self):
        """Get custom prompts for this API."""
        return [
            ls.MCPPrompt(
                name="analyze_document",
                description="Analyze a document using the AI model",
                arguments=[
                    {"name": "document", "description": "Document content", "required": True},
                    {"name": "focus", "description": "Analysis focus area", "required": False},
                ],
            )
        ]

    def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom prompt requests."""
        if name == "analyze_document":
            document = arguments.get("document", "")
            focus = arguments.get("focus", "general analysis")

            return {
                "description": f"Document analysis with focus on {focus}",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please analyze this document with focus on {focus}:\n\n{document}\n\n"
                        "Provide insights about:\n"
                        "1. Key themes and topics\n"
                        "2. Writing style and tone\n"
                        "3. Main conclusions\n"
                        "4. Potential improvements",
                    }
                ],
            }
        raise ValueError(f"Unknown prompt: {name}")


def create_server():
    """Create and configure the MCP server."""
    # Create MCP spec with session management enabled
    mcp_spec = ls.MCPSpec(
        server_name="StreamableMCP Example Server",
        server_version="1.0.0",
        enable_sessions=True,
        stateless_mode=False,  # Enable stateful sessions
    )

    # Create server with the MCP API
    return ls.LitServer(StreamableMCPAPI(), spec=mcp_spec, port=8000, workers_per_device=1)


async def demo_client():
    """Demonstrate client interactions with the new MCP features."""
    import httpx

    base_url = "http://localhost:8000"

    print("🚀 MCP 2025-03-26 Streamable HTTP Demo")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # 1. Initialize the MCP connection
        print("\n📡 1. Initializing MCP connection...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "Demo Client", "version": "1.0.0"},
            },
        }

        response = await client.post(f"{base_url}/mcp", json=init_request)
        session_id = response.headers.get("Mcp-Session-Id")
        print(f"✅ Connected! Session ID: {session_id}")
        print(f"Protocol Version: {response.json()['result']['protocolVersion']}")

        headers = {"Mcp-Session-Id": session_id} if session_id else {}

        # 2. List available tools (including decorated tools)
        print("\n🔧 2. Listing available tools...")
        tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await client.post(f"{base_url}/mcp", json=tools_request, headers=headers)
        tools = response.json()["result"]["tools"]
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")

        # 3. Call a decorator-registered tool
        print("\n➕ 3. Calling decorator-registered tool (calculate_sum)...")
        calc_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculate_sum", "arguments": {"a": 15, "b": 27}},
        }

        response = await client.post(f"{base_url}/mcp", json=calc_request, headers=headers)
        result = response.json()["result"]
        print(f"✅ Sum result: {result['content'][0]['text']}")

        # 4. Call a custom API tool
        print("\n🤖 4. Calling custom API tool (analyze_text)...")
        analyze_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "analyze_text", "arguments": {"text": "This is a good example of text analysis!"}},
        }

        response = await client.post(f"{base_url}/mcp", json=analyze_request, headers=headers)
        result = response.json()["result"]
        print(f"✅ Analysis result: {json.dumps(json.loads(result['content'][0]['text']), indent=2)}")

        # 5. Demonstrate JSON-RPC batching
        print("\n📦 5. Testing JSON-RPC batching...")
        batch_request = [
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "calculate_sum", "arguments": {"a": 10, "b": 20}},
            },
            {"jsonrpc": "2.0", "id": 6, "method": "resources/list", "params": {}},
        ]

        response = await client.post(f"{base_url}/mcp", json=batch_request, headers=headers)
        batch_results = response.json()
        print(f"✅ Batch completed with {len(batch_results)} responses")

        # 6. Test SSE streaming (if available)
        print("\n🌊 6. Testing SSE streaming...")
        try:
            sse_headers = {**headers, "Accept": "text/event-stream"}
            async with client.stream("GET", f"{base_url}/mcp", headers=sse_headers) as stream:
                print("✅ SSE connection established")
                event_count = 0
                async for line in stream.aiter_lines():
                    if line.startswith("data:"):
                        event_count += 1
                        print(f"📨 Received event {event_count}: {line}")
                        if event_count >= 2:  # Just demo a couple events
                            break
        except Exception as e:
            print(f"ℹ️ SSE not fully implemented in demo: {e}")

        # 7. Clean up session
        print("\n🧹 7. Cleaning up session...")
        response = await client.delete(f"{base_url}/mcp", headers=headers)
        if response.status_code == 204:
            print("✅ Session terminated successfully")


if __name__ == "__main__":
    # Start the server
    server = create_server()

    print("Starting MCP 2025-03-26 Streamable HTTP Example Server...")
    print("Features enabled:")
    print("  ✅ Streamable HTTP transport")
    print("  ✅ Session management")
    print("  ✅ Tool registration decorators")
    print("  ✅ JSON-RPC batching")
    print("  ✅ SSE streaming support")
    print("  ✅ Custom tools, resources, and prompts")
    print("\nServer will start on http://localhost:8000")
    print(
        "Run the demo client with: python -c 'import asyncio; from example_streamable_mcp import demo_client;"
        " asyncio.run(demo_client())'"
    )

    # Run the server
    server.run()
