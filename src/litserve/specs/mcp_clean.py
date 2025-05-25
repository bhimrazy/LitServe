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

import inspect
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from litserve.specs.base import LitSpec

if TYPE_CHECKING:
    from litserve.api import LitAPI
    from litserve.server import LitServer

logger = logging.getLogger(__name__)


# MCP Protocol Message Types (JSON-RPC 2.0)
class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCErrorResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int, None]
    error: JSONRPCError


class JSONRPCNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None


# MCP Primitive Types (using protocol-compliant field names)
class MCPTool(BaseModel):
    """MCP Tool definition following the protocol specification."""

    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None

    class Config:
        # Serialize as camelCase for MCP protocol compliance
        alias_generator = lambda field_name: "inputSchema" if field_name == "input_schema" else field_name
        allow_population_by_field_name = True


class MCPResource(BaseModel):
    """MCP Resource definition following the protocol specification."""

    uri: str
    name: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = None

    class Config:
        alias_generator = lambda field_name: "mimeType" if field_name == "mime_type" else field_name
        allow_population_by_field_name = True


class MCPPrompt(BaseModel):
    """MCP Prompt definition following the protocol specification."""

    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None


class MCPMessage(BaseModel):
    """MCP Message definition following the protocol specification."""

    role: str
    content: Union[str, Dict[str, Any]]


# MCP Request/Response Types
class InitializeRequest(BaseModel):
    """MCP Initialize request following the protocol specification."""

    protocol_version: str = "2024-11-05"
    capabilities: Dict[str, Any]
    client_info: Dict[str, str]

    class Config:
        alias_generator = lambda field_name: {"protocol_version": "protocolVersion", "client_info": "clientInfo"}.get(
            field_name, field_name
        )
        allow_population_by_field_name = True


class InitializeResponse(BaseModel):
    """MCP Initialize response following the protocol specification."""

    protocol_version: str = "2024-11-05"
    capabilities: Dict[str, Any]
    server_info: Dict[str, str]

    class Config:
        alias_generator = lambda field_name: {"protocol_version": "protocolVersion", "server_info": "serverInfo"}.get(
            field_name, field_name
        )
        allow_population_by_field_name = True


class ListToolsResponse(BaseModel):
    """MCP List Tools response following the protocol specification."""

    tools: List[MCPTool]


class CallToolRequest(BaseModel):
    """MCP Call Tool request following the protocol specification."""

    name: str
    arguments: Optional[Dict[str, Any]] = None


class CallToolResponse(BaseModel):
    """MCP Call Tool response following the protocol specification."""

    content: List[Dict[str, Any]]
    is_error: Optional[bool] = False

    class Config:
        alias_generator = lambda field_name: "isError" if field_name == "is_error" else field_name
        allow_population_by_field_name = True


class ListResourcesResponse(BaseModel):
    """MCP List Resources response following the protocol specification."""

    resources: List[MCPResource]


class ReadResourceRequest(BaseModel):
    """MCP Read Resource request following the protocol specification."""

    uri: str


class ReadResourceResponse(BaseModel):
    """MCP Read Resource response following the protocol specification."""

    contents: List[Dict[str, Any]]


class ListPromptsResponse(BaseModel):
    """MCP List Prompts response following the protocol specification."""

    prompts: List[MCPPrompt]


class GetPromptRequest(BaseModel):
    """MCP Get Prompt request following the protocol specification."""

    name: str
    arguments: Optional[Dict[str, Any]] = None


class GetPromptResponse(BaseModel):
    """MCP Get Prompt response following the protocol specification."""

    description: Optional[str] = None
    messages: List[MCPMessage]


# Server capabilities
DEFAULT_SERVER_CAPABILITIES = {
    "logging": {},
    "prompts": {"listChanged": True},
    "resources": {"subscribe": True, "listChanged": True},
    "tools": {"listChanged": True},
}


class MCPSpec(LitSpec):
    """Model Context Protocol (MCP) specification for LitServe.

    This spec implements the MCP HTTP Streamable transport specification,
    allowing LitServe to serve as an MCP server that provides tools, resources,
    and prompts to MCP clients.

    Usage:
        server = LitServer(api, spec=MCPSpec())
        server.run()

    The spec supports:
    - Tools: Functions exposed to LLM for execution
    - Resources: Data/content served to clients
    - Prompts: Template interactions for LLMs
    - Session management and capability negotiation

    """

    def __init__(
        self,
        server_name: str = "LitServe MCP Server",
        server_version: str = "1.0.0",
        capabilities: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.server_name = server_name
        self.server_version = server_version
        self.capabilities = capabilities or DEFAULT_SERVER_CAPABILITIES.copy()
        self.session_id: Optional[str] = None
        self.initialized = False

        # Storage for MCP primitives
        self._tools: Dict[str, Any] = {}
        self._resources: Dict[str, Any] = {}
        self._prompts: Dict[str, Any] = {}

        # Register MCP endpoints
        self._register_mcp_endpoints()

    def _register_mcp_endpoints(self):
        """Register all MCP protocol endpoints."""
        # Core protocol endpoints
        self.add_endpoint("/mcp", self.handle_mcp_request, ["POST"])
        self.add_endpoint("/mcp", self.handle_options, ["OPTIONS"])

        # Optional: RESTful endpoints for easier debugging
        self.add_endpoint("/mcp/tools", self.list_tools_rest, ["GET"])
        self.add_endpoint("/mcp/resources", self.list_resources_rest, ["GET"])
        self.add_endpoint("/mcp/prompts", self.list_prompts_rest, ["GET"])

    @property
    def stream(self):
        """MCP supports streaming responses."""
        return True

    def pre_setup(self, lit_api: "LitAPI"):
        """Pre-setup validation and configuration."""
        # Validate that required methods are implemented if using custom MCP features
        if hasattr(lit_api, "get_tools") and not callable(lit_api.get_tools):
            raise ValueError("get_tools must be callable if defined")

        if hasattr(lit_api, "get_resources") and not callable(lit_api.get_resources):
            raise ValueError("get_resources must be callable if defined")

        if hasattr(lit_api, "get_prompts") and not callable(lit_api.get_prompts):
            raise ValueError("get_prompts must be callable if defined")

    def setup(self, server: "LitServer"):
        """Setup the MCP spec with the server."""
        super().setup(server)
        logger.info(f"MCP spec setup complete for {self.server_name}")

    def decode_request(self, request, context_kwargs: Optional[dict] = None):
        """Decode MCP JSON-RPC request."""
        if isinstance(request, dict):
            return request
        if hasattr(request, "json"):
            return request.json()
        return request

    async def handle_options(self, request: Request):
        """Handle CORS preflight requests."""
        return JSONResponse(
            content={},
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    async def handle_mcp_request(self, request: Request):
        """Main MCP request handler implementing JSON-RPC 2.0."""
        try:
            body = await request.body()
            if not body:
                return self._create_error_response("Empty request body", -32600)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                return self._create_error_response(f"Parse error: {str(e)}", -32700)

            # Validate JSON-RPC structure
            if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
                return self._create_error_response("Invalid JSON-RPC request", -32600)

            method = data.get("method")
            if not method:
                return self._create_error_response("Missing method", -32600)

            request_id = data.get("id")
            params = data.get("params", {})

            # Route to appropriate handler
            handler = self._get_method_handler(method)
            if not handler:
                return self._create_error_response(f"Method not found: {method}", -32601, request_id)

            try:
                result = await handler(params, request_id)

                # Handle notifications (no response expected)
                if request_id is None:
                    return JSONResponse(content={}, status_code=202)

                # Handle streaming responses
                if isinstance(result, dict) and result.get("_stream"):
                    return self._create_streaming_response(result, request_id)

                return self._create_success_response(result, request_id)

            except Exception as e:
                logger.exception(f"Error handling method {method}")
                return self._create_error_response(str(e), -32603, request_id)

        except Exception as e:
            logger.exception("Error handling MCP request")
            return self._create_error_response(f"Internal error: {str(e)}", -32603)

    def _get_method_handler(self, method: str):
        """Get the appropriate handler for an MCP method."""
        handlers = {
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "prompts/list": self._handle_list_prompts,
            "prompts/get": self._handle_get_prompt,
            "logging/setLevel": self._handle_set_log_level,
            "ping": self._handle_ping,
        }
        return handlers.get(method)

    async def _handle_initialize(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Handle initialize request - establishes MCP session."""
        client_info = params.get("clientInfo", {})

        # Generate session ID
        self.session_id = str(uuid.uuid4())

        logger.info(f"Initializing MCP session {self.session_id} with client {client_info}")

        return InitializeResponse(
            capabilities=self.capabilities,
            server_info={
                "name": self.server_name,
                "version": self.server_version,
            },
        ).model_dump()

    async def _handle_initialized(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Handle initialized notification."""
        self.initialized = True
        logger.info("MCP session initialized")
        return {}

    async def _handle_list_tools(self, params: Dict[str, Any], request_id: Union[str, int]):
        """List available tools."""
        tools = []

        # Get tools from LitAPI if available
        if hasattr(self._server.lit_api, "get_tools"):
            api_tools = await self._maybe_await(self._server.lit_api.get_tools())
            tools.extend(api_tools)

        # Add default tools
        tools.extend(self._get_default_tools())

        return ListToolsResponse(tools=tools).model_dump()

    async def _handle_call_tool(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Execute a tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        # Call tool implementation
        if hasattr(self._server.lit_api, "call_tool"):
            result = await self._maybe_await(self._server.lit_api.call_tool(tool_name, arguments))
        else:
            result = await self._call_default_tool(tool_name, arguments)

        return CallToolResponse(content=[{"type": "text", "text": str(result)}]).model_dump()

    async def _handle_list_resources(self, params: Dict[str, Any], request_id: Union[str, int]):
        """List available resources."""
        resources = []

        # Get resources from LitAPI if available
        if hasattr(self._server.lit_api, "get_resources"):
            api_resources = await self._maybe_await(self._server.lit_api.get_resources())
            resources.extend(api_resources)

        # Add default resources
        resources.extend(self._get_default_resources())

        return ListResourcesResponse(resources=resources).model_dump()

    async def _handle_read_resource(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Read a specific resource."""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")

        # Read resource implementation
        if hasattr(self._server.lit_api, "read_resource"):
            content = await self._maybe_await(self._server.lit_api.read_resource(uri))
        else:
            content = await self._read_default_resource(uri)

        return ReadResourceResponse(contents=[{"type": "text", "text": str(content)}]).model_dump()

    async def _handle_list_prompts(self, params: Dict[str, Any], request_id: Union[str, int]):
        """List available prompts."""
        prompts = []

        # Get prompts from LitAPI if available
        if hasattr(self._server.lit_api, "get_prompts"):
            api_prompts = await self._maybe_await(self._server.lit_api.get_prompts())
            prompts.extend(api_prompts)

        # Add default prompts
        prompts.extend(self._get_default_prompts())

        return ListPromptsResponse(prompts=prompts).model_dump()

    async def _handle_get_prompt(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Get a specific prompt."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise ValueError("Prompt name is required")

        # Get prompt implementation
        if hasattr(self._server.lit_api, "get_prompt"):
            result = await self._maybe_await(self._server.lit_api.get_prompt(name, arguments))
        else:
            result = await self._get_default_prompt(name, arguments)

        return result

    async def _handle_set_log_level(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Set logging level."""
        level = params.get("level")
        if level:
            logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
        return {}

    async def _handle_ping(self, params: Dict[str, Any], request_id: Union[str, int]):
        """Handle ping request."""
        return {}

    def _get_default_tools(self) -> List[MCPTool]:
        """Get default tools provided by LitServe."""
        return [
            MCPTool(
                name="predict",
                description="Run model prediction on input data",
                input_schema={
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "Input data for prediction"}},
                    "required": ["input"],
                },
            ),
            MCPTool(
                name="health_check",
                description="Check server health status",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    def _get_default_resources(self) -> List[MCPResource]:
        """Get default resources provided by LitServe."""
        return [
            MCPResource(
                uri="litserve://server/info",
                name="Server Information",
                description="Information about the LitServe server",
                mime_type="application/json",
            ),
            MCPResource(
                uri="litserve://server/capabilities",
                name="Server Capabilities",
                description="Server capabilities and configuration",
                mime_type="application/json",
            ),
        ]

    def _get_default_prompts(self) -> List[MCPPrompt]:
        """Get default prompts provided by LitServe."""
        return [
            MCPPrompt(
                name="analyze_prediction",
                description="Analyze and explain a model prediction",
                arguments=[
                    {"name": "input", "description": "The input data", "required": True},
                    {"name": "output", "description": "The model output", "required": True},
                ],
            )
        ]

    async def _call_default_tool(self, name: str, arguments: Dict[str, Any]):
        """Execute default tools."""
        if name == "predict":
            input_data = arguments.get("input")
            if not input_data:
                raise ValueError("Input data is required for prediction")

            # Use LitAPI predict method
            context = {}
            decoded_input = self._server.lit_api.decode_request({"input": input_data})

            if self._server.lit_api.enable_async:
                result = await self._server.lit_api.predict(decoded_input, context=context)
            else:
                result = self._server.lit_api.predict(decoded_input, context=context)

            return self._server.lit_api.encode_response(result, context=context)

        if name == "health_check":
            return {"status": "healthy", "timestamp": time.time()}

        raise ValueError(f"Unknown tool: {name}")

    async def _read_default_resource(self, uri: str):
        """Read default resources."""
        if uri == "litserve://server/info":
            return {
                "name": self.server_name,
                "version": self.server_version,
                "session_id": self.session_id,
                "initialized": self.initialized,
                "timestamp": time.time(),
            }
        if uri == "litserve://server/capabilities":
            return self.capabilities
        raise ValueError(f"Unknown resource: {uri}")

    async def _get_default_prompt(self, name: str, arguments: Dict[str, str]):
        """Get default prompts."""
        if name == "analyze_prediction":
            input_data = arguments.get("input", "")
            output_data = arguments.get("output", "")

            prompt_text = f"""Please analyze this model prediction:

Input: {input_data}
Output: {output_data}

Provide insights about:
1. The relationship between input and output
2. Potential reasoning behind the prediction
3. Any notable patterns or observations
"""

            return GetPromptResponse(
                description="Analyze and explain a model prediction",
                messages=[MCPMessage(role="user", content=prompt_text)],
            ).model_dump()
        raise ValueError(f"Unknown prompt: {name}")

    async def _maybe_await(self, obj):
        """Await if coroutine, otherwise return directly."""
        if inspect.iscoroutine(obj):
            return await obj
        return obj

    def _create_success_response(self, result: Any, request_id: Union[str, int]):
        """Create a successful JSON-RPC response."""
        return JSONResponse(
            content=JSONRPCResponse(id=request_id, result=result).model_dump(),
            headers={"Content-Type": "application/json"},
        )

    def _create_error_response(self, message: str, code: int, request_id: Union[str, int, None] = None):
        """Create an error JSON-RPC response."""
        return JSONResponse(
            content=JSONRPCErrorResponse(id=request_id, error=JSONRPCError(code=code, message=message)).model_dump(),
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

    def _create_streaming_response(self, result: Dict[str, Any], request_id: Union[str, int]):
        """Create a streaming response for MCP."""

        async def generate():
            # Start with the response header
            yield f"data: {json.dumps(JSONRPCResponse(id=request_id, result=result).model_dump())}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # RESTful endpoints for debugging/development
    async def list_tools_rest(self, request: Request):
        """REST endpoint to list tools."""
        result = await self._handle_list_tools({}, "rest")
        return JSONResponse(content=result)

    async def list_resources_rest(self, request: Request):
        """REST endpoint to list resources."""
        result = await self._handle_list_resources({}, "rest")
        return JSONResponse(content=result)

    async def list_prompts_rest(self, request: Request):
        """REST endpoint to list prompts."""
        result = await self._handle_list_prompts({}, "rest")
        return JSONResponse(content=result)
