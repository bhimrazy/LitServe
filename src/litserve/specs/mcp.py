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
"""Model Context Protocol (MCP) specification for LitServe.

This module implements the MCP specification allowing any LitServe application to become an MCP-compliant server with
minimal configuration.

"""

import json
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from litserve.specs.base import LitSpec

if TYPE_CHECKING:
    pass


class MCPError(BaseModel):
    """MCP error response model."""

    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class MCPRequest(BaseModel):
    """MCP JSON-RPC request model."""

    jsonrpc: str = Field(..., description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(None, description="Request ID")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")


class MCPResponse(BaseModel):
    """MCP JSON-RPC response model."""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None


class MCPCapabilities(BaseModel):
    """MCP server capabilities."""

    tools: Optional[Dict[str, Any]] = Field(default_factory=dict)
    resources: Optional[Dict[str, Any]] = Field(default_factory=dict)
    prompts: Optional[Dict[str, Any]] = Field(default_factory=dict)
    logging: Optional[Dict[str, Any]] = None


class MCPServerInfo(BaseModel):
    """MCP server information."""

    name: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")


class MCPInitializeParams(BaseModel):
    """MCP initialize parameters."""

    protocol_version: str = Field(..., alias="protocolVersion")
    capabilities: Dict[str, Any]
    client_info: MCPServerInfo = Field(..., alias="clientInfo")


class MCPInitializeResult(BaseModel):
    """MCP initialize result."""

    protocol_version: str = Field("2024-11-05", alias="protocolVersion")
    capabilities: MCPCapabilities
    server_info: MCPServerInfo = Field(..., alias="serverInfo")


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict[str, Any] = Field(..., alias="inputSchema", description="Tool input schema")


class MCPResource(BaseModel):
    """MCP resource definition."""

    uri: str = Field(..., description="Resource URI")
    name: str = Field(..., description="Resource name")
    description: Optional[str] = Field(None, description="Resource description")
    mime_type: Optional[str] = Field(None, alias="mimeType", description="Resource MIME type")


class MCPPrompt(BaseModel):
    """MCP prompt definition."""

    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    arguments: Optional[List[Dict[str, Any]]] = Field(None, description="Prompt arguments")


class MCPMessage(BaseModel):
    """MCP message definition."""

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class GetPromptResponse(BaseModel):
    """Response for prompts/get method."""

    description: Optional[str] = None
    messages: List[MCPMessage]


class MCPToolCallContent(BaseModel):
    """Tool call content."""

    type: str = "text"
    text: str


class MCPToolCallResult(BaseModel):
    """Tool call result."""

    content: List[MCPToolCallContent]
    is_error: bool = Field(False, alias="isError")


class MCPResourceContent(BaseModel):
    """Resource content."""

    uri: str
    mime_type: Optional[str] = Field(None, alias="mimeType")
    text: Optional[str] = None
    blob: Optional[bytes] = None


class MCPSpec(LitSpec):
    """Model Context Protocol specification for LitServe.

    This specification implements the MCP protocol allowing any LitServe application
    to become an MCP-compliant server. It provides default implementations for
    common functionality while allowing customization through optional LitAPI methods.

    Args:
        server_name: Name of the MCP server (default: "LitServe MCP Server")
        server_version: Version of the MCP server (default: "1.0.0")

    Examples:
        Basic usage:
        ```python
        import litserve as ls

        class MyAPI(ls.LitAPI):
            def predict(self, x):
                return {"result": x}

        server = ls.LitServer(MyAPI(), spec=ls.MCPSpec())
        ```

        With custom tools:
        ```python
        class MyAPI(ls.LitAPI):
            def get_tools(self):
                return [ls.MCPTool(
                    name="custom_tool",
                    description="My custom tool",
                    input_schema={"type": "object", "properties": {...}}
                )]

            def call_tool(self, name, arguments):
                if name == "custom_tool":
                    return "Custom result"
        ```

    """

    def __init__(self, server_name: str = "LitServe MCP Server", server_version: str = "1.0.0"):
        super().__init__()
        self.server_name = server_name
        self.server_version = server_version
        self.capabilities = MCPCapabilities()
        self.initialized = False

        # Register MCP endpoints
        self.add_endpoint("/mcp", self.handle_mcp_request, ["POST"])
        self.add_endpoint("/mcp", self.handle_cors_preflight, ["OPTIONS"])

        # Add REST endpoints for debugging
        self.add_endpoint("/mcp/tools", self.get_tools_rest, ["GET"])
        self.add_endpoint("/mcp/resources", self.get_resources_rest, ["GET"])
        self.add_endpoint("/mcp/prompts", self.get_prompts_rest, ["GET"])

    @property
    def stream(self):
        """MCP supports streaming by default."""
        return True

    def setup(self, server):
        """Setup the MCP specification with the server."""
        super().setup(server)
        # Store reference to the API for accessing custom methods
        self.api = server.lit_api

    async def handle_cors_preflight(self, request: Request):
        """Handle CORS preflight requests."""
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        )

    async def handle_mcp_request(self, request: Request):
        """Handle MCP JSON-RPC requests."""
        try:
            # Parse JSON-RPC request
            body = await request.body()
            data = json.loads(body)

            # Validate JSON-RPC format
            if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
                raise ValueError("Invalid JSON-RPC request")

            mcp_request = MCPRequest(**data)

            # Route to appropriate handler
            result = await self._route_method(mcp_request)

            response = MCPResponse(id=mcp_request.id, result=result)

            return JSONResponse(
                content=response.model_dump(exclude_none=True, by_alias=True),
                headers={"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
            )

        except Exception as e:
            error_response = MCPResponse(
                id=data.get("id") if "data" in locals() else None,
                error=MCPError(
                    code=-32603, message=f"Internal error: {str(e)}", data={"traceback": traceback.format_exc()}
                ),
            )
            return JSONResponse(
                content=error_response.model_dump(exclude_none=True, by_alias=True),
                status_code=400,
                headers={"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
            )

    async def _route_method(self, request: MCPRequest) -> Dict[str, Any]:
        """Route MCP method to appropriate handler."""
        method = request.method
        params = request.params or {}

        if method == "initialize":
            return await self._handle_initialize(params)
        if method == "tools/list":
            return await self._handle_tools_list(params)
        if method == "tools/call":
            return await self._handle_tools_call(params)
        if method == "resources/list":
            return await self._handle_resources_list(params)
        if method == "resources/read":
            return await self._handle_resources_read(params)
        if method == "prompts/list":
            return await self._handle_prompts_list(params)
        if method == "prompts/get":
            return await self._handle_prompts_get(params)
        if method == "ping":
            return {}
        raise ValueError(f"Method not found: {method}")

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize method."""
        MCPInitializeParams(**params)  # Validate params
        self.initialized = True

        server_info = MCPServerInfo(name=self.server_name, version=self.server_version)

        # Create result using alias names for proper JSON representation
        result_dict = {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities.model_dump(),
            "serverInfo": server_info.model_dump(),
        }

        # Create using model_validate to parse from dict with aliases
        result = MCPInitializeResult.model_validate(result_dict)

        return result.model_dump(by_alias=True)

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list method."""
        tools = []

        # Get tools from API if available
        if hasattr(self.api, "get_tools"):
            try:
                custom_tools = self.api.get_tools()
                if custom_tools:
                    tools.extend([tool.model_dump(by_alias=True) for tool in custom_tools])
            except Exception:
                pass  # Ignore errors in custom tool retrieval

        # Add default tools
        default_tools = self._get_default_tools()
        tools.extend([tool.model_dump(by_alias=True) for tool in default_tools])

        return {"tools": tools}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call method."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise ValueError("Tool name is required")

        # Try custom tool handler first
        if hasattr(self.api, "call_tool"):
            try:
                result = self.api.call_tool(name, arguments)
                return MCPToolCallResult(content=[MCPToolCallContent(text=str(result))]).model_dump(by_alias=True)
            except Exception:
                pass  # Fall through to default tools

        # Handle default tools
        if name == "predict":
            input_data = arguments.get("input")
            if input_data is None:
                raise ValueError("Input is required for predict tool")

            # Use the API's predict method
            try:
                decoded_input = self.api.decode_request(input_data)
                prediction = self.api.predict(decoded_input)
                result = self.api.encode_response(prediction)

                return MCPToolCallResult(content=[MCPToolCallContent(text=str(result))]).model_dump(by_alias=True)
            except Exception as e:
                return MCPToolCallResult(
                    content=[MCPToolCallContent(text=f"Error: {str(e)}")], is_error=True
                ).model_dump(by_alias=True)

        elif name == "health_check":
            return MCPToolCallResult(content=[MCPToolCallContent(text="Server is healthy")]).model_dump(by_alias=True)

        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list method."""
        resources = []

        # Get resources from API if available
        if hasattr(self.api, "get_resources"):
            try:
                custom_resources = self.api.get_resources()
                if custom_resources:
                    resources.extend([resource.model_dump(by_alias=True) for resource in custom_resources])
            except Exception:
                pass  # Ignore errors in custom resource retrieval

        # Add default resources
        default_resources = self._get_default_resources()
        resources.extend([resource.model_dump(by_alias=True) for resource in default_resources])

        return {"resources": resources}

    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read method."""
        uri = params.get("uri")
        if not uri:
            raise ValueError("Resource URI is required")

        # Try custom resource handler first
        if hasattr(self.api, "read_resource"):
            try:
                content = self.api.read_resource(uri)
                return {
                    "contents": [
                        MCPResourceContent(
                            uri=uri,
                            mime_type="application/json",
                            text=json.dumps(content) if not isinstance(content, str) else content,
                        ).model_dump(by_alias=True)
                    ]
                }
            except Exception:
                pass  # Fall through to default resources

        # Handle default resources
        if uri == "litserve://server/info":
            content = {
                "server_name": self.server_name,
                "server_version": self.server_version,
                "api_class": self.api.__class__.__name__,
                "initialized": self.initialized,
            }
            return {
                "contents": [
                    MCPResourceContent(uri=uri, mime_type="application/json", text=json.dumps(content)).model_dump(
                        by_alias=True
                    )
                ]
            }

        if uri == "litserve://server/capabilities":
            return {
                "contents": [
                    MCPResourceContent(
                        uri=uri, mime_type="application/json", text=json.dumps(self.capabilities.model_dump())
                    ).model_dump(by_alias=True)
                ]
            }

        raise ValueError(f"Unknown resource: {uri}")

    async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list method."""
        prompts = []

        # Get prompts from API if available
        if hasattr(self.api, "get_prompts"):
            try:
                custom_prompts = self.api.get_prompts()
                if custom_prompts:
                    prompts.extend([prompt.model_dump(by_alias=True) for prompt in custom_prompts])
            except Exception:
                pass  # Ignore errors in custom prompt retrieval

        # Add default prompts
        default_prompts = self._get_default_prompts()
        prompts.extend([prompt.model_dump(by_alias=True) for prompt in default_prompts])

        return {"prompts": prompts}

    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get method."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise ValueError("Prompt name is required")

        # Try custom prompt handler first
        if hasattr(self.api, "get_prompt"):
            try:
                return self.api.get_prompt(name, arguments)
            except Exception:
                pass  # Fall through to default prompts

        # Handle default prompts
        if name == "analyze_prediction":
            input_data = arguments.get("input", "N/A")
            output_data = arguments.get("output", "N/A")

            response = GetPromptResponse(
                description=f"Analyze prediction for input: {input_data}",
                messages=[
                    MCPMessage(
                        role="user",
                        content=f"Please analyze this prediction:\nInput: {input_data}\nOutput: {output_data}\n\n"
                        "Provide insights about the model's performance and any patterns you notice.",
                    )
                ],
            )
            return response.model_dump(by_alias=True)

        raise ValueError(f"Unknown prompt: {name}")

    def _get_default_tools(self) -> List[MCPTool]:
        """Get default MCP tools."""
        tools_data = [
            {
                "name": "predict",
                "description": "Make a prediction using the LitServe API",
                "inputSchema": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "Input data for prediction"}},
                    "required": ["input"],
                },
            },
            {
                "name": "health_check",
                "description": "Check if the server is healthy",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        return [MCPTool.model_validate(tool_data) for tool_data in tools_data]

    def _get_default_resources(self) -> List[MCPResource]:
        """Get default MCP resources."""
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
                description="Server capabilities and supported features",
                mime_type="application/json",
            ),
        ]

    def _get_default_prompts(self) -> List[MCPPrompt]:
        """Get default MCP prompts."""
        return [
            MCPPrompt(
                name="analyze_prediction",
                description="Analyze a prediction made by the model",
                arguments=[
                    {"name": "input", "description": "The input that was provided to the model", "required": False},
                    {"name": "output", "description": "The output produced by the model", "required": False},
                ],
            )
        ]

    async def get_tools_rest(self):
        """REST endpoint for listing tools."""
        result = await self._handle_tools_list({})
        return JSONResponse(content=result)

    async def get_resources_rest(self):
        """REST endpoint for listing resources."""
        result = await self._handle_resources_list({})
        return JSONResponse(content=result)

    async def get_prompts_rest(self):
        """REST endpoint for listing prompts."""
        result = await self._handle_prompts_list({})
        return JSONResponse(content=result)


# Export all the classes that users might need
__all__ = [
    "MCPSpec",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPMessage",
    "GetPromptResponse",
    "MCPCapabilities",
    "MCPServerInfo",
]
