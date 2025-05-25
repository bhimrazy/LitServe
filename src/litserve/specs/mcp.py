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
minimal configuration. Supports the latest 2025-03-26 specification with Streamable HTTP transport.

Key Features:
- Streamable HTTP transport with single /mcp endpoint
- Session management with Mcp-Session-Id header
- SSE streaming for real-time responses
- Tool registration decorators
- Integration with LitAPI predict functions
- Stateful and stateless operation modes
- JSON-RPC 2.0 compliance
- Event batching support
- Resumability with event stores

"""

import asyncio
import json
import re
import traceback
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from litserve.specs.base import LitSpec

if TYPE_CHECKING:
    pass

# Constants for Streamable HTTP transport
MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
LAST_EVENT_ID_HEADER = "Last-Event-ID"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"
SESSION_ID_PATTERN = re.compile(r"^[\x21-\x7E]+$")  # Visible ASCII characters only
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB limit for messages


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

    protocol_version: str = Field("2025-03-26", alias="protocolVersion")
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


# Streamable HTTP transport models and utilities


class EventMessage(BaseModel):
    """Event message for SSE streaming."""

    id: Optional[str] = None
    event: str = "message"
    data: str
    retry: Optional[int] = None


class SessionInfo(BaseModel):
    """Session information for stateful connections."""

    session_id: str
    created_at: datetime
    last_activity: datetime
    active: bool = True


class MCPSession:
    """Session management for MCP connections."""

    def __init__(self, session_id: str):
        if not SESSION_ID_PATTERN.match(session_id):
            raise ValueError("Session ID must only contain visible ASCII characters (0x21-0x7E)")

        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.active = True
        self.event_store: List[EventMessage] = []
        self._event_counter = 0

    def generate_event_id(self) -> str:
        """Generate a unique event ID for this session."""
        self._event_counter += 1
        return f"{self.session_id}-{self._event_counter}"

    def store_event(self, event: EventMessage) -> None:
        """Store an event for replay capability."""
        if not event.id:
            event.id = self.generate_event_id()
        self.event_store.append(event)
        self.last_activity = datetime.now()

    def get_events_after(self, last_event_id: Optional[str] = None) -> List[EventMessage]:
        """Get events after a specific event ID for replay."""
        if not last_event_id:
            return self.event_store

        # Find the index of the last event
        start_index = 0
        for i, event in enumerate(self.event_store):
            if event.id == last_event_id:
                start_index = i + 1
                break

        return self.event_store[start_index:]

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class ToolRegistration:
    """Registry for MCP tools with decorator support."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, name: str, description: str, input_schema: Dict[str, Any]):
        """Decorator for registering MCP tools."""

        def decorator(func):
            self._tools[name] = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
                "handler": func,
            }
            return func

        return decorator

    def get_tools(self) -> List[MCPTool]:
        """Get all registered tools as MCPTool objects."""
        return [
            MCPTool(name=tool["name"], description=tool["description"], input_schema=tool["input_schema"])
            for tool in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a registered tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")

        handler = self._tools[name]["handler"]
        return handler(**arguments)


# Global tool registry for decorator pattern
_global_tool_registry = ToolRegistration()


def mcp_tool(name: str, description: str, input_schema: Dict[str, Any]):
    """Global decorator for registering MCP tools."""
    return _global_tool_registry.tool(name, description, input_schema)


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

    def __init__(
        self,
        server_name: str = "LitServe MCP Server",
        server_version: str = "1.0.0",
        enable_sessions: bool = True,
        stateless_mode: bool = False,
    ):
        super().__init__()
        self.server_name = server_name
        self.server_version = server_version
        self.capabilities = MCPCapabilities()
        self.initialized = False
        self.enable_sessions = enable_sessions
        self.stateless_mode = stateless_mode

        # Session management for Streamable HTTP transport
        self.sessions: Dict[str, MCPSession] = {}
        self.tool_registry = ToolRegistration()

        # Single MCP endpoint with multiple HTTP methods (Streamable HTTP pattern)
        self.add_endpoint("/mcp", self.handle_mcp_request, ["POST", "GET", "DELETE"])
        self.add_endpoint("/mcp", self.handle_cors_preflight, ["OPTIONS"])

        # Legacy REST endpoints for debugging (optional)
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
        """Handle MCP Streamable HTTP requests with support for POST, GET, and DELETE."""
        method = request.method

        # Extract session ID from header if present
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)

        if method == "POST":
            return await self._handle_post_request(request, session_id)
        if method == "GET":
            return await self._handle_get_request(request, session_id)
        if method == "DELETE":
            return await self._handle_delete_request(request, session_id)

        return JSONResponse(
            content={"error": "Method not allowed"}, status_code=405, headers={"Access-Control-Allow-Origin": "*"}
        )

    async def _handle_post_request(self, request: Request, session_id: Optional[str]):
        """Handle POST requests (JSON-RPC messages)."""
        try:
            # Parse JSON-RPC request
            body = await request.body()

            # Check message size limit
            if len(body) > MAX_MESSAGE_SIZE:
                raise ValueError(f"Message too large: {len(body)} bytes (max {MAX_MESSAGE_SIZE})")

            data = json.loads(body)

            # Handle batch requests (JSON-RPC 2.0 batching)
            if isinstance(data, list):
                return await self._handle_batch_request(data, session_id)

            # Validate JSON-RPC format
            if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
                raise ValueError("Invalid JSON-RPC request")

            mcp_request = MCPRequest(**data)

            # Get or create session if sessions are enabled
            session = None
            if self.enable_sessions and not self.stateless_mode:
                session = await self._get_or_create_session(session_id)

            # Route to appropriate handler
            result = await self._route_method(mcp_request, session)

            response = MCPResponse(id=mcp_request.id, result=result)

            # Set session ID header in response if new session was created
            headers = {"Access-Control-Allow-Origin": "*", "Content-Type": CONTENT_TYPE_JSON}
            if session and not session_id:
                headers[MCP_SESSION_ID_HEADER] = session.session_id

            return JSONResponse(content=response.model_dump(exclude_none=True, by_alias=True), headers=headers)

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
                headers={"Access-Control-Allow-Origin": "*", "Content-Type": CONTENT_TYPE_JSON},
            )

    async def _handle_batch_request(self, batch_data: List[Dict[str, Any]], session_id: Optional[str]):
        """Handle JSON-RPC batch requests."""
        responses = []
        session = None

        if self.enable_sessions and not self.stateless_mode:
            session = await self._get_or_create_session(session_id)

        for item in batch_data:
            try:
                if not isinstance(item, dict) or item.get("jsonrpc") != "2.0":
                    continue

                mcp_request = MCPRequest(**item)
                result = await self._route_method(mcp_request, session)
                response_data = MCPResponse(id=mcp_request.id, result=result).model_dump(
                    exclude_none=True, by_alias=True
                )
                responses.append(response_data)
            except Exception as e:
                responses.append(
                    MCPResponse(id=item.get("id"), error=MCPError(code=-32603, message=str(e))).model_dump(
                        exclude_none=True, by_alias=True
                    )
                )

        headers = {"Access-Control-Allow-Origin": "*", "Content-Type": CONTENT_TYPE_JSON}
        if session and not session_id:
            headers[MCP_SESSION_ID_HEADER] = session.session_id

        return JSONResponse(content=responses, headers=headers)

    async def _handle_get_request(self, request: Request, session_id: Optional[str]):
        """Handle GET requests (SSE streaming)."""
        # Validate Accept header for SSE
        accept_header = request.headers.get("accept", "")
        if CONTENT_TYPE_SSE not in accept_header:
            return JSONResponse(
                content={"error": "Not Acceptable: Client must accept text/event-stream"},
                status_code=406,
                headers={"Access-Control-Allow-Origin": "*"},
            )

        # Get or validate session
        session = None
        if self.enable_sessions:
            if not session_id:
                return JSONResponse(
                    content={"error": "Bad Request: Session ID required for SSE"},
                    status_code=400,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            session = self.sessions.get(session_id)
            if not session:
                return JSONResponse(
                    content={"error": "Not Found: Invalid session ID"},
                    status_code=404,
                    headers={"Access-Control-Allow-Origin": "*"},
                )

        # Handle resumability with Last-Event-ID
        last_event_id = request.headers.get(LAST_EVENT_ID_HEADER)

        return await self._create_sse_response(session, last_event_id)

    async def _handle_delete_request(self, request: Request, session_id: Optional[str]):
        """Handle DELETE requests (session termination)."""
        if not self.enable_sessions:
            return JSONResponse(
                content={"error": "Sessions not enabled"}, status_code=405, headers={"Access-Control-Allow-Origin": "*"}
            )

        if not session_id:
            return JSONResponse(
                content={"error": "Bad Request: Session ID required"},
                status_code=400,
                headers={"Access-Control-Allow-Origin": "*"},
            )

        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.active = False
            del self.sessions[session_id]

            return JSONResponse(
                content={"message": "Session terminated"}, status_code=204, headers={"Access-Control-Allow-Origin": "*"}
            )

        return JSONResponse(
            content={"error": "Not Found: Invalid session ID"},
            status_code=404,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def _get_or_create_session(self, session_id: Optional[str]) -> MCPSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_activity()
            return session

        # Create new session
        new_session_id = session_id or self._generate_session_id()
        session = MCPSession(new_session_id)
        self.sessions[new_session_id] = session
        return session

    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return uuid.uuid4().hex

    async def _create_sse_response(self, session: Optional[MCPSession], last_event_id: Optional[str]):
        """Create SSE streaming response."""

        async def event_generator():
            # Send initial connection event
            yield "event: connected\n"
            session_data = {"type": "connected", "session_id": session.session_id if session else None}
            yield f"data: {json.dumps(session_data)}\n\n"

            # Replay events if last_event_id is provided and session exists
            if session and last_event_id:
                events_to_replay = session.get_events_after(last_event_id)
                for event in events_to_replay:
                    if event.id:
                        yield f"id: {event.id}\n"
                    yield f"event: {event.event}\n"
                    yield f"data: {event.data}\n\n"

            # Keep connection alive for new events
            # In a real implementation, this would listen for new events from the session
            # For now, we'll just send periodic ping events
            while True:
                try:
                    await asyncio.sleep(30)  # Send ping every 30 seconds
                    ping_event = EventMessage(
                        event="ping", data=json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()})
                    )
                    if session:
                        session.store_event(ping_event)

                    if ping_event.id:
                        yield f"id: {ping_event.id}\n"
                    yield f"event: {ping_event.event}\n"
                    yield f"data: {ping_event.data}\n\n"

                except asyncio.CancelledError:
                    # Connection closed
                    break

        headers = {
            "Content-Type": CONTENT_TYPE_SSE,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }

        if session:
            headers[MCP_SESSION_ID_HEADER] = session.session_id

        return StreamingResponse(content=event_generator(), media_type=CONTENT_TYPE_SSE, headers=headers)

    async def _route_method(self, request: MCPRequest, session: Optional[MCPSession] = None) -> Dict[str, Any]:
        """Route MCP method to appropriate handler."""
        method = request.method
        params = request.params or {}

        if method == "initialize":
            return await self._handle_initialize(params, session)
        if method == "tools/list":
            return await self._handle_tools_list(params, session)
        if method == "tools/call":
            return await self._handle_tools_call(params, session)
        if method == "resources/list":
            return await self._handle_resources_list(params, session)
        if method == "resources/read":
            return await self._handle_resources_read(params, session)
        if method == "prompts/list":
            return await self._handle_prompts_list(params, session)
        if method == "prompts/get":
            return await self._handle_prompts_get(params, session)
        if method == "ping":
            return {}
        raise ValueError(f"Method not found: {method}")

    async def _handle_initialize(self, params: Dict[str, Any], session: Optional[MCPSession] = None) -> Dict[str, Any]:
        """Handle MCP initialize method."""
        MCPInitializeParams(**params)  # Validate params
        self.initialized = True

        server_info = MCPServerInfo(name=self.server_name, version=self.server_version)

        # Create result using alias names for proper JSON representation
        result_dict = {
            "protocolVersion": "2025-03-26",
            "capabilities": self.capabilities.model_dump(),
            "serverInfo": server_info.model_dump(),
        }

        # Create using model_validate to parse from dict with aliases
        result = MCPInitializeResult.model_validate(result_dict)

        return result.model_dump(by_alias=True)

    async def _handle_tools_list(self, params: Dict[str, Any], session: Optional[MCPSession] = None) -> Dict[str, Any]:
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

        # Add tools from global tool registry
        global_tools = _global_tool_registry.get_tools()
        tools.extend([tool.model_dump(by_alias=True) for tool in global_tools])

        # Add tools from instance tool registry
        instance_tools = self.tool_registry.get_tools()
        tools.extend([tool.model_dump(by_alias=True) for tool in instance_tools])

        # Add default tools
        default_tools = self._get_default_tools()
        tools.extend([tool.model_dump(by_alias=True) for tool in default_tools])

        return {"tools": tools}

    async def _handle_tools_call(self, params: Dict[str, Any], session: Optional[MCPSession] = None) -> Dict[str, Any]:
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
                pass  # Fall through to registered tools

        # Try global tool registry
        try:
            result = _global_tool_registry.call_tool(name, arguments)
            return MCPToolCallResult(content=[MCPToolCallContent(text=str(result))]).model_dump(by_alias=True)
        except ValueError:
            pass  # Tool not found in global registry
        except Exception as e:
            return MCPToolCallResult(content=[MCPToolCallContent(text=f"Error: {str(e)}")], is_error=True).model_dump(
                by_alias=True
            )

        # Try instance tool registry
        try:
            result = self.tool_registry.call_tool(name, arguments)
            return MCPToolCallResult(content=[MCPToolCallContent(text=str(result))]).model_dump(by_alias=True)
        except ValueError:
            pass  # Tool not found in instance registry
        except Exception as e:
            return MCPToolCallResult(content=[MCPToolCallContent(text=f"Error: {str(e)}")], is_error=True).model_dump(
                by_alias=True
            )

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

        raise ValueError(f"Unknown tool: {name}")

    async def _handle_resources_list(
        self, params: Dict[str, Any], session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
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

    async def _handle_resources_read(
        self, params: Dict[str, Any], session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
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

    async def _handle_prompts_list(
        self, params: Dict[str, Any], session: Optional[MCPSession] = None
    ) -> Dict[str, Any]:
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

    async def _handle_prompts_get(self, params: Dict[str, Any], session: Optional[MCPSession] = None) -> Dict[str, Any]:
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
    "mcp_tool",
    "ToolRegistration",
    "MCPSession",
]
