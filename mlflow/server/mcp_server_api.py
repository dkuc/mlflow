"""FastAPI router for MCP Server Registry."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus
from mlflow.exceptions import MlflowException
from mlflow.server.handlers import _get_tracking_store

mcp_server_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/mcp-servers",
    tags=["MCP Servers"],
)


def _get_store():
    return _get_tracking_store()


def _mlflow_exception_to_http(e: MlflowException) -> HTTPException:
    return HTTPException(status_code=e.get_http_status_code(), detail=e.message)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ServerJSONPayload(BaseModel):
    model_config = {"extra": "allow"}

    name: str
    version: str
    title: str | None = None
    description: str | None = None
    packages: list[Any] | None = None
    remotes: list[Any] | None = None
    repository: str | None = None
    websiteUrl: str | None = None
    _meta: dict | None = None


class MCPToolPayload(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict | None = None
    outputSchema: dict | None = None
    annotations: dict | None = None
    icons: list | None = None
    execution: dict | None = None


class CreateMCPServerRequest(BaseModel):
    name: str
    description: str | None = None
    icon: str | None = None


class UpdateMCPServerRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    icon: str | None = None
    latest_version: str | None = None


class CreateMCPServerVersionRequest(BaseModel):
    server_json: ServerJSONPayload
    display_name: str | None = None
    status: str = "draft"
    source: str | None = None
    tools: list[MCPToolPayload] | None = None
    create_access_bindings_from_remotes: bool = False


class UpdateMCPServerVersionRequest(BaseModel):
    display_name: str | None = None
    status: str | None = None
    tools: list[MCPToolPayload] | None = None


class CreateMCPAccessBindingRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    endpoint_url: str
    transport_type: str = "streamable-http"


class UpdateMCPAccessBindingRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    endpoint_url: str | None = None
    transport_type: str | None = None


class SetAliasRequest(BaseModel):
    alias: str
    version: str


class SetTagRequest(BaseModel):
    key: str
    value: str


class AliasResponse(BaseModel):
    alias: str
    version: str


class MCPAccessBindingSummaryResponse(BaseModel):
    binding_id: int
    endpoint_url: str
    transport_type: str = "streamable-http"
    server_version: str | None = None
    server_alias: str | None = None


class MCPServerResponse(BaseModel):
    name: str
    display_name: str | None = None
    description: str | None = None
    icon: str | None = None
    status: str | None = None
    access_bindings: list[MCPAccessBindingSummaryResponse] = Field(default_factory=list)
    latest_version: str | None = None
    aliases: list[AliasResponse] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None


class MCPServerVersionResponse(BaseModel):
    name: str
    version: str
    server_json: dict
    display_name: str | None = None
    status: str = "draft"
    tools: list[MCPToolPayload] | None = None
    aliases: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    source: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None


class MCPAccessBindingResponse(BaseModel):
    binding_id: int
    server_name: str
    endpoint_url: str
    transport_type: str = "streamable-http"
    tools: list[MCPToolPayload] | None = None
    server_version: str | None = None
    server_alias: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _server_to_response(server) -> MCPServerResponse:
    return MCPServerResponse(
        name=server.name,
        display_name=server.display_name,
        description=server.description,
        icon=server.icon,
        status=server.status.value if server.status else None,
        access_bindings=[
            MCPAccessBindingSummaryResponse(
                binding_id=b.binding_id,
                endpoint_url=b.endpoint_url,
                transport_type=b.transport_type.value
                if isinstance(b.transport_type, MCPRemoteTransportType)
                else b.transport_type,
                server_version=b.server_version,
                server_alias=b.server_alias,
            )
            for b in server.access_bindings
        ],
        latest_version=server.latest_version,
        aliases=[AliasResponse(alias=k, version=v) for k, v in server.aliases.items()],
        tags=server.tags,
        created_by=server.created_by,
        last_updated_by=server.last_updated_by,
        creation_timestamp=server.creation_timestamp,
        last_updated_timestamp=server.last_updated_timestamp,
    )


def _version_to_response(ver) -> MCPServerVersionResponse:
    tools = None
    if ver.tools is not None:
        tools = [
            MCPToolPayload(
                name=t.name,
                title=t.title,
                description=t.description,
                inputSchema=t.input_schema,
                outputSchema=t.output_schema,
                annotations=t.annotations,
                icons=t.icons,
                execution=t.execution,
            )
            for t in ver.tools
        ]
    return MCPServerVersionResponse(
        name=ver.name,
        version=ver.version,
        server_json=ver.server_json,
        display_name=ver.display_name,
        status=ver.status.value if isinstance(ver.status, MCPStatus) else ver.status,
        tools=tools,
        aliases=ver.aliases,
        tags=ver.tags,
        source=ver.source,
        created_by=ver.created_by,
        last_updated_by=ver.last_updated_by,
        creation_timestamp=ver.creation_timestamp,
        last_updated_timestamp=ver.last_updated_timestamp,
    )


def _binding_to_response(binding, tools=None) -> MCPAccessBindingResponse:
    return MCPAccessBindingResponse(
        binding_id=binding.binding_id,
        server_name=binding.server_name,
        endpoint_url=binding.endpoint_url,
        transport_type=binding.transport_type.value
        if isinstance(binding.transport_type, MCPRemoteTransportType)
        else binding.transport_type,
        tools=tools,
        server_version=binding.server_version,
        server_alias=binding.server_alias,
        created_by=binding.created_by,
        last_updated_by=binding.last_updated_by,
        creation_timestamp=binding.creation_timestamp,
        last_updated_timestamp=binding.last_updated_timestamp,
    )


def _payload_to_tools(tool_payloads):
    from mlflow.entities.mcp_server import MCPTool

    if tool_payloads is None:
        return None
    return [
        MCPTool(
            name=t.name,
            title=t.title,
            description=t.description,
            input_schema=t.inputSchema,
            output_schema=t.outputSchema,
            annotations=t.annotations,
            icons=t.icons,
            execution=t.execution,
        )
        for t in tool_payloads
    ]


# ---------------------------------------------------------------------------
# Routes — IMPORTANT: static paths like /bindings must be registered BEFORE
# the /{name} catch-all to avoid "bindings" being treated as a server name.
# ---------------------------------------------------------------------------


@mcp_server_router.get("/bindings")
def list_bindings_workspace(request: Request):
    store = _get_store()
    filter_string = request.query_params.get("filter_string")
    try:
        result = store.search_mcp_access_bindings(filter_string=filter_string)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"bindings": [_binding_to_response(b).model_dump() for b in result.to_list()]}


# --- MCPServer CRUD ---


@mcp_server_router.post("/")
def create_server(body: CreateMCPServerRequest):
    store = _get_store()
    try:
        server = store.create_mcp_server(
            name=body.name,
            description=body.description,
            icon=body.icon,
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server": _server_to_response(server).model_dump()}


@mcp_server_router.get("/")
def search_servers(request: Request):
    store = _get_store()
    filter_string = request.query_params.get("filter_string")
    max_results = int(request.query_params.get("max_results", 100))
    try:
        result = store.search_mcp_servers(
            filter_string=filter_string, max_results=max_results
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    resp = {"mcp_servers": [_server_to_response(s).model_dump() for s in result.to_list()]}
    if result.token:
        resp["next_page_token"] = result.token
    return resp


@mcp_server_router.get("/{name:path}/versions/{version}/tags/{key}")
def get_version_tag_placeholder(name: str, version: str, key: str):
    raise HTTPException(status_code=405, detail="Use POST to set tags")


@mcp_server_router.get("/{name:path}/versions/{version}")
def get_server_version(name: str, version: str):
    store = _get_store()
    try:
        if version == "latest":
            ver = store.get_latest_mcp_server_version(name)
        else:
            ver = store.get_mcp_server_version(name, version)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server_version": _version_to_response(ver).model_dump()}


@mcp_server_router.patch("/{name:path}/versions/{version}")
def update_server_version(name: str, version: str, body: UpdateMCPServerVersionRequest):
    store = _get_store()
    try:
        ver = store.update_mcp_server_version(
            name=name,
            version=version,
            display_name=body.display_name,
            status=body.status,
            tools=_payload_to_tools(body.tools),
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server_version": _version_to_response(ver).model_dump()}


@mcp_server_router.delete("/{name:path}/versions/{version}")
def delete_server_version(name: str, version: str):
    store = _get_store()
    try:
        store.delete_mcp_server_version(name, version)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.post("/{name:path}/versions/{version}/tags")
def set_version_tag(name: str, version: str, body: SetTagRequest):
    store = _get_store()
    try:
        store.set_mcp_server_version_tag(name, version, body.key, body.value)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.delete("/{name:path}/versions/{version}/tags/{key}")
def delete_version_tag(name: str, version: str, key: str):
    store = _get_store()
    try:
        store.delete_mcp_server_version_tag(name, version, key)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.get("/{name:path}/versions")
def list_server_versions(name: str, request: Request):
    store = _get_store()
    filter_string = request.query_params.get("filter_string")
    try:
        result = store.search_mcp_server_versions(name, filter_string=filter_string)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {
        "mcp_server_versions": [
            _version_to_response(v).model_dump() for v in result.to_list()
        ]
    }


@mcp_server_router.post("/{name:path}/versions")
def create_server_version(name: str, body: CreateMCPServerVersionRequest):
    store = _get_store()
    server_json_dict = body.server_json.model_dump(exclude_none=True)
    if server_json_dict.get("name") != name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"server_json.name '{server_json_dict.get('name')}' must match "
                f"path parameter '{name}'"
            ),
        )
    try:
        ver = store.create_mcp_server_version(
            server_json=server_json_dict,
            display_name=body.display_name,
            source=body.source,
            status=body.status,
            tools=_payload_to_tools(body.tools),
        )
        if body.create_access_bindings_from_remotes:
            for remote in (server_json_dict.get("remotes") or []):
                url = remote.get("url")
                transport = remote.get("type", "streamable-http")
                if url:
                    store.create_mcp_access_binding(
                        server_name=name,
                        endpoint_url=url,
                        transport_type=MCPRemoteTransportType(transport),
                        server_version=ver.version,
                    )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server_version": _version_to_response(ver).model_dump()}


@mcp_server_router.get("/{name:path}/bindings/{binding_id}")
def get_binding(name: str, binding_id: int):
    store = _get_store()
    try:
        binding = store.get_mcp_access_binding(name, binding_id)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"binding": _binding_to_response(binding).model_dump()}


@mcp_server_router.patch("/{name:path}/bindings/{binding_id}")
def update_binding(name: str, binding_id: int, body: UpdateMCPAccessBindingRequest):
    store = _get_store()
    transport = MCPRemoteTransportType(body.transport_type) if body.transport_type else None
    try:
        binding = store.update_mcp_access_binding(
            server_name=name,
            binding_id=binding_id,
            server_version=body.server_version,
            server_alias=body.server_alias,
            endpoint_url=body.endpoint_url,
            transport_type=transport,
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"binding": _binding_to_response(binding).model_dump()}


@mcp_server_router.delete("/{name:path}/bindings/{binding_id}")
def delete_binding(name: str, binding_id: int):
    store = _get_store()
    try:
        store.delete_mcp_access_binding(name, binding_id)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.get("/{name:path}/bindings")
def list_bindings(name: str, request: Request):
    store = _get_store()
    filter_string = request.query_params.get("filter_string")
    try:
        result = store.search_mcp_access_bindings(server_name=name, filter_string=filter_string)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"bindings": [_binding_to_response(b).model_dump() for b in result.to_list()]}


@mcp_server_router.post("/{name:path}/bindings")
def create_binding(name: str, body: CreateMCPAccessBindingRequest):
    store = _get_store()
    try:
        transport = MCPRemoteTransportType(body.transport_type)
        binding = store.create_mcp_access_binding(
            server_name=name,
            endpoint_url=body.endpoint_url,
            transport_type=transport,
            server_version=body.server_version,
            server_alias=body.server_alias,
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"binding": _binding_to_response(binding).model_dump()}


@mcp_server_router.post("/{name:path}/tags")
def set_server_tag(name: str, body: SetTagRequest):
    store = _get_store()
    try:
        store.set_mcp_server_tag(name, body.key, body.value)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.delete("/{name:path}/tags/{key}")
def delete_server_tag(name: str, key: str):
    store = _get_store()
    try:
        store.delete_mcp_server_tag(name, key)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.post("/{name:path}/aliases")
def set_alias(name: str, body: SetAliasRequest):
    store = _get_store()
    try:
        store.set_mcp_server_alias(name, body.alias, body.version)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.get("/{name:path}/aliases/{alias}")
def get_alias(name: str, alias: str):
    store = _get_store()
    try:
        ver = store.get_mcp_server_version_by_alias(name, alias)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server_version": _version_to_response(ver).model_dump()}


@mcp_server_router.delete("/{name:path}/aliases/{alias}")
def delete_alias(name: str, alias: str):
    store = _get_store()
    try:
        store.delete_mcp_server_alias(name, alias)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}


@mcp_server_router.get("/{name:path}")
def get_server(name: str):
    store = _get_store()
    try:
        server = store.get_mcp_server(name)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server": _server_to_response(server).model_dump()}


@mcp_server_router.patch("/{name:path}")
def update_server(name: str, body: UpdateMCPServerRequest):
    store = _get_store()
    try:
        server = store.update_mcp_server(
            name=name,
            description=body.description,
            display_name=body.display_name,
            icon=body.icon,
            latest_version=body.latest_version,
        )
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {"mcp_server": _server_to_response(server).model_dump()}


@mcp_server_router.delete("/{name:path}")
def delete_server(name: str):
    store = _get_store()
    try:
        store.delete_mcp_server(name)
    except MlflowException as e:
        raise _mlflow_exception_to_http(e) from e
    return {}
