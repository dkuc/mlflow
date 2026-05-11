"""REST client implementation of MCP Server Registry mixin."""
from __future__ import annotations

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

_MCP_API_PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"


def _raise_on_error(response):
    from mlflow.utils.rest_utils import verify_rest_response

    verify_rest_response(response, _MCP_API_PREFIX)


def _parse_tool(t: dict) -> MCPTool:
    return MCPTool(
        name=t.get("name", ""),
        title=t.get("title"),
        description=t.get("description"),
        input_schema=t.get("inputSchema"),
        output_schema=t.get("outputSchema"),
        annotations=t.get("annotations"),
        icons=t.get("icons"),
        execution=t.get("execution"),
    )


def _parse_binding(data: dict) -> MCPAccessBinding:
    return MCPAccessBinding(
        binding_id=data["binding_id"],
        server_name=data.get("server_name", ""),
        endpoint_url=data["endpoint_url"],
        transport_type=MCPRemoteTransportType(data.get("transport_type", "streamable-http")),
        server_version=data.get("server_version"),
        server_alias=data.get("server_alias"),
        created_by=data.get("created_by"),
        last_updated_by=data.get("last_updated_by"),
        creation_timestamp=data.get("creation_timestamp"),
        last_updated_timestamp=data.get("last_updated_timestamp"),
    )


def _parse_version(data: dict) -> MCPServerVersion:
    tools = None
    if data.get("tools") is not None:
        tools = [_parse_tool(t) for t in data["tools"]]
    return MCPServerVersion(
        name=data["name"],
        version=data["version"],
        server_json=data["server_json"],
        display_name=data.get("display_name"),
        status=MCPStatus(data.get("status", "draft")),
        tools=tools,
        aliases=data.get("aliases", []),
        tags=data.get("tags", {}),
        source=data.get("source"),
        workspace=data.get("workspace"),
        created_by=data.get("created_by"),
        last_updated_by=data.get("last_updated_by"),
        creation_timestamp=data.get("creation_timestamp"),
        last_updated_timestamp=data.get("last_updated_timestamp"),
    )


def _parse_server(data: dict) -> MCPServer:
    bindings = [
        MCPAccessBinding(
            binding_id=b["binding_id"],
            server_name=data["name"],
            endpoint_url=b["endpoint_url"],
            transport_type=MCPRemoteTransportType(b.get("transport_type", "streamable-http")),
            server_version=b.get("server_version"),
            server_alias=b.get("server_alias"),
        )
        for b in data.get("access_bindings", [])
    ]
    aliases = {a["alias"]: a["version"] for a in data.get("aliases", [])}
    status_str = data.get("status")
    status = MCPStatus(status_str) if status_str else None
    return MCPServer(
        name=data["name"],
        display_name=data.get("display_name"),
        description=data.get("description"),
        icon=data.get("icon"),
        workspace=data.get("workspace"),
        status=status,
        tags=data.get("tags", {}),
        aliases=aliases,
        access_bindings=bindings,
        latest_version=data.get("latest_version"),
        created_by=data.get("created_by"),
        last_updated_by=data.get("last_updated_by"),
        creation_timestamp=data.get("creation_timestamp"),
        last_updated_timestamp=data.get("last_updated_timestamp"),
    )


class RestMCPServerRegistryMixin:
    """REST client mixin for MCP Server Registry.

    Uses http_request (not _call_endpoint) because MCP server registry
    endpoints use FastAPI + Pydantic, not Flask + protobuf. Auth credentials
    are obtained from self.get_host_creds() as usual.
    """

    def _mcp_request(self, method: str, path: str, body: dict | None = None):
        from mlflow.utils.rest_utils import http_request

        host_creds = self.get_host_creds()
        endpoint = f"{_MCP_API_PREFIX}{path}"
        kwargs = {"method": method, "endpoint": endpoint}
        if body is not None:
            kwargs["json"] = body
        response = http_request(host_creds=host_creds, **kwargs)
        _raise_on_error(response)
        return response.json()

    # --- MCPServer operations ---

    def create_mcp_server(
        self,
        name: str,
        description: str | None = None,
        icon: str | None = None,
    ) -> MCPServer:
        data = self._mcp_request(
            "POST", "/", {"name": name, "description": description, "icon": icon}
        )
        return _parse_server(data["mcp_server"])

    def get_mcp_server(self, name: str) -> MCPServer:
        data = self._mcp_request("GET", f"/{name}")
        return _parse_server(data["mcp_server"])

    def search_mcp_servers(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServer]:
        params = f"?max_results={max_results}"
        if filter_string:
            from urllib.parse import quote

            params += f"&filter_string={quote(filter_string)}"
        if page_token:
            params += f"&page_token={page_token}"
        data = self._mcp_request("GET", f"/{params}")
        servers = [_parse_server(s) for s in data.get("mcp_servers", [])]
        return PagedList(servers, data.get("next_page_token"))

    def update_mcp_server(
        self,
        name: str,
        description: str | None = None,
        display_name: str | None = None,
        icon: str | None = None,
        latest_version: str | None = None,
    ) -> MCPServer:
        body = {}
        if description is not None:
            body["description"] = description
        if display_name is not None:
            body["display_name"] = display_name
        if icon is not None:
            body["icon"] = icon
        if latest_version is not None:
            body["latest_version"] = latest_version
        data = self._mcp_request("PATCH", f"/{name}", body)
        return _parse_server(data["mcp_server"])

    def delete_mcp_server(self, name: str) -> None:
        self._mcp_request("DELETE", f"/{name}")

    # --- MCPServerVersion operations ---

    def create_mcp_server_version(
        self,
        server_json: dict,
        display_name: str | None = None,
        source: str | None = None,
        status: str | None = None,
        tools: list | None = None,
    ) -> MCPServerVersion:
        name = server_json.get("name", "")
        body: dict = {"server_json": server_json}
        if display_name is not None:
            body["display_name"] = display_name
        if source is not None:
            body["source"] = source
        if status is not None:
            body["status"] = status
        if tools is not None:
            body["tools"] = [
                t.to_dict() if isinstance(t, MCPTool) else t for t in tools
            ]
        data = self._mcp_request("POST", f"/{name}/versions", body)
        return _parse_version(data["mcp_server_version"])

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        data = self._mcp_request("GET", f"/{name}/versions/{version}")
        return _parse_version(data["mcp_server_version"])

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        data = self._mcp_request("GET", f"/{name}/aliases/{alias}")
        return _parse_version(data["mcp_server_version"])

    def get_latest_mcp_server_version(self, name: str) -> MCPServerVersion:
        return self.get_mcp_server_version_by_alias(name, "latest")

    def search_mcp_server_versions(
        self,
        name: str,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServerVersion]:
        params = f"?max_results={max_results}"
        if filter_string:
            from urllib.parse import quote

            params += f"&filter_string={quote(filter_string)}"
        if page_token:
            params += f"&page_token={page_token}"
        data = self._mcp_request("GET", f"/{name}/versions{params}")
        versions = [_parse_version(v) for v in data.get("mcp_server_versions", [])]
        return PagedList(versions, data.get("next_page_token"))

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        display_name: str | None = None,
        status: str | None = None,
        tools: list | None = None,
    ) -> MCPServerVersion:
        body = {}
        if display_name is not None:
            body["display_name"] = display_name
        if status is not None:
            body["status"] = status
        if tools is not None:
            body["tools"] = [
                t.to_dict() if isinstance(t, MCPTool) else t for t in tools
            ]
        data = self._mcp_request("PATCH", f"/{name}/versions/{version}", body)
        return _parse_version(data["mcp_server_version"])

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        self._mcp_request("DELETE", f"/{name}/versions/{version}")

    # --- MCPAccessBinding operations ---

    def create_mcp_access_binding(
        self,
        server_name: str,
        endpoint_url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
    ) -> MCPAccessBinding:
        body = {
            "endpoint_url": endpoint_url,
            "transport_type": transport_type.value
            if isinstance(transport_type, MCPRemoteTransportType)
            else transport_type,
        }
        if server_version is not None:
            body["server_version"] = server_version
        if server_alias is not None:
            body["server_alias"] = server_alias
        data = self._mcp_request("POST", f"/{server_name}/bindings", body)
        return _parse_binding(data["binding"])

    def get_mcp_access_binding(self, server_name: str, binding_id: int) -> MCPAccessBinding:
        data = self._mcp_request("GET", f"/{server_name}/bindings/{binding_id}")
        return _parse_binding(data["binding"])

    def search_mcp_access_bindings(
        self,
        server_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessBinding]:
        if server_name is not None:
            data = self._mcp_request("GET", f"/{server_name}/bindings")
        else:
            data = self._mcp_request("GET", "/bindings")
        bindings = [_parse_binding(b) for b in data.get("bindings", [])]
        return PagedList(bindings, None)

    def update_mcp_access_binding(
        self,
        server_name: str,
        binding_id: int,
        server_version: str | None = None,
        server_alias: str | None = None,
        endpoint_url: str | None = None,
        transport_type: MCPRemoteTransportType | None = None,
    ) -> MCPAccessBinding:
        body = {}
        if server_version is not None:
            body["server_version"] = server_version
        if server_alias is not None:
            body["server_alias"] = server_alias
        if endpoint_url is not None:
            body["endpoint_url"] = endpoint_url
        if transport_type is not None:
            body["transport_type"] = (
                transport_type.value
                if isinstance(transport_type, MCPRemoteTransportType)
                else transport_type
            )
        data = self._mcp_request("PATCH", f"/{server_name}/bindings/{binding_id}", body)
        return _parse_binding(data["binding"])

    def delete_mcp_access_binding(self, server_name: str, binding_id: int) -> None:
        self._mcp_request("DELETE", f"/{server_name}/bindings/{binding_id}")

    # --- Tag operations ---

    def set_mcp_server_tag(self, name: str, key: str, value: str) -> None:
        self._mcp_request("POST", f"/{name}/tags", {"key": key, "value": value})

    def delete_mcp_server_tag(self, name: str, key: str) -> None:
        self._mcp_request("DELETE", f"/{name}/tags/{key}")

    def set_mcp_server_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self._mcp_request("POST", f"/{name}/versions/{version}/tags", {"key": key, "value": value})

    def delete_mcp_server_version_tag(self, name: str, version: str, key: str) -> None:
        self._mcp_request("DELETE", f"/{name}/versions/{version}/tags/{key}")

    # --- Alias operations ---

    def set_mcp_server_alias(self, name: str, alias: str, version: str) -> None:
        self._mcp_request("POST", f"/{name}/aliases", {"alias": alias, "version": version})

    def delete_mcp_server_alias(self, name: str, alias: str) -> None:
        self._mcp_request("DELETE", f"/{name}/aliases/{alias}")

    # --- Trace linking ---

    def link_mcp_server_versions_to_trace(
        self,
        trace_id: str,
        mcp_servers: list[MCPServerVersion],
    ) -> None:
        for ver in mcp_servers:
            self._mcp_request(
                "POST",
                f"/{ver.name}/versions/{ver.version}/link-trace",
                {"trace_id": trace_id},
            )

    def get_mcp_server_versions_for_trace(
        self,
        trace_id: str,
    ) -> list[MCPServerVersion]:
        raise MlflowException(
            "get_mcp_server_versions_for_trace is not supported via REST client",
            error_code=ENDPOINT_NOT_FOUND,
        )
