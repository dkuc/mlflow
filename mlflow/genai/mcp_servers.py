"""MLflow GenAI MCP Server Registry SDK functions."""
from __future__ import annotations

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPServer
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking.client import MlflowClient


# ---------------------------------------------------------------------------
# Convenience registration helpers
# ---------------------------------------------------------------------------


def register_mcp_server(
    *,
    server_json: dict,
    display_name: str | None = None,
    source: str | None = None,
    status: str = "draft",
    tools: list | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    """Register an MCP server from a server.json payload, auto-creating the parent server."""
    client = MlflowClient()
    ver = client.create_mcp_server_version(
        server_json=server_json,
        display_name=display_name,
        source=source,
        status=status,
        tools=tools,
    )
    if create_access_bindings_from_remotes:
        for remote in (server_json.get("remotes") or []):
            url = remote.get("url")
            transport = remote.get("type", "streamable-http")
            if url:
                client.create_mcp_access_binding(
                    server_name=ver.name,
                    endpoint_url=url,
                    transport_type=transport,
                    server_version=ver.version,
                )
    return ver


def register_mcp_server_from_url(
    *,
    url: str,
    display_name: str | None = None,
    source: str | None = None,
    status: str = "draft",
    tools: list | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    """Fetch a server.json payload from a URL and register it."""
    import urllib.request

    with urllib.request.urlopen(url) as resp:
        import json

        server_json = json.loads(resp.read().decode("utf-8"))
    return register_mcp_server(
        server_json=server_json,
        display_name=display_name,
        source=source or url,
        status=status,
        tools=tools,
        create_access_bindings_from_remotes=create_access_bindings_from_remotes,
    )


# ---------------------------------------------------------------------------
# MCPServer CRUD
# ---------------------------------------------------------------------------


def create_mcp_server(
    *,
    name: str,
    description: str | None = None,
    icon: str | None = None,
) -> MCPServer:
    return MlflowClient().create_mcp_server(name=name, description=description, icon=icon)


def get_mcp_server(*, name: str) -> MCPServer:
    return MlflowClient().get_mcp_server(name)


def search_mcp_servers(
    *,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPServer]:
    return MlflowClient().search_mcp_servers(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_server(
    *,
    name: str,
    display_name: str | None = None,
    description: str | None = None,
    icon: str | None = None,
    latest_version: str | None = None,
) -> MCPServer:
    return MlflowClient().update_mcp_server(
        name=name,
        display_name=display_name,
        description=description,
        icon=icon,
        latest_version=latest_version,
    )


def delete_mcp_server(*, name: str) -> None:
    MlflowClient().delete_mcp_server(name)


# ---------------------------------------------------------------------------
# MCPServerVersion CRUD
# ---------------------------------------------------------------------------


def create_mcp_server_version(
    *,
    server_json: dict,
    display_name: str | None = None,
    source: str | None = None,
    status: str = "draft",
    tools: list | None = None,
    create_access_bindings_from_remotes: bool = False,
) -> MCPServerVersion:
    return register_mcp_server(
        server_json=server_json,
        display_name=display_name,
        source=source,
        status=status,
        tools=tools,
        create_access_bindings_from_remotes=create_access_bindings_from_remotes,
    )


def get_mcp_server_version(*, name: str, version: str) -> MCPServerVersion:
    return MlflowClient().get_mcp_server_version(name, version)


def get_mcp_server_version_by_alias(*, name: str, alias: str) -> MCPServerVersion:
    return MlflowClient().get_mcp_server_version_by_alias(name, alias)


def get_latest_mcp_server_version(*, name: str) -> MCPServerVersion:
    return MlflowClient().get_latest_mcp_server_version(name)


def search_mcp_server_versions(
    *,
    name: str,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPServerVersion]:
    return MlflowClient().search_mcp_server_versions(
        name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_server_version(
    *,
    name: str,
    version: str,
    display_name: str | None = None,
    status: str | None = None,
    tools: list | None = None,
) -> MCPServerVersion:
    return MlflowClient().update_mcp_server_version(
        name=name,
        version=version,
        display_name=display_name,
        status=status,
        tools=tools,
    )


def delete_mcp_server_version(*, name: str, version: str) -> None:
    MlflowClient().delete_mcp_server_version(name, version)


# ---------------------------------------------------------------------------
# MCPAccessBinding CRUD
# ---------------------------------------------------------------------------


def create_mcp_access_binding(
    *,
    server_name: str,
    endpoint_url: str,
    transport_type: str = "streamable-http",
    server_version: str | None = None,
    server_alias: str | None = None,
) -> MCPAccessBinding:
    return MlflowClient().create_mcp_access_binding(
        server_name=server_name,
        endpoint_url=endpoint_url,
        transport_type=transport_type,
        server_version=server_version,
        server_alias=server_alias,
    )


def get_mcp_access_binding(*, server_name: str, binding_id: int) -> MCPAccessBinding:
    return MlflowClient().get_mcp_access_binding(server_name, binding_id)


def search_mcp_access_bindings(
    *,
    server_name: str | None = None,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[MCPAccessBinding]:
    return MlflowClient().search_mcp_access_bindings(
        server_name=server_name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


def update_mcp_access_binding(
    *,
    server_name: str,
    binding_id: int,
    endpoint_url: str | None = None,
    transport_type: str | None = None,
    server_version: str | None = None,
    server_alias: str | None = None,
) -> MCPAccessBinding:
    return MlflowClient().update_mcp_access_binding(
        server_name=server_name,
        binding_id=binding_id,
        endpoint_url=endpoint_url,
        transport_type=transport_type,
        server_version=server_version,
        server_alias=server_alias,
    )


def delete_mcp_access_binding(*, server_name: str, binding_id: int) -> None:
    MlflowClient().delete_mcp_access_binding(server_name, binding_id)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def set_mcp_server_tag(*, name: str, key: str, value: str) -> None:
    MlflowClient().set_mcp_server_tag(name, key, value)


def delete_mcp_server_tag(*, name: str, key: str) -> None:
    MlflowClient().delete_mcp_server_tag(name, key)


def set_mcp_server_version_tag(*, name: str, version: str, key: str, value: str) -> None:
    MlflowClient().set_mcp_server_version_tag(name, version, key, value)


def delete_mcp_server_version_tag(*, name: str, version: str, key: str) -> None:
    MlflowClient().delete_mcp_server_version_tag(name, version, key)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def set_mcp_server_alias(*, name: str, alias: str, version: str) -> None:
    MlflowClient().set_mcp_server_alias(name, alias, version)


def delete_mcp_server_alias(*, name: str, alias: str) -> None:
    MlflowClient().delete_mcp_server_alias(name, alias)


# ---------------------------------------------------------------------------
# Trace linking
# ---------------------------------------------------------------------------


def link_mcp_server_versions_to_trace(
    *,
    trace_id: str,
    mcp_servers: list[MCPServerVersion],
) -> None:
    MlflowClient().link_mcp_server_versions_to_trace(trace_id, mcp_servers)
