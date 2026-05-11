"""Tests for SqlAlchemyMCPServerRegistryMixin."""
from pathlib import Path

import pytest

from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

_SERVER_JSON = {
    "name": "io.github.test/server",
    "version": "1.0.0",
    "description": "Test server",
}


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


# ---- MCPServer CRUD ---


def test_create_mcp_server(store):
    server = store.create_mcp_server("io.github.test/server", description="A test server")
    assert server.name == "io.github.test/server"
    assert server.description == "A test server"
    assert server.creation_timestamp is not None


def test_create_mcp_server_duplicate_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException, match="already exists") as exc_info:
        store.create_mcp_server("io.github.test/server")
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_get_mcp_server(store):
    store.create_mcp_server("io.github.test/server", description="desc")
    server = store.get_mcp_server("io.github.test/server")
    assert server.name == "io.github.test/server"
    assert server.description == "desc"


def test_get_mcp_server_not_found(store):
    with pytest.raises(MlflowException) as exc_info:
        store.get_mcp_server("nonexistent/server")
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_update_mcp_server(store):
    store.create_mcp_server("io.github.test/server")
    updated = store.update_mcp_server(
        "io.github.test/server",
        description="Updated",
        display_name="Test Server",
    )
    assert updated.description == "Updated"
    assert updated.display_name == "Test Server"


def test_delete_mcp_server(store):
    store.create_mcp_server("io.github.test/server")
    store.delete_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException):
        store.get_mcp_server("io.github.test/server")


def test_search_mcp_servers_empty(store):
    result = store.search_mcp_servers()
    assert result.to_list() == []


def test_search_mcp_servers_returns_all(store):
    store.create_mcp_server("io.github.test/server1")
    store.create_mcp_server("io.github.test/server2")
    result = store.search_mcp_servers()
    assert len(result.to_list()) == 2


def test_search_mcp_servers_filter_by_name(store):
    store.create_mcp_server("io.github.test/alpha")
    store.create_mcp_server("io.github.test/beta")
    result = store.search_mcp_servers(filter_string="name = 'io.github.test/alpha'")
    assert len(result.to_list()) == 1
    assert result.to_list()[0].name == "io.github.test/alpha"


def test_search_mcp_servers_filter_by_status(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE.value
    )
    result = store.search_mcp_servers(filter_string="status = 'active'")
    names = [s.name for s in result.to_list()]
    assert "io.github.test/server" in names


# ---- MCPServerVersion CRUD ---


def test_create_mcp_server_version(store):
    ver = store.create_mcp_server_version(server_json=_SERVER_JSON)
    assert ver.name == "io.github.test/server"
    assert ver.version == "1.0.0"
    assert ver.status == MCPStatus.DRAFT


def test_create_version_auto_creates_server(store):
    ver = store.create_mcp_server_version(server_json=_SERVER_JSON)
    server = store.get_mcp_server("io.github.test/server")
    assert server.name == "io.github.test/server"


def test_create_version_duplicate_raises(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    with pytest.raises(MlflowException) as exc_info:
        store.create_mcp_server_version(server_json=_SERVER_JSON)
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_create_version_missing_name_raises(store):
    with pytest.raises(MlflowException, match="'name'") as exc_info:
        store.create_mcp_server_version(server_json={"version": "1.0.0"})
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_create_version_missing_version_raises(store):
    with pytest.raises(MlflowException, match="'version'") as exc_info:
        store.create_mcp_server_version(server_json={"name": "test/server"})
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_mcp_server_version(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert ver.server_json == _SERVER_JSON


def test_get_mcp_server_version_not_found(store):
    with pytest.raises(MlflowException) as exc_info:
        store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_update_mcp_server_version_status_transitions(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)

    ver = store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE.value
    )
    assert ver.status == MCPStatus.ACTIVE

    ver = store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED.value
    )
    assert ver.status == MCPStatus.DEPRECATED

    ver = store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE.value
    )
    assert ver.status == MCPStatus.ACTIVE


def test_update_mcp_server_version_invalid_transition(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    with pytest.raises(MlflowException, match="Invalid status transition") as exc_info:
        store.update_mcp_server_version(
            "io.github.test/server", "1.0.0", status=MCPStatus.DEPRECATED.value
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_update_mcp_server_version_with_tools(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    tools = [MCPTool(name="search", description="Web search")]
    ver = store.update_mcp_server_version("io.github.test/server", "1.0.0", tools=tools)
    assert ver.tools is not None
    assert ver.tools[0].name == "search"


def test_delete_mcp_server_version(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.delete_mcp_server_version("io.github.test/server", "1.0.0")
    with pytest.raises(MlflowException):
        store.get_mcp_server_version("io.github.test/server", "1.0.0")


def test_get_latest_mcp_server_version(store):
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "1.0.0"}
    )
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "2.0.0"}
    )
    store.update_mcp_server_version(
        "io.github.test/server", "2.0.0", status=MCPStatus.ACTIVE.value
    )
    ver = store.get_latest_mcp_server_version("io.github.test/server")
    assert ver.version == "2.0.0"


def test_get_latest_ignores_drafts(store):
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "1.0.0"}
    )
    store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE.value
    )
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "2.0.0"}
    )
    # 2.0.0 is draft — latest should still resolve to 1.0.0
    ver = store.get_latest_mcp_server_version("io.github.test/server")
    assert ver.version == "1.0.0"


def test_search_mcp_server_versions(store):
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "1.0.0"}
    )
    store.create_mcp_server_version(
        server_json={"name": "io.github.test/server", "version": "2.0.0"}
    )
    result = store.search_mcp_server_versions("io.github.test/server")
    assert len(result.to_list()) == 2


# ---- Tags ---


def test_set_and_delete_mcp_server_tag(store):
    store.create_mcp_server("io.github.test/server")
    store.set_mcp_server_tag("io.github.test/server", "team", "platform")
    server = store.get_mcp_server("io.github.test/server")
    assert server.tags["team"] == "platform"

    store.delete_mcp_server_tag("io.github.test/server", "team")
    server = store.get_mcp_server("io.github.test/server")
    assert "team" not in server.tags


def test_set_and_delete_mcp_server_version_tag(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_version_tag("io.github.test/server", "1.0.0", "env", "prod")
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert ver.tags["env"] == "prod"

    store.delete_mcp_server_version_tag("io.github.test/server", "1.0.0", "env")
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert "env" not in ver.tags


# ---- Aliases ---


def test_set_and_resolve_alias(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    ver = store.get_mcp_server_version_by_alias("io.github.test/server", "production")
    assert ver.version == "1.0.0"


def test_alias_latest_is_reserved(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    with pytest.raises(MlflowException, match="reserved") as exc_info:
        store.set_mcp_server_alias("io.github.test/server", "latest", "1.0.0")
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_version_by_alias_latest(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.update_mcp_server_version(
        "io.github.test/server", "1.0.0", status=MCPStatus.ACTIVE.value
    )
    ver = store.get_mcp_server_version_by_alias("io.github.test/server", "latest")
    assert ver.version == "1.0.0"


def test_delete_alias(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    store.delete_mcp_server_alias("io.github.test/server", "production")
    with pytest.raises(MlflowException):
        store.get_mcp_server_version_by_alias("io.github.test/server", "production")


def test_alias_exposed_on_server(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    server = store.get_mcp_server("io.github.test/server")
    assert "production" in server.aliases
    assert server.aliases["production"] == "1.0.0"


def test_alias_exposed_on_version(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    assert "production" in ver.aliases


# ---- Access Bindings ---


def test_create_mcp_access_binding_with_version(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    binding = store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    assert binding.endpoint_url == "https://example.com/mcp"
    assert binding.server_version == "1.0.0"
    assert binding.server_alias is None
    assert binding.binding_id is not None


def test_create_mcp_access_binding_with_alias(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    binding = store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_alias="production",
    )
    assert binding.server_alias == "production"
    assert binding.server_version is None


def test_create_binding_both_version_and_alias_raises(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.set_mcp_server_alias("io.github.test/server", "production", "1.0.0")
    with pytest.raises(MlflowException) as exc_info:
        store.create_mcp_access_binding(
            server_name="io.github.test/server",
            endpoint_url="https://example.com/mcp",
            server_version="1.0.0",
            server_alias="production",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_create_binding_neither_raises(store):
    store.create_mcp_server("io.github.test/server")
    with pytest.raises(MlflowException) as exc_info:
        store.create_mcp_access_binding(
            server_name="io.github.test/server",
            endpoint_url="https://example.com/mcp",
        )
    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_mcp_access_binding(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    binding = store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    fetched = store.get_mcp_access_binding("io.github.test/server", binding.binding_id)
    assert fetched.endpoint_url == "https://example.com/mcp"


def test_search_mcp_access_bindings(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    result = store.search_mcp_access_bindings()
    assert len(result.to_list()) == 1


def test_search_mcp_access_bindings_by_server(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    result = store.search_mcp_access_bindings(server_name="io.github.test/server")
    assert len(result.to_list()) == 1
    result2 = store.search_mcp_access_bindings(server_name="io.github.other/server")
    assert len(result2.to_list()) == 0


def test_update_mcp_access_binding(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    binding = store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://old.example.com/mcp",
        server_version="1.0.0",
    )
    updated = store.update_mcp_access_binding(
        server_name="io.github.test/server",
        binding_id=binding.binding_id,
        endpoint_url="https://new.example.com/mcp",
    )
    assert updated.endpoint_url == "https://new.example.com/mcp"


def test_delete_mcp_access_binding(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    binding = store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    store.delete_mcp_access_binding("io.github.test/server", binding.binding_id)
    with pytest.raises(MlflowException):
        store.get_mcp_access_binding("io.github.test/server", binding.binding_id)


def test_access_binding_on_server(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    store.create_mcp_access_binding(
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    server = store.get_mcp_server("io.github.test/server")
    assert len(server.access_bindings) == 1
    assert server.access_bindings[0].endpoint_url == "https://example.com/mcp"


# ---- Trace linking ---


def test_link_and_get_mcp_server_versions_for_trace(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    store.link_mcp_server_versions_to_trace("trace-123", [ver])
    linked = store.get_mcp_server_versions_for_trace("trace-123")
    assert len(linked) == 1
    assert linked[0].name == "io.github.test/server"
    assert linked[0].version == "1.0.0"


def test_link_mcp_server_versions_idempotent(store):
    store.create_mcp_server_version(server_json=_SERVER_JSON)
    ver = store.get_mcp_server_version("io.github.test/server", "1.0.0")
    store.link_mcp_server_versions_to_trace("trace-123", [ver])
    store.link_mcp_server_versions_to_trace("trace-123", [ver])
    linked = store.get_mcp_server_versions_for_trace("trace-123")
    assert len(linked) == 1
