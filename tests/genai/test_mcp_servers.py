"""Tests for mlflow.genai MCP server SDK functions."""
from pathlib import Path
from unittest import mock

import pytest

import mlflow.genai
from mlflow.entities.mcp_server import MCPStatus
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

_SERVER_JSON = {
    "name": "io.github.test/sdk-server",
    "version": "1.0.0",
    "description": "SDK test server",
}


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture(autouse=True)
def patch_store(store):
    with mock.patch(
        "mlflow.genai.mcp_servers._store", return_value=store
    ):
        yield


def test_register_mcp_server():
    ver = mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    assert ver.name == "io.github.test/sdk-server"
    assert ver.version == "1.0.0"
    assert ver.status == MCPStatus.DRAFT


def test_register_mcp_server_with_status():
    ver = mlflow.genai.register_mcp_server(server_json=_SERVER_JSON, status="active")
    assert ver.status == MCPStatus.ACTIVE


def test_create_mcp_server():
    server = mlflow.genai.create_mcp_server(name="io.github.test/sdk-server")
    assert server.name == "io.github.test/sdk-server"


def test_get_mcp_server():
    mlflow.genai.create_mcp_server(name="io.github.test/sdk-server")
    server = mlflow.genai.get_mcp_server(name="io.github.test/sdk-server")
    assert server.name == "io.github.test/sdk-server"


def test_search_mcp_servers():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    servers = mlflow.genai.search_mcp_servers()
    assert len(servers.to_list()) >= 1


def test_update_mcp_server():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    updated = mlflow.genai.update_mcp_server(
        name="io.github.test/sdk-server",
        description="Updated via SDK",
    )
    assert updated.description == "Updated via SDK"


def test_delete_mcp_server():
    mlflow.genai.create_mcp_server(name="io.github.test/sdk-server")
    mlflow.genai.delete_mcp_server(name="io.github.test/sdk-server")
    servers = mlflow.genai.search_mcp_servers()
    names = [s.name for s in servers.to_list()]
    assert "io.github.test/sdk-server" not in names


def test_get_mcp_server_version():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    ver = mlflow.genai.get_mcp_server_version(
        name="io.github.test/sdk-server", version="1.0.0"
    )
    assert ver.version == "1.0.0"


def test_update_mcp_server_version():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    updated = mlflow.genai.update_mcp_server_version(
        name="io.github.test/sdk-server",
        version="1.0.0",
        status="active",
    )
    assert updated.status == MCPStatus.ACTIVE


def test_set_and_get_alias():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    mlflow.genai.set_mcp_server_alias(
        name="io.github.test/sdk-server", alias="production", version="1.0.0"
    )
    ver = mlflow.genai.get_mcp_server_version_by_alias(
        name="io.github.test/sdk-server", alias="production"
    )
    assert ver.version == "1.0.0"


def test_create_access_binding():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    binding = mlflow.genai.create_mcp_access_binding(
        server_name="io.github.test/sdk-server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    assert binding.endpoint_url == "https://example.com/mcp"
    assert binding.binding_id is not None


def test_search_access_bindings():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    mlflow.genai.create_mcp_access_binding(
        server_name="io.github.test/sdk-server",
        endpoint_url="https://example.com/mcp",
        server_version="1.0.0",
    )
    bindings = mlflow.genai.search_mcp_access_bindings()
    assert len(bindings.to_list()) >= 1


def test_link_mcp_server_versions_to_trace():
    mlflow.genai.register_mcp_server(server_json=_SERVER_JSON)
    ver = mlflow.genai.get_mcp_server_version(
        name="io.github.test/sdk-server", version="1.0.0"
    )
    # Should not raise
    mlflow.genai.link_mcp_server_versions_to_trace(
        trace_id="test-trace-123", mcp_servers=[ver]
    )
