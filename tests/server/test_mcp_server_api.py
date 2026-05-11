"""Integration tests for MCP Server Registry FastAPI endpoints."""
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server.mcp_server_api import mcp_server_router
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


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(mcp_server_router)
    with mock.patch(
        "mlflow.server.mcp_server_api._get_tracking_store", return_value=store
    ):
        with TestClient(app) as c:
            yield c


# ---- Server CRUD ---


def test_create_server(client):
    resp = client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mcp_server"]["name"] == "io.github.test/server"


def test_get_server(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server")
    assert resp.status_code == 200
    assert resp.json()["mcp_server"]["name"] == "io.github.test/server"


def test_search_servers(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/")
    assert resp.status_code == 200
    servers = resp.json()["mcp_servers"]
    assert len(servers) >= 1


def test_update_server(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    resp = client.patch(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server",
        json={"description": "Updated description"},
    )
    assert resp.status_code == 200
    assert resp.json()["mcp_server"]["description"] == "Updated description"


def test_delete_server(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    resp = client.delete("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server")
    assert resp.status_code == 200
    resp2 = client.get("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server")
    assert resp2.status_code == 404


# ---- Version CRUD ---


def test_create_version(client):
    resp = client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    assert resp.status_code == 200
    ver = resp.json()["mcp_server_version"]
    assert ver["version"] == "1.0.0"
    assert ver["status"] == "draft"


def test_create_version_name_mismatch(client):
    resp = client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.other/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    assert resp.status_code == 400


def test_get_version(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions/1.0.0")
    assert resp.status_code == 200
    assert resp.json()["mcp_server_version"]["version"] == "1.0.0"


def test_list_versions(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions")
    assert resp.status_code == 200
    assert len(resp.json()["mcp_server_versions"]) == 1


def test_update_version_status(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.patch(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions/1.0.0",
        json={"status": "active"},
    )
    assert resp.status_code == 200
    assert resp.json()["mcp_server_version"]["status"] == "active"


def test_delete_version(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.delete(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions/1.0.0"
    )
    assert resp.status_code == 200


# ---- Tags ---


def test_set_server_tag(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    resp = client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/tags",
        json={"key": "team", "value": "platform"},
    )
    assert resp.status_code == 200


def test_delete_server_tag(client):
    client.post("/ajax-api/3.0/mlflow/mcp-servers/", json={"name": "io.github.test/server"})
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/tags",
        json={"key": "team", "value": "platform"},
    )
    resp = client.delete("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/tags/team")
    assert resp.status_code == 200


# ---- Aliases ---


def test_set_alias(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/aliases",
        json={"alias": "production", "version": "1.0.0"},
    )
    assert resp.status_code == 200


def test_get_alias(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/aliases",
        json={"alias": "production", "version": "1.0.0"},
    )
    resp = client.get(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/aliases/production"
    )
    assert resp.status_code == 200
    assert resp.json()["mcp_server_version"]["version"] == "1.0.0"


def test_delete_alias(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/aliases",
        json={"alias": "production", "version": "1.0.0"},
    )
    resp = client.delete(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/aliases/production"
    )
    assert resp.status_code == 200


# ---- Access Bindings ---


def test_create_binding(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    resp = client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/bindings",
        json={
            "endpoint_url": "https://example.com/mcp",
            "server_version": "1.0.0",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["binding"]["endpoint_url"] == "https://example.com/mcp"


def test_list_bindings(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/bindings",
        json={"endpoint_url": "https://example.com/mcp", "server_version": "1.0.0"},
    )
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/bindings")
    assert resp.status_code == 200
    assert len(resp.json()["bindings"]) == 1


def test_workspace_list_bindings(client):
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/versions",
        json={"server_json": _SERVER_JSON},
    )
    client.post(
        "/ajax-api/3.0/mlflow/mcp-servers/io.github.test/server/bindings",
        json={"endpoint_url": "https://example.com/mcp", "server_version": "1.0.0"},
    )
    resp = client.get("/ajax-api/3.0/mlflow/mcp-servers/bindings")
    assert resp.status_code == 200
    assert len(resp.json()["bindings"]) == 1
