import pytest

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import (
    MCPRemoteTransportType,
    MCPServer,
    MCPServerTag,
    MCPStatus,
    MCPTool,
)
from mlflow.entities.mcp_server_alias import MCPServerAlias
from mlflow.entities.mcp_server_version import MCPServerVersion


def test_mcp_status_values():
    assert MCPStatus.DRAFT == "draft"
    assert MCPStatus.ACTIVE == "active"
    assert MCPStatus.DEPRECATED == "deprecated"
    assert MCPStatus.DELETED == "deleted"


def test_mcp_remote_transport_type_values():
    assert MCPRemoteTransportType.STREAMABLE_HTTP == "streamable-http"
    assert MCPRemoteTransportType.SSE == "sse"


def test_mcp_tool_creation():
    tool = MCPTool(
        name="web_search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert tool.name == "web_search"
    assert tool.description == "Search the web"
    assert tool.title is None
    assert tool.output_schema is None


def test_mcp_tool_is_frozen():
    tool = MCPTool(name="tool")
    with pytest.raises(Exception):
        tool.name = "other"


def test_mcp_server_tag():
    tag = MCPServerTag(key="team", value="platform")
    assert tag.key == "team"
    assert tag.value == "platform"


def test_mcp_server_default_fields():
    server = MCPServer(name="io.github.test/server")
    assert server.name == "io.github.test/server"
    assert server.display_name is None
    assert server.description is None
    assert server.icon is None
    assert server.workspace == "default"
    assert server.status is None
    assert server.tags == {}
    assert server.aliases == {}
    assert server.access_bindings == []
    assert server.latest_version is None
    assert server.created_by is None
    assert server.last_updated_by is None
    assert server.creation_timestamp is None
    assert server.last_updated_timestamp is None


def test_mcp_server_full_fields():
    server = MCPServer(
        name="io.github.test/server",
        display_name="Test Server",
        description="A test server",
        status=MCPStatus.ACTIVE,
        tags={"team": "platform"},
        aliases={"production": "1.0.0"},
        latest_version="1.0.0",
        created_by="user1",
        creation_timestamp=1000,
    )
    assert server.display_name == "Test Server"
    assert server.status == MCPStatus.ACTIVE
    assert server.tags["team"] == "platform"
    assert server.aliases["production"] == "1.0.0"


def test_mcp_server_version_defaults():
    version = MCPServerVersion(
        name="io.github.test/server",
        version="1.0.0",
        server_json={"name": "io.github.test/server", "version": "1.0.0"},
    )
    assert version.status == MCPStatus.DRAFT
    assert version.tools is None
    assert version.aliases == []
    assert version.tags == {}
    assert version.source is None


def test_mcp_server_version_with_tools():
    tools = [MCPTool(name="search", description="Search tool")]
    version = MCPServerVersion(
        name="io.github.test/server",
        version="1.0.0",
        server_json={"name": "io.github.test/server", "version": "1.0.0"},
        tools=tools,
        status=MCPStatus.ACTIVE,
    )
    assert len(version.tools) == 1
    assert version.tools[0].name == "search"
    assert version.status == MCPStatus.ACTIVE


def test_mcp_access_binding_defaults():
    binding = MCPAccessBinding(
        binding_id=1,
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
    )
    assert binding.binding_id == 1
    assert binding.server_name == "io.github.test/server"
    assert binding.endpoint_url == "https://example.com/mcp"
    assert binding.transport_type == MCPRemoteTransportType.STREAMABLE_HTTP
    assert binding.server_version is None
    assert binding.server_alias is None


def test_mcp_access_binding_with_alias():
    binding = MCPAccessBinding(
        binding_id=2,
        server_name="io.github.test/server",
        endpoint_url="https://example.com/mcp",
        transport_type=MCPRemoteTransportType.SSE,
        server_alias="production",
    )
    assert binding.transport_type == MCPRemoteTransportType.SSE
    assert binding.server_alias == "production"
    assert binding.server_version is None


def test_mcp_server_alias():
    alias = MCPServerAlias(
        name="io.github.test/server",
        alias="production",
        version="1.0.0",
    )
    assert alias.name == "io.github.test/server"
    assert alias.alias == "production"
    assert alias.version == "1.0.0"


def test_mcp_server_alias_is_frozen():
    alias = MCPServerAlias(name="test/server", alias="prod", version="1.0.0")
    with pytest.raises(Exception):
        alias.alias = "staging"


def test_mcp_tool_to_dict():
    tool = MCPTool(
        name="search",
        description="Search tool",
        input_schema={"type": "object"},
    )
    d = tool.to_dict()
    assert d["name"] == "search"
    assert d["inputSchema"] == {"type": "object"}
    assert "outputSchema" not in d


def test_mcp_tool_from_dict():
    data = {"name": "search", "inputSchema": {"type": "object"}, "description": "Search"}
    tool = MCPTool.from_dict(data)
    assert tool.name == "search"
    assert tool.input_schema == {"type": "object"}
    assert tool.description == "Search"


def test_mcp_server_workspace_resolution():
    server = MCPServer(name="test/server")
    assert server.workspace == "default"

    server2 = MCPServer(name="test/server", workspace="custom")
    assert server2.workspace == "custom"


def test_mcp_server_version_workspace_resolution():
    version = MCPServerVersion(
        name="test/server",
        version="1.0.0",
        server_json={"name": "test/server", "version": "1.0.0"},
    )
    assert version.workspace == "default"


def test_mcp_access_binding_workspace_resolution():
    binding = MCPAccessBinding(
        binding_id=1,
        server_name="test/server",
        endpoint_url="https://example.com/mcp",
    )
    assert binding.workspace == "default"


def test_mcp_server_to_dict():
    server = MCPServer(
        name="test/server",
        status=MCPStatus.ACTIVE,
        tags={"team": "platform"},
    )
    d = server.to_dict()
    assert d["name"] == "test/server"
    assert d["status"] == "active"
    assert d["tags"] == {"team": "platform"}


def test_mcp_server_version_to_dict():
    version = MCPServerVersion(
        name="test/server",
        version="1.0.0",
        server_json={"name": "test/server", "version": "1.0.0"},
        tools=[MCPTool(name="search")],
    )
    d = version.to_dict()
    assert d["name"] == "test/server"
    assert d["version"] == "1.0.0"
    assert len(d["tools"]) == 1
    assert d["tools"][0]["name"] == "search"


def test_mcp_access_binding_to_dict():
    binding = MCPAccessBinding(
        binding_id=1,
        server_name="test/server",
        endpoint_url="https://example.com/mcp",
    )
    d = binding.to_dict()
    assert d["binding_id"] == 1
    assert d["transport_type"] == "streamable-http"
