from dataclasses import dataclass

from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@dataclass
class MCPAccessBinding:
    binding_id: int
    server_name: str
    endpoint_url: str
    transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP
    server_version: str | None = None
    server_alias: str | None = None
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    def to_dict(self) -> dict:
        return {
            "binding_id": self.binding_id,
            "server_name": self.server_name,
            "endpoint_url": self.endpoint_url,
            "transport_type": self.transport_type.value
            if isinstance(self.transport_type, MCPRemoteTransportType)
            else self.transport_type,
            "server_version": self.server_version,
            "server_alias": self.server_alias,
            "workspace": self.workspace,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
        }
