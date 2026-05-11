from dataclasses import dataclass, field

from mlflow.entities.mcp_server import MCPStatus, MCPTool
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@dataclass
class MCPServerVersion:
    name: str
    version: str
    server_json: dict
    display_name: str | None = None
    status: MCPStatus = MCPStatus.DRAFT
    tools: list[MCPTool] | None = None
    aliases: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    source: str | None = None
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "server_json": self.server_json,
            "display_name": self.display_name,
            "status": self.status.value if isinstance(self.status, MCPStatus) else self.status,
            "tools": [t.to_dict() for t in self.tools] if self.tools else None,
            "aliases": self.aliases,
            "tags": self.tags,
            "source": self.source,
            "workspace": self.workspace,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
        }
