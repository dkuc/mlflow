from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mlflow.utils.workspace_utils import resolve_entity_workspace_name

if TYPE_CHECKING:
    from mlflow.entities.mcp_access_binding import MCPAccessBinding


class MCPStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"

    def __str__(self):
        return self.value


class MCPRemoteTransportType(str, Enum):
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class MCPTool:
    name: str
    title: str | None = None
    description: str | None = None
    input_schema: dict | None = None
    output_schema: dict | None = None
    annotations: dict | None = None
    icons: list | None = None
    execution: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "annotations": self.annotations,
            "icons": self.icons,
            "execution": self.execution,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "MCPTool":
        return cls(
            name=data.get("name", ""),
            title=data.get("title"),
            description=data.get("description"),
            input_schema=data.get("inputSchema"),
            output_schema=data.get("outputSchema"),
            annotations=data.get("annotations"),
            icons=data.get("icons"),
            execution=data.get("execution"),
        )


@dataclass(frozen=True)
class MCPServerTag:
    key: str
    value: str


@dataclass
class MCPServer:
    name: str
    display_name: str | None = None
    description: str | None = None
    icon: str | None = None
    workspace: str | None = None
    status: MCPStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    access_bindings: list["MCPAccessBinding"] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def __post_init__(self):
        self.workspace = resolve_entity_workspace_name(self.workspace)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "workspace": self.workspace,
            "status": self.status.value if self.status else None,
            "tags": self.tags,
            "aliases": self.aliases,
            "access_bindings": [
                {
                    "binding_id": b.binding_id,
                    "endpoint_url": b.endpoint_url,
                    "transport_type": b.transport_type.value
                    if isinstance(b.transport_type, MCPRemoteTransportType)
                    else b.transport_type,
                    "server_version": b.server_version,
                    "server_alias": b.server_alias,
                }
                for b in self.access_bindings
            ],
            "latest_version": self.latest_version,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
        }
