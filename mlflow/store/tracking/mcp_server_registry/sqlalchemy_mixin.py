from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.dbmodels.models import (
    SqlMCPAccessBinding,
    SqlMCPServer,
    SqlMCPServerAlias,
    SqlMCPServerTag,
    SqlMCPServerVersion,
    SqlMCPServerVersionTag,
)
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.time import get_current_time_millis

_RESERVED_ALIAS = "latest"

_VALID_TRANSITIONS: dict[MCPStatus, set[MCPStatus]] = {
    MCPStatus.DRAFT: {MCPStatus.ACTIVE, MCPStatus.DELETED},
    MCPStatus.ACTIVE: {MCPStatus.DRAFT, MCPStatus.DEPRECATED},
    MCPStatus.DEPRECATED: {MCPStatus.ACTIVE, MCPStatus.DELETED},
    MCPStatus.DELETED: set(),
}


class SqlAlchemyMCPServerRegistryMixin:
    def _get_mcp_server_or_raise(self, session, name: str) -> SqlMCPServer:
        server = (
            self._get_query(session, SqlMCPServer)
            .options(
                joinedload(SqlMCPServer.tags),
                joinedload(SqlMCPServer.server_aliases),
                joinedload(SqlMCPServer.access_bindings),
                joinedload(SqlMCPServer.server_versions),
            )
            .filter(SqlMCPServer.name == name)
            .first()
        )
        if server is None:
            raise MlflowException(
                f"MCP server '{name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return server

    def _get_mcp_version_or_raise(self, session, name: str, version: str) -> SqlMCPServerVersion:
        ver = (
            self._get_query(session, SqlMCPServerVersion)
            .options(
                joinedload(SqlMCPServerVersion.version_tags),
            )
            .filter(
                SqlMCPServerVersion.name == name,
                SqlMCPServerVersion.version == version,
            )
            .first()
        )
        if ver is None:
            raise MlflowException(
                f"MCP server version '{name}@{version}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return ver

    # --- MCPServer operations ---

    def create_mcp_server(
        self,
        name: str,
        description: str | None = None,
        icon: str | None = None,
    ) -> MCPServer:
        now = get_current_time_millis()
        with self.ManagedSessionMaker() as session:
            server = self._with_workspace_field(
                SqlMCPServer(
                    name=name,
                    description=description,
                    icon=icon,
                    created_at=now,
                    last_updated_at=now,
                )
            )
            try:
                session.add(server)
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"MCP server '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return server.to_mlflow_entity(versions=[], access_bindings=[])

    def get_mcp_server(self, name: str) -> MCPServer:
        with self.ManagedSessionMaker() as session:
            server = self._get_mcp_server_or_raise(session, name)
            return server.to_mlflow_entity()

    def search_mcp_servers(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServer]:
        self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        with self.ManagedSessionMaker() as session:
            query = (
                self._get_query(session, SqlMCPServer)
                .options(
                    joinedload(SqlMCPServer.tags),
                    joinedload(SqlMCPServer.server_aliases),
                    joinedload(SqlMCPServer.access_bindings),
                    joinedload(SqlMCPServer.server_versions),
                )
            )

            filters = _parse_filter_string(filter_string or "")
            if "name" in filters:
                op, val = filters["name"]
                if op == "=":
                    query = query.filter(SqlMCPServer.name == val)
                elif op in ("LIKE", "like"):
                    query = query.filter(SqlMCPServer.name.like(val))
            if "has_access_bindings" in filters:
                _, val = filters["has_access_bindings"]
                if val is True:
                    query = query.filter(SqlMCPServer.access_bindings.any())

            query = query.order_by(SqlMCPServer.created_at.desc())

            # Status is a derived field, so we need post-query filtering
            if "status" in filters:
                servers = query.all()
                op, val = filters["status"]
                if isinstance(val, list):
                    target_statuses = [MCPStatus(v) for v in val]
                    servers = [s for s in servers if _derive_status(s) in target_statuses]
                elif op == "=":
                    servers = [s for s in servers if _derive_status(s) == MCPStatus(val)]
                entities = [s.to_mlflow_entity() for s in servers]
                paged = entities[offset : offset + max_results]
                next_token = None
                if len(entities) > offset + max_results:
                    next_token = SearchUtils.create_page_token(offset + max_results)
                return PagedList(paged, next_token)

            query = query.offset(offset).limit(max_results + 1)
            servers = query.all()
            entities = [s.to_mlflow_entity() for s in servers]
            next_token = None
            if len(entities) > max_results:
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList(entities[:max_results], next_token)

    def update_mcp_server(
        self,
        name: str,
        description: str | None = None,
        display_name: str | None = None,
        icon: str | None = None,
        latest_version: str | None = None,
    ) -> MCPServer:
        with self.ManagedSessionMaker() as session:
            server = self._get_mcp_server_or_raise(session, name)
            if description is not None:
                server.description = description
            if display_name is not None:
                server.display_name = display_name
            if icon is not None:
                server.icon = icon
            if latest_version is not None:
                self._get_mcp_version_or_raise(session, name, latest_version)
                server.latest_version = latest_version
            server.last_updated_at = get_current_time_millis()
            session.flush()
            return server.to_mlflow_entity()

    def delete_mcp_server(self, name: str) -> None:
        with self.ManagedSessionMaker() as session:
            server = self._get_mcp_server_or_raise(session, name)
            session.delete(server)

    # --- MCPServerVersion operations ---

    def create_mcp_server_version(
        self,
        server_json: dict,
        display_name: str | None = None,
        source: str | None = None,
        status: str | None = None,
        tools: list | None = None,
    ) -> MCPServerVersion:
        name = server_json.get("name")
        version_str = server_json.get("version")
        if not name:
            raise MlflowException(
                "server_json must include 'name'", error_code=INVALID_PARAMETER_VALUE
            )
        if not version_str:
            raise MlflowException(
                "server_json must include 'version'", error_code=INVALID_PARAMETER_VALUE
            )

        status_val = MCPStatus(status) if status else MCPStatus.DRAFT
        now = get_current_time_millis()

        tools_json = None
        if tools is not None:
            tools_json = [
                t.to_dict() if isinstance(t, MCPTool) else t for t in tools
            ]

        with self.ManagedSessionMaker() as session:
            # Auto-create parent MCPServer if it doesn't exist
            existing_server = (
                self._get_query(session, SqlMCPServer)
                .filter(SqlMCPServer.name == name)
                .first()
            )
            if existing_server is None:
                existing_server = self._with_workspace_field(
                    SqlMCPServer(
                        name=name,
                        created_at=now,
                        last_updated_at=now,
                    )
                )
                session.add(existing_server)
                session.flush()

            ver = self._with_workspace_field(
                SqlMCPServerVersion(
                    name=name,
                    version=version_str,
                    server_json=server_json,
                    display_name=display_name,
                    status=status_val.value,
                    tools=tools_json,
                    source=source,
                    created_at=now,
                    last_updated_at=now,
                )
            )
            try:
                session.add(ver)
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"MCP server version '{name}@{version_str}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e

            existing_server.last_updated_at = now

            return ver.to_mlflow_entity()

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        with self.ManagedSessionMaker() as session:
            ver = self._get_mcp_version_or_raise(session, name, version)
            return ver.to_mlflow_entity()

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        if alias == _RESERVED_ALIAS:
            return self.get_latest_mcp_server_version(name)
        with self.ManagedSessionMaker() as session:
            alias_row = (
                self._get_query(session, SqlMCPServerAlias)
                .filter(
                    SqlMCPServerAlias.name == name,
                    SqlMCPServerAlias.alias == alias,
                )
                .first()
            )
            if alias_row is None:
                raise MlflowException(
                    f"Alias '{alias}' not found on MCP server '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return self.get_mcp_server_version(name, alias_row.version)

    def get_latest_mcp_server_version(self, name: str) -> MCPServerVersion:
        with self.ManagedSessionMaker() as session:
            server = self._get_mcp_server_or_raise(session, name)
            if server.latest_version:
                return self.get_mcp_server_version(name, server.latest_version)
            ver = (
                self._get_query(session, SqlMCPServerVersion)
                .filter(
                    SqlMCPServerVersion.name == name,
                    SqlMCPServerVersion.status != MCPStatus.DRAFT.value,
                )
                .order_by(SqlMCPServerVersion.created_at.desc())
                .first()
            )
            if ver is None:
                raise MlflowException(
                    f"No non-draft version found for MCP server '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return ver.to_mlflow_entity()

    def search_mcp_server_versions(
        self,
        name: str,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServerVersion]:
        self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        with self.ManagedSessionMaker() as session:
            query = (
                self._get_query(session, SqlMCPServerVersion)
                .options(joinedload(SqlMCPServerVersion.version_tags))
                .filter(
                    SqlMCPServerVersion.name == name,
                    SqlMCPServerVersion.status != MCPStatus.DELETED.value,
                )
            )
            filters = _parse_filter_string(filter_string or "")
            if "status" in filters:
                op, val = filters["status"]
                if isinstance(val, list):
                    query = query.filter(
                        SqlMCPServerVersion.status.in_([MCPStatus(v).value for v in val])
                    )
                elif op == "=":
                    query = query.filter(
                        SqlMCPServerVersion.status == MCPStatus(val).value
                    )

            query = (
                query
                .order_by(SqlMCPServerVersion.created_at.desc())
                .offset(offset)
                .limit(max_results + 1)
            )
            versions = query.all()
            entities = [v.to_mlflow_entity() for v in versions]
            next_token = None
            if len(entities) > max_results:
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList(entities[:max_results], next_token)

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        display_name: str | None = None,
        status: str | None = None,
        tools: list | None = None,
    ) -> MCPServerVersion:
        with self.ManagedSessionMaker() as session:
            ver = self._get_mcp_version_or_raise(session, name, version)
            if display_name is not None:
                ver.display_name = display_name
            if tools is not None:
                ver.tools = [
                    t.to_dict() if isinstance(t, MCPTool) else t for t in tools
                ]
            if status is not None:
                new_status = MCPStatus(status)
                current_status = MCPStatus(ver.status)
                if new_status not in _VALID_TRANSITIONS[current_status]:
                    raise MlflowException(
                        f"Invalid status transition: {current_status.value} -> {new_status.value}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                ver.status = new_status.value
            ver.last_updated_at = get_current_time_millis()
            session.flush()
            return ver.to_mlflow_entity()

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        with self.ManagedSessionMaker() as session:
            ver = self._get_mcp_version_or_raise(session, name, version)
            session.delete(ver)

    # --- MCPAccessBinding operations ---

    def create_mcp_access_binding(
        self,
        server_name: str,
        endpoint_url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
    ) -> MCPAccessBinding:
        if (server_version is None) == (server_alias is None):
            raise MlflowException(
                "Exactly one of server_version or server_alias must be set",
                error_code=INVALID_PARAMETER_VALUE,
            )
        now = get_current_time_millis()
        with self.ManagedSessionMaker() as session:
            self._get_mcp_server_or_raise(session, server_name)

            if server_version is not None:
                self._get_mcp_version_or_raise(session, server_name, server_version)
            else:
                alias_row = (
                    self._get_query(session, SqlMCPServerAlias)
                    .filter(
                        SqlMCPServerAlias.name == server_name,
                        SqlMCPServerAlias.alias == server_alias,
                    )
                    .first()
                )
                if alias_row is None:
                    raise MlflowException(
                        f"Alias '{server_alias}' not found on MCP server '{server_name}'",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

            transport_type_val = (
                transport_type.value
                if isinstance(transport_type, MCPRemoteTransportType)
                else (transport_type or MCPRemoteTransportType.STREAMABLE_HTTP.value)
            )
            binding = self._with_workspace_field(
                SqlMCPAccessBinding(
                    server_name=server_name,
                    endpoint_url=endpoint_url,
                    transport_type=transport_type_val,
                    server_version=server_version,
                    server_alias=server_alias,
                    created_at=now,
                    last_updated_at=now,
                )
            )
            session.add(binding)
            session.flush()
            return binding.to_mlflow_entity()

    def get_mcp_access_binding(self, server_name: str, binding_id: int) -> MCPAccessBinding:
        with self.ManagedSessionMaker() as session:
            binding = (
                self._get_query(session, SqlMCPAccessBinding)
                .filter(
                    SqlMCPAccessBinding.binding_id == binding_id,
                    SqlMCPAccessBinding.server_name == server_name,
                )
                .first()
            )
            if binding is None:
                raise MlflowException(
                    f"MCP access binding {binding_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return binding.to_mlflow_entity()

    def search_mcp_access_bindings(
        self,
        server_name: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessBinding]:
        self._validate_max_results_param(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        with self.ManagedSessionMaker() as session:
            query = self._get_query(session, SqlMCPAccessBinding)
            if server_name is not None:
                query = query.filter(SqlMCPAccessBinding.server_name == server_name)
            query = (
                query
                .order_by(SqlMCPAccessBinding.created_at.desc())
                .offset(offset)
                .limit(max_results + 1)
            )
            bindings = query.all()
            entities = [b.to_mlflow_entity() for b in bindings]
            next_token = None
            if len(entities) > max_results:
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList(entities[:max_results], next_token)

    def update_mcp_access_binding(
        self,
        server_name: str,
        binding_id: int,
        server_version: str | None = None,
        server_alias: str | None = None,
        endpoint_url: str | None = None,
        transport_type: MCPRemoteTransportType | None = None,
    ) -> MCPAccessBinding:
        with self.ManagedSessionMaker() as session:
            binding = (
                self._get_query(session, SqlMCPAccessBinding)
                .filter(
                    SqlMCPAccessBinding.binding_id == binding_id,
                    SqlMCPAccessBinding.server_name == server_name,
                )
                .first()
            )
            if binding is None:
                raise MlflowException(
                    f"MCP access binding {binding_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            if endpoint_url is not None:
                binding.endpoint_url = endpoint_url
            if transport_type is not None:
                binding.transport_type = (
                    transport_type.value
                    if isinstance(transport_type, MCPRemoteTransportType)
                    else transport_type
                )
            if server_version is not None or server_alias is not None:
                if server_version is not None and server_alias is not None:
                    raise MlflowException(
                        "Only one of server_version or server_alias may be updated at a time",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if server_version is not None:
                    self._get_mcp_version_or_raise(session, server_name, server_version)
                    binding.server_version = server_version
                    binding.server_alias = None
                else:
                    binding.server_alias = server_alias
                    binding.server_version = None
            binding.last_updated_at = get_current_time_millis()
            session.flush()
            return binding.to_mlflow_entity()

    def delete_mcp_access_binding(self, server_name: str, binding_id: int) -> None:
        with self.ManagedSessionMaker() as session:
            binding = (
                self._get_query(session, SqlMCPAccessBinding)
                .filter(
                    SqlMCPAccessBinding.binding_id == binding_id,
                    SqlMCPAccessBinding.server_name == server_name,
                )
                .first()
            )
            if binding is None:
                raise MlflowException(
                    f"MCP access binding {binding_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(binding)

    # --- Tag operations ---

    def set_mcp_server_tag(self, name: str, key: str, value: str) -> None:
        with self.ManagedSessionMaker() as session:
            self._get_mcp_server_or_raise(session, name)
            existing = (
                self._get_query(session, SqlMCPServerTag)
                .filter(
                    SqlMCPServerTag.name == name,
                    SqlMCPServerTag.key == key,
                )
                .first()
            )
            if existing is not None:
                existing.value = value
            else:
                tag = self._with_workspace_field(
                    SqlMCPServerTag(name=name, key=key, value=value)
                )
                session.add(tag)

    def delete_mcp_server_tag(self, name: str, key: str) -> None:
        with self.ManagedSessionMaker() as session:
            tag = (
                self._get_query(session, SqlMCPServerTag)
                .filter(
                    SqlMCPServerTag.name == name,
                    SqlMCPServerTag.key == key,
                )
                .first()
            )
            if tag is None:
                raise MlflowException(
                    f"Tag '{key}' not found on MCP server '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(tag)

    def set_mcp_server_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        with self.ManagedSessionMaker() as session:
            self._get_mcp_version_or_raise(session, name, version)
            existing = (
                self._get_query(session, SqlMCPServerVersionTag)
                .filter(
                    SqlMCPServerVersionTag.name == name,
                    SqlMCPServerVersionTag.version == version,
                    SqlMCPServerVersionTag.key == key,
                )
                .first()
            )
            if existing is not None:
                existing.value = value
            else:
                tag = self._with_workspace_field(
                    SqlMCPServerVersionTag(
                        name=name, version=version, key=key, value=value
                    )
                )
                session.add(tag)

    def delete_mcp_server_version_tag(self, name: str, version: str, key: str) -> None:
        with self.ManagedSessionMaker() as session:
            tag = (
                self._get_query(session, SqlMCPServerVersionTag)
                .filter(
                    SqlMCPServerVersionTag.name == name,
                    SqlMCPServerVersionTag.version == version,
                    SqlMCPServerVersionTag.key == key,
                )
                .first()
            )
            if tag is None:
                raise MlflowException(
                    f"Tag '{key}' not found on MCP server version '{name}@{version}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(tag)

    # --- Alias operations ---

    def set_mcp_server_alias(self, name: str, alias: str, version: str) -> None:
        if alias == _RESERVED_ALIAS:
            raise MlflowException(
                f"'{_RESERVED_ALIAS}' is a reserved alias name and cannot be set",
                error_code=INVALID_PARAMETER_VALUE,
            )
        with self.ManagedSessionMaker() as session:
            self._get_mcp_version_or_raise(session, name, version)
            existing = (
                self._get_query(session, SqlMCPServerAlias)
                .filter(
                    SqlMCPServerAlias.name == name,
                    SqlMCPServerAlias.alias == alias,
                )
                .first()
            )
            if existing is not None:
                existing.version = version
            else:
                alias_obj = self._with_workspace_field(
                    SqlMCPServerAlias(name=name, alias=alias, version=version)
                )
                session.add(alias_obj)

    def delete_mcp_server_alias(self, name: str, alias: str) -> None:
        with self.ManagedSessionMaker() as session:
            alias_row = (
                self._get_query(session, SqlMCPServerAlias)
                .filter(
                    SqlMCPServerAlias.name == name,
                    SqlMCPServerAlias.alias == alias,
                )
                .first()
            )
            if alias_row is None:
                raise MlflowException(
                    f"Alias '{alias}' not found on MCP server '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(alias_row)

    # --- Trace linking ---

    def link_mcp_server_versions_to_trace(
        self,
        trace_id: str,
        mcp_servers: list[MCPServerVersion],
    ) -> None:
        from mlflow.store.tracking.dbmodels.models import SqlEntityAssociation

        now = get_current_time_millis()
        with self.ManagedSessionMaker() as session:
            for ver in mcp_servers:
                ws = getattr(self, "_workspace", "default")
                dest_id = f"{ws}:{ver.name}:{ver.version}"
                existing = (
                    session.query(SqlEntityAssociation)
                    .filter(
                        SqlEntityAssociation.source_type == "trace",
                        SqlEntityAssociation.source_id == trace_id,
                        SqlEntityAssociation.destination_type == "mcp_server_version",
                        SqlEntityAssociation.destination_id == dest_id,
                    )
                    .first()
                )
                if existing is None:
                    import uuid

                    session.add(
                        SqlEntityAssociation(
                            association_id=str(uuid.uuid4()),
                            source_type="trace",
                            source_id=trace_id,
                            destination_type="mcp_server_version",
                            destination_id=dest_id,
                            created_time=now,
                        )
                    )

    def get_mcp_server_versions_for_trace(
        self,
        trace_id: str,
    ) -> list[MCPServerVersion]:
        from mlflow.store.tracking.dbmodels.models import SqlEntityAssociation

        ws = getattr(self, "_workspace", "default")
        with self.ManagedSessionMaker() as session:
            associations = (
                session.query(SqlEntityAssociation)
                .filter(
                    SqlEntityAssociation.source_type == "trace",
                    SqlEntityAssociation.source_id == trace_id,
                    SqlEntityAssociation.destination_type == "mcp_server_version",
                    SqlEntityAssociation.destination_id.like(f"{ws}:%"),
                )
                .all()
            )
            results = []
            for assoc in associations:
                parts = assoc.destination_id.split(":", 2)
                if len(parts) == 3:
                    _, name, version = parts
                    try:
                        ver = self.get_mcp_server_version(name, version)
                        results.append(ver)
                    except MlflowException:
                        pass
            return results


def _derive_status(sql_server: SqlMCPServer) -> MCPStatus | None:
    from mlflow.store.tracking.dbmodels.models import _resolve_latest_version

    resolved = _resolve_latest_version(sql_server.latest_version, sql_server.server_versions)
    if resolved is None:
        return None
    return MCPStatus(resolved.status)


def _parse_filter_string(filter_string: str) -> dict:
    import re

    filters = {}
    if not filter_string.strip():
        return filters

    in_pattern = re.compile(
        r"(\w+)\s+IN\s*\(([^)]+)\)", re.IGNORECASE
    )
    for match in in_pattern.finditer(filter_string):
        field = match.group(1).lower()
        values_str = match.group(2)
        values = [v.strip().strip("'\"") for v in values_str.split(",")]
        filters[field] = ("IN", values)

    eq_pattern = re.compile(r"(\w+)\s*(=|LIKE|like)\s*'([^']*)'", re.IGNORECASE)
    for match in eq_pattern.finditer(filter_string):
        field = match.group(1).lower()
        op = match.group(2).upper()
        val = match.group(3)
        if field not in filters:
            filters[field] = (op, val)

    bool_pattern = re.compile(r"(\w+)\s*=\s*(true|false)", re.IGNORECASE)
    for match in bool_pattern.finditer(filter_string):
        field = match.group(1).lower()
        val = match.group(2).lower() == "true"
        if field not in filters:
            filters[field] = ("=", val)

    return filters
