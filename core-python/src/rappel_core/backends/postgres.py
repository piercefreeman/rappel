"""Postgres backend for persisting runner state and action results."""

from __future__ import annotations

import asyncio
import pickle
import sys
from typing import Any, Sequence
from uuid import uuid4

import psycopg

from .base import ActionDone, BaseBackend, GraphUpdate, InstanceDone, QueuedInstance

DEFAULT_DSN = "postgresql://rappel:rappel@localhost:5432/rappel_core"


def _ensure_proto_aliases() -> None:
    """Ensure protobuf modules are importable by their generated names."""
    if "ast_pb2" not in sys.modules:
        try:
            from proto import ast_pb2 as proto_ast
        except ImportError:
            proto_ast = None
        if proto_ast is not None:
            sys.modules["ast_pb2"] = proto_ast
    if "messages_pb2" not in sys.modules:
        try:
            from proto import messages_pb2 as proto_messages
        except ImportError:
            proto_messages = None
        if proto_messages is not None:
            sys.modules["messages_pb2"] = proto_messages


class PostgresBackend(BaseBackend):
    """Persist runner state and action results in Postgres."""

    def __init__(self, dsn: str = DEFAULT_DSN) -> None:
        _ensure_proto_aliases()
        self._dsn = dsn
        self._ensure_schema()

    def save_graphs(self, graphs: Sequence[GraphUpdate]) -> None:
        if not graphs:
            return
        payloads = [(self._serialize(graph.state),) for graph in graphs]
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO runner_graph_updates (state) VALUES (%s)", payloads
                )

    def save_actions_done(self, actions: Sequence[ActionDone]) -> None:
        if not actions:
            return
        payloads = [
            (
                action.node_id,
                action.action_name,
                action.attempt,
                self._serialize(action.result),
            )
            for action in actions
        ]
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO runner_actions_done (node_id, action_name, attempt, result)
                    VALUES (%s, %s, %s, %s)
                    """,
                    payloads,
                )

    def save_instances_done(self, instances: Sequence[InstanceDone]) -> None:
        if not instances:
            return
        payloads = [
            (
                instance.executor_id,
                instance.entry_node,
                self._serialize_optional(instance.result),
                self._serialize_optional(instance.error),
            )
            for instance in instances
        ]
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO runner_instances_done (executor_id, entry_node, result, error)
                    VALUES (%s, %s, %s, %s)
                    """,
                    payloads,
                )

    async def get_queued_instances(self, size: int) -> list[QueuedInstance]:
        if size <= 0:
            return []
        return await asyncio.to_thread(self._fetch_queued_instances, size)

    def queue_instances(self, instances: Sequence[QueuedInstance]) -> None:
        """Insert queued instances for run-loop consumption."""
        if not instances:
            return
        payloads = [(uuid4(), self._serialize(instance)) for instance in instances]
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO queued_instances (instance_id, payload) VALUES (%s, %s)",
                    payloads,
                )

    def clear_queue(self) -> None:
        """Delete all queued instances from the backing table."""
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM queued_instances")

    def _fetch_queued_instances(self, size: int) -> list[QueuedInstance]:
        if size <= 0:
            return []
        with psycopg.connect(self._dsn) as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT instance_id, payload
                        FROM queued_instances
                        ORDER BY created_at
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                        """,
                        (size,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return []
                    instance_ids = [row[0] for row in rows]
                    cur.execute(
                        "DELETE FROM queued_instances WHERE instance_id = ANY(%s)",
                        (instance_ids,),
                    )
        return [self._deserialize(row[1]) for row in rows]

    def _ensure_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runner_graph_updates (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        state BYTEA NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runner_actions_done (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        node_id UUID NOT NULL,
                        action_name TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        result BYTEA
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runner_instances_done (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        executor_id UUID NOT NULL,
                        entry_node UUID NOT NULL,
                        result BYTEA,
                        error BYTEA
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queued_instances (
                        instance_id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload BYTEA NOT NULL
                    )
                    """
                )

    @staticmethod
    def _serialize(value: Any) -> bytes:
        _ensure_proto_aliases()
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _serialize_optional(cls, value: Any | None) -> bytes | None:
        if value is None:
            return None
        return cls._serialize(value)

    @staticmethod
    def _deserialize(payload: bytes | memoryview) -> QueuedInstance:
        _ensure_proto_aliases()
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        value = pickle.loads(payload)
        if not isinstance(value, QueuedInstance):
            raise TypeError("queued instance payload did not decode to QueuedInstance")
        return value
